import json
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ("0,1,2")
import random
import torch
from torch.utils.data import Dataset, DataLoader
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# =====================
# 配置
# =====================
MODEL_PATH = "/disk3/model/gemma-4-31B"
TRAIN_PATH = "/disk2/disk4/chenp/2024-chenpeng/data_strengthen/IMAGE/original_code/image_keyword/dialogcc_gemma/train.json"
DEV_PATH = "/disk2/disk4/chenp/2024-chenpeng/data_strengthen/IMAGE/original_code/image_keyword/dialogcc_gemma/dev.json"

MAX_LENGTH = 600
MAX_GEN_TOKENS = 256
BATCH_SIZE = 1
GRAD_ACC = 16
LR = 1e-4
EPOCHS = 1
TRAIN_SAMPLES = 200
DEV_SAMPLES = 300
EVAL_STEPS = 2000
PATIENCE = 10
MIN_DELTA = 0.0005

# =====================
# Dataset
# =====================
class MultiTaskDataset(Dataset):
    def __init__(self, path, tokenizer, max_samples=None, is_train=True):
        self.samples = []
        self.raw_samples = []
        self.tokenizer = tokenizer
        self.num_max = 0

        data = json.load(open(path))

        # if is_train:
        #     data = dict(list(data.items())[:TRAIN_SAMPLES])  # 可调试

        print(len(data))

        self.num_task1 = 0
        self.num_task2 = 0
        self.num_task3 = 0

        for _, turns in data.items():
            self.build_task1(turns)
            self.build_task23(turns)

        combined = list(zip(self.samples, self.raw_samples))
        # print(combined[0])

        # ===== train 才 shuffle =====
        if is_train:
            random.shuffle(combined)

        # ===== dev：固定 + 三任务均衡 =====
        if max_samples:
            task_groups = {
                "image_position": [],
                "keyword": [],
                "description": []
            }

            for s, r in combined:
                task_groups[r[0]].append((s, r))

            per_task = max_samples // 3

            new_combined = []
            for task in ["image_position", "keyword", "description"]:
                group = task_groups[task]

                # ⚠️ 不 shuffle → 保证固定
                new_combined.extend(group[:per_task])

            combined = new_combined

        self.samples, self.raw_samples = zip(*combined)

        print(f"task1:{self.num_task1}, task2:{self.num_task2}, task3:{self.num_task3}, raw_samples:{len(self.raw_samples)}, num_max:{self.num_max}")

    def build_task1(self, turns):

        if not turns:
            return

        prompt = (
            "<TASK_IMAGE_POSITION>\n"
            "You are an expert in multimodal conversation understanding and content planning.\n\n"

            "You are given a conversation consisting of numbered utterances.\n"
            "Your task is to decide where images should be inserted.\n"
            "Each image should be inserted immediately AFTER a specific utterance.\n\n"

            "Output format:\n"
            "<IMAGE_INSERTIONS>\n"
            "<INSERTION>\n"
            "id: <int>\n"
            "utterance: \"<exact utterance text>\"\n"
            "rationale: <reason>\n"
            "</INSERTION>\n"
            "</IMAGE_INSERTIONS>\n\n"

            "Rules:\n"
            "- The id must correspond to the utterance index\n"
            "- The utterance text must EXACTLY match the original utterance\n"
            "- Do not modify or paraphrase the utterance text\n"
            "- Insert at least one image\n"
            "- Do not output anything outside the required format\n\n"

            "Conversation:\n"
        )

        # 深拷贝，避免修改原始数据
        turns_copy = copy.deepcopy(turns)

        flag= 0

        while True:

            utterances = []
            insertions = []

            utt_id = 0
            last_id = None
            last_text = None

            # ========= 构造对话 =========
            for t in turns_copy:

                if t[0] == "txt":

                    utterance_text = t[1].strip()

                    utterances.append(
                        f"[{utt_id}] {utterance_text}"
                    )

                    last_id = utt_id
                    last_text = utterance_text

                    utt_id += 1

                elif t[0] == "img" and last_id is not None:

                    insertions.append({
                        "id": last_id,
                        "utterance": last_text,
                        "reason": t[1].strip()
                    })

            # 没有有效数据
            if len(utterances) == 0 or len(insertions) == 0:
                return

            # ========= 输入 =========
            context_text = "\n".join(utterances)

            input_text = (
                    prompt +
                    context_text +
                    "\n\nOutput:\n"
            )

            # ========= 标签 =========
            insertion_blocks = []

            for ins in insertions:
                block = (
                    "<INSERTION>\n"
                    f"id: {ins['id']}\n"
                    f'utterance: "{ins["utterance"]}"\n'
                    f"rationale: {ins['reason']}\n"
                    "</INSERTION>"
                )

                insertion_blocks.append(block)

            label = (
                    "<IMAGE_INSERTIONS>\n\n" +
                    "\n\n".join(insertion_blocks) +
                    "\n\n</IMAGE_INSERTIONS>"
            )

            # ========= tokenize instruction =========
            instruction = self.tokenizer(
                input_text,
                add_special_tokens=False,
                truncation=False,
                padding=False
            )

            # ========= tokenize response =========
            response = self.tokenizer(
                label,
                add_special_tokens=False,
                truncation=False,
                padding=False
            )

            # ========= 拼接 =========
            input_ids = (
                    instruction["input_ids"] +
                    response["input_ids"] +
                    [self.tokenizer.eos_token_id]
            )

            # ========= 长度检测 =========
            if len(input_ids) <= MAX_LENGTH:
                if flag== 0:
                    self.num_max += 1
                break
            flag=1

            # 超长则删除最前面的 txt 以及其后的 img
            for i, t in enumerate(turns_copy):

                if t[0] == "txt":

                    del turns_copy[i]

                    if i < len(turns_copy) and turns_copy[i][0] == "img":
                        del turns_copy[i]

                    break

            if not turns_copy:
                return

        # ========= 保存 =========
        self.samples.append((input_text, label))

        self.num_task1 += 1

        self.raw_samples.append((
            "image_position",
            input_text,
            label
        ))
    def build_task23(self, turns):

        # 深拷贝，保证原始数据不变
        turns_copy = copy.deepcopy(turns)

        history = []
        speaker_flag = True

        for t in turns_copy:

            if t[0] == "txt":
                speaker = "User" if speaker_flag else "Assistant"
                history.append(f"[{speaker}] {t[1].strip()}")
                speaker_flag = not speaker_flag

            elif t[0] == "img":

                if len(history) < 1:
                    continue

                context = "\n".join(history[:-1])
                response = history[-1]
                keywords_list = [k.strip() for k in t[3]]
                keywords = ", ".join(keywords_list)
                description = t[2].strip()

                task2_prompt_template = (
                    "<TASK_KEYWORD_PREDICTION>\n\n"
                    "You are an expert in multimodal conversation understanding.\n\n" 
                    "You are given a dialogue context and a user or assistant response.\n" 
                    "In addition to the textual response, the speaker also shares an image.\n\n" 
                    "Your task is to predict the key content of the shared image as keywords, based on both the dialogue context and the response.\n\n" 
                    "The keywords should reflect what is likely shown in the image implied by the conversation.\n\n" 
                    "Input description:\n" 
                    "- Context: previous dialogue history\n" 
                    "- Response: the current utterance associated with the shared image\n\n" 
                    "Output format:\n" 
                     "<IMAGE_KEYWORDS>keyword1, keyword2, keyword3</IMAGE_KEYWORDS>\n\n"
                    "Rules:\n" 
                    "- Keywords should reflect the main content of the image\n" 
                    "- Keywords may come from the context or the response, or be reasonably inferred from them\n" 
                    "- Prefer meaningful words or short phrases\n" 
                    "- Cover objects, people, actions, attributes, events, and scenes when possible\n" 
                    "- Avoid unimportant words (e.g., 'a', 'the')\n" 
                    "- No duplication\n" 
                    "- Separate keywords with commas\n" 
                    "- Do not output anything else\n\n" 
                    "Context:\n{context}\n\n"
                    "Response:\n{response}\n\n"
                    "Output:\n"
                )

                task3_prompt_template = (
                    "<TASK_IMAGE_DESCRIPTION>\n\n" 
                    "You are an expert in multimodal conversation understanding.\n\n" 
                    "You are given a dialogue context, a response, and a set of keywords.\n" 
                    "In the response, the speaker shares an image.\n\n" 
                    "Your task is to generate TWO complementary descriptions of the image:\n" 
                    "1) An image caption describing the visible content of the image\n" 
                    "2) A context-aware description that explains the image in the dialogue\n\n" 
                    "The two descriptions must be separated by a vertical bar '|'.\n\n" 
                    "Input description:\n" 
                    "- Context: previous dialogue history\n" 
                    "- Response: the utterance associated with the image\n" 
                    "- Keywords: key elements related to the image (extracted from both descriptions)\n\n" 
                    "Output format:\n" "<IMAGE_DESCRIPTION>image_caption | context_aware_description</IMAGE_DESCRIPTION>\n\n"
                    "Field description:\n" 
                    "- image_caption: a concise and natural caption describing what is directly visible in the image\n" 
                    "- context_aware_description: a description that incorporates dialogue context, " 
                    "including identities, intentions, or specific situations implied by the conversation\n\n" 
                    "Rules:\n" 
                    "- You MUST output exactly one line\n" 
                    "- You MUST include the '|' separator\n" 
                    "- The two descriptions must refer to the SAME image\n" 
                    "- The image_caption should focus on visible content only\n" 
                    "- The context_aware_description should incorporate dialogue understanding\n" 
                    "- Keywords may appear in both descriptions when appropriate\n" 
                    "- Both descriptions should be consistent with the provided keywords\n" 
                    "- Prefer clear, natural, and fluent language\n" 
                    "- Avoid hallucination or unrelated content\n" 
                    "- Do not output anything else\n\n" 
                    "Context:\n{context}\n\n"
                    "Response:\n{response}\n\n"
                    "Keywords:\n{keywords}\n\n"
                    "Output:\n"
                )

                keyword_label = (
                    "<IMAGE_KEYWORDS>"
                    f"{keywords}"
                    "</IMAGE_KEYWORDS>"
                )
                description_label = (
                    "<IMAGE_DESCRIPTION>"
                    f"{description}"
                    "</IMAGE_DESCRIPTION>"
                )
                # ===== 循环检查 Task2 token 长度 =====
                flag=0
                context_task2 = copy.deepcopy(context)
                while True:
                    prompt_task2 = task2_prompt_template.format(
                        context=context_task2,
                        response=response
                    )

                    # ===== instruction =====
                    instruction_task2 = self.tokenizer(
                        prompt_task2,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False
                    )

                    # ===== response =====
                    response_task2 = self.tokenizer(
                        keyword_label,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False
                    )

                    # ===== 与训练保持一致 =====
                    input_ids_task2 = (
                            instruction_task2["input_ids"] +
                            response_task2["input_ids"] +
                            [self.tokenizer.eos_token_id]
                    )

                    if len(input_ids_task2) <= MAX_LENGTH or not context_task2:
                        if flag==0:
                            self.num_max +=1
                        break
                    # 超长则删掉 context_task2 的第一句
                    flag=1
                    context_task2 = "\n".join(context_task2.split("\n")[1:])

                self.samples.append((prompt_task2, keyword_label))
                self.num_task2 +=1
                self.raw_samples.append(("keyword", prompt_task2, keyword_label))

                # ===== 循环检查 Task3 token 长度 =====
                context_task3 = copy.deepcopy(context)
                flag=0
                while True:
                    prompt_task3 = task3_prompt_template.format(
                        context=context_task3,
                        response=response,
                        keywords=keywords
                    )

                    # ===== instruction =====
                    instruction_task3 = self.tokenizer(
                        prompt_task3,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False
                    )

                    # ===== response =====
                    response_task3 = self.tokenizer(
                        description_label,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False
                    )

                    # ===== 与训练保持一致 =====
                    input_ids_task3 = (
                            instruction_task3["input_ids"] +
                            response_task3["input_ids"] +
                            [self.tokenizer.eos_token_id]
                    )

                    if len(input_ids_task3) <= MAX_LENGTH or not context_task3:
                        if flag==0:
                            self.num_max +=1
                        break
                    # 超长则删掉 context_task3 的第一句
                    flag=1
                    context_task3 = "\n".join(context_task3.split("\n")[1:])

                self.samples.append((prompt_task3, description_label))
                self.num_task3 +=1
                self.raw_samples.append(("description", prompt_task3, description_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        prompt, label = self.samples[idx]

        # ===== instruction =====
        instruction = self.tokenizer(
            prompt,
            add_special_tokens=False
        )

        # ===== response =====
        response = self.tokenizer(
            label,
            add_special_tokens=False
        )

        # ===== 拼接 =====
        input_ids = (
                instruction["input_ids"] +
                response["input_ids"] +
                [self.tokenizer.eos_token_id]
        )

        attention_mask = (
                instruction["attention_mask"] +
                response["attention_mask"] +
                [1]
        )

        # ===== labels =====
        labels = (
                [-100] * len(instruction["input_ids"]) +
                response["input_ids"] +
                [self.tokenizer.eos_token_id]
        )

        # ===== truncate =====
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        # ===== padding =====
        pad_len = MAX_LENGTH - len(input_ids)

        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        # ===== tensor =====
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# =====================
# Eval
# =====================
def evaluate(model, dataloader):
    model.eval()
    losses = []
    from tqdm import tqdm


    device = next(model.parameters()).device  # ✅ 提前拿一次

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}  # ✅ 必须加

            loss = model(**batch).loss
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if len(losses) > 0 else 0

# =====================
# 🔥 生成监控（每个任务都看）
# =====================
def generate_samples(model, tokenizer, dataset, max_gen_tokens=256):
    """
    对 dataset 中每个任务生成示例，避免生成被截断。
    max_gen_tokens: 单条生成最大 token 数，可根据显存调节
    """
    model.eval()
    print("\n========== GENERATION CHECK ==========")

    task_count = {
        "image_position": 0,
        "keyword": 0,
        "description": 0
    }

    device = next(model.parameters()).device
    for task, prompt, gt in dataset.raw_samples:

        if task_count[task] >= 3:
            continue

        # ⚠️ 增加 pad_token_id，避免生成提前结束
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        # pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("=====================")
        # print(pred)
        # print("====================")

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        pred = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        print(f"\n===== {task} #{task_count[task]+1} =====")
        print("=================GT================:\n", gt[:500])   # 打印更多字符
        print("=================PRED================:\n", pred[:500])

        task_count[task] += 1

        # 提前结束（3个任务 * 3个）
        if all(v >= 3 for v in task_count.values()):
            break

    model.train()

# =====================
# Train
# =====================
def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )

    model = prepare_model_for_kbit_training(model)

    # ❗冻结vision（如果你不训练视觉）
    for n, p in model.named_parameters():
        if "vision_tower" in n:
            p.requires_grad = False

    model.config.use_cache = False

    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    import torch.nn as nn

    def unwrap_linear(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Module):
                if hasattr(child, "linear"):
                    setattr(module, name, child.linear)
                else:
                    unwrap_linear(child)

    unwrap_linear(model)  # ⭐⭐⭐ 就放这里

    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    ))

    model.print_trainable_parameters()
    # print("\n===== TRAINABLE PARAMS =====")
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #
    # print("============================\n")


    train_dataset = MultiTaskDataset(TRAIN_PATH, tokenizer, is_train=True)

    dev_dataset = MultiTaskDataset(
        DEV_PATH,
        tokenizer,
        max_samples=DEV_SAMPLES,  # ⚠️ 必须是3的倍数
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1,pin_memory=True)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR)

    best_loss = float("inf")
    patience = 0
    global_step = 0

    from tqdm import tqdm

    for epoch in range(EPOCHS):
        device = next(model.parameters()).device
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}",mininterval=1)

        for step, batch in enumerate(pbar):
            # batch = {k: v.to(model.device) for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss = loss / GRAD_ACC
            loss.backward()

            if (step + 1) % GRAD_ACC == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * GRAD_ACC:.4f}",
                # "loss": f"{loss.item():.4f}",
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "accum": f"{(step % GRAD_ACC) + 1}/{GRAD_ACC}"
            })

            # ===== eval =====
            if global_step % EVAL_STEPS == 0:

                dev_loss = evaluate(model, dev_loader)

                print(f"\nDEV LOSS: {dev_loss}")

                generate_samples(model, tokenizer, dev_dataset)

                if best_loss - dev_loss > MIN_DELTA:
                    best_loss = dev_loss
                    patience = 0

                    model.save_pretrained("best_lora")
                    print("✅ save best")
                #
                # else:
                #     patience += 1
                #
                # if patience >= PATIENCE:
                #     print("⛔ early stop")
                #     return

    model.save_pretrained("lora_output")

if __name__ == "__main__":
    main()