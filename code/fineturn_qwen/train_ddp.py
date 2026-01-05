import json, os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from accelerate import Accelerator
import random

# =======================
# 配置
# =======================
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
DATA_PATH = "/root/autodl-tmp/data_strengthen/DialogCC/rationale_caption/train.json"
OUTPUT_DIR = "./lora_rationale_caption"

BATCH_SIZE = 1
MAX_LENGTH = 1024
LR = 3e-4
NUM_EPOCHS = 5

# =======================
# Accelerator 初始化
# =======================
accelerator = Accelerator()
device = accelerator.device
print(f"[INFO] Using device: {device}")

# =======================
# 数据加载
# =======================
def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 可选下采样，保持你原来的行为
        data = {k: v for i, (k, v) in enumerate(data.items()) if i % 1 == 0}
        print(f"[INFO] Load {len(data)} samples")
    return data

# =======================
# 构建单个样本
# =======================
def build_sample(dialogue_seq, tokenizer=None, max_tokens=None):
    """
    构建单个样本，自动检测 token 数量：
    - 如果 tokenizer+max_tokens 给出，超过长度则从后往前删对话，同时删对应 img/description
    - 每句 txt 之后可能跟 img，如果有就一起算
    - 最终 label 必须有 description，否则返回 None, None
    """
    def token_count(text):
        if tokenizer is None:
            return 0  # 如果没传 tokenizer，不做长度检测
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    def generate_prompt():
        return (
            "You are an expert in multimodal conversation understanding and content planning.\n\n"
            "You are given a conversation consisting of numbered utterances.\n"
            "Your task is to decide at which points in the conversation an image should be inserted.\n\n"
            "You must insert at least ONE image.\n"
            "Each image must be inserted immediately AFTER a specific utterance.\n\n"
            "Output MUST follow the exact structured format below.\n"
            "Do NOT add any text outside this format.\n\n"
            "<IMAGE_INSERTIONS>\n"
            "- utterance_id: <int>  # the index of the utterance after which the image is inserted\n"
            "- utterance: <text>    # the exact utterance text at this position\n"
            "- rationale: a clear explanation of the communicative need for the image\n"
            "- description: a concrete and detailed visual description of the image content\n"
            "</IMAGE_INSERTIONS>\n\n"
            "Conversation:\n"
        )

    dialogue_seq = dialogue_seq.copy()
    while True:
        utterances = []
        utterance_text_map = {}
        image_insertions = []
        utterance_id = 0
        last_txt_id = None

        i = 0
        while i < len(dialogue_seq):
            item = dialogue_seq[i]

            if item[0] == "txt":
                # 添加 txt
                utterances.append(f"utterance_{utterance_id}: {item[1]}")
                utterance_text_map[utterance_id] = item[1]
                last_txt_id = utterance_id
                utterance_id += 1

                # 检查下一句是否是 img
                if i + 1 < len(dialogue_seq) and dialogue_seq[i + 1][0] == "img":
                    img_item = dialogue_seq[i + 1]
                    if last_txt_id is not None:
                        image_insertions.append({
                            "utterance_id": last_txt_id,
                            "utterance": utterance_text_map[last_txt_id],
                            "rationale": img_item[1],
                            "description": img_item[2]
                        })
                    i += 1  # 跳过 img
            i += 1

        prompt = generate_prompt()
        context = "\n".join(utterances)
        input_text = f"{prompt}{context}\n\nOutput:\n"

        # 生成 label_text
        blocks = []
        for ins in image_insertions:
            blocks.append(
                f"- utterance_id: {ins['utterance_id']}\n"
                f"- utterance: {ins['utterance']}\n"
                f"- rationale: {ins['rationale']}\n"
                f"- description: {ins['description']}"
            )
        label_text = "<IMAGE_INSERTIONS>\n" + "\n\n".join(blocks) + "\n</IMAGE_INSERTIONS>"

        # label 中必须有 description
        has_desc = any("description:" in b and b.strip() != "description:" for b in blocks)
        if not has_desc:
            return None, None

        # 检查长度
        if tokenizer is None or max_tokens is None or token_count(input_text + label_text) <= max_tokens:
            return input_text, label_text

        # 超长，从后往前删 txt + 对应 img/description
        removed = False
        for j in reversed(range(len(dialogue_seq))):
            if dialogue_seq[j][0] == "img":
                # 删除 img
                dialogue_seq.pop(j)
                # 删除它前面的 txt
                if j - 1 >= 0 and dialogue_seq[j - 1][0] == "txt":
                    dialogue_seq.pop(j - 1)
                removed = True
                break
            elif dialogue_seq[j][0] == "txt":
                # 删除 txt 就够了
                dialogue_seq.pop(j)
                removed = True
                break
        if not removed:
            # 没有 txt 可删，丢弃样本
            return None, None


# =======================
# 构建数据集
# =======================
def prepare_dataset(data_path, tokenizer):
    raw_data = load_data(data_path)
    samples = []

    for _, dialogue_seq in raw_data.items():
        inp, lbl = build_sample(dialogue_seq, tokenizer=tokenizer, max_tokens=MAX_LENGTH)
        if inp is None or lbl is None:
             continue
        samples.append((inp, lbl))

    print(f"[Dataset] total samples: {len(samples)}")
    return samples

# =======================
# Dataset
# =======================
class DialogueDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, label = self.samples[idx]
        full_text = prompt + label

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        prompt_len = len(
            self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        )

        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# =======================
# LoRA 配置
# =======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# =======================
# 加载模型
# =======================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    use_fast=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16
)

model = get_peft_model(model, lora_config)
model.train()

# =======================
# 数据 & 优化器
# =======================
samples = prepare_dataset(DATA_PATH, tokenizer)
dataset = DialogueDataset(samples, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# =======================
# 训练
# =======================
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    loop = tqdm(dataloader, desc="Training", ncols=100)

    for batch in loop:
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# =======================
# 保存 LoRA
# =======================
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] LoRA weights saved to {OUTPUT_DIR}")
