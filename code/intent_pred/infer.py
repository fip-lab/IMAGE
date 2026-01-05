import json
import os
import torch,random
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

local_rank = setup_ddp()
device = torch.device(f"cuda:{local_rank}")
world_size = dist.get_world_size()

BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
LORA_PATH = "./output/lora_intent_qwen3"
# TEST_DATA_PATH = "../Photochat/no_sel/test.json"
# OUTPUT_PATH = "./no_sel/predictions_test.json"
# METRIC_PATH = "./no_sel/metrics_test.txt"
TEST_DATA_PATH = "./ori_data/test.json"
OUTPUT_PATH = "./intent/predictions_test.json"
METRIC_PATH = "./intent/metrics_test.txt"


MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 2

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(device)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

def build_prompt(text_history):
    prompt = (
        "### Task: Predict whether the next utterance will share an image.\n"
        "### Conversation context:\n"
    )
    speakers = ["User", "Assistant"]
    for i, txt in enumerate(text_history):
        prompt += f"{speakers[i % 2]}: {txt}\n"

    prompt += (
        "\n### System: Please predict whether the next utterance will share an image. "
        "Answer 'Yes' or 'No'.\nAnswer:"
    )
    return prompt

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dialog_items = list(raw_data.items())
# dialog_items = dialog_items[:100]
dialog_items = dialog_items[local_rank::world_size]

final_outputs = {}
tp = fp = fn = tn = 0

for dialog_id, dialogue in tqdm(dialog_items, disable=(local_rank != 0)):
    txt_history = []
    step_preds = []

    # 构造 GT（仅针对 txt）
    gt_labels = []
    for i, item in enumerate(dialogue):
        if item[0] != "txt":
            continue
        next_is_img = (
            i + 1 < len(dialogue) and dialogue[i + 1][0] == "img"
        )
        gt_labels.append(1 if next_is_img else 0)

    # ============================
    # 预先确定用于统计的 txt 编号
    # ============================
    POS_NEG_RATIO = 3  # 你想要的比例 1:9

    pos_indices = [i for i, v in enumerate(gt_labels) if v == 1]
    neg_indices = [i for i, v in enumerate(gt_labels) if v == 0]

    num_pos = len(pos_indices)
    max_neg = num_pos * POS_NEG_RATIO

    if len(neg_indices) <= max_neg:
        selected_neg_indices = set(neg_indices)
    else:
        selected_neg_indices = set(
            random.sample(neg_indices, max_neg)
        )

    gt_ptr = 0
    num_txt = len(gt_labels)

    gen_texts = []

    for item in dialogue:
        if item[0] != "txt":
            continue

        txt_history.append(item[1])
        prompt = build_prompt(txt_history)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        gen_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        gen_texts.append(gen_text)

        gen = gen_text.strip().lower()

        pred = 1 if gen.startswith("yes") else 0

        gt = gt_labels[gt_ptr]

        cur_txt_idx = gt_ptr

        gt_ptr += 1

        if gt == 1:
            # 正样本：一定统计
            if pred == 1:
                tp += 1
            else:
                fn += 1

        else:
            # 负样本：只有被选中才统计
            if cur_txt_idx in selected_neg_indices:
                if pred == 1:
                    fp += 1
                else:
                    tn += 1
            # 否则直接跳过

        # # 指标统计
        # if pred == 1 and gt == 1:
        #     tp += 1
        # elif pred == 1 and gt == 0:
        #     fp += 1
        # elif pred == 0 and gt == 1:
        #     fn += 1
        # else:
        #     tn += 1

        # 只保存是否预测正确
        step_preds.append(1 if pred == gt else 0)

    final_outputs[dialog_id] = [dialogue,step_preds]
    # if local_rank == 0:
    #     tqdm.write(f"{dialog_id}: {gen_texts}")

# DDP gather
gathered = [None for _ in range(world_size)] if local_rank == 0 else None
dist.gather_object(
    (final_outputs, tp, fp, fn, tn),
    gathered,
    dst=0
)

if local_rank == 0:
    merged = {}
    TP = FP = FN = TN = 0

    for out, t, f, n, tn_ in gathered:
        merged.update(out)
        TP += t
        FP += f
        FN += n
        TN += tn_

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / (total + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n========== Evaluation ==========")
    print(f"Total samples: {total}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # ✅ 保存预测结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # ✅ 保存评价指标
    with open(METRIC_PATH, "w", encoding="utf-8") as f:
        f.write("Intent Prediction Evaluation\n")
        f.write("============================\n")
        f.write(f"Total samples: {total}\n\n")

        f.write(f"TP: {TP}\n")
        f.write(f"FP: {FP}\n")
        f.write(f"FN: {FN}\n")
        f.write(f"TN: {TN}\n\n")

        f.write(f"Accuracy : {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall   : {recall:.6f}\n")
        f.write(f"F1-score : {f1:.6f}\n")

cleanup_ddp()
