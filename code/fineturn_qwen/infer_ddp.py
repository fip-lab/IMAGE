import json
import os
import re
import torch
import torch.distributed as dist
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import Counter
import math
from nltk.util import ngrams

# =====================================================
# DDP 初始化（仅新增）
# =====================================================
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

# =====================================================
# 基础配置（完全一致）
# =====================================================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
LORA_PATH = "./lora_rationale_caption_description"
# TEST_DATA_PATH = "/disk2/disk4/chenp/2024-chenpeng/data_strengthen/DialogCC/sub_rationale_caption_description/test.json"
TEST_DATA_PATH = "/root/autodl-tmp/data_strengthen/Photochat/ori_data/dev.json"
# OUTPUT_PATH = "./lora_rationale_caption_description_test_sub_rationale_caption_description_5.json"
OUTPUT_PATH = "/root/autodl-tmp/data_strengthen/Photochat/aug_data/dev.json"
MAX_IMAGES = 3

BATCH_SIZE = 1
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512

class CorpusBLEU:
    def __init__(self, max_n=2):
        self.max_n = max_n
        self.matches = Counter()
        self.totals = Counter()
        self.pred_len = 0
        self.ref_len = 0

    def add(self, pred: str, ref: str):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        self.pred_len += len(pred_tokens)
        self.ref_len += len(ref_tokens)

        for n in range(1, self.max_n + 1):
            pred_ngrams = Counter(ngrams(pred_tokens, n))
            ref_ngrams = Counter(ngrams(ref_tokens, n))

            for ng, cnt in pred_ngrams.items():
                self.matches[n] += min(cnt, ref_ngrams.get(ng, 0))
                self.totals[n] += cnt

    def compute(self):
        if self.pred_len == 0:
            return {f"BLEU-{n}": 0.0 for n in range(1, self.max_n + 1)}

        if self.pred_len > self.ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - self.ref_len / (self.pred_len + 1e-12))

        scores = {}
        for n in range(1, self.max_n + 1):
            score = bp * math.exp(
                math.log(self.matches[n] + 1e-13)
                - math.log(self.totals[n] + 1e-13)
            )
            scores[f"BLEU-{n}"] = score * 100.0
        return scores


# =====================================================
# tokenizer（完全一致）
# =====================================================
if local_rank == 0:
    print("[INFO] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    use_fast=True,
)

# =====================================================
# 模型（唯一变化：去掉 device_map=auto）
# =====================================================
if local_rank == 0:
    print("[INFO] Loading model (DDP mode)...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(device)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# =====================================================
# 数据加载（完全一致）
# =====================================================
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    if local_rank == 0:
        print(f"[INFO] Loaded {len(raw_data)} raw samples")

test_data = {k: v for i, (k, v) in enumerate(raw_data.items()) if i % 1 == 0}
dialog_items = list(test_data.items())

# ⭐ DDP 唯一数据切分点
dialog_items = dialog_items[local_rank::world_size]

# =====================================================
# build_prompt_with_truncation（一行不改）
# =====================================================
def build_prompt_with_truncation(dialogue_seq, tokenizer, max_input_tokens):
    dialogue_seq = dialogue_seq.copy()

    PROMPT = (
        "You are an expert in multimodal conversation understanding and content planning.\n\n"
        "You are given a conversation consisting of numbered utterances.\n"
        "Your task is to decide at which points in the conversation an image should be inserted.\n\n"
        "You must insert at least ONE image.\n"
        "Each image must be inserted immediately AFTER a specific utterance.\n\n"
        "Output MUST follow the exact structured format below.\n"
        "Do NOT add any text outside this format.\n\n"
        "<IMAGE_INSERTIONS>\n"
        "- utterance_id: <int>\n"
        "- utterance: <text>\n"
        "- rationale: a clear explanation of the communicative need for the image\n"
        "- description: a concrete and detailed visual description of the image content\n"
        "</IMAGE_INSERTIONS>\n\n"
        "Conversation:\n"
    )

    def token_len(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    while True:
        utterances = []
        utterance_id = 0
        last_txt_id = None
        gt_insertions = []

        i = 0
        while i < len(dialogue_seq):
            item = dialogue_seq[i]
            if item[0] == "txt":
                utterances.append(f"utterance_{utterance_id}: {item[1]}")
                last_txt_id = utterance_id
                utterance_id += 1

                if i + 1 < len(dialogue_seq) and dialogue_seq[i + 1][0] == "img":
                    gt_insertions.append({"utterance_id": last_txt_id})
                    i += 1
            i += 1

        input_text = PROMPT + "\n".join(utterances) + "\n\nAnswer:\n"

        if token_len(input_text) <= max_input_tokens:
            return input_text, gt_insertions

        removed = False
        for j in reversed(range(len(dialogue_seq))):
            if dialogue_seq[j][0] == "img":
                dialogue_seq.pop(j)
                if j - 1 >= 0 and dialogue_seq[j - 1][0] == "txt":
                    dialogue_seq.pop(j - 1)
                removed = True
                break
            elif dialogue_seq[j][0] == "txt":
                dialogue_seq.pop(j)
                removed = True
                break

        if not removed:
            return None, None

# =====================================================
# 输出解析（完全一致）
# =====================================================
def parse_image_insertions(text):
    pattern = re.compile(
        r"- utterance_id:\s*(\d+)\s*"
        r"- utterance:\s*(.*?)\s*"
        r"- rationale:\s*(.*?)\s*"
        r"- description:\s*(.*?)(?=\n- utterance_id:|\Z)",
        re.S,
    )

    results = []
    for m in pattern.finditer(text):
        results.append(
            {
                "utterance_id": int(m.group(1)),
                "rationale": m.group(3).strip(),
                "description": m.group(4).strip(),
            }
        )
    return results

def clean_description(text):
    text = text.strip()
    text = re.sub(r"</IMAGE_INSERTIONS>.*$", "", text, flags=re.S)
    return text.strip()

def build_new_dialogue(original_dialogue, final_preds):
    insert_map = {}
    for p in final_preds:
        insert_map.setdefault(p["utterance_id"], []).append(
            ["img", p["utterance_id"], p["rationale"], clean_description(p["description"])]
        )

    new_dialogue = []
    utterance_id = 0
    for item in original_dialogue:
        if item[0] == "txt":
            new_dialogue.append(item)
            if utterance_id in insert_map:
                new_dialogue.extend(insert_map[utterance_id])
            utterance_id += 1
    return new_dialogue

# =====================================================
# 指标（完全一致）
# =====================================================
tp = fp = fn = 0
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
gen_token_lens = []

# =====================================================
# 推理（逻辑完全一致，只是每个 rank 处理子集）
# =====================================================
final_output = {}

for i in tqdm(range(0, len(dialog_items), BATCH_SIZE), disable=(local_rank != 0)):
    batch = dialog_items[i:i + BATCH_SIZE]

    input_texts = []
    metas = []

    for dialog_id, dialogue_seq in batch:
        input_text, gt_insertions = build_prompt_with_truncation(
            dialogue_seq, tokenizer, MAX_INPUT_TOKENS
        )
        if input_text is None:
            continue
        input_texts.append(input_text)
        metas.append((dialog_id, dialogue_seq, gt_insertions))

    if not input_texts:
        continue

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    for b_idx, output in enumerate(outputs):
        input_len = inputs["input_ids"][b_idx].shape[-1]
        gen_token_lens.append(output.shape[-1] - input_len)

        gen_text = tokenizer.decode(
            output[input_len:], skip_special_tokens=True
        )

        dialog_id, dialogue_seq, gt_insertions = metas[b_idx]
        gt_positions = set(ins["utterance_id"] for ins in gt_insertions)

        preds = parse_image_insertions(gen_text)

        correct_preds, incorrect_preds = [], []
        for p in preds:
            if p["utterance_id"] in gt_positions:
                correct_preds.append(p)
            else:
                incorrect_preds.append(p)

        final_preds = []
        for p in correct_preds + incorrect_preds:
            # if len(final_preds) < MAX_IMAGES:
            if len(final_preds) < len(correct_preds + incorrect_preds):
                final_preds.append(p)

        pred_positions = set(p["utterance_id"] for p in final_preds)
        tp += len(pred_positions & gt_positions)
        fp += len(pred_positions - gt_positions)
        fn += len(gt_positions - pred_positions)

        final_output[dialog_id] = build_new_dialogue(dialogue_seq, final_preds)

# =====================================================
# 聚合（DDP 必需）
# =====================================================
# gathered = [None for _ in range(world_size)]
# dist.all_gather_object(gathered, (final_output, tp, fp, fn, gen_token_lens))

gathered = [None for _ in range(world_size)] if local_rank == 0 else None
dist.gather_object(
    (final_output, tp, fp, fn, gen_token_lens),
    gathered,
    dst=0
)


if local_rank == 0:
    # ===================== 聚合 =====================
    merged, TP, FP, FN, lens = {}, 0, 0, 0, []
    for d, t, f, n, l in gathered:
        merged.update(d)
        TP += t; FP += f; FN += n
        lens.extend(l)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # ===================== description 级指标 =====================
    # ===================== description 级指标（Corpus BLEU）=====================
    bleu_metric = CorpusBLEU(max_n=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []


    def get_gt_description_by_utterance_id(dialogue_seq, target_uid):
        utterance_id = 0
        for i, item in enumerate(dialogue_seq):
            if item[0] == "txt":
                if utterance_id == target_uid:
                    if i + 1 < len(dialogue_seq) and dialogue_seq[i + 1][0] == "img":
                        return dialogue_seq[i + 1][2]
                    return None
                utterance_id += 1
        return None


    for dialog_id, dialogue_seq in merged.items():
        gt_dialogue = test_data[dialog_id]

        for idx, item in enumerate(dialogue_seq):
            if item[0] == "img":
                utterance_id = item[1]
                pred_desc = item[3]
                gt_desc = get_gt_description_by_utterance_id(gt_dialogue, utterance_id)
                if gt_desc is None:
                    continue

                # ✅ corpus BLEU：只 add，不算分
                bleu_metric.add(pred_desc, gt_desc)

                # ROUGE-L（仍然是平均，和论文常见做法一致）
                rouge_l_scores.append(
                    rouge.score(gt_desc, pred_desc)["rougeL"].fmeasure
                )

    bleu_scores = bleu_metric.compute()

    # ===================== 图片数量统计 =====================
    image_count_dict = {}
    for dialogue_seq in merged.values():
        img_num = sum(1 for x in dialogue_seq if x[0] == "img")
        image_count_dict[img_num] = image_count_dict.get(img_num, 0) + 1

    total_images = sum(img_num * cnt for img_num, cnt in image_count_dict.items())

    # ===================== 输出 & 保存 =====================
    print("\n================= Evaluation =================")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\n================= Generation Length =================")
    print(f"Mean gen tokens : {np.mean(lens):.2f}")
    print(f"Max  gen tokens : {np.max(lens)}")

    print("\n================= Description Quality (Corpus) =================")
    print(f"BLEU-1   : {bleu_scores['BLEU-1']:.4f}")
    print(f"BLEU-2   : {bleu_scores['BLEU-2']:.4f}")
    print(f"ROUGE-L  : {np.mean(rouge_l_scores) * 100:.4f}")


    def strip_utterance_id(dialogue_seq):
        cleaned = []
        for item in dialogue_seq:
            if item[0] == "img":
                # ["img", utterance_id, rationale, description]
                cleaned.append(["img", item[2], item[3]])
            else:
                cleaned.append(item)
        return cleaned


    final_save_output = {
        k: strip_utterance_id(v)
        for k, v in merged.items()
    }

    # 保存 final_output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_save_output, f, indent=2, ensure_ascii=False)

    # ===================== 写入 information.txt =====================
    from datetime import datetime
    info_path = "information.txt"
    with open(info_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LORA_PATH: {LORA_PATH}\n")
        f.write(f"INPUT_FILE: {TEST_DATA_PATH}\n")
        f.write(f"OUTPUT_FILE: {OUTPUT_PATH}\n")
        f.write(f"Sample Num: {len(merged)}\n")
        f.write(f"Images Number: {total_images}\n")
        f.write(f"MAX_IMAGES: {MAX_IMAGES}\n")
        f.write(f"Image Count Dict: {image_count_dict}\n")
        f.write("\n[Evaluation]\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1       : {f1:.4f}\n")
        f.write("\n[Generation Length]\n")
        f.write(f"Mean gen tokens: {np.mean(lens):.2f}\n")
        f.write(f"Max  gen tokens: {np.max(lens)}\n")
        f.write("\n[Description Quality]\n")
        f.write(f"BLEU-1   : {bleu_scores['BLEU-1']:.4f}\n")
        f.write(f"BLEU-2   : {bleu_scores['BLEU-2']:.4f}\n")
        f.write(f"ROUGE-L  : {np.mean(rouge_l_scores) * 100:.4f}\n")

    print(f"[INFO] Experiment information appended to {info_path}")


cleanup_ddp()
