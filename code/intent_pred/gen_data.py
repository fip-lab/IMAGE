import json
import os,random
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer

# ======================
# 路径配置
# ======================
# INPUT_PATH = "../Photochat/no_sel/train.json"
# OUTPUT_PATH = "./no_sel/train2.json"
# MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
INPUT_PATH = "./ori_data/train.json"
OUTPUT_PATH = "./intent/train.json"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
MAX_LENGTH = 1024

# ======================
# IO
# ======================
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================
# 构建子样本
# ======================
def build_intent_samples(
    raw_data: Dict[str, List[List[Any]]],
    tokenizer: AutoTokenizer,
    max_length: int,
    neg_pos_ratio: int = 99,  # 负样本最大倍数
) -> List[Dict[str, Any]]:

    samples = []

    def build_prompt_from_history(txt_history: List[str]) -> str:
        prompt = (
            "### Task: Predict whether the next utterance will share an image.\n"
            "### Conversation context:\n"
        )
        speakers = ["User", "Assistant"]
        for i, txt in enumerate(txt_history):
            prompt += f"{speakers[i % 2]}: {txt}\n"
        prompt += (
            "\n### System: Please predict whether the next utterance will share an image. "
            "Answer 'Yes' or 'No'. Only output one word.\nAnswer:"
        )
        return prompt

    LABEL_TOKEN_LEN = 2  # " Yes" / " No"

    for dialogue_id, dialogue in tqdm(raw_data.items(), desc="Processing dialogues"):
        txt_positions = [idx for idx, item in enumerate(dialogue) if item[0] == "txt"]

        pos_samples = []
        neg_samples = []

        # 构建每个 txt 的子样本
        for t_i, txt_idx in enumerate(txt_positions):
            txt_history = [dialogue[i][1] for i in txt_positions[: t_i + 1]]

            next_is_img = txt_idx + 1 < len(dialogue) and dialogue[txt_idx + 1][0] == "img"
            label = 1 if next_is_img else 0

            # ===== history-level 裁剪 =====
            while True:
                prompt = build_prompt_from_history(txt_history)
                token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids) + LABEL_TOKEN_LEN

                if token_len <= max_length:
                    break
                if len(txt_history) > 1:
                    txt_history.pop(0)
                else:
                    break

            sample = {
                "dialogue_id": dialogue_id,
                "text": txt_history,
                "label": label
            }

            if label == 1:
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)

        # ===== 控制负样本数量 =====
        if len(pos_samples) > 0:
            max_neg = neg_pos_ratio * len(pos_samples)
            if len(neg_samples) > max_neg:
                neg_samples = random.sample(neg_samples, max_neg)
        else:
            # 对话里没有正样本，保留少量负样本避免完全丢失
            if len(neg_samples) > 0:
                neg_samples = random.sample(neg_samples, min(len(neg_samples), 1))

        # 合并
        samples.extend(pos_samples + neg_samples)

    return samples


# ======================
# 主函数
# ======================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    raw_data = load_json(INPUT_PATH)
    samples = build_intent_samples(raw_data, tokenizer, max_length=MAX_LENGTH)
    save_json(samples, OUTPUT_PATH)
    print(f"✔ Saved {len(samples)} intent samples to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
