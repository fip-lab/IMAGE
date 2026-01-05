import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from accelerate import Accelerator

# =======================
# 配置
# =======================
# MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
# DATA_PATH  = "./no_sel/train3.json"
# OUTPUT_DIR = "./output/lora_intent_qwen3"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
DATA_PATH  = "./intent/train.json"
OUTPUT_DIR = "./output/lora_intent_qwen3"

MAX_LENGTH = 1024
BATCH_SIZE = 1
LR = 1e-4
NUM_EPOCHS = 3

# =======================
# Accelerator（⚠️ 不使用 gradient_accumulation）
# =======================
accelerator = Accelerator()
device = accelerator.device

# =======================
# 数据加载
# =======================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

raw_data = load_json(DATA_PATH)
# raw_data = raw_data[:10000]

# =======================
# Dataset
# =======================
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        context_text = (
            "### Task: Predict whether the next utterance will share an image.\n"
            "### Conversation context:\n"
        )

        speakers = ["User", "Assistant"]
        for i, txt in enumerate(example["text"]):
            context_text += f"{speakers[i % 2]}: {txt}\n"

        context_text += (
            "\n### System: Please predict whether the next utterance will share an image. "
            "Answer 'Yes' or 'No'.\nAnswer:"
        )

        label_text = " Yes" if example["label"] == 1 else " No"
        full_text = context_text + label_text

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        context_len = len(
            self.tokenizer(
                context_text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        )

        labels[:context_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_label": torch.tensor(example["label"], dtype=torch.long),
        }

# =======================
# Tokenizer / Model（⚠️ 关键：FP16 参数）
# =======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# =======================
# LoRA
# =======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.train()

# =======================
# DataLoader
# =======================
dataset = IntentDataset(raw_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# =======================
# DDP prepare
# =======================
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# =======================
# Training Loop（⚠️ 不用 accumulate）
# =======================
for epoch in range(NUM_EPOCHS):
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    loop = tqdm(dataloader, disable=not accelerator.is_main_process)

    POS_WEIGHT = 3

    for batch in loop:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss

        sample_labels = batch["sample_label"]
        weights = torch.where(
            sample_labels == 1,
            torch.tensor(POS_WEIGHT, device=loss.device),
            torch.tensor(1.0, device=loss.device)
        )
        # batch_size=1 时可以直接：
        weighted_loss = loss * weights.mean()

        accelerator.backward(weighted_loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# =======================
# Save LoRA
# =======================
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✔ LoRA saved to {OUTPUT_DIR}")
