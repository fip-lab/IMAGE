import json
import copy
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# =====================
# 配置
# =====================
MODEL_PATH = "/disk3/model/gemma-4-31B"
LORA_PATH = "./best_lora"


TEST_PATH = "/disk2/disk4/chenp/2024-chenpeng/data_strengthen/IMAGE/new_code/dataset/NaturalConv/sub_train.json"

SAVE_PATH = "/disk2/disk4/chenp/2024-chenpeng/data_strengthen/IMAGE/new_code/dataset/NaturalConv_aug/sub_train.json"

MAX_LENGTH = 600
MAX_GEN_TOKENS = 256

# =====================
# tokenizer + model
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)

import torch.nn as nn

def unwrap_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            if hasattr(child, "linear"):
                setattr(module, name, child.linear)
            else:
                unwrap_linear(child)

unwrap_linear(base_model)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

device = next(model.parameters()).device

# =====================
# prompts（完全保持训练一致）
# =====================
TASK1_PROMPT = (
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

TASK2_PROMPT = (
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

TASK3_PROMPT = (
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

# =====================
# generate
# =====================
def generate(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    pred = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return pred

# =====================
# truncate helper
# =====================
def fit_task1_context(turns):

    turns_copy = copy.deepcopy(turns)

    while True:

        utterances = []
        utt_id = 0

        for t in turns_copy:
            if t[0] == "txt":
                utterances.append(
                    f"[{utt_id}] {t[1].strip()}"
                )
                utt_id += 1

        context_text = "\n".join(utterances)

        prompt = (
                TASK1_PROMPT +
                context_text +
                "\n\nOutput:\n"
        )

        instruction = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        input_ids = (
                instruction["input_ids"] +
                [tokenizer.eos_token_id]
        )

        if len(input_ids) <= MAX_LENGTH:
            return prompt

        for i, t in enumerate(turns_copy):

            if t[0] == "txt":

                del turns_copy[i]

                if i < len(turns_copy):
                    if turns_copy[i][0] == "img":
                        del turns_copy[i]

                break

# =====================
# task23 truncate
# =====================
def fit_task23_prompt(
        template,
        context,
        response,
        keywords=None
):

    context_copy = copy.deepcopy(context)

    while True:

        if keywords is None:

            prompt = template.format(
                context=context_copy,
                response=response
            )

        else:

            prompt = template.format(
                context=context_copy,
                response=response,
                keywords=keywords
            )

        instruction = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        input_ids = (
                instruction["input_ids"] +
                [tokenizer.eos_token_id]
        )

        if len(input_ids) <= MAX_LENGTH:
            return prompt

        if not context_copy:
            return prompt

        context_copy = "\n".join(
            context_copy.split("\n")[1:]
        )

# =====================
# regex parse
# =====================
def parse_task1_output(pred):

    results = []

    pattern = re.compile(
        r"<INSERTION>\s*"
        r"id:\s*(\d+)\s*"
        r'utterance:\s*"(.*?)"\s*'
        r"rationale:\s*(.*?)\s*"
        r"</INSERTION>",
        re.DOTALL
    )

    matches = pattern.findall(pred)

    for m in matches:

        results.append({
            "utt_id": int(m[0]),
            "utterance": m[1].strip(),
            "rationale": m[2].strip()
        })

    return results


def parse_keywords(pred):
    match = re.search(
        r"<IMAGE_KEYWORDS>(.*?)</IMAGE_KEYWORDS>",
        pred,
        re.DOTALL
    )

    if not match:
        return []

    content = match.group(1).strip()

    keywords = [
        x.strip()
        for x in content.split(",")
        if x.strip()
    ]

    # ===== 去重（保持顺序）=====
    seen = set()
    unique_keywords = []

    for kw in keywords:

        kw_lower = kw.lower()  # 忽略大小写重复

        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return unique_keywords


def parse_description(pred):
    match = re.search(
        r"<IMAGE_DESCRIPTION>(.*?)</IMAGE_DESCRIPTION>",
        pred,
        re.DOTALL
    )

    if not match:
        return pred.strip()

    return match.group(1).strip()
# =====================
# load test
# =====================
test_data = json.load(
    open(TEST_PATH, "r", encoding="utf-8")
)
# test_data = dict(list(test_data.items())[:2])  # 可调试

results = {}

# =====================
# task1 metrics
# =====================
total_tp = 0
total_fp = 0
total_fn = 0

# =====================
# inference
# =====================
from tqdm import tqdm

for sample_id, turns in tqdm(test_data.items()):

    original_dialog = copy.deepcopy(turns)

    ground_truth = []
    predictions = []
    gt_positions = set()
    new_dialog = []
    # =====================
    # extract ground truth
    # =====================
    txt_id = 0

    for i, t in enumerate(turns):

        if t[0] != "txt":
            continue

        current_txt_id = txt_id
        txt_id += 1

        # 后面是否跟着图片
        if i + 1 < len(turns):

            next_item = turns[i + 1]
            if next_item[0] == "img":
                gt_positions.add(current_txt_id)

                gt_item = {
                    "utt_id": current_txt_id,
                    "rationale": next_item[1],
                    "description": next_item[2]
                }

                ground_truth.append(gt_item)

    # =====================
    # task1
    # =====================
    task1_prompt = fit_task1_context(turns)

    task1_pred = generate(task1_prompt)

    insertions = parse_task1_output(task1_pred)

    insertion_map = {}

    for ins in insertions:
        if ins["utt_id"] not in insertion_map:
            insertion_map[ins["utt_id"]] = ins["rationale"]

    pred_positions = set(
        insertion_map.keys()
    )
    tp = len(
        pred_positions & gt_positions
    )

    fp = len(
        pred_positions - gt_positions
    )

    fn = len(
        gt_positions - pred_positions
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    # =====================
    # build dialogue
    # =====================
    history = []
    speaker_flag = True
    utt_id = 0

    for t in turns:

        if t[0] != "txt":
            continue

        txt = t[1]

        new_dialog.append(
            ["txt", txt]
        )

        speaker = (
            "User"
            if speaker_flag
            else "Assistant"
        )


        history.append(
            f"[{speaker}] {txt.strip()}"
        )

        speaker_flag = not speaker_flag

        current_utt = utt_id


        # =====================
        # need image
        # =====================
        if current_utt in insertion_map:

            rationale = insertion_map[current_utt]

            context = "\n".join(history[:-1])
            response = history[-1]

            # =====================
            # task2
            # =====================
            task2_prompt = fit_task23_prompt(
                TASK2_PROMPT,
                context,
                response
            )

            keyword_pred = generate(task2_prompt)

            keywords = parse_keywords(keyword_pred)

            if len(keywords) == 0:
                keywords = ["no_keyword"]

            keywords_text = ", ".join(keywords)


            # =====================
            # task3
            # =====================
            task3_prompt = fit_task23_prompt(
                TASK3_PROMPT,
                context,
                response,
                keywords_text
            )

            description_pred = generate(task3_prompt)

            description_pred = parse_description(description_pred)

            # =====================
            # img item
            # =====================
            pred_item = {
                "utt_id": current_utt,
                "rationale": rationale,
                "keywords": keywords_text,
                "description": description_pred
            }

            predictions.append(pred_item)

            img_item = [
                "img",
                rationale,
                description_pred,
                keywords_text
            ]

            new_dialog.append(img_item)

        utt_id += 1

    results[sample_id] = {
        "original_dialog": original_dialog,
        "new_dialog": new_dialog,
        "ground_truth": ground_truth,
        "predictions": predictions
    }
    # print(results[sample_id])

# =====================
# save
# =====================
with open(
        SAVE_PATH,
        "w",
        encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print("saved:", SAVE_PATH)
# =====================
# final task1 metrics
# =====================
precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\n====================")
print("TASK1 RESULTS")
print("====================")
print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")