import copy
import re

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

# =========================================================
# fixed path
# =========================================================

MODEL_PATH = "/disk3/model/gemma-4-31B"

LORA_PATH = "./gemma_lora"

# =========================================================
# description generation config
# =========================================================

DESCRIPTION_MAX_LENGTH = 600

DESCRIPTION_MAX_GEN_TOKENS = 256

# =========================================================
# scoring config
# =========================================================

SCORING_MAX_LENGTH = 800

SCORING_MAX_GEN_TOKENS = 64

# =========================================================
# description prompt
# =========================================================

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

# =========================================================
# scoring prompt
# =========================================================

SCORING_PROMPT = """
<TASK_MULTIMODAL_DIALOGUE_SCORING>

You are an expert in multimodal dialogue evaluation.

You are given a short multimodal conversation.

In the dialogue, content inside:

[Image Description] ... [/Image Description]

represents an ACTUAL image shared in the conversation and should be treated as the shared image itself.

Your task is to independently evaluate the OVERALL multimodal dialogue on TWO dimensions.

1) Aesthetic Score

Evaluate the overall aesthetic quality by considering:
- dialogue text aesthetics (fluency, emotional appeal, engagement)
- image aesthetics reflected by the image description
- image-text aesthetic consistency

2) Novelty Score

Evaluate the overall novelty by considering:
- dialogue text novelty (interestingness, originality, informativeness)
- image novelty reflected by the image description
- image-text novelty

IMPORTANT:
- Aesthetic and Novelty are TWO INDEPENDENT dimensions.
- A high aesthetic score does NOT imply a high novelty score.
- A high novelty score does NOT imply a high aesthetic score.
- You MUST score them separately.

Score scale:
1 = very low
2 = low
3 = medium
4 = high
5 = very high

Evaluate the WHOLE multimodal dialogue rather than only the text or image.

Output format:
<AESTHETIC_SCORE>number</AESTHETIC_SCORE>
<NOVELTY_SCORE>number</NOVELTY_SCORE>

Example 1:

Dialogue:
[User] I really want a dog.
[Assistant] What type would you get?
[User] I love pugs. [Image Description] this pug exudes an air of calm and peace . | Image of a cute pug puppy.[/Image Description]
[Assistant] Pugs are adorable.

Output:
<AESTHETIC_SCORE>4</AESTHETIC_SCORE>
<NOVELTY_SCORE>2</NOVELTY_SCORE>


Example 2:

Dialogue:
[User] Look what I ate.
[Assistant] okay
[User] Fries again. [Image Description] a blurry photo of plain fries on a plate.[/Image Description]
[Assistant] cool

Output:
<AESTHETIC_SCORE>2</AESTHETIC_SCORE>
<NOVELTY_SCORE>1</NOVELTY_SCORE>

Now evaluate the following dialogue.

Dialogue:
{dialogue}

Output:
<AESTHETIC_SCORE>
""".strip()
# =========================================================
# unwrap linear
# =========================================================

def unwrap_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            if hasattr(child, "linear"):
                setattr(module, name, child.linear)
            else:
                unwrap_linear(child)

# =========================================================
# load gemma
# =========================================================

def load_gemma(MODEL_PATH, LORA_PATH, device):

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
        # device_map="auto",
        attn_implementation="sdpa"
    )

    unwrap_linear(base_model)

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    )

    model.eval()

    return model, tokenizer

# =========================================================
# fit description prompt
# =========================================================

def fit_description_prompt(
    tokenizer,
    context,
    response,
    keywords
):

    context_copy = copy.deepcopy(context)

    while True:

        prompt = TASK3_PROMPT.format(
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
            instruction["input_ids"]
            + [tokenizer.eos_token_id]
        )

        if len(input_ids) <= DESCRIPTION_MAX_LENGTH:

            return prompt

        if not context_copy:

            return prompt

        context_copy = "\n".join(
            context_copy.split("\n")[1:]
        )

# =========================================================
# fit scoring prompt
# =========================================================

def fit_score_prompt(
    tokenizer,
    dialogue
):

    dialogue_lines = dialogue.split("\n")

    while True:

        prompt = SCORING_PROMPT.format(
            dialogue="\n".join(dialogue_lines)
        )

        instruction = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        input_ids = (
            instruction["input_ids"]
            + [tokenizer.eos_token_id]
        )

        if len(input_ids) <= SCORING_MAX_LENGTH:

            return prompt

        if len(dialogue_lines) <= 1:

            return prompt

        dialogue_lines = dialogue_lines[1:]

# =========================================================
# parse description
# =========================================================

def parse_description(text):

    match = re.search(
        r"<IMAGE_DESCRIPTION>(.*?)</IMAGE_DESCRIPTION>",
        text,
        re.DOTALL
    )

    if match:

        return match.group(1).strip()

    return text.strip()

# =========================================================
# parse score
# =========================================================

def parse_score(text):

    aesthetic_match = re.search(
        r"(?:<AESTHETIC_SCORE>)?\s*(\d)\s*</AESTHETIC_SCORE>",
        text,
        re.DOTALL
    )

    novelty_match = re.search(
        r"<NOVELTY_SCORE>\s*(\d)\s*</NOVELTY_SCORE>",
        text,
        re.DOTALL
    )

    aesthetic_score = (
        float(aesthetic_match.group(1))
        if aesthetic_match
        else -1
    )

    novelty_score = (
        float(novelty_match.group(1))
        if novelty_match
        else -1
    )

    return aesthetic_score, novelty_score

# =========================================================
# build multimodal dialogue
# =========================================================

def build_multimodal_dialogue(
    turns,
    txt_index,
    image_description,
    context_window
):

    txt_turns = turns

    start = max(0, txt_index - context_window)
    end = min(
        len(txt_turns),
        txt_index + context_window
    )

    dialogue_lines = []

    for idx in range(start, end):

        speaker = (
            "User"
            if idx % 2 == 0
            else "Assistant"
        )

        text = txt_turns[idx].strip()

        # 只对目标response拼接目标img
        if idx == txt_index:

            text = (
                text
                + " "
                + f"[Image Description] "
                + f"{image_description}"
                + "[/Image Description]"
            )

        dialogue_lines.append(
            f"[{speaker}] {text}"
        )

    return "\n".join(dialogue_lines)

# =========================================================
# generate description
# =========================================================

def gemma_generate_description(
    model,
    tokenizer,
    context,
    response,
    keywords,
    use_keywords=True
):
    if not use_keywords:
        keywords = ""

    device = next(model.parameters()).device

    prompt = fit_description_prompt(
        tokenizer=tokenizer,
        context=context,
        response=response,
        keywords=keywords,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=DESCRIPTION_MAX_LENGTH
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=DESCRIPTION_MAX_GEN_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    pred = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    description = parse_description(pred)

    return description

# =========================================================
# gemma score
# =========================================================

def gemma_score(
    model,
    tokenizer,
    turns,
    image_position,
    image_description,
    context_window
):

    device = next(model.parameters()).device

    multimodal_dialogue = (
        build_multimodal_dialogue(
            turns,
            image_position,
            image_description,
            context_window
        )
    )

    prompt = fit_score_prompt(
        tokenizer,
        multimodal_dialogue
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SCORING_MAX_LENGTH
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=SCORING_MAX_GEN_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            min_new_tokens=8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    pred = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    aesthetic_score, novelty_score = parse_score(pred)

    return aesthetic_score, novelty_score