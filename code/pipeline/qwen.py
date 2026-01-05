# qwen.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Any

class QwenGenerator:
    def __init__(
        self,
        base_model_path: str,
        lora_path: str = None,
        use_lora: bool = False,
        device: str = "cuda:0",
        max_input_tokens: int = 1024,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            # device_map="auto",
        )

        # Apply LoRA if needed
        if use_lora and lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.to(self.device)
        self.model.eval()

    # =========================================
    # Prompt 构建 + 超长裁剪
    # =========================================
    def build_prompt_with_truncation(self, dialogue_seq: List[List[Any]]):
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
            "- utterance_id: <int>  # the index of the utterance after which the image is inserted\n"
            "- utterance: <text>    # the exact utterance text at this position\n"
            "- rationale: a clear explanation of the communicative need for the image\n"
            "- description: a concrete and detailed visual description of the image content\n"
            "</IMAGE_INSERTIONS>\n\n"
            "Conversation:\n"
        )

        def token_len(text):
            return len(self.tokenizer(text, add_special_tokens=False).input_ids)

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

            if token_len(input_text) <= self.max_input_tokens:
                return input_text, gt_insertions

            # 超长裁剪：从后往前删
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

            # 如果删完还不够，循环会继续，直到 token_len(input_text) <= max_input_tokens

    # =========================================
    # 文本解析
    # =========================================
    def parse_image_insertions(self, text: str) -> List[Dict[str, Any]]:
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

    def clean_description(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"</IMAGE_INSERTIONS>.*$", "", text, flags=re.S)
        return text.strip()

    # =========================================
    # 主接口：生成图片描述
    # =========================================
    def generate_captions(self, dialogue_seq: List[List[Any]], seed: int) -> List[Dict[str, Any]]:
        torch.manual_seed(seed)
        import random
        random.seed(seed)

        input_text, _ = self.build_prompt_with_truncation(dialogue_seq)
        if input_text is None:
            return []

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p
            )

        gen_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        patterns = self.parse_image_insertions(gen_text)

        results = []
        for p in patterns:
            uid = p["utterance_id"]
            results.append({
                "caption": p["description"].strip(),
                "utterance_id": uid,
                "rationale": p["rationale"].strip()
            })
        return results
