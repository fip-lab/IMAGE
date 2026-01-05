# fg.py
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class FGClip:
    def __init__(self, model_path: str, device: str):
        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

        self.model.eval()

    @torch.no_grad()
    def compute_score(self, image_path: str, text: str) -> float:
        """
        计算单张图片 + 单条文本的 fg-clip 相似度
        """
        image = Image.open(image_path).convert("RGB")

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # 用 tokenizer 的结果覆盖 processor 的 text 部分（与你原代码一致）
        inputs["input_ids"] = tokens["input_ids"]
        inputs["attention_mask"] = tokens["attention_mask"]

        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # L2 归一化
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        score = torch.sum(image_embeds * text_embeds).item()
        return score
