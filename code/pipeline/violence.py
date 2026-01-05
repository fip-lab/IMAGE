# violence.py
import clip
import torch
import numpy as np
from PIL import Image


class ViolenceChecker:
    """
    CLIP-based Violence Detector
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "ViT-B/32",
        violence_threshold: float = 0.23
    ):
        self.device = device
        self.model_name = model_name
        self.violence_threshold = violence_threshold

        # labels（与你的 settings.yaml 完全一致）
        self.labels = [
            "fight on a street",
            "fire on a street",
            "street violence",
            "violence in office",
            "fire in office",
        ]

        self.default_label = "Unknown"

        # CLIP model
        self.model, self.preprocess = clip.load(
            self.model_name,
            device=self.device
        )

        # 文本特征
        self.text_prompts = [
            "a photo of " + label for label in self.labels
        ]
        self.text_features = self._encode_text(self.text_prompts)

    @torch.no_grad()
    def _encode_text(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def _encode_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image_tensor)

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        返回:
        {
            'label': str,
            'confidence': float
        }
        """
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        image_features = self._encode_image(image)

        # cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(
            dim=-1, keepdim=True
        )

        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)

        confidence = abs(values[0].item())
        label_index = indices[0].item()

        label = self.default_label
        if confidence >= self.violence_threshold:
            label = self.labels[label_index]

        return {
            "label": label,
            "confidence": confidence
        }
