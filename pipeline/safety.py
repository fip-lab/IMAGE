import numpy as np
import torch
import clip

from PIL import Image

from nudenet import NudeDetector

from better_profanity import profanity

import easyocr

# =========================================================
# violence checker
# =========================================================

class ViolenceChecker:

    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
    ):

        self.device = device

        self.labels = [
            "fight on a street",
            "fire on a street",
            "street violence",
            "violence in office",
            "fire in office",
        ]

        self.default_label = "Unknown"

        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device
        )

        self.text_prompts = [
            "a photo of " + l
            for l in self.labels
        ]

        self.text_features = self._encode_text(
            self.text_prompts
        )

    @torch.no_grad()
    def _encode_text(self, texts):

        tokens = clip.tokenize(
            texts
        ).to(self.device)

        feats = self.model.encode_text(
            tokens
        )

        feats = feats / feats.norm(
            dim=-1,
            keepdim=True
        )

        return feats

    @torch.no_grad()
    def _encode_image(self, image):

        image = Image.fromarray(
            image
        ).convert("RGB")

        image = self.preprocess(
            image
        ).unsqueeze(0).to(self.device)

        feats = self.model.encode_image(
            image
        )

        feats = feats / feats.norm(
            dim=-1,
            keepdim=True
        )

        return feats

    @torch.no_grad()
    def predict(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(image)

        image_features = self._encode_image(
            image
        )

        similarity = (
            image_features
            @ self.text_features.T
        )

        values, indices = similarity[0].topk(1)

        confidence = values[0].item()

        label_idx = indices[0].item()

        label = self.default_label

        return {
            "label": label,
            "confidence": confidence,
            "label_idx": label_idx,
        }

# =========================================================
# load nude detector
# =========================================================

def load_nude_detector():

    profanity.load_censor_words()

    detector = NudeDetector()

    reader = easyocr.Reader(
        ['en'],
        gpu=False
    )

    return detector, reader

# =========================================================
# load violence detector
# =========================================================

def load_violence_detector(
    device="cuda",
):

    checker = ViolenceChecker(
        device=device
    )

    return checker

# =========================================================
# nude check helper
# =========================================================

def check_nude_content(
    detections,
    threshold,
):

    exposed_parts = [

        "FEMALE_BREAST_EXPOSED",

        "MALE_GENITALIA_EXPOSED",

        "FEMALE_GENITALIA_EXPOSED",

        "BUTTOCKS_EXPOSED",

        "ANUS_EXPOSED",

        "FEET_EXPOSED",
    ]

    for detection in detections:

        if (
            detection['class'] in exposed_parts
            and detection['score'] > threshold
        ):

            return True

    return False

# =========================================================
# profanity check helper
# =========================================================

def check_adult_text(
    reader,
    image_path,
    threshold=0.25,
):

    img = Image.open(
        image_path
    ).convert("RGB")

    img_np = np.array(img)

    text_results = reader.readtext(
        img_np
    )

    for item in text_results:

        text = item[1]
        score = item[2]

        # OCR 低置信度过滤
        if score < threshold:
            continue

        if profanity.contains_profanity(
            text
        ):
            return True

    return False

# =========================================================
# nude detection
# =========================================================

def is_nsfw_image(
    detector,
    reader,
    image_path,
    threshold=0.5,
):

    try:

        detections = detector.detect(
            image_path
        )

        is_nude = check_nude_content(
            detections=detections,
            threshold=threshold,
        )

        is_adult_text = check_adult_text(
            reader=reader,
            image_path=image_path,
        )

        return (
            is_nude
            or is_adult_text
        )

    except Exception as e:

        print(f"NSFW Detection Failed: {image_path}")

        print(e)

        return True

# =========================================================
# violence detection
# =========================================================

def is_violence_image(
    checker,
    image_path,
    threshold=0.23,
):

    try:

        result = checker.predict(
            image_path
        )

        confidence = result[
            "confidence"
        ]

        is_violence = (
            confidence >= threshold
        )

        return is_violence

    except Exception as e:

        print(f"Violence Detection Failed: {image_path}")

        print(e)

        return True