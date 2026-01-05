# nude.py
from typing import List
from PIL import Image
import numpy as np

from nudenet import NudeDetector
from better_profanity import profanity
import easyocr


# =====================
# 裸体类别判定
# =====================
EXPOSED_PARTS = {
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
}


def check_nude_content(
    detections: List[dict],
    threshold: float
) -> bool:
    """True = 检测到裸露身体部位"""
    for det in detections:
        if det["class"] in EXPOSED_PARTS and det["score"] >= threshold:
            return True
    return False


def check_adult_text(
    image_path: str,
    ocr_reader: easyocr.Reader
) -> bool:
    """True = 检测到色情/成人文本"""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    texts = ocr_reader.readtext(img_np)
    for _, text, _ in texts:
        if profanity.contains_profanity(text):
            return True
    return False


class NudeChecker:
    """
    Nude / Adult Content Detector

    返回语义：
        True  -> 不安全（应过滤）
        False -> 安全
    """

    def __init__(self):
        self.nude_detector = NudeDetector()
        profanity.load_censor_words()

        # OCR 固定 CPU
        self.ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

    def check(
        self,
        image_path: str,
        nude_threshold: float
    ) -> bool:
        """
        True  = 不安全（裸露 OR 成人文本）
        False = 安全
        """
        detections = self.nude_detector.detect(image_path)

        is_nude = check_nude_content(
            detections,
            nude_threshold
        )

        is_adult_text = check_adult_text(
            image_path,
            self.ocr_reader
        )

        return is_nude or is_adult_text
