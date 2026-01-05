# IMAGE: Enriching Multi-Modal Dialogue Dataset with Generative Models

This repository contains the implementation for the IMAGE framework,
which focuses on image-guided multimodal dialogue data augmentation, model fine-tuning,
and downstream task evaluation.

---

## Directory Structure
```
IMAGE/
├── README.md
├── requirement.txt
|── code/
    ├── fineturn_qwen/        # Fine-tuning code for Qwen models
    ├── fineturn_sd_3.5/      # Fine-tuning code for Stable Diffusion 3.5
    ├── intent_pred/          # Downstream task: intent detection
    └── pipeline/             # Integrated IMAGE data augmentation pipeline

```

## Installation

```
pip install -r requirement.txt
```

---

## Code

The `code/` directory contains the main components of the IMAGE framework:

- fineturn_qwen: fine-tuning scripts for Qwen-based language models.
- fineturn_sd_3.5: fine-tuning scripts for Stable Diffusion 3.5.
- pipeline: the IMAGE data augmentation and generation pipeline.
- intent_pred: implementation of the downstream intent detection task.

---

## Notes

- The Qwen3-4B model used in this project can be easily replaced with larger variants of the Qwen series if more computational resources are available.
- The image generation module is model-agnostic and can be substituted with other image generation models as needed.
- Pretrained model weights and the complete original PhotoChat dataset are not included in this repository and should be obtained separately by users according to their own requirements and licenses.

