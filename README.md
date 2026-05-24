# IMAGE: Enriching Multi-Modal Dialogue Dataset with Generative Models

Test data for this project can be downloaded from the anonymous repository at: https://anonymous.4open.science/r/IMAGE-0557/

This repository contains the implementation for the IMAGE framework,
which focuses on image-guided multimodal dialogue data augmentation, model fine-tuning,
and downstream task evaluation.

---

## Directory Structure
```
IMAGE/
├── README.md
├── requirement.txt
|── train_gemma/          # Gemma train
└── pipeline/             # Integrated IMAGE data augmentation pipeline

```

## Installation

```
pip install -r requirement.txt
```

---

The repository mainly consists of the following components:

- **train_gemma**: training scripts for the three-task multimodal model based on Gemma.
- **pipeline**: the implementation of the **IMAGE** framework, including all major components for multimodal dialogue data augmentation and generation.

---

## Notes

- The **Gemma3-31B** model used in this project can be replaced with either larger or smaller LLMs according to available computational resources.
- The image generation module is model-agnostic and can be substituted with other image generation tools or models as needed.
- This repository does **not** include pretrained model weights, original datasets, or source code of external tools. Users should obtain the required resources separately according to their own requirements and licenses.