import torch

from PIL import Image

from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    AutoModel,
    AutoImageProcessor,
)

from sentence_transformers.util import (
    pytorch_cos_sim
)

# =========================================================
# fixed path
# =========================================================

DINO_PATH = "/disk3/model/dinov2-large"

FG_CLIP_PATH = "/disk3/model/fg-clip-base"

# =========================================================
# load novelty image models
# =========================================================

def load_novelty_image_models(
    device="cuda",
):

    processor_dino = (
        AutoImageProcessor.from_pretrained(
            DINO_PATH
        )
    )

    model_dino = (
        AutoModel.from_pretrained(
            DINO_PATH
        ).to(device)
    )

    model_dino.eval()

    tokenizer_clip = (
        CLIPTokenizer.from_pretrained(
            FG_CLIP_PATH
        )
    )

    processor_clip = (
        CLIPProcessor.from_pretrained(
            FG_CLIP_PATH
        )
    )

    model_clip = (
        CLIPModel.from_pretrained(
            FG_CLIP_PATH
        ).to(device)
    )

    model_clip.eval()

    return {
        "processor_dino": processor_dino,
        "model_dino": model_dino,
        "tokenizer_clip": tokenizer_clip,
        "processor_clip": processor_clip,
        "model_clip": model_clip,
        "device": device,
    }

# =========================================================
# build context
# =========================================================

def build_context(
    dialogue,
    image_position,
    context_window,
):
    txt_turns = []

    for t in dialogue:
        if t[0] == "txt":
            txt_turns.append(t[1])


    start = max(
        0,
        image_position - context_window
    )

    end = min(
        len(txt_turns),
        image_position + context_window
    )

    context = txt_turns[start:end]

    return ". ".join(context)

# =========================================================
# novelty image score
# =========================================================

def novelty_image_score(
    models,
    image_paths,
    image_description,
    dialogue,
    image_position,
    context_window=2,
):

    processor_dino = models[
        "processor_dino"
    ]

    model_dino = models[
        "model_dino"
    ]

    processor_clip = models[
        "processor_clip"
    ]

    model_clip = models[
        "model_clip"
    ]

    device = models[
        "device"
    ]

    context = build_context(
        dialogue=dialogue,
        image_position=image_position,
        context_window=context_window,
    )

    images = [
        Image.open(x).convert("RGB")
        for x in image_paths
    ]

    # =====================================================
    # dino image features
    # =====================================================

    with torch.no_grad():

        dino_inputs = processor_dino(
            images=images,
            return_tensors="pt"
        ).to(device)

        dino_features = model_dino(
            **dino_inputs
        ).last_hidden_state[:, 0, :]

    # =====================================================
    # clip image features
    # =====================================================

    with torch.no_grad():

        image_inputs = processor_clip(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        image_outputs = (
            model_clip.get_image_features(
                **image_inputs
            )
        )

        image_features = (
            image_outputs.pooler_output
            if hasattr(
                image_outputs,
                "pooler_output"
            )
            else image_outputs
        )

    # =====================================================
    # description feature
    # =====================================================

    with torch.no_grad():

        prompt_inputs = processor_clip(
            text=[image_description],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        prompt_outputs = (
            model_clip.get_text_features(
                **prompt_inputs
            )
        )

        prompt_features = (
            prompt_outputs.pooler_output
            if hasattr(
                prompt_outputs,
                "pooler_output"
            )
            else prompt_outputs
        )

    # =====================================================
    # context feature
    # =====================================================

    with torch.no_grad():

        context_inputs = processor_clip(
            text=[context],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        context_outputs = (
            model_clip.get_text_features(
                **context_inputs
            )
        )

        context_features = (
            context_outputs.pooler_output
            if hasattr(
                context_outputs,
                "pooler_output"
            )
            else context_outputs
        )

    # =====================================================
    # compute novelty
    # =====================================================

    scores = []

    for idx in range(len(image_paths)):

        similarities = []

        for j in range(len(image_paths)):

            if idx == j:
                continue

            sim = float(
                pytorch_cos_sim(
                    dino_features[idx],
                    dino_features[j]
                )
            )

            similarities.append(sim)

        avg_similarity = (
            sum(similarities)
            / len(similarities)
        )

        prompt_penalty = float(
            1 - pytorch_cos_sim(
                image_features[idx],
                prompt_features
            )
        )

        context_penalty = float(
            1 - pytorch_cos_sim(
                image_features[idx],
                context_features
            )
        )

        novelty_score = (
            1
            - avg_similarity
            * prompt_penalty
            * context_penalty
        )

        novelty_score = max(
            0.0,
            min(1.0, novelty_score)
        )

        scores.append(
            float(novelty_score)
        )

    return scores