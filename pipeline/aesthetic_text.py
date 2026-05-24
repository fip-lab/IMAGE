import sys
from pathlib import Path
OA_ROOT = "./Open-Assistant"
sys.path.insert(0, OA_ROOT)
sys.path.insert(0,str(Path(OA_ROOT) / "model"))
import numpy as np
import torch
import model_training.models.reward_model
from transformers import AutoTokenizer
from model_training.models.reward_model import GPTNeoXRewardModel
MODEL_PATH = "/disk3/model/oasst-rm-2.1-pythia-1.4b"
# =========================================================
# normalization
# =========================================================

P1 = -4.0703

P99 = 0.9160

# =========================================================
# normalize reward score
# =========================================================

def normalize_rm(x):

    x = np.clip(
        x,
        P1,
        P99
    )

    score = (
        1
        + 4
        * (x - P1)
        / (P99 - P1)
    )

    return float(score)

# =========================================================
# load aesthetic text model
# =========================================================

def load_aesthetic_text_model(
    device="cuda",
):

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # =====================================================
    # critical patch
    # =====================================================

    GPTNeoXRewardModel.all_tied_weights_keys = {}

    GPTNeoXRewardModel._tied_weights_keys = {}

    # =====================================================
    # load model
    # =====================================================

    model = (
        GPTNeoXRewardModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            _fast_init=False,
            ignore_mismatched_sizes=True
        )
    )

    model.all_tied_weights_keys = {}

    model._tied_weights_keys = {}

    model = model.to(device)

    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }

# =========================================================
# aesthetic response score
# =========================================================

def aesthetic_response_score(
    model_dict,
    context,
    response,
    max_length=1024,
):

    tokenizer = model_dict[
        "tokenizer"
    ]

    model = model_dict[
        "model"
    ]

    device = model_dict[
        "device"
    ]

    text = (
        f"<|prompter|>{context}"
        f"<|endoftext|>"
        f"<|assistant|>{response}"
        f"<|endoftext|>"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        raw_score = (
            model(**inputs)
            .logits[0]
            .item()
        )

    norm_score = normalize_rm(
        raw_score
    )

    return float(norm_score)