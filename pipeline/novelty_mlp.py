# =========================================================
# novelty_mlp.py
# =========================================================

import joblib
import numpy as np

import torch
import torch.nn as nn

# =========================================================
# device
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =========================================================
# path
# =========================================================

MODEL_PATH = "./mlp/novelty_mlp.pt"

SCALER_PATH = "./mlp/novelty_scaler.pkl"

# =========================================================
# mlp
# =========================================================

class NoveltyMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(3, 16),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(16, 1),
        )

    def forward(self, x):

        return (
            1
            + 4 * torch.sigmoid(self.net(x))
        )

# =========================================================
# load
# =========================================================

def load_novelty_mlp():

    model = (
        NoveltyMLP()
        .to(DEVICE)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler

# =========================================================
# predict
# =========================================================

def predict_novelty_score(
    model,
    scaler,
    novelty_image_score,
    novelty_description_score,
    gemma_novelty_score,
):

    x = np.array(
        [[
            novelty_image_score,
            novelty_description_score,
            gemma_novelty_score,
        ]],
        dtype=np.float32,
    )

    x = scaler.transform(x)

    x = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        score = model(x).item()

    return float(score)