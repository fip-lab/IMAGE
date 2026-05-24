import joblib
import numpy as np
import torch
import torch.nn as nn

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
MODEL_PATH = "./mlp/aesthetic_mlp.pt"
SCALER_PATH = "./mlp/aesthetic_scaler.pkl"
class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(4, 1),
        )
    def forward(self, x):
        return (
            1
            + 4 * torch.sigmoid(self.net(x))
        )

def load_aesthetic_mlp():

    model = (
        AestheticMLP()
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

def predict_aesthetic_score(
    model,
    scaler,
    image_score,
    response_score,
    gemma_score,
):
    x = np.array(
        [[
            image_score,
            response_score,
            gemma_score,
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