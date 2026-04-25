# import torch
# import torchvision.transforms as transforms
# import numpy as np

# # Tạm thời fake model để test server
# def predict_image(image):
#     # trả random để test API trước
#     return np.random.rand()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

BASE_MODEL_FILES = [
    "models/resnet18_d1_best.pth",
    "models/resnet18_d2_best.pth",
    "models/resnet18_d3_best.pth",
    "models/resnet18_d4_best.pth",
]

GATE_PATH = "models/gated_ensemble_d2.pth"

# =========================
# LOAD BASE MODELS
# =========================
def load_base_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

base_models = [load_base_model(p) for p in BASE_MODEL_FILES]

# =========================
# LOAD GATE
# =========================
class GatingNet(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CLASSES * num_models, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

checkpoint = torch.load(GATE_PATH, map_location=DEVICE)
gate = GatingNet(len(base_models))
gate.load_state_dict(checkpoint["gate_state_dict"])
gate.to(DEVICE)
gate.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = [F.softmax(m(image), dim=1) for m in base_models]

        concat_probs = torch.cat(probs, dim=1)
        weights = gate(concat_probs)

        stacked = torch.stack(probs, dim=1)
        final_probs = (weights.unsqueeze(2) * stacked).sum(dim=1)

    ai_prob = final_probs[0][1].item()  # class 1 = AI
    return ai_prob