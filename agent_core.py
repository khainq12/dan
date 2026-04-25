import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import cv2
from torchvision import models, transforms
from PIL import Image
import google.generativeai as genai

from db import save_to_db

# ================= GEMINI =================
API_KEY = os.getenv("AIzaSyAlN1YrDAdIvqvouA1HkuLWtKxsRnSIAmo")  # 🔥 FIX ENV

if API_KEY:
    genai.configure(api_key=API_KEY)

try:
    gemini = genai.GenerativeModel("gemini-2.5-flash") if API_KEY else None
    print("✅ Gemini loaded!")
except Exception as e:
    print("❌ Gemini init error:", e)
    gemini = None

# ================= PATH =================
BASE = r"D:\do_an_nganh\code\ai_detector_backend\models"

model_paths = [
    os.path.join(BASE, "resnet18_d1_best.pth"),
    os.path.join(BASE, "resnet18_d2_best.pth"),
    os.path.join(BASE, "resnet18_d3_best.pth"),
    os.path.join(BASE, "resnet18_d4_best.pth"),
]

DATA_DIR = r"D:\do_an_nganh\code\ai_detector_backend\dataset\Data Set 1\Data Set 1\train"

# ================= LOAD MODEL =================
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(path, map_location="cpu")

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model

models_list = [load_model(p) for p in model_paths]
print("✅ Loaded 4 models")

# ================= FEATURE EXTRACTOR =================
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

extractor = FeatureExtractor(models_list[0]).eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= FAISS =================
image_paths, labels = [], []

for label in ["real", "fake"]:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder):
        continue

    for img_name in os.listdir(folder)[:100]:
        path = os.path.join(folder, img_name)
        if path.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(path)
            labels.append(label)

embeddings = []

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            emb = extractor(x).numpy().astype("float32")[0]

        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append(emb)
    except:
        continue

embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("✅ FAISS ready")

# ================= GRADCAM =================
def gradcam(x):
    model = models_list[0]
    model.train()

    gradients, activations = [], []

    def forward_hook(m, i, o): activations.append(o)
    def backward_hook(m, gi, go): gradients.append(go[0])

    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    out = model(x)
    pred = out.argmax()

    model.zero_grad()
    out[0, pred].backward()

    g = gradients[0]
    a = activations[0]

    w = torch.mean(g, dim=(2, 3), keepdim=True)
    cam = torch.sum(w * a, dim=1).squeeze()

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam.detach().numpy(), (224, 224))

    h1.remove()
    h2.remove()

    model.eval()
    return cam

# ================= PIPELINE =================
class Pipeline:
    def run(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        outputs = []
        print("\n===== MODEL DEBUG =====")

        for i, m in enumerate(models_list):
            out = torch.softmax(m(x), dim=1)
            print(f"Model {i}: real={out[0,0].item():.4f}, fake={out[0,1].item():.4f}")
            outputs.append(out)

        # 🔥 bỏ model lỗi
        use_idx = [0, 2, 3]
        outputs = [outputs[i] for i in use_idx]

        final_out = sum(outputs) / len(outputs)

        real_prob = final_out[0][0].item()
        fake_prob = final_out[0][1].item()

        label = "fake" if fake_prob > real_prob else "real"
        conf = max(real_prob, fake_prob)

        print("Final:", label, conf)

        # ===== EMB =====
        with torch.no_grad():
            emb = extractor(x).numpy().astype("float32")[0]

        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # ===== SAVE DB =====
        try:
            save_to_db(image_path, label, conf, "HIGH", emb)
        except Exception as e:
            print("DB error:", e)

        # ===== FAISS =====
        D, I = index.search(np.array([emb], dtype="float32"), k=6)
        I = I[0][1:]

        sim_labels = [labels[i] for i in I if i < len(labels)]
        sim_paths = [image_paths[i] for i in I if i < len(image_paths)]  # 🔥 FIX

        fake_count = sim_labels.count("fake")
        real_count = sim_labels.count("real")

        sim_score = fake_count / (fake_count + real_count + 1e-6)

        risk = (
            "HIGH" if conf > 0.75
            else "MEDIUM" if conf > 0.5
            else "LOW"
        )

        return {
            "label": label,
            "confidence": conf,
            "similarity": sim_score,
            "risk": risk,
            "similar_labels": sim_labels,
            "similar_paths": sim_paths,  # 🔥 FIX
            "cam": gradcam(x)
        }

# ================= AGENT =================
class ImageAgent:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def handle(self, query, image_path):
        r = self.pipeline.run(image_path)

        base_answer = self.build_answer(r)

        if gemini:
            try:
                prompt = f"""
Bạn là AI phát hiện ảnh fake.

Label: {r['label']}
Confidence: {r['confidence']:.2f}
Risk: {r['risk']}

User: {query}

Trả lời tự nhiên 2-3 câu.
"""
                response = gemini.generate_content(prompt)
                answer = response.text.strip() if response.text else base_answer
            except:
                answer = base_answer
        else:
            answer = base_answer

        return {
            "text": answer,
            "raw": r
        }

    def build_answer(self, r):
        label = "giả" if r["label"] == "fake" else "thật"
        conf = r["confidence"]

        if conf > 0.85:
            level = "rất cao"
        elif conf > 0.7:
            level = "khá cao"
        elif conf > 0.5:
            level = "trung bình"
        else:
            level = "không chắc chắn"

        return f"Ảnh này có khả năng là {label}. Độ tin cậy {level} (≈ {conf:.2f})."