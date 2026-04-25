import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import cv2
from torchvision import models, transforms
from PIL import Image

# ================= GEMINI =================
# Tự động dùng package mới (google-genai) hoặc cũ (google-generativeai)
_gemini_mode = None  # "new" hoặc "old"
_gemini_client = None  # genai.Client (new) hoặc GenerativeModel (old)

import time

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAlN1YrDAdIvqvouA1HkuLWtKxsRnSIAmo")
GEMINI_MODEL = "gemini-2.0-flash"  # free quota cao hơn gemini-2.5-flash

try:
    from google import genai as _genai_new
    if API_KEY:
        _gemini_client = _genai_new.Client(api_key=API_KEY)
        _gemini_mode = "new"
    print("✅ Gemini loaded (google-genai)")
except ImportError:
    try:
        import google.generativeai as _genai_old
        if API_KEY:
            _genai_old.configure(api_key=API_KEY)
            _gemini_client = _genai_old.GenerativeModel(GEMINI_MODEL)
            _gemini_mode = "old"
        print("✅ Gemini loaded (google-generativeai, fallback)")
    except ImportError:
        print("⚠️ No Gemini package found. Chat will use fallback answers.")


def _call_gemini(prompt, max_retries=3):
    """Gọi Gemini với retry khi bị rate limit (429)."""
    if not _gemini_mode or not _gemini_client:
        return None

    for attempt in range(max_retries):
        try:
            if _gemini_mode == "new":
                response = _gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                return response.text.strip() if response.text else None
            elif _gemini_mode == "old":
                response = _gemini_client.generate_content(prompt)
                return response.text.strip() if response.text else None
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"⚠️ Gemini rate limited, retry sau {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                print(f"⚠️ Gemini error: {err_msg[:120]}")
                return None
    return None


from db import save_to_db

# ================= PATH =================
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

model_paths = [
    os.path.join(BASE, "resnet18_d1_best.pth"),
    os.path.join(BASE, "resnet18_d2_best.pth"),
    os.path.join(BASE, "resnet18_d3_best.pth"),
    os.path.join(BASE, "resnet18_d4_best.pth"),
]

GATE_PATH = os.path.join(BASE, "gated_ensemble_d2.pth")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Data Set 1", "Data Set 1", "train")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD MODEL =================
def load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Model file not found: {path}")
        return None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(path, map_location="cpu")

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model

# ================= GATING NETWORK =================
class GatingNet(nn.Module):
    """Cùng kiến trúc với lúc train trên Kaggle."""
    def __init__(self, num_models):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_models, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# ================= LAZY INIT =================
models_list = []
extractor = None
image_paths = []
labels = []
index = None
gate = None
EPS = 1e-8

def init_models():
    """Load models, gate, và FAISS index lazily, chỉ gọi khi cần."""
    global models_list, extractor, image_paths, labels, index, gate

    if models_list:
        return

    # ----- Load base models -----
    for p in model_paths:
        m = load_model(p)
        if m is not None:
            models_list.append(m)

    if not models_list:
        print("❌ No models loaded! Check model paths.")
        return

    print(f"✅ Loaded {len(models_list)} base models")

    # ----- Load gating network -----
    if os.path.exists(GATE_PATH):
        checkpoint = torch.load(GATE_PATH, map_location="cpu")
        gate = GatingNet(len(models_list))
        gate.load_state_dict(checkpoint["gate_state_dict"])
        gate.eval()
        print("✅ Gating network loaded")
    else:
        print(f"⚠️ Gate file not found: {GATE_PATH} — will use simple average")

    # ----- Feature extractor -----
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

    # ----- FAISS -----
    _image_paths, _labels = [], []

    for label_name in ["fake", "real"]:
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            continue

        for img_name in os.listdir(folder)[:100]:
            path = os.path.join(folder, img_name)
            if path.lower().endswith((".jpg", ".png", ".jpeg")):
                _image_paths.append(path)
                _labels.append(label_name)

    _embeddings = []

    for path in _image_paths:
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0)

            with torch.no_grad():
                emb = extractor(x).numpy().astype("float32")[0]

            emb = emb / (np.linalg.norm(emb) + EPS)
            _embeddings.append(emb)
        except (IOError, OSError, RuntimeError) as e:
            print(f"⚠️ Skip image {path}: {e}")
            continue

    if _embeddings:
        _embeddings = np.array(_embeddings, dtype="float32")
        index = faiss.IndexFlatIP(_embeddings.shape[1])
        index.add(_embeddings)
        print("✅ FAISS ready")
    else:
        print("⚠️ No embeddings built — FAISS search disabled.")

    image_paths = _image_paths
    labels = _labels

# ================= GRADCAM =================
def gradcam(x):
    model = models_list[0]
    x = x.clone().requires_grad_(True)

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
    cam = cam / (cam.max() + EPS)

    cam = cv2.resize(cam.detach().numpy(), (224, 224))

    h1.remove()
    h2.remove()

    return cam

# ================= PIPELINE =================
class Pipeline:
    def __init__(self):
        init_models()

    def run(self, image_path, skip_db=False, image_bytes=None, filename=None):
        """Chạy full pipeline. skip_db=True khi không cần lưu DB (chat reuse)."""
        init_models()

        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        # ===== ENSEMBLE =====
        print("\n===== MODEL DEBUG =====")
        gate_weights = None

        with torch.no_grad():
            probs = [torch.softmax(m(x), dim=1) for m in models_list]

            for i, p in enumerate(probs):
                print(f"Model {i}: fake={p[0,0].item():.4f}, real={p[0,1].item():.4f}")

            # ----- Gated ensemble (cùng logic với lúc train) -----
            if gate is not None:
                concat_probs = torch.cat(probs, dim=1)
                weights = gate(concat_probs)

                stacked_probs = torch.stack(probs, dim=1)
                final_out = (weights.unsqueeze(2) * stacked_probs).sum(dim=1)

                gate_weights = weights[0].tolist()
                print(f"Gate weights: {[f'{w:.3f}' for w in gate_weights]}")
            else:
                # Fallback: simple average (không có gate)
                final_out = torch.stack(probs).mean(dim=0)

        # ===== LABEL MAPPING =====
        # ImageFolder sắp xếp alphabet: fake=0, real=1
        # (Giống với lúc train gate trên Kaggle)
        fake_prob = final_out[0][0].item()
        real_prob = final_out[0][1].item()

        label = "fake" if fake_prob > real_prob else "real"
        conf = max(fake_prob, real_prob)

        print(f"Final: {label} ({conf:.4f})")

        # ===== EMB =====
        with torch.no_grad():
            emb = extractor(x).numpy().astype("float32")[0]

        emb = emb / (np.linalg.norm(emb) + EPS)

        # ===== RISK =====
        if label == "fake":
            if conf > 0.85:
                risk = "HIGH"
            elif conf > 0.7:
                risk = "MEDIUM"
            else:
                risk = "LOW"
        else:
            risk = "SAFE"

        # ===== SAVE DB =====
        if not skip_db:
            try:
                save_to_db(image_path, label, conf, risk, emb, image_bytes, filename)
            except Exception as e:
                print("DB error:", e)

        # ===== FAISS =====
        sim_labels = []
        sim_paths = []

        if index is not None and len(image_paths) > 0:
            D, I = index.search(np.array([emb], dtype="float32"), k=6)
            I = I[0][1:]

            sim_labels = [labels[i] for i in I if i < len(labels)]
            sim_paths = [image_paths[i] for i in I if i < len(image_paths)]

        fake_count = sim_labels.count("fake")
        real_count = sim_labels.count("real")

        sim_score = fake_count / (fake_count + real_count + EPS)

        return {
            "label": label,
            "confidence": conf,
            "similarity": sim_score,
            "risk": risk,
            "similar_labels": sim_labels,
            "similar_paths": sim_paths,
            "cam": gradcam(x)
        }

    def get_embedding(self, image_path):
        """Trích xuất embedding vector từ ảnh (cho similar search)."""
        init_models()
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = extractor(x).numpy().astype("float32")[0]
        return emb / (np.linalg.norm(emb) + EPS)

# ================= AGENT =================
class ImageAgent:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, image_path, image_bytes=None, filename=None):
        """Chạy prediction + lưu DB (chỉ gọi 1 lần per ảnh)."""
        r = self.pipeline.run(image_path, skip_db=False, image_bytes=image_bytes, filename=filename)
        return {"text": self.build_answer(r), "raw": r}

    def chat(self, query, prev_result):
        """Chat reuse kết quả đã có, KHÔNG chạy lại model, KHÔNG lưu DB."""
        r = prev_result["raw"]

        prompt = f"""Bạn là AI phát hiện ảnh fake.

Label: {r['label']}
Confidence: {r['confidence']:.2f}
Risk: {r['risk']}

User: {query}

Trả lời tự nhiên 2-3 câu."""

        answer = _call_gemini(prompt)
        if not answer:
            answer = self.build_answer(r)

        return {"text": answer, "raw": r}

    def handle(self, query, image_path):
        """Giữ lại cho backward compat — chạy full pipeline."""
        r = self.pipeline.run(image_path, skip_db=True)

        prompt = f"""Bạn là AI phát hiện ảnh fake.

Label: {r['label']}
Confidence: {r['confidence']:.2f}
Risk: {r['risk']}

User: {query}

Trả lời tự nhiên 2-3 câu."""

        answer = _call_gemini(prompt)
        if not answer:
            answer = self.build_answer(r)

        return {"text": answer, "raw": r}

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
