import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Class definitions (HAM10000) ──────────────────────────────────────────────
CLASSES = {
    0: {
        "code": "akiec",
        "name": "Actinic Keratoses",
        "risk": "Moderate",
        "color": "#F59E0B",
        "description": "Pre-cancerous rough, scaly patch caused by years of sun exposure. Can develop into squamous cell carcinoma.",
        "action": "Consult a dermatologist. Treatment options include cryotherapy, topical creams, or photodynamic therapy.",
    },
    1: {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "risk": "High",
        "color": "#EF4444",
        "description": "Most common type of skin cancer. Grows slowly and rarely spreads, but needs prompt treatment.",
        "action": "See a dermatologist immediately. Highly treatable when caught early.",
    },
    2: {
        "code": "bkl",
        "name": "Benign Keratosis",
        "risk": "Low",
        "color": "#10B981",
        "description": "Non-cancerous skin growth including seborrheic keratoses and solar lentigines. Generally harmless.",
        "action": "Usually requires no treatment. Monitor for changes. Cosmetic removal is an option.",
    },
    3: {
        "code": "df",
        "name": "Dermatofibroma",
        "risk": "Very Low",
        "color": "#10B981",
        "description": "Benign skin lesion — a small, firm bump. Common on the legs. Harmless.",
        "action": "No treatment needed. Surgical removal possible if bothersome.",
    },
    4: {
        "code": "mel",
        "name": "Melanoma",
        "risk": "Critical",
        "color": "#7F1D1D",
        "description": "The most dangerous form of skin cancer. Develops from melanocytes. Can spread to other organs.",
        "action": "Seek urgent medical attention. Early detection is critical for survival.",
    },
    5: {
        "code": "nv",
        "name": "Melanocytic Nevi",
        "risk": "Very Low",
        "color": "#10B981",
        "description": "Common moles caused by clusters of pigmented cells. Usually benign.",
        "action": "Monitor using ABCDE rule. See a doctor if mole changes in size, shape, or color.",
    },
    6: {
        "code": "vasc",
        "name": "Vascular Lesion",
        "risk": "Low",
        "color": "#3B82F6",
        "description": "Includes angiomas, angiokeratomas, and pyogenic granulomas. Blood vessel abnormalities.",
        "action": "Generally benign. Consult a dermatologist for evaluation and possible removal.",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(path="best_model.pth"):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 7)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ── Image preprocessing ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    results = []
    for idx, prob in enumerate(probs.tolist()):
        entry = dict(CLASSES[idx])
        entry["probability"] = round(prob * 100, 2)
        entry["index"] = idx
        results.append(entry)
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    results = predict(img)

    # Encode thumbnail for display
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"results": results, "thumbnail": thumb_b64})

if __name__ == "__main__":
    print("\n🔬 Skin Cancer Classifier running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
