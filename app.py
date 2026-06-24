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
# Each key is the class index (0–6) as output by the model's final layer.
# To add/remove classes: update CLASSES dict + change the Linear layer output size in load_model().
# 'risk' and 'color' are UI hints only — not used in model inference.
CLASSES = {
    0: {
        "code": "akiec",
        "name": "Actinic Keratoses",
        "risk": "Moderate",
        "color": "#F59E0B",          # Amber — moderate risk
        "description": "Pre-cancerous rough, scaly patch caused by years of sun exposure. Can develop into squamous cell carcinoma.",
        "action": "Consult a dermatologist. Treatment options include cryotherapy, topical creams, or photodynamic therapy.",
    },
    1: {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "risk": "High",
        "color": "#EF4444",          # Red — high risk
        "description": "Most common type of skin cancer. Grows slowly and rarely spreads, but needs prompt treatment.",
        "action": "See a dermatologist immediately. Highly treatable when caught early.",
    },
    2: {
        "code": "bkl",
        "name": "Benign Keratosis",
        "risk": "Low",
        "color": "#10B981",          # Green — low/benign
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
        "color": "#7F1D1D",          # Dark red — critical risk
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
        "color": "#3B82F6",          # Blue — vascular/benign
        "description": "Includes angiomas, angiokeratomas, and pyogenic granulomas. Blood vessel abnormalities.",
        "action": "Generally benign. Consult a dermatologist for evaluation and possible removal.",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
# Architecture: DenseNet-121 pretrained backbone with a custom 7-class head.
# Checkpoint must contain 'model_state_dict' key (saved via torch.save({'model_state_dict': ...})).
# To swap backbone: replace densenet121 and update the in_features (1024) accordingly.
#   e.g. ResNet-50 → model.fc = nn.Linear(2048, NUM_CLASSES)
#        EfficientNet-B0 → model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
# To change number of classes: update the second arg of nn.Linear and len(CLASSES).
NUM_CLASSES = 7  # ← change here if you extend/reduce the dataset

def load_model(path="best_model.pth"):
    model = models.densenet121(weights=None)  # weights=None: we load our own checkpoint
    model.classifier = nn.Linear(1024, NUM_CLASSES)
    # weights_only=False needed if checkpoint stores non-tensor objects (e.g. epoch, optimizer state)
    # Switch to weights_only=True if checkpoint only has model weights (safer, PyTorch recommended)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()   # disables dropout/batchnorm training behaviour
    return model

# Resolve path relative to this file so the app works from any working directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ── Image preprocessing ───────────────────────────────────────────────────────
# Must match the preprocessing used during training exactly.
# If you retrain with different input size or normalization stats, update here too.
# mean/std below are ImageNet defaults — recompute from your dataset if fine-tuning from scratch.
INPUT_SIZE = (224, 224)          # ← update if you retrain with a different resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),                             # scales [0,255] → [0.0,1.0]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def predict(image: Image.Image):
    """
    Run inference on a PIL Image and return all 7 classes sorted by confidence.

    Returns:
        list[dict]: Each dict is a copy of the CLASSES entry plus:
            - 'probability' (float): confidence in %, rounded to 2 decimal places
            - 'index' (int): original class index (0–6)
        Sorted descending by probability (highest confidence first).
    """
    tensor = transform(image).unsqueeze(0)   # unsqueeze: [3,H,W] → [1,3,H,W] (add batch dim)
    with torch.no_grad():                    # no_grad: skip gradient tracking for faster inference
        logits = model(tensor)               # raw scores, shape [1, NUM_CLASSES]
        probs = torch.softmax(logits, dim=1)[0]   # convert to probabilities, then drop batch dim

    results = []
    for idx, prob in enumerate(probs.tolist()):
        entry = dict(CLASSES[idx])           # shallow copy so we don't mutate CLASSES
        entry["probability"] = round(prob * 100, 2)
        entry["index"] = idx
        results.append(entry)

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Renders templates/index.html — add any template context variables here if needed
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Expects a multipart/form-data POST with an 'image' file field.
    Returns JSON:
        {
            "results":   [ { class fields + probability + index }, ... ],  # sorted by confidence
            "thumbnail": "<base64 JPEG string>"                            # for frontend preview
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        # convert("RGB") handles grayscale, RGBA, and palette-mode images uniformly
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    results = predict(img)

    # ── Thumbnail generation ──────────────────────────────────────────────────
    # thumbnail() is in-place and preserves aspect ratio within the given bounds.
    # Sent back to the frontend as a base64 data URI so no static file storage is needed.
    # To change preview size: update the tuple below (width, height max bounds).
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)   # quality=85: good balance of size vs fidelity
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"results": results, "thumbnail": thumb_b64})

if __name__ == "__main__":
    print("\n🔬 Skin Cancer Classifier running at http://127.0.0.1:5000\n")
    # debug=False in production; set to True locally for auto-reload on code changes
    # To change port: update below OR pass --port flag if using `flask run`
    app.run(debug=False, port=5000)
