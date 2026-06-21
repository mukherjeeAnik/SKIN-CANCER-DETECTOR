# DermaScan — Skin Lesion Classifier

> **⚠️ Medical Disclaimer:** This tool is for **educational and research purposes only**. 
> It does **not** constitute medical advice, diagnosis, or treatment.
> Always consult a qualified dermatologist for any skin concerns.

---

## Table of Contents

- [Overview](#overview) 
- [Demo](#demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Supported Classes](#supported-classes)
- [Model Details](#model-details)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**DermaScan** is a locally-hosted web application that classifies dermoscopic skin lesion images into 7 diagnostic categories using a DenseNet-121 model trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection).

**Key properties:**

- Fully offline — no data leaves your machine
- CPU-only inference — no GPU required
- Sub-second prediction on modern hardware
- Returns ranked probabilities across all 7 classes

---

## Demo

```
Upload image → GET /predict → JSON response with ranked class probabilities + base64 thumbnail
```

Example response:

```json
{
  "thumbnail": "<base64-encoded JPEG>",
  "results": [
    {
      "index": 5,
      "code": "nv",
      "name": "Melanocytic Nevi",
      "risk": "Very Low",
      "color": "#10B981",
      "probability": 94.31,
      "description": "Common moles caused by clusters of pigmented cells. Usually benign.",
      "action": "Monitor using ABCDE rule. See a doctor if mole changes in size, shape, or color."
    },
    ...
  ]
}
```

> **[REVISION NOTE]** Add a GIF or screenshot here once the UI is finalized.
> Recommended dimensions: 900×500 px, saved as `assets/demo.gif`.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Browser (Client)                │
│   Upload image  ──►  POST /predict                  │
│   Render result ◄──  JSON { results, thumbnail }    │
└─────────────────────────┬───────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────┐
│                   Flask Server (app.py)              │
│                                                     │
│  /predict route                                     │
│    │                                                │
│    ├─ PIL Image decode + RGB conversion             │
│    ├─ Resize → 224×224, ImageNet normalize          │
│    ├─ DenseNet-121 forward pass (CPU)               │
│    ├─ Softmax → 7-class probability vector          │
│    └─ Sort + serialize + encode thumbnail           │
└─────────────────────────────────────────────────────┘
```

**Model pipeline:**

```
Raw Image
    │
    ▼
Resize (224×224)
    │
    ▼
ToTensor  →  [0.0, 1.0]
    │
    ▼
Normalize (ImageNet μ/σ)
    │
    ▼
DenseNet-121 backbone (pretrained ImageNet weights, fine-tuned on HAM10000)
    │
    ▼
Linear(1024 → 7)
    │
    ▼
Softmax  →  Class probabilities (sums to 1.0)
```

> **[REVISION NOTE]** If you swap the backbone (e.g. EfficientNet-B3, ResNet-50),
> update this diagram and the in-features value in `load_model()`.

---

## Project Structure

```
dermascan/
│
├── app.py                  # Flask app — routes, model loading, inference
├── best_model.pth          # Trained model checkpoint (NOT included in repo — see Setup)
├── requirements.txt        # Python dependencies
│
├── templates/
│   └── index.html          # Frontend UI (Jinja2 template)
│
├── static/                 # (Optional) CSS, JS, favicon
│
└── README.md
```

> **[REVISION NOTE]** If you add a `static/` directory, a `Dockerfile`, or a `tests/` folder,
> update this tree accordingly.

---

## Setup & Installation

### Prerequisites

| Requirement   | Minimum Version | Notes                              |
|---------------|-----------------|------------------------------------|
| Python        | 3.9+            | 3.10 or 3.11 recommended           |
| pip           | 22+             | `pip install --upgrade pip`        |
| RAM           | 2 GB free       | DenseNet-121 ~30 MB; tensors ~1 GB |
| Disk          | 500 MB free     | Model + dependencies               |

> **[REVISION NOTE]** If you upgrade PyTorch or add new dependencies, update the table above
> and `requirements.txt` together. Mismatches between these two are the #1 source of setup failures.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/dermascan.git
cd dermascan
```

> **[REVISION NOTE]** Replace the URL above once the repo is public.

---

### Step 2 — Place the model checkpoint

Download `best_model.pth` and place it in the project root:

```
dermascan/
└── best_model.pth   ← here
```

The checkpoint must contain the key `model_state_dict`.
To verify:

```python
import torch
ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
print(ckpt.keys())   # should include 'model_state_dict'
```

> **[REVISION NOTE]** If you retrain and save additional keys (e.g. `optimizer_state_dict`,
> `epoch`, `val_accuracy`), document them here so future contributors know what to expect.

---

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

### Step 4 — Install dependencies

**Standard install (all platforms):**

```bash
pip install -r requirements.txt
```

**Windows — CPU-only PyTorch (smaller download, ~500 MB faster):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask Pillow
```

**Linux / macOS — CPU-only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask Pillow
```

> **[REVISION NOTE]** The `--index-url` flag pins to a CPU-only wheel.
> Remove it if you want CUDA support and have a compatible GPU + CUDA toolkit.
> Tested with PyTorch 2.x; update `requirements.txt` if you change the version.

---

## Running the App

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

The terminal will print:

```
Loading model from /path/to/best_model.pth...
Model loaded successfully!

🔬 Skin Cancer Classifier running at http://127.0.0.1:5000
```

**To stop the server:** `Ctrl + C`

> **[REVISION NOTE]** For production deployment, do **not** use `app.run()` directly.
> Use a production WSGI server instead:
> ```bash
> pip install gunicorn
> gunicorn -w 2 -b 0.0.0.0:5000 app:app
> ```
> Also set `debug=False` and configure a reverse proxy (nginx) in front of it.

---

## API Reference

### `GET /`

Returns the main HTML UI.

---

### `POST /predict`

Accepts a skin lesion image and returns classification results.

**Request:**

| Parameter | Type           | Required | Description                       |
|-----------|----------------|----------|-----------------------------------|
| `image`   | File (binary)  | ✅        | JPEG, PNG, or any PIL-readable format |

Content-Type: `multipart/form-data`

**Example (curl):**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@/path/to/lesion.jpg"
```

**Example (Python):**

```python
import requests

with open("lesion.jpg", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"image": f}
    )

data = response.json()
top_prediction = data["results"][0]
print(f"{top_prediction['name']}: {top_prediction['probability']}%")
```

**Success Response — `200 OK`:**

```json
{
  "thumbnail": "<base64 JPEG string, max 300×300 px>",
  "results": [
    {
      "index": 0,
      "code": "akiec",
      "name": "Actinic Keratoses",
      "risk": "Moderate",
      "color": "#F59E0B",
      "probability": 72.14,
      "description": "...",
      "action": "..."
    }
  ]
}
```

`results` is sorted descending by `probability`. All 7 classes are always returned.

**Error Responses:**

| Status | Body                              | Cause                            |
|--------|-----------------------------------|----------------------------------|
| `400`  | `{"error": "No image uploaded"}`  | `image` field missing in request |
| `400`  | `{"error": "Invalid image file"}` | File is corrupt or unreadable    |

> **[REVISION NOTE]** If you add new endpoints (e.g. `/health`, `/classes`, batch prediction),
> document them in this section following the same format.

---

## Supported Classes

HAM10000 contains 7 diagnostic categories:

| Index | Code    | Name                  | Risk Level | Notes                                          |
|-------|---------|-----------------------|------------|------------------------------------------------|
| 0     | `akiec` | Actinic Keratoses     | Moderate   | Pre-cancerous; can progress to SCC             |
| 1     | `bcc`   | Basal Cell Carcinoma  | High       | Most common skin cancer; rarely metastasizes   |
| 2     | `bkl`   | Benign Keratosis      | Low        | Includes seborrheic keratoses & solar lentigines |
| 3     | `df`    | Dermatofibroma        | Very Low   | Benign fibrous nodule                          |
| 4     | `mel`   | Melanoma              | Critical   | Most dangerous; can metastasize to organs      |
| 5     | `nv`    | Melanocytic Nevi      | Very Low   | Common moles; monitor with ABCDE rule          |
| 6     | `vasc`  | Vascular Lesion       | Low        | Angiomas, angiokeratomas, pyogenic granulomas  |

> **[REVISION NOTE]** Index order is fixed by how the model was trained and must match
> the `CLASSES` dict in `app.py` exactly. If you retrain with a different class order or
> add/remove classes, update BOTH this table AND `CLASSES` in `app.py`, and change
> `nn.Linear(1024, 7)` → `nn.Linear(1024, N)` where N is your new class count.

---

## Model Details

| Property           | Value                                                       |
|--------------------|-------------------------------------------------------------|
| Architecture       | DenseNet-121                                                |
| Backbone weights   | ImageNet (fine-tuned on HAM10000)                           |
| Classifier head    | `Linear(in_features=1024, out_features=7)`                  |
| Input size         | 224 × 224 px (auto-resized, aspect ratio not preserved)     |
| Preprocessing      | ImageNet normalization — μ `[0.485, 0.456, 0.406]`, σ `[0.229, 0.224, 0.225]` |
| Output             | 7-class softmax probability vector                          |
| Inference device   | CPU (MPS / CUDA not used)                                   |
| Checkpoint format  | `torch.save({'model_state_dict': ...})`                     |
| Framework          | PyTorch 2.x + torchvision                                   |

> **[REVISION NOTE]** If you retrain with different normalization stats (e.g., computed from
> HAM10000 directly instead of ImageNet), update μ/σ here and in `app.py`'s `transform`.
> Mismatch between training and inference normalization is a silent accuracy killer.

---

## Performance

| Metric             | Value   |
|--------------------|---------|
| Validation Accuracy | ~89.45% |
| Dataset            | HAM10000 (10,015 dermoscopic images) |
| Train/Val split    | 80% / 20%                            |
| Inference time (CPU) | ~150–400 ms per image (hardware-dependent) |
| Model size         | ~30 MB                               |

> **[REVISION NOTE]** Update this table after every retraining run.
> Add per-class precision/recall/F1 if available — overall accuracy alone can be misleading
> on the imbalanced HAM10000 dataset (nv class is ~67% of samples).
> Recommended: log experiments with [Weights & Biases](https://wandb.ai) or MLflow
> and link the run here.

**Class distribution in HAM10000 (approximate):**

| Class   | Samples | % of Dataset |
|---------|---------|--------------|
| nv      | ~6,705  | 67%          |
| mel     | ~1,113  | 11%          |
| bkl     | ~1,099  | 11%          |
| bcc     | ~514    | 5%           |
| akiec   | ~327    | 3%           |
| vasc    | ~142    | 1.4%         |
| df      | ~115    | 1.1%         |

> **[REVISION NOTE]** High class imbalance means the model will naturally favour `nv`.
> If you retrain, consider weighted cross-entropy loss or oversampling strategies.
> Document any changes to the training strategy here.

---

## Troubleshooting

### `FileNotFoundError: best_model.pth not found`

The model checkpoint is not in the project root. See [Step 2 of Setup](#step-2--place-the-model-checkpoint).

---

### `RuntimeError: size mismatch for classifier.weight`

The checkpoint was trained with a different number of classes or a different backbone.
Verify the architecture matches: `DenseNet-121` with `Linear(1024, 7)`.

```python
import torch
ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
for k, v in ckpt["model_state_dict"].items():
    if "classifier" in k:
        print(k, v.shape)
# Expected:
# classifier.weight  torch.Size([7, 1024])
# classifier.bias    torch.Size([7])
```

---

### `ModuleNotFoundError: No module named 'torch'`

Your virtual environment is not activated, or dependencies are not installed.

```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### Server starts but browser shows nothing / 404

Confirm `templates/index.html` exists. Flask's `render_template()` looks for templates in
a `templates/` subdirectory relative to `app.py`.

---

### Prediction is very slow (> 5 seconds)

DenseNet-121 on CPU takes ~150–400 ms on modern hardware. Values above 1–2 seconds suggest:

- An older / low-power CPU
- Contention from other processes

This is normal — the app is not designed for high-throughput serving.

> **[REVISION NOTE]** If you add GPU support later (`model.to("cuda")`), note the expected
> GPU inference time here and update the Prerequisites table with CUDA/MPS requirements.

---

## Roadmap

- [ ] Add per-class ABCDE rule overlay in the UI
- [ ] Grad-CAM heatmap visualization to highlight decision regions
- [ ] Batch prediction endpoint (`POST /predict/batch`)
- [ ] Docker container for one-command deployment
- [ ] ONNX export for lighter inference without PyTorch
- [ ] Per-class precision/recall metrics in `/health` endpoint
- [ ] Mobile-responsive UI

> **[REVISION NOTE]** Move items to a `CHANGELOG.md` or GitHub milestone once completed.
> Update version tags accordingly.

---

## License

> **[REVISION NOTE]** Add your chosen license here (MIT, Apache 2.0, etc.) and include
> a `LICENSE` file in the repository root. If this project uses the HAM10000 dataset,
> review its [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
> terms — your project's license must be compatible.
