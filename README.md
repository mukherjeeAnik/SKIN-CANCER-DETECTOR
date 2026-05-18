# 🔬 DermaScan — Skin Cancer Classifier

A local web application for classifying skin lesions using DenseNet-121 trained on the HAM10000 dataset.
Runs fully on CPU — no GPU required.

---

## 📁 Setup (one-time)

### 1. Place your model file
Copy `best_model.pth` into this folder (same directory as `app.py`).

### 2. Install Python dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

> **Tip for Windows:** If you want a smaller CPU-only PyTorch install (faster download), run:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install flask Pillow
> ```

---

## 🚀 Running the app

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## 🩺 Supported Classes (HAM10000)

| Code   | Name                        | Risk Level |
|--------|-----------------------------|------------|
| akiec  | Actinic Keratoses           | Moderate   |
| bcc    | Basal Cell Carcinoma        | High       |
| bkl    | Benign Keratosis            | Low        |
| df     | Dermatofibroma              | Very Low   |
| mel    | Melanoma                    | Critical   |
| nv     | Melanocytic Nevi            | Very Low   |
| vasc   | Vascular Lesion             | Low        |

---

## ⚙️ Technical Details

- **Architecture:** DenseNet-121
- **Input size:** 224 × 224 px (auto-resized)
- **Preprocessing:** ImageNet normalization (mean/std)
- **Validation accuracy:** ~89.45%
- **Device:** CPU (no GPU required)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.  
It does **not** provide medical advice, diagnosis, or treatment.  
Always consult a qualified dermatologist for any skin concerns.
