# 🫁 PneumoVision — AI Pulmonary Diagnostics

> DenseNet121 · GradCAM · Bilateral Lung Asymmetry Analysis (BLAA) · Firebase · FastAPI

---

## 📁 Project Structure

```
pneumovision/
├── frontend/
│   ├── index.html        ← Main UI
│   ├── style.css         ← Dark medical theme
│   └── app.js            ← API calls, Firebase, UI logic
├── backend/
│   ├── main.py           ← FastAPI server
│   ├── model.py          ← DenseNet121 + GradCAM + BLAA
│   ├── train.py          ← Training script
│   └── requirements.txt
└── README.md
```

---

## ⚙️ Backend Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Download dataset (Kaggle)
- Dataset: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Unzip and organise as:
```
backend/dataset/
├── COVID/            (~3616 images)
├── Normal/           (~10192 images)
└── Viral Pneumonia/  → rename folder to: Pneumonia
```

### 3. Train the model
```bash
cd backend
python train.py
# This creates xray_model.h5 (~97% val accuracy expected)
# Training takes 30-90 min depending on GPU
```

### 4. Set up Firebase
1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create a project → Firestore Database → Start in test mode
3. Project Settings → Service Accounts → Generate new private key
4. Save as `backend/serviceAccountKey.json`

### 5. Run the API server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 Frontend Setup

Simply open `frontend/index.html` in a browser **or** serve it:
```bash
cd frontend
python -m http.server 5500
# Open http://localhost:5500
```

If your FastAPI runs on a different port, update `API_BASE` in `app.js`.

---

## 🔬 Key Features

| Feature | Description |
|---|---|
| **DenseNet121** | Transfer learning from ImageNet, fine-tuned in 2 phases |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalisation for better X-ray visibility |
| **GradCAM** | Gradient-weighted Class Activation Mapping — highlights pathological regions |
| **BLAA** ⭐ | **Bilateral Lung Asymmetry Analysis** — independently quantifies left/right lung involvement from GradCAM activations. COVID → bilateral symmetric; Pneumonia → unilateral/asymmetric |
| **Firebase** | Firestore stores all prediction history server-side |
| **Report Export** | Download HTML diagnostic report per scan |

---

## 🎯 Expected Accuracy

| Model | Val Accuracy | AUC |
|---|---|---|
| DenseNet121 (this config) | ~96–98% | ~0.99 |
| Phase 1 (frozen backbone) | ~90–93% | — |
| Phase 2 (fine-tuned) | **~96–98%** | ~0.99 |

Trained on COVID-19 Radiography Database (Kaggle).

---

## 🧪 Novelty: BLAA (Bilateral Lung Asymmetry Analysis)

**What it does:**
- Splits the GradCAM heatmap vertically into left and right lung halves
- Calculates the percentage of each half showing high activation (threshold = 0.45)
- Computes an asymmetry score = |left% − right%|
- Classifies the pattern:
  - **Bilateral symmetric** (asym < 15%) → COVID-19 consistent
  - **Unilateral dominant** (asym ≥ 15%) → Bacterial pneumonia consistent
  - **Minimal involvement** → Normal / early-stage

**Clinical relevance:**
COVID-19 pneumonia typically shows diffuse bilateral ground-glass opacities, while bacterial pneumonia tends to show lobar/segmental consolidation, often in one lung. BLAA gives a second differential signal beyond the raw classifier output.

---

## ⚠️ Disclaimer

This system is for **academic/research purposes only**. It is not clinically validated and must not be used for real medical diagnosis.
