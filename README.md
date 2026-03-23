# PneumoVision — Chest X-Ray Analysis System
 
A full-stack medical imaging web application that analyses chest X-rays to detect COVID-19 and Pneumonia using deep learning.
 
---
 
## Model
 
| Detail | Value |
|---|---|
| Architecture | MobileNetV2 (Transfer Learning) |
| Dataset | COVID-19 Radiography Database (Kaggle) |
| Total Images | 11,537 |
| Train / Validation Split | 80% / 20% (9,230 / 2,307) |
| Classes | COVID-19, Normal, Pneumonia |
| Validation Accuracy | 80.2% |
| Validation AUC | 0.9470 |
| Training Phase | Phase 1 only (CPU hardware constraint) |
 
---
 
## Features
 
- **COVID-19 & Pneumonia Detection** — classifies chest X-rays into 3 categories: COVID-19, Normal, Pneumonia
- **GradCAM Heatmap** — visually highlights the regions in the X-ray that influenced the model's decision
- **BLAA (Bilateral Lung Asymmetry Analysis)** — splits the GradCAM heatmap into left and right lung halves, calculates independent involvement percentage for each lung, and computes an asymmetry score. COVID-19 typically shows bilateral symmetric patterns while bacterial pneumonia tends to show unilateral dominance
- **Scan History** — all predictions are stored in Firebase Firestore and viewable in the History tab
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| ML Model | TensorFlow, MobileNetV2 |
| Explainability | GradCAM, OpenCV |
| Database | Firebase Firestore |
 
---
 
## Project Structure
 
```
PneumoVision/
├── backend/
│   ├── main.py          # FastAPI server and API endpoints
│   ├── model.py         # MobileNetV2 model, GradCAM, BLAA logic
│   ├── train.py         # Model training script
│   ├── requirements.txt # Python dependencies
│   └── xray_model.h5   # Trained model weights
├── frontend/
│   ├── index.html       # Main UI
│   ├── style.css        # Styling
│   └── app.js           # Frontend logic and API calls
└── README.md
```
 
---
 
## Running Locally
 
**1 — Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```
 
**2 — Add Firebase credentials:**
 
Place your `serviceAccountKey.json` inside the `backend/` folder.
 
**3 — Start the backend:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
 
**4 — Start the frontend:**
```bash
cd frontend
python -m http.server 5500
```
 
**5 — Open in browser:**
```
http://localhost:5500
```
 
---
 
## Dataset
 
COVID-19 Radiography Database — [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
 
| Class | Images |
|---|---|
| COVID-19 | 3,616 |
| Normal | 10,192 |
| Pneumonia | 1,345 |
 
> Note: Dataset is not included in this repository due to size. Download from Kaggle and place images in `backend/dataset/COVID/`, `backend/dataset/Normal/`, `backend/dataset/Pneumonia/`.
 
---
 
## Notes
 
- Validation AUC of 0.947 indicates strong class separation ability even at 80.2% raw accuracy
- Full Phase 2 fine-tuning on GPU is expected to push accuracy to 95%+
- Model was trained on CPU (Intel Core i7 13th Gen) with Phase 1 training only

<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/6a7792af-2eb1-44b3-9a86-180fba9edc26" />
<img width="1915" height="1017" alt="image" src="https://github.com/user-attachments/assets/bc1c3940-4da4-446e-994b-ac6e646608ad" />
<img width="1915" height="912" alt="image" src="https://github.com/user-attachments/assets/87ab3153-e754-4fad-a476-2b3643ad0bb8" />
<img width="1917" height="962" alt="image" src="https://github.com/user-attachments/assets/e89c78e4-4587-43eb-9d4d-a16f1a908262" />
<img width="1915" height="962" alt="image" src="https://github.com/user-attachments/assets/b8c5a04c-1a15-4a06-8a4a-8d32756312a5" />
<img width="1897" height="963" alt="image" src="https://github.com/user-attachments/assets/4bc1ca08-aeae-4e1e-8e1d-bede0846b207" />





