from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from PIL import Image
import io
import uuid
import datetime
import os
import traceback

from model import load_xray_model, predict_xray, generate_gradcam_heatmap
import gdown

MODEL_PATH = "xray_model.h5"
if not os.path.exists(MODEL_PATH):
    drive_id = os.environ.get("MODEL_DRIVE_ID")
    if drive_id:
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id=1U-H__E3eB_6XJg52Qqas-SldqSuIO96w", MODEL_PATH, quiet=False)
        print("Model downloaded!")
# Firebase setup (optional - graceful fallback)
firebase_enabled = False
db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_enabled = True
        print(" Firebase connected")
    else:
        print("  serviceAccountKey.json not found — Firebase disabled")
except Exception as e:
    print(f"  Firebase init failed: {e}")

app = FastAPI(title="PneumoVision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading DenseNet121 model...")
model = load_xray_model()
print(" Model ready")


def encode_image_bgr(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_severity(class_name: str, pis: float) -> str:
    if class_name == "Normal":
        return "None"
    if pis < 20:
        return "Mild"
    elif pis < 45:
        return "Moderate"
    elif pis < 70:
        return "Severe"
    return "Critical"


def get_recommendations(class_name: str, severity: str) -> list:
    base = {
        "Normal": [
            "No significant pulmonary pathology detected.",
            "Routine follow-up as clinically indicated.",
            "Maintain healthy lifestyle and regular check-ups.",
        ],
        "COVID-19": [
            "Immediate isolation recommended per local health guidelines.",
            "RT-PCR confirmation strongly advised.",
            "Monitor SpO₂ levels closely; target ≥ 95%.",
            "Adequate hydration, rest, and symptomatic management.",
        ],
        "Pneumonia": [
            "Clinical correlation and sputum culture recommended.",
            "Antibiotic therapy may be warranted (consult physician).",
            "Follow-up chest X-ray in 4–6 weeks post-treatment.",
            "Monitor for signs of pleural effusion or abscess.",
        ],
    }
    recs = base.get(class_name, [])
    if severity in ("Severe", "Critical"):
        recs.append("⚠️ Significant bilateral involvement — consider hospitalization.")
    return recs


@app.get("/")
def root():
    return {"status": "PneumoVision API running", "firebase": firebase_enabled}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Prediction
        result = predict_xray(model, pil_img)

        # GradCAM + bilateral analysis
        gradcam_bgr, blaa = generate_gradcam_heatmap(model, pil_img, result["class_idx"])

        # PIS = overall pulmonary involvement score (from BLAA)
        pis_score = round((blaa["left_pct"] + blaa["right_pct"]) / 2, 2)
        asymmetry_score = round(abs(blaa["left_pct"] - blaa["right_pct"]), 2)

        severity = get_severity(result["class_name"], pis_score)
        recommendations = get_recommendations(result["class_name"], severity)

        record_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        payload = {
            "id": record_id,
            "timestamp": timestamp,
            "filename": file.filename,
            "prediction": result["class_name"],
            "confidence": round(result["confidence"], 2),
            "probabilities": {k: round(v, 2) for k, v in result["probabilities"].items()},
            "gradcam_b64": encode_image_bgr(gradcam_bgr),
            "original_b64": encode_pil(pil_img),
            "pis_score": pis_score,
            "blaa": {
                "left_lung_pct": blaa["left_pct"],
                "right_lung_pct": blaa["right_pct"],
                "asymmetry_score": asymmetry_score,
                "pattern": blaa["pattern"],
            },
            "severity": severity,
            "recommendations": recommendations,
        }

        # Save to Firestore (without large image blobs)
        if firebase_enabled:
            db.collection("predictions").document(record_id).set(
                {k: v for k, v in payload.items() if k not in ("gradcam_b64", "original_b64")}
            )

        return JSONResponse(content=payload)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history(limit: int = 10):
    if not firebase_enabled:
        return []
    try:
        docs = (
            db.collection("predictions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{record_id}")
def delete_record(record_id: str):
    if not firebase_enabled:
        raise HTTPException(status_code=503, detail="Firebase not configured")
    db.collection("predictions").document(record_id).delete()
    return {"deleted": record_id}
