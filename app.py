import os
import uuid
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from model_utils import DeepfakeDetector
from db import get_collection
from local_store import LocalStore

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
MODEL_NAME = os.getenv("MODEL_NAME", "tf_efficientnet_b0.ns_jft_in1k")
RESULT_TTL = int(os.getenv("RESULT_TTL", "300"))  # 5 minutes default
RESULT_DIR = os.getenv("RESULT_DIR", "/tmp/deepfake_results")

app = FastAPI(title="Deepfake Detection API (MongoDB + Temporary Files)")

# CORS
allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None  # loaded at startup
store = LocalStore(RESULT_DIR)

def _now_ts() -> float:
    return time.time()

@app.on_event("startup")
def startup():
    global detector
    detector = DeepfakeDetector(model_path=MODEL_PATH, model_name=MODEL_NAME)

    # Create TTL index (MongoDB deletes docs automatically after RESULT_TTL seconds)
    coll = get_collection()
    coll.create_index("created_at", expireAfterSeconds=RESULT_TTL)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image or video. Returns label, percentage, and a URL to fetch the heatmap output."""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename.lower())[1]
    content_type = (file.content_type or "").lower()
    data = await file.read()

    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    # IMAGE
    if ext in IMAGE_EXTS or content_type.startswith("image/"):
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = detector.predict_image(img)
        if "error" in result:
            return {"job_id": job_id, "type": "image", "label": "UNKNOWN", "percentage": None, "error": result["error"]}

        fake_prob = float(result["probability"])
        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        percentage = round(fake_prob * 100.0, 2)

        out_path = store.make_path(job_id, "heatmap.png")
        ok, png = cv2.imencode(".png", result["overlay_bgr"])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        with open(out_path, "wb") as f:
            f.write(png.tobytes())

        coll = get_collection()
        coll.insert_one({
            "job_id": job_id,
            "type": "image",
            "label": label,
            "fake_probability": fake_prob,
            "percentage": percentage,
            "output_path": out_path,
            "mime": "image/png",
            "created_at": created_at,
        })

        return {
            "job_id": job_id,
            "type": "image",
            "label": label,
            "percentage": percentage,
            "fake_probability": round(fake_prob, 6),
            "heatmap_url": f"/result/{job_id}",
            "expires_in_seconds": RESULT_TTL
        }

    # VIDEO
    if ext in VIDEO_EXTS or content_type.startswith("video/"):
        suffix = ext if ext else ".mp4"
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        in_path = in_tmp.name
        in_tmp.close()
        with open(in_path, "wb") as f:
            f.write(data)

        out_path = store.make_path(job_id, "heatmap.mp4")

        try:
            summary = detector.predict_video(in_path, out_path, sample_every=1)
        except Exception as e:
            try:
                os.remove(in_path)
            except Exception:
                pass
            store.delete_path(out_path)
            raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

        try:
            os.remove(in_path)
        except Exception:
            pass

        avg_fake_prob = float(summary["avg_probability"])
        label = "FAKE" if avg_fake_prob >= 0.5 else "REAL"
        percentage = round(avg_fake_prob * 100.0, 2)

        coll = get_collection()
        coll.insert_one({
            "job_id": job_id,
            "type": "video",
            "label": label,
            "fake_probability": avg_fake_prob,
            "percentage": percentage,
            "frames_scored": int(summary.get("frames_scored", 0)),
            "output_path": out_path,
            "mime": "video/mp4",
            "created_at": created_at,
        })

        return {
            "job_id": job_id,
            "type": "video",
            "label": label,
            "percentage": percentage,
            "fake_probability": round(avg_fake_prob, 6),
            "frames_scored": int(summary.get("frames_scored", 0)),
            "heatmap_url": f"/result/{job_id}",
            "expires_in_seconds": RESULT_TTL
        }

    raise HTTPException(status_code=400, detail="Unsupported file type. Upload an image or a video.")

@app.get("/result/{job_id}")
def get_result(job_id: str):
    coll = get_collection()
    doc = coll.find_one({"job_id": job_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found or expired.")

    created_at = doc.get("created_at")
    if created_at is not None:
        age = _now_ts() - created_at.timestamp()
        if age > RESULT_TTL:
            try:
                store.delete_path(doc.get("output_path", ""))
            except Exception:
                pass
            coll.delete_one({"job_id": job_id})
            raise HTTPException(status_code=404, detail="Result expired.")

    path = doc.get("output_path")
    mime = doc.get("mime", "application/octet-stream")
    if not path or not store.exists(path):
        coll.delete_one({"job_id": job_id})
        raise HTTPException(status_code=404, detail="File not found (expired).")

    return FileResponse(path, media_type=mime, filename=os.path.basename(path))
