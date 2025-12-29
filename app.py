import os
import uuid
import time
import tempfile
from typing import Dict, Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from model_utils import DeepfakeDetector

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
MODEL_NAME = os.getenv("MODEL_NAME", "tf_efficientnet_b0.ns_jft_in1k")

RESULT_DIR = os.getenv("RESULT_DIR", "/tmp/deepfake_results")
RESULT_TTL = int(os.getenv("RESULT_TTL", "300"))
SERVE_ONCE = os.getenv("SERVE_ONCE", "1") == "1"

app = FastAPI(title="Deepfake Detection API (Serve-once, No DB)")

allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None
os.makedirs(RESULT_DIR, exist_ok=True)

RESULTS: Dict[str, Dict[str, Any]] = {}

def _purge_expired() -> None:
    now = time.time()
    expired = [k for k, v in RESULTS.items() if (now - float(v.get("created_at", 0))) > RESULT_TTL]
    for job_id in expired:
        v = RESULTS.pop(job_id, None)
        if not v:
            continue
        path = v.get("path")
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

@app.on_event("startup")
def startup():
    global detector
    detector = DeepfakeDetector(model_path=MODEL_PATH, model_name=MODEL_NAME)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    _purge_expired()

    ext = os.path.splitext(file.filename.lower())[1]
    content_type = (file.content_type or "").lower()
    data = await file.read()

    job_id = str(uuid.uuid4())

    # IMAGE
    if ext in IMAGE_EXTS or content_type.startswith("image/"):
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = detector.predict_image(img)
        if "error" in result:
            return JSONResponse({
                "job_id": job_id,
                "type": "image",
                "label": "UNKNOWN",
                "percentage": None,
                "error": result["error"],
            })

        fake_prob = float(result["probability"])
        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        percentage = round(fake_prob * 100.0, 2)

        out_path = os.path.join(RESULT_DIR, f"{job_id}.png")
        ok, png = cv2.imencode(".png", result["overlay_bgr"])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        with open(out_path, "wb") as f:
            f.write(png.tobytes())

        RESULTS[job_id] = {"path": out_path, "mime": "image/png", "created_at": time.time()}

        return {
            "job_id": job_id,
            "type": "image",
            "label": label,
            "percentage": percentage,
            "fake_probability": round(fake_prob, 6),
            "result_url": f"/result/{job_id}",
            "expires_in_seconds": RESULT_TTL,
            "serve_once": SERVE_ONCE,
        }

    # VIDEO
    if ext in VIDEO_EXTS or content_type.startswith("video/"):
        suffix = ext if ext else ".mp4"
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        in_path = in_tmp.name
        in_tmp.close()
        with open(in_path, "wb") as f:
            f.write(data)

        out_path = os.path.join(RESULT_DIR, f"{job_id}.mp4")

        try:
            summary = detector.predict_video(in_path, out_path, sample_every=1)
        except Exception as e:
            try:
                os.remove(in_path)
            except Exception:
                pass
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

        try:
            os.remove(in_path)
        except Exception:
            pass

        avg_fake_prob = float(summary["avg_probability"])
        label = "FAKE" if avg_fake_prob >= 0.5 else "REAL"
        percentage = round(avg_fake_prob * 100.0, 2)

        RESULTS[job_id] = {"path": out_path, "mime": "video/mp4", "created_at": time.time()}

        return {
            "job_id": job_id,
            "type": "video",
            "label": label,
            "percentage": percentage,
            "fake_probability": round(avg_fake_prob, 6),
            "frames_scored": int(summary.get("frames_scored", 0)),
            "result_url": f"/result/{job_id}",
            "expires_in_seconds": RESULT_TTL,
            "serve_once": SERVE_ONCE,
        }

    raise HTTPException(status_code=400, detail="Unsupported file type. Upload an image or a video.")

@app.get("/result/{job_id}")
def get_result(job_id: str):
    _purge_expired()

    item = RESULTS.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Result not found or expired.")

    path = item.get("path")
    mime = item.get("mime", "application/octet-stream")
    created_at = float(item.get("created_at", 0))
    if (time.time() - created_at) > RESULT_TTL:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        RESULTS.pop(job_id, None)
        raise HTTPException(status_code=404, detail="Result expired.")

    if not path or not os.path.exists(path):
        RESULTS.pop(job_id, None)
        raise HTTPException(status_code=404, detail="File missing (expired).")

    resp = FileResponse(path, media_type=mime, filename=os.path.basename(path))

    if SERVE_ONCE:
        try:
            os.remove(path)
        except Exception:
            pass
        RESULTS.pop(job_id, None)

    return resp
