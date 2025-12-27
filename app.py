import os
import uuid
import asyncio
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from model_utils import DeepfakeDetector
from local_store import LocalStore
from db_data_api import insert_result, find_result, delete_result

APP_NAME = "Deepfake Detection API"

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
MODEL_NAME = os.getenv("MODEL_NAME", "tf_efficientnet_b0.ns_jft_in1k")

RESULT_TTL = int(os.getenv("RESULT_TTL", "300"))
STORE_DIR = os.getenv("STORE_DIR", "/tmp/deepfake_results")

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS.strip() in ("*", "") else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[DeepfakeDetector] = None
store = LocalStore(STORE_DIR)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def _is_image(filename: str, content_type: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    if ext in IMAGE_EXTS:
        return True
    if content_type and content_type.startswith("image/"):
        return True
    return False

def _is_video(filename: str, content_type: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    if ext in VIDEO_EXTS:
        return True
    if content_type and content_type.startswith("video/"):
        return True
    return False

async def _cleanup_loop():
    while True:
        try:
            store.cleanup_older_than(RESULT_TTL)
        except Exception:
            pass
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    global detector
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    detector = DeepfakeDetector(model_path=MODEL_PATH, model_name=MODEL_NAME)
    asyncio.create_task(_cleanup_loop())

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Unified endpoint for image OR video.

    Returns JSON:
      - label: REAL/FAKE
      - percentage: fake probability * 100
      - job_id
      - heatmap_url: URL to GET /result/{job_id}
      - output_type: image/png or video/mp4
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    is_image = _is_image(file.filename or "", file.content_type or "")
    is_video = _is_video(file.filename or "", file.content_type or "")

    if not (is_image or is_video):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload an image or a video (.jpg/.png/.mp4 etc).")

    job_id = uuid.uuid4().hex

    if is_image:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = detector.predict_image(img)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        overlay = result["overlay_bgr"]
        ok, png = cv2.imencode(".png", overlay)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode output image")

        out_path = store.make_path(job_id, "heatmap.png")
        with open(out_path, "wb") as f:
            f.write(png.tobytes())

        fake_prob = float(result["probability"])
        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        percentage = round(fake_prob * 100.0, 2)

        # Store metadata in Atlas Data API (best-effort; don't block user if it fails)
        try:
            await insert_result({
                "job_id": job_id,
                "label": label,
                "percentage": percentage,
                "output_type": "image/png",
                "output_path": out_path,
                "ttl_seconds": RESULT_TTL,
            })
        except Exception:
            pass

        base = str(request.base_url).rstrip("/")
        return {
            "job_id": job_id,
            "label": label,
            "percentage": percentage,
            "output_type": "image/png",
            "heatmap_url": f"{base}/result/{job_id}",
        }

    # video
    in_fd, in_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1] or ".mp4")
    os.close(in_fd)
    with open(in_path, "wb") as f:
        f.write(content)

    out_path = store.make_path(job_id, "heatmap.mp4")
    try:
        summary = detector.predict_video(in_path=in_path, out_path=out_path, sample_every=int(os.getenv("VIDEO_SAMPLE_EVERY","1")))
    except Exception as e:
        try: os.remove(in_path)
        except Exception: pass
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    try:
        os.remove(in_path)
    except Exception:
        pass

    avg_prob = float(summary.get("avg_probability", 0.0))
    label = "FAKE" if avg_prob >= 0.5 else "REAL"
    percentage = round(avg_prob * 100.0, 2)

    try:
        await insert_result({
            "job_id": job_id,
            "label": label,
            "percentage": percentage,
            "output_type": "video/mp4",
            "output_path": out_path,
            "frames_scored": int(summary.get("frames_scored", 0)),
            "ttl_seconds": RESULT_TTL,
        })
    except Exception:
        pass

    base = str(request.base_url).rstrip("/")
    return {
        "job_id": job_id,
        "label": label,
        "percentage": percentage,
        "output_type": "video/mp4",
        "frames_scored": int(summary.get("frames_scored", 0)),
        "heatmap_url": f"{base}/result/{job_id}",
    }

@app.get("/result/{job_id}")
async def result(job_id: str):
    # Prefer DB metadata if available; otherwise fall back to local store paths.
    doc = None
    try:
        doc = await find_result(job_id)
    except Exception:
        doc = None

    candidates = []
    if doc and isinstance(doc, dict) and doc.get("output_path"):
        candidates.append(doc["output_path"])
    candidates.append(store.make_path(job_id, "heatmap.mp4"))
    candidates.append(store.make_path(job_id, "heatmap.png"))

    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        # best-effort cleanup DB record
        try:
            await delete_result(job_id)
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="Result not found or expired")

    # Enforce TTL based on file mtime
    age = (asyncio.get_event_loop().time())
    try:
        import time
        if (time.time() - os.path.getmtime(path)) > RESULT_TTL:
            try:
                os.remove(path)
            except Exception:
                pass
            try:
                await delete_result(job_id)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="Result expired")
    except HTTPException:
        raise
    except Exception:
        pass

    mime = "video/mp4" if path.lower().endswith(".mp4") else "image/png"
    return FileResponse(path, media_type=mime, headers={"Content-Disposition": "inline"})
