import os
import uuid
import time
import mimetypes
import asyncio
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from model_utils import DeepfakeDetector

load_dotenv()

# ===== Config =====
RESULT_TTL = int(os.getenv("RESULT_TTL", "300"))  # seconds
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

TMP_DIR = Path("/tmp/deepfake")
IN_DIR = TMP_DIR / "in"
OUT_DIR = TMP_DIR / "out"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== App =====
app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[DeepfakeDetector] = None


def is_video_file(filename: str, content_type: str) -> bool:
    ext = Path(filename).suffix.lower()
    return content_type.startswith("video/") or ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def is_image_file(filename: str, content_type: str) -> bool:
    ext = Path(filename).suffix.lower()
    return content_type.startswith("image/") or ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


async def cleanup_loop():
    while True:
        now = time.time()
        for folder in (IN_DIR, OUT_DIR):
            for p in folder.glob("*"):
                try:
                    if now - p.stat().st_mtime > RESULT_TTL:
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
        await asyncio.sleep(60)


@app.on_event("startup")
async def startup():
    global detector
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    detector = DeepfakeDetector(model_path=MODEL_PATH)
    asyncio.create_task(cleanup_loop())


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "ttl_seconds": RESULT_TTL}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...), sample_every: int = 5):
    """
    Upload an image or video.
    Returns JSON with prediction and download url for the heatmap output.
    For video, you can control speed via sample_every (higher = faster).
    """
    global detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ct = (file.content_type or "").lower()
    is_vid = is_video_file(file.filename, ct)
    is_img = is_image_file(file.filename, ct)

    if not (is_vid or is_img):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type} ({Path(file.filename).suffix})",
        )

    job_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower() or (".mp4" if is_vid else ".png")

    in_path = IN_DIR / f"{job_id}{ext}"
    out_ext = ".mp4" if is_vid else ".png"
    out_path = OUT_DIR / f"{job_id}_heatmap{out_ext}"

    # Save upload to /tmp
    data = await file.read()
    in_path.write_bytes(data)

    try:
        if is_vid:
            # Video path -> produce heatmap MP4
            result = detector.predict_video(str(in_path), str(out_path), sample_every=sample_every)
            prob = float(result["avg_probability"])
            label = result["label"]
            output_type = "video/mp4"
        else:
            # Image -> produce heatmap PNG
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Could not decode image")

            result = detector.predict_image(bgr)
            if "error" in result:
                raise RuntimeError(result["error"])

            prob = float(result["probability"])
            label = result["label"]
            output_type = "image/png"

            # save overlay image
            overlay_bgr = result["overlay_bgr"]
            ok = cv2.imwrite(str(out_path), overlay_bgr)
            if not ok:
                raise RuntimeError("Failed to write output image")

        percentage = round(prob * 100.0, 2)

        return JSONResponse(
            {
                "job_id": job_id,
                "label": label,
                "probability": prob,
                "percentage": percentage,
                "output_type": output_type,
                "download_url": f"/download/{job_id}",
                "expires_in_seconds": RESULT_TTL,
                "sample_every": sample_every if output_type == "video/mp4" else None,
            }
        )

    except Exception as e:
        # cleanup if failed
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{job_id}")
def download(job_id: str):
    # try video then image
    candidates = [
        OUT_DIR / f"{job_id}_heatmap.mp4",
        OUT_DIR / f"{job_id}_heatmap.png",
    ]
    out_path = next((p for p in candidates if p.exists()), None)
    if not out_path:
        raise HTTPException(status_code=404, detail="Output not found or expired")

    mime, _ = mimetypes.guess_type(str(out_path))
    return FileResponse(
        path=str(out_path),
        media_type=mime or "application/octet-stream",
        filename=out_path.name,
    )
