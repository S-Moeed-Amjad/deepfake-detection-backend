# Backend (FastAPI) — MongoDB (TTL) + Temporary Heatmap Files

## Endpoints
- `POST /predict` (multipart/form-data, field: `file`)  
  Returns JSON: `label`, `percentage`, and `heatmap_url`.

- `GET /result/{job_id}`  
  Streams the heatmap output (PNG or MP4). Expires after `RESULT_TTL` seconds.

## Auto-expiry (5 minutes)
- MongoDB TTL index deletes metadata automatically (not instantaneous).
- `/result/{job_id}` also enforces TTL and removes expired file + record on access.

## Local run
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
# edit .env to set MONGODB_URI

python -m uvicorn app:app --reload --port 8000
```

Swagger:
- http://localhost:8000/docs

## Deploy on Render
Set env vars:
- `MONGODB_URI`
- `ALLOWED_ORIGINS` (your Netlify URL)
- `MODEL_PATH=best.pt` (included in repo)
- `RESULT_TTL=300`
# deepfake-detection-backend
