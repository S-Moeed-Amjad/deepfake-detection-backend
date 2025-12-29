# Deepfake Backend (FastAPI) — No DB, Serve-once Results

## Endpoints
- `POST /predict` (multipart field: `file`)  
  Returns JSON with `label`, `percentage`, and `result_url`.

- `GET /result/{job_id}`  
  Streams the heatmap output (PNG/MP4). If `SERVE_ONCE=1`, it deletes the output immediately after serving.

## Local run
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
python -m uvicorn app:app --reload --port 8000
```

Swagger:
- http://localhost:8000/docs

## Deploy on Render
- Set `ALLOWED_ORIGINS` to your Netlify domain.
- Render filesystem is ephemeral, which is perfect for temporary results.
