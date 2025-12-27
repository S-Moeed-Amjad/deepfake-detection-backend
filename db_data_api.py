import os
from datetime import datetime, timezone
import httpx

APP_ID = os.getenv("MONGO_DATA_API_APP_ID")
API_KEY = os.getenv("MONGO_DATA_API_KEY")
DATA_SOURCE = os.getenv("MONGO_DATA_API_DATA_SOURCE")
DB = os.getenv("MONGO_DB", "deepfake")
COLL = os.getenv("MONGO_COLLECTION", "results")

BASE = lambda app_id: f"https://data.mongodb-api.com/app/{app_id}/endpoint/data/v1"

def _require():
    missing = []
    if not APP_ID: missing.append("MONGO_DATA_API_APP_ID")
    if not API_KEY: missing.append("MONGO_DATA_API_KEY")
    if not DATA_SOURCE: missing.append("MONGO_DATA_API_DATA_SOURCE")
    if missing:
        raise RuntimeError("Missing env vars: " + ", ".join(missing))

def _headers():
    return {"Content-Type": "application/json", "api-key": API_KEY}

async def insert_result(doc: dict) -> None:
    _require()
    payload = {
        "dataSource": DATA_SOURCE,
        "database": DB,
        "collection": COLL,
        "document": {**doc, "created_at": datetime.now(timezone.utc).isoformat()},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{BASE(APP_ID)}/action/insertOne", headers=_headers(), json=payload)
        r.raise_for_status()

async def find_result(job_id: str):
    _require()
    payload = {"dataSource": DATA_SOURCE, "database": DB, "collection": COLL, "filter": {"job_id": job_id}}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{BASE(APP_ID)}/action/findOne", headers=_headers(), json=payload)
        r.raise_for_status()
        return r.json().get("document")

async def delete_result(job_id: str) -> None:
    _require()
    payload = {"dataSource": DATA_SOURCE, "database": DB, "collection": COLL, "filter": {"job_id": job_id}}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{BASE(APP_ID)}/action/deleteOne", headers=_headers(), json=payload)
        r.raise_for_status()
