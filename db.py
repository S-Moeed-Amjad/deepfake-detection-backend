import os
from pymongo import MongoClient
from pymongo.collection import Collection

def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("Missing MONGODB_URI")
    return MongoClient(uri)

def get_collection() -> Collection:
    client = get_client()
    db_name = os.getenv("MONGODB_DB", "deepfake")
    coll_name = os.getenv("MONGODB_COLLECTION", "results")
    return client[db_name][coll_name]
