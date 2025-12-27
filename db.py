import os
from pymongo import MongoClient
from pymongo.collection import Collection
import certifi

def get_client():
    uri = os.environ["mongodb+srv://syedmoeedamjad_db_user:Yolopolo123@deepfake.wnk5pzu.mongodb.net/?appName=deepfake"]
    return MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=20000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
    )


def get_collection() -> Collection:
    client = get_client()
    db_name = os.getenv("MONGODB_DB", "deepfake")
    coll_name = os.getenv("MONGODB_COLLECTION", "results")
    return client[db_name][coll_name]
