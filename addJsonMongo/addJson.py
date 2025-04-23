import json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["DB명"]
collection = db["컬렉션명"]

with open("파일경로.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    if isinstance(data, list):
        collection.insert_many(data)
    else:
        collection.insert_one(data)