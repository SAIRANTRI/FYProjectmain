import os
import motor.motor_asyncio
from bson import ObjectId
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# MongoDB connection string from environment variable
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "fyproject")

# Create MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
photos_collection = db["photos"]
results_collection = db["results"]

async def get_photo_by_id(photo_id: str) -> Optional[Dict[str, Any]]:
    try:
        photo = await photos_collection.find_one({"_id": ObjectId(photo_id)})
        if photo:
            photo["_id"] = str(photo["_id"])  # Convert ObjectId to string
        return photo
    except Exception as e:
        logger.error(f"Error fetching photo {photo_id}: {str(e)}")
        return None

async def get_photos_by_ids(photo_ids: List[str]) -> List[Dict[str, Any]]:
    try:
        # Convert string IDs to ObjectId
        object_ids = [ObjectId(pid) for pid in photo_ids]
        
        # Query the database
        cursor = photos_collection.find({"_id": {"$in": object_ids}})
        
        # Convert results to list and format _id as string
        photos = []
        async for photo in cursor:
            photo["_id"] = str(photo["_id"])
            photos.append(photo)
            
        return photos
    except Exception as e:
        logger.error(f"Error fetching photos: {str(e)}")
        return []

async def save_classification_result(result_data: Dict[str, Any]) -> str:
    try:
        # Insert the result document
        result = await results_collection.insert_one(result_data)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving classification result: {str(e)}")
        raise