import os
import sys
import json
import requests
import pymongo
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db-connector")

class DatabaseConnector:
    
    def __init__(self, mongo_uri: Optional[str] = None):
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        if not self.mongo_uri:
            logger.warning("MongoDB URI not provided. Database operations will not work.")
            self.client = None
            self.db = None
        else:
            try:
                self.client = pymongo.MongoClient(self.mongo_uri)
                self.db = self.client.get_database()
                logger.info(f"Connected to MongoDB database: {self.db.name}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.client = None
                self.db = None
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_user_images(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all images for a specific user
        
        Args:
            user_id: User ID to fetch images for
        
        Returns:
            List of image documents
        """
        if not self.db:
            logger.error("Database not connected. Cannot retrieve user images.")
            return []
            
        try:
            # Convert string user_id to ObjectId if needed
            from bson.objectid import ObjectId
            user_object_id = ObjectId(user_id)
            
            # Query the photos collection based on the schema in backend/models/photo.model.js
            images = list(self.db.photos.find({"userId": user_object_id}))
            
            # Convert ObjectId to string for JSON serialization
            for img in images:
                img["_id"] = str(img["_id"])
                if "userId" in img and isinstance(img["userId"], ObjectId):
                    img["userId"] = str(img["userId"])
                if "referenceId" in img and img["referenceId"] and isinstance(img["referenceId"], ObjectId):
                    img["referenceId"] = str(img["referenceId"])
                
            logger.info(f"Retrieved {len(images)} images for user {user_id}")
            return images
            
        except Exception as e:
            logger.error(f"Error retrieving user images: {e}")
            return []
    
    def save_classification_result(self, 
                                task_id: str,
                                reference_image_id: str,
                                reference_image_url: str,
                                matched_images: List[Dict[str, Any]],
                                unmatched_images: List[Dict[str, Any]]) -> str:
        if not self.db:
            logger.error("Database not connected. Cannot save classification result.")
            return None
            
        try:
            # Prepare document
            doc = {
                "taskId": task_id,
                "referenceImageId": reference_image_id,
                "referenceImageUrl": reference_image_url,
                "matchedImages": matched_images,
                "unmatchedImages": unmatched_images,
                "processedAt": datetime.now()
            }
            
            # Insert into collection
            result = self.db.classificationResults.insert_one(doc)
            logger.info(f"Classification result saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving classification result: {e}")
            return None
    
    def get_classification_result(self, task_id: str) -> Dict[str, Any]:
        if not self.db:
            logger.error("Database not connected. Cannot retrieve classification result.")
            return None
            
        try:
            result = self.db.classificationResults.find_one({"taskId": task_id})
            if result:
                # Convert ObjectId to string for JSON serialization
                result["_id"] = str(result["_id"])
                return result
            else:
                logger.warning(f"No classification result found for task ID: {task_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving classification result: {e}")
            return None


class CloudinaryConnector:
    
    def __init__(self, 
                cloud_name: Optional[str] = None, 
                api_key: Optional[str] = None, 
                api_secret: Optional[str] = None):
        """Initialize Cloudinary connection"""
        self.cloud_name = cloud_name or os.environ.get("CLOUDINARY_CLOUD_NAME")
        self.api_key = api_key or os.environ.get("CLOUDINARY_API_KEY")
        self.api_secret = api_secret or os.environ.get("CLOUDINARY_API_SECRET")
        
        if not all([self.cloud_name, self.api_key, self.api_secret]):
            logger.warning("Cloudinary credentials not provided. Image operations will not work.")
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Image downloaded successfully to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return False
