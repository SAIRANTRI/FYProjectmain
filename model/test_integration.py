import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from utils.db_connector import DatabaseConnector, CloudinaryConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-integration")

load_dotenv()

def test_mongodb_connection():
    """Test connection to MongoDB"""
    db = DatabaseConnector()
    if db.db:
        logger.info("MongoDB connection successful!")
        images = db.get_all_images()
        logger.info(f"Retrieved {len(images)} images from MongoDB")
    else:
        logger.error("MongoDB connection failed!")
    db.close()

def test_cloudinary_connection():
    """Test connection to Cloudinary by downloading a test image"""
    cloudinary = CloudinaryConnector()
    
    test_url = "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
    
    test_dir = Path("./test_downloads")
    test_dir.mkdir(exist_ok=True)
    
    success = cloudinary.download_image(test_url, str(test_dir / "test_image.jpg"))
    
    if success:
        logger.info("Cloudinary download successful!")
    else:
        logger.error("Cloudinary download failed!")

if __name__ == "__main__":
    logger.info("Testing MongoDB and Cloudinary integration...")
    test_mongodb_connection()
    test_cloudinary_connection()
    logger.info("Integration tests completed!")