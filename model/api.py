import os
import sys
import uuid
import shutil
import json
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import model components
from Model.main import FaceClassifier
from Model.utils.db_connector import DatabaseConnector, CloudinaryConnector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-recognition-api")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and classification using MongoDB and Cloudinary",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create temp directory for uploads
UPLOAD_DIR = Path("./Model/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR = Path("./Model/results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize face classifier
MODEL_PATH = os.environ.get("MODEL_PATH", None)  # Get model path from environment variable
classifier = None

# Database and Cloudinary connectors
db_connector = None
cloudinary_connector = None

# Pydantic models for request/response
class ImageInfo(BaseModel):
    _id: str
    imageUrl: str
    imageType: str

class ClassificationRequest(BaseModel):
    userId: str

class ClassificationResult(BaseModel):
    taskId: str
    referenceImageId: str
    referenceImageUrl: str
    matchedImages: List[Dict[str, Any]] = []
    unmatchedImages: List[Dict[str, Any]] = []
    processedAt: datetime = Field(default_factory=datetime.now)

class ProcessingStatus(BaseModel):
    status: str
    message: str
    taskId: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global classifier, db_connector, cloudinary_connector
    try:
        # Initialize classifier with GPU if available
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        classifier = FaceClassifier(
            model_path=MODEL_PATH,
            device=device,
            similarity_threshold=0.85,
            embedding_dim=512,
            use_arcface=True
        )
        logger.info(f"Face classifier initialized on {device}")
        
        # Initialize database connector
        db_connector = DatabaseConnector()
        
        # Initialize Cloudinary connector
        cloudinary_connector = CloudinaryConnector()
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    # Close database connection
    if db_connector:
        db_connector.close()

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/process-images/", response_model=ProcessingStatus)
async def process_images(
    background_tasks: BackgroundTasks,
    request: ClassificationRequest = Body(...),
):
    """
    Process images for a specific user based on their uploaded reference and pool images
    """
    try:
        # Create unique task ID
        task_id = str(uuid.uuid4())
        
        # Create directories for this task
        task_input_dir = UPLOAD_DIR / task_id / "input"
        task_output_dir = OUTPUT_DIR / task_id
        os.makedirs(task_input_dir, exist_ok=True)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Add processing task to background tasks
        background_tasks.add_task(
            process_user_images,
            task_id=task_id,
            user_id=request.userId,
            input_dir=str(task_input_dir),
            output_dir=str(task_output_dir)
        )
        
        return {
            "status": "processing",
            "message": "Image processing task started",
            "taskId": task_id
        }
    except Exception as e:
        logger.error(f"Error starting processing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing-result/{task_id}", response_model=ClassificationResult)
async def get_processing_result(task_id: str):
    """Get the results of a processing task from MongoDB"""
    try:
        result = db_connector.get_classification_result(task_id)
        
        if not result:
            # Check if there's a local result file (task might still be processing)
            result_file = OUTPUT_DIR / task_id / "result.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    result = json.load(f)
                return result
            else:
                raise HTTPException(status_code=404, detail="Task not found or still processing")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving processing result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-results/{task_id}")
async def download_results(task_id: str):
    """Download the zip file with classification results"""
    zip_path = OUTPUT_DIR / task_id / "classified_images.zip"
    
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        path=zip_path,
        filename="classified_images.zip",
        media_type="application/zip"
    )

async def process_user_images(
    task_id: str, 
    user_id: str,
    input_dir: str, 
    output_dir: str
):
    """Background task to process user's images from MongoDB/Cloudinary"""
    try:
        # Fetch user's images from MongoDB
        user_images = db_connector.get_user_images(user_id)
        
        if not user_images:
            raise Exception(f"No images found for user ID: {user_id}")
        
        # Separate reference and pool images
        reference_images = [img for img in user_images if img["imageType"] == "reference"]
        pool_images = [img for img in user_images if img["imageType"] == "pool"]
        
        if not reference_images:
            raise Exception("No reference images found")
        
        if not pool_images:
            raise Exception("No pool images found")
        
        # Use the first reference image (or you could implement logic to select a specific one)
        reference_image = reference_images[0]
        
        # Download reference image
        reference_path = os.path.join(UPLOAD_DIR, task_id, f"reference.jpg")
        success = cloudinary_connector.download_image(reference_image["imageUrl"], reference_path)
        
        if not success:
            raise Exception(f"Failed to download reference image: {reference_image['imageUrl']}")
        
        # Download pool images - ensure filenames match the expected format in main.py
        for img in pool_images:
            # Use the MongoDB _id as the filename to ensure uniqueness
            img_path = os.path.join(input_dir, f"{img['_id']}.jpg")
            success = cloudinary_connector.download_image(img["imageUrl"], img_path)
            if not success:
                logger.warning(f"Failed to download image: {img['imageUrl']}")
        
        # Run classification
        matched_files, unmatched_files = classifier.classify_images(
            input_dir=input_dir,
            reference_image_path=reference_path,
            output_dir=output_dir
        )
        
        # Create zip file
        zip_path = os.path.join(output_dir, "classified_images.zip")
        classifier.create_output_zip(output_dir, zip_path)
        
        # Prepare data for MongoDB - ensure we're matching the correct files
        matched_images = []
        for filename in matched_files:
            # Extract image ID from filename (remove extension)
            image_id = os.path.splitext(os.path.basename(filename))[0]
            # Find original image info
            original_image = next((img for img in pool_images if str(img["_id"]) == image_id), None)
            if original_image:
                matched_images.append({
                    "imageId": original_image["_id"],
                    "imageUrl": original_image["imageUrl"],
                    "confidence": 1.0,  # You could add actual confidence scores if available
                    "processedAt": datetime.now().isoformat()
                })
        
        unmatched_images = []
        for filename in unmatched_files:
            # Extract image ID from filename (remove extension)
            image_id = os.path.splitext(os.path.basename(filename))[0]
            # Find original image info
            original_image = next((img for img in pool_images if str(img["_id"]) == image_id), None)
            if original_image:
                unmatched_images.append({
                    "imageId": original_image["_id"],
                    "imageUrl": original_image["imageUrl"],
                    "processedAt": datetime.now().isoformat()
                })
        
        # Save results to MongoDB
        db_connector.save_classification_result(
            task_id=task_id,
            reference_image_id=reference_image["_id"],
            reference_image_url=reference_image["imageUrl"],
            matched_images=matched_images,
            unmatched_images=unmatched_images
        )
        
        # Also save results to local JSON file for backup
        result = {
            "taskId": task_id,
            "referenceImageId": reference_image["_id"],
            "referenceImageUrl": reference_image["imageUrl"],
            "matchedImages": matched_images,
            "unmatchedImages": unmatched_images,
            "processedAt": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result, f)
            
        logger.info(f"Processing task {task_id} completed successfully")
    except Exception as e:
        logger.error(f"Error in processing task {task_id}: {e}")
        # Save error to result file
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump({"error": str(e)}, f)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)