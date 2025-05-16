import os
import logging
import shutil
import tempfile
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.models.schemas import ClassificationRequest, ClassificationResponse, MatchedPhoto, UnmatchedPhoto
from app.services.mongodb import get_photo_by_id, get_photos_by_ids, save_classification_result
from app.services.cloudinary import download_image
from app.services.classifier import ClassifierService
from app.utils.helpers import generate_task_id, create_temp_directory, prepare_classification_result

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize classifier service
MODEL_PATH = os.getenv("MODEL_PATH", None)
classifier_service = ClassifierService(model_path=MODEL_PATH)

async def process_classification(
    task_id: str,
    user_id: str,
    reference_photo_id: str,
    reference_image_path: str,
    pool_photos: List[Dict[str, Any]],
    pool_image_paths: List[str],
    temp_dir: str
):
    try:
        # Get photo IDs
        photo_ids = [photo["_id"] for photo in pool_photos]
        
        # Classify images
        matched_photos, unmatched_photos = await classifier_service.classify_images(
            reference_image_path=reference_image_path,
            pool_image_paths=pool_image_paths,
            photo_ids=photo_ids
        )
        
        # Prepare result document
        result_data = prepare_classification_result(
            task_id=task_id,
            user_id=user_id,
            reference_photo_id=reference_photo_id,
            matched_photos=matched_photos,
            unmatched_photos=unmatched_photos
        )
        
        # Save result to database
        await save_classification_result(result_data)
        
        logger.info(f"Classification task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in classification task {task_id}: {str(e)}")
    finally:
        # Clean up temporary directory
        classifier_service.cleanup_temp_dir(temp_dir)

@router.post("/classify", response_model=ClassificationResponse)
async def classify_images(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks
):
    try:
        # Generate task ID
        task_id = generate_task_id()
        logger.info(f"Starting classification task {task_id}")
        
        # Get reference photo
        reference_photo = await get_photo_by_id(request.referencePhotoId)
        if not reference_photo:
            raise HTTPException(status_code=404, detail=f"Reference photo {request.referencePhotoId} not found")
        
        # Get pool photos
        pool_photos = await get_photos_by_ids(request.poolPhotoIds)
        if not pool_photos:
            raise HTTPException(status_code=404, detail="No pool photos found")
            
        # Check if we have all requested pool photos
        found_ids = {photo["_id"] for photo in pool_photos}
        missing_ids = set(request.poolPhotoIds) - found_ids
        if missing_ids:
            logger.warning(f"Some pool photos were not found: {missing_ids}")
        
        # Create temporary directory
        temp_dir = create_temp_directory()
        
        # Download reference image
        reference_image_path = os.path.join(temp_dir, f"reference_{request.referencePhotoId}.jpg")
        reference_download_success = await download_image(reference_photo["imageUrl"], reference_image_path)
        
        if not reference_download_success:
            classifier_service.cleanup_temp_dir(temp_dir)
            raise HTTPException(status_code=500, detail="Failed to download reference image")
        
        # Download pool images
        pool_image_paths = []
        for photo in pool_photos:
            image_path = os.path.join(temp_dir, f"pool_{photo['_id']}.jpg")
            download_success = await download_image(photo["imageUrl"], image_path)
            
            if download_success:
                pool_image_paths.append(image_path)
            else:
                logger.warning(f"Failed to download pool image {photo['_id']}")
        
        if not pool_image_paths:
            classifier_service.cleanup_temp_dir(temp_dir)
            raise HTTPException(status_code=500, detail="Failed to download any pool images")
        
        # Add background task for classification
        background_tasks.add_task(
            process_classification,
            task_id=task_id,
            user_id=request.userId,
            reference_photo_id=request.referencePhotoId,
            reference_image_path=reference_image_path,
            pool_photos=pool_photos,
            pool_image_paths=pool_image_paths,
            temp_dir=temp_dir
        )
        
        # Prepare initial response
        # Since processing happens in the background, we return empty lists initially
        # The client can poll for results using the task ID
        return ClassificationResponse(
            taskId=task_id,
            matched=[],
            unmatched=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/classify/{task_id}", response_model=ClassificationResponse)
async def get_classification_result(task_id: str):
    try:
        # Query the database for the result
        result = await results_collection.find_one({"taskId": task_id})
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Classification result for task {task_id} not found")
        
        # Get photo details for matched photos
        matched_photo_ids = [match["photoId"] for match in result["matches"]]
        matched_photos = await get_photos_by_ids(matched_photo_ids)
        
        # Get photo details for unmatched photos
        unmatched_photo_ids = [photo["photoId"] for photo in result["unmatchedImages"]]
        unmatched_photos = await get_photos_by_ids(unmatched_photo_ids)
        
        # Create response
        matched = []
        for match in result["matches"]:
            photo = next((p for p in matched_photos if p["_id"] == match["photoId"]), None)
            if photo:
                matched.append(MatchedPhoto(
                    photoId=match["photoId"],
                    imageUrl=photo["imageUrl"],
                    confidence=match["confidence"]
                ))
        
        unmatched = []
        for unmatch in result["unmatchedImages"]:
            photo = next((p for p in unmatched_photos if p["_id"] == unmatch["photoId"]), None)
            if photo:
                unmatched.append(UnmatchedPhoto(
                    photoId=unmatch["photoId"],
                    imageUrl=photo["imageUrl"]
                ))
        
        return ClassificationResponse(
            taskId=task_id,
            matched=matched,
            unmatched=unmatched
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_classification_result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify/upload")
async def classify_uploaded_images(
    reference_image: UploadFile = File(...),
    pool_images: List[UploadFile] = File(...),
    user_id: str = Form(...)
):
    """
    Process uploaded images and return a ZIP file with classification results
    
    Args:
        reference_image: Reference image file
        pool_images: List of pool image files
        user_id: User ID
        
    Returns:
        ZIP file containing classified images
    """
    try:
        logger.info(f"Received classification request with {len(pool_images)} pool images")
        
        # Create temporary directories
        temp_dir = create_temp_directory()
        input_dir = create_temp_directory("temp_input_pool")  # Separate directory for pool images only
        output_dir = create_temp_directory("temp_output")
        
        try:
            # Save reference image (not in the input_dir)
            reference_path = os.path.join(temp_dir, f"reference_{reference_image.filename}")
            with open(reference_path, "wb") as f:
                f.write(await reference_image.read())
            
            logger.info(f"Saved reference image to {reference_path}")
            
            # Save pool images to the input_dir (separate from reference)
            pool_paths = []
            for pool_image in pool_images:
                pool_path = os.path.join(input_dir, f"pool_{pool_image.filename}")
                with open(pool_path, "wb") as f:
                    f.write(await pool_image.read())
                pool_paths.append(pool_path)
            
            logger.info(f"Saved {len(pool_paths)} pool images to input directory")
            
            # Run classification - using input_dir which contains only pool images
            matched_files, unmatched_files = classifier_service.classifier.classify_images(
                input_dir=input_dir,  # Only contains pool images
                reference_image_path=reference_path,
                output_dir=output_dir
            )
            
            logger.info(f"Classification completed. Matched: {len(matched_files)}, Unmatched: {len(unmatched_files)}")
            
            # Create ZIP file
            zip_path = os.path.join(tempfile.gettempdir(), f"classified_images_{generate_task_id()}.zip")
            classifier_service.classifier.create_output_zip(output_dir, zip_path)
            
            logger.info(f"Created ZIP file at {zip_path}")
            
            # Return ZIP file as response
            return FileResponse(
                path=zip_path,
                filename="classified_images.zip",
                media_type="application/zip",
                background=BackgroundTasks().add_task(cleanup_files, temp_dir, input_dir, output_dir, zip_path)
            )
            
        except Exception as e:
            # Clean up in case of error
            cleanup_files(temp_dir, input_dir, output_dir)
            logger.error(f"Error in classification: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_files(*paths):
    """Clean up temporary files and directories"""
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"Error cleaning up {path}: {str(e)}")