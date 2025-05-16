import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def generate_task_id() -> str:
    return str(uuid.uuid4())

def create_temp_directory(base_dir: str = "temp_input") -> str:
    # Create a unique directory name
    unique_dir = f"{base_dir}_{uuid.uuid4().hex}"
    os.makedirs(unique_dir, exist_ok=True)
    logger.info(f"Created temporary directory: {unique_dir}")
    return unique_dir

def prepare_classification_result(
    task_id: str,
    user_id: str,
    reference_photo_id: str,
    matched_photos: List[Dict[str, Any]],
    unmatched_photos: List[Dict[str, Any]]
) -> Dict[str, Any]:

    # Add timestamps to matched and unmatched photos
    now = datetime.utcnow()
    
    for photo in matched_photos:
        photo["matchedAt"] = now
        
    for photo in unmatched_photos:
        photo["processedAt"] = now
    
    # Create result document
    result = {
        "taskId": task_id,
        "userId": user_id,
        "referenceImage": reference_photo_id,
        "matches": matched_photos,
        "unmatchedImages": unmatched_photos,
        "createdAt": now
    }
    
    return result

def save_uploaded_file(upload_file, destination_path):
    """
    Save an uploaded file to the specified path
    
    Args:
        upload_file: FastAPI UploadFile
        destination_path: Path to save the file
        
    Returns:
        Path to the saved file
    """
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise