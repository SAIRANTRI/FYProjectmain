import os
import sys
import logging
import shutil
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch

"""
#Add the model directory to the Python path
sys.path.append(os.path.abspath("model"))

from main import FaceClassifier
from sklearn.metrics.pairwise import cosine_similarity
"""

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import the FaceClassifier
from model.main import FaceClassifier
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ClassifierService:
    def __init__(self, model_path: str = None, device: str = None):
        # Determine device
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
        logger.info(f"Initializing FaceClassifier on device: {device}")
        
        # Initialize the FaceClassifier
        self.classifier = FaceClassifier(
            model_path=model_path,
            device=device,
            similarity_threshold=0.75,
            embedding_dim=512,
            use_arcface=True
        )
        
        logger.info("FaceClassifier initialized successfully")
        
    def cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    async def classify_images(self, 
                             reference_image_path: str, 
                             pool_image_paths: List[str],
                             photo_ids: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            logger.info(f"Processing reference image: {reference_image_path}")
            
            # Process reference image to get embedding
            reference_embedding = self.classifier.process_reference_image(reference_image_path)
            
            if reference_embedding is None:
                raise ValueError("Failed to generate embedding for reference image")
                
            # Process each pool image
            matched_photos = []
            unmatched_photos = []
            
            logger.info(f"Processing {len(pool_image_paths)} pool images")
            
            for i, (image_path, photo_id) in enumerate(zip(pool_image_paths, photo_ids)):
                try:
                    logger.info(f"Processing pool image {i+1}/{len(pool_image_paths)}: {image_path}")
                    
                    # Load and preprocess the image
                    image = self.classifier._robust_image_load(image_path)
                    
                    if image is None:
                        logger.warning(f"Could not read image: {image_path}")
                        unmatched_photos.append({
                            "photoId": photo_id,
                            "processedAt": None
                        })
                        continue
                    
                    # Detect faces
                    detections = self.classifier.preprocessor.detect_faces(image)
                    
                    if not detections:
                        logger.info(f"No faces detected in image: {image_path}")
                        unmatched_photos.append({
                            "photoId": photo_id,
                            "processedAt": None
                        })
                        continue
                    
                    # Check each detected face for a match
                    match_found = False
                    best_confidence = 0.0
                    
                    for detection in detections:
                        face = self.classifier.preprocessor.preprocess_face(image, detection)
                        embedding = self.classifier.model.get_embedding(face)
                        
                        # Calculate similarity
                        similarity = cosine_similarity(
                            reference_embedding,
                            embedding
                        )[0][0]
                        
                        if similarity >= self.classifier.similarity_threshold:
                            match_found = True
                            confidence = float(similarity)
                            if confidence > best_confidence:
                                best_confidence = confidence
                    
                    if match_found:
                        matched_photos.append({
                            "photoId": photo_id,
                            "confidence": best_confidence,
                            "matchedAt": None  # Will be set by the caller
                        })
                    else:
                        unmatched_photos.append({
                            "photoId": photo_id,
                            "processedAt": None  # Will be set by the caller
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing pool image {image_path}: {str(e)}")
                    unmatched_photos.append({
                        "photoId": photo_id,
                        "processedAt": None
                    })
            
            logger.info(f"Classification completed. Matched: {len(matched_photos)}, Unmatched: {len(unmatched_photos)}")
            return matched_photos, unmatched_photos
            
        except Exception as e:
            logger.error(f"Error in classify_images: {str(e)}")
            raise