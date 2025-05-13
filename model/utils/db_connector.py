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
