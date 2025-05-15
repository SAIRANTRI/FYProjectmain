import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def download_image(image_url: str, save_path: str) -> bool:
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Save the image
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Successfully downloaded image to {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {image_url}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {str(e)}")
        return False