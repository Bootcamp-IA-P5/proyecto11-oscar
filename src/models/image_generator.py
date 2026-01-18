"""
This module provides functions for generating images.

It includes a function to generate an image from a text prompt using the
Hugging Face Inference API, with retry logic and error handling. It also
includes a function to create a placeholder image as a fallback.
"""
from PIL import Image
import time
from huggingface_hub import InferenceClient
from huggingface_hub.utils import RepositoryNotFoundError
import requests
import replicate
from pyunsplash import PyUnsplash
from io import BytesIO

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup 
from config.settings import (
    HF_TOKEN, HF_MODEL, 
    REPLICATE_API_TOKEN, REPLICATE_MODEL,
    PEXELS_API_KEY, UNSPLASH_ACCESS_KEY
)

log_setup()
log = Logger().log

def extract_keywords(prompt: str, max_words: int = 3) -> str:
    """
    Extracts the most relevant keywords from an image prompt.
    
    This function removes common words and extracts the core subject matter
    from a detailed image generation prompt, making it suitable for stock
    photo search APIs like Pexels and Unsplash.
    
    Args:
        prompt (str): The detailed image generation prompt.
        max_words (int): Maximum number of keywords to extract (default: 3).
    
    Returns:
        str: Space-separated keywords extracted from the prompt.
    
    Example:
        >>> extract_keywords("A futuristic drone flying over a cityscape")
        "drone cityscape futuristic"
    """
    # Common words to filter out
    stop_words = {
        'a', 'an', 'the', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
        'photorealistic', 'cinematic', 'lighting', 'professional', 'photography',
        '8k', 'high', 'resolution', 'detailed', 'image', 'picture', 'photo'
    }
    
    # Clean and tokenize
    words = prompt.lower().replace(',', ' ').replace('.', ' ').split()
    
    # Filter and extract important words
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Return top keywords
    return ' '.join(keywords[:max_words])

def create_placeholder_image(text_input: str) -> Image.Image:
    """
    Creates a solid color placeholder image.

    Args:
        text_input (str):   The text input for which the placeholder is created.
                            This is not used in the image generation itself but
                            is kept for API consistency.

    Returns:
        Image.Image: A PIL Image object representing the placeholder.
    """
    W, H = 640, 480
    img = Image.new('RGB', (W, H), color = (73, 109, 137))
    return img

def generate_image_from_huggingface(prompt: str) -> Image.Image | None:
    """
    Generates an image from a text prompt using the Hugging Face Inference API.

    It attempts to generate an image up to MAX_RETRIES times. If the model is
    loading, it will wait and retry. If other errors occur, or all retries fail,
    it returns a placeholder image.

    Args:
        prompt (str): The text prompt to generate the image from.

    Returns:
        Image.Image | None: A PIL Image object of the generated image, or a
                            placeholder image if generation fails.
    """
    client = InferenceClient(token=HF_TOKEN)

    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES):
        try:
            log.info(f"⏳ Trying to create an image with HF. (Try {attempt + 1})...")
            
            generated_image = client.text_to_image(
                model=HF_MODEL,
                prompt=prompt,
                num_inference_steps=60,
                negative_prompt="bad quality, low resolution, blurry, distorted, ugly",
                guidance_scale=8.5,      # Add this for better prompt adherence
                width=1024,
                height=1024
            )
            
            log.info("✅ Image generated successfully (Hugging Face Hub).")
            return generated_image
            

        except RepositoryNotFoundError:
            log.error(f"❌ Error: Model '{HF_MODEL}' not found or not accessible with this token.")
            break
        except Exception as e:
            error_message = str(e).lower()
            
            if "not ready" in error_message or "loading" in error_message:
                wait_time = 15
                log.info(f"⏳ Model loading in HF. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue 
            
            if "rate limit" in error_message or "limit exceeded" in error_message:
                log.error("❌ Free limit reached. Using Mockup.")
                break
                
            log.error(f"❌ Unknown error in HF: {e}")
            break
        
    log.info("⚠️ All try attempts failed. Using Mockup")
    return create_placeholder_image(prompt)

def generate_image_from_replicate(prompt: str) -> str | None:
    """
    Generates an image from a text prompt using the Replicate API.

    This function sends a prompt to the Replicate API to generate an image.
    If the image is generated successfully, it is downloaded and saved to a
    local file.

    Args:
        prompt (str): The text prompt to generate the image from.

    Returns:
        str | None: The file path of the generated image (e.g.,
                    'generated_content.webp') if successful, otherwise None.
    """
    try:
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "webp",
                "output_quality": 90
            }
        )

        image_url = output[0]
        
        response = requests.get(image_url)
        if response.status_code == 200:
            image_path = "generated_content.webp"
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
            
        return None

    except Exception as e:
        log.error(f"Error con Replicate: {e}")
        return None

def search_image_from_unsplash(prompt: str) -> Image.Image | None:
    """
    Searches for a high-quality stock photo on Unsplash based on the prompt.
    
    This function extracts keywords from the AI image generation prompt and
    searches Unsplash's vast library of free, high-quality photos. It returns
    the most relevant image as a PIL Image object.
    
    Args:
        prompt (str): The text prompt describing the desired image.
    
    Returns:
        Image.Image | None: A PIL Image object of the found photo, or None if
                           the search fails or no results are found.
    
    Note:
        Requires UNSPLASH_ACCESS_KEY to be configured in environment variables.
    """
    if not UNSPLASH_ACCESS_KEY:
        log.error("❌ UNSPLASH_ACCESS_KEY not configured")
        return None
    
    try:
        # Extract relevant keywords from the prompt
        keywords = extract_keywords(prompt)
        log.info(f"🔍 Searching Unsplash for: '{keywords}'")
        
        # Initialize Unsplash client
        pu = PyUnsplash(api_key=UNSPLASH_ACCESS_KEY)
        
        # Search for photos
        search = pu.photos(type_='random', count=1, featured=True, query=keywords)
        
        # Get the first result
        photo = search.entries[0]
        image_url = photo.link_download
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            log.info(f"✅ Image found on Unsplash (by {photo.body.get('user', {}).get('name', 'Unknown')})")
            return image
        else:
            log.error(f"❌ Failed to download image from Unsplash: {response.status_code}")
            return None
            
    except IndexError:
        log.warning(f"⚠️ No results found on Unsplash for '{keywords}'")
        return None
    except Exception as e:
        log.error(f"❌ Error searching Unsplash: {e}")
        return None

def search_image_from_pexels(prompt: str) -> Image.Image | None:
    """
    Searches for a high-quality stock photo on Pexels based on the prompt.
    
    This function extracts keywords from the AI image generation prompt and
    searches Pexels' collection of free stock photos. It returns the most
    relevant image as a PIL Image object.
    
    Args:
        prompt (str): The text prompt describing the desired image.
    
    Returns:
        Image.Image | None: A PIL Image object of the found photo, or None if
                           the search fails or no results are found.
    
    Note:
        Requires PEXELS_API_KEY to be configured in environment variables.
    """
    if not PEXELS_API_KEY:
        log.error("❌ PEXELS_API_KEY not configured")
        return None
    
    try:
        # Extract relevant keywords from the prompt
        keywords = extract_keywords(prompt)
        log.info(f"🔍 Searching Pexels for: '{keywords}'")
        
        # Pexels API endpoint
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": PEXELS_API_KEY}
        params = {
            "query": keywords,
            "per_page": 1,
            "orientation": "landscape"
        }
        
        # Search for photos
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('photos') and len(data['photos']) > 0:
                photo = data['photos'][0]
                image_url = photo['src']['large2x']  # High quality version
                photographer = photo['photographer']
                
                # Download the image
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    image = Image.open(BytesIO(img_response.content))
                    log.info(f"✅ Image found on Pexels (by {photographer})")
                    return image
                else:
                    log.error(f"❌ Failed to download image from Pexels: {img_response.status_code}")
                    return None
            else:
                log.warning(f"⚠️ No results found on Pexels for '{keywords}'")
                return None
        else:
            log.error(f"❌ Pexels API error: {response.status_code}")
            return None
            
    except Exception as e:
        log.error(f"❌ Error searching Pexels: {e}")
        return None
