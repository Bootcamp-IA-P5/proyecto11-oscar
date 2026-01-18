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

def extract_keywords(prompt: str, max_words: int = 5) -> str:
    """
    Extracts the most relevant keywords from an image prompt.
    
    This function removes common words and extracts the core subject matter
    from a detailed image generation prompt, making it suitable for stock
    photo search APIs like Pexels and Unsplash.
    
    Args:
        prompt (str): The detailed image generation prompt.
        max_words (int): Maximum number of keywords to extract (default: 5).
    
    Returns:
        str: Space-separated keywords extracted from the prompt.
    
    Example:
        >>> extract_keywords("A futuristic drone flying over a cityscape")
        "futuristic drone flying cityscape"
    """
    # Extended stop words to filter out
    stop_words = {
        'a', 'an', 'the', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'photorealistic', 'cinematic', 'lighting', 'professional', 'photography',
        '8k', '4k', 'high', 'resolution', 'detailed', 'image', 'picture', 'photo',
        'quality', 'render', 'shot', 'view', 'scene', 'background', 'style'
    }
    
    # Clean and tokenize
    words = prompt.lower()
    # Remove punctuation
    for char in ',.:;!?()[]{}"\'\'`':
        words = words.replace(char, ' ')
    words = words.split()
    
    # Filter and extract important words (nouns, adjectives, verbs)
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    # If we got too few keywords, be less restrictive
    if len(keywords) < 2:
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Return top keywords
    result = ' '.join(keywords[:max_words])
    return result if result else 'abstract art'  # Fallback

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
        keywords = extract_keywords(prompt, max_words=5)
        log.info(f"🔍 Searching Unsplash for: '{keywords}'")
        
        # Use Unsplash API directly for better reliability
        url = "https://api.unsplash.com/photos/random"
        headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
        params = {
            "query": keywords,
            "orientation": "landscape",
            "content_filter": "high"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            image_url = data['urls']['regular']  # High quality version
            photographer = data['user']['name']
            
            # Download the image
            img_response = requests.get(image_url, timeout=30)
            if img_response.status_code == 200:
                image = Image.open(BytesIO(img_response.content))
                log.info(f"✅ Image found on Unsplash (by {photographer})")
                return image
            else:
                log.error(f"❌ Failed to download image from Unsplash: {img_response.status_code}")
                return None
        elif response.status_code == 404:
            log.warning(f"⚠️ No results found on Unsplash for '{keywords}'")
            return None
        elif response.status_code == 403:
            log.error(f"❌ Unsplash API: Access forbidden. Check your API key.")
            return None
        elif response.status_code == 401:
            log.error(f"❌ Unsplash API: Unauthorized. Invalid API key.")
            return None
        else:
            log.error(f"❌ Unsplash API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        log.error("❌ Unsplash request timed out")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"❌ Network error with Unsplash: {e}")
        return None
    except Exception as e:
        log.error(f"❌ Unexpected error searching Unsplash: {type(e).__name__}: {e}")
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
        # Extract relevant keywords from the prompt with more words for better results
        keywords = extract_keywords(prompt, max_words=5)
        log.info(f"🔍 Searching Pexels for: '{keywords}'")
        
        # Pexels API endpoint
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": PEXELS_API_KEY}
        params = {
            "query": keywords,
            "per_page": 5,  # Get top 5 results for better relevance
            "orientation": "landscape",
            "size": "large"
        }
        
        # Search for photos
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('photos') and len(data['photos']) > 0:
                # Get the first (most relevant) result
                photo = data['photos'][0]
                image_url = photo['src']['large2x']  # High quality version
                photographer = photo['photographer']
                
                log.info(f"📊 Found {len(data['photos'])} results on Pexels, using top match")
                
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
                # Try with fewer keywords as fallback
                if len(keywords.split()) > 2:
                    log.info("🔄 Retrying with fewer keywords...")
                    simple_keywords = ' '.join(keywords.split()[:2])
                    params["query"] = simple_keywords
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('photos') and len(data['photos']) > 0:
                            photo = data['photos'][0]
                            image_url = photo['src']['large2x']
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code == 200:
                                image = Image.open(BytesIO(img_response.content))
                                log.info(f"✅ Image found on Pexels with simpler search: '{simple_keywords}'")
                                return image
                return None
        elif response.status_code == 403:
            log.error(f"❌ Pexels API: Access forbidden. Check your API key.")
            return None
        elif response.status_code == 429:
            log.error(f"❌ Pexels API: Rate limit exceeded. Try again later.")
            return None
        else:
            log.error(f"❌ Pexels API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        log.error("❌ Pexels request timed out")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"❌ Network error with Pexels: {e}")
        return None
    except Exception as e:
        log.error(f"❌ Unexpected error searching Pexels: {type(e).__name__}: {e}")
        return None
