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

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup 
from config.settings import HF_TOKEN, HF_MODEL, REPLICATE_API_TOKEN, REPLICATE_MODEL

log_setup()
log = Logger().log

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
