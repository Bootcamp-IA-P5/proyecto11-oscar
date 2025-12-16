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

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup 
from config.settings import HF_TOKEN, HF_MODEL

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

def generate_image_from_prompt(prompt: str) -> Image.Image | None:
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
                num_inference_steps=8,
                # negative_prompt="bad quality, low resolution, blurry, distorted, ugly",
                width=640,
                height=480
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