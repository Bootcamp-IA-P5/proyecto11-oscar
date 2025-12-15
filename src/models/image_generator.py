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
    W, H = 640, 480
    img = Image.new('RGB', (W, H), color = (73, 109, 137))
    return img

def generate_image_from_prompt(prompt: str) -> Image.Image | None:
    client = InferenceClient(token=HF_TOKEN)

    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES):
        try:
            log.info(f"⏳ Trying to create an image with HF. (Try {attempt + 1})...")
            
            generated_image = client.text_to_image(
                model=HF_MODEL,
                prompt=prompt,
                negative_prompt="bad quality, low resolution, blurry, distorted, ugly",
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