import os

try:
    from dotenv import load_dotenv
    
    load_dotenv()
except:
    pass

LOG_FILE_NAME = os.getenv("LOG_FILE_NAME","llm_model.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_BASE_DIR = os.getenv("LOG_BASE_DIR", "log")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")