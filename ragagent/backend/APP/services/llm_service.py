import google.generativeai as genai
from config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("Missing Gemini API key")
                
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info("Gemini service initialized")
        except Exception as e:
            logger.error(f"Gemini init failed: {str(e)}")
            raise