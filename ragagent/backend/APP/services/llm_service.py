import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )

    async def generate(self, prompt: str) -> str:
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"<div class='error'>Gemini error: {str(e)}</div>"
        
    # To enable SemanticCache to call it
    async def embed_text(self, text: str):
        from langchain.embeddings import FakeEmbeddings  # Replace with real one if needed
        embedder = FakeEmbeddings(size=384)  # Replace with real embedder for production
        return embedder.embed_query(text)