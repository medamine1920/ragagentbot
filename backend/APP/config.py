import os
from dotenv import load_dotenv
from pydantic import Field
from typing import Optional
from pydantic_settings import BaseSettings
load_dotenv()

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(..., env="ALGORITHM")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ✅ Expose them here so you can import directly
#SECRET_KEY = settings.SECRET_KEY
#ALGORITHM = settings.ALGORITHM