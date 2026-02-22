import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Google Drive
    GOOGLE_DRIVE_FOLDER_ID: str
    GOOGLE_CREDENTIALS_PATH: str = "./credentials.json"

    # LLM
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4-turbo-preview"
    GOOGLE_API_KEY: Optional[str] = None
    
    SECOND_LLM_PROVIDER: str = "google"
    SECOND_LLM_MODEL: str = "gemini-3-pro-image-preview"

    # Vertex AI Batch (optional — only needed when using batch mode)
    VERTEX_PROJECT_ID: Optional[str] = None
    VERTEX_LOCATION: str = "us-central1"
    VERTEX_MODEL: str = "publishers/google/models/gemini-2.0-flash-001"
    GCS_BUCKET: Optional[str] = None
    GCS_BATCH_PREFIX: str = "batch-jobs"
    VERTEX_BATCH_POLL_INTERVAL: int = 60  # seconds between status checks

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET: str
    AWS_REGION: str = "us-east-1"

    # App
    LOG_LEVEL: str = "INFO"
    DEBUG_SAVE_IMAGES: bool = False
    MAX_CONCURRENT_JOBS: int = 5
    PROMPT_CONFIG_PATH: str = "./app/prompts/config.yaml"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
