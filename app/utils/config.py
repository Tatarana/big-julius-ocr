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

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET: str
    AWS_REGION: str = "us-east-1"

    # App
    LOG_LEVEL: str = "INFO"
    MAX_CONCURRENT_JOBS: int = 5
    PROMPT_CONFIG_PATH: str = "./app/prompts/config.yaml"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
