from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SendMode(str, Enum):
    CHUNKS = "chunks"     # PDF → N images → N separate LLM calls → merge
    IMAGES = "images"     # PDF → all images → 1 LLM call
    PDF = "pdf"           # raw PDF → 1 LLM call (Google only)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    results: List[Any] = []
    errors: List[str] = []
    created_at: datetime
    updated_at: datetime


class ProcessFilesRequest(BaseModel):
    llm_provider: str = "google"                        # "openai" | "google"
    llm_model: str = "gemini-3.1-pro-preview"           # override anytime
    send_mode: SendMode = SendMode.PDF                  # "chunks" | "images" | "pdf"


class JobStats(BaseModel):
    total_files: int
    processed_files: int
    failed_files: int
    start_time: datetime
    end_time: Optional[datetime] = None


class ProcessFilesResponse(BaseModel):
    job_id: str
    message: str


class FileMetadata(BaseModel):
    id: str
    name: str
    size: str
    created_time: datetime
    status: str = "pending"  # pending, processing, processed, failed


class Transaction(BaseModel):
    date: str
    description: str
    installments: str = ""
    amount: float
    balance: Optional[float] = None
    category: str

    @field_validator('amount')
    def round_amount(cls, v):
        return round(v, 2)


class ServiceStatus(BaseModel):
    status: str
    latency_ms: float
    message: Optional[str] = None


class ConnectionCheckResponse(BaseModel):
    google_drive: ServiceStatus
    llm_api: ServiceStatus
    aws_s3: ServiceStatus
    vertex_ai: Optional[ServiceStatus] = None
    timestamp: datetime
