from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class ProcessFilesRequest(BaseModel):
    folder_id: Optional[str] = None

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
    status: str = "pending" # pending, processing, processed, failed

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
    timestamp: datetime
