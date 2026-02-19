from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.models import (
    ProcessFilesRequest, ProcessFilesResponse, 
    ConnectionCheckResponse, ServiceStatus, FileMetadata
)
from app.services.google_drive import drive_service
from app.services.llm_service import llm_service
from app.services.s3_service import s3_service
from app.services.ocr_processor import ocr_processor
from app.utils.config import settings
from app.utils.logger import logger
from datetime import datetime
import uuid

router = APIRouter()

@router.post("/process-files", response_model=ProcessFilesResponse)
async def process_files(request: ProcessFilesRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # Handle the case where Swagger UI sends "string" as default value
    folder_id = request.folder_id
    if not folder_id or folder_id.lower() == "string":
        folder_id = settings.GOOGLE_DRIVE_FOLDER_ID
    
    logger.info(f"[Job {job_id}] Starting process-files for folder_id={folder_id}")
    
    background_tasks.add_task(ocr_processor.process_folder, folder_id, job_id)
    
    return ProcessFilesResponse(
        job_id=job_id,
        message="Processing started in background"
    )

@router.get("/list-input-files", response_model=list[FileMetadata])
async def list_input_files(folder_id: str = None):
    # Handle the case where Swagger UI sends "string" as default value
    fid = folder_id
    if not fid or fid.lower() == "string":
        fid = settings.GOOGLE_DRIVE_FOLDER_ID
        
    try:
        return drive_service.list_files_in_folder(fid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check-all-connections", response_model=ConnectionCheckResponse)
async def check_all_connections():
    drive_status = drive_service.check_connection()
    llm_status = await llm_service.check_connection()
    s3_status = s3_service.check_connection()
    
    return ConnectionCheckResponse(
        google_drive=ServiceStatus(status="connected" if drive_status else "disconnected", latency_ms=0),
        llm_api=ServiceStatus(status="connected" if llm_status else "disconnected", latency_ms=0),
        aws_s3=ServiceStatus(status="connected" if s3_status else "disconnected", latency_ms=0),
        timestamp=datetime.utcnow()
    )

@router.get("/show-config")
async def show_config():
    # Return sanitized config
    return {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_MODEL": settings.LLM_MODEL,
        "AWS_REGION": settings.AWS_REGION,
        "AWS_S3_BUCKET": settings.AWS_S3_BUCKET,
        "GOOGLE_DRIVE_FOLDER_ID": settings.GOOGLE_DRIVE_FOLDER_ID,
        "LOG_LEVEL": settings.LOG_LEVEL,
        "DEBUG_SAVE_IMAGES": settings.DEBUG_SAVE_IMAGES
    }
