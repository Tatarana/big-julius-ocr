from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.models import (
    ProcessFilesRequest, ProcessFilesResponse,
    ConnectionCheckResponse, ServiceStatus, FileMetadata,
    JobStatusResponse
)
from app.services.job_registry import job_registry
from app.services.google_drive import drive_service
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
    folder_id = settings.GOOGLE_DRIVE_FOLDER_ID

    logger.info(f"[Job {job_id}] process-files | folder={folder_id} | Vertex AI Batch")

    job_registry.create_job(job_id)

    background_tasks.add_task(
        ocr_processor.process_folder,
        folder_id,
        job_id,
    )

    return ProcessFilesResponse(
        job_id=job_id,
        message="Processing started via Vertex AI Batch",
    )


@router.get("/process-status/{job_id}", response_model=JobStatusResponse)
async def get_process_status(job_id: str):
    status = job_registry.get_job(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status


@router.get("/list-input-files", response_model=list[FileMetadata])
async def list_input_files(folder_id: str = None):
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
    s3_status = s3_service.check_connection()

    # Vertex AI check
    try:
        from google.cloud import storage as gcs_storage
        from google.cloud import aiplatform

        client = gcs_storage.Client(project=settings.VERTEX_PROJECT_ID)
        bucket = client.bucket(settings.GCS_BUCKET)
        bucket.reload()

        aiplatform.init(
            project=settings.VERTEX_PROJECT_ID,
            location=settings.VERTEX_LOCATION,
        )

        vertex_status = ServiceStatus(
            status="connected",
            latency_ms=0,
            message=f"project={settings.VERTEX_PROJECT_ID}, bucket={settings.GCS_BUCKET}",
        )
    except Exception as e:
        vertex_status = ServiceStatus(
            status="disconnected",
            latency_ms=0,
            message=str(e) or repr(e),
        )

    from datetime import timezone
    return ConnectionCheckResponse(
        google_drive=ServiceStatus(status="connected" if drive_status else "disconnected", latency_ms=0),
        aws_s3=ServiceStatus(status="connected" if s3_status else "disconnected", latency_ms=0),
        vertex_ai=vertex_status,
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/show-config")
async def show_config():
    return {
        "VERTEX_PROJECT_ID": settings.VERTEX_PROJECT_ID,
        "VERTEX_LOCATION": settings.VERTEX_LOCATION,
        "VERTEX_MODEL": settings.VERTEX_MODEL,
        "GCS_BUCKET": settings.GCS_BUCKET,
        "AWS_REGION": settings.AWS_REGION,
        "AWS_S3_BUCKET": settings.AWS_S3_BUCKET,
        "GOOGLE_DRIVE_FOLDER_ID": settings.GOOGLE_DRIVE_FOLDER_ID,
        "LOG_LEVEL": settings.LOG_LEVEL,
    }
