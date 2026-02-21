from datetime import datetime
from typing import Dict, Any, Optional
from app.api.models import JobStatus, JobStatusResponse

class JobRegistry:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str):
        now = datetime.utcnow()
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "results": [],
            "errors": [],
            "created_at": now,
            "updated_at": now
        }

    def update_job(self, job_id: str, **kwargs):
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)
            self._jobs[job_id]["updated_at"] = datetime.utcnow()

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        job_data = self._jobs.get(job_id)
        if job_data:
            return JobStatusResponse(**job_data)
        return None

job_registry = JobRegistry()
