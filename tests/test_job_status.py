import pytest
from app.services.job_registry import job_registry
from app.api.models import JobStatus

def test_job_registry_lifecycle():
    job_id = "test-job-123"
    
    # 1. Create
    job_registry.create_job(job_id)
    status = job_registry.get_job(job_id)
    assert status.job_id == job_id
    assert status.status == JobStatus.PENDING
    
    # 2. Update status
    job_registry.update_job(job_id, status=JobStatus.PROCESSING, total_files=5)
    status = job_registry.get_job(job_id)
    assert status.status == JobStatus.PROCESSING
    assert status.total_files == 5
    
    # 3. Update progress
    job_registry.update_job(job_id, processed_files=1, results=[{"file": "test.pdf", "status": "success"}])
    status = job_registry.get_job(job_id)
    assert status.processed_files == 1
    assert len(status.results) == 1
    
    # 4. Complete
    job_registry.update_job(job_id, status=JobStatus.COMPLETED)
    status = job_registry.get_job(job_id)
    assert status.status == JobStatus.COMPLETED

def test_endpoint_status(client):
    job_id = "test-endpoint-job"
    job_registry.create_job(job_id)
    
    response = client.get(f"/process-status/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert response.json()["status"] == "pending"

def test_endpoint_status_not_found(client):
    response = client.get("/process-status/non-existent-job")
    assert response.status_code == 404
