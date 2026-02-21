def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_input_files(client, mock_google_drive):
    response = client.get("/list-input-files")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_check_connections(client, mock_google_drive, mock_llm, mock_s3):
    response = client.get("/check-all-connections")
    assert response.status_code == 200
    data = response.json()
    assert data["google_drive"]["status"] == "connected"
    assert data["llm_api"]["status"] == "connected"
    assert data["aws_s3"]["status"] == "connected"


def test_process_files_default(client, mock_ocr):
    """Default request: google provider, pdf mode."""
    response = client.post("/process-files", json={})
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert "google" in body["message"]
    assert "pdf" in body["message"]


def test_process_files_openai_chunks(client, mock_ocr):
    """Explicit OpenAI + chunks mode."""
    response = client.post(
        "/process-files",
        json={"llm_provider": "openai", "send_mode": "chunks"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "openai" in body["message"]
    assert "chunks" in body["message"]


def test_process_files_google_pdf(client, mock_ocr):
    """Gemini PDF mode request."""
    response = client.post(
        "/process-files",
        json={"llm_provider": "google", "send_mode": "pdf"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "google" in body["message"]
    assert "pdf" in body["message"]


def test_process_files_invalid_mode(client, mock_ocr):
    """Unknown send_mode should be rejected by the Pydantic model."""
    response = client.post(
        "/process-files",
        json={"send_mode": "telepathy"},
    )
    assert response.status_code == 422  # Validation error
