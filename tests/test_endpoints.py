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

def test_process_files(client, mock_ocr):
    response = client.post("/process-files", json={"folder_id": "123"})
    assert response.status_code == 200
    assert "job_id" in response.json()
