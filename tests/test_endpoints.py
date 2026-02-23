def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_input_files(client, mock_google_drive):
    response = client.get("/list-input-files")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_process_files(client, mock_ocr):
    """Default request — no params needed."""
    response = client.post("/process-files", json={})
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert "Vertex AI" in body["message"]


def test_show_config(client):
    response = client.get("/show-config")
    assert response.status_code == 200
    data = response.json()
    assert "VERTEX_PROJECT_ID" in data
    assert "GCS_BUCKET" in data
