import sys
from unittest.mock import MagicMock, AsyncMock

# Mock pandas before importing app modules
sys.modules["pandas"] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_google_drive(mocker):
    mock = mocker.patch("app.services.google_drive.drive_service")
    mock.list_files_in_folder.return_value = []
    mock.check_connection.return_value = True
    return mock


@pytest.fixture
def mock_s3(mocker):
    mock = mocker.patch("app.services.s3_service.s3_service")
    mock.check_connection.return_value = True
    return mock


@pytest.fixture
def mock_vertex(mocker):
    """Mock GCS + Vertex AI for /check-all-connections."""
    mock_storage = MagicMock()
    mock_bucket = MagicMock()
    mock_storage.return_value.bucket.return_value = mock_bucket
    mocker.patch("app.api.endpoints.gcs_storage", mock_storage, create=True)
    return mock_storage


@pytest.fixture
def mock_ocr(mocker):
    mock = mocker.patch("app.api.endpoints.ocr_processor")
    mock.process_folder = AsyncMock(return_value=[])
    return mock
