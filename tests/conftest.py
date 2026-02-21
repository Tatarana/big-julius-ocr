import sys
from unittest.mock import MagicMock, AsyncMock

# Mock pandas before importing app modules
sys.modules["pandas"] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from app.main import app
import os


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
def mock_llm(mocker):
    """Patch the llm_service shim (used by /check-all-connections)."""
    mock = mocker.patch("app.services.llm_service.llm_service")
    mock.check_connection = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_llm_provider(mocker):
    """Patch the provider factory used by ocr_processor."""
    mock_provider = MagicMock()
    mock_provider.call_images = AsyncMock(return_value=[])
    mock_provider.call_chunks = AsyncMock(return_value=[])
    mock_provider.call_pdf = AsyncMock(return_value=[])
    mocker.patch("app.services.llm.factory.get_provider", return_value=mock_provider)
    return mock_provider


@pytest.fixture
def mock_s3(mocker):
    mock = mocker.patch("app.services.s3_service.s3_service")
    mock.check_connection.return_value = True
    return mock


@pytest.fixture
def mock_ocr(mocker):
    # Patch at the endpoint's import location so the background task is intercepted
    mock = mocker.patch("app.api.endpoints.ocr_processor")
    mock.process_folder = AsyncMock(return_value=[])
    return mock
