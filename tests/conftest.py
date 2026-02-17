import sys
from unittest.mock import MagicMock

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
    mock = mocker.patch("app.services.llm_service.llm_service")
    mock.check_connection.return_value = True
    mock.extract_transactions.return_value = '{"transactions": []}'
    return mock

@pytest.fixture
def mock_s3(mocker):
    mock = mocker.patch("app.services.s3_service.s3_service")
    mock.check_connection.return_value = True
    return mock
    
@pytest.fixture
def mock_ocr(mocker):
    mock = mocker.patch("app.services.ocr_processor.ocr_processor")
    return mock
