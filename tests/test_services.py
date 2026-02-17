import pytest
from app.services.ocr_processor import OCRProcessor
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_detect_bank():
    processor = OCRProcessor()
    assert processor.detect_bank("picpay_statement.pdf") == "picpay"
    assert processor.detect_bank("itau_2023.pdf") == "itau"
    assert processor.detect_bank("unknown_file.pdf") == "unknown"

# Add more service tests here
