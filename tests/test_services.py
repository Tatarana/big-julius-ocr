import pytest
from app.services.ocr_processor import OCRProcessor


@pytest.mark.asyncio
async def test_detect_bank():
    processor = OCRProcessor()
    assert processor.detect_bank("picpay_statement.pdf") == "picpay"
    assert processor.detect_bank("itau_2023.pdf") == "itau"
    assert processor.detect_bank("xp_invest.pdf") == "xp"
    assert processor.detect_bank("unknown_file.pdf") == "unknown"


@pytest.mark.asyncio
async def test_detect_type():
    processor = OCRProcessor()
    assert processor.detect_type("picpay_fatura_2024.pdf") == "ccstatement"
    assert processor.detect_type("itau_cc_march.pdf") == "ccstatement"
    assert processor.detect_type("itau_extrato_2024.pdf") == "bankstatement"


def test_get_prompt():
    processor = OCRProcessor()
    cc_prompt = processor._get_prompt("ccstatement")
    assert "credit card" in cc_prompt.lower()
    
    bank_prompt = processor._get_prompt("bankstatement")
    assert "bank" in bank_prompt.lower()
