"""
Tests for OCRProcessor — prompt loading and filename heuristic.
"""
import pytest
from app.services.ocr_processor import OCRProcessor


class TestDetectTypeFromFilename:
    """Filename heuristic is only used for prompt selection."""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_fatura_detected_as_cc(self):
        assert self.processor._detect_type_from_filename("picpay_fatura_2024.pdf") == "ccstatement"

    def test_cc_detected(self):
        assert self.processor._detect_type_from_filename("itau_cc_march.pdf") == "ccstatement"

    def test_default_is_bankstatement(self):
        assert self.processor._detect_type_from_filename("itau_extrato_2024.pdf") == "bankstatement"

    def test_credit_keyword(self):
        assert self.processor._detect_type_from_filename("credit_card_stmt.pdf") == "ccstatement"


class TestLoadPrompt:
    """Verify prompts load correctly."""

    def setup_method(self):
        self.processor = OCRProcessor()

    def test_load_ccstatement_prompt(self):
        prompt = self.processor._load_prompt("ccstatement")
        assert "credit card" in prompt.lower()

    def test_load_bankstatement_prompt(self):
        prompt = self.processor._load_prompt("bankstatement")
        assert "bank" in prompt.lower()

    def test_load_classify_prompt(self):
        prompt = self.processor._load_prompt("classify")
        assert "bank" in prompt.lower()
        assert "doc_type" in prompt.lower()
        assert "owner" in prompt.lower()

    def test_load_nonexistent_returns_empty(self):
        prompt = self.processor._load_prompt("nonexistent_prompt")
        assert prompt == ""
