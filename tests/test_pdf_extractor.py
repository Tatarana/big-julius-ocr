"""
Tests for the dual-batch architecture components.
Tests the metadata parsing and prompt loading logic.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.vertex_batch_service import VertexBatchService


class TestParseMetadata:
    """Test the _parse_metadata helper that processes classify-batch output."""

    def setup_method(self):
        self.svc = VertexBatchService()

    def test_parse_valid_json(self):
        raw = '{"bank": "itau", "doc_type": "ccstatement", "owner": "FERNANDO REIS"}'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "itau"
        assert result["doc_type"] == "ccstatement"
        assert result["owner"] == "FERNANDO REIS"

    def test_parse_with_markdown_fences(self):
        raw = '```json\n{"bank": "picpay", "doc_type": "ccstatement", "owner": "MARIA"}\n```'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "picpay"
        assert result["doc_type"] == "ccstatement"
        assert result["owner"] == "MARIA"

    def test_parse_unknown_bank(self):
        raw = '{"bank": "nubank", "doc_type": "bankstatement", "owner": "JOAO"}'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "nubank"
        assert result["doc_type"] == "bankstatement"
        assert result["owner"] == "JOAO"

    def test_parse_invalid_json_returns_defaults(self):
        raw = "this is not json"
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "unknown"
        assert result["doc_type"] == "bankstatement"
        assert result["owner"] == "unknown"

    def test_parse_empty_string_returns_defaults(self):
        raw = ""
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "unknown"

    def test_parse_normalizes_case(self):
        raw = '{"bank": "ITAU", "doc_type": "CcStatement", "owner": "Someone"}'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "itau"
        assert result["doc_type"] == "ccstatement"

    def test_parse_missing_fields_uses_defaults(self):
        raw = '{"bank": "xp"}'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "xp"
        assert result["doc_type"] == "bankstatement"
        assert result["owner"] == "unknown"


class TestDetectTypeFromFilename:
    """Test the filename heuristic used for prompt selection."""

    def setup_method(self):
        from app.services.ocr_processor import OCRProcessor
        self.processor = OCRProcessor()

    def test_fatura_in_filename(self):
        assert self.processor._detect_type_from_filename("Fatura-itau.pdf") == "ccstatement"

    def test_cc_in_filename(self):
        assert self.processor._detect_type_from_filename("cc-statement.pdf") == "ccstatement"

    def test_default_bankstatement(self):
        assert self.processor._detect_type_from_filename("9223724-Xp-25-02-2026.pdf") == "bankstatement"

    def test_picpay_fatura(self):
        assert self.processor._detect_type_from_filename("PicPay_Fatura_122025.pdf") == "ccstatement"
