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
        raw = '{"bank": "itau", "doc_type": "ccstatement", "owner": "FERNANDO REIS", "payment_date": "15-03-2026"}'
        result = self.svc._parse_metadata(raw)
        assert result["bank"] == "itau"
        assert result["doc_type"] == "ccstatement"
        assert result["owner"] == "FERNANDO REIS"
        assert result["payment_date"] == "15-03-2026"

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
        assert result["payment_date"] == ""

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
        assert result["payment_date"] == ""

    def test_parse_payment_date_for_cc(self):
        raw = '{"bank": "xp", "doc_type": "ccstatement", "owner": "ANA", "payment_date": "10-04-2026"}'
        result = self.svc._parse_metadata(raw)
        assert result["payment_date"] == "10-04-2026"

    def test_parse_payment_date_empty_for_bankstatement(self):
        raw = '{"bank": "itau", "doc_type": "bankstatement", "owner": "JOAO", "payment_date": ""}'
        result = self.svc._parse_metadata(raw)
        assert result["payment_date"] == ""


class TestExtractFileIdFromRow:
    """Test the _extract_file_id_from_row helper that matches output rows to input files."""

    def setup_method(self):
        self.svc = VertexBatchService()

    def test_extract_valid_file_id(self):
        row = {
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "application/pdf", "data": "..."}},
                        {"text": "classify this document"},
                        {"text": "__TRACK__:abc123"},
                    ]
                }]
            },
            "response": {"candidates": []}
        }
        assert self.svc._extract_file_id_from_row(row) == "abc123"

    def test_extract_file_id_with_colons(self):
        """file_id may contain colons — only split on the first one."""
        row = {
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": "__TRACK__:id:with:colons"},
                    ]
                }]
            }
        }
        assert self.svc._extract_file_id_from_row(row) == "id:with:colons"

    def test_missing_track_tag_returns_none(self):
        row = {
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": "just a normal prompt"},
                    ]
                }]
            }
        }
        assert self.svc._extract_file_id_from_row(row) is None

    def test_extract_file_id_with_null_text(self):
        """Vertex AI might return parts with 'text': null, which should be ignored."""
        row = {
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "application/pdf", "data": "..."}},
                        {"text": None},
                        {"text": "__TRACK__:xyz789"},
                    ]
                }]
            }
        }
        assert self.svc._extract_file_id_from_row(row) == "xyz789"

    def test_malformed_row_returns_none(self):
        assert self.svc._extract_file_id_from_row({}) is None
        assert self.svc._extract_file_id_from_row({"request": {}}) is None
        assert self.svc._extract_file_id_from_row({"request": {"contents": []}}) is None


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
