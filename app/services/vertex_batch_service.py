"""
app/services/vertex_batch_service.py

Vertex AI Batch Prediction service for Gemini PDF processing.

Supports dual-batch mode:
  - Batch A (classify): identify bank, doc_type, owner per PDF
  - Batch B (extract):  extract transactions per PDF
Both batches run in parallel via asyncio.gather.
Once both complete, results are merged to build the final CSVs.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from google.api_core.exceptions import GoogleAPIError
from google.cloud import aiplatform, storage

from app.api.models import JobStatus
from app.services.job_registry import job_registry
from app.services.s3_service import s3_service
from app.utils.config import settings
from app.utils.logger import logger


# ──────────────────────────────── data classes ─────────────────────────────

@dataclass
class BatchFileEntry:
    """One file to be included in a batch job."""
    file_id: str
    file_name: str
    pdf_bytes: bytes
    user_prompt: str
    system_prompt: str


# ──────────────────────────────── service ──────────────────────────────────

class VertexBatchService:
    """Manages the full lifecycle of Vertex AI Batch Prediction jobs."""

    # ---------------------------------------------------------------- helpers

    def _gcs_client(self) -> storage.Client:
        return storage.Client(project=settings.VERTEX_PROJECT_ID)

    def _build_request_row(self, entry: BatchFileEntry) -> str:
        """Serialize one BatchFileEntry to a JSONL line (GenerateContentRequest)."""
        b64 = base64.b64encode(entry.pdf_bytes).decode("utf-8")
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": b64,
                            }
                        },
                        {"text": entry.user_prompt},
                    ],
                }
            ],
            "system_instruction": {
                "parts": [{"text": entry.system_prompt}]
            },
            "generation_config": {
                "temperature": 0,
                "max_output_tokens": 50000,
                "response_mime_type": "application/json",
            },
        }
        return json.dumps({"request": request})

    def _upload_jsonl_to_gcs(self, content: str, job_id: str, suffix: str = "") -> str:
        """Upload JSONL content to GCS, return gs:// URI."""
        gcs = self._gcs_client()
        bucket = gcs.bucket(settings.GCS_BUCKET)
        filename = f"input{suffix}.jsonl" if suffix else "input.jsonl"
        blob_name = f"{settings.GCS_BATCH_PREFIX}/{job_id}/{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content.encode("utf-8"), content_type="application/jsonl")
        uri = f"gs://{settings.GCS_BUCKET}/{blob_name}"
        logger.info(f"[Job {job_id}] JSONL uploaded → {uri}")
        return uri

    def _submit_vertex_job(
        self, gcs_input_uri: str, gcs_output_prefix: str, display_suffix: str = ""
    ) -> aiplatform.BatchPredictionJob:
        """Submit a Vertex AI BatchPredictionJob and return the job object."""
        aiplatform.init(
            project=settings.VERTEX_PROJECT_ID,
            location=settings.VERTEX_LOCATION,
        )
        job = aiplatform.BatchPredictionJob.submit(
            job_display_name=f"big-julius-ocr-batch{display_suffix}",
            model_name=settings.VERTEX_MODEL,
            gcs_source=gcs_input_uri,
            gcs_destination_prefix=gcs_output_prefix,
            instances_format="jsonl",
            predictions_format="jsonl",
        )
        return job

    def _download_output_jsonl(self, output_directory: str) -> list[dict]:
        """Download all prediction JSONL files from the output GCS directory."""
        prefix = output_directory
        if prefix.startswith("gs://"):
            parts = prefix[len("gs://"):].split("/", 1)
            bucket_name = parts[0]
            prefix_path = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Unexpected output directory format: {output_directory}")

        gcs = self._gcs_client()
        bucket = gcs.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix_path))

        rows: list[dict] = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            logger.info(f"Downloading output blob: {blob.name}")
            content = blob.download_as_text()
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _extract_text_from_response(self, row: dict) -> str | None:
        """Pull the generated text out of a batch output row."""
        try:
            return (
                row["response"]["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Could not extract text from response row: {list(row.keys())}")
            return None

    def _parse_transactions(self, raw: str, bank: str) -> list[dict]:
        """Parse LLM JSON response into a list of transaction dicts."""
        try:
            clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
            data = json.loads(clean)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("transactions", "data", "results", "items"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
            logger.warning(f"[{bank}] Unexpected JSON structure; returning empty list.")
            return []
        except json.JSONDecodeError as exc:
            logger.error(f"[{bank}] JSON parse error: {exc}\nRaw: {raw[:300]}")
            return []

    def _parse_metadata(self, raw: str) -> dict:
        """Parse the classify-batch LLM response into bank/doc_type/owner."""
        defaults = {"bank": "unknown", "doc_type": "bankstatement", "owner": "unknown"}
        try:
            clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
            data = json.loads(clean)
            if isinstance(data, dict):
                return {
                    "bank": str(data.get("bank", "unknown")).lower().strip(),
                    "doc_type": str(data.get("doc_type", "bankstatement")).lower().strip(),
                    "owner": str(data.get("owner", "unknown")).strip(),
                }
        except json.JSONDecodeError as exc:
            logger.error(f"[Classify] JSON parse error: {exc}\nRaw: {raw[:300]}")
        return defaults

    def _transactions_to_csv(
        self, transactions: list[dict], bank: str, doc_type: str, owner: str, extraction_date: str
    ) -> str:
        cols = ["bank", "doc_type", "owner", "extraction_date", "date", "description",
                "installments", "amount", "balance", "category"]
        normalized = []
        for tx in transactions:
            norm = {
                "bank": bank,
                "doc_type": doc_type,
                "owner": owner,
                "extraction_date": extraction_date
            }
            for col in ["date", "description", "installments", "amount", "balance", "category"]:
                val = tx.get(col, tx.get(col.capitalize(), ""))
                if col in ("amount", "balance") and val is not None and val != "":
                    val = str(val).replace(".", ",")
                norm[col] = val
            normalized.append(norm)
        df = pd.DataFrame(normalized)[cols]
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep="|")
        return buf.getvalue()

    # --------------------------------------------------------- batch lifecycle

    async def _build_and_submit(
        self,
        entries: list[BatchFileEntry],
        job_id: str,
        suffix: str,
    ) -> tuple[aiplatform.BatchPredictionJob, str]:
        """Build JSONL, upload to GCS, submit batch job. Returns (job, job_name)."""
        step = f"[Vertex Batch{suffix}]"
        logger.info(f"{step}[Job {job_id}] Building JSONL for {len(entries)} files…")
        lines = [self._build_request_row(e) for e in entries]
        jsonl_content = "\n".join(lines)

        gcs_input_uri = await asyncio.to_thread(
            self._upload_jsonl_to_gcs, jsonl_content, job_id, suffix
        )
        gcs_output_prefix = (
            f"gs://{settings.GCS_BUCKET}/{settings.GCS_BATCH_PREFIX}/{job_id}/output{suffix}/"
        )

        logger.info(f"{step}[Job {job_id}] Submitting Vertex AI batch job…")
        try:
            batch_job = await asyncio.to_thread(
                self._submit_vertex_job, gcs_input_uri, gcs_output_prefix, suffix
            )
        except GoogleAPIError as exc:
            raise RuntimeError(f"Failed to submit Vertex AI batch job{suffix}: {exc}") from exc

        job_name = batch_job.resource_name
        logger.info(f"{step}[Job {job_id}] Job submitted: {job_name}")
        return batch_job, job_name

    async def _poll_until_done(
        self,
        batch_job: aiplatform.BatchPredictionJob,
        job_id: str,
        suffix: str,
    ) -> list[dict]:
        """Poll a batch job until terminal state, then download results."""
        step = f"[Vertex Batch{suffix}]"
        batch_job_name = batch_job.resource_name
        poll_interval = settings.VERTEX_BATCH_POLL_INTERVAL
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while True:
            await asyncio.sleep(poll_interval)
            try:
                batch_job = await asyncio.to_thread(
                    aiplatform.BatchPredictionJob, batch_job_name
                )
                state = batch_job.state.name
            except Exception as exc:
                logger.warning(f"{step}[Job {job_id}] Error refreshing job state: {exc}")
                continue

            logger.info(f"{step}[Job {job_id}] State: {state}")

            if state in terminal_states:
                if state != "JOB_STATE_SUCCEEDED":
                    error_msg = f"Vertex AI batch job{suffix} ended with state {state}"
                    logger.error(f"{step}[Job {job_id}] {error_msg}")
                    raise RuntimeError(error_msg)
                break

        output_dir = batch_job.output_info.gcs_output_directory
        logger.info(f"{step}[Job {job_id}] Downloading results from {output_dir}…")
        output_rows = await asyncio.to_thread(self._download_output_jsonl, output_dir)
        logger.info(f"{step}[Job {job_id}] Got {len(output_rows)} output rows")
        return output_rows

    async def _run_single_pipeline(
        self,
        entries: list[BatchFileEntry],
        job_id: str,
        suffix: str,
    ) -> list[dict]:
        """Submit → poll → download for one batch job. Returns output rows."""
        batch_job, _ = await self._build_and_submit(entries, job_id, suffix)
        return await self._poll_until_done(batch_job, job_id, suffix)

    # -------------------------------------------------------- public interface

    async def run_dual_batch_job(
        self,
        classify_entries: list[BatchFileEntry],
        extract_entries: list[BatchFileEntry],
        job_id: str,
    ) -> None:
        """
        Run two batch jobs in parallel:
          - classify_entries → metadata (bank, doc_type, owner)
          - extract_entries  → transactions

        When both complete, merge results and upload CSVs to S3.
        """
        if not settings.VERTEX_PROJECT_ID or not settings.GCS_BUCKET:
            raise RuntimeError(
                "VERTEX_PROJECT_ID and GCS_BUCKET must be set in .env to use batch mode."
            )

        step = "[Vertex Dual]"
        logger.info(
            f"{step}[Job {job_id}] Starting dual batch: "
            f"{len(classify_entries)} classify + {len(extract_entries)} extract"
        )

        job_registry.update_job(job_id, status=JobStatus.EXTRACTING)

        # ── Fire both batch jobs in parallel ──────────────────────────────
        classify_task = self._run_single_pipeline(classify_entries, job_id, "-classify")
        extract_task = self._run_single_pipeline(extract_entries, job_id, "-extract")

        classify_rows, extract_rows = await asyncio.gather(classify_task, extract_task)

        logger.info(
            f"{step}[Job {job_id}] Both batches done: "
            f"{len(classify_rows)} classify rows, {len(extract_rows)} extract rows"
        )

        # ── Parse classify results ────────────────────────────────────────
        metadata_list: list[dict] = []
        for i, row in enumerate(classify_rows):
            raw = self._extract_text_from_response(row)
            if raw:
                meta = self._parse_metadata(raw)
            else:
                meta = {"bank": "unknown", "doc_type": "bankstatement", "owner": "unknown"}
            logger.info(
                f"{step}[Job {job_id}] Classify result [{i}] "
                f"{classify_entries[i].file_name}: {meta}"
            )
            metadata_list.append(meta)

        # ── Build CSVs using both results ─────────────────────────────────
        job_registry.update_job(job_id, status=JobStatus.BUILDING_CSV)

        extraction_date = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        results = []

        for i, (entry, tx_row) in enumerate(zip(extract_entries, extract_rows)):
            meta = metadata_list[i] if i < len(metadata_list) else {
                "bank": "unknown", "doc_type": "bankstatement", "owner": "unknown"
            }
            bank = meta["bank"]
            doc_type = meta["doc_type"]
            owner = meta["owner"]

            raw_text = self._extract_text_from_response(tx_row)
            if not raw_text:
                logger.warning(f"{step}[Job {job_id}] No text for file {entry.file_name}")
                results.append({"file_id": entry.file_id, "status": "failed",
                                 "error": "Empty response from Vertex AI"})
                continue

            transactions = self._parse_transactions(raw_text, bank)
            if not transactions:
                logger.warning(f"{step}[Job {job_id}] No transactions for {entry.file_name}")
                results.append({"file_id": entry.file_id, "status": "failed",
                                 "error": "No transactions parsed"})
                continue

            human_doc_type = (
                "bank statement" if doc_type == "bankstatement"
                else "credit card statement"
            )
            csv_content = self._transactions_to_csv(
                transactions, bank, human_doc_type, owner, extraction_date
            )

            count = len(transactions)
            output_filename = (
                f"{job_id}-{ts}-{bank}-{doc_type}"
                f"-google-batch-{count}.csv"
            )

            job_registry.update_job(job_id, status=JobStatus.UPLOADING)
            s3_service.upload_file(csv_content, output_filename)
            logger.info(
                f"{step}[Job {job_id}] ✅ {entry.file_name} → {output_filename} ({count} txs)"
            )
            results.append({
                "file_id": entry.file_id,
                "status": "success",
                "output_file": output_filename,
                "transactions_count": count,
                "bank": bank,
                "doc_type": doc_type,
                "owner": owner,
                "llm_provider": "google",
                "llm_model": settings.VERTEX_MODEL,
                "send_mode": "batch-pdf",
            })

        # ── Update job registry ───────────────────────────────────────────
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            processed_files=success,
            failed_files=failed,
            results=results,
        )
        logger.info(
            f"{step}[Job {job_id}] Done — {success} succeeded, {failed} failed."
        )


vertex_batch_service = VertexBatchService()
