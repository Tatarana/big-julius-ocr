"""
app/services/ocr_processor.py

Simplified OCR processor — uses Vertex AI Batch exclusively.
All processing goes through vertex_batch_service.
"""
from pathlib import Path

from app.services.google_drive import drive_service
from app.services.s3_service import s3_service
from app.utils.logger import logger
from app.utils.config import settings
from app.services.job_registry import job_registry
from app.api.models import JobStatus
from app.services.vertex_batch_service import vertex_batch_service, BatchFileEntry


class OCRProcessor:
    def __init__(self):
        self.prompts_dir = Path(__file__).resolve().parent.parent / "prompts"

    # ----------------------------------------------------------------- detection

    def detect_bank(self, filename: str) -> str:
        filename = filename.lower()
        if "picpay" in filename:
            return "picpay"
        elif "itau" in filename:
            return "itau"
        elif "xp" in filename:
            return "xp"
        return "unknown"

    def detect_type(self, filename: str) -> str:
        filename = filename.lower()
        if "fatura" in filename or "cc" in filename or "credit" in filename:
            return "ccstatement"
        return "bankstatement"

    # ----------------------------------------------------------------- prompts

    def _get_prompt(self, stmt_type: str) -> str:
        """Load the universal prompt for the given statement type."""
        prompt_path = self.prompts_dir / f"{stmt_type}.txt"
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            return ""
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_system_prompt(self) -> str:
        return (
            "Extract bank transactions. Respond ONLY with a JSON object. "
            "Minify output (no whitespace)."
        )

    # ----------------------------------------------------------------- public API

    async def process_folder(self, folder_id: str, job_id: str):
        """Download all files, build batch entries, delegate to Vertex AI Batch."""
        logger.info(f"=== Starting folder processing: {folder_id} (Job: {job_id}) ===")
        job_registry.update_job(job_id, status=JobStatus.PROCESSING)

        try:
            files = drive_service.list_files_in_folder(
                folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
            )
            logger.info(f"Found {len(files)} files to process")
            job_registry.update_job(job_id, total_files=len(files))

            # Build batch entries
            entries: list[BatchFileEntry] = []
            for i, file in enumerate(files, 1):
                logger.info(f"[Job {job_id}] Downloading {i}/{len(files)}: {file.name}")
                pdf_bytes = drive_service.download_file(file.id)

                bank = self.detect_bank(file.name)
                if bank == "unknown":
                    logger.warning(f"[Job {job_id}] Unknown bank for {file.name} — skipping.")
                    continue

                stmt_type = self.detect_type(file.name)
                user_prompt = self._get_prompt(stmt_type)
                sys_prompt = self._get_system_prompt()

                entries.append(BatchFileEntry(
                    file_id=file.id,
                    file_name=file.name,
                    bank=bank,
                    stmt_type=stmt_type,
                    pdf_bytes=pdf_bytes,
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                ))

            if not entries:
                logger.error(f"[Job {job_id}] No valid files to batch — aborting.")
                job_registry.update_job(
                    job_id, status=JobStatus.FAILED,
                    errors=["No files with known bank detected."],
                )
                return []

            # Delegate to Vertex AI Batch — blocks until done
            await vertex_batch_service.run_batch_job(entries, job_id)

        except Exception as exc:
            logger.error(f"[Job {job_id}] Processing failed: {exc}", exc_info=True)
            job_registry.update_job(job_id, status=JobStatus.FAILED, errors=[str(exc)])
            raise


ocr_processor = OCRProcessor()
