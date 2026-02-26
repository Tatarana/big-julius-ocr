"""
app/services/ocr_processor.py

OCR processor — uses Vertex AI Batch with dual-call architecture.
  - Call 1 (classify): identify bank, doc_type, owner
  - Call 2 (extract):  extract transactions
Both calls run in parallel, results merged for CSV generation.
"""
from pathlib import Path

from app.services.google_drive import drive_service
from app.utils.logger import logger
from app.utils.config import settings
from app.services.job_registry import job_registry
from app.api.models import JobStatus
from app.services.vertex_batch_service import vertex_batch_service, BatchFileEntry


class OCRProcessor:
    def __init__(self):
        self.prompts_dir = Path(__file__).resolve().parent.parent / "prompts"

    # ----------------------------------------------------------------- prompts

    def _load_prompt(self, name: str) -> str:
        """Load a prompt file by name (without extension)."""
        prompt_path = self.prompts_dir / f"{name}.txt"
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            return ""
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _detect_type_from_filename(self, filename: str) -> str:
        """Use filename heuristic to guess doc type for prompt selection only."""
        fn = filename.lower()
        if "fatura" in fn or "cc" in fn or "credit" in fn:
            return "ccstatement"
        return "bankstatement"

    def _get_system_prompt(self) -> str:
        return (
            "Extract bank transactions. Respond ONLY with a JSON object. "
            "Minify output (no whitespace)."
        )

    def _get_classify_system_prompt(self) -> str:
        return (
            "Classify this financial document. Respond ONLY with a JSON object. "
            "Minify output (no whitespace)."
        )

    # ----------------------------------------------------------------- public API

    async def process_folder(self, folder_id: str, job_id: str):
        """Download all files, build dual batch entries, delegate to Vertex AI."""
        logger.info(f"=== Starting folder processing: {folder_id} (Job: {job_id}) ===")
        job_registry.update_job(job_id, status=JobStatus.DOWNLOADING)

        try:
            files = drive_service.list_files_in_folder(
                folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
            )
            logger.info(f"Found {len(files)} files to process")
            job_registry.update_job(job_id, total_files=len(files))

            # Load prompts
            classify_prompt = self._load_prompt("classify")
            classify_sys = self._get_classify_system_prompt()
            extract_sys = self._get_system_prompt()

            classify_entries: list[BatchFileEntry] = []
            extract_entries: list[BatchFileEntry] = []

            for i, file in enumerate(files, 1):
                logger.info(f"[Job {job_id}] Downloading {i}/{len(files)}: {file.name}")
                pdf_bytes = drive_service.download_file(file.id)

                # Classify entry — same prompt for all files
                classify_entries.append(BatchFileEntry(
                    file_id=file.id,
                    file_name=file.name,
                    pdf_bytes=pdf_bytes,
                    user_prompt=classify_prompt,
                    system_prompt=classify_sys,
                ))

                # Extract entry — prompt selected by filename heuristic
                stmt_type = self._detect_type_from_filename(file.name)
                extract_prompt = self._load_prompt(stmt_type)
                logger.info(
                    f"[Job {job_id}] {file.name}: prompt={stmt_type} (filename heuristic)"
                )

                extract_entries.append(BatchFileEntry(
                    file_id=file.id,
                    file_name=file.name,
                    pdf_bytes=pdf_bytes,
                    user_prompt=extract_prompt,
                    system_prompt=extract_sys,
                ))

            if not extract_entries:
                logger.error(f"[Job {job_id}] No files to process — aborting.")
                job_registry.update_job(
                    job_id, status=JobStatus.FAILED,
                    errors=["No files found in folder."],
                )
                return []

            # Fire dual batch — classify + extract in parallel
            await vertex_batch_service.run_dual_batch_job(
                classify_entries, extract_entries, job_id
            )

        except Exception as exc:
            logger.error(f"[Job {job_id}] Processing failed: {exc}", exc_info=True)
            job_registry.update_job(job_id, status=JobStatus.FAILED, errors=[str(exc)])
            raise


ocr_processor = OCRProcessor()
