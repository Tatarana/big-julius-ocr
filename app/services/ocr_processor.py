import json
import pandas as pd
import io
import asyncio
from datetime import datetime
from pathlib import Path
import yaml
import fitz  # PyMuPDF
import base64
from app.services.google_drive import drive_service
from app.services.llm.factory import get_provider
from app.services.s3_service import s3_service
from app.utils.logger import logger
from app.utils.config import settings
from app.services.job_registry import job_registry
from app.api.models import JobStatus
from app.services.vertex_batch_service import vertex_batch_service, BatchFileEntry


class OCRProcessor:
    def __init__(self):
        # Determine base directory for prompts
        self.base_dir = Path(__file__).resolve().parent.parent
        self.prompts_dir = self.base_dir / "prompts"
        self.prompts_config = self._load_prompts_config()

    # ----------------------------------------------------------------- config

    def _load_prompts_config(self) -> dict:
        try:
            config_path = Path(settings.PROMPT_CONFIG_PATH)
            if not config_path.is_absolute():
                config_path = self.prompts_dir / "config.yaml"
            logger.info(f"Loading prompt config from: {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load prompt config: {e}")
            return {"banks": {}}

    # --------------------------------------------------------------- detection

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

    # -------------------------------------------------------------- prompt I/O

    def _resolve_prompt(
        self,
        bank: str,
        provider_key: str,
        stmt_type: str,
        send_mode: str,
    ) -> tuple[str, str]:
        """
        Return (user_prompt_text, system_prompt_text) for the given combination.
        Falls back gracefully if a specific prompt file is missing.
        """
        banks_cfg = self.prompts_config.get("banks", {})
        prompt_rel = (
            banks_cfg
            .get(bank, {})
            .get(provider_key, {})
            .get(stmt_type, {})
            .get(send_mode)
        )

        user_prompt = ""
        if prompt_rel:
            prompt_path = self.prompts_dir / prompt_rel
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    user_prompt = f.read()
            else:
                logger.warning(
                    f"Prompt file not found: {prompt_path}. Using empty prompt."
                )
        else:
            logger.warning(
                f"No prompt configured for bank={bank}, provider={provider_key}, "
                f"type={stmt_type}, mode={send_mode}. Using empty prompt."
            )

        system_prompt = self.prompts_config.get("default_system_prompt", "")
        return user_prompt, system_prompt

    # ------------------------------------------------------- page preparation

    def _pdf_to_b64_images(
        self, doc: fitz.Document, page_indices: list[int], job_id: str, file_name: str
    ) -> list[str]:
        """Convert selected PDF pages to base64-encoded JPEG strings."""
        b64_images = []
        for page_num in page_indices:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("jpg")

            if settings.DEBUG_SAVE_IMAGES:
                project_root = Path(settings.PROMPT_CONFIG_PATH).resolve().parent.parent
                temp_dir = project_root / "temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                img_path = temp_dir / f"{job_id or 'debug'}_{file_name}_p{page_num}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                logger.info(f"DEBUG: Saved page image to {img_path}")

            b64_images.append(base64.b64encode(img_data).decode("utf-8"))
        return b64_images

    def _get_page_indices(self, bank: str, doc: fitz.Document) -> list[int]:
        """Return the page indices to process (bank-specific exclusions)."""
        num_pages = len(doc)
        page_indices = list(range(num_pages))

        if bank == "picpay":
            if num_pages > 3:
                page_indices = list(range(2, num_pages - 1))
                logger.info(
                    f"PicPay: skipping first 2 and last page → processing {len(page_indices)} pages"
                )
            else:
                page_indices = []
                logger.warning(
                    f"PicPay PDF has only {num_pages} pages; all skipped by exclusion rule."
                )

        return page_indices

    # ----------------------------------------- processing strategy dispatchers

    async def _run_chunks(
        self, provider, pages_b64: list[str], prompt: str, sys_prompt: str, bank: str
    ) -> list[dict]:
        return await provider.call_chunks(pages_b64, prompt, sys_prompt, bank)

    async def _run_images(
        self, provider, pages_b64: list[str], prompt: str, sys_prompt: str, bank: str
    ) -> list[dict]:
        return await provider.call_images(pages_b64, prompt, sys_prompt, bank)

    async def _run_pdf(
        self, provider, pdf_bytes: bytes, prompt: str, sys_prompt: str, bank: str
    ) -> list[dict]:
        return await provider.call_pdf(pdf_bytes, prompt, sys_prompt, bank)

    # ----------------------------------------------------------- CSV + upload

    def _transactions_to_csv(
        self,
        transactions: list[dict],
        bank: str,
        doc_type: str,
        extraction_date: str
    ) -> str:
        cols = [
            "bank",
            "doc_type",
            "extraction_date",
            "date",
            "description",
            "installments",
            "amount",
            "balance",
            "category",
        ]
        normalized = []
        for tx in transactions:
            norm = {
                "bank": bank,
                "doc_type": doc_type,
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

    # ------------------------------------------------------------- public API

    async def process_file(
        self,
        file_id: str,
        file_name: str,
        job_id: str = None,
        llm_provider: str = "google",
        llm_model: str = None,
        send_mode: str = "images",
    ) -> dict:
        try:
            # ---- 1. Download
            logger.info(f"[Step 1/5] Downloading: {file_name} ({file_id})")
            content_bytes = drive_service.download_file(file_id)
            logger.info(f"[Step 1/5] Done: {len(content_bytes)} bytes")

            # ---- 2. Detect bank & type
            bank = self.detect_bank(file_name)
            if bank == "unknown":
                raise ValueError(f"Could not detect bank for file: {file_name}")
            stmt_type = self.detect_type(file_name)
            logger.info(f"[Step 2/5] bank={bank}, type={stmt_type}")

            # ---- 3. Resolve model
            if not llm_model:
                llm_model = (
                    settings.LLM_MODEL
                    if llm_provider == "openai"
                    else settings.SECOND_LLM_MODEL
                )

            # ---- 4. Resolve prompt
            user_prompt, sys_prompt = self._resolve_prompt(
                bank, llm_provider, stmt_type, send_mode
            )

            # ---- 5. Get provider
            provider = get_provider(llm_provider, llm_model)

            # ---- 6. Process based on send_mode
            all_transactions: list[dict] = []

            if send_mode == "pdf":
                logger.info(f"[Step 3/5] send_mode=pdf — sending raw PDF to {llm_provider}/{llm_model}")
                all_transactions = await self._run_pdf(
                    provider, content_bytes, user_prompt, sys_prompt, bank
                )

            else:  # chunks or images — need page images
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                page_indices = self._get_page_indices(bank, doc)

                if not page_indices:
                    doc.close()
                    raise ValueError(f"No pages to process for {file_name}")

                logger.info(f"[Step 3/5] Rasterising {len(page_indices)} pages…")
                pages_b64 = self._pdf_to_b64_images(doc, page_indices, job_id or "debug", file_name)
                doc.close()

                if send_mode == "chunks":
                    logger.info(f"[Step 3/5] send_mode=chunks → {len(pages_b64)} separate LLM calls")
                    all_transactions = await self._run_chunks(
                        provider, pages_b64, user_prompt, sys_prompt, bank
                    )
                else:  # images
                    logger.info(f"[Step 3/5] send_mode=images → 1 LLM call with {len(pages_b64)} images")
                    all_transactions = await self._run_images(
                        provider, pages_b64, user_prompt, sys_prompt, bank
                    )

            if not all_transactions:
                logger.warning(f"No transactions extracted from: {file_name}")
                return {"file_id": file_id, "status": "failed", "error": f"No transactions extracted from {file_name}"}

            # ---- 7. Generate CSV
            logger.info(f"[Step 4/5] Generating CSV for {len(all_transactions)} transactions…")
            
            # Map internal doc type to human-readable
            human_doc_type = "bank statement" if stmt_type == "bankstatement" else "credit card statement"
            extraction_date = datetime.now().strftime("%Y-%m-%d")
            
            csv_content = self._transactions_to_csv(
                all_transactions,
                bank=bank,
                doc_type=human_doc_type,
                extraction_date=extraction_date
            )
            logger.info(f"[Step 4/5] CSV ready: {len(csv_content)} chars, {len(all_transactions)} rows")

            # ---- 8. Upload to S3
            exec_id = job_id or "no-exec-id"
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            count = len(all_transactions)
            output_filename = f"{exec_id}-{ts}-{bank}-{stmt_type}-{llm_provider}-{send_mode}-{count}.csv"

            logger.info(f"[Step 5/5] Uploading to S3: {output_filename}")
            s3_service.upload_file(csv_content, output_filename)
            logger.info(f"✅ Done: {file_name} → {output_filename}")

            return {
                "file_id": file_id,
                "status": "success",
                "output_file": output_filename,
                "transactions_count": count,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "send_mode": send_mode,
            }

        except Exception as e:
            error_detail = str(e) or repr(e)
            logger.error(
                f"❌ Error processing {file_name} ({file_id}): "
                f"[{type(e).__name__}] {error_detail}",
                exc_info=True,
            )
            return {"file_id": file_id, "status": "failed", "error": error_detail}

    async def process_folder(
        self,
        folder_id: str,
        job_id: str = None,
        llm_provider: str = "google",
        llm_model: str = None,
        send_mode: str = "images",
    ):
        logger.info(
            f"=== Starting folder processing: {folder_id} (Job: {job_id}) "
            f"provider={llm_provider}, model={llm_model or 'default'}, mode={send_mode} ==="
        )
        if job_id:
            job_registry.update_job(job_id, status=JobStatus.PROCESSING)

        try:
            files = drive_service.list_files_in_folder(
                folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
            )
            logger.info(f"Found {len(files)} files to process")
            if job_id:
                job_registry.update_job(job_id, total_files=len(files))

            # ── Vertex AI Batch path ──────────────────────────────────────────
            # Use when: provider=google, mode=pdf, Vertex AI env vars are set.
            use_vertex_batch = (
                llm_provider == "google"
                and send_mode == "pdf"
                and bool(settings.VERTEX_PROJECT_ID)
                and bool(settings.GCS_BUCKET)
            )

            if use_vertex_batch:
                return await self._process_folder_vertex_batch(
                    files, job_id, llm_model
                )

            # ── Standard per-file path (all other cases) ──────────────────────
            results = []
            for i, file in enumerate(files, 1):
                logger.info(f"--- Processing file {i}/{len(files)}: {file.name} ---")
                res = await self.process_file(
                    file.id,
                    file.name,
                    job_id=job_id,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    send_mode=send_mode,
                )
                results.append(res)

                if job_id:
                    processed = sum(1 for r in results if r["status"] == "success")
                    failed = sum(1 for r in results if r["status"] == "failed")
                    job_registry.update_job(
                        job_id,
                        processed_files=processed,
                        failed_files=failed,
                        results=results,
                    )

            success = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "failed")
            logger.info(f"=== Folder done: {success} succeeded, {failed} failed ===")

            if job_id:
                job_registry.update_job(job_id, status=JobStatus.COMPLETED)

            return results

        except Exception as e:
            logger.error(f"Error in process_folder: {e}")
            if job_id:
                job_registry.update_job(job_id, status=JobStatus.FAILED, errors=[str(e)])
            raise

    async def _process_folder_vertex_batch(
        self,
        files: list,
        job_id: str,
        llm_model: str | None,
    ):
        """
        Download all files, build BatchFileEntry list, then hand off to
        vertex_batch_service which handles the full async lifecycle.
        """
        logger.info(
            f"[Job {job_id}] Using Vertex AI Batch path for {len(files)} files."
        )
        try:
            entries: list[BatchFileEntry] = []
            for i, file in enumerate(files, 1):
                logger.info(f"[Job {job_id}] Downloading {i}/{len(files)}: {file.name}")
                pdf_bytes = drive_service.download_file(file.id)

                bank = self.detect_bank(file.name)
                if bank == "unknown":
                    logger.warning(f"[Job {job_id}] Unknown bank for {file.name} — skipping.")
                    continue

                stmt_type = self.detect_type(file.name)
                user_prompt, sys_prompt = self._resolve_prompt(
                    bank, "google", stmt_type, "pdf"
                )

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
                job_registry.update_job(job_id, status=JobStatus.FAILED,
                                        errors=["No files with known bank detected."])
                return []

            # Delegate to the batch service — this blocks (polls) until done
            await vertex_batch_service.run_batch_job(entries, job_id)

        except Exception as exc:
            logger.error(f"[Job {job_id}] Vertex AI Batch failed: {exc}")
            job_registry.update_job(job_id, status=JobStatus.FAILED, errors=[str(exc)])
            raise


ocr_processor = OCRProcessor()
