import json
import pandas as pd
import io
import asyncio
from datetime import datetime
from pathlib import Path
from app.services.google_drive import drive_service
from app.services.llm_service import llm_service
from app.services.s3_service import s3_service
from app.utils.logger import logger
from app.utils.config import settings
import yaml
import pdfplumber

class OCRProcessor:
    def __init__(self):
        # Determine base directory for prompts
        self.base_dir = Path(__file__).resolve().parent.parent
        self.prompts_dir = self.base_dir / "prompts"
        self.prompts = self.load_prompts()
    
    def load_prompts(self):
        try:
            # Respect the configured path if it's absolute, otherwise use the project's prompts directory
            config_path = Path(settings.PROMPT_CONFIG_PATH)
            if not config_path.is_absolute():
                config_path = self.prompts_dir / "config.yaml"
                
            logger.info(f"Loading prompt config from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load prompt config: {e}")
            return {"banks": {}}

    def detect_bank(self, filename: str) -> str:
        # Simple heuristic based on filename or could be more complex
        filename = filename.lower()
        if "picpay" in filename:
            return "picpay"
        elif "itau" in filename:
            return "itau"
        elif "xp" in filename:
            return "xp"
        else:
            return "unknown"

    async def process_file(self, file_id: str, file_name: str) -> dict:
        try:
            logger.info(f"[Step 1/5] Downloading file: {file_name} ({file_id})")
            
            # 1. Download
            content_bytes = drive_service.download_file(file_id)
            logger.info(f"[Step 1/5] Download complete: {len(content_bytes)} bytes")
            
            # 2. Extract text from PDF using pdfplumber
            pdf_stream = io.BytesIO(content_bytes)
            text_pages = []
            with pdfplumber.open(pdf_stream) as pdf:
                logger.info(f"[Step 1/5] PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                        logger.debug(f"Page {i+1}: {len(page_text)} chars extracted")
                    else:
                        logger.warning(f"Page {i+1}: no text extracted (may be scanned/image)")
            
            text_content = "\n\n".join(text_pages)
            logger.info(f"[Step 1/5] Extracted {len(text_content)} chars from {len(text_pages)} pages")
            
            if not text_content.strip():
                raise Exception(f"No text could be extracted from {file_name}. The PDF may be scanned/image-based and require OCR.") 

            bank = self.detect_bank(file_name)
            logger.info(f"[Step 2/5] Detected bank: {bank} for file: {file_name}")
            if bank == "unknown":
                raise Exception(f"Could not detect bank for file: {file_name}")

            # 2. Extract with LLM
            prompt_file_rel = self.prompts.get('banks', {}).get(bank, {}).get('prompt_file')
            prompt_template_content = ""
            
            if prompt_file_rel:
                # Use prompts_dir for resolving the individual bank prompt files
                prompt_path = self.prompts_dir / prompt_file_rel
                logger.info(f"[Step 3/5] Loading prompt template from: {prompt_path}")
                if prompt_path.exists():
                    with open(prompt_path, 'r') as f:
                         prompt_template_content = f.read()
                    logger.info(f"[Step 3/5] Prompt template loaded ({len(prompt_template_content)} chars)")
                else:
                    logger.warning(f"[Step 3/5] Prompt file not found: {prompt_path}, using fallback")
            
            if not prompt_template_content:
                 # Fallback
                 prompt_template_content = "{{DOCUMENT_TEXT}}"
                 logger.info("[Step 3/5] Using fallback prompt template")

            logger.info(f"[Step 3/5] Calling LLM for transaction extraction...")
            llm_response = llm_service.extract_transactions(text_content, bank, prompt_template_content)
            data = json.loads(llm_response)
            
            # 3. Process & Categorize
            transactions = data.get("transactions", [])
            logger.info(f"[Step 4/5] LLM returned {len(transactions)} transactions, categorizing...")
            processed_txs = []
            for tx in transactions:
                # Add categorization
                tx['category'] = llm_service.categorize_transaction(tx['description'])
                processed_txs.append(tx)

            # 4. Generate CSV
            df = pd.DataFrame(processed_txs)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, sep='|')
            csv_content = csv_buffer.getvalue()
            logger.info(f"[Step 4/5] CSV generated ({len(csv_content)} chars, {len(processed_txs)} rows)")

            # 5. Upload to S3
            output_filename = f"processed_{datetime.now().strftime('%Y%m%d')}_{file_name}.csv"
            logger.info(f"[Step 5/5] Uploading to S3: {output_filename}")
            s3_service.upload_file(csv_content, output_filename)
            
            logger.info(f"✅ File processed successfully: {file_name} -> {output_filename}")
            return {"file_id": file_id, "status": "success", "output_file": output_filename}

        except Exception as e:
            logger.error(f"❌ Error processing file {file_name} ({file_id}): {str(e)}")
            return {"file_id": file_id, "status": "failed", "error": str(e)}

    async def process_folder(self, folder_id: str):
        logger.info(f"=== Starting folder processing: {folder_id} ===")
        files = drive_service.list_files_in_folder(folder_id or settings.GOOGLE_DRIVE_FOLDER_ID)
        logger.info(f"Found {len(files)} files to process")
        results = []
        for i, file in enumerate(files, 1):
            logger.info(f"--- Processing file {i}/{len(files)}: {file.name} ---")
            res = await self.process_file(file.id, file.name)
            results.append(res)
        
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        logger.info(f"=== Folder processing complete: {success} succeeded, {failed} failed ===")
        return results

ocr_processor = OCRProcessor()

