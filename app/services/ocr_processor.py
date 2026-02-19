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
from app.services.llm_service import llm_service
from app.services.s3_service import s3_service
from app.utils.logger import logger
from app.utils.config import settings

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

    def detect_type(self, filename: str) -> str:
        filename = filename.lower()
        # "fatura" is credit card statement in Portuguese
        if "fatura" in filename or "cc" in filename or "credit" in filename:
            return "ccstatement"
        return "bankstatement"

    async def process_file(self, file_id: str, file_name: str, job_id: str = None) -> dict:
        try:
            logger.info(f"[Step 1/5] Downloading file: {file_name} ({file_id})")
            content_bytes = drive_service.download_file(file_id)
            logger.info(f"[Step 1/5] Download complete: {len(content_bytes)} bytes")
            
            bank = self.detect_bank(file_name)
            logger.info(f"[Step 2/5] Detected bank: {bank} for file: {file_name}")
            if bank == "unknown":
                raise Exception(f"Could not detect bank for file: {file_name}")

            # 2. Convert PDF to images and process in chunks
            all_transactions = []
            
            # Load prompt template once
            prompt_file_rel = self.prompts.get('banks', {}).get(bank, {}).get('prompt_file')
            prompt_template_content = "{{DOCUMENT_TEXT}}"
            if prompt_file_rel:
                prompt_path = self.prompts_dir / prompt_file_rel
                if prompt_path.exists():
                    with open(prompt_path, 'r') as f:
                        prompt_template_content = f.read()
            
            system_prompt = self.prompts.get('default_system_prompt')

            # Open PDF with PyMuPDF
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            num_pages = len(doc)
            
            # Determine which pages to process
            page_indices = list(range(num_pages))
            if bank == 'picpay':
                # Remove first two (0, 1) and the last page (num_pages - 1)
                # Ensure we have enough pages to perform the exclusion
                if num_pages > 3:
                    page_indices = list(range(2, num_pages - 1))
                    logger.info(f"PicPay detected: Skipping first 2 and last page. Processing {len(page_indices)} pages (Indices: {page_indices})")
                else:
                    page_indices = []
                    logger.warning(f"PicPay PDF has only {num_pages} pages. Skipping all pages based on exclusion rule (first 2 and last).")
            
            logger.info(f"[Step 3/5] Processing {len(page_indices)} total pages in a single call...")
            
            base64_images = []
            for page_num in page_indices:
                page = doc.load_page(page_num)
                # Use higher DPI for better OCR (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("jpg")
                
                # Debug Mode: Save images to disk
                if settings.DEBUG_SAVE_IMAGES:
                    project_root = Path(settings.PROMPT_CONFIG_PATH).resolve().parent.parent
                    temp_dir = project_root / "temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    img_path = temp_dir / f"{job_id or 'debug'}_{file_name}_p{page_num}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    logger.info(f"DEBUG: Saved page image to {img_path}")

                base64_images.append(base64.b64encode(img_data).decode('utf-8'))
            
            if base64_images:
                logger.info(f"Calling LLM (Vision) for all {len(base64_images)} pages...")
                llm_response = await llm_service.extract_transactions_from_images(base64_images, bank, prompt_template_content, system_prompt)
                
                try:
                    data = json.loads(llm_response)
                    # Find transactions list in data
                    txs = data.get("transactions")
                    if txs is None:
                        for val in data.values():
                            if isinstance(val, list):
                                txs = val
                                break
                    
                    if txs:
                        logger.info(f"Extracted {len(txs)} transactions from LLM response")
                        all_transactions.extend(txs)
                    else:
                        logger.warning(f"No transactions found in LLM response")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response: {e}")
            
            doc.close()            
            if not all_transactions:
                logger.warning(f"No transactions extracted from the entire file: {file_name}")
            
            # 3. Process & Categorize
            logger.info(f"[Step 4/5] Processing {len(all_transactions)} total transactions...")
            processed_txs = []
            for tx in all_transactions:
                # Add categorization if not already present or "unknown"
                desc = tx.get('description', tx.get('Description', ''))
                current_cat = tx.get('category', tx.get('Category', 'unknown'))
                
                if not current_cat or current_cat.lower() == 'unknown':
                    tx['category'] = llm_service.categorize_transaction(desc)
                else:
                    tx['category'] = current_cat
                
                processed_txs.append(tx)

            # 4. Generate CSV
            cols = ['date', 'description', 'installments', 'amount', 'balance', 'category']
            # Normalize to ensure all columns exist and are lowercase
            normalized_txs = []
            for tx in processed_txs:
                norm_tx = {}
                for col in cols:
                    # Try lowercase and Capitalized versions
                    val = tx.get(col, tx.get(col.capitalize(), ""))
                    
                    # Ensure amount and balance use comma as decimal separator
                    if col in ['amount', 'balance'] and val is not None and val != "":
                        # Convert to string and replace dot with comma
                        val = str(val).replace('.', ',')
                    
                    norm_tx[col] = val
                normalized_txs.append(norm_tx)
            
            df = pd.DataFrame(normalized_txs)[cols]
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, sep='|')
            csv_content = csv_buffer.getvalue()
            logger.info(f"[Step 4/5] CSV generated ({len(csv_content)} chars, {len(processed_txs)} rows)")

            # 5. Upload to S3
            # Pattern: <exec_id>-<timestamp>-<bank>-<type>-<sum_rec>.csv
            exec_id = job_id or "no-exec-id"
            ts = datetime.now().strftime('%Y%m%d%H%M%S')
            stmt_type = self.detect_type(file_name)
            count = len(processed_txs)
            
            output_filename = f"{exec_id}-{ts}-{bank}-{stmt_type}-{count}.csv"
            
            logger.info(f"[Step 5/5] Uploading to S3: {output_filename}")
            s3_service.upload_file(csv_content, output_filename)
            
            logger.info(f"✅ File processed successfully: {file_name} -> {output_filename}")
            return {"file_id": file_id, "status": "success", "output_file": output_filename, "transactions_count": count}

        except Exception as e:
            logger.error(f"❌ Error processing file {file_name} ({file_id}): {str(e)}")
            return {"file_id": file_id, "status": "failed", "error": str(e)}

    async def process_folder(self, folder_id: str, job_id: str = None):
        logger.info(f"=== Starting folder processing: {folder_id} (Job: {job_id}) ===")
        files = drive_service.list_files_in_folder(folder_id or settings.GOOGLE_DRIVE_FOLDER_ID)
        logger.info(f"Found {len(files)} files to process")
        results = []
        for i, file in enumerate(files, 1):
            logger.info(f"--- Processing file {i}/{len(files)}: {file.name} ---")
            res = await self.process_file(file.id, file.name, job_id)
            results.append(res)
        
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        logger.info(f"=== Folder processing complete: {success} succeeded, {failed} failed ===")
        return results

ocr_processor = OCRProcessor()

