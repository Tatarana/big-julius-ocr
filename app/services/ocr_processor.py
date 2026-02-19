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
            content_bytes = drive_service.download_file(file_id)
            logger.info(f"[Step 1/5] Download complete: {len(content_bytes)} bytes")
            
            bank = self.detect_bank(file_name)
            logger.info(f"[Step 2/5] Detected bank: {bank} for file: {file_name}")
            if bank == "unknown":
                raise Exception(f"Could not detect bank for file: {file_name}")

            # 2. Extract text and process in chunks
            pdf_stream = io.BytesIO(content_bytes)
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

            with pdfplumber.open(pdf_stream) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"[Step 3/5] Processing {num_pages} pages in chunks...")
                
                # Process in chunks of 2 pages to stay well within token limits
                chunk_size = 2
                for i in range(0, num_pages, chunk_size):
                    chunk_pages = pdf.pages[i:i+chunk_size]
                    logger.info(f"--- Processing pages {i+1} to {min(i+chunk_size, num_pages)} ---")
                    
                    chunk_text = ""
                    for page in chunk_pages:
                        page_text = page.extract_text()
                        if page_text:
                            chunk_text += page_text + "\n\n"
                    
                    if not chunk_text.strip():
                        logger.warning(f"No text extracted from pages {i+1}-{i+chunk_size}")
                        continue
                    
                    logger.info(f"Calling LLM for chunk {i//chunk_size + 1}...")
                    llm_response = llm_service.extract_transactions(chunk_text, bank, prompt_template_content, system_prompt)
                    
                    try:
                        data = json.loads(llm_response)
                        # Find transactions list in data
                        chunk_txs = data.get("transactions")
                        if chunk_txs is None:
                            for val in data.values():
                                if isinstance(val, list):
                                    chunk_txs = val
                                    break
                        
                        if chunk_txs:
                            logger.info(f"Extracted {len(chunk_txs)} transactions from chunk")
                            all_transactions.extend(chunk_txs)
                        else:
                            logger.warning(f"No transactions found in LLM response for chunk {i//chunk_size + 1}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response for chunk: {e}")
                        # Could try to fix the JSON here or just continue
            
            if not all_transactions:
                logger.warning(f"No transactions extracted from the entire file: {file_name}")
            
            # 3. Process & Categorize
            logger.info(f"[Step 4/5] Processing {len(all_transactions)} total transactions...")
            processed_txs = []
            for tx in all_transactions:
                # Add categorization
                desc = tx.get('description', '')
                tx['category'] = llm_service.categorize_transaction(desc)
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
            return {"file_id": file_id, "status": "success", "output_file": output_filename, "transactions_count": len(all_transactions)}

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

