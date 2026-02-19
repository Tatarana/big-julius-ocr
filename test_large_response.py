
import asyncio
import json
import pandas as pd
import io
from app.services.ocr_processor import ocr_processor
from unittest.mock import MagicMock, patch

async def test_large_csv_generation():
    from app.utils.config import settings
    print(f"DEBUG_SAVE_IMAGES: {settings.DEBUG_SAVE_IMAGES}")
    print("Testing CSV generation with 120 transactions...")
    
    # 1. Create 120 mock transactions
    mock_transactions = []
    for i in range(1, 121):
        mock_transactions.append({
            "date": f"{i%28+1:02d}-12-2025",
            "description": f"Transaction Number {i}",
            "amount": -10.50 - i,
            "category": "Testing"
        })
    
    llm_payload = json.dumps({"transactions": mock_transactions})
    
    # 2. Mock the LLM service to return this payload
    with patch('app.services.llm_service.llm_service.extract_transactions_from_images') as mock_llm:
        mock_llm.return_value = llm_payload
        
        # We also need to mock drive_service.download_file and fitz.open
        # since we want to test process_file
        with patch('app.services.google_drive.drive_service.download_file') as mock_download, \
             patch('fitz.open') as mock_fitz, \
             patch('app.services.s3_service.s3_service.upload_file') as mock_s3:
            
            mock_download.return_value = b"dummy pdf content"
            
            # Mock PDF doc
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 10
            mock_page = MagicMock()
            mock_page.get_pixmap.return_value.tobytes.return_value = b"img"
            mock_doc.load_page.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            # Run processing
            result = await ocr_processor.process_file("file123", "PicPay_Test.pdf", "job123")
            
            print(f"Result status: {result['status']}")
            if result['status'] == 'success':
                print(f"Transactions count: {result['transactions_count']}")
                print(f"Output file: {result['output_file']}")
                
                # Check the CSV count from the mock_s3 call
                csv_content = mock_s3.call_args[0][0]
                lines = csv_content.strip().split('\n')
                # 120 transactions + 1 header = 121 lines
                print(f"CSV line count (including header): {len(lines)}")
                
                if len(lines) == 121:
                    print("✅ SUCCESS: CSV contains all 120 registries plus header.")
                else:
                    print(f"❌ FAILURE: CSV contains {len(lines)-1} registries, expected 120.")
            else:
                print(f"❌ FAILURE: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_large_csv_generation())
