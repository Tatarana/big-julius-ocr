from openai import OpenAI
from app.utils.config import settings
from app.utils.logger import logger
import json

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.client = None
        
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # Add other providers here

    def extract_transactions(self, text_content: str, bank_name: str, prompt_template: str, system_prompt: str = None) -> str:
        """
        Sends text to LLM to extract transactions.
        Returns raw JSON string response.
        """
        if not self.client:
            raise Exception("LLM Client not initialized")

        try:
            full_prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", text_content)
            
            sys_prompt = system_prompt or "You are a helpful banking assistant that extracts transaction data from statements. Always respond in valid JSON format."
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_prompt}
            ]
            logger.info(f"Sending LLM request for bank={bank_name} (model={self.model}, prompt length={len(full_prompt)} chars)")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content
            logger.debug(f"LLM RESPONSE BODY: {result}")
            logger.info(f"LLM usage: {response.usage}")
            return result
        except Exception as e:
            logger.error(f"LLM extraction failed for bank {bank_name}: {str(e)}")
            raise

    def categorize_transaction(self, description: str) -> str:
        # Simple categorization logic or another LLM call
        # For cost efficiency, maybe just regex or a smaller model
        return "Uncategorized"

    def check_connection(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"LLM connection check failed: {str(e)}")
            return False

llm_service = LLMService()
