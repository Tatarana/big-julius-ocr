from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.logger import logger
import json

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.client = None
        
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=600.0  # Extremely high timeout (10 mins) for massive vision tasks
            )
        # Add other providers here

    async def extract_transactions(self, text_content: str, bank_name: str, prompt_template: str, system_prompt: str = None) -> str:
        """
        Sends text to LLM to extract transactions.
        """
        prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", text_content)
        return await self._call_llm(prompt, bank_name, system_prompt)

    async def extract_transactions_from_images(self, base64_images: list[str], bank_name: str, prompt_template: str, system_prompt: str = None) -> str:
        """
        Sends images to LLM (Vision) to extract transactions.
        """
        # Remove placeholder if it exists in the template when sending images
        prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", "[See attached images for document content]")
        
        content = [{"type": "text", "text": prompt}]
        for b64 in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"  # Crucial for reading small text/numbers
                }
            })
            
        return await self._call_llm(content, bank_name, system_prompt)

    async def _call_llm(self, user_content, bank_name: str, system_prompt: str = None) -> str:
        if not self.client:
            raise Exception("LLM Client not initialized")

        try:
            sys_prompt = system_prompt or "You are a helpful banking assistant that extracts transaction data from statements. Always respond in valid JSON format."
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Create a structured log representation (without exhausting disk space with images)
            log_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if isinstance(content, list):
                    log_content = []
                    for item in content:
                        if item["type"] == "text":
                            log_content.append(item)
                        elif item["type"] == "image_url":
                            # Truncate image URL for logging
                            url = item["image_url"]["url"]
                            truncated_url = f"{url[:50]}...[TRUNCATED {len(url)} chars]...{url[-10:]}"
                            log_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": truncated_url,
                                    "detail": item["image_url"].get("detail")
                                }
                            })
                    log_messages.append({"role": role, "content": log_content})
                else:
                    log_messages.append(msg)

            full_payload_log = {
                "model": self.model,
                "messages": log_messages,
                "temperature": 0,
                "max_tokens": 16384,
                "response_format": {"type": "json_object"}
            }
            
            logger.debug(f"LLM REQUEST PAYLOAD:\n{json.dumps(full_payload_log, indent=2, ensure_ascii=False)}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=16384,  # Increased to maximum for large statements
                response_format={"type": "json_object"}
            )
            logger.info(f"LLM response received for bank={bank_name}")
            result = response.choices[0].message.content
            
            if response.choices[0].finish_reason == "length":
                logger.warning(f"⚠️ LLM response was truncated due to length (max_tokens reached). Attempting to repair JSON.")
                result = self._repair_json(result)
            
            logger.debug(f"LLM RESPONSE BODY: {result}")
            logger.info(f"LLM usage: {response.usage}")
            return result
        except Exception as e:
            logger.error(f"LLM call failed for bank {bank_name}: {str(e)}")
            raise

    def _repair_json(self, broken_json: str) -> str:
        """
        Very basic repair for truncated JSON lists. 
        Tries to find the last complete object and close the JSON.
        """
        logger.debug(f"Original broken JSON snippet: {broken_json[-50:]}")
        
        # If it's a list of objects, try to find the last complete one
        # Pattern: ... }, { "incomplete": "..."
        last_comma = broken_json.rfind('},')
        if last_comma != -1:
            repaired = broken_json[:last_comma + 1] + "]}"
            try:
                json.loads(repaired) # Verify
                logger.warning("⚠️ REPAIR LOG: Successfully repaired truncated JSON by closing list after the last complete object. NOTE: The extraction is PARTIAL and some transactions were lost.")
                return repaired
            except:
                pass
        
        # Second attempt: just add closing brackets and see if it works
        for suffix in ["]}", "}", " ] } "]:
            try:
                repaired = broken_json + suffix
                json.loads(repaired)
                logger.warning(f"⚠️ REPAIR LOG: Successfully repaired truncated JSON by appending '{suffix}'. NOTE: The extraction is PARTIAL and some transactions were lost.")
                return repaired
            except:
                continue
                
        logger.error("❌ Failed to repair truncated JSON. Returning original string.")
        return broken_json

    def categorize_transaction(self, description: str) -> str:
        # Simple categorization logic or another LLM call
        # For cost efficiency, maybe just regex or a smaller model
        return "Uncategorized"

    async def check_connection(self) -> bool:
        if not self.client:
            return False
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"LLM connection check failed: {str(e)}")
            return False

llm_service = LLMService()
