
import json
import base64
import os
from pathlib import Path

def generate_gpt_request():
    temp_dir = Path("e:/dev/big-julius-ocr/app/temp")
    # Get the latest set of images (using the job ID prefix)
    images = sorted([f for f in temp_dir.glob("*.jpg") if "PicPay_Fatura" in f.name])
    
    if not images:
        print("No real images found in temp folder.")
        return

    # Load prompts
    # 1. System Prompt
    system_prompt = "Extract bank transactions. Respond ONLY with a JSON object.\nMinify output (no whitespace).\nFormat:\n{\"transactions\": [{\"date\": \"DD-MM-YYYY\", \"description\": \"text\", \"amount\": 1.23, \"balance\": 4.56}]}\n"
    
    # 2. User Prompt
    prompt_path = "e:/dev/big-julius-ocr/app/prompts/picpay/extraction_prompt.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        user_text = f.read().replace("{{DOCUMENT_TEXT}}", "[See attached images for document content]")

    # Build messages
    content = [{"type": "text", "text": user_text}]
    for img_path in images:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}",
                    "detail": "high"
                }
            })

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"}
    }

    with open("gpt_request_payload.json", "w", encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    
    print(f"✅ Generated gpt_request_payload.json with {len(images)} images.")
    print("\nYou can now run the following curl command:")
    print("curl https://api.openai.com/v1/chat/completions \\")
    print("  -H \"Content-Type: application/json\" \\")
    print("  -H \"Authorization: Bearer YOUR_OPENAI_API_KEY\" \\")
    print("  -d @gpt_request_payload.json")

if __name__ == "__main__":
    generate_gpt_request()
