# Gemini API Test Script (PowerShell)
# Replicates the exact request logic used in the app for:
# Provider: Google Gemini
# Model: gemini-3.1-pro-preview
# File: .\temp\PicPay_Fatura_122025.pdf
# Mode: PDF (Direct upload)

$apiKey = "AIzaSyBkWIGmUIWm2VTj5oRITDkiuUlSBzYoGiY"
if (-not $apiKey) {
    Write-Host "Error: GOOGLE_API_KEY environment variable not set." -ForegroundColor Red
    exit
}

$filePath = "PicPay_Fatura_122025.pdf"
if (-not (Test-Path $filePath)) {
    Write-Host "Error: File not found at $filePath" -ForegroundColor Red
    exit
}

Write-Host "Preparing Base64 encoding for $filePath..."
$base64File = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path $filePath)))

# --- SYSTEM PROMPT (from app/prompts/config.yaml) ---
$systemPrompt = @"
Extract bank transactions. Respond ONLY with a JSON object.
Minify output (no whitespace).
Format: {"transactions": [{"date": "DD-MM-YYYY", "description": "text", "installments": "", "amount": 1.23, "balance": 4.56, "category": "Outros"}]}
"@

# --- USER PROMPT (from app/prompts/picpay/google/ccstatement/pdf.txt) ---
$userPrompt = @"
You are a senior financial analyst specialized in PicPay Mastercard credit card statements. A PDF file of a PicPay credit card statement is attached. Extract, normalize, and reconcile ALL transactions from the entire document.

METADATA
Extract from the statement:
- Closing date from line Fechamento: → use this year for all transactions
- Target total from line Total geral dos lançamentos → reconciliation target
- Ignore any line containing PAGAMENTO DE FATURA

PicPay statements are formatted with TWO COLUMNS per page. Each column contains independent transaction blocks. You MUST:
- Process the LEFT column first, top to bottom, extracting all its transactions.
- Then process the RIGHT column, top to bottom, extracting all its transactions.
- Never mix or interleave lines from different columns.
- Be aware that a transaction section may START in one column and CONTINUE in the next column or on the next page.

TRANSACTION SECTIONS
Transactions exist ONLY inside sections headed by Picpay Card final XXXX → Transações Nacionais.
TRANSACTION FORMAT: DD/MM DESCRIPTION VALUE

date = DD/MM → DD-MM-YYYY (use closing year)
description = text between date and value
amount = last numeric value (negative for purchases, positive for CASHBACK/ESTORNO/CREDITO/PAGAMENTO RECEBIDO)

INSTALLMENTS: Pattern PARCXX/YY → installments: "[XX/YY]", remove from description.
CATEGORIES: RESTAURANTE→Restaurante, MERCADO→Mercado, DROGARIA→Saúde, POSTO→Combustivel, AMAZON/SHOPPIN→Compras, ESTACIONAMENTO→Transporte, LATAM/GOL/HOTEL→Viagem, SEGUROS→Seguros, CASHBACK→Crédito, default→Outros

RECONCILIATION: Compute NET_TOTAL = sum(abs(negatives)) - sum(positives). Compare to target. If mismatch, append: ||RECONCILIATION_DIFFERENCE|[difference]||Financeiro

OUTPUT: Return ONLY a valid JSON object. Minify.
{
  "transactions": [
    {"date": "DD-MM-YYYY", "description": "text", "installments": "[XX/YY]", "amount": -123.45, "balance": 0, "category": "category name"}
  ]
}
Rules: (1) amounts/balance must be NUMBERS with dot decimal. (2) DD-MM-YYYY format. (3) No markdown. (4) ONLY the JSON object.
"@

# --- API BODY ---
# Using v1beta for gemini-3.1-pro-preview support
# Note: Using inlineData to simulate the payload directness
$body = @{
    system_instruction = @{
        parts = @( @{ text = $systemPrompt } )
    }
    contents = @(
        @{
            parts = @(
                @{ text = $userPrompt },
                @{
                    inlineData = @{
                        mimeType = "application/pdf"
                        data = $base64File
                    }
                }
            )
        }
    )
    generationConfig = @{
        responseMimeType = "application/json"
        temperature = 0
        maxOutputTokens = 50000
    }
} | ConvertTo-Json -Depth 10

$tempFile = "gemini_request.json"
$body | Set-Content -Path $tempFile -Encoding utf8

Write-Host "Sending request to Gemini API (reading from temp file to avoid 'extension too long' error)..."
curl.exe -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key=$apiKey" `
  -H "Content-Type: application/json" `
  -d "@$tempFile"

Remove-Item $tempFile
