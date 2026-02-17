# Big Julius OCR

Bank Statement OCR Microservice that reads files from Google Drive, processes them using LLM APIs to extract transactions, and saves structured CSVs to AWS S3.

## Features

- **Google Drive Integration**: Automatically reads PDF/Image statements.
- **LLM extraction**: Uses OpenAI GPT-4 (or others) to parse transactions.
- **AWS S3 Output**: Stores processed CSV files.
- **Bank Support**: PicPay, Itaú, XP Investimentos.
- **REST API**: FastAPI endpoints for control and monitoring.

## Setup

1.  **Clone the repository**
2.  **Create `.env` file**: Copy `.env.example` to `.env` and fill in your credentials.
    ```bash
    cp .env.example .env
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Valid pandas installation required.*

4.  **Run with Docker**:
    ```bash
    docker-compose up --build
    ```

## API Endpoints

- `POST /process-files`: Trigger processing of files in Drive folder.
- `GET /list-input-files`: List files available for processing.
- `GET /health`: Service health check.
- `GET /check-all-connections`: Verify external service connectivity.
- `GET /show-config`: View current configuration.

## Development

Run locally:
```bash
uvicorn app.main:app --reload
```

Run tests:
```bash
pytest tests/
```

## Configuration

Prompts are located in `app/prompts/config.yaml` and subdirectories.
To add a new bank, create a new prompt file and update the config.
