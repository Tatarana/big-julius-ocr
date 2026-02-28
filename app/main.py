from fastapi import FastAPI
from app.utils.config import settings
from app.utils.logger import logger
from app.api import endpoints

app = FastAPI(
    title="Big Julius OCR",
    description="Bank Statement OCR Microservice",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Big Julius OCR Service")
    logger.info(f"Configuration loaded: Env={settings.LOG_LEVEL}")

@app.get("/")
async def root():
    return {"message": "Big Julius OCR Service is running"}

@app.get("/health")
async def health_check():
    from datetime import datetime, timezone
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Include routers
app.include_router(endpoints.router)
