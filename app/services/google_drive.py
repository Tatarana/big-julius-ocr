import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from app.utils.config import settings
from app.utils.logger import logger
from app.api.models import FileMetadata

class GoogleDriveService:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    def __init__(self):
        self.creds = None
        if os.path.exists(settings.GOOGLE_CREDENTIALS_PATH):
            self.creds = service_account.Credentials.from_service_account_file(
                settings.GOOGLE_CREDENTIALS_PATH, scopes=self.SCOPES)
        else:
            logger.warning(f"Google Credentials file not found at {settings.GOOGLE_CREDENTIALS_PATH}")

    def get_service(self):
        if not self.creds:
            raise Exception("Google Drive credentials not initialized")
        return build('drive', 'v3', credentials=self.creds)

    def list_files_in_folder(self, folder_id: str) -> list[FileMetadata]:
        try:
            service = self.get_service()
            query = f"'{folder_id}' in parents and trashed = false and mimeType = 'application/pdf'"
            results = service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, size, createdTime)"
            ).execute()
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")
            
            return [
                FileMetadata(
                    id=f['id'],
                    name=f['name'],
                    size=f.get('size', '0'),
                    created_time=f['createdTime'],
                    status="pending"
                ) for f in files
            ]
        except Exception as e:
            logger.error(f"Error listing files from Drive: {str(e)}")
            raise

    def download_file(self, file_id: str) -> bytes:
        try:
            service = self.get_service()
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            content = file_content.getvalue()
            logger.info(f"Downloaded file {file_id} ({len(content)} bytes)")
            return content
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            raise
            
    def check_connection(self) -> bool:
        try:
            self.get_service().files().list(pageSize=1).execute()
            return True
        except Exception as e:
            logger.error(f"Google Drive connection check failed: {str(e)}")
            return False

drive_service = GoogleDriveService()
