import boto3
from botocore.exceptions import ClientError
from app.utils.config import settings
from app.utils.logger import logger
import io

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_S3_BUCKET

    def upload_file(self, file_content: str, file_name: str):
        """
        Uploads string content (CSV) to S3
        """
        try:
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                file_name,
                ExtraArgs={'ContentType': 'text/csv'}
            )
            logger.info(f"Successfully uploaded {file_name} to S3")
            return True
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise

    def check_connection(self) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error(f"S3 connection check failed: {str(e)}")
            return False

s3_service = S3Service()
