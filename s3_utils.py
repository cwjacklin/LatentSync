import boto3
import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID", os.environ.get("S3_ACCESS_KEY", "minioadmin")),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY", os.environ.get("S3_SECRET_KEY", "YourStrongPasswordHere")),
        config=boto3.session.Config(signature_version='s3v4')
    )

def download_from_s3(s3_key: str, local_path: str, bucket: Optional[str] = None):
    s3 = get_s3_client()
    env_bucket = os.environ.get("S3_BUCKET")
    if not bucket and not env_bucket:
        raise ValueError("S3 bucket must be provided via parameter or S3_BUCKET environment variable")
    bucket = bucket or env_bucket
    logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
    s3.download_file(bucket, s3_key, local_path)

def upload_to_s3(local_path: str, s3_key: str, bucket: Optional[str] = None):
    s3 = get_s3_client()
    env_bucket = os.environ.get("S3_BUCKET")
    if not bucket and not env_bucket:
        raise ValueError("S3 bucket must be provided via parameter or S3_BUCKET environment variable")
    bucket = bucket or env_bucket
    logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)

def notify_webhook(webhook_url: str, payload: dict):
    secret = os.environ.get("APP_WEBHOOK_SECRET")
    if not secret:
        logger.error("APP_WEBHOOK_SECRET is not set. Cannot notify webhook.")
        return
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret}"
    }
    try:
        response = requests.post(webhook_url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Webhook notified successfully: {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to notify webhook: {e}")
