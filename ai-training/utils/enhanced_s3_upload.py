"""
Enhanced S3 Upload System for GPU Training
Provides efficient, reliable S3 uploads with progress tracking and retry logic
"""

import boto3
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError, NoCredentialsError
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedS3Uploader:
    """Enhanced S3 uploader with progress tracking and retry logic"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=region)
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 client initialized for bucket: {bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Failed to access S3 bucket {bucket_name}: {e}")
            raise
    
    def upload_file(self, 
                   file_path: str, 
                   s3_key: str, 
                   content_type: str = None,
                   metadata: Dict[str, str] = None,
                   max_retries: int = 3) -> Tuple[bool, str]:
        """
        Upload a single file to S3 with retry logic
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            content_type: MIME type
            metadata: Additional metadata
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (success, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Uploading {file_path} ({file_size} bytes) to s3://{self.bucket_name}/{s3_key}")
        
        # Prepare upload parameters
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        if metadata:
            extra_args['Metadata'] = metadata
        
        # Add file hash for integrity checking
        file_hash = self._calculate_file_hash(file_path)
        extra_args['Metadata'] = extra_args.get('Metadata', {})
        extra_args['Metadata']['file-hash'] = file_hash
        
        # Upload with retry logic
        for attempt in range(max_retries + 1):
            try:
                if file_size > 100 * 1024 * 1024:  # 100MB
                    # Use multipart upload for large files
                    self._multipart_upload(file_path, s3_key, extra_args)
                else:
                    # Use simple upload for smaller files
                    self.s3_client.upload_file(file_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
                
                # Verify upload
                if self._verify_upload(s3_key, file_hash):
                    logger.info(f"Successfully uploaded {s3_key}")
                    return True, None
                else:
                    logger.warning(f"Upload verification failed for {s3_key}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return False, "Upload verification failed"
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.warning(f"Upload attempt {attempt + 1} failed: {error_code} - {e}")
                
                if attempt < max_retries:
                    if error_code in ['NoSuchBucket', 'AccessDenied']:
                        # Don't retry for these errors
                        return False, f"S3 error: {error_code}"
                    
                    # Wait before retry
                    time.sleep(2 ** attempt)
                else:
                    return False, f"S3 error: {error_code} - {e}"
            
            except Exception as e:
                logger.error(f"Unexpected error during upload: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    return False, f"Unexpected error: {e}"
        
        return False, "Max retries exceeded"
    
    def upload_multiple_files(self, 
                            file_uploads: List[Dict[str, Any]], 
                            max_workers: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Upload multiple files concurrently
        
        Args:
            file_uploads: List of upload dictionaries with keys:
                - file_path: Local file path
                - s3_key: S3 object key
                - content_type: MIME type (optional)
                - metadata: Additional metadata (optional)
            max_workers: Maximum concurrent uploads
            
        Returns:
            Dictionary mapping s3_key to result dictionary
        """
        results = {}
        
        logger.info(f"Starting concurrent upload of {len(file_uploads)} files with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_key = {}
            for upload_info in file_uploads:
                file_path = upload_info['file_path']
                s3_key = upload_info['s3_key']
                content_type = upload_info.get('content_type')
                metadata = upload_info.get('metadata')
                
                future = executor.submit(
                    self.upload_file,
                    file_path,
                    s3_key,
                    content_type,
                    metadata
                )
                future_to_key[future] = s3_key
            
            # Process completed uploads
            for future in as_completed(future_to_key):
                s3_key = future_to_key[future]
                try:
                    success, error = future.result()
                    results[s3_key] = {
                        'success': success,
                        'error': error,
                        'timestamp': time.time()
                    }
                except Exception as e:
                    results[s3_key] = {
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    }
        
        # Log summary
        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        logger.info(f"Upload completed: {successful} successful, {failed} failed")
        
        return results
    
    def upload_training_data(self, 
                           training_data: Dict[str, Any], 
                           job_id: str) -> Tuple[bool, str]:
        """
        Upload training data to S3
        
        Args:
            training_data: Training data dictionary
            job_id: Job identifier
            
        Returns:
            Tuple of (success, s3_key)
        """
        try:
            # Serialize training data
            data_json = json.dumps(training_data, indent=2)
            data_bytes = data_json.encode('utf-8')
            
            # Create temporary file
            temp_file = f"/tmp/training_data_{job_id}.json"
            with open(temp_file, 'w') as f:
                f.write(data_json)
            
            # Upload to S3
            s3_key = f"training_data/{job_id}.json"
            success, error = self.upload_file(
                temp_file,
                s3_key,
                content_type='application/json',
                metadata={
                    'job_id': job_id,
                    'upload_type': 'training_data',
                    'timestamp': str(time.time())
                }
            )
            
            # Cleanup temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if success:
                logger.info(f"Training data uploaded to s3://{self.bucket_name}/{s3_key}")
                return True, s3_key
            else:
                logger.error(f"Failed to upload training data: {error}")
                return False, error
                
        except Exception as e:
            logger.error(f"Error uploading training data: {e}")
            return False, str(e)
    
    def upload_training_script(self, 
                             script_path: str, 
                             job_id: str) -> Tuple[bool, str]:
        """
        Upload training script to S3
        
        Args:
            script_path: Path to training script
            job_id: Job identifier
            
        Returns:
            Tuple of (success, s3_key)
        """
        try:
            s3_key = f"scripts/{job_id}/train_gpu.py"
            success, error = self.upload_file(
                script_path,
                s3_key,
                content_type='text/plain',
                metadata={
                    'job_id': job_id,
                    'upload_type': 'training_script',
                    'timestamp': str(time.time())
                }
            )
            
            if success:
                logger.info(f"Training script uploaded to s3://{self.bucket_name}/{s3_key}")
                return True, s3_key
            else:
                logger.error(f"Failed to upload training script: {error}")
                return False, error
                
        except Exception as e:
            logger.error(f"Error uploading training script: {e}")
            return False, str(e)
    
    def upload_model_files(self, 
                          model_files: List[Dict[str, str]], 
                          job_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Upload model files to S3
        
        Args:
            model_files: List of model file dictionaries with keys:
                - file_path: Local file path
                - model_type: Type of model (embedding, classification, etc.)
            job_id: Job identifier
            
        Returns:
            Dictionary mapping model_type to upload result
        """
        uploads = []
        for model_file in model_files:
            file_path = model_file['file_path']
            model_type = model_file['model_type']
            
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found: {file_path}")
                continue
            
            s3_key = f"models/{job_id}/{model_type}.keras"
            
            uploads.append({
                'file_path': file_path,
                's3_key': s3_key,
                'content_type': 'application/octet-stream',
                'metadata': {
                    'job_id': job_id,
                    'model_type': model_type,
                    'upload_type': 'model_file',
                    'timestamp': str(time.time())
                }
            })
        
        return self.upload_multiple_files(uploads)
    
    def download_file(self, s3_key: str, local_path: str) -> Tuple[bool, str]:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to download {s3_key}: {error_code} - {e}")
            return False, f"S3 error: {error_code} - {e}"
        
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_key}: {e}")
            return False, str(e)
    
    def list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with given prefix
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List of object dictionaries
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> Tuple[bool, str]:
        """
        Delete an object from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to delete {s3_key}: {error_code} - {e}")
            return False, f"S3 error: {error_code} - {e}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _verify_upload(self, s3_key: str, expected_hash: str) -> bool:
        """Verify uploaded file integrity"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            metadata = response.get('Metadata', {})
            actual_hash = metadata.get('file-hash', '')
            return actual_hash == expected_hash
        except:
            return False
    
    def _multipart_upload(self, file_path: str, s3_key: str, extra_args: Dict[str, Any]):
        """Upload large file using multipart upload"""
        try:
            # Create multipart upload
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                **extra_args
            )
            upload_id = response['UploadId']
            
            # Upload parts
            part_number = 1
            parts = []
            chunk_size = 50 * 1024 * 1024  # 50MB chunks
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    response = self.s3_client.upload_part(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk
                    )
                    
                    parts.append({
                        'ETag': response['ETag'],
                        'PartNumber': part_number
                    })
                    
                    part_number += 1
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                self.s3_client.abort_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    UploadId=upload_id
                )
            except:
                pass
            raise e

# Global uploader instance
_uploader = None

def get_s3_uploader(bucket_name: str = None, region: str = "us-east-1") -> EnhancedS3Uploader:
    """Get or create S3 uploader instance"""
    global _uploader
    
    if _uploader is None or (bucket_name and _uploader.bucket_name != bucket_name):
        if bucket_name is None:
            bucket_name = os.getenv('S3_BUCKET', 'signatureai-uploads')
        
        _uploader = EnhancedS3Uploader(bucket_name, region)
    
    return _uploader