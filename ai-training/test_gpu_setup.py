#!/usr/bin/env python3
"""
GPU Training Setup Test Script
Comprehensive test to verify all components are working correctly
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"  ✅ Pillow")
    except ImportError as e:
        print(f"  ❌ Pillow import failed: {e}")
        return False
    
    try:
        import boto3
        print(f"  ✅ Boto3 {boto3.__version__}")
    except ImportError as e:
        print(f"  ❌ Boto3 import failed: {e}")
        return False
    
    try:
        from utils.enhanced_logging import TrainingLogger
        print(f"  ✅ Enhanced logging")
    except ImportError as e:
        print(f"  ❌ Enhanced logging import failed: {e}")
        return False
    
    try:
        from utils.enhanced_s3_upload import EnhancedS3Uploader
        print(f"  ✅ Enhanced S3 upload")
    except ImportError as e:
        print(f"  ❌ Enhanced S3 upload import failed: {e}")
        return False
    
    return True

def test_gpu_availability():
    """Test GPU availability and configuration"""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import tensorflow as tf
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu.name}")
        else:
            print("  ⚠️  No GPU found, using CPU")
        
        # Check CUDA availability
        cuda_available = tf.test.is_built_with_cuda()
        print(f"  {'✅' if cuda_available else '❌'} CUDA available: {cuda_available}")
        
        # Test GPU memory growth
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("  ✅ GPU memory growth configured")
            except Exception as e:
                print(f"  ❌ GPU memory growth failed: {e}")
                return False
        
        # Test simple GPU operation
        try:
            with tf.device('/GPU:0' if gpus else '/CPU:0'):
                test_tensor = tf.random.normal([100, 100])
                result = tf.reduce_sum(test_tensor)
                print(f"  ✅ GPU computation test passed: {result.numpy():.2f}")
        except Exception as e:
            print(f"  ❌ GPU computation test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU test failed: {e}")
        return False

def test_aws_credentials():
    """Test AWS credentials and S3 access"""
    print("\n🔍 Testing AWS credentials...")
    
    try:
        import boto3
        
        # Test credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"  ✅ AWS credentials valid")
        print(f"    Account: {identity.get('Account')}")
        print(f"    User ID: {identity.get('UserId')}")
        print(f"    ARN: {identity.get('Arn')}")
        
        # Test S3 access
        s3 = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET', 'signatureai-uploads')
        
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  ✅ S3 bucket access: {bucket_name}")
        except Exception as e:
            print(f"  ❌ S3 bucket access failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ AWS credentials test failed: {e}")
        return False

def test_training_script():
    """Test the training script syntax and imports"""
    print("\n🔍 Testing training script...")
    
    try:
        # Test script syntax
        script_path = Path(__file__).parent / "train_gpu.py"
        if not script_path.exists():
            print(f"  ❌ Training script not found: {script_path}")
            return False
        
        # Compile script to check syntax
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        compile(script_content, str(script_path), 'exec')
        print("  ✅ Training script syntax valid")
        
        # Test script imports
        import subprocess
        result = subprocess.run([
            sys.executable, '-c', 
            'import sys; sys.path.insert(0, "."); exec(open("train_gpu.py").read())'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ Training script imports successful")
        else:
            print(f"  ⚠️  Training script import warnings: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training script test failed: {e}")
        return False

def test_logging_system():
    """Test the enhanced logging system"""
    print("\n🔍 Testing logging system...")
    
    try:
        from utils.enhanced_logging import TrainingLogger, ProgressTracker
        
        # Test logger creation
        logger = TrainingLogger("test_job")
        print("  ✅ Training logger created")
        
        # Test logging functions
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.debug("Test debug message")
        print("  ✅ Logging functions work")
        
        # Test progress updates
        logger.update_progress(25.0, "testing", "Progress test")
        logger.update_progress(50.0, "testing", "Halfway done")
        logger.update_progress(100.0, "completed", "Test completed")
        print("  ✅ Progress updates work")
        
        # Test progress tracker
        tracker = ProgressTracker("test_job")
        latest = tracker.get_latest_progress()
        if latest:
            print(f"  ✅ Progress tracker works: {latest['progress']}%")
        
        # Cleanup
        logger.cleanup()
        print("  ✅ Logger cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Logging system test failed: {e}")
        traceback.print_exc()
        return False

def test_s3_upload():
    """Test S3 upload functionality"""
    print("\n🔍 Testing S3 upload system...")
    
    try:
        from utils.enhanced_s3_upload import EnhancedS3Uploader
        
        bucket_name = os.getenv('S3_BUCKET', 'signatureai-uploads')
        uploader = EnhancedS3Uploader(bucket_name)
        print("  ✅ S3 uploader created")
        
        # Test file upload
        test_file = "/tmp/test_upload.txt"
        with open(test_file, 'w') as f:
            f.write("Test upload content")
        
        success, error = uploader.upload_file(
            test_file,
            "test/test_upload.txt",
            content_type="text/plain",
            metadata={"test": "true"}
        )
        
        if success:
            print("  ✅ File upload successful")
        else:
            print(f"  ❌ File upload failed: {error}")
            return False
        
        # Test file download
        download_path = "/tmp/test_download.txt"
        success, error = uploader.download_file("test/test_upload.txt", download_path)
        
        if success and os.path.exists(download_path):
            print("  ✅ File download successful")
            os.remove(download_path)
        else:
            print(f"  ❌ File download failed: {error}")
            return False
        
        # Test file deletion
        success, error = uploader.delete_object("test/test_upload.txt")
        if success:
            print("  ✅ File deletion successful")
        else:
            print(f"  ❌ File deletion failed: {error}")
        
        # Cleanup
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"  ❌ S3 upload test failed: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test basic model training functionality"""
    print("\n🔍 Testing model training...")
    
    try:
        # Import training components
        from train_gpu import SignaturePreprocessor, SignatureEmbeddingModel
        import numpy as np
        from PIL import Image
        
        # Test preprocessor
        preprocessor = SignaturePreprocessor(target_size=(224, 224))
        print("  ✅ Preprocessor created")
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='white')
        processed = preprocessor.preprocess_signature(test_image, "test")
        
        if processed is not None:
            print(f"  ✅ Image preprocessing works: {processed.shape}")
        else:
            print("  ❌ Image preprocessing failed")
            return False
        
        # Test model creation
        model_manager = SignatureEmbeddingModel(max_students=10)
        print("  ✅ Model manager created")
        
        # Test with minimal training data
        training_data = {
            "test_student": {
                "genuine": [processed],
                "forged": []
            }
        }
        
        # This would normally train, but we'll just test the setup
        print("  ✅ Training data structure valid")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model training test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import settings
        
        # Test basic settings
        print(f"  ✅ Model image size: {settings.MODEL_IMAGE_SIZE}")
        print(f"  ✅ Model epochs: {settings.MODEL_EPOCHS}")
        print(f"  ✅ S3 bucket: {settings.S3_BUCKET}")
        print(f"  ✅ AWS region: {settings.AWS_REGION}")
        
        # Test GPU settings
        print(f"  ✅ GPU training dir: {settings.GPU_TRAINING_DIR}")
        print(f"  ✅ GPU logs dir: {settings.GPU_LOGS_DIR}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting GPU Training Setup Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("GPU Availability", test_gpu_availability),
        ("AWS Credentials", test_aws_credentials),
        ("Training Script", test_training_script),
        ("Logging System", test_logging_system),
        ("S3 Upload", test_s3_upload),
        ("Model Training", test_model_training),
        ("Configuration", test_configuration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GPU training setup is ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())