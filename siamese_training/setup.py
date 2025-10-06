"""
Setup script for Siamese Signature Verification System
Run this first to set up the environment and test the installation
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_installation():
    """Test if TensorFlow and other packages work correctly"""
    print("\nTesting installation...")
    try:
        import tensorflow as tf
        import numpy as np
        import cv2
        from PIL import Image
        import sklearn
        
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"✅ PIL version: {Image.__version__}")
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
        else:
            print("ℹ️  No GPU detected, using CPU")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = ['models', 'data', 'data/example_student/genuine', 'data/example_student/forged']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("Siamese Signature Verification Setup")
    print("=" * 40)
    print("This script will set up the environment for signature verification")
    print()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during installation test")
        return
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your signature images to the data/ directories")
    print("2. Run: python train_siamese.py --student_id example --genuine_dir data/example_student/genuine --forged_dir data/example_student/forged")
    print("3. Run: python verify_signature.py --student_id example --reference <ref_image> --test <test_image>")
    print("\nFor detailed usage, see README.md")

if __name__ == "__main__":
    main()