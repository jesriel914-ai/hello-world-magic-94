#!/usr/bin/env python3
# filepath: siamese_training/test_imports.py
"""
Quick test to verify all modules can be imported
Run this before deploying to Colab to catch any import errors
"""

import sys

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...\n")
    
    errors = []
    
    # Test preprocessing
    try:
        import preprocessing
        print("✅ preprocessing.py - OK")
    except Exception as e:
        errors.append(f"❌ preprocessing.py - {e}")
        print(errors[-1])
    
    # Test model
    try:
        import model
        print("✅ model.py - OK")
    except Exception as e:
        errors.append(f"❌ model.py - {e}")
        print(errors[-1])
    
    # Test training
    try:
        import training
        print("✅ training.py - OK")
    except Exception as e:
        errors.append(f"❌ training.py - {e}")
        print(errors[-1])
    
    # Test identification
    try:
        import identification
        print("✅ identification.py - OK")
    except Exception as e:
        errors.append(f"❌ identification.py - {e}")
        print(errors[-1])
    
    # Test main
    try:
        import main
        print("✅ main.py - OK")
    except Exception as e:
        errors.append(f"❌ main.py - {e}")
        print(errors[-1])
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"❌ FAILED: {len(errors)} module(s) have errors")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ SUCCESS: All modules imported successfully!")
        print("   Ready to deploy to Google Colab")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
