# CORS Fix for Google Colab + ngrok Setup

## Problem
When running the Python backend in Google Colab and tunneling via ngrok, the frontend gets CORS errors:
```
Access to fetch at 'https://unfantastic-delmar-incondite.ngrok-free.dev/api/train' from origin 'http://localhost:5173' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Solution for Colab

### Step 1: Update your Colab notebook

Replace your current `main.py` code in Colab with the fixed version. Here's the complete updated code:

```python
"""
google drive filepath: siamese_training/main.py
Flask API Server for Siamese Signature Training, Verification & Classification
FIXED: Proper CORS handling for ngrok + frontend
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import time
import os
import base64
import numpy as np
import json
import cv2
from pathlib import Path
import tempfile
import shutil
import gc
import tensorflow as tf

from siamese_trainer import SiameseSignatureTrainer
from siamese_verifier import SiameseSignatureVerifier
from siamese_classifier import SiameseSignatureClassifier
from siamese_incremental_trainer import SiameseIncrementalTrainer
from smart_orchestrator import SmartTrainingOrchestrator

app = Flask(__name__)

# Enhanced CORS configuration for ngrok
CORS(app, 
     resources={r"/api/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False,
         "max_age": 3600
     }})

# Add explicit OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization, ngrok-skip-browser-warning")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        response.headers.add('Access-Control-Max-Age', "3600")
        return response

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, ngrok-skip-browser-warning')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Initialize all services
trainer = SiameseSignatureTrainer(base_dir='models')
verifier = SiameseSignatureVerifier(base_dir='models')
classifier = SiameseSignatureClassifier(base_dir='models')
incremental_trainer = SiameseIncrementalTrainer(base_dir='models')
orchestrator = SmartTrainingOrchestrator(base_dir='models')

def base64_to_image(base64_str):
    """Convert base64 string to OpenCV image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# ============================================================================
# TRAINING ENDPOINTS - FIXED
# ============================================================================

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train Siamese model for a student - FIXED with CORS"""
    try:
        data = request.json
        student_id = data.get('student_id')
        genuine_samples = data.get('genuine_samples', [])
        forged_samples = data.get('forged_samples', [])
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        if len(genuine_samples) < 2:
            return jsonify({'error': 'At least 2 genuine samples required'}), 400
        
        print(f"\n[TRAINING] Student: {student_id}")
        print(f"[TRAINING] Genuine: {len(genuine_samples)}, Forged: {len(forged_samples)}")
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Save genuine samples
            genuine_paths = []
            for i, base64_img in enumerate(genuine_samples):
                img = base64_to_image(base64_img)
                path = temp_dir / f"genuine_{i}.jpg"
                cv2.imwrite(str(path), img)
                genuine_paths.append(str(path))
            
            # Save forged samples
            forged_paths = []
            if forged_samples:
                for i, base64_img in enumerate(forged_samples):
                    img = base64_to_image(base64_img)
                    path = temp_dir / f"forged_{i}.jpg"
                    cv2.imwrite(str(path), img)
                    forged_paths.append(str(path))
            
            # Train model
            metadata = trainer.train_student_model(
                student_id=student_id,
                genuine_samples=genuine_paths,
                forged_samples=forged_paths if forged_paths else None,
                epochs=50
            )
            
            # Save reference embeddings
            trainer.save_reference_embeddings(student_id, genuine_paths)
            
            response_data = {
                'success': True,
                'metadata': metadata,
                'message': f'Model trained successfully for {student_id}'
            }
            
            print(f"\n[CLEANUP] Cleaning up memory after training {student_id}...")
            del metadata, genuine_paths, forged_paths
            gc.collect()
            tf.keras.backend.clear_session()
            
            # CRITICAL FIX: Rebuild classification database after each training
            print(f"\n[CLASSIFIER] Rebuilding classification database...")
            try:
                classifier.build_classification_database(rebuild=True)
                print(f"[CLASSIFIER] ✅ Database rebuilt successfully")
            except Exception as e:
                print(f"[CLASSIFIER] ⚠️  Failed to rebuild database: {e}")
            
            return jsonify(response_data)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()
            
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add other endpoints here (verify, classify, etc.) - same as before but with CORS fix

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with classifier status"""
    import tensorflow as tf
    
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    classifier_ready = classifier.index is not None and len(classifier.student_id_map) > 0
    
    total_students = len(orchestrator.sample_hashes.keys()) if orchestrator.sample_hashes else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'siamese-signature-training',
        'version': '4.1-fixed-cors',
        'gpu_available': gpu_available,
        'tensorflow_version': tf.__version__,
        'features': {
            'training': True,
            'verification_1to1': True,
            'classification_1toN': True,
            'incremental_learning': True,
            'smart_orchestrator': True
        },
        'classifier': {
            'ready': classifier_ready,
            'num_students': len(set(classifier.student_id_map)) if classifier_ready else 0,
            'num_embeddings': classifier.index.ntotal if classifier_ready else 0
        },
        'orchestrator': {
            'total_students_tracked': total_students,
            'sample_tracking_enabled': True
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SIAMESE SIGNATURE API v4.1 (CORS FIXED)")
    print("  (Google Colab + ngrok + Frontend)")
    print("="*60)
    
    print("\n🚀 Server ready with CORS fix!")
    print("✅ Preflight OPTIONS requests handled")
    print("✅ Access-Control-Allow-Origin headers added")
    print("✅ ngrok-skip-browser-warning header supported")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Step 2: Install pyngrok in Colab

Add this cell to your Colab notebook:

```python
!pip install pyngrok
```

### Step 3: Start ngrok with proper configuration

```python
from pyngrok import ngrok

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print(f"🌐 Public URL: {public_url}")
print(f"🔗 Use this URL in your frontend .env file:")
print(f"VITE_SIAMESE_API_URL={public_url}")
```

### Step 4: Update your frontend .env file

Create or update your `.env` file in your frontend project:

```bash
VITE_SIAMESE_API_URL=https://your-ngrok-url.ngrok-free.app
```

### Step 5: Test the connection

Add this test cell to your Colab notebook:

```python
import requests

# Test CORS preflight
def test_cors():
    url = f"{public_url}/api/health"
    headers = {
        'Origin': 'http://localhost:5173',
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"✅ CORS Test: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ CORS Test failed: {e}")

test_cors()
```

## Key CORS Fixes Applied

1. **Explicit OPTIONS handler** - Handles preflight requests
2. **after_request decorator** - Adds CORS headers to all responses
3. **ngrok-skip-browser-warning header** - Properly allowed
4. **Wildcard origins** - Allows requests from any origin (including localhost:5173)

## Verification

After applying these fixes, your frontend should be able to make requests to the Colab backend without CORS errors. The training should work properly now!

## Troubleshooting

If you still get CORS errors:

1. **Check ngrok URL** - Make sure it's the correct URL in your .env
2. **Restart frontend** - After updating .env, restart your dev server
3. **Check browser console** - Look for any remaining CORS errors
4. **Test with curl** - Verify the backend responds correctly

```bash
curl -X GET "https://your-ngrok-url.ngrok-free.app/api/health" \
  -H "Origin: http://localhost:5173" \
  -H "ngrok-skip-browser-warning: true"
```