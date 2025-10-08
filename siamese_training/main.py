"""
Verification-Only Flask API - Local Deployment
No training capabilities, just signature verification
Optimized for R5 3400G (CPU only)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os
import base64
import numpy as np
import json
import cv2
import gc

from siamese_verifier import SiameseSignatureVerifier

app = Flask(__name__)

# Universal CORS - works for both ngrok and localhost
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "ngrok-skip-browser-warning"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})
# Initialize verifier only
verifier = SiameseSignatureVerifier(base_dir='models')

def base64_to_image(base64_str):
    """Convert base64 string to OpenCV image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/api/verify', methods=['POST'])
def verify_signature():
    """
    Verify signature against trained model
    POST /api/verify
    Body: {
        "student_id": "STU0000389",
        "signature_image": "base64_image"
    }
    """
    try:
        data = request.json
        student_id = data.get('student_id')
        signature_base64 = data.get('signature_image')
        
        if not student_id or not signature_base64:
            return jsonify({'error': 'Missing student_id or signature_image'}), 400
        
        print(f"[VERIFICATION] Student: {student_id}")
        
        # Remove data URL prefix
        if ',' in signature_base64:
            signature_base64 = signature_base64.split(',')[1]
        
        # Decode to image
        img_data = base64.b64decode(signature_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Save temporarily
        temp_path = f'temp_{student_id}_{int(time.time())}.jpg'
        cv2.imwrite(temp_path, img)
        
        # Verify
        result = verifier.verify_signature(student_id, temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        del img, nparr
        gc.collect()
        
        print(f"[VERIFICATION] Result: {result['is_verified']}")
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/list', methods=['GET'])
def list_trained_models():
    """List all students with trained models"""
    try:
        from pathlib import Path
        models_dir = Path('models')
        
        if not models_dir.exists():
            return jsonify({'students': []})
        
        trained_students = []
        for student_dir in models_dir.iterdir():
            if student_dir.is_dir():
                metadata_file = student_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        trained_students.append({
                            'student_id': student_dir.name,
                            'metadata': metadata
                        })
        
        return jsonify({'students': trained_students})
        
    except Exception as e:
        print(f"[ERROR] Failed to list models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/status/<student_id>', methods=['GET'])
def model_status(student_id):
    """Check if model exists for student"""
    try:
        exists, metadata = verifier.check_model_exists(student_id)
        
        if exists:
            return jsonify({'exists': True, 'metadata': metadata})
        else:
            return jsonify({'exists': False})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'siamese-signature-verification',
        'version': '1.0-local',
        'mode': 'verification-only'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SIAMESE SIGNATURE VERIFICATION (LOCAL)")
    print("="*60)
    print("  Environment: Local (CPU Only)")
    print("  Server: http://localhost:5000")
    print("  Mode: Verification Only")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  POST /api/verify             - Verify signature")
    print("  GET  /api/models/list        - List trained models")
    print("  GET  /api/model/status/:id   - Check model status")
    print("  GET  /api/health             - Health check")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)