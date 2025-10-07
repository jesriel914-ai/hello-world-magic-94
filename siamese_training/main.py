# filepath: siamese_training/main.py
"""
Flask API Server for Siamese Signature Training & Verification
Run with: python main.py
"""


from flask import Flask, request, jsonify
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

from siamese_trainer import SiameseSignatureTrainer
from siamese_verifier import SiameseSignatureVerifier

app = Flask(__name__)
CORS(app)

# Initialize trainer and verifier
trainer = SiameseSignatureTrainer(base_dir='models')
verifier = SiameseSignatureVerifier(base_dir='models')

def base64_to_image(base64_str):
    """Convert base64 string to OpenCV image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train Siamese model for a student
    POST /api/train
    Body: {
        "student_id": "2021-0001",
        "genuine_samples": ["base64_image1", ...],
        "forged_samples": ["base64_image1", ...]  // Optional
    }
    """
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
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Save genuine samples
            genuine_paths = []
            for i, base64_img in enumerate(genuine_samples):
                img = base64_to_image(base64_img)
                path = temp_dir / f"genuine_{i}.jpg"
                cv2.imwrite(str(path), img)
                genuine_paths.append(str(path))
            
            # Save forged samples if provided
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
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'message': f'Model trained successfully for {student_id}'
            })
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/list', methods=['GET'])
def list_trained_models():
    """
    List all students with trained models
    GET /api/models/list
    """
    try:
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
# In your Flask app.py, update the verify endpoint:

@app.route('/api/verify', methods=['POST'])
def verify_signature():
    try:
        data = request.json
        student_id = data.get('student_id')
        signature_base64 = data.get('signature_image')
        
        if not student_id or not signature_base64:
            return jsonify({'error': 'Missing student_id or signature_image'}), 400
        
        print(f"[VERIFICATION] Student: {student_id}")
        
        # Remove data URL prefix if present
        if ',' in signature_base64:
            signature_base64 = signature_base64.split(',')[1]
        
        # Decode base64 to image
        img_data = base64.b64decode(signature_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Save temporarily for verification
        temp_path = f'temp_{student_id}_{int(time.time())}.jpg'
        cv2.imwrite(temp_path, img)
        
        # Verify using the verifier
        verifier = SiameseSignatureVerifier()
        result = verifier.verify_signature(student_id, temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"[VERIFICATION] Result: {result}")
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/api/model/status/<student_id>', methods=['GET'])
def model_status(student_id):
    """
    Check if model exists for student
    GET /api/model/status/{student_id}
    """
    try:
        exists, metadata = verifier.check_model_exists(student_id)
        
        if exists:
            return jsonify({
                'exists': True,
                'metadata': metadata
            })
        else:
            return jsonify({'exists': False})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'siamese-signature-training',
        'version': '1.0'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SIAMESE SIGNATURE TRAINING API")
    print("="*60)
    print(f"  Server: http://localhost:5000")
    print(f"  Status: Starting...")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  POST /api/train              - Train student model")
    print("  POST /api/verify             - Verify signature")
    print("  GET  /api/model/status/:id   - Check model status")
    print("  GET  /api/health             - Health check")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)