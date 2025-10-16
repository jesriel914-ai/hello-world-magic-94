"""
google drive filepath: siamese_training/main.py
Flask API Server for Siamese Signature Training, Verification & Classification
NOW with 1:N Classification and Incremental Learning
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
import gc
import tensorflow as tf

from siamese_trainer import SiameseSignatureTrainer
from siamese_verifier import SiameseSignatureVerifier
from siamese_classifier import SiameseSignatureClassifier  # NEW
from siamese_incremental_trainer import SiameseIncrementalTrainer  # NEW

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     resources={r"/api/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False,
         "max_age": 3600
     }})

# Initialize all services
trainer = SiameseSignatureTrainer(base_dir='models')
verifier = SiameseSignatureVerifier(base_dir='models')
classifier = SiameseSignatureClassifier(base_dir='models')  # NEW
incremental_trainer = SiameseIncrementalTrainer(base_dir='models')  # NEW

def base64_to_image(base64_str):
    """Convert base64 string to OpenCV image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# ============================================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================================

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train Siamese model for a student"""
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
            
            # Rebuild classification database after training
            print(f"[CLASSIFIER] Rebuilding classification database...")
            classifier.build_classification_database(rebuild=True)
            
            return jsonify(response_data)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()
            
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/batch', methods=['POST'])
def train_batch():
    """Train multiple students in batch"""
    try:
        data = request.json
        students = data.get('students', [])
        
        if not students:
            return jsonify({'error': 'No students provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"BATCH TRAINING: {len(students)} students")
        print(f"{'='*60}\n")
        
        results = []
        failed = []
        
        for idx, student_data in enumerate(students):
            student_id = student_data.get('student_id')
            genuine_samples = student_data.get('genuine_samples', [])
            forged_samples = student_data.get('forged_samples', [])
            
            print(f"\n[{idx+1}/{len(students)}] Training student: {student_id}")
            
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                
                temp_dir = Path(tempfile.mkdtemp())
                
                try:
                    genuine_paths = []
                    for i, base64_img in enumerate(genuine_samples):
                        img = base64_to_image(base64_img)
                        path = temp_dir / f"genuine_{i}.jpg"
                        cv2.imwrite(str(path), img)
                        genuine_paths.append(str(path))
                    
                    forged_paths = []
                    if forged_samples:
                        for i, base64_img in enumerate(forged_samples):
                            img = base64_to_image(base64_img)
                            path = temp_dir / f"forged_{i}.jpg"
                            cv2.imwrite(str(path), img)
                            forged_paths.append(str(path))
                    
                    metadata = trainer.train_student_model(
                        student_id=student_id,
                        genuine_samples=genuine_paths,
                        forged_samples=forged_paths if forged_paths else None,
                        epochs=50
                    )
                    
                    trainer.save_reference_embeddings(student_id, genuine_paths)
                    
                    results.append({
                        'student_id': student_id,
                        'success': True,
                        'metadata': metadata
                    })
                    
                    print(f"[{idx+1}/{len(students)}] ✓ Success: {student_id}")
                    
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    del genuine_paths, forged_paths, metadata
                    gc.collect()
                    tf.keras.backend.clear_session()
                    
            except Exception as e:
                print(f"[{idx+1}/{len(students)}] ✗ Failed: {student_id} - {str(e)}")
                failed.append({
                    'student_id': student_id,
                    'error': str(e)
                })
        
        print(f"\n{'='*60}")
        print(f"BATCH TRAINING COMPLETE")
        print(f"  Success: {len(results)}/{len(students)}")
        print(f"  Failed: {len(failed)}/{len(students)}")
        print(f"{'='*60}\n")
        
        # Rebuild classification database after batch training
        if len(results) > 0:
            print(f"[CLASSIFIER] Rebuilding classification database...")
            classifier.build_classification_database(rebuild=True)
        
        return jsonify({
            'success': len(failed) == 0,
            'total': len(students),
            'succeeded': len(results),
            'failed': len(failed),
            'results': results,
            'errors': failed
        })
        
    except Exception as e:
        print(f"[ERROR] Batch training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_signature():
    """Verify signature against trained model (1:1 verification)"""
    try:
        data = request.json
        student_id = data.get('student_id')
        signature_base64 = data.get('signature_image')
        
        if not student_id or not signature_base64:
            return jsonify({'error': 'Missing student_id or signature_image'}), 400
        
        print(f"[VERIFICATION 1:1] Student: {student_id}")
        
        if ',' in signature_base64:
            signature_base64 = signature_base64.split(',')[1]
        
        img_data = base64.b64decode(signature_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        temp_path = f'temp_{student_id}_{int(time.time())}.jpg'
        cv2.imwrite(temp_path, img)
        
        result = verifier.verify_signature(student_id, temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        del img, nparr
        gc.collect()
        
        print(f"[VERIFICATION 1:1] Result: {result}")
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# NEW ENDPOINTS - Classification (1:N)
# ============================================================================

@app.route('/api/classify', methods=['POST'])
def classify_signature():
    """
    NEW: Classify signature to identify owner (1:N classification)
    Automatically identifies which student the signature belongs to
    
    Body: {
        "signature_image": "base64_image",
        "top_k": 3  (optional, default=3)
    }
    """
    try:
        data = request.json
        signature_base64 = data.get('signature_image')
        top_k = data.get('top_k', 3)
        
        if not signature_base64:
            return jsonify({'error': 'signature_image is required'}), 400
        
        print(f"[CLASSIFICATION 1:N] Identifying signature owner...")
        
        # Decode image
        if ',' in signature_base64:
            signature_base64 = signature_base64.split(',')[1]
        
        img_data = base64.b64decode(signature_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Save temporarily
        temp_path = f'temp_classify_{int(time.time())}.jpg'
        cv2.imwrite(temp_path, img)
        
        # Classify
        result = classifier.classify_signature(temp_path, top_k=top_k)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        del img, nparr
        gc.collect()
        
        print(f"[CLASSIFICATION 1:N] Result: {result.get('student_id', 'UNKNOWN')}")
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"[ERROR] Classification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify/realtime', methods=['POST'])
def classify_realtime():
    """
    NEW: Real-time classification for webcam frames
    Optimized for speed (no file I/O)
    
    Body: {
        "frame": "base64_image"
    }
    """
    try:
        data = request.json
        frame_base64 = data.get('frame')
        
        if not frame_base64:
            return jsonify({'error': 'frame is required'}), 400
        
        # Decode image
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        
        img_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Classify (no file I/O)
        result = classifier.realtime_classify_frame(frame)
        
        del frame, nparr
        gc.collect()
        
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"[ERROR] Real-time classification failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/classifier/rebuild', methods=['POST'])
def rebuild_classifier():
    """
    NEW: Force rebuild classification database
    Use after training new models or if database is corrupted
    """
    try:
        print(f"[CLASSIFIER] Rebuilding classification database...")
        classifier.build_classification_database(rebuild=True)
        
        return jsonify({
            'success': True,
            'message': 'Classification database rebuilt successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild classifier: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# NEW ENDPOINTS - Incremental Learning
# ============================================================================

@app.route('/api/incremental/add-genuine', methods=['POST'])
def add_genuine_samples():
    """
    NEW: Add new genuine samples to existing student model
    WITHOUT retraining from scratch
    
    Body: {
        "student_id": "2025001",
        "new_samples": ["base64_image1", "base64_image2", ...],
        "update_threshold": true  (optional)
    }
    """
    try:
        data = request.json
        student_id = data.get('student_id')
        new_samples = data.get('new_samples', [])
        update_threshold = data.get('update_threshold', True)
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        if len(new_samples) == 0:
            return jsonify({'error': 'No new samples provided'}), 400
        
        print(f"\n[INCREMENTAL] Adding {len(new_samples)} genuine samples for {student_id}")
        
        # Check if retraining is recommended
        check = incremental_trainer.check_if_retraining_needed(student_id, len(new_samples))
        
        if check['needs_retraining']:
            return jsonify({
                'success': False,
                'needs_retraining': True,
                'reason': check['reason'],
                'recommendation': check['recommendation']
            }), 400
        
        # Save new samples temporarily
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            sample_paths = []
            for i, base64_img in enumerate(new_samples):
                img = base64_to_image(base64_img)
                path = temp_dir / f"new_genuine_{i}.jpg"
                cv2.imwrite(str(path), img)
                sample_paths.append(str(path))
            
            # Add samples incrementally
            metadata = incremental_trainer.add_new_genuine_samples(
                student_id=student_id,
                new_genuine_samples=sample_paths,
                update_threshold=update_threshold
            )
            
            # Update classification database
            print(f"[CLASSIFIER] Updating database for {student_id}...")
            classifier.update_database_for_student(student_id)
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'message': f'Added {len(new_samples)} genuine samples to {student_id}'
            })
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()
            
    except Exception as e:
        print(f"[ERROR] Incremental learning failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/incremental/add-forged', methods=['POST'])
def add_forged_samples():
    """
    NEW: Add new forged samples to existing student model
    
    Body: {
        "student_id": "2025001",
        "new_samples": ["base64_image1", "base64_image2", ...]
    }
    """
    try:
        data = request.json
        student_id = data.get('student_id')
        new_samples = data.get('new_samples', [])
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        if len(new_samples) == 0:
            return jsonify({'error': 'No new samples provided'}), 400
        
        print(f"\n[INCREMENTAL] Adding {len(new_samples)} forged samples for {student_id}")
        
        # Save new samples temporarily
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            sample_paths = []
            for i, base64_img in enumerate(new_samples):
                img = base64_to_image(base64_img)
                path = temp_dir / f"new_forged_{i}.jpg"
                cv2.imwrite(str(path), img)
                sample_paths.append(str(path))
            
            # Add forged samples
            metadata = incremental_trainer.add_new_forged_samples(
                student_id=student_id,
                new_forged_samples=sample_paths,
                retrain_contrastive=False
            )
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'message': f'Added {len(new_samples)} forged samples to {student_id}'
            })
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()
            
    except Exception as e:
        print(f"[ERROR] Adding forged samples failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/incremental/check', methods=['POST'])
def check_incremental():
    """
    NEW: Check if incremental learning is suitable or if full retraining is needed
    
    Body: {
        "student_id": "2025001",
        "new_sample_count": 5
    }
    """
    try:
        data = request.json
        student_id = data.get('student_id')
        new_sample_count = data.get('new_sample_count', 0)
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        check = incremental_trainer.check_if_retraining_needed(student_id, new_sample_count)
        
        return jsonify(check)
        
    except Exception as e:
        print(f"[ERROR] Check failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================================

@app.route('/api/models/list', methods=['GET'])
def list_trained_models():
    """List all students with trained models"""
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

@app.route('/api/model/status/<student_id>', methods=['GET'])
def model_status(student_id):
    """Check if model exists for student"""
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
    import tensorflow as tf
    
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    # Check classifier status
    classifier_ready = classifier.index is not None and len(classifier.student_id_map) > 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'siamese-signature-training',
        'version': '3.0-gpu-classifier-incremental',
        'gpu_available': gpu_available,
        'tensorflow_version': tf.__version__,
        'features': {
            'training': True,
            'verification_1to1': True,
            'classification_1toN': True,
            'incremental_learning': True
        },
        'classifier': {
            'ready': classifier_ready,
            'num_students': len(set(classifier.student_id_map)) if classifier_ready else 0,
            'num_embeddings': classifier.index.ntotal if classifier_ready else 0
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SIAMESE SIGNATURE API (GPU + CLASSIFIER + INCREMENTAL)")
    print("="*60)
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("  Environment: Google Colab")
        
        # Setup ngrok
        from pyngrok import ngrok
        
        NGROK_TOKEN = os.environ.get('NGROK_AUTH_TOKEN')
        if not NGROK_TOKEN:
            print("\n⚠️  WARNING: NGROK_AUTH_TOKEN not set!")
            print("Please set it with:")
            print('  os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"')
        else:
            ngrok.set_auth_token(NGROK_TOKEN)
        
        public_url = ngrok.connect(5000)
        print(f"  Public URL: {public_url}")
        print(f"\n  🌍 UPDATE YOUR FRONTEND .env WITH:")
        print(f"  VITE_SIAMESE_API_URL={public_url}")
        
    except ImportError:
        IN_COLAB = False
        print("  Environment: Local")
        print(f"  Server: http://localhost:5000")
    
    print(f"  Status: Starting...")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  🎓 TRAINING:")
    print("    POST /api/train              - Train single student")
    print("    POST /api/train/batch        - Train multiple students")
    print("  🔍 VERIFICATION (1:1):")
    print("    POST /api/verify             - Verify signature")
    print("  🎯 CLASSIFICATION (1:N):")
    print("    POST /api/classify           - Identify signature owner")
    print("    POST /api/classify/realtime  - Real-time classification")
    print("    POST /api/classifier/rebuild - Rebuild database")
    print("  🔄 INCREMENTAL LEARNING:")
    print("    POST /api/incremental/add-genuine - Add genuine samples")
    print("    POST /api/incremental/add-forged  - Add forged samples")
    print("    POST /api/incremental/check       - Check if retraining needed")
    print("  📊 MANAGEMENT:")
    print("    GET  /api/models/list        - List trained models")
    print("    GET  /api/model/status/:id   - Check model status")
    print("    GET  /api/health             - Health check")
    print("\n" + "="*60 + "\n")
    
    # Build classification database on startup
    print("🔧 Initializing classification database...")
    try:
        classifier.build_classification_database(rebuild=False)
    except Exception as e:
        print(f"⚠️  Failed to initialize classifier: {e}")
        print("   Classifier will be built after first training")
    
    print("\n🚀 Server ready!\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False)