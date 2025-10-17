"""
google drive filepath: siamese_training/main.py
Flask API Server for Siamese Signature Training, Verification & Classification
FIXED: Proper classification database updates after training
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
from siamese_classifier import SiameseSignatureClassifier
from siamese_incremental_trainer import SiameseIncrementalTrainer
from smart_orchestrator import SmartTrainingOrchestrator

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
    """Train Siamese model for a student - FIXED with classifier rebuild"""
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

@app.route('/api/train/batch', methods=['POST'])
def train_batch():
    """Train multiple students in batch - FIXED with classifier rebuild"""
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
        
        # CRITICAL FIX: Rebuild classification database after batch training
        if len(results) > 0:
            print(f"[CLASSIFIER] Rebuilding classification database for {len(results)} students...")
            try:
                classifier.build_classification_database(rebuild=True)
                print(f"[CLASSIFIER] ✅ Database rebuilt successfully")
            except Exception as e:
                print(f"[CLASSIFIER] ⚠️  Failed to rebuild database: {e}")
        
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

@app.route('/api/train/smart', methods=['POST'])
def smart_train():
    """
    FIXED: Smart training - automatically detects:
    - New students (full training)
    - Existing students with new samples (incremental)
    - Students with no changes (skip)
    
    Now properly rebuilds classifier after all updates
    """
    try:
        data = request.json
        students = data.get('students', [])
        
        if not students:
            return jsonify({'error': 'No students provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"SMART TRAINING (FIXED): {len(students)} students")
        print(f"{'='*60}\n")
        
        # Execute smart training (now with proper classifier rebuild)
        results = orchestrator.execute_smart_training(students)
        
        return jsonify({
            'success': results['failed'] == 0,
            'total': results['total'],
            'new_trained': results['new_trained'],
            'incremental_updated': results['incremental_updated'],
            'retrained': results['retrained'],
            'skipped': results['skipped'],
            'failed': results['failed'],
            'results': results['results']
        })
        
    except Exception as e:
        print(f"[ERROR] Smart training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# VERIFICATION & CLASSIFICATION ENDPOINTS (unchanged)
# ============================================================================

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

@app.route('/api/classify', methods=['POST'])
def classify_signature():
    """Classify signature to identify owner (1:N classification)"""
    try:
        data = request.json
        signature_base64 = data.get('signature_image')
        top_k = data.get('top_k', 3)
        
        if not signature_base64:
            return jsonify({'error': 'signature_image is required'}), 400
        
        print(f"[CLASSIFICATION 1:N] Identifying signature owner...")
        
        if ',' in signature_base64:
            signature_base64 = signature_base64.split(',')[1]
        
        img_data = base64.b64decode(signature_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        temp_path = f'temp_classify_{int(time.time())}.jpg'
        cv2.imwrite(temp_path, img)
        
        result = classifier.classify_signature(temp_path, top_k=top_k)
        
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

@app.route('/api/classifier/rebuild', methods=['POST'])
def rebuild_classifier():
    """Force rebuild classification database"""
    try:
        print(f"[CLASSIFIER] Force rebuilding classification database...")
        classifier.build_classification_database(rebuild=True)
        
        return jsonify({
            'success': True,
            'message': 'Classification database rebuilt successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild classifier: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# INCREMENTAL LEARNING ENDPOINTS (unchanged)
# ============================================================================

@app.route('/api/incremental/add-genuine', methods=['POST'])
def add_genuine_samples():
    """Add new genuine samples to existing student model"""
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
        
        check = incremental_trainer.check_if_retraining_needed(student_id, len(new_samples))
        
        if check['needs_retraining']:
            return jsonify({
                'success': False,
                'needs_retraining': True,
                'reason': check['reason'],
                'recommendation': check['recommendation']
            }), 400
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            sample_paths = []
            for i, base64_img in enumerate(new_samples):
                img = base64_to_image(base64_img)
                path = temp_dir / f"new_genuine_{i}.jpg"
                cv2.imwrite(str(path), img)
                sample_paths.append(str(path))
            
            metadata = incremental_trainer.add_new_genuine_samples(
                student_id=student_id,
                new_genuine_samples=sample_paths,
                update_threshold=update_threshold
            )
            
            # Update classification database
            print(f"[CLASSIFIER] Updating database for {student_id}...")
            classifier.build_classification_database(rebuild=True)
            
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

# ============================================================================
# MANAGEMENT ENDPOINTS (unchanged)
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
        'version': '4.1-fixed-incremental',
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
    print("  SIAMESE SIGNATURE API v4.1 (FIXED)")
    print("  (Proper Incremental Learning + Classification)")
    print("="*60)
    
    print("\nAvailable Endpoints:")
    print("  🎓 TRAINING:")
    print("    POST /api/train              - Train single student")
    print("    POST /api/train/batch        - Train multiple students")
    print("    POST /api/train/smart        - 🔧 FIXED: Smart training (auto-detect)")
    print("  🔍 VERIFICATION & CLASSIFICATION:")
    print("    POST /api/verify             - Verify signature (1:1)")
    print("    POST /api/classify           - Identify signature owner (1:N)")
    print("    POST /api/classifier/rebuild - Rebuild classification DB")
    print("  🔄 INCREMENTAL LEARNING:")
    print("    POST /api/incremental/add-genuine - Add genuine samples")
    print("  📊 MANAGEMENT:")
    print("    GET  /api/models/list        - List trained models")
    print("    GET  /api/health             - Health check")
    print("\n" + "="*60 + "\n")
    
    # Build classification database on startup
    print("🔧 Initializing classification database...")
    try:
        classifier.build_classification_database(rebuild=False)
        print("✅ Classification database ready")
    except Exception as e:
        print(f"⚠️  Failed to initialize classifier: {e}")
    
    print("\n🚀 Server ready!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)