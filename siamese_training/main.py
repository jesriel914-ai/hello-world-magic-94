# filepath: siamese_training/main.py
"""
Flask API Server for Siamese Signature Identification
Optimized for Cloudflare Tunnel deployment
"""

import os
import sys
import json
import zipfile
import io
import traceback
from datetime import datetime
from typing import Dict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import threading

# Import our modules
from training import train_batch_students, get_trainer
from identification import identify_signature, verify_signature, get_identifier

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Allow all origins (as requested)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global training state
training_state = {
    'is_training': False,
    'progress': 0.0,
    'current_student': None,
    'total_students': 0,
    'completed_students': 0,
    'error': None,
    'start_time': None
}

training_lock = threading.Lock()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        trainer = get_trainer()
        status = trainer.get_status()
        
        return jsonify({
            'status': 'online',
            'message': 'Siamese Signature Identification API',
            'version': '1.0.0',
            'model_loaded': status['is_trained'],
            'total_students': status['total_students'],
            'total_embeddings': status['total_embeddings'],
            'last_updated': status['last_updated'],
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# MODEL STATUS
# ============================================================================

@app.route('/status', methods=['GET'])
def get_status():
    """Get model status and statistics"""
    try:
        trainer = get_trainer()
        status = trainer.get_status()
        
        return jsonify({
            'is_trained': status['is_trained'],
            'total_students': status['total_students'],
            'total_embeddings': status['total_embeddings'],
            'last_updated': status['last_updated'],
            'students': status['students'],
            'architecture': 'Siamese Network (MobileNetV2)',
            'returns_single_result': True,
            'nonsignature_detection': True,
            'incremental_learning': True
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TRAINING
# ============================================================================

@app.route('/train_batch', methods=['POST'])
def train_batch():
    """
    Batch train multiple students at once
    Training happens in background thread
    
    Request body:
    {
        "students": [
            {
                "student_id": "2020-001",
                "genuine_samples": ["base64_image1", "base64_image2", ...],
                "forged_samples": []
            },
            ...
        ],
        "epochs": 30,  # optional
        "batch_size": 96  # optional
    }
    """
    global training_state
    
    try:
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({'error': 'Missing students data'}), 400
        
        students_list = data['students']
        epochs = data.get('epochs', 30)
        batch_size = data.get('batch_size', 96)
        
        if not students_list or len(students_list) == 0:
            return jsonify({'error': 'No students provided'}), 400
        
        # Check if already training
        with training_lock:
            if training_state['is_training']:
                return jsonify({
                    'error': 'Training already in progress',
                    'current_progress': training_state['progress']
                }), 409
            
            # Set training state
            training_state['is_training'] = True
            training_state['progress'] = 0.0
            training_state['current_student'] = None
            training_state['total_students'] = len(students_list)
            training_state['completed_students'] = 0
            training_state['error'] = None
            training_state['start_time'] = datetime.now().timestamp()
        
        # Convert to training format
        students_data = {}
        for student in students_list:
            student_id = student['student_id']
            genuine = student.get('genuine_samples', [])
            forged = student.get('forged_samples', [])
            
            students_data[student_id] = {
                'genuine': genuine,
                'forged': forged
            }
        
        print(f"\n🚀 Starting batch training for {len(students_data)} students...")
        
        # Start training in background thread
        def train_background():
            global training_state
            
            try:
                # Progress callback
                def progress_callback(progress, message):
                    with training_lock:
                        training_state['progress'] = progress
                        training_state['current_student'] = message
                
                # Train
                result = train_batch_students(
                    students_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    progress_callback=progress_callback
                )
                
                # Update state
                with training_lock:
                    training_state['is_training'] = False
                    training_state['progress'] = 100.0
                    training_state['completed_students'] = result['students_trained']
                    training_state['error'] = None
                
                print(f"✅ Training completed successfully!")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Training failed: {error_msg}")
                print(traceback.format_exc())
                
                with training_lock:
                    training_state['is_training'] = False
                    training_state['error'] = error_msg
        
        # Start training thread
        training_thread = threading.Thread(target=train_background, daemon=True)
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Training started for {len(students_data)} students',
            'total_students': len(students_data)
        }), 200
        
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        print(traceback.format_exc())
        
        with training_lock:
            training_state['is_training'] = False
            training_state['error'] = str(e)
        
        return jsonify({'error': str(e)}), 500


@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Get current training status (for polling)"""
    with training_lock:
        return jsonify({
            'is_training': training_state['is_training'],
            'progress': training_state['progress'],
            'current_student': training_state['current_student'],
            'total_students': training_state['total_students'],
            'completed_students': training_state['completed_students'],
            'error': training_state['error'],
            'start_time': training_state['start_time']
        }), 200


# ============================================================================
# IDENTIFICATION
# ============================================================================

@app.route('/classify', methods=['POST'])
def classify_signature():
    """
    Identify the owner of a signature
    Returns ONLY the best match (single result)
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    
    Response:
    {
        "identified": true/false,
        "student_id": "2020-001" or null,
        "confidence": 0.95,
        "distance": 0.234,
        "decision": "ACCEPT" / "UNCERTAIN" / "UNKNOWN" / "NON_SIGNATURE",
        "message": "...",
        "threshold_info": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        image_data = data['image']
        
        # Identify signature
        result = identify_signature(image_data, is_base64=True)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Classification error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify_signature_endpoint():
    """
    Verify if a signature belongs to a specific student (1:1 verification)
    
    Request body:
    {
        "image": "base64_encoded_image",
        "student_id": "2020-001"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'student_id' not in data:
            return jsonify({'error': 'Missing image or student_id'}), 400
        
        image_data = data['image']
        student_id = data['student_id']
        
        # Verify signature
        result = verify_signature(image_data, student_id, is_base64=True)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STUDENT MANAGEMENT
# ============================================================================

@app.route('/delete_student', methods=['POST'])
def delete_student():
    """
    Delete a student from the database
    
    Request body:
    {
        "student_id": "2020-001"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'student_id' not in data:
            return jsonify({'error': 'Missing student_id'}), 400
        
        student_id = data['student_id']
        
        trainer = get_trainer()
        trainer.delete_student(student_id)
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_student/<student_id>', methods=['GET'])
def get_student_info(student_id):
    """Get information about a specific student"""
    try:
        identifier = get_identifier()
        info = identifier.get_student_info(student_id)
        
        if info is None:
            return jsonify({'error': 'Student not found'}), 404
        
        return jsonify(info), 200
        
    except Exception as e:
        print(f"❌ Error getting student info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/list_students', methods=['GET'])
def list_students():
    """Get list of all trained students"""
    try:
        identifier = get_identifier()
        students = identifier.get_all_students()
        
        return jsonify({
            'students': students,
            'total': len(students)
        }), 200
        
    except Exception as e:
        print(f"❌ Error listing students: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MODEL EXPORT / DOWNLOAD
# ============================================================================

@app.route('/export_cloud', methods=['POST'])
def export_to_cloud():
    """
    Export model to cloud storage (S3)
    Note: This is a placeholder - actual S3 upload needs AWS credentials
    """
    try:
        # TODO: Implement S3 upload
        # For now, just return success
        
        return jsonify({
            'success': True,
            'message': 'Model export to cloud not yet implemented',
            'url': None
        }), 200
        
    except Exception as e:
        print(f"❌ Cloud export error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download_model', methods=['GET'])
def download_model():
    """
    Download trained model as a zip file
    Includes encoder, embeddings, and metadata
    """
    try:
        from training import MODEL_DIR, ENCODER_PATH, EMBEDDINGS_PATH, METADATA_PATH
        
        # Create zip file in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add encoder model
            if os.path.exists(ENCODER_PATH):
                zipf.write(ENCODER_PATH, 'signature_encoder.h5')
            
            # Add embeddings
            if os.path.exists(EMBEDDINGS_PATH):
                zipf.write(EMBEDDINGS_PATH, 'embeddings.pkl')
            
            # Add metadata
            if os.path.exists(METADATA_PATH):
                zipf.write(METADATA_PATH, 'metadata.json')
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'siamese_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import tensorflow as tf
    
    print("\n" + "="*70)
    print("🚀 SIAMESE SIGNATURE IDENTIFICATION API")
    print("="*70)
    print(f"📦 TensorFlow version: {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("⚠️  No GPU detected - using CPU")
    
    print(f"\n📂 Model storage: {os.path.abspath('model_storage')}")
    
    # Load existing model if available
    print("\n🔄 Initializing model...")
    trainer = get_trainer()
    status = trainer.get_status()
    
    if status['is_trained']:
        print(f"✅ Model loaded with {status['total_students']} students")
        print(f"   Total embeddings: {status['total_embeddings']}")
        print(f"   Last updated: {status['last_updated']}")
    else:
        print("🆕 No trained model found - ready for first training")
    
    print("\n" + "="*70)
    print("🌐 Starting Flask server...")
    print("="*70)
    print("\n💡 API Endpoints:")
    print("   GET  /health              - Health check")
    print("   GET  /status              - Model status")
    print("   POST /train_batch         - Batch train students")
    print("   GET  /training_status     - Training progress")
    print("   POST /classify            - Identify signature owner")
    print("   POST /verify              - Verify signature (1:1)")
    print("   POST /delete_student      - Delete student")
    print("   GET  /list_students       - List all students")
    print("   POST /export_cloud        - Export to cloud")
    print("   GET  /download_model      - Download model")
    print("\n" + "="*70 + "\n")
    
    # Run Flask app
    # Note: For Cloudflare tunnel, we use 0.0.0.0:5000
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
