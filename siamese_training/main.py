# filepath: siamese_training/main.py
"""
Flask API Server for Siamese Signature Identification
Optimized for Cloudflare Tunnel deployment
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time
import traceback
import os
import zipfile
import io
from datetime import datetime

from model import SiameseModelWrapper
from training import IncrementalTrainer
from classification import SignatureClassifier

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Allow all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize model components
print("🚀 Initializing Siamese Network...")
model_wrapper = SiameseModelWrapper(input_shape=(224, 224, 1))
trainer = IncrementalTrainer(model_wrapper, save_dir='models')
classifier = SignatureClassifier(model_wrapper, trainer)

# Training state management
training_state = {
    'is_training': False,
    'progress': 0.0,
    'current_student': None,
    'total_students': 0,
    'completed_students': 0,
    'error': None,
    'started_at': None,
    'completed_at': None
}

training_lock = threading.Lock()

def background_training_worker(students_data):
    """
    Background worker for training (runs in separate thread).
    """
    global training_state
    
    try:
        with training_lock:
            training_state['is_training'] = True
            training_state['progress'] = 0.0
            training_state['current_student'] = None
            training_state['total_students'] = len(students_data)
            training_state['completed_students'] = 0
            training_state['error'] = None
            training_state['started_at'] = datetime.now().isoformat()
            training_state['completed_at'] = None
        
        def progress_callback(progress, current_student=None, stage='training', **kwargs):
            """Update training progress."""
            with training_lock:
                training_state['progress'] = progress
                if current_student:
                    training_state['current_student'] = current_student
                if stage == 'training' and 'epoch' in kwargs:
                    completed = int((progress - 30) / 50 * len(students_data))
                    training_state['completed_students'] = max(0, min(completed, len(students_data)))
        
        # Train model
        result = trainer.train_batch(
            students_data=students_data,
            epochs=50,
            batch_size=32,
            progress_callback=progress_callback
        )
        
        with training_lock:
            training_state['is_training'] = False
            training_state['progress'] = 100.0
            training_state['completed_students'] = training_state['total_students']
            training_state['completed_at'] = datetime.now().isoformat()
            
        print(f"✅ Background training completed: {result}")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        
        with training_lock:
            training_state['is_training'] = False
            training_state['error'] = str(e)
            training_state['completed_at'] = datetime.now().isoformat()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'online',
        'model_trained': len(trainer.embeddings_db) > 0,
        'total_students': len(trainer.embeddings_db),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/train/batch', methods=['POST'])
def train_batch():
    """
    Train model with batch of students (background training).
    
    Request body:
    {
        "students": [
            {
                "studentId": "2021-00001",
                "genuineSamples": [
                    {"thumbnail": "data:image/jpeg;base64,...", "timestamp": 123456789},
                    ...
                ]
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({'error': 'Missing students data'}), 400
        
        students_data = data['students']
        
        if len(students_data) == 0:
            return jsonify({'error': 'No students provided'}), 400
        
        # Check if already training
        with training_lock:
            if training_state['is_training']:
                return jsonify({
                    'error': 'Training already in progress',
                    'progress': training_state['progress']
                }), 409
        
        # Start background training
        training_thread = threading.Thread(
            target=background_training_worker,
            args=(students_data,),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'message': 'Training started in background',
            'total_students': len(students_data),
            'status': 'training'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    with training_lock:
        return jsonify(training_state)


@app.route('/api/classify', methods=['POST'])
def classify_signature():
    """
    Classify/identify signature owner (1:N identification).
    
    Request body:
    {
        "image": "data:image/jpeg;base64,..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        image_base64 = data['image']
        
        # Classify
        result = classifier.classify(image_base64)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_signature():
    """
    Verify if signature belongs to claimed student (1:1 verification).
    
    Request body:
    {
        "image": "data:image/jpeg;base64,...",
        "studentId": "2021-00001"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'studentId' not in data:
            return jsonify({'error': 'Missing image or studentId'}), 400
        
        image_base64 = data['image']
        student_id = data['studentId']
        
        # Verify
        result = classifier.verify(image_base64, student_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/students', methods=['GET'])
def get_students():
    """Get list of all trained students."""
    try:
        result = classifier.get_all_students()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/download', methods=['GET'])
def download_model():
    """Download model as ZIP file."""
    try:
        # Create in-memory ZIP file
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model weights
            if os.path.exists(trainer.weights_path):
                zf.write(trainer.weights_path, 'model_weights.weights.h5')
            
            # Add embeddings
            if os.path.exists(trainer.embeddings_path):
                zf.write(trainer.embeddings_path, 'embeddings.pkl')
            
            # Add metadata
            if os.path.exists(trainer.metadata_path):
                zf.write(trainer.metadata_path, 'metadata.json')
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'siamese_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/export', methods=['POST'])
def export_to_cloud():
    """
    Export model to cloud storage.
    Note: Implement your cloud storage logic here (S3, GCS, etc.)
    """
    try:
        # TODO: Implement cloud storage upload
        # For now, just return success if files exist
        
        if not os.path.exists(trainer.weights_path):
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Placeholder for cloud upload
        # upload_to_s3(trainer.weights_path)
        # upload_to_s3(trainer.embeddings_path)
        # upload_to_s3(trainer.metadata_path)
        
        return jsonify({
            'message': 'Model exported successfully (local storage only)',
            'note': 'Cloud storage integration not implemented yet'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 Siamese Signature Recognition API Server")
    print("="*60)
    print(f"📊 Loaded {len(trainer.embeddings_db)} trained students")
    print("\n🌐 Starting Flask server on port 5000...")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )