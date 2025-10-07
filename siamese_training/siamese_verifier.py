"""
google drive filepath: siamese_training/siamese_verifier.py
Siamese Network Verification Module - GPU-Optimized
Enhanced preprocessing matching training pipeline
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import json

class SiameseSignatureVerifier:
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.img_size = (224, 224)
        
    def preprocess_image(self, image_path):
        """
        Load and preprocess image - MUST MATCH TRAINING PREPROCESSING
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        
        img = cv2.resize(img, self.img_size)
        
        # Apply same preprocessing as training
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        img = img.astype('float32') / 255.0
        return img
    
    def verify_signature(self, student_id, test_image_path):
        """
        Verify signature using enhanced distance-based matching
        Uses stricter thresholds for higher precision
        """
        print(f"\n{'='*60}")
        print(f"Verifying Signature for Student: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        
        # Load feature extractor
        keras_model_path = student_dir / 'feature_extractor.keras'
        h5_model_path = student_dir / 'feature_extractor.h5'
        
        if keras_model_path.exists():
            feature_path = keras_model_path
        elif h5_model_path.exists():
            feature_path = h5_model_path
        else:
            raise FileNotFoundError(f"No trained model found for student {student_id}")
        
        reference_path = student_dir / 'reference_genuine.npy'
        metadata_path = student_dir / 'metadata.json'
        
        print(f"Loading model from: {feature_path}")
        feature_extractor = keras.models.load_model(feature_path)
        
        # Load metadata
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found for student {student_id}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        threshold = metadata.get('threshold', 0.45)
        
        # Load reference embeddings
        if not reference_path.exists():
            print("❌ Error: No reference embeddings found")
            return {
                'is_verified': False,
                'confidence': 0.0,
                'min_distance': 0.0,
                'average_distance': 0.0,
                'threshold_used': threshold,
                'student_id': student_id,
                'error': 'No reference embeddings available'
            }
        
        reference_embeddings = np.load(reference_path)
        
        # Preprocess test image with same pipeline as training
        test_img = self.preprocess_image(test_image_path)
        
        # Generate test embedding
        test_embedding = feature_extractor.predict(
            np.expand_dims(test_img, axis=0), 
            verbose=0
        )[0]
        
        # L2 normalize
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Calculate Euclidean distances to all references
        distances = []
        for ref_emb in reference_embeddings:
            distance = np.linalg.norm(test_embedding - ref_emb)
            distances.append(distance)
        
        # Statistics
        min_distance = float(np.min(distances))
        avg_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))
        std_distance = float(np.std(distances))
        
        # Enhanced decision logic
        # 1. Minimum distance must be below threshold
        # 2. Average distance should also be reasonable
        # 3. Standard deviation should be low (consistent matching)
        
        is_verified = (
            min_distance < threshold and 
            avg_distance < (threshold * 1.3) and
            std_distance < 0.3
        )
        
        # Enhanced confidence calculation
        # Consider multiple factors
        distance_confidence = max(0, 1 - (min_distance / threshold))
        avg_confidence = max(0, 1 - (avg_distance / (threshold * 1.3)))
        consistency_confidence = max(0, 1 - (std_distance / 0.3))
        
        # Weighted combination
        confidence = (
            distance_confidence * 0.5 + 
            avg_confidence * 0.3 + 
            consistency_confidence * 0.2
        )
        
        print(f"\n📊 Verification Metrics:")
        print(f"  Min Distance: {min_distance:.4f}")
        print(f"  Avg Distance: {avg_distance:.4f}")
        print(f"  Max Distance: {max_distance:.4f}")
        print(f"  Std Distance: {std_distance:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  References: {len(distances)} samples")
        print(f"\n🎯 Decision: {'✅ VERIFIED' if is_verified else '❌ NOT VERIFIED'}")
        print(f"  Confidence: {confidence*100:.1f}%")
        
        # Enhanced warnings and diagnostics
        if is_verified:
            if min_distance > (threshold * 0.85):
                print("  ⚠️  Borderline match - close to threshold")
            elif min_distance < (threshold * 0.3):
                print("  ✅ Strong match - very close to references")
            else:
                print("  ✅ Good match - within acceptable range")
        else:
            if min_distance < (threshold * 1.15):
                print("  ⚠️  Borderline rejection - consider manual review")
            elif avg_distance > 1.0:
                print("  ❌ Significant difference from training samples")
            elif std_distance > 0.4:
                print("  ❌ Inconsistent matching across references")
        
        # Additional diagnostics
        if avg_distance < 0.25:
            print("  ℹ️  Very close match to all references")
        
        # Precision/Recall info from training
        if 'precision' in metadata and 'recall' in metadata:
            print(f"\n📈 Model Performance (from training):")
            print(f"  Precision: {metadata['precision']*100:.1f}%")
            print(f"  Recall: {metadata['recall']*100:.1f}%")
            print(f"  F1 Score: {metadata.get('f1_score', 0)*100:.1f}%")
        
        print(f"{'='*60}\n")
        
        return {
            'is_verified': bool(is_verified),
            'confidence': float(confidence),
            'min_distance': float(min_distance),
            'average_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'std_distance': float(std_distance),
            'threshold_used': float(threshold),
            'student_id': student_id,
            'num_references': len(distances),
            'model_accuracy': metadata.get('final_accuracy', 0),
            'model_precision': metadata.get('precision', 0),
            'model_recall': metadata.get('recall', 0)
        }
    
    def check_model_exists(self, student_id):
        """Check if trained model exists for student"""
        student_dir = self.base_dir / student_id
        
        keras_model = student_dir / 'siamese_model.keras'
        h5_model = student_dir / 'siamese_model.h5'
        metadata_path = student_dir / 'metadata.json'
        
        model_exists = keras_model.exists() or h5_model.exists()
        
        if model_exists and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return True, metadata
        
        return False, None