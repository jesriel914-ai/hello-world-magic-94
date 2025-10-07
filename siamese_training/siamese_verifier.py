"""
Siamese Network Verification Module
Works with contrastive loss trained models
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
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        return img
    
    def verify_signature(self, student_id, test_image_path):
        """
        Verify signature using distance-based matching
        Lower distance = more similar
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
        
        threshold = metadata.get('threshold', 0.5)
        
        # Load reference embeddings
        if not reference_path.exists():
            print("❌ Error: No reference embeddings found")
            return {
                'is_verified': False,
                'confidence': 0.0,
                'average_distance': 0.0,
                'threshold_used': threshold,
                'student_id': student_id,
                'error': 'No reference embeddings available'
            }
        
        reference_embeddings = np.load(reference_path)
        
        # Preprocess test image
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
        
        # Use minimum distance (closest match)
        min_distance = float(np.min(distances))
        avg_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))
        
        # Decision: if distance < threshold, it's verified
        is_verified = min_distance < threshold
        
        # Convert distance to confidence score (0-1, higher = more confident)
        # Confidence inversely proportional to distance
        confidence = max(0, 1 - (min_distance / threshold))
        
        print(f"\n📊 Verification Metrics:")
        print(f"  Min Distance: {min_distance:.4f}")
        print(f"  Avg Distance: {avg_distance:.4f}")
        print(f"  Max Distance: {max_distance:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Comparisons: {len(distances)} reference signatures")
        print(f"\n🎯 Decision: {'✅ VERIFIED' if is_verified else '❌ NOT VERIFIED'}")
        print(f"  Confidence: {confidence*100:.1f}%")
        
        # Warnings
        if is_verified and min_distance > (threshold * 0.8):
            print("  ⚠️  Borderline match - distance close to threshold")
        elif not is_verified and min_distance < (threshold * 1.2):
            print("  ⚠️  Borderline rejection - consider manual review")
        
        if avg_distance < 0.3:
            print("  ℹ️  Very close match across all references")
        elif avg_distance > 1.0:
            print("  ℹ️  Significantly different from training samples")
        
        print(f"{'='*60}\n")
        
        return {
            'is_verified': bool(is_verified),
            'confidence': float(confidence),
            'min_distance': float(min_distance),
            'average_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'threshold_used': float(threshold),
            'student_id': student_id,
            'num_references': len(distances)
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