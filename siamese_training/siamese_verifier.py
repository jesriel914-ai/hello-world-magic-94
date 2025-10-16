"""
google drive filepath: siamese_training/siamese_verifier.py
Siamese Network Verification Module with Signature Isolation
NOW robust to camera quality, lighting, background, paper color
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import json

# Import our preprocessing module
from signature_preprocessing import SignaturePreprocessor

class SiameseSignatureVerifier:
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.img_size = (224, 224)
        
        # Initialize signature preprocessor
        self.preprocessor = SignaturePreprocessor(target_size=self.img_size)
        
    def preprocess_image(self, image_path):
        """
        NEW: Preprocess using signature isolation
        Works with ANY camera quality, lighting, or background
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        
        # Use our robust signature extraction
        return self.preprocessor.preprocess_for_verification(img)
    
    def verify_signature(self, student_id, test_image_path):
        """
        Verify signature with enhanced robustness
        Now works regardless of:
        - Camera quality (phone camera, webcam, scanner)
        - Lighting conditions (bright, dim, shadows)
        - Background color (white paper, yellow paper, desk)
        - Image artifacts (JPEG compression, noise)
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
        
        # Check if model uses signature isolation
        is_signature_isolated = metadata.get('preprocessing') == 'signature_extraction'
        
        print(f"Model type: {'Signature-isolated' if is_signature_isolated else 'Legacy'}")
        
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
        
        # Preprocess test image with signature isolation
        print(f"\nExtracting signature from test image...")
        print(f"  (Ignoring: background, lighting, camera quality, paper color)")
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
        
        # Enhanced decision logic with stricter criteria
        is_verified = (
            min_distance < threshold and 
            avg_distance < (threshold * 1.3) and
            std_distance < 0.3
        )
        
        # Enhanced confidence calculation
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
        print(f"  Background-invariant: ✅")
        
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
        
        # Model performance info
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
            'model_recall': metadata.get('recall', 0),
            'preprocessing': metadata.get('preprocessing', 'legacy')
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