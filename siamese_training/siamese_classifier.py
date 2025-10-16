"""
google drive filepath: siamese_training/siamese_classifier.py
Real-time 1:N Signature Classification using FAISS
Identifies signature owner automatically WITHOUT selecting student first
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import json
import faiss
from typing import List, Dict, Tuple, Optional
import time

from signature_preprocessing import SignaturePreprocessor

class SiameseSignatureClassifier:
    """
    Fast 1:N signature classification using FAISS
    Can identify signature owner in real-time from database of all trained students
    """
    
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.img_size = (224, 224)
        self.preprocessor = SignaturePreprocessor(target_size=self.img_size)
        
        # FAISS index for fast nearest neighbor search
        self.index = None
        self.student_id_map = []  # Maps index position to student_id
        self.student_metadata = {}  # Store metadata for each student
        
        # Confidence thresholds
        self.identification_threshold = 0.55  # Distance threshold for "known" vs "unknown"
        self.confidence_threshold = 0.65  # Minimum confidence to show result
        
        print("🔍 Signature Classifier initialized")
    
    def build_classification_database(self, rebuild=False):
        """
        Build FAISS index from all trained student models
        This creates a searchable database of all signature embeddings
        
        Args:
            rebuild: Force rebuild even if index exists
        """
        print("\n" + "="*60)
        print("BUILDING CLASSIFICATION DATABASE")
        print("="*60)
        
        index_path = self.base_dir / 'classification_index.faiss'
        metadata_path = self.base_dir / 'classification_metadata.json'
        
        # Load existing if available and not forcing rebuild
        if not rebuild and index_path.exists() and metadata_path.exists():
            print("📂 Loading existing classification database...")
            try:
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.student_id_map = data['student_id_map']
                    self.student_metadata = data['student_metadata']
                
                print(f"✅ Loaded database with {len(self.student_id_map)} students")
                print(f"   Total embeddings: {self.index.ntotal}")
                print("="*60 + "\n")
                return
            except Exception as e:
                print(f"⚠️  Failed to load existing database: {e}")
                print("   Rebuilding from scratch...")
        
        # Build from scratch
        print("🔨 Building new classification database...")
        
        all_embeddings = []
        self.student_id_map = []
        self.student_metadata = {}
        
        # Scan all student directories
        student_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        if len(student_dirs) == 0:
            print("⚠️  No trained models found!")
            print("="*60 + "\n")
            return
        
        print(f"Found {len(student_dirs)} trained students")
        
        for student_dir in student_dirs:
            student_id = student_dir.name
            
            # Check for reference embeddings
            ref_path = student_dir / 'reference_genuine.npy'
            metadata_file = student_dir / 'metadata.json'
            
            if not ref_path.exists():
                print(f"⚠️  Skipping {student_id}: No reference embeddings")
                continue
            
            try:
                # Load embeddings
                embeddings = np.load(ref_path)
                
                # Load metadata
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Add to database
                for emb in embeddings:
                    all_embeddings.append(emb)
                    self.student_id_map.append(student_id)
                
                self.student_metadata[student_id] = {
                    'num_embeddings': len(embeddings),
                    'training_date': metadata.get('training_date', 'unknown'),
                    'accuracy': metadata.get('final_accuracy', 0),
                    'preprocessing': metadata.get('preprocessing', 'legacy')
                }
                
                print(f"✓ Added {student_id}: {len(embeddings)} embeddings")
                
            except Exception as e:
                print(f"⚠️  Error loading {student_id}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            print("❌ No valid embeddings found!")
            print("="*60 + "\n")
            return
        
        # Convert to numpy array
        all_embeddings = np.array(all_embeddings, dtype='float32')
        embedding_dim = all_embeddings.shape[1]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total students: {len(set(self.student_id_map))}")
        print(f"   Total embeddings: {len(all_embeddings)}")
        print(f"   Embedding dimension: {embedding_dim}")
        
        # Create FAISS index
        # Using L2 (Euclidean distance) - same as Siamese training
        print(f"\n🔧 Creating FAISS index...")
        
        # For T4 GPU with 15GB: Use flat index (exact search, no approximation)
        # If you have more students (>10k), consider IndexIVFFlat for speed
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Add all embeddings
        self.index.add(all_embeddings)
        
        print(f"✅ FAISS index created with {self.index.ntotal} vectors")
        
        # Save to disk
        print(f"\n💾 Saving classification database...")
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, 'w') as f:
            json.dump({
                'student_id_map': self.student_id_map,
                'student_metadata': self.student_metadata,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_students': len(set(self.student_id_map)),
                'total_embeddings': len(all_embeddings)
            }, f, indent=2)
        
        print(f"✅ Database saved to: {self.base_dir}")
        print("="*60 + "\n")
    
    def classify_signature(self, image_path_or_array, top_k=3) -> Dict:
        """
        Identify signature owner (1:N classification)
        
        Args:
            image_path_or_array: Path to signature image or numpy array
            top_k: Return top K most similar students
        
        Returns:
            {
                'identified': bool,
                'student_id': str or None,
                'confidence': float,
                'top_matches': [
                    {'student_id': str, 'distance': float, 'confidence': float},
                    ...
                ]
            }
        """
        if self.index is None or len(self.student_id_map) == 0:
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'top_matches': [],
                'error': 'Classification database not built. Please train models first.'
            }
        
        print(f"\n{'='*60}")
        print("SIGNATURE CLASSIFICATION (1:N)")
        print(f"{'='*60}")
        
        try:
            # Preprocess image
            if isinstance(image_path_or_array, (str, Path)):
                img = cv2.imread(str(image_path_or_array))
                if img is None:
                    raise ValueError(f"Failed to load image: {image_path_or_array}")
                test_img = self.preprocessor.preprocess_for_verification(img)
            else:
                # Assume it's already preprocessed
                test_img = image_path_or_array
            
            # Load a feature extractor (use any trained model)
            # They all share the same architecture
            first_student = list(self.student_metadata.keys())[0]
            feature_extractor_path = self.base_dir / first_student / 'feature_extractor.keras'
            
            if not feature_extractor_path.exists():
                raise FileNotFoundError("No feature extractor found")
            
            feature_extractor = keras.models.load_model(feature_extractor_path)
            
            # Generate embedding
            test_embedding = feature_extractor.predict(
                np.expand_dims(test_img, axis=0),
                verbose=0
            )[0]
            
            # L2 normalize
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            test_embedding = test_embedding.astype('float32').reshape(1, -1)
            
            # Search in FAISS index
            print(f"🔍 Searching among {len(set(self.student_id_map))} students...")
            
            distances, indices = self.index.search(test_embedding, top_k * 5)  # Get more for grouping
            
            distances = distances[0]
            indices = indices[0]
            
            # Group by student_id and get best match per student
            student_best_matches = {}
            for dist, idx in zip(distances, indices):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                student_id = self.student_id_map[idx]
                
                if student_id not in student_best_matches:
                    student_best_matches[student_id] = dist
                else:
                    # Keep the minimum distance for this student
                    student_best_matches[student_id] = min(student_best_matches[student_id], dist)
            
            # Sort by distance
            sorted_matches = sorted(student_best_matches.items(), key=lambda x: x[1])
            
            # Convert to confidence scores
            top_matches = []
            for student_id, distance in sorted_matches[:top_k]:
                # Convert distance to confidence (0-1 scale)
                # Lower distance = higher confidence
                confidence = max(0, 1 - (distance / self.identification_threshold))
                
                top_matches.append({
                    'student_id': student_id,
                    'distance': float(distance),
                    'confidence': float(confidence),
                    'metadata': self.student_metadata.get(student_id, {})
                })
            
            # Determine if signature is identified
            if len(top_matches) > 0:
                best_match = top_matches[0]
                
                identified = (
                    best_match['distance'] < self.identification_threshold and
                    best_match['confidence'] > self.confidence_threshold
                )
                
                if identified:
                    print(f"\n✅ IDENTIFIED: {best_match['student_id']}")
                    print(f"   Distance: {best_match['distance']:.4f}")
                    print(f"   Confidence: {best_match['confidence']*100:.1f}%")
                else:
                    print(f"\n❓ UNKNOWN SIGNATURE")
                    print(f"   Best match: {best_match['student_id']}")
                    print(f"   Distance: {best_match['distance']:.4f} (threshold: {self.identification_threshold:.4f})")
                    print(f"   Confidence: {best_match['confidence']*100:.1f}% (threshold: {self.confidence_threshold*100:.1f}%)")
                
                print(f"\n📊 Top {len(top_matches)} matches:")
                for i, match in enumerate(top_matches, 1):
                    print(f"   {i}. {match['student_id']}: {match['confidence']*100:.1f}% (dist: {match['distance']:.4f})")
                
                result = {
                    'identified': identified,
                    'student_id': best_match['student_id'] if identified else None,
                    'confidence': best_match['confidence'],
                    'distance': best_match['distance'],
                    'top_matches': top_matches
                }
            else:
                print(f"\n❓ UNKNOWN SIGNATURE")
                print("   No matches found in database")
                
                result = {
                    'identified': False,
                    'student_id': None,
                    'confidence': 0.0,
                    'distance': float('inf'),
                    'top_matches': []
                }
            
            print(f"{'='*60}\n")
            
            # Cleanup
            del feature_extractor
            tf.keras.backend.clear_session()
            
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'top_matches': [],
                'error': str(e)
            }
    
    def realtime_classify_frame(self, frame: np.ndarray) -> Dict:
        """
        Real-time classification for webcam frames
        Optimized for speed (no file I/O)
        
        Args:
            frame: OpenCV image (BGR)
        
        Returns:
            Same format as classify_signature()
        """
        if self.index is None:
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'top_matches': [],
                'error': 'Database not initialized'
            }
        
        try:
            # Preprocess frame
            processed = self.preprocessor.preprocess_for_verification(frame)
            
            # Classify
            return self.classify_signature(processed, top_k=3)
            
        except Exception as e:
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'top_matches': [],
                'error': str(e)
            }
    
    def update_database_for_student(self, student_id: str):
        """
        Update classification database when a student's model is updated
        (For incremental learning support)
        
        Args:
            student_id: Student whose model was updated
        """
        print(f"\n🔄 Updating classification database for {student_id}...")
        
        # For now, just rebuild the entire database
        # TODO: Optimize to only update specific student's embeddings
        self.build_classification_database(rebuild=True)
        
        print(f"✅ Database updated for {student_id}")


# Standalone testing
if __name__ == "__main__":
    classifier = SiameseSignatureClassifier(base_dir='models')
    
    # Build database
    classifier.build_classification_database()
    
    # Test classification
    test_image = 'test_signature.jpg'
    if Path(test_image).exists():
        result = classifier.classify_signature(test_image, top_k=5)
        print(f"\nResult: {result}")