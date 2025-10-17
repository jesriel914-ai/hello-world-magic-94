"""
google drive filepath: siamese_training/siamese_incremental_trainer.py
Incremental Learning for Siamese Signature Verification
Add new samples WITHOUT retraining from scratch
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import cv2
from pathlib import Path
import gc
from typing import List, Dict, Optional

from signature_preprocessing import SignaturePreprocessor

class SiameseIncrementalTrainer:
    """
    Incremental learning for Siamese models
    
    Strategy:
    1. Reuse existing feature extractor (no retraining backbone)
    2. Generate embeddings for new samples
    3. Merge with existing reference embeddings
    4. Optionally fine-tune if new samples show different characteristics
    """
    
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.img_size = (224, 224)
        self.preprocessor = SignaturePreprocessor(target_size=self.img_size)
        
        print("🔄 Incremental Learning Trainer initialized")
    
    def add_new_genuine_samples(
        self,
        student_id: str,
        new_genuine_samples: List[str],
        update_threshold: bool = True
    ) -> Dict:
        """
        Add new GENUINE samples to existing student model
        WITHOUT retraining the entire model
        
        Args:
            student_id: Student identifier
            new_genuine_samples: List of image paths for new genuine samples
            update_threshold: Whether to recalculate verification threshold
        
        Returns:
            Updated metadata
        """
        print(f"\n{'='*60}")
        print(f"INCREMENTAL LEARNING: Adding Genuine Samples")
        print(f"Student: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        
        if not student_dir.exists():
            return {
                'needs_retraining': True,
                'reason': 'No existing model found',
                'recommendation': 'Train new model from scratch'
            }
        
        # Load metadata
        metadata_path = student_dir / 'metadata.json'
        if not metadata_path.exists():
            return {
                'needs_retraining': True,
                'reason': 'No metadata found',
                'recommendation': 'Train new model'
            }
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        existing_samples = metadata.get('genuine_samples', 0)
        incremental_updates = metadata.get('incremental_updates', 0)
        
        # Decision rules
        reasons = []
        
        # Rule 1: Too many incremental updates (drift risk)
        if incremental_updates >= 5:
            reasons.append(f"Too many incremental updates ({incremental_updates})")
        
        # Rule 2: New samples represent >50% of total
        if new_sample_count > existing_samples:
            reasons.append(f"New samples ({new_sample_count}) exceed existing ({existing_samples})")
        
        # Rule 3: Old model uses legacy preprocessing
        if metadata.get('preprocessing') != 'signature_extraction':
            reasons.append("Model uses legacy preprocessing")
        
        # Rule 4: Low accuracy
        if metadata.get('final_accuracy', 1.0) < 0.85:
            reasons.append(f"Low model accuracy ({metadata.get('final_accuracy', 0)*100:.1f}%)")
        
        needs_retraining = len(reasons) > 0
        
        if needs_retraining:
            return {
                'needs_retraining': True,
                'reason': '; '.join(reasons),
                'recommendation': 'Full retraining recommended for optimal performance'
            }
        else:
            return {
                'needs_retraining': False,
                'reason': 'Model is suitable for incremental update',
                'recommendation': 'Proceed with incremental learning'
            }


# Standalone testing
if __name__ == "__main__":
    trainer = SiameseIncrementalTrainer(base_dir='models')
    
    # Example: Add new genuine samples
    student_id = "2025001"
    new_samples = ["new_signature_1.jpg", "new_signature_2.jpg"]
    
    # Check if retraining is needed
    check = trainer.check_if_retraining_needed(student_id, len(new_samples))
    print(f"Retraining check: {check}")
    
    if not check['needs_retraining']:
        # Proceed with incremental learning
        metadata = trainer.add_new_genuine_samples(student_id, new_samples)
        print(f"Updated metadata: {metadata}")
    else:
        print(f"Recommendation: {check['recommendation']}")
        
        if not student_dir.exists():
            raise FileNotFoundError(f"No trained model found for {student_id}")
        
        # Load existing data
        print("📂 Loading existing model...")
        
        feature_extractor_path = student_dir / 'feature_extractor.keras'
        reference_path = student_dir / 'reference_genuine.npy'
        metadata_path = student_dir / 'metadata.json'
        
        if not feature_extractor_path.exists():
            raise FileNotFoundError(f"Feature extractor not found for {student_id}")
        
        # Load feature extractor
        feature_extractor = keras.models.load_model(feature_extractor_path)
        
        # Load existing embeddings
        existing_embeddings = None
        if reference_path.exists():
            existing_embeddings = np.load(reference_path)
            print(f"✓ Loaded {len(existing_embeddings)} existing embeddings")
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Process new samples
        print(f"\n🆕 Processing {len(new_genuine_samples)} new genuine samples...")
        
        new_embeddings = []
        for i, img_path in enumerate(new_genuine_samples):
            try:
                # Preprocess with signature isolation
                img = self.preprocessor.preprocess_for_training(str(img_path))
                
                # Generate embedding
                embedding = feature_extractor.predict(
                    np.expand_dims(img, axis=0),
                    verbose=0
                )[0]
                
                # L2 normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                new_embeddings.append(embedding)
                
                if (i + 1) % 5 == 0:
                    print(f"   Processed {i + 1}/{len(new_genuine_samples)} samples...")
                
            except Exception as e:
                print(f"⚠️  Failed to process sample {i}: {e}")
                continue
        
        new_embeddings = np.array(new_embeddings)
        print(f"✓ Generated {len(new_embeddings)} new embeddings")
        
        # Merge with existing embeddings
        if existing_embeddings is not None:
            merged_embeddings = np.vstack([existing_embeddings, new_embeddings])
            print(f"\n✅ Merged embeddings:")
            print(f"   Old: {len(existing_embeddings)} embeddings")
            print(f"   New: {len(new_embeddings)} embeddings")
            print(f"   Total: {len(merged_embeddings)} embeddings")
        else:
            merged_embeddings = new_embeddings
            print(f"\n✅ Created initial embeddings: {len(merged_embeddings)}")
        
        # Optional: Remove outliers or oldest samples if too many
        # This prevents the reference set from growing too large
        max_references = 50  # Configurable
        if len(merged_embeddings) > max_references:
            print(f"\n⚠️  Too many references ({len(merged_embeddings)}), keeping best {max_references}...")
            merged_embeddings = self._select_representative_samples(
                merged_embeddings,
                max_samples=max_references
            )
            print(f"✓ Reduced to {len(merged_embeddings)} representative embeddings")
        
        # Save updated embeddings
        print(f"\n💾 Saving updated reference embeddings...")
        np.save(reference_path, merged_embeddings)
        
        # Update metadata
        old_sample_count = metadata.get('genuine_samples', 0)
        metadata['genuine_samples'] = old_sample_count + len(new_genuine_samples)
        metadata['total_reference_embeddings'] = len(merged_embeddings)
        metadata['last_incremental_update'] = datetime.now().isoformat()
        metadata['incremental_updates'] = metadata.get('incremental_updates', 0) + 1
        
        # Optionally update threshold based on new embeddings
        if update_threshold:
            new_threshold = self._calculate_optimal_threshold(
                merged_embeddings,
                feature_extractor,
                student_dir
            )
            if new_threshold is not None:
                metadata['threshold'] = new_threshold
                print(f"✓ Updated threshold: {new_threshold:.4f}")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata updated")
        print("="*60 + "\n")
        
        # Cleanup
        del feature_extractor
        gc.collect()
        tf.keras.backend.clear_session()
        
        return metadata
    
    def add_new_forged_samples(
        self,
        student_id: str,
        new_forged_samples: List[str],
        retrain_contrastive: bool = False
    ) -> Dict:
        """
        Add new FORGED samples to improve model's ability to reject forgeries
        
        Args:
            student_id: Student identifier
            new_forged_samples: List of image paths for new forged samples
            retrain_contrastive: Whether to fine-tune the model with new negatives
        
        Returns:
            Updated metadata
        """
        print(f"\n{'='*60}")
        print(f"INCREMENTAL LEARNING: Adding Forged Samples")
        print(f"Student: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        
        if not student_dir.exists():
            raise FileNotFoundError(f"No trained model found for {student_id}")
        
        # Load metadata
        metadata_path = student_dir / 'metadata.json'
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # For now, we'll just update the count
        # Full retraining with contrastive loss would require implementing fine-tuning
        old_forged_count = metadata.get('forged_samples', 0)
        metadata['forged_samples'] = old_forged_count + len(new_forged_samples)
        metadata['last_incremental_update'] = datetime.now().isoformat()
        
        if retrain_contrastive:
            print("⚠️  Contrastive fine-tuning not yet implemented")
            print("   For now, forged samples are just counted in metadata")
            print("   Consider full retraining if model performance degrades")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Updated forged sample count:")
        print(f"   Old: {old_forged_count}")
        print(f"   New: {len(new_forged_samples)}")
        print(f"   Total: {metadata['forged_samples']}")
        print("="*60 + "\n")
        
        return metadata
    
    def _select_representative_samples(
        self,
        embeddings: np.ndarray,
        max_samples: int = 50
    ) -> np.ndarray:
        """
        Select most representative embeddings using k-means clustering
        This prevents the reference set from growing infinitely
        
        Args:
            embeddings: All embeddings
            max_samples: Maximum number to keep
        
        Returns:
            Representative subset of embeddings
        """
        if len(embeddings) <= max_samples:
            return embeddings
        
        from sklearn.cluster import KMeans
        
        print(f"   Running k-means clustering (k={max_samples})...")
        
        kmeans = KMeans(n_clusters=max_samples, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        # Keep the cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Normalize centroids
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        return centroids
    
    def _calculate_optimal_threshold(
        self,
        genuine_embeddings: np.ndarray,
        feature_extractor: keras.Model,
        student_dir: Path
    ) -> Optional[float]:
        """
        Calculate optimal verification threshold based on genuine embeddings
        Uses intra-class distances (distance between genuine samples)
        
        Args:
            genuine_embeddings: Reference genuine embeddings
            feature_extractor: Feature extraction model
            student_dir: Student directory
        
        Returns:
            Optimal threshold or None if can't calculate
        """
        try:
            print(f"   Calculating optimal threshold...")
            
            # Calculate pairwise distances between genuine samples
            from scipy.spatial.distance import cdist
            
            distances = cdist(genuine_embeddings, genuine_embeddings, metric='euclidean')
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = distances[np.triu_indices_from(distances, k=1)]
            
            if len(upper_triangle) == 0:
                return None
            
            # Use statistics to set threshold
            mean_dist = np.mean(upper_triangle)
            std_dist = np.std(upper_triangle)
            max_dist = np.max(upper_triangle)
            
            # Threshold = mean + 2*std (captures ~95% of genuine variations)
            # But cap it at a reasonable value
            threshold = min(mean_dist + 2 * std_dist, 0.6)
            
            print(f"   Genuine distances: mean={mean_dist:.4f}, std={std_dist:.4f}, max={max_dist:.4f}")
            print(f"   Calculated threshold: {threshold:.4f}")
            
            return float(threshold)
            
        except Exception as e:
            print(f"⚠️  Failed to calculate threshold: {e}")
            return None
    
    def check_if_retraining_needed(
        self,
        student_id: str,
        new_sample_count: int
    ) -> Dict[str, any]:
        """
        Determine if full retraining is recommended instead of incremental update
        
        Args:
            student_id: Student identifier
            new_sample_count: Number of new samples to add
        
        Returns:
            {
                'needs_retraining': bool,
                'reason': str,
                'recommendation': str
            }
        """
        student_dir = self.base_dir / student_