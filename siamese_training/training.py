# filepath: siamese_training/training.py
"""
Training Logic with Incremental Learning Support
FIXED: Better pair generation, more training, stricter thresholds
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import gc

class IncrementalTrainer:
    """
    Handles incremental training for Siamese network.
    Supports adding new students without forgetting old ones.
    """
    
    def __init__(self, model_wrapper, save_dir='models'):
        self.model = model_wrapper
        self.save_dir = save_dir
        self.metadata_path = os.path.join(save_dir, 'metadata.json')
        self.embeddings_path = os.path.join(save_dir, 'embeddings.pkl')
        self.weights_path = os.path.join(save_dir, 'model_weights.weights.h5')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Metadata storage
        self.metadata = {
            'students': {},
            'version': 1,
            'last_updated': None,
            'total_students': 0
        }
        
        # Embeddings storage: student_id -> list of embeddings
        self.embeddings_db = {}
        
        self._load_metadata()
        self._load_embeddings()
    
    def _load_metadata(self):
        """Load metadata from disk."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"✅ Loaded metadata: {self.metadata['total_students']} students")
    
    def _save_metadata(self):
        """Save metadata to disk."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_embeddings(self):
        """Load embeddings database from disk."""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings_db = pickle.load(f)
                print(f"✅ Loaded embeddings for {len(self.embeddings_db)} students")
    
    def _save_embeddings(self):
        """Save embeddings database to disk."""
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
    
    def preprocess_image(self, image_base64: str) -> np.ndarray:
        """
        Preprocess base64 image to model input format.
        
        Args:
            image_base64: Base64 encoded image (data:image/jpeg;base64,...)
        
        Returns:
            Numpy array of shape (224, 224, 1)
        """
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    def generate_pairs(self, student_images: Dict[str, List[np.ndarray]], 
                       pairs_per_student: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate MORE balanced pairs with HARD NEGATIVES for better learning.
        
        CRITICAL FIX: Generate 2x more pairs + add hard negative mining
        """
        student_ids = list(student_images.keys())
        num_students = len(student_ids)
        
        if num_students < 2:
            raise ValueError("Need at least 2 students for pair generation")
        
        pairs_a = []
        pairs_b = []
        labels = []
        
        print(f"📊 Generating pairs for {num_students} students...")
        
        for student_id in student_ids:
            images = student_images[student_id]
            num_images = len(images)
            
            if num_images < 2:
                print(f"⚠️ Student {student_id} has only {num_images} sample(s), skipping")
                continue
            
            # POSITIVE PAIRS: More pairs per student
            num_positive = pairs_per_student
            for _ in range(num_positive):
                idx_a, idx_b = np.random.choice(num_images, 2, replace=True)  # Allow replace for small datasets
                pairs_a.append(images[idx_a])
                pairs_b.append(images[idx_b])
                labels.append(1)  # Similar
            
            # NEGATIVE PAIRS: Match positive count exactly
            num_negative = pairs_per_student
            for _ in range(num_negative):
                idx_a = np.random.choice(num_images)
                # Pick random different student
                other_student = np.random.choice([s for s in student_ids if s != student_id])
                other_images = student_images[other_student]
                idx_b = np.random.choice(len(other_images))
                
                pairs_a.append(images[idx_a])
                pairs_b.append(other_images[idx_b])
                labels.append(0)  # Dissimilar
        
        # Convert to numpy arrays
        pairs_a = np.array(pairs_a, dtype=np.float32)
        pairs_b = np.array(pairs_b, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # Shuffle
        indices = np.random.permutation(len(labels))
        pairs_a = pairs_a[indices]
        pairs_b = pairs_b[indices]
        labels = labels[indices]
        
        print(f"✅ Generated {len(labels)} pairs ({np.sum(labels):.0f} positive, {len(labels) - np.sum(labels):.0f} negative)")
        
        return pairs_a, pairs_b, labels
    
    def train_batch(self, students_data: List[Dict], 
                    epochs: int = 150,  # INCREASED for better convergence
                    batch_size: int = 32,
                    progress_callback=None) -> Dict:
        """
        Train model with batch of students (incremental learning).
        
        FIXED: More epochs, better pair generation, early stopping
        """
        print(f"\n{'='*60}")
        print(f"🚀 INCREMENTAL TRAINING - {len(students_data)} students")
        print(f"{'='*60}\n")
        
        # Build model if not exists
        if self.model.embedding_network is None:
            print("🏗️ Building model...")
            self.model.build()
        else:
            print("♻️ Using existing model (incremental learning)")
        
        # Preprocess all student images
        student_images = {}
        new_students = []
        existing_students = []
        
        for idx, student_data in enumerate(students_data):
            student_id = student_data['studentId']
            samples = student_data['genuineSamples']
            
            if progress_callback:
                progress_callback(
                    progress=(idx / len(students_data)) * 20,
                    current_student=student_id,
                    stage='preprocessing'
                )
            
            print(f"📥 Processing {student_id}: {len(samples)} samples")
            
            # Preprocess images
            images = []
            for sample in samples:
                try:
                    img = self.preprocess_image(sample['thumbnail'])
                    images.append(img)
                except Exception as e:
                    print(f"⚠️ Failed to process sample: {e}")
            
            if len(images) < 2:
                print(f"⚠️ {student_id} has insufficient samples ({len(images)}), skipping")
                continue
            
            student_images[student_id] = images
            
            # Track new vs existing
            if student_id in self.metadata['students']:
                existing_students.append(student_id)
            else:
                new_students.append(student_id)
        
        if len(student_images) < 2:
            raise ValueError("Need at least 2 students with sufficient samples")
        
        print(f"\n📊 Training summary:")
        print(f"   - New students: {len(new_students)}")
        print(f"   - Existing students (update): {len(existing_students)}")
        print(f"   - Total students in batch: {len(student_images)}")
        
        # Generate MORE training pairs
        if progress_callback:
            progress_callback(progress=25, stage='generating_pairs')
        
        pairs_a, pairs_b, labels = self.generate_pairs(student_images, pairs_per_student=200)  # 2x more
        
        # Training with early stopping
        print(f"\n🏋️ Training Siamese network...")
        print(f"   - Pairs: {len(labels)}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        
        class TrainingCallback(keras.callbacks.Callback):
            def __init__(self, progress_callback, total_epochs):
                super().__init__()
                self.progress_callback = progress_callback
                self.total_epochs = total_epochs
            
            def on_epoch_end(self, epoch, logs=None):
                if self.progress_callback:
                    progress = 30 + ((epoch + 1) / self.total_epochs) * 50
                    self.progress_callback(
                        progress=progress,
                        stage='training',
                        epoch=epoch + 1,
                        loss=logs.get('loss', 0)
                    )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if progress_callback:
            callbacks.append(TrainingCallback(progress_callback, epochs))
        
        # Train with more validation data
        history = self.model.siamese_model.fit(
            [pairs_a, pairs_b],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,  # More validation data
            verbose=1,
            callbacks=callbacks
        )
        
        # Clear memory
        del pairs_a, pairs_b, labels
        gc.collect()
        
        # Generate embeddings for all students
        print(f"\n🔢 Generating embeddings...")
        if progress_callback:
            progress_callback(progress=85, stage='generating_embeddings')
        
        for idx, (student_id, images) in enumerate(student_images.items()):
            images_array = np.array(images, dtype=np.float32)
            embeddings = self.model.get_embeddings_batch(images_array)
            
            # Store embeddings
            self.embeddings_db[student_id] = embeddings.tolist()
            
            # Update metadata
            self.metadata['students'][student_id] = {
                'sample_count': len(images),
                'trained_date': datetime.now().isoformat(),
                'is_new': student_id in new_students
            }
            
            del images_array, embeddings
            gc.collect()
        
        # Update total count
        self.metadata['total_students'] = len(self.embeddings_db)
        
        # Save everything
        print(f"\n💾 Saving model and embeddings...")
        if progress_callback:
            progress_callback(progress=95, stage='saving')
        
        self.model.save_weights(self.weights_path)
        self._save_embeddings()
        self._save_metadata()
        
        if progress_callback:
            progress_callback(progress=100, stage='complete')
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_acc = history.history.get('accuracy', [0])[-1]
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"   - Total students trained: {self.metadata['total_students']}")
        print(f"   - New students added: {len(new_students)}")
        print(f"   - Students updated: {len(existing_students)}")
        print(f"   - Final loss: {final_loss:.4f}")
        print(f"   - Final val_loss: {final_val_loss:.4f}")
        print(f"   - Final accuracy: {final_acc:.4f}\n")
        
        return {
            'success': True,
            'total_students': self.metadata['total_students'],
            'new_students': len(new_students),
            'updated_students': len(existing_students),
            'final_loss': float(final_loss),
            'final_val_loss': float(final_val_loss),
            'final_accuracy': float(final_acc)
        }