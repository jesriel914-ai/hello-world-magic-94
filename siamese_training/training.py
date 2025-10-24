# filepath: siamese_training/training.py
"""
Training Pipeline for Siamese Network
Supports incremental learning and batch training
"""

import os
import json
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras

from preprocessing import preprocess_batch, augment_signature
from model import SiameseNetwork
from data_generator import SiamesePairGenerator, ValidationPairGenerator

# Storage paths
MODEL_DIR = 'model_storage'
ENCODER_PATH = os.path.join(MODEL_DIR, 'signature_encoder.h5')
SIAMESE_PATH = os.path.join(MODEL_DIR, 'siamese_model.h5')
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'embeddings.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')

# Training configuration
BATCH_SIZE = 96  # Optimized for T4 GPU (12GB)
EPOCHS = 30
LEARNING_RATE = 5e-4  # Increased from 1e-4 to prevent accuracy collapse


class IncrementalTrainer:
    """
    Incremental Training Manager for Siamese Network
    Supports adding new students without retraining from scratch
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.model = None
        self.embeddings_db = {}  # {student_id: [embeddings]}
        self.metadata = {
            'students': {},
            'total_students': 0,
            'total_embeddings': 0,
            'last_updated': None,
            'training_history': []
        }
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Load existing model and data if available
        self.load_state()
    
    def load_state(self):
        """Load existing model, embeddings, and metadata"""
        try:
            # Load embeddings
            if os.path.exists(EMBEDDINGS_PATH):
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                print(f"✅ Loaded embeddings for {len(self.embeddings_db)} students")
            
            # Load metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata (last updated: {self.metadata.get('last_updated', 'Never')})")
            
            # Load model
            if os.path.exists(ENCODER_PATH):
                self.model = SiameseNetwork()
                self.model.load_model(ENCODER_PATH)
                print("✅ Loaded existing model")
            else:
                self.model = SiameseNetwork()
                print("🆕 Created new model")
                
        except Exception as e:
            print(f"⚠️  Error loading state: {e}")
            self.model = SiameseNetwork()
            print("🆕 Created new model")
    
    def save_state(self):
        """Save model, embeddings, and metadata"""
        try:
            # Save models
            self.model.save_model(ENCODER_PATH, SIAMESE_PATH)
            
            # Save embeddings
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(self.embeddings_db, f)
            
            # Update metadata
            self.metadata['total_students'] = len(self.embeddings_db)
            self.metadata['total_embeddings'] = sum(len(embs) for embs in self.embeddings_db.values())
            self.metadata['last_updated'] = datetime.now().isoformat()
            
            # Save metadata
            with open(METADATA_PATH, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print("✅ State saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving state: {e}")
            raise
    
    def detect_changes(self, new_students_data: Dict[str, Dict]) -> Dict:
        """
        Detect what needs to be trained
        
        Args:
            new_students_data: Dict of {student_id: {'genuine': [...], 'forged': [...]}}
        
        Returns:
            Dict with categorized students
        """
        existing_students = set(self.embeddings_db.keys())
        new_students_ids = set(new_students_data.keys())
        
        # Categorize students
        brand_new = new_students_ids - existing_students
        existing_updated = new_students_ids & existing_students
        unchanged = existing_students - new_students_ids
        
        changes = {
            'new_students': list(brand_new),
            'existing_updated': list(existing_updated),
            'unchanged': list(unchanged),
            'total_to_train': len(brand_new) + len(existing_updated)
        }
        
        print(f"\n📊 Training Changes Detected:")
        print(f"   🆕 New students: {len(changes['new_students'])}")
        print(f"   🔄 Updated students: {len(changes['existing_updated'])}")
        print(f"   ✅ Unchanged students: {len(changes['unchanged'])}")
        print(f"   📝 Total to train: {changes['total_to_train']}")
        
        return changes
    
    def prepare_training_data(self, students_data: Dict[str, Dict], 
                             use_augmentation: bool = False) -> Tuple[Dict, int]:
        """
        Prepare training data WITHOUT creating all pairs at once
        Returns processed images dict instead
        
        Args:
            students_data: Dict of {student_id: {'genuine': [...], 'forged': [...]}}
            use_augmentation: Whether to use data augmentation (DISABLED by default)
        
        Returns:
            processed_data: Dict of {student_id: preprocessed_images}
            estimated_pairs: Estimated number of pairs
        """
        print("\n🔄 Preparing training data...")
        
        # Preprocess all images
        processed_data = {}
        
        for student_id, data in students_data.items():
            genuine_samples = data.get('genuine', [])
            
            if len(genuine_samples) == 0:
                print(f"⚠️  Skipping {student_id}: No genuine samples")
                continue
            
            # Preprocess genuine samples WITHOUT augmentation
            genuine_processed = preprocess_batch(
                genuine_samples, 
                is_base64=True, 
                normalize=True,
                apply_augmentation=False  # Never augment
            )
            
            print(f"   ✅ {student_id}: {len(genuine_processed)} samples prepared")
            
            processed_data[student_id] = genuine_processed
        
        # Estimate pairs (for info only)
        total_samples = sum(len(imgs) for imgs in processed_data.values())
        estimated_pairs = total_samples * (total_samples - 1) // 2
        
        print(f"\n📊 Total samples: {total_samples}")
        print(f"   Estimated pairs: {estimated_pairs} (generated on-the-fly)")
        
        return processed_data, estimated_pairs
    
    def train_batch(self, students_data: Dict[str, Dict], 
                   epochs: int = EPOCHS,
                   batch_size: int = BATCH_SIZE,
                   progress_callback=None) -> Dict:
        """
        Train on a batch of students (incremental learning)
        
        Args:
            students_data: Dict of {student_id: {'genuine': [...], 'forged': [...]}}
            epochs: Number of training epochs
            batch_size: Batch size for training
            progress_callback: Callback function for progress updates
        
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*60}")
        print(f"🚀 BATCH TRAINING STARTED")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        # Detect changes
        changes = self.detect_changes(students_data)
        
        if changes['total_to_train'] == 0:
            print("ℹ️  No new or updated students to train")
            return {
                'success': True,
                'message': 'No training needed',
                'students_trained': 0
            }
        
        # Prepare training data (memory-efficient - no pairs yet)
        # Augmentation disabled by default to avoid confusion
        processed_data, estimated_pairs = self.prepare_training_data(students_data, use_augmentation=False)
        
        if len(processed_data) == 0:
            raise ValueError("No training data generated. Check your data.")
        
        # For incremental learning: If we already have a trained model,
        # we do fine-tuning with lower learning rate
        is_incremental = len(self.embeddings_db) > 0
        
        if is_incremental:
            print("\n🔄 Incremental Learning Mode")
            print("   Using existing model for fine-tuning...")
            
            # Lower learning rate for fine-tuning
            self.model.learning_rate = 5e-5
            self.model.compile_model()
            
            # Unfreeze more layers for adaptation
            self.model.unfreeze_layers(num_layers=75)
        else:
            print("\n🆕 Fresh Training Mode")
            print("   Training from pretrained weights...")
        
        # Create data generators (memory-efficient)
        print("\n🔧 Creating data generators...")
        train_generator = SiamesePairGenerator(
            processed_data=processed_data,
            batch_size=batch_size,
            positive_ratio=0.5
        )
        
        val_generator = ValidationPairGenerator(
            processed_data=processed_data,
            batch_size=batch_size,
            num_pairs=min(1000, len(processed_data) * 20)
        )
        
        # Create callbacks
        training_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Custom progress callback
        if progress_callback:
            class ProgressCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = ((epoch + 1) / epochs) * 100
                    progress_callback(progress, f"Epoch {epoch+1}/{epochs}")
            
            training_callbacks.append(ProgressCallback())
        
        # Train model with generator
        print(f"\n🎯 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Learning Rate: {self.model.learning_rate}")
        print(f"   Using memory-efficient generators")
        print(f"\n{'='*60}\n")
        
        history = self.model.siamese_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1,
            callbacks=training_callbacks
        )
        
        # Generate embeddings for all students
        print("\n🧮 Generating embeddings for all students...")
        
        for student_id, data in students_data.items():
            genuine_samples = data.get('genuine', [])
            
            if len(genuine_samples) == 0:
                continue
            
            # Preprocess
            genuine_processed = preprocess_batch(genuine_samples, is_base64=True, normalize=True)
            
            # Generate embeddings
            embeddings = self.model.get_embeddings_batch(genuine_processed)
            
            # Store embeddings (merge with existing if incremental)
            if is_incremental and student_id in self.embeddings_db:
                # Merge with existing embeddings
                old_embeddings = self.embeddings_db[student_id]
                merged = np.vstack([old_embeddings, embeddings])
                
                # Keep only the most recent embeddings (limit to 100 per student)
                if len(merged) > 100:
                    merged = merged[-100:]
                
                self.embeddings_db[student_id] = merged
                print(f"   🔄 {student_id}: Updated embeddings ({len(old_embeddings)} → {len(merged)})")
            else:
                self.embeddings_db[student_id] = embeddings
                print(f"   🆕 {student_id}: Created {len(embeddings)} embeddings")
            
            # Update metadata
            self.metadata['students'][student_id] = {
                'genuine_count': len(genuine_samples),
                'forged_count': len(data.get('forged', [])),
                'embedding_count': len(self.embeddings_db[student_id]),
                'last_trained': datetime.now().isoformat()
            }
        
        # Save state
        self.save_state()
        
        # Record training history
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        training_record = {
            'timestamp': end_time.isoformat(),
            'students_trained': changes['total_to_train'],
            'new_students': changes['new_students'],
            'updated_students': changes['existing_updated'],
            'duration_seconds': duration,
            'epochs': epochs,
            'final_loss': float(history.history['loss'][-1]) if history.history else None,
            'final_val_loss': float(history.history['val_loss'][-1]) if history.history else None
        }
        
        self.metadata['training_history'].append(training_record)
        
        # Save updated metadata
        with open(METADATA_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Students trained: {changes['total_to_train']}")
        print(f"   Total students in DB: {len(self.embeddings_db)}")
        print(f"   Total embeddings: {sum(len(embs) for embs in self.embeddings_db.values())}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'message': f'Successfully trained {changes["total_to_train"]} students',
            'students_trained': changes['total_to_train'],
            'total_students': len(self.embeddings_db),
            'duration': duration,
            'history': training_record
        }
    
    def delete_student(self, student_id: str):
        """
        Delete a student from the database
        
        Args:
            student_id: Student ID to delete
        """
        if student_id in self.embeddings_db:
            del self.embeddings_db[student_id]
            
            if student_id in self.metadata['students']:
                del self.metadata['students'][student_id]
            
            self.save_state()
            print(f"✅ Deleted student: {student_id}")
        else:
            print(f"⚠️  Student not found: {student_id}")
    
    def get_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_trained': len(self.embeddings_db) > 0,
            'total_students': len(self.embeddings_db),
            'total_embeddings': sum(len(embs) for embs in self.embeddings_db.values()),
            'last_updated': self.metadata.get('last_updated'),
            'students': list(self.embeddings_db.keys()),
            'metadata': self.metadata
        }


# Global trainer instance
trainer = IncrementalTrainer()


def train_batch_students(students_data: Dict[str, Dict], 
                        epochs: int = EPOCHS,
                        batch_size: int = BATCH_SIZE,
                        progress_callback=None) -> Dict:
    """
    Main entry point for batch training
    
    Args:
        students_data: Dict of {student_id: {'genuine': [...], 'forged': [...]}}
        epochs: Number of epochs
        batch_size: Batch size
        progress_callback: Progress callback function
    
    Returns:
        Training results
    """
    return trainer.train_batch(students_data, epochs, batch_size, progress_callback)


def get_trainer():
    """Get the global trainer instance"""
    return trainer
