"""
Siamese Network Training Module - PROPER CONTRASTIVE LOSS
Fixed: Uses contrastive loss instead of broken binary classification
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from datetime import datetime
import cv2
from pathlib import Path
import gc

# CPU Optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

try:
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Custom Contrastive Loss
def contrastive_loss(margin=1.0):
    """
    Contrastive loss for Siamese networks
    - For similar pairs (y=1): penalize large distances
    - For dissimilar pairs (y=0): penalize small distances
    """
    def loss(y_true, y_pred):
        # y_pred is the distance between embeddings
        # y_true: 1 for similar (genuine), 0 for dissimilar (forged)
        # Cast to float32 to fix type mismatch
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

class SiameseSignatureTrainer:
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.img_size = (224, 224)
        self.threshold = 0.5  # Distance threshold (lower = more similar)
        self.margin = 1.0  # Contrastive loss margin
        
    def build_siamese_network(self):
        """Build Siamese network with proper distance output"""
        input_shape = (*self.img_size, 3)
        
        # Feature extractor
        base = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128)
        ], name='feature_extractor')
        
        # Siamese inputs
        input_a = layers.Input(shape=input_shape, name='input_a')
        input_b = layers.Input(shape=input_shape, name='input_b')
        
        # Get embeddings
        embedding_a = base(input_a)
        embedding_b = base(input_b)
        
        # L2 normalize
        embedding_a = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding_a)
        embedding_b = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding_b)
        
        # Calculate Euclidean distance (this is our output)
        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(
                tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)
            ),
            name='distance'
        )([embedding_a, embedding_b])
        
        model = models.Model(inputs=[input_a, input_b], outputs=distance)
        return model, base
    
    def create_pairs(self, genuine_images, forged_images):
        """Create training pairs"""
        pairs = []
        labels = []
        n_genuine = len(genuine_images)
        n_forged = len(forged_images)
        
        print(f"Creating pairs from {n_genuine} genuine and {n_forged} forged samples...")
        
        # Positive pairs (genuine vs genuine) - label = 1 (similar)
        for i in range(n_genuine):
            for j in range(i + 1, min(i + 4, n_genuine)):
                pairs.append([genuine_images[i], genuine_images[j]])
                labels.append(1)
        
        # Negative pairs (genuine vs forged) - label = 0 (dissimilar)
        for genuine in genuine_images:
            # Use up to 3 forged samples per genuine
            for forged in forged_images[:min(3, n_forged)]:
                pairs.append([genuine, forged])
                labels.append(0)
        
        return np.array(pairs), np.array(labels)
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        return img
    
    def train_student_model(self, student_id, genuine_samples, forged_samples=None, epochs=50):
        """Train Siamese model with contrastive loss"""
        print(f"\n{'='*60}")
        print(f"Training Model for Student: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        student_dir.mkdir(exist_ok=True)
        
        # Load images
        print(f"Loading images...")
        genuine_images = [self.preprocess_image(img) for img in genuine_samples]
        
        if not forged_samples or len(forged_samples) < 5:
            raise ValueError(
                f"\n{'='*60}\n"
                f"TRAINING BLOCKED: Insufficient forged samples\n"
                f"{'='*60}\n"
                f"Current forged samples: {len(forged_samples) if forged_samples else 0}\n"
                f"Required minimum: 5 forged samples\n"
                f"Recommended: 20+ forged samples\n\n"
                f"Why forged samples are required:\n"
                f"- The model needs examples of 'different' signatures\n"
                f"- Using other students' signatures as 'forged' is CORRECT\n"
                f"- Without negatives, model cannot learn to distinguish\n"
                f"{'='*60}\n"
            )
        
        forged_images = [self.preprocess_image(img) for img in forged_samples]
        
        print(f"✓ Genuine: {len(genuine_images)}")
        print(f"✓ Forged: {len(forged_images)}")
        
        # Create pairs
        pairs, labels = self.create_pairs(genuine_images, forged_images)
        
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count
        print(f"✓ Generated {len(pairs)} pairs")
        print(f"  - Positive (similar): {pos_count}")
        print(f"  - Negative (different): {neg_count}")
        print(f"  - Balance: {pos_count/len(labels)*100:.1f}% positive")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            pairs, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train_a = np.array([pair[0] for pair in X_train])
        X_train_b = np.array([pair[1] for pair in X_train])
        X_val_a = np.array([pair[0] for pair in X_val])
        X_val_b = np.array([pair[1] for pair in X_val])
        
        del pairs
        gc.collect()
        
        # Build model
        print(f"\nBuilding Siamese network...")
        model, feature_extractor = self.build_siamese_network()
        
        # CRITICAL: Use contrastive loss, not binary crossentropy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=contrastive_loss(margin=self.margin),
            metrics=['mae']  # Mean Absolute Error for distance
        )
        
        print(f"✓ Model parameters: {model.count_params():,}")
        print(f"✓ Loss function: Contrastive Loss (margin={self.margin})")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\nStarting training (max {epochs} epochs)...")
        print("="*60)
        start_time = datetime.now()
        
        history = model.fit(
            [X_train_a, X_train_b], y_train,
            validation_data=([X_val_a, X_val_b], y_val),
            epochs=epochs,
            batch_size=4,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        print("\n" + "="*60)
        print("Evaluating model performance...")
        
        try:
            val_loss = model.evaluate([X_val_a, X_val_b], y_val, verbose=0)[0]
        except Exception as e:
            print(f"Warning: Could not evaluate model: {e}")
            val_loss = history.history['val_loss'][-1]
        
        # Calculate accuracy manually on validation set
        try:
            val_distances = model.predict([X_val_a, X_val_b], verbose=0)
            predictions = (val_distances.flatten() < self.threshold).astype(int)
            accuracy = np.mean(predictions == y_val)
        except Exception as e:
            print(f"Warning: Could not calculate accuracy: {e}")
            accuracy = 0.0
        
        print(f"\n✅ Training completed!")
        print(f"Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Accuracy (threshold={self.threshold}): {accuracy*100:.2f}%")
        
        # Save models
        print(f"\nSaving models...")
        try:
            model.save(student_dir / 'siamese_model.keras')
            print(f"✓ Saved siamese_model.keras")
        except Exception as e:
            print(f"⚠️  Warning: Could not save full model: {e}")
        
        try:
            feature_extractor.save(student_dir / 'feature_extractor.keras')
            print(f"✓ Saved feature_extractor.keras")
        except Exception as e:
            print(f"❌ Error: Could not save feature extractor: {e}")
            raise
        
        metadata = {
            'student_id': student_id,
            'training_date': datetime.now().isoformat(),
            'genuine_samples': len(genuine_images),
            'forged_samples': len(forged_images),
            'training_pairs': len(X_train) + len(X_val),
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': training_time,
            'final_val_loss': float(val_loss),
            'final_accuracy': float(accuracy) if accuracy > 0 else 0.0,
            'threshold': self.threshold,
            'margin': self.margin,
            'img_size': list(self.img_size)
        }
        
        try:
            with open(student_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved metadata.json")
        except Exception as e:
            print(f"⚠️  Warning: Could not save metadata: {e}")
        
        # Clear memory
        try:
            del model, feature_extractor, X_train_a, X_train_b, X_val_a, X_val_b
            gc.collect()
        except:
            pass
        
        print(f"✓ Models saved to: {student_dir}")
        print("="*60 + "\n")
        
        return metadata
    
    def save_reference_embeddings(self, student_id, genuine_samples):
        """Save reference embeddings for verification"""
        student_dir = self.base_dir / student_id
        feature_extractor = keras.models.load_model(student_dir / 'feature_extractor.keras')
        
        genuine_images = [self.preprocess_image(img) for img in genuine_samples]
        embeddings = feature_extractor.predict(np.array(genuine_images), verbose=0)
        
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        np.save(student_dir / 'reference_genuine.npy', embeddings)
        print(f"✓ Saved {len(embeddings)} reference embeddings")