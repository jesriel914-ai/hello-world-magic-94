"""
google drive filepath: siamese_training/siamese_trainer.py
Siamese Network Training Module - GPU-Optimized for Colab
Enhanced architecture for maximum signature matching precision
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

# GPU Configuration
print("="*60)
print("GPU CONFIGURATION")
print("="*60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU(s) detected and configured")
        print(f"   GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - using CPU")

print("="*60 + "\n")

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision training enabled (float16)")

def contrastive_loss(margin=1.0):
    """
    Enhanced contrastive loss for Siamese networks
    - For similar pairs (y=1): penalize large distances
    - For dissimilar pairs (y=0): penalize small distances
    """
    def loss(y_true, y_pred):
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
        self.threshold = 0.45  # Stricter threshold for higher precision
        self.margin = 1.2  # Larger margin for better separation
        
    def build_enhanced_siamese_network(self):
        """
        Build enhanced Siamese network with deeper architecture
        Optimized for GPU training with maximum accuracy
        """
        input_shape = (*self.img_size, 3)
        
        # Enhanced feature extractor with more layers
        base = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for embeddings
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, dtype='float32')  # Keep output as float32
        ], name='feature_extractor')
        
        # Siamese inputs
        input_a = layers.Input(shape=input_shape, name='input_a')
        input_b = layers.Input(shape=input_shape, name='input_b')
        
        # Get embeddings
        embedding_a = base(input_a)
        embedding_b = base(input_b)
        
        # L2 normalize embeddings
        embedding_a = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding_a)
        embedding_b = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding_b)
        
        # Calculate Euclidean distance
        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(
                tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)
            ),
            name='distance'
        )([embedding_a, embedding_b])
        
        model = models.Model(inputs=[input_a, input_b], outputs=distance)
        return model, base
    
    def create_augmented_pairs(self, genuine_images, forged_images):
        """
        Create training pairs with data augmentation
        """
        pairs = []
        labels = []
        n_genuine = len(genuine_images)
        n_forged = len(forged_images)
        
        print(f"Creating augmented pairs from {n_genuine} genuine and {n_forged} forged samples...")
        
        # Positive pairs (genuine vs genuine) - MORE PAIRS
        for i in range(n_genuine):
            for j in range(i + 1, n_genuine):
                pairs.append([genuine_images[i], genuine_images[j]])
                labels.append(1)
                
                # Add augmented versions
                pairs.append([self.augment_image(genuine_images[i]), genuine_images[j]])
                labels.append(1)
                pairs.append([genuine_images[i], self.augment_image(genuine_images[j])])
                labels.append(1)
        
        # Negative pairs (genuine vs forged) - BALANCED
        for genuine in genuine_images:
            for forged in forged_images:
                pairs.append([genuine, forged])
                labels.append(0)
                
                # Add augmented versions
                pairs.append([self.augment_image(genuine), forged])
                labels.append(0)
        
        print(f"✓ Generated {len(pairs)} training pairs")
        return np.array(pairs), np.array(labels)
    
    def augment_image(self, img):
        """Apply random augmentation to image"""
        aug_img = img.copy()
        
        # Random rotation (-5 to 5 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=(1, 1, 1))
        
        # Random brightness
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            aug_img = np.clip(aug_img * brightness, 0, 1)
        
        # Random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, aug_img.shape)
            aug_img = np.clip(aug_img + noise, 0, 1)
        
        return aug_img
    
    def preprocess_image(self, image_path):
        """Load and preprocess image with enhanced preprocessing"""
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, self.img_size)
        
        # Convert to grayscale and back to RGB for consistency
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better signature extraction
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Convert back to 3 channels
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        img = img.astype('float32') / 255.0
        return img
    
    def train_student_model(self, student_id, genuine_samples, forged_samples=None, epochs=100):
        """Train enhanced Siamese model with GPU acceleration"""
        print(f"\n{'='*60}")
        print(f"Training Enhanced Model for Student: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        student_dir.mkdir(exist_ok=True)
        
        # Load images with enhanced preprocessing
        print(f"Loading and preprocessing images...")
        genuine_images = [self.preprocess_image(img) for img in genuine_samples]
        
        if not forged_samples or len(forged_samples) < 5:
            raise ValueError(
                f"\n{'='*60}\n"
                f"TRAINING BLOCKED: Insufficient forged samples\n"
                f"{'='*60}\n"
                f"Current forged samples: {len(forged_samples) if forged_samples else 0}\n"
                f"Required minimum: 5 forged samples\n"
                f"Recommended: 20+ forged samples\n"
                f"{'='*60}\n"
            )
        
        forged_images = [self.preprocess_image(img) for img in forged_samples]
        
        print(f"✓ Genuine: {len(genuine_images)}")
        print(f"✓ Forged: {len(forged_images)}")
        
        # Create augmented pairs
        pairs, labels = self.create_augmented_pairs(genuine_images, forged_images)
        
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count
        print(f"✓ Total pairs: {len(pairs)}")
        print(f"  - Positive (similar): {pos_count}")
        print(f"  - Negative (different): {neg_count}")
        print(f"  - Balance: {pos_count/len(labels)*100:.1f}% positive")
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            pairs, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        X_train_a = np.array([pair[0] for pair in X_train])
        X_train_b = np.array([pair[1] for pair in X_train])
        X_val_a = np.array([pair[0] for pair in X_val])
        X_val_b = np.array([pair[1] for pair in X_val])
        
        del pairs
        gc.collect()
        
        # Build enhanced model
        print(f"\nBuilding enhanced Siamese network...")
        model, feature_extractor = self.build_enhanced_siamese_network()
        
        # Compile with optimized settings for GPU
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=contrastive_loss(margin=self.margin),
            metrics=['mae']
        )
        
        print(f"✓ Model parameters: {model.count_params():,}")
        print(f"✓ Loss function: Enhanced Contrastive Loss (margin={self.margin})")
        print(f"✓ GPU acceleration: ENABLED")
        
        # Enhanced callbacks
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
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(student_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nStarting GPU training (max {epochs} epochs)...")
        print("="*60)
        start_time = datetime.now()
        
        # Train with larger batch size for GPU
        history = model.fit(
            [X_train_a, X_train_b], y_train,
            validation_data=([X_val_a, X_val_b], y_val),
            epochs=epochs,
            batch_size=32,  # Larger batch size for GPU
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        print("\n" + "="*60)
        print("Evaluating model performance...")
        
        val_loss = model.evaluate([X_val_a, X_val_b], y_val, verbose=0)[0]
        
        # Calculate accuracy with threshold
        val_distances = model.predict([X_val_a, X_val_b], verbose=0)
        predictions = (val_distances.flatten() < self.threshold).astype(int)
        accuracy = np.mean(predictions == y_val)
        
        # Calculate precision, recall, F1
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)
        
        print(f"\n✅ Training completed!")
        print(f"Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        
        # Save models
        print(f"\nSaving models...")
        model.save(student_dir / 'siamese_model.keras')
        feature_extractor.save(student_dir / 'feature_extractor.keras')
        
        metadata = {
            'student_id': student_id,
            'training_date': datetime.now().isoformat(),
            'genuine_samples': len(genuine_images),
            'forged_samples': len(forged_images),
            'training_pairs': len(X_train) + len(X_val),
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': training_time,
            'final_val_loss': float(val_loss),
            'final_accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'threshold': self.threshold,
            'margin': self.margin,
            'img_size': list(self.img_size),
            'architecture': 'enhanced_gpu_optimized'
        }
        
        with open(student_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Models saved to: {student_dir}")
        print("="*60 + "\n")
        
        # Clear memory
        del model, feature_extractor, X_train_a, X_train_b, X_val_a, X_val_b
        gc.collect()
        tf.keras.backend.clear_session()
        
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