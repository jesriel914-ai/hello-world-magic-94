"""
google drive filepath: siamese_training/siamese_trainer.py
Memory-Optimized Siamese Network Trainer with Signature Isolation
NOW focuses ONLY on signature strokes, ignoring background/lighting/camera
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

# Import our new preprocessing module
from signature_preprocessing import SignaturePreprocessor

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
    """Enhanced contrastive loss for Siamese networks"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

class PairGenerator(keras.utils.PyDataset):
    """
    Memory-efficient pair generator with signature-focused augmentation
    """
    def __init__(self, genuine_imgs, forged_imgs, batch_size=32, 
                 augment=True, validation=False, **kwargs):
        super().__init__(**kwargs)
        self.genuine_imgs = genuine_imgs
        self.forged_imgs = forged_imgs
        self.batch_size = batch_size
        self.augment = augment and not validation
        self.validation = validation
        
        n_genuine = len(genuine_imgs)
        n_forged = len(forged_imgs)
        
        # Conservative pair counts for free Colab
        self.positive_pairs_per_epoch = min(n_genuine * (n_genuine - 1), 600)
        self.negative_pairs_per_epoch = min(n_genuine * n_forged, 600)
        
        if self.augment:
            self.positive_pairs_per_epoch *= 2
            self.negative_pairs_per_epoch = int(self.negative_pairs_per_epoch * 1.5)
        
        self.total_pairs = self.positive_pairs_per_epoch + self.negative_pairs_per_epoch
        
        print(f"  Generator initialized:")
        print(f"    Positive pairs/epoch: {self.positive_pairs_per_epoch}")
        print(f"    Negative pairs/epoch: {self.negative_pairs_per_epoch}")
        print(f"    Total pairs/epoch: {self.total_pairs}")
        print(f"    Batches/epoch: {len(self)}")
    
    def __len__(self):
        return int(np.ceil(self.total_pairs / self.batch_size))
    
    def __getitem__(self, idx):
        batch_size = min(self.batch_size, self.total_pairs - idx * self.batch_size)
        
        pairs_a = []
        pairs_b = []
        labels = []
        
        for _ in range(batch_size):
            if np.random.random() < 0.5:
                # Positive pair
                i, j = np.random.choice(len(self.genuine_imgs), 2, replace=False)
                img_a = self.genuine_imgs[i].copy()
                img_b = self.genuine_imgs[j].copy()
                
                if self.augment:
                    img_a = self.augment_signature(img_a)
                    img_b = self.augment_signature(img_b)
                
                pairs_a.append(img_a)
                pairs_b.append(img_b)
                labels.append(1)
            else:
                # Negative pair
                i = np.random.randint(len(self.genuine_imgs))
                j = np.random.randint(len(self.forged_imgs))
                
                img_a = self.genuine_imgs[i].copy()
                img_b = self.forged_imgs[j].copy()
                
                if self.augment:
                    img_a = self.augment_signature(img_a)
                
                pairs_a.append(img_a)
                pairs_b.append(img_b)
                labels.append(0)
        
        return (np.array(pairs_a), np.array(pairs_b)), np.array(labels)
    
    def on_epoch_end(self):
        """Memory cleanup after each epoch"""
        gc.collect()
    
    def augment_signature(self, img):
        """
        Signature-aware augmentation
        Only geometric transforms - NO color/background changes
        """
        aug_img = img.copy()
        
        # Random rotation (-8 to 8 degrees) - signatures can vary in angle
        if np.random.random() > 0.5:
            angle = np.random.uniform(-8, 8)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            # Use white border (255) since we have white background
            aug_img = cv2.warpAffine(aug_img, M, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(1, 1, 1))
        
        # Slight stroke thickness variation (simulate pen pressure)
        if np.random.random() > 0.6:
            kernel_size = np.random.choice([1, 2])
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                # Dilate to thicken strokes slightly
                aug_img_uint8 = (aug_img * 255).astype(np.uint8)
                aug_img_uint8 = cv2.dilate(aug_img_uint8, kernel, iterations=1)
                aug_img = aug_img_uint8.astype('float32') / 255.0
        
        # Very slight elastic distortion (simulate paper bend)
        # This makes the model robust to slightly warped captures
        if np.random.random() > 0.7:
            aug_img = self.elastic_transform(aug_img, alpha=5, sigma=2)
        
        # Small random noise (simulate scan artifacts)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.008, aug_img.shape)
            aug_img = np.clip(aug_img + noise, 0, 1)
        
        return aug_img
    
    def elastic_transform(self, image, alpha=10, sigma=3):
        """
        Elastic deformation for signature augmentation
        Simulates slight paper warping or hand tremor
        """
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), 
            (0, 0), sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), 
            (0, 0), sigma
        ) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Add displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply transformation with white border
        distorted = cv2.remap(
            image, map_x, map_y, 
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1, 1, 1)
        )
        
        return distorted

class SiameseSignatureTrainer:
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.img_size = (224, 224)
        self.threshold = 0.45
        self.margin = 1.2
        
        # Initialize our new preprocessor
        self.preprocessor = SignaturePreprocessor(target_size=self.img_size)
        
    def build_efficient_siamese_network(self):
        """
        Memory-efficient Siamese network with slightly increased capacity
        Can peak at ~9GB RAM and ~11GB GPU (well within free Colab limits)
        """
        input_shape = (*self.img_size, 3)
        
        base = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Block 1 - Initial feature extraction
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2 - Enhanced features
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3 - Deep features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4 - High-level features
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Embedding layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, dtype='float32')
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
        
        # Calculate Euclidean distance
        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(
                tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)
            ),
            name='distance'
        )([embedding_a, embedding_b])
        
        model = models.Model(inputs=[input_a, input_b], outputs=distance)
        return model, base
    
    def preprocess_image(self, image_path):
        """
        NEW: Use signature isolation preprocessing
        Returns ONLY the signature strokes, ignoring everything else
        """
        return self.preprocessor.preprocess_for_training(str(image_path))
    
    def train_student_model(self, student_id, genuine_samples, forged_samples=None, epochs=100):
        """Train with isolated signatures"""
        print(f"\n{'='*60}")
        print(f"Training Signature-Focused Model for: {student_id}")
        print(f"{'='*60}")
        
        student_dir = self.base_dir / student_id
        student_dir.mkdir(exist_ok=True)
        
        # Load and preprocess images with signature isolation
        print(f"Loading and isolating signatures...")
        genuine_images = np.array([self.preprocess_image(img) for img in genuine_samples])
        
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
        
        forged_images = np.array([self.preprocess_image(img) for img in forged_samples])
        
        print(f"✓ Genuine: {len(genuine_images)} (isolated signatures)")
        print(f"✓ Forged: {len(forged_images)} (isolated signatures)")
        
        # Create generators
        print(f"\nCreating data generators...")
        
        train_generator = PairGenerator(
            genuine_images, forged_images,
            batch_size=20,  # Slightly larger batch for better GPU utilization
            augment=True,
            validation=False
        )
        
        val_generator = PairGenerator(
            genuine_images, forged_images,
            batch_size=20,
            augment=False,
            validation=True
        )
        
        # Build model
        print(f"\nBuilding enhanced Siamese network...")
        model, feature_extractor = self.build_efficient_siamese_network()
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=contrastive_loss(margin=self.margin),
            metrics=['mae']
        )
        
        print(f"✓ Model parameters: {model.count_params():,}")
        print(f"✓ Estimated memory: ~{model.count_params() * 4 / 1024 / 1024:.1f} MB")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # More patience for better convergence
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(student_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: gc.collect()
            )
        ]
        
        print(f"\nStarting training (max {epochs} epochs)...")
        print(f"Focus: Signature strokes only (background-invariant)")
        print("="*60)
        start_time = datetime.now()
        
        # Train
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        print("\n" + "="*60)
        print("Evaluating model performance...")
        
        val_loss = history.history['val_loss'][-1]
        
        # Calculate metrics on sample
        print("Calculating accuracy metrics...")
        val_pairs_a = []
        val_pairs_b = []
        val_labels = []
        
        for _ in range(250):  # More samples for better metrics
            if np.random.random() < 0.5:
                i, j = np.random.choice(len(genuine_images), 2, replace=False)
                val_pairs_a.append(genuine_images[i])
                val_pairs_b.append(genuine_images[j])
                val_labels.append(1)
            else:
                i = np.random.randint(len(genuine_images))
                j = np.random.randint(len(forged_images))
                val_pairs_a.append(genuine_images[i])
                val_pairs_b.append(forged_images[j])
                val_labels.append(0)
        
        val_pairs_a = np.array(val_pairs_a)
        val_pairs_b = np.array(val_pairs_b)
        val_labels = np.array(val_labels)
        
        val_distances = model.predict([val_pairs_a, val_pairs_b], batch_size=20, verbose=0)
        predictions = (val_distances.flatten() < self.threshold).astype(int)
        accuracy = np.mean(predictions == val_labels)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(val_labels, predictions, zero_division=0)
        recall = recall_score(val_labels, predictions, zero_division=0)
        f1 = f1_score(val_labels, predictions, zero_division=0)
        
        print(f"\n✅ Training completed!")
        print(f"Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        
        # Clean up
        del val_pairs_a, val_pairs_b, val_labels, val_distances, predictions
        gc.collect()
        
        # Save models
        print(f"\nSaving models...")
        model.save(student_dir / 'siamese_model.keras')
        feature_extractor.save(student_dir / 'feature_extractor.keras')
        
        metadata = {
            'student_id': student_id,
            'training_date': datetime.now().isoformat(),
            'genuine_samples': len(genuine_images),
            'forged_samples': len(forged_images),
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
            'architecture': 'signature_isolated_v2',
            'preprocessing': 'signature_extraction',
            'background_invariant': True
        }
        
        with open(student_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Models saved to: {student_dir}")
        print(f"✓ Training focused on signature strokes only")
        print(f"✓ Model is now robust to camera quality, lighting, and background")
        print("="*60 + "\n")
        
        # Final cleanup
        del model, feature_extractor, genuine_images, forged_images
        del train_generator, val_generator
        gc.collect()
        tf.keras.backend.clear_session()
        
        return metadata
    
    def save_reference_embeddings(self, student_id, genuine_samples):
        """Save reference embeddings using isolated signatures"""
        student_dir = self.base_dir / student_id
        feature_extractor = keras.models.load_model(student_dir / 'feature_extractor.keras')
        
        # Process with signature isolation
        genuine_images = [self.preprocess_image(img) for img in genuine_samples]
        embeddings = []
        
        batch_size = 10
        for i in range(0, len(genuine_images), batch_size):
            batch = genuine_images[i:i+batch_size]
            batch_embeddings = feature_extractor.predict(np.array(batch), verbose=0)
            embeddings.append(batch_embeddings)
            gc.collect()
        
        embeddings = np.vstack(embeddings)
        
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        np.save(student_dir / 'reference_genuine.npy', embeddings)
        print(f"✓ Saved {len(embeddings)} reference embeddings (signature-focused)")
        
        del feature_extractor, embeddings
        gc.collect()
        tf.keras.backend.clear_session()