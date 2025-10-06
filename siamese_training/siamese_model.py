"""
Siamese Network for Signature Verification
Optimized for Ryzen 5 3400G, 16GB RAM, Python 3.10.11
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for your hardware
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

class SiameseNetwork:
    def __init__(self, input_shape=(224, 224, 1), embedding_dim=128):
        """
        Initialize Siamese Network
        
        Args:
            input_shape: Input image shape (height, width, channels)
            embedding_dim: Dimension of the embedding vector
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = None
        self.encoder = None
        self.threshold = 0.5
        
    def create_encoder(self):
        """Create the shared encoder network"""
        inputs = keras.Input(shape=self.input_shape, name='input_image')
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Embedding layer
        embeddings = layers.Dense(self.embedding_dim, activation='linear', name='embeddings')(x)
        
        self.encoder = keras.Model(inputs, embeddings, name='encoder')
        return self.encoder
    
    def create_siamese_network(self):
        """Create the complete Siamese network"""
        if self.encoder is None:
            self.create_encoder()
        
        # Input layers for two images
        input_1 = keras.Input(shape=self.input_shape, name='input_1')
        input_2 = keras.Input(shape=self.input_shape, name='input_2')
        
        # Get embeddings for both inputs
        embedding_1 = self.encoder(input_1)
        embedding_2 = self.encoder(input_2)
        
        # Compute distance between embeddings
        distance = layers.Lambda(self._euclidean_distance, name='distance')([embedding_1, embedding_2])
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(distance)
        
        self.model = keras.Model([input_1, input_2], output, name='siamese_network')
        return self.model
    
    def _euclidean_distance(self, vectors):
        """Compute Euclidean distance between two vectors"""
        x, y = vectors
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for the network"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = image_path
            
            if image is None:
                return None
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            return image
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def create_pairs(self, genuine_images, forged_images, max_pairs_per_student=100):
        """Create positive and negative pairs for training"""
        pairs = []
        labels = []
        
        # Positive pairs (genuine-genuine)
        for i in range(len(genuine_images)):
            for j in range(i + 1, min(i + 3, len(genuine_images))):  # Limit pairs to avoid memory issues
                pairs.append([genuine_images[i], genuine_images[j]])
                labels.append(1)  # Same person
        
        # Negative pairs (genuine-forged)
        for genuine in genuine_images:
            for forged in forged_images:
                pairs.append([genuine, forged])
                labels.append(0)  # Different person
        
        # Limit total pairs to prevent memory overflow
        if len(pairs) > max_pairs_per_student:
            indices = np.random.choice(len(pairs), max_pairs_per_student, replace=False)
            pairs = [pairs[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        return np.array(pairs), np.array(labels)
    
    def train(self, genuine_images, forged_images, validation_split=0.2, epochs=50, batch_size=16):
        """Train the Siamese network"""
        print("Creating training pairs...")
        pairs, labels = self.create_pairs(genuine_images, forged_images)
        
        print(f"Created {len(pairs)} training pairs")
        print(f"Positive pairs: {np.sum(labels)}")
        print(f"Negative pairs: {len(labels) - np.sum(labels)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            pairs, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Prepare data for training
        X_train_1 = np.array([self.preprocess_image(pair[0]) for pair in X_train])
        X_train_2 = np.array([self.preprocess_image(pair[1]) for pair in X_train])
        X_val_1 = np.array([self.preprocess_image(pair[0]) for pair in X_val])
        X_val_2 = np.array([self.preprocess_image(pair[1]) for pair in X_val])
        
        # Remove None values
        valid_indices = ~(np.isnan(X_train_1).any(axis=(1, 2, 3)) | np.isnan(X_train_2).any(axis=(1, 2, 3)))
        X_train_1 = X_train_1[valid_indices]
        X_train_2 = X_train_2[valid_indices]
        y_train = y_train[valid_indices]
        
        valid_indices = ~(np.isnan(X_val_1).any(axis=(1, 2, 3)) | np.isnan(X_val_2).any(axis=(1, 2, 3)))
        X_val_1 = X_val_1[valid_indices]
        X_val_2 = X_val_2[valid_indices]
        y_val = y_val[valid_indices]
        
        print(f"Training samples: {len(X_train_1)}")
        print(f"Validation samples: {len(X_val_1)}")
        
        # Create and compile model
        self.create_siamese_network()
        self.compile_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        print("Starting training...")
        history = self.model.fit(
            [X_train_1, X_train_2], y_train,
            validation_data=([X_val_1, X_val_2], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Find optimal threshold
        self.find_optimal_threshold(X_val_1, X_val_2, y_val)
        
        return history
    
    def find_optimal_threshold(self, X_val_1, X_val_2, y_val):
        """Find optimal threshold for verification"""
        # Get predictions
        predictions = self.model.predict([X_val_1, X_val_2], verbose=0)
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (predictions > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {self.threshold:.3f}")
        print(f"Best F1 score: {best_f1:.3f}")
    
    def verify_signature(self, reference_image, test_image):
        """Verify if two signatures belong to the same person"""
        # Preprocess images
        ref_processed = self.preprocess_image(reference_image)
        test_processed = self.preprocess_image(test_image)
        
        if ref_processed is None or test_processed is None:
            return False, 0.0
        
        # Get prediction
        prediction = self.model.predict([
            np.expand_dims(ref_processed, axis=0),
            np.expand_dims(test_processed, axis=0)
        ], verbose=0)[0][0]
        
        # Determine if verified
        is_verified = prediction > self.threshold
        confidence = float(prediction)
        
        return is_verified, confidence
    
    def save_model(self, model_path, metadata_path):
        """Save the trained model and metadata"""
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'embedding_dim': self.embedding_dim,
            'threshold': self.threshold,
            'model_type': 'siamese'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path, metadata_path):
        """Load a trained model and metadata"""
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.input_shape = tuple(metadata['input_shape'])
        self.embedding_dim = metadata['embedding_dim']
        self.threshold = metadata['threshold']
        
        print(f"Model loaded from {model_path}")
        print(f"Threshold: {self.threshold}")

def main():
    """Example usage"""
    print("Siamese Network for Signature Verification")
    print("=" * 50)
    
    # Initialize network
    siamese = SiameseNetwork()
    
    # Example: Load your data here
    # genuine_images = load_genuine_images()  # List of image paths
    # forged_images = load_forged_images()    # List of image paths
    
    # Train model
    # history = siamese.train(genuine_images, forged_images)
    
    # Save model
    # siamese.save_model('siamese_model.h5', 'siamese_metadata.json')
    
    print("Training pipeline ready!")
    print("To use:")
    print("1. Load your genuine and forged signature images")
    print("2. Call siamese.train(genuine_images, forged_images)")
    print("3. Use siamese.verify_signature(ref_image, test_image) for verification")

if __name__ == "__main__":
    main()