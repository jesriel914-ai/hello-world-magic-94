# filepath: siamese_training/model.py
"""
Siamese Network Architecture for Signature Identification
Optimized for Colab T4 GPU with TensorFlow 2.19
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

def create_embedding_network(input_shape=(224, 224, 1)):
    """
    Create the embedding network (encoder) for Siamese architecture.
    SIMPLIFIED: Uses MobileNetV2 instead of EfficientNet for better generalization.
    
    Args:
        input_shape: Input image shape (height, width, channels)
    
    Returns:
        Keras Model that outputs 128-dimensional embeddings
    """
    inputs = layers.Input(shape=input_shape, name='signature_input')
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)
    
    # Convert grayscale to RGB for MobileNet (requires 3 channels)
    x = layers.Concatenate()([x, x, x])
    
    # Load MobileNetV2 (LIGHTER than EfficientNet, better for signatures)
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        alpha=0.75,  # Reduced complexity (was 1.0)
        pooling='avg'
    )
    
    # Fine-tune from layer 80 onwards (was 100)
    base_model.trainable = True
    for layer in base_model.layers[:80]:
        layer.trainable = False
    
    x = base_model(x)
    
    # SIMPLIFIED embedding layers - Less overfitting
    x = layers.Dense(256, activation='relu', name='dense1')(x)  # Reduced from 512
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Increased dropout (was 0.3)
    
    # Removed second dense layer (was 256 units)
    
    # Final embedding - L2 normalized
    embeddings = layers.Dense(128, activation=None, name='embeddings')(x)
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='l2_norm')(embeddings)
    
    model = Model(inputs=inputs, outputs=embeddings, name='embedding_network')
    return model


def create_siamese_model(input_shape=(224, 224, 1)):
    """
    Create Siamese Network with contrastive loss.
    
    Args:
        input_shape: Input image shape
    
    Returns:
        Siamese model and embedding network
    """
    # Create embedding network
    embedding_network = create_embedding_network(input_shape)
    
    # Define inputs for pairs
    input_a = layers.Input(shape=input_shape, name='input_a')
    input_b = layers.Input(shape=input_shape, name='input_b')
    
    # Get embeddings
    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)
    
    # Compute L2 distance
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)),
        name='euclidean_distance'
    )([embedding_a, embedding_b])
    
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance, name='siamese_network')
    
    return siamese_model, embedding_network


def contrastive_loss(y_true, y_pred, margin=0.6):
    """
    RELAXED Contrastive loss - Allows more variation within classes.
    
    For signatures: People's handwriting varies naturally, so we use:
    - Larger margin (0.6 instead of 0.5)
    - Lower weight on dissimilar loss (0.6 instead of 0.8)
    
    This makes the model less "picky" about exact matches.
    
    Args:
        y_true: Ground truth labels (1 for similar, 0 for dissimilar)
        y_pred: Predicted distances
        margin: Margin for dissimilar pairs (increased to 0.6)
    
    Returns:
        Loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.squeeze(y_pred)
    
    # Similar pairs: minimize distance
    similar_loss = y_true * tf.square(y_pred)
    
    # Dissimilar pairs: maximize distance up to margin
    dissimilar_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    
    # Reduced weight on dissimilar (0.6) = more lenient on similar pairs
    return tf.reduce_mean(similar_loss + 0.6 * dissimilar_loss)


def triplet_loss(y_true, y_pred, margin=0.5):
    """
    Alternative: Triplet loss for Siamese networks.
    
    Args:
        y_true: Not used (kept for compatibility)
        y_pred: [anchor_embedding, positive_embedding, negative_embedding]
        margin: Margin for triplet loss
    
    Returns:
        Loss value
    """
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


class SiameseModelWrapper:
    """
    Wrapper class for Siamese model with training and inference utilities.
    """
    
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.siamese_model = None
        self.embedding_network = None
        self.history = None
        
    def build(self):
        """Build the Siamese model with RELAXED loss function."""
        self.siamese_model, self.embedding_network = create_siamese_model(self.input_shape)
        
        # Compile with RELAXED contrastive loss (larger margin, lower dissimilar weight)
        self.siamese_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=0.6),
            metrics=['accuracy']
        )
        
    def load_weights(self, filepath):
        """Load model weights."""
        if self.embedding_network is None:
            self.build()
        self.embedding_network.load_weights(filepath)
        
    def save_weights(self, filepath):
        """Save model weights."""
        self.embedding_network.save_weights(filepath)
        
    def get_embedding(self, image):
        """
        Get embedding for a single image.
        
        Args:
            image: numpy array of shape (224, 224, 1)
        
        Returns:
            128-dimensional embedding
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return self.embedding_network.predict(image, verbose=0)[0]
    
    def get_embeddings_batch(self, images):
        """
        Get embeddings for batch of images.
        
        Args:
            images: numpy array of shape (N, 224, 224, 1)
        
        Returns:
            Array of embeddings (N, 128)
        """
        return self.embedding_network.predict(images, verbose=0)
    
    def compute_distance(self, embedding1, embedding2):
        """
        Compute Euclidean distance between two embeddings.
        
        Args:
            embedding1: First embedding (128-dimensional)
            embedding2: Second embedding (128-dimensional)
        
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def summary(self):
        """Print model summary."""
        if self.embedding_network:
            self.embedding_network.summary()