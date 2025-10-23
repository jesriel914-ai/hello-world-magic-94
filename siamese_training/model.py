# filepath: siamese_training/model.py
"""
Siamese Neural Network Model for Signature Identification
Architecture: MobileNetV2 encoder + L2-normalized embeddings
Optimized for T4 GPU (12GB RAM)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np

# Model configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
EMBEDDING_DIM = 128


def create_base_encoder(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                        embedding_dim=EMBEDDING_DIM,
                        trainable_layers=50):
    """
    Create base encoder using MobileNetV2
    
    Args:
        input_shape: Input image shape (224, 224, 3)
        embedding_dim: Output embedding dimension (128)
        trainable_layers: Number of last layers to make trainable (for fine-tuning)
    
    Returns:
        Keras Model that outputs L2-normalized embeddings
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='signature_input')
    
    # MobileNetV2 backbone (pretrained on ImageNet)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Remove classification head
        weights='imagenet',
        alpha=1.0,  # Width multiplier (1.0 = full model)
        pooling='avg'  # Global average pooling
    )
    
    # Freeze most layers initially (for incremental learning)
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Get features from MobileNetV2
    x = base_model(inputs, training=False)  # Use in inference mode for stability
    
    # Embedding projection head
    x = layers.Dense(256, activation='relu', name='projection_dense1')(x)
    x = layers.BatchNormalization(name='projection_bn1')(x)
    x = layers.Dropout(0.3, name='projection_dropout')(x)
    
    x = layers.Dense(embedding_dim, activation=None, name='embedding_dense')(x)
    
    # L2 normalization (critical for similarity comparison)
    embeddings = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name='l2_normalize'
    )(x)
    
    # Create model
    encoder = Model(inputs, embeddings, name='signature_encoder')
    
    return encoder


def create_siamese_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                         embedding_dim=EMBEDDING_DIM):
    """
    Create Siamese network with shared encoder
    
    Args:
        input_shape: Input image shape
        embedding_dim: Embedding dimension
    
    Returns:
        Siamese model that takes two images and outputs distance
    """
    # Shared encoder
    encoder = create_base_encoder(input_shape, embedding_dim)
    
    # Two input branches
    input_a = layers.Input(shape=input_shape, name='signature_a')
    input_b = layers.Input(shape=input_shape, name='signature_b')
    
    # Get embeddings for both inputs
    embedding_a = encoder(input_a)
    embedding_b = encoder(input_b)
    
    # Compute L2 distance between embeddings
    distance = layers.Lambda(
        lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)),
        name='euclidean_distance'
    )([embedding_a, embedding_b])
    
    # Create Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance, name='siamese_network')
    
    return siamese_model, encoder


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function for Siamese networks
    
    Args:
        y_true: Ground truth labels (1 for same, 0 for different)
        y_pred: Predicted distances
        margin: Margin for negative pairs
    
    Returns:
        Loss value
    """
    # y_true: 1 if same student, 0 if different
    # y_pred: Euclidean distance
    
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    
    loss = y_true * square_pred + (1 - y_true) * margin_square
    
    return tf.reduce_mean(loss)


def triplet_loss(embeddings, labels, margin=0.5):
    """
    Triplet loss for more robust training
    
    Args:
        embeddings: Batch of embeddings
        labels: Student IDs for each embedding
        margin: Margin for triplet loss
    
    Returns:
        Loss value
    """
    # Implementation of triplet loss with semi-hard negative mining
    # This is more advanced and can be used for better performance
    
    # For now, we use contrastive loss as it's simpler and works well
    # Triplet loss can be added later for improvement
    pass


class SiameseNetwork:
    """
    Wrapper class for Siamese Network with training and inference utilities
    """
    
    def __init__(self, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                 embedding_dim=EMBEDDING_DIM,
                 learning_rate=1e-4):
        """
        Initialize Siamese Network
        
        Args:
            input_shape: Input image shape
            embedding_dim: Embedding dimension
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Create models
        self.siamese_model, self.encoder = create_siamese_model(input_shape, embedding_dim)
        
        # Compile model
        self.compile_model()
        
    def compile_model(self):
        """Compile the Siamese model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.siamese_model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=1.0),
            metrics=['accuracy']
        )
        
    def train(self, pairs, labels, epochs=30, batch_size=96, validation_split=0.1, 
              verbose=1, callbacks=None):
        """
        Train the Siamese network
        
        Args:
            pairs: Training pairs (N, 2, 224, 224, 3)
            labels: Labels (N,) - 1 for same, 0 for different
            epochs: Number of training epochs
            batch_size: Batch size (optimized for T4 GPU)
            validation_split: Validation data fraction
            verbose: Verbosity level
            callbacks: List of Keras callbacks
        
        Returns:
            Training history
        """
        # Prepare data
        input_a = pairs[:, 0]
        input_b = pairs[:, 1]
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ]
        
        # Train model
        history = self.siamese_model.fit(
            [input_a, input_b],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
        )
        
        return history
    
    def get_embedding(self, image):
        """
        Get embedding for a single image
        
        Args:
            image: Preprocessed image (224, 224, 3)
        
        Returns:
            Embedding vector (128,)
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        embedding = self.encoder.predict(image, verbose=0)
        return embedding[0]
    
    def get_embeddings_batch(self, images):
        """
        Get embeddings for a batch of images
        
        Args:
            images: Batch of images (N, 224, 224, 3)
        
        Returns:
            Embeddings (N, 128)
        """
        return self.encoder.predict(images, verbose=0, batch_size=32)
    
    def compute_distance(self, embedding1, embedding2):
        """
        Compute Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Distance value
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity value (0 to 1)
        """
        dot_product = np.dot(embedding1, embedding2)
        # Already L2 normalized, so this is cosine similarity
        similarity = (dot_product + 1) / 2  # Convert from [-1, 1] to [0, 1]
        return similarity
    
    def save_model(self, encoder_path='signature_encoder.h5', 
                   siamese_path='siamese_model.h5'):
        """
        Save both encoder and siamese model
        
        Args:
            encoder_path: Path to save encoder model
            siamese_path: Path to save siamese model
        """
        self.encoder.save(encoder_path)
        self.siamese_model.save(siamese_path)
        print(f"✅ Models saved: {encoder_path}, {siamese_path}")
    
    def load_model(self, encoder_path='signature_encoder.h5'):
        """
        Load encoder model
        
        Args:
            encoder_path: Path to encoder model
        """
        self.encoder = keras.models.load_model(
            encoder_path,
            custom_objects={'tf': tf},
            compile=False
        )
        print(f"✅ Encoder loaded from {encoder_path}")
    
    def unfreeze_layers(self, num_layers=100):
        """
        Unfreeze more layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        # Get base model (MobileNetV2)
        base_model = self.encoder.layers[1]  # MobileNetV2 is the second layer
        
        # Unfreeze last num_layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        # Recompile
        self.compile_model()
        
        print(f"✅ Unfroze {num_layers} layers for fine-tuning")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        print("\n" + "="*60)
        print("ENCODER MODEL SUMMARY")
        print("="*60)
        self.encoder.summary()
        
        print("\n" + "="*60)
        print("SIAMESE MODEL SUMMARY")
        print("="*60)
        self.siamese_model.summary()


# GPU Memory Management for T4 GPU
def configure_gpu_memory():
    """
    Configure GPU memory growth to prevent OOM errors
    Optimized for T4 GPU with 12GB RAM
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth (allocate memory as needed)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ GPU Memory Growth enabled for {len(gpus)} GPU(s)")
            print(f"   Using TensorFlow {tf.__version__}")
            
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU detected. Training will use CPU.")


# Call this at module import
configure_gpu_memory()
