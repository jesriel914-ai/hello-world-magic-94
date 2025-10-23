# filepath: siamese_training/data_generator.py
"""
Memory-efficient data generator for Siamese network training
Generates pairs on-the-fly instead of loading all pairs into memory
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

class SiamesePairGenerator(keras.utils.Sequence):
    """
    Generates pairs on-the-fly during training to save memory
    """
    
    def __init__(self, processed_data, batch_size=96, positive_ratio=0.5):
        """
        Initialize generator
        
        Args:
            processed_data: Dict of {student_id: numpy array of images}
            batch_size: Number of pairs per batch
            positive_ratio: Ratio of positive pairs (same student)
        """
        self.processed_data = processed_data
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        
        # Create student ID to index mapping
        self.student_ids = list(processed_data.keys())
        self.num_students = len(self.student_ids)
        
        # Calculate total samples
        self.total_samples = sum(len(imgs) for imgs in processed_data.values())
        
        # Estimate epochs
        self.pairs_per_epoch = max(1000, self.total_samples * 10)
        
        print(f"📊 Generator initialized:")
        print(f"   Students: {self.num_students}")
        print(f"   Total samples: {self.total_samples}")
        print(f"   Pairs per epoch: {self.pairs_per_epoch}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Batches per epoch: {len(self)}")
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.pairs_per_epoch / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of pairs
        
        Args:
            index: Batch index
        
        Returns:
            ([batch_a, batch_b], labels)
        """
        batch_a = []
        batch_b = []
        labels = []
        
        num_positive = int(self.batch_size * self.positive_ratio)
        num_negative = self.batch_size - num_positive
        
        # Generate positive pairs (same student)
        for _ in range(num_positive):
            # Random student
            student_id = np.random.choice(self.student_ids)
            student_images = self.processed_data[student_id]
            
            # Need at least 2 images
            if len(student_images) < 2:
                # If only 1 image, use it twice (not ideal but rare)
                img_a = student_images[0]
                img_b = student_images[0]
            else:
                # Random two different images from same student
                idx = np.random.choice(len(student_images), size=2, replace=False)
                img_a = student_images[idx[0]]
                img_b = student_images[idx[1]]
            
            batch_a.append(img_a)
            batch_b.append(img_b)
            labels.append(1.0)  # Same student
        
        # Generate negative pairs (different students)
        if self.num_students < 2:
            # If only 1 student, can't create negative pairs
            # Just duplicate positive pairs (not ideal)
            for _ in range(num_negative):
                student_id = self.student_ids[0]
                student_images = self.processed_data[student_id]
                idx = np.random.choice(len(student_images), size=2, replace=True)
                batch_a.append(student_images[idx[0]])
                batch_b.append(student_images[idx[1]])
                labels.append(0.0)
        else:
            for _ in range(num_negative):
                # Random two different students
                student_ids = np.random.choice(self.student_ids, size=2, replace=False)
                
                images_a = self.processed_data[student_ids[0]]
                images_b = self.processed_data[student_ids[1]]
                
                # Random image from each
                img_a = images_a[np.random.randint(0, len(images_a))]
                img_b = images_b[np.random.randint(0, len(images_b))]
                
                batch_a.append(img_a)
                batch_b.append(img_b)
                labels.append(0.0)  # Different students
        
        # Convert to numpy arrays
        batch_a = np.array(batch_a, dtype=np.float32)
        batch_b = np.array(batch_b, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # Shuffle batch
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        
        batch_a = batch_a[indices]
        batch_b = batch_b[indices]
        labels = labels[indices]
        
        return [batch_a, batch_b], labels
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        # Random seed for shuffling
        pass


class ValidationPairGenerator(keras.utils.Sequence):
    """
    Fixed validation pairs (not random)
    """
    
    def __init__(self, processed_data, batch_size=96, num_pairs=500):
        """
        Initialize validation generator with fixed pairs
        
        Args:
            processed_data: Dict of {student_id: numpy array of images}
            batch_size: Number of pairs per batch
            num_pairs: Total validation pairs to generate
        """
        self.batch_size = batch_size
        
        # Generate fixed validation pairs
        print(f"📊 Generating {num_pairs} fixed validation pairs...")
        
        student_ids = list(processed_data.keys())
        num_students = len(student_ids)
        
        pairs_a = []
        pairs_b = []
        labels = []
        
        num_positive = num_pairs // 2
        num_negative = num_pairs - num_positive
        
        # Generate positive pairs
        for _ in range(num_positive):
            student_id = np.random.choice(student_ids)
            student_images = processed_data[student_id]
            
            if len(student_images) >= 2:
                idx = np.random.choice(len(student_images), size=2, replace=False)
                pairs_a.append(student_images[idx[0]])
                pairs_b.append(student_images[idx[1]])
                labels.append(1.0)
        
        # Generate negative pairs
        if num_students >= 2:
            for _ in range(num_negative):
                student_ids_pair = np.random.choice(student_ids, size=2, replace=False)
                
                images_a = processed_data[student_ids_pair[0]]
                images_b = processed_data[student_ids_pair[1]]
                
                pairs_a.append(images_a[np.random.randint(0, len(images_a))])
                pairs_b.append(images_b[np.random.randint(0, len(images_b))])
                labels.append(0.0)
        
        self.pairs_a = np.array(pairs_a, dtype=np.float32)
        self.pairs_b = np.array(pairs_b, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        
        self.num_pairs = len(self.labels)
        
        print(f"   ✅ Created {self.num_pairs} validation pairs")
    
    def __len__(self):
        """Number of batches"""
        return int(np.ceil(self.num_pairs / self.batch_size))
    
    def __getitem__(self, index):
        """Get batch"""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_pairs)
        
        batch_a = self.pairs_a[start_idx:end_idx]
        batch_b = self.pairs_b[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        return [batch_a, batch_b], batch_labels
