# filepath: siamese_training/preprocessing.py
"""
Image Preprocessing for Signature Identification
Optimized for T4 GPU training - preserves signature quality
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Target size for MobileNetV2
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3


def decode_base64_image(base64_string):
    """
    Decode base64 image string to numpy array
    
    Args:
        base64_string: Base64 encoded image (with or without data URI prefix)
    
    Returns:
        numpy array: Decoded image in BGR format
    """
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_bytes))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif len(img_array.shape) == 2:
        # Grayscale to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    return img_array


def preprocess_signature(image_data, is_base64=True, normalize=True):
    """
    Preprocess signature image for model input
    
    Args:
        image_data: Either base64 string or numpy array
        is_base64: Whether input is base64 encoded
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        numpy array: Preprocessed image ready for model (224x224x3)
    """
    # Decode if base64
    if is_base64:
        img = decode_base64_image(image_data)
    else:
        img = image_data.copy()
    
    # Step 1: Convert to grayscale for signature processing
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Step 2: Apply adaptive thresholding to enhance signature
    # This helps separate signature from background
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 3: Remove noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Step 4: Find signature bounding box and crop
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours (signature region)
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:  # Filter out very small contours (noise)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        # Add padding around signature
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(gray.shape[1], x_max + padding)
        y_max = min(gray.shape[0], y_max + padding)
        
        # Crop to signature region
        if x_max > x_min and y_max > y_min:
            gray = gray[y_min:y_max, x_min:x_max]
    
    # Step 5: Resize to target size (224x224) while maintaining aspect ratio
    h, w = gray.shape
    aspect_ratio = w / h
    
    if aspect_ratio > 1:
        # Wider than tall
        new_w = IMG_WIDTH
        new_h = int(IMG_WIDTH / aspect_ratio)
    else:
        # Taller than wide
        new_h = IMG_HEIGHT
        new_w = int(IMG_HEIGHT * aspect_ratio)
    
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 6: Create canvas and center the signature
    canvas = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255
    
    y_offset = (IMG_HEIGHT - new_h) // 2
    x_offset = (IMG_WIDTH - new_w) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Step 7: Invert so signature is white on black (better for neural networks)
    inverted = 255 - canvas
    
    # Step 8: Convert to 3 channels (RGB)
    rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    
    # Step 9: Normalize to [0, 1] range
    if normalize:
        rgb = rgb.astype(np.float32) / 255.0
    
    return rgb


def preprocess_batch(images_data, is_base64=True, normalize=True, apply_augmentation=False):
    """
    Preprocess a batch of signature images
    
    Args:
        images_data: List of base64 strings or numpy arrays
        is_base64: Whether inputs are base64 encoded
        normalize: Whether to normalize pixel values
        apply_augmentation: Whether to apply augmentation (DEFAULT: False)
    
    Returns:
        numpy array: Batch of preprocessed images (N, 224, 224, 3)
    """
    processed = []
    
    for img_data in images_data:
        processed_img = preprocess_signature(img_data, is_base64, normalize)
        
        if apply_augmentation:
            # Apply augmentation (doubles samples)
            augmented = augment_signature(processed_img, augmentation_factor=2)
            processed.extend(augmented)
        else:
            # No augmentation - use original only
            processed.append(processed_img)
    
    return np.array(processed, dtype=np.float32)


def create_pairs_from_data(genuine_samples_dict, forged_samples_dict=None):
    """
    Create training pairs (positive and negative) from genuine samples
    For genuine-only training, creates pairs within and across students
    
    Args:
        genuine_samples_dict: Dict of {student_id: [list of genuine samples]}
        forged_samples_dict: Optional dict of {student_id: [list of forged samples]}
    
    Returns:
        pairs: numpy array of image pairs (N, 2, 224, 224, 3)
        labels: numpy array of labels (N,) - 1 for same, 0 for different
    """
    pairs = []
    labels = []
    
    student_ids = list(genuine_samples_dict.keys())
    
    # For each student
    for student_id in student_ids:
        genuine_samples = genuine_samples_dict[student_id]
        
        if len(genuine_samples) < 2:
            continue
        
        # Create positive pairs (same student)
        for i in range(len(genuine_samples)):
            for j in range(i + 1, len(genuine_samples)):
                pairs.append([genuine_samples[i], genuine_samples[j]])
                labels.append(1)  # Same student
        
        # Create negative pairs (different students)
        for other_id in student_ids:
            if other_id == student_id:
                continue
            
            other_samples = genuine_samples_dict[other_id]
            
            # Create some negative pairs
            num_negative = min(len(genuine_samples) * 2, len(other_samples) * 2)
            
            for _ in range(num_negative):
                idx1 = np.random.randint(0, len(genuine_samples))
                idx2 = np.random.randint(0, len(other_samples))
                pairs.append([genuine_samples[idx1], other_samples[idx2]])
                labels.append(0)  # Different students
    
    return np.array(pairs), np.array(labels, dtype=np.float32)


def augment_signature(image, augmentation_factor=2):
    """
    Apply data augmentation to signature image
    
    Args:
        image: Preprocessed signature image (224, 224, 3)
        augmentation_factor: Number of augmented versions to create
    
    Returns:
        List of augmented images
    """
    augmented = [image]  # Include original
    
    for _ in range(augmentation_factor - 1):
        aug_img = image.copy()
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        h, w = aug_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(0, 0, 0))
        
        # Random scaling (0.9 to 1.1)
        scale = np.random.uniform(0.9, 1.1)
        new_size = (int(w * scale), int(h * scale))
        scaled = cv2.resize(aug_img, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Center crop/pad back to original size
        if scale > 1:
            # Crop
            y_start = (scaled.shape[0] - h) // 2
            x_start = (scaled.shape[1] - w) // 2
            aug_img = scaled[y_start:y_start+h, x_start:x_start+w]
        else:
            # Pad
            aug_img = np.zeros((h, w, 3), dtype=aug_img.dtype)
            y_start = (h - scaled.shape[0]) // 2
            x_start = (w - scaled.shape[1]) // 2
            aug_img[y_start:y_start+scaled.shape[0], 
                   x_start:x_start+scaled.shape[1]] = scaled
        
        # Random brightness adjustment (0.8 to 1.2)
        brightness = np.random.uniform(0.8, 1.2)
        aug_img = np.clip(aug_img * brightness, 0, 1)
        
        augmented.append(aug_img.astype(np.float32))
    
    return augmented


def is_likely_signature(image_data, is_base64=True):
    """
    Detect if the uploaded image is likely a signature
    Returns False for random photos, blank pages, etc.
    
    Args:
        image_data: Base64 string or numpy array
        is_base64: Whether input is base64 encoded
    
    Returns:
        bool: True if likely a signature, False otherwise
    """
    # Decode if base64
    if is_base64:
        img = decode_base64_image(image_data)
    else:
        img = image_data.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Check 1: Image should have some content (not blank)
    mean_intensity = np.mean(gray)
    if mean_intensity > 250 or mean_intensity < 5:
        return False  # Too white or too black
    
    # Check 2: Should have edges (signature has strokes)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    if edge_density < 0.01 or edge_density > 0.5:
        return False  # Too few or too many edges
    
    # Check 3: Should have some elongated contours (signature strokes)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 1:
        return False  # No contours found
    
    # Check aspect ratios of largest contours
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0 or max(areas) < 100:
        return False  # No significant contours
    
    return True


# TensorFlow preprocessing function for compatibility
def preprocess_for_mobilenet(image):
    """
    Preprocess image for MobileNetV2 input
    Note: We use our custom preprocessing, but this ensures compatibility
    
    Args:
        image: Preprocessed signature image (224, 224, 3) in [0, 1]
    
    Returns:
        Tensor ready for MobileNetV2
    """
    # MobileNetV2 expects inputs in [-1, 1] range
    # Convert from [0, 1] to [-1, 1]
    return (image * 2.0) - 1.0
