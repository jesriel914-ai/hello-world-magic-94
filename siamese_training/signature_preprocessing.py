"""
Enhanced Signature Preprocessing Module
Extracts ONLY the signature strokes, ignoring background, paper color, shadows, etc.
Works with various camera qualities and lighting conditions
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class SignaturePreprocessor:
    """
    Robust signature extraction and preprocessing
    Handles real-world conditions: different cameras, lighting, paper colors, shadows
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def extract_signature(self, img: np.ndarray) -> np.ndarray:
        """
        Extract ONLY signature strokes from image
        Returns clean white background with black strokes
        
        Steps:
        1. Convert to grayscale
        2. Remove background/paper color automatically
        3. Isolate dark strokes (pen ink)
        4. Clean up noise
        5. Normalize to consistent appearance
        """
        
        # Step 1: Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Step 2: Apply bilateral filter to smooth while preserving edges
        # This helps with camera noise and JPEG artifacts
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 3: Adaptive thresholding to handle varying lighting
        # This automatically adjusts for different paper colors and shadows
        # CRITICAL: This separates dark ink from light background
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so strokes are WHITE on BLACK
            blockSize=15,  # Larger block for handling shadows
            C=10  # Constant subtracted (higher = more aggressive)
        )
        
        # Step 4: Morphological operations to clean up noise
        # Remove small noise while keeping signature strokes intact
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
        
        # Step 5: Connect nearby strokes (pen lifts)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
        
        # Step 6: Find signature contours and create clean mask
        contours, _ = cv2.findContours(
            connected, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter out tiny noise contours (area threshold)
        # Keep only substantial strokes
        min_contour_area = 50  # Adjust based on image size
        significant_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > min_contour_area
        ]
        
        # Create clean signature mask
        signature_mask = np.zeros_like(connected)
        cv2.drawContours(signature_mask, significant_contours, -1, 255, -1)
        
        # Step 7: Crop to signature bounding box (remove excess white space)
        signature_cropped = self.crop_to_content(signature_mask)
        
        # Step 8: Pad and resize to target size while maintaining aspect ratio
        signature_padded = self.pad_and_resize(signature_cropped, self.target_size)
        
        # Step 9: Invert back (black strokes on white background)
        # This is the standard convention for signature images
        signature_final = cv2.bitwise_not(signature_padded)
        
        return signature_final
    
    def crop_to_content(self, img: np.ndarray, margin: int = 10) -> np.ndarray:
        """
        Crop image to bounding box of content with margin
        Removes excess white space around signature
        """
        # Find non-zero pixels (the signature)
        coords = cv2.findNonZero(img)
        
        if coords is None:
            # No signature found, return original
            return img
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add margin (but don't exceed image boundaries)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        # Crop to signature region
        cropped = img[y:y+h, x:x+w]
        
        return cropped
    
    def pad_and_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio
        Pad with white background
        """
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor (maintain aspect ratio)
        scale = min(target_w / w, target_h / h)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create white canvas
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Center the signature
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def preprocess_for_training(self, image_path: str) -> np.ndarray:
        """
        Full preprocessing pipeline for training
        Returns: normalized float32 array (0-1) with shape (224, 224, 3)
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Extract signature (white background, black strokes)
        signature_clean = self.extract_signature(img)
        
        # Convert to 3-channel (model expects BGR)
        signature_bgr = cv2.cvtColor(signature_clean, cv2.COLOR_GRAY2BGR)
        
        # Normalize to [0, 1]
        signature_normalized = signature_bgr.astype('float32') / 255.0
        
        return signature_normalized
    
    def preprocess_for_verification(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess captured webcam image for verification
        Input: BGR image from camera
        Returns: normalized float32 array matching training format
        """
        # Extract signature (handles any background, lighting, camera quality)
        signature_clean = self.extract_signature(img)
        
        # Convert to 3-channel
        signature_bgr = cv2.cvtColor(signature_clean, cv2.COLOR_GRAY2BGR)
        
        # Normalize to [0, 1]
        signature_normalized = signature_bgr.astype('float32') / 255.0
        
        return signature_normalized
    
    def debug_visualization(self, img: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize preprocessing steps for debugging
        Shows: Original -> Grayscale -> Binary -> Cleaned -> Final
        """
        import matplotlib.pyplot as plt
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Denoise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Binary threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Final
        final = self.extract_signature(img)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale + Denoised')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('Binary (Adaptive Threshold)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(cleaned, cmap='gray')
        axes[1, 0].set_title('Noise Removed')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(final, cmap='gray')
        axes[1, 1].set_title('Final Extracted Signature')
        axes[1, 1].axis('off')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Debug visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    preprocessor = SignaturePreprocessor(target_size=(224, 224))
    
    # For training
    processed_img = preprocessor.preprocess_for_training('sample_signature.jpg')
    print(f"Processed shape: {processed_img.shape}")
    print(f"Value range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
    
    # For verification (from webcam)
    img = cv2.imread('webcam_capture.jpg')
    processed_img = preprocessor.preprocess_for_verification(img)
    
    # Debug visualization
    preprocessor.debug_visualization(img, save_path='preprocessing_steps.png')