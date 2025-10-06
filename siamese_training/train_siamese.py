"""
Training Script for Siamese Signature Verification
Usage: python train_siamese.py --student_id <student_id> --genuine_dir <path> --forged_dir <path>
"""

import argparse
import os
import glob
from siamese_model import SiameseNetwork
import matplotlib.pyplot as plt

def load_images_from_directory(directory):
    """Load all images from a directory"""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
        image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    print(f"Found {len(image_paths)} images in {directory}")
    return image_paths

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network for Signature Verification')
    parser.add_argument('--student_id', required=True, help='Student ID')
    parser.add_argument('--genuine_dir', required=True, help='Directory containing genuine signatures')
    parser.add_argument('--forged_dir', required=True, help='Directory containing forged signatures')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--output_dir', default='./models', help='Output directory for saved models')
    
    args = parser.parse_args()
    
    print("Siamese Signature Verification Training")
    print("=" * 50)
    print(f"Student ID: {args.student_id}")
    print(f"Genuine directory: {args.genuine_dir}")
    print(f"Forged directory: {args.forged_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    print("Loading images...")
    genuine_images = load_images_from_directory(args.genuine_dir)
    forged_images = load_images_from_directory(args.forged_dir)
    
    if len(genuine_images) == 0:
        print("ERROR: No genuine images found!")
        return
    
    if len(forged_images) == 0:
        print("WARNING: No forged images found! Training with only genuine images.")
        forged_images = []
    
    print(f"Loaded {len(genuine_images)} genuine images")
    print(f"Loaded {len(forged_images)} forged images")
    print()
    
    # Initialize and train model
    print("Initializing Siamese Network...")
    siamese = SiameseNetwork()
    
    print("Starting training...")
    history = siamese.train(
        genuine_images=genuine_images,
        forged_images=forged_images,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f"siamese_{args.student_id}.h5")
    metadata_path = os.path.join(args.output_dir, f"siamese_{args.student_id}_metadata.json")
    
    siamese.save_model(model_path, metadata_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, f"training_history_{args.student_id}.png")
    plot_training_history(history, plot_path)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Training plots saved to: {plot_path}")

if __name__ == "__main__":
    main()