# Siamese Signature Verification System

A Python-based Siamese network implementation for signature verification, optimized for Ryzen 5 3400G, 16GB RAM, and Python 3.10.11.

## Features

- **Lightweight Architecture**: Optimized for your hardware specs
- **Memory Efficient**: Handles large datasets without memory overflow
- **Real-time Verification**: Fast signature matching
- **Automatic Threshold Tuning**: Finds optimal verification threshold
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python siamese_model.py
   ```

## Usage

### 1. Training a Model

Train a Siamese model for a specific student:

```bash
python train_siamese.py \
  --student_id "STU001" \
  --genuine_dir "./data/student1/genuine" \
  --forged_dir "./data/student1/forged" \
  --epochs 50 \
  --batch_size 16
```

**Parameters**:
- `--student_id`: Unique identifier for the student
- `--genuine_dir`: Directory containing genuine signature images
- `--forged_dir`: Directory containing forged signature images
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 16)
- `--output_dir`: Output directory for models (default: ./models)

### 2. Verifying Signatures

Verify if two signatures belong to the same person:

```bash
python verify_signature.py \
  --student_id "STU001" \
  --reference "./reference_signature.jpg" \
  --test "./test_signature.jpg"
```

**Parameters**:
- `--student_id`: Student ID (must match training)
- `--reference`: Path to reference signature image
- `--test`: Path to test signature image
- `--model_dir`: Directory containing trained models (default: ./models)

## File Structure

```
siamese_training/
├── siamese_model.py          # Core Siamese network implementation
├── train_siamese.py          # Training script
├── verify_signature.py       # Verification script
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── models/                  # Trained models (created after training)
    ├── siamese_STU001.h5
    ├── siamese_STU001_metadata.json
    └── training_history_STU001.png
```

## Hardware Optimization

This implementation is optimized for your system:

- **Memory Management**: Automatic memory growth for GPU
- **Batch Size**: Default 16 (adjustable based on available RAM)
- **Image Size**: 224x224 pixels (optimal for your hardware)
- **Pair Limiting**: Prevents memory overflow with large datasets

## Expected Performance

With your hardware (Ryzen 5 3400G, 16GB RAM):

- **Training Time**: ~2-5 minutes per student (50 epochs)
- **Verification Speed**: ~100-200ms per verification
- **Memory Usage**: ~2-4GB during training
- **Model Size**: ~10-15MB per student model

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce `--batch_size` to 8 or 4
   - Reduce image resolution in `siamese_model.py`

2. **No Images Found**:
   - Check file paths and extensions
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF

3. **Model Not Found**:
   - Ensure training completed successfully
   - Check model directory path

### Performance Tips

1. **For Better Accuracy**:
   - Use more training images (10+ genuine, 5+ forged)
   - Increase training epochs
   - Ensure good quality images

2. **For Faster Training**:
   - Reduce image size to 128x128
   - Use fewer training pairs
   - Reduce embedding dimension

## Next Steps

After training, you can:

1. **Integrate with Web App**: Use the trained models in your React app
2. **Batch Verification**: Process multiple signatures at once
3. **Model Export**: Convert to TensorFlow.js for browser use
4. **Performance Monitoring**: Track verification accuracy over time

## Support

If you encounter issues:

1. Check the console output for error messages
2. Verify all file paths are correct
3. Ensure sufficient disk space for models
4. Check Python and TensorFlow versions