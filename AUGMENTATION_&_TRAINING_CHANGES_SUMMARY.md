# Augmentation & Training Pipeline Changes

## Summary of Changes

### 1. Augmentation System Refactoring

**Before:** 15+ separate augmentation cases  
**After:** 5 merged, parametric augmentation groups

#### New Augmentation Groups

| Group | Weight | Description | Real-World Impact |
|-------|--------|-------------|-------------------|
| **Geometric Transforms** | 28% | Rotation, perspective skew, cropping | Handles phone angles, partial signatures |
| **Focus & Motion** | 25% | Distance scaling, blur types, hand shake | Simulates camera focus issues, movement |
| **Lighting & Ink** | 22% | Brightness, contrast, color temp, glare | Different lighting conditions, pen colors |
| **Background & Context** | 15% | Paper types, document elements, distractors | Real document scanning scenarios |
| **Camera Artifacts** | 10% | Sensor noise, resolution loss, compression | Low-end phone cameras, video streaming |

### 2. Augmentation Count Change

**Changed from 2x to 4x augmentation**

For 25 classes × 50 samples:
- **Before:** 1,250 original + 2,500 augmented = **3,750 total**
- **After:** 1,250 original + 5,000 augmented = **6,250 total**

### 3. Training Pipeline Improvements

#### Model Architecture Enhancements
```
Input (1280 features from MobileNet)
  ↓
Dense(256) + BatchNorm + Dropout(0.5)  ← Increased from 128
  ↓
Dense(128) + BatchNorm + Dropout(0.4)  ← New layer
  ↓
Dense(64) + Dropout(0.3)
  ↓
Output (num_classes)
```

#### Training Configuration Changes

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Epochs | 100 | 80 | More data = fewer epochs needed |
| Batch Size | 32 | 16 | Better stability with more data |
| Validation Split | 15% | 20% | Better generalization monitoring |
| Learning Rate | 0.0001 | 0.0001 | Kept optimal for stability |

### 4. Expected Results

#### Performance Predictions

**With 2x Augmentation:**
- Training accuracy: 85-92%
- Validation accuracy: 78-85%
- Overfitting risk: Moderate

**With 4x Augmentation:**
- Training accuracy: 80-86% (lower, but healthier)
- Validation accuracy: 82-88% (better generalization)
- Overfitting risk: Lower
- Real-world camera performance: Significantly better

#### Why Training Accuracy Decreases

This is **expected and good**:
- More diverse augmented data = harder to memorize
- Lower training acc + higher validation acc = **less overfitting**
- Better generalization to real-world camera variations

### 5. Memory Optimization

Added throughout the pipeline:
- Periodic `tf.nextFrame()` calls during feature extraction
- Cleanup every 50 samples during processing
- Cleanup every 10 epochs during training
- Immediate disposal of individual feature tensors after stacking

### 6. Real-World Camera Coverage

The refactored augmentations now cover all critical mobile camera scenarios:

✅ **Distance variations** - Users moving phone closer/farther  
✅ **Focus issues** - Out-of-focus, motion blur, autofocus hunting  
✅ **Lighting conditions** - Bright, dim, warm, cool, shadowed  
✅ **Paper backgrounds** - White, yellow pad, notebook, beige, with lines  
✅ **Hand shake** - Natural tremor during handheld capture  
✅ **Angle variations** - Rotated, skewed, perspective distortion  
✅ **Partial signatures** - Cropped by bounding box detection  
✅ **Camera quality** - Noise, low resolution, compression artifacts  
✅ **Document context** - Form fields, text, distractors  
✅ **Glare/overexposure** - Lighting reflections on paper  

### 7. Training Time Estimate

For 25 classes × 50 samples with 4x augmentation:
- Feature extraction: ~8-12 minutes
- Training (80 epochs): ~15-20 minutes
- **Total: ~25-30 minutes** (on mid-range hardware)

### 8. Usage Notes

#### Memory Considerations
- Your system: R5 3400G, 16GB RAM, 2GB VRAM
- 6,250 samples should fit comfortably
- Watch for memory warnings in console
- If issues occur, reduce to 3x augmentation

#### Progressive Training Strategy
1. Start with 3x augmentation
2. Monitor validation accuracy
3. If still overfitting, increase to 4x
4. If memory issues, drop back to 3x

### 9. Code Integration

Both artifacts are **drop-in replacements**:

1. **augmentImage()** - Replace your existing function
2. **trainModel()** - Replace your existing training function
3. Ensure `AUGMENTATION_COUNT = 4` is set
4. No other changes needed to your pipeline

### 10. Next Steps

After training with 4x augmentation:
1. Test with live camera feed
2. Measure real-world accuracy
3. Identify remaining failure cases
4. Fine-tune augmentation weights if needed

The refactored system is cleaner, more maintainable, and better aligned with real-world mobile camera signature detection scenarios.