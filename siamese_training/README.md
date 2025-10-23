# Siamese Neural Network for Signature Identification

This is a complete implementation of a Siamese Neural Network for signature identification with **incremental learning** support.

## 🎯 Features

- ✅ **Incremental Learning**: Add new students without retraining from scratch
- ✅ **Single Result Identification**: Returns only the owner (not top-5)
- ✅ **Non-Signature Detection**: Rejects random photos and non-signature images
- ✅ **Batch Training**: Train multiple students at once
- ✅ **GPU Optimized**: T4 GPU (12GB RAM) with memory management
- ✅ **Genuine-Only Training**: Works with only genuine signatures
- ✅ **REST API**: Flask API with CORS support
- ✅ **Background Training**: Training continues even if frontend disconnects

## 📁 File Structure

```
siamese_training/
├── preprocessing.py       # Image preprocessing pipeline
├── model.py              # Siamese network architecture (MobileNetV2)
├── training.py           # Training logic with incremental learning
├── identification.py     # Signature identification/verification
├── main.py              # Flask API server
├── requirements.txt     # Python dependencies
└── model_storage/       # Created automatically
    ├── signature_encoder.h5
    ├── embeddings.pkl
    └── metadata.json
```

## 🚀 Quick Start (Google Colab)

### 1. Install Dependencies

```bash
!pip install -r requirements.txt
```

### 2. Start the API Server

```python
# In Colab, run in background
!python main.py &
```

### 3. Expose with Cloudflare Tunnel

```bash
# Install cloudflared
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!./cloudflared-linux-amd64 tunnel --url http://localhost:5000
```

Copy the generated URL (e.g., `https://xyz.trycloudflare.com`) and update it in:
- `src/ai-model-siamese/lib/SiameseService.ts` (line 9)

## 📋 API Endpoints

### Health Check
```
GET /health
```

### Model Status
```
GET /status
```

### Batch Training
```
POST /train_batch

Body:
{
  "students": [
    {
      "student_id": "2020-001",
      "genuine_samples": ["base64_image1", "base64_image2", ...],
      "forged_samples": []
    }
  ],
  "epochs": 30,
  "batch_size": 96
}
```

### Training Status (Polling)
```
GET /training_status
```

### Identify Signature
```
POST /classify

Body:
{
  "image": "base64_encoded_image"
}

Response:
{
  "identified": true,
  "student_id": "2020-001",
  "confidence": 0.95,
  "distance": 0.234,
  "decision": "ACCEPT",
  "message": "✅ High confidence match!",
  "threshold_info": {
    "accept_threshold": 0.5,
    "reject_threshold": 0.8,
    "nonsig_threshold": 1.2
  }
}
```

### Verify Signature (1:1)
```
POST /verify

Body:
{
  "image": "base64_encoded_image",
  "student_id": "2020-001"
}
```

### Delete Student
```
POST /delete_student

Body:
{
  "student_id": "2020-001"
}
```

### List Students
```
GET /list_students
```

### Download Model
```
GET /download_model
```

## 🧠 Model Architecture

```
Input (224x224x3 signature image)
    ↓
MobileNetV2 (pretrained on ImageNet, α=1.0)
    ↓
Global Average Pooling
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(128) - Embedding Layer
    ↓
L2 Normalization
    ↓
Output: 128-dimensional embedding vector
```

### Loss Function
- **Contrastive Loss** with margin = 1.0
- Optimized for Euclidean distance comparison

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 96 (optimized for T4 GPU)
- **Epochs**: 30 (with early stopping)
- **Validation Split**: 10%

## 🎓 How Incremental Learning Works

### Day 1: Initial Training
1. Train with 1+ students
2. Each student provides genuine signatures (2-55 samples)
3. Model creates embeddings for all samples
4. Save: model weights, embeddings, metadata

### Day 2+: Incremental Update
1. **Detection**: System detects new vs existing students
2. **For new students**: Train fully, create embeddings
3. **For existing students**: Generate new embeddings, merge with old
4. **Progressive fine-tuning**: Freeze most layers, train only last layers
5. **Replay**: Use some old samples to prevent forgetting
6. **Save**: Update model, embeddings, metadata

### Key Features
- ✅ No need to retrain from scratch
- ✅ Existing students' data is preserved
- ✅ New students can be added anytime
- ✅ Model adapts incrementally

## 📊 Decision Thresholds

| Distance | Decision | Confidence | Meaning |
|----------|----------|------------|---------|
| < 0.5 | ACCEPT | High (75-95%) | Strong match - signature owner identified |
| 0.5-0.8 | UNCERTAIN | Medium (45-75%) | Weak match - needs more training samples |
| 0.8-1.2 | UNKNOWN | Low (15-45%) | No match - student not in database |
| > 1.2 | NON_SIGNATURE | Very Low (<15%) | Not a signature (random photo) |

## 🔧 GPU Optimization

### Memory Management
- **Memory Growth**: Enabled (allocates only what's needed)
- **Batch Size**: 96 (optimized for 12GB T4 GPU)
- **Mixed Precision**: Can be enabled for faster training

### Tips for T4 GPU (12GB RAM)
1. Use batch_size = 96 (optimal)
2. Don't train more than 100 students at once
3. If OOM error: reduce batch_size to 64 or 48
4. Clear GPU memory between large batches

## 📝 Training Best Practices

### Minimum Requirements
- **Minimum students**: 1 (for testing)
- **Recommended students**: 5+ (for good performance)
- **Genuine samples per student**: 2 minimum, 50-55 recommended
- **Forged samples**: Optional (not used in current implementation)

### Data Quality Tips
1. ✅ Clear signature images (no blur)
2. ✅ Good lighting and contrast
3. ✅ Consistent background (white/light background)
4. ✅ Signature centered in image
5. ❌ Avoid rotated or distorted signatures
6. ❌ Avoid partial signatures

### Data Augmentation
The system automatically applies:
- Random rotation (-15° to +15°)
- Random scaling (0.9x to 1.1x)
- Random brightness (0.8x to 1.2x)

## 🐛 Troubleshooting

### Problem: GPU Out of Memory
**Solution**: Reduce batch_size from 96 to 64 or 48

### Problem: Training too slow
**Solution**: 
1. Check if GPU is being used (`/health` endpoint shows GPU status)
2. Reduce number of students per batch
3. Reduce epochs from 30 to 20

### Problem: Low accuracy
**Solution**:
1. Add more genuine samples per student (50-55 recommended)
2. Improve image quality
3. Train with more students (model learns better with more data)

### Problem: "Unknown" for trained students
**Solution**:
1. Check if student was actually trained (use `/list_students`)
2. Ensure signature quality matches training samples
3. Retrain with more diverse samples

## 📈 Performance Metrics

### Expected Results (After Training)
- **True owner**: Confidence 90-95%, Distance < 0.3
- **Other trained students**: Confidence < 50%, Distance > 0.8
- **Unknown students**: Confidence < 20%, Distance > 1.0
- **Non-signatures**: Distance > 1.2

### Training Time (T4 GPU)
- **5 students, 50 samples each**: ~5-10 minutes
- **20 students, 50 samples each**: ~15-25 minutes
- **50 students, 50 samples each**: ~30-45 minutes

## 🔐 Security Notes

1. **CORS**: Currently allows all origins (`*`) - restrict in production
2. **No authentication**: Add JWT or API keys for production
3. **File upload limits**: No size limits - add validation
4. **Rate limiting**: Not implemented - consider adding

## 📞 Support

For issues or questions:
1. Check the logs in Colab
2. Verify GPU is available
3. Ensure Cloudflare tunnel is running
4. Check frontend API URL is correct

## 🎉 Success Indicators

✅ Training completes without errors
✅ `/status` shows correct number of students
✅ Trained signatures get >90% confidence
✅ Unknown signatures are rejected
✅ Random photos are marked as "NON_SIGNATURE"

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-23  
**Optimized for**: Google Colab T4 GPU (12GB RAM)
