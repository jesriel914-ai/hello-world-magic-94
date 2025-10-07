# Google Colab Siamese Training Backend Setup Guide

## Prerequisites Checklist
- ✅ Google account
- ✅ `siamese_training` folder uploaded to Google Drive containing:
  - `main.py`
  - `siamese_trainer.py`
  - `siamese_verifier.py`
  - `requirements.txt`
- ✅ Ngrok account (free) - Sign up at [ngrok.com](https://ngrok.com)

---

## Step 1: Prepare Ngrok Auth Token

1. Go to [ngrok.com](https://ngrok.com) and create a free account
2. Navigate to [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Copy your authtoken (looks like: `2a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p`)
4. Keep this token handy - you'll need it in Step 5

---

## Step 2: Open Your Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Open your `Siamese-training-backend.ipynb` notebook
3. Make sure you're using GPU:
   - Click **Runtime** → **Change runtime type**
   - Select **T4 GPU** under Hardware accelerator
   - Click **Save**

---

## Step 3: Mount Google Drive

Create a **new code cell** at the top of your notebook and paste this code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**What happens:**
- Click **Run** (or press `Shift + Enter`)
- A popup will appear asking for permission
- Click on your Google account
- Allow Colab to access your Drive
- You'll see: `Mounted at /content/drive`

---

## Step 4: Navigate to Your Project Directory

In the **next code cell**, add:

```python
import os

# Change to your siamese_training directory
# Adjust the path based on where you uploaded the folder in Drive
os.chdir('/content/drive/MyDrive/siamese_training')

# Verify you're in the right place
print("Current directory:", os.getcwd())
print("\nFiles in directory:")
!ls -la
```

**Expected output:**
```
Current directory: /content/drive/MyDrive/siamese_training
Files in directory:
main.py
siamese_trainer.py
siamese_verifier.py
requirements.txt
```

**Note:** If your folder is in a different location, adjust the path:
- Root of Drive: `/content/drive/MyDrive/siamese_training`
- Inside a folder: `/content/drive/MyDrive/Projects/siamese_training`

---

## Step 5: Install Dependencies

Create a **new code cell**:

```python
# Install required packages
!pip install -r /content/drive/MyDrive/siamese_training/requirements.txt

print("✅ Installation complete!")
```

**This will take 2-3 minutes.** You'll see lots of text - that's normal!

---

## Step 6: Set Up Ngrok

Create a **new code cell**:

```python
import os
from pyngrok import ngrok

# Set your ngrok authtoken (replace with YOUR token)
NGROK_TOKEN = "33iR8WTI5iZB8riQF3S3b9xagRc_6ww25VCQzCyDzyCJ636qk"  # ← Paste your token here
os.environ['NGROK_AUTH_TOKEN'] = NGROK_TOKEN

# Configure ngrok
ngrok.set_auth_token(NGROK_TOKEN)

print("✅ Ngrok configured successfully!")
```

**⚠️ IMPORTANT:** Replace `"YOUR_NGROK_TOKEN_HERE"` with your actual token from Step 1!

---

## Step 7: Verify GPU Availability

Create a **new code cell**:

```python
import tensorflow as tf

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU AVAILABLE: {gpus[0].name}")
    print(f"   GPU Type: {tf.test.gpu_device_name()}")
else:
    print("⚠️ No GPU found - using CPU (will be slower)")

# Check TensorFlow version
print(f"\n📦 TensorFlow version: {tf.__version__}")
```

**Expected output:**
```
✅ GPU AVAILABLE: /physical_device:GPU:0
   GPU Type: /device:GPU:0
📦 TensorFlow version: 2.19.0
```

---

## Step 8: Start the Training Server

Create a **new code cell**:

```python
# Start the Flask training API server
!python main.py
```

**What you'll see:**
```
==============================================================
  SIAMESE SIGNATURE TRAINING API (GPU-OPTIMIZED)
==============================================================
  Environment: Google Colab
  Public URL: https://abc123.ngrok-free.app
  
  🌍 UPDATE YOUR FRONTEND .env WITH:
  VITE_SIAMESE_API_URL=https://abc123.ngrok-free.app
  
  Status: Starting...
==============================================================

Available Endpoints:
  POST /api/train              - Train student model
  POST /api/verify             - Verify signature
  GET  /api/models/list        - List trained models
  GET  /api/model/status/:id   - Check model status
  GET  /api/health             - Health check
==============================================================
```

**🎉 Your server is now running!**

---

## Step 9: Update Your Frontend

1. Copy the **Public URL** from the output (e.g., `https://abc123.ngrok-free.app`)
2. In your frontend project, update `.env` file:

```bash
VITE_SIAMESE_API_URL=https://abc123.ngrok-free.app
```

3. Restart your frontend dev server

---

## Step 10: Test the Connection

In your frontend, you should now be able to:
- ✅ Add students
- ✅ Upload signatures
- ✅ Train models
- ✅ Verify signatures

---

## Complete Notebook Structure

Here's what your final notebook should look like:

```
Cell 1: Mount Drive
Cell 2: Navigate to project
Cell 3: Install dependencies
Cell 4: Configure ngrok
Cell 5: Check GPU
Cell 6: Start server (keep running)
```

---

## Troubleshooting Guide

### Issue: "No such file or directory"
**Solution:** Check your Drive path in Step 4. Try:
```python
# List all directories in MyDrive
!ls /content/drive/MyDrive/
```

### Issue: "GPU not found"
**Solution:** 
1. Go to Runtime → Change runtime type
2. Select **T4 GPU**
3. Save and restart runtime

### Issue: "Ngrok authentication failed"
**Solution:**
- Verify your token is correct (no extra spaces)
- Make sure you copied the entire token
- Try regenerating token from ngrok dashboard

### Issue: "Module not found"
**Solution:**
```python
# Reinstall dependencies
!pip install --upgrade -r requirements.txt
```

### Issue: "Port already in use"
**Solution:**
```python
# Restart the runtime
# Runtime → Restart runtime
```

### Issue: Frontend can't connect
**Solution:**
1. Make sure the ngrok URL is correct in `.env`
2. Check that Cell 6 is still running (has the spinner)
3. Try visiting the ngrok URL in browser - you should see "healthy"

---

## Important Notes

### ⏰ Session Timeout
- Colab free disconnects after ~12 hours or 90 minutes idle
- You'll need to rerun all cells when this happens
- The ngrok URL will change each time

### 💾 Saving Models
- Models are saved to: `/content/drive/MyDrive/siamese_training/models/`
- They persist in Drive even after session ends
- Each student gets their own folder

### 🔄 Restarting
When you restart Colab:
1. Run Cell 1 (Mount Drive) - Click "Connect to Google Drive"
2. Run Cell 2 (Navigate)
3. Run Cell 3 (Install - can be skipped if recently done)
4. Run Cell 4 (Ngrok config)
5. Run Cell 5 (Check GPU)
6. Run Cell 6 (Start server)
7. Update frontend `.env` with new ngrok URL

### 📊 Monitoring
Watch the Cell 6 output for:
- Training progress
- Verification requests
- Any errors

---

## Quick Command Reference

### Check current directory:
```python
!pwd
```

### List files:
```python
!ls -la
```

### View file content:
```python
!cat main.py
```

### Check Python version:
```python
!python --version
```

### Check disk space:
```python
!df -h
```

### View logs:
```python
# Server logs appear in Cell 6 output
```

---

## Performance Tips

### Training Speed
- **With GPU:** ~30-60 seconds per student
- **Without GPU:** ~5-10 minutes per student

### Batch Processing
- You can train multiple students at once
- GPU handles 32 batch size efficiently

### Memory Management
- Free plan: 12.7 GB RAM, 15 GB GPU RAM
- Can train ~50 students before needing restart
- Models are automatically saved to Drive

---

## Need Help?

**Cell is stuck?**
- Click the stop button (⏹️) next to the cell
- Runtime → Restart runtime

**Want to see more details?**
- Check the Cell 6 output for detailed logs
- Training shows: loss, accuracy, precision, recall
- Verification shows: distance metrics, confidence

**Models not saving?**
- Check Drive storage space
- Verify path in Cell 2 is correct
- Models folder should auto-create

---

## Success Checklist

Before starting training, verify:
- ✅ Drive is mounted
- ✅ GPU is available
- ✅ Dependencies installed
- ✅ Ngrok configured
- ✅ Server running
- ✅ Frontend `.env` updated
- ✅ Can see "healthy" at ngrok URL

---

## Example Training Flow

1. **Frontend:** Add student "2021-0001"
2. **Frontend:** Upload 10 genuine + 20 forged signatures
3. **Frontend:** Click "Train Model"
4. **Colab Cell 6 shows:**
   ```
   Training Enhanced Model for Student: 2021-0001
   Genuine: 10, Forged: 20
   Total pairs: 450
   Epochs: 35/100
   Final accuracy: 96.8%
   ✅ Training completed!
   ```
5. **Frontend:** Model ready for verification

---

## Next Steps After Setup

1. Test with 1-2 students first
2. Verify signatures work correctly
3. Then batch upload more students
4. Monitor Colab Cell 6 for any issues

**Happy Training! 🚀**