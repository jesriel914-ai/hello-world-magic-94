# 🎓 Google Colab Setup Instructions

Follow these steps to run the Siamese Signature Identification API on Google Colab with a T4 GPU.

## 📋 Prerequisites

1. **Google Account** with access to Colab
2. **GPU Runtime**: You need a T4 GPU (free tier or Colab Pro)

## 🚀 Step-by-Step Setup

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **GPU type** to **T4** (if available)
4. Click **Save**

### Step 3: Upload Code to Colab

**Option A: Clone from GitHub**
```python
!git clone https://github.com/jesriel914-ai/hello-world-magic-94.git
%cd hello-world-magic-94/siamese_training
```

**Option B: Upload Files Manually**
1. Click the folder icon on the left sidebar
2. Upload all files from `siamese_training/` folder:
   - `preprocessing.py`
   - `model.py`
   - `training.py`
   - `identification.py`
   - `main.py`
   - `requirements.txt`
   - `run_colab.py`

### Step 4: Run the Setup Script

```python
!python run_colab.py
```

This script will:
1. ✅ Check GPU availability
2. ✅ Install all dependencies
3. ✅ Download Cloudflare tunnel
4. ✅ Start Flask API server
5. ✅ Expose server via Cloudflare tunnel

### Step 5: Copy the URL

After running, you'll see output like:

```
🌍 Starting Cloudflare tunnel...
   Copy the URL that starts with https://

+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://addressing-connectors-moisture-twice.trycloudflare.com                           |
+--------------------------------------------------------------------------------------------+
```

**Copy this URL!** (e.g., `https://addressing-connectors-moisture-twice.trycloudflare.com`)

### Step 6: Update Frontend Configuration

1. Open your project: `src/ai-model-siamese/lib/SiameseService.ts`
2. Find line 9: `const API_BASE_URL = "..."`
3. Replace with your Cloudflare URL:

```typescript
const API_BASE_URL = "https://addressing-connectors-moisture-twice.trycloudflare.com";
```

4. Save the file

### Step 7: Test the Connection

In a new Colab cell, run:

```python
import requests

# Replace with your actual URL
API_URL = "https://your-url.trycloudflare.com"

# Test health check
response = requests.get(f"{API_URL}/health")
print(response.json())
```

You should see:
```json
{
  "status": "online",
  "message": "Siamese Signature Identification API",
  "model_loaded": false,
  "total_students": 0,
  ...
}
```

### Step 8: Start Training from Frontend

1. Go to your web app
2. Navigate to **Siamese Signature Model Training** page
3. Add students and upload signature samples
4. Click **Train Model**
5. Training will happen in Colab (you can monitor progress in frontend)

## 🎯 Alternative: Manual Setup

If you prefer manual control:

### 1. Install Dependencies
```python
!pip install -q -r requirements.txt
```

### 2. Start Flask Server (Background)
```python
import subprocess
import threading

flask_process = subprocess.Popen(
    ["python", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
```

### 3. Install Cloudflare
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
```

### 4. Start Tunnel
```python
!./cloudflared-linux-amd64 tunnel --url http://localhost:5000
```

## 📊 Monitoring Training

### Check Training Status

```python
import requests

response = requests.get(f"{API_URL}/training_status")
status = response.json()

print(f"Training: {status['is_training']}")
print(f"Progress: {status['progress']:.1f}%")
print(f"Current: {status['current_student']}")
```

### Check Model Status

```python
response = requests.get(f"{API_URL}/status")
status = response.json()

print(f"Trained: {status['is_trained']}")
print(f"Students: {status['total_students']}")
print(f"Embeddings: {status['total_embeddings']}")
```

## 🐛 Troubleshooting

### Problem: "No GPU detected"

**Solution**:
1. Runtime → Change runtime type → GPU → T4
2. Restart runtime
3. Re-run setup script

### Problem: "Module not found"

**Solution**:
```python
!pip install tensorflow==2.19.0 flask flask-cors opencv-python pillow
```

### Problem: Cloudflare tunnel not starting

**Solution**:
1. Check Flask is running: `!curl http://localhost:5000/health`
2. Restart tunnel manually:
   ```python
   !./cloudflared-linux-amd64 tunnel --url http://localhost:5000
   ```

### Problem: Training runs but no progress

**Solution**:
1. Check logs in Colab output
2. Verify GPU is being used:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Problem: "Training already in progress" error

**Solution**:
Wait for current training to finish or restart runtime

## 💡 Tips

1. **Keep Colab Tab Open**: Don't close Colab while training
2. **Colab Timeout**: Free tier disconnects after ~12 hours
3. **Save Models**: Download models before closing Colab
4. **GPU Quota**: Free tier has limited GPU hours per day
5. **Tunnel URL Changes**: URL changes each time you restart tunnel

## 📥 Download Trained Model

After training, download the model:

```python
# Via API
import requests

response = requests.get(f"{API_URL}/download_model")
with open('siamese_model.zip', 'wb') as f:
    f.write(response.content)

# Download to your PC
from google.colab import files
files.download('siamese_model.zip')
```

## 🔄 Restart Instructions

If Colab disconnects:

1. Reconnect to runtime
2. Re-run setup script
3. Update frontend with new Cloudflare URL
4. Model is saved and will auto-load

## ✅ Success Checklist

- [ ] GPU is enabled (T4)
- [ ] Dependencies installed
- [ ] Flask server running
- [ ] Cloudflare tunnel active
- [ ] Frontend can reach API (`/health` works)
- [ ] Training completes successfully
- [ ] Identification works

## 📞 Need Help?

Common issues:
1. **GPU out of memory**: Reduce batch_size in training request
2. **Slow training**: Ensure GPU is enabled
3. **API unreachable**: Check Cloudflare URL is correct
4. **Training fails**: Check image data is valid base64

---

**Happy Training! 🎉**
