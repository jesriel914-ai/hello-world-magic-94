#!/bin/bash
# GPU Instance Setup Script for Signature AI Training
# This script sets up a fresh GPU instance with all required dependencies

set -e

echo "🚀 Starting GPU instance setup for Signature AI Training"
echo "Timestamp: $(date)"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    htop \
    vim \
    build-essential \
    cmake \
    pkg-config

# Install AWS CLI v2
echo "☁️ Installing AWS CLI v2..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    echo "✅ AWS CLI installed successfully"
else
    echo "✅ AWS CLI already installed"
fi

# Install Docker (optional, for containerized training)
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
    echo "✅ Docker installed successfully"
else
    echo "✅ Docker already installed"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip

# Install TensorFlow with GPU support
echo "🧠 Installing TensorFlow with GPU support..."
pip3 install tensorflow[and-cuda]==2.13.0

# Install other ML dependencies
echo "📚 Installing ML dependencies..."
pip3 install \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    scipy==1.10.1 \
    pillow==10.2.0 \
    opencv-python==4.9.0.80 \
    boto3==1.34.34 \
    botocore==1.34.34 \
    fastapi==0.92.0 \
    uvicorn[standard]==0.22.0 \
    python-multipart==0.0.9 \
    supabase>=2.6.0 \
    httpx>=0.27 \
    requests==2.31.0 \
    python-dotenv==1.0.1 \
    orjson==3.9.15 \
    pydantic==1.10.13 \
    aiohttp>=3.8 \
    typing-extensions==4.5.0

# Verify GPU availability
echo "🔍 Verifying GPU setup..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
"

# Create training directory
echo "📁 Creating training directory..."
sudo mkdir -p /home/ubuntu/ai-training
sudo chown -R ubuntu:ubuntu /home/ubuntu/ai-training

# Create logs directory
echo "📝 Creating logs directory..."
sudo mkdir -p /var/log/ai-training
sudo chown -R ubuntu:ubuntu /var/log/ai-training

# Install monitoring tools (optional)
echo "📊 Installing monitoring tools..."
pip3 install psutil nvidia-ml-py3

# Create systemd service for training (optional)
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/ai-training.service > /dev/null <<EOF
[Unit]
Description=Signature AI Training Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-training
ExecStart=/usr/bin/python3 /home/ubuntu/ai-training/train_gpu.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service (but don't start it)
sudo systemctl daemon-reload
sudo systemctl enable ai-training.service

# Create health check script
echo "🏥 Creating health check script..."
sudo tee /home/ubuntu/ai-training/health_check.py > /dev/null <<'EOF'
#!/usr/bin/env python3
import tensorflow as tf
import sys
import json

def check_gpu_health():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("❌ No GPU detected")
            return False
        
        # Test GPU memory
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.random.normal([1000, 1000])
                result = tf.reduce_sum(test_tensor)
                print(f"✅ GPU computation test passed: {result.numpy()}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ GPU health check failed: {e}")
        return False

if __name__ == "__main__":
    healthy = check_gpu_health()
    sys.exit(0 if healthy else 1)
EOF

sudo chmod +x /home/ubuntu/ai-training/health_check.py

# Create cleanup script
echo "🧹 Creating cleanup script..."
sudo tee /home/ubuntu/ai-training/cleanup.sh > /dev/null <<'EOF'
#!/bin/bash
# Cleanup script for GPU instance

echo "🧹 Starting cleanup process..."

# Clean up temporary files
sudo rm -rf /tmp/*_models
sudo rm -rf /tmp/training_*
sudo rm -rf /tmp/*.json

# Clean up logs older than 7 days
find /var/log/ai-training -name "*.log" -mtime +7 -delete

# Clean up Python cache
find /home/ubuntu -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ubuntu -name "*.pyc" -delete 2>/dev/null || true

# Clean up package cache
sudo apt-get autoremove -y
sudo apt-get autoclean

echo "✅ Cleanup completed"
EOF

sudo chmod +x /home/ubuntu/ai-training/cleanup.sh

# Set up log rotation
echo "📋 Setting up log rotation..."
sudo tee /etc/logrotate.d/ai-training > /dev/null <<EOF
/var/log/ai-training/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
}
EOF

# Create monitoring script
echo "📊 Creating monitoring script..."
sudo tee /home/ubuntu/ai-training/monitor.py > /dev/null <<'EOF'
#!/usr/bin/env python3
import psutil
import json
import time
from datetime import datetime

def get_system_stats():
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg(),
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0 if 'tf' in globals() else False
    }

if __name__ == "__main__":
    try:
        import tensorflow as tf
        stats = get_system_stats()
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error: {e}")
EOF

sudo chmod +x /home/ubuntu/ai-training/monitor.py

# Final verification
echo "🔍 Running final verification..."
python3 /home/ubuntu/ai-training/health_check.py

# Set up cron job for cleanup
echo "⏰ Setting up cleanup cron job..."
(crontab -l 2>/dev/null; echo "0 2 * * * /home/ubuntu/ai-training/cleanup.sh") | crontab -

echo ""
echo "🎉 GPU instance setup completed successfully!"
echo ""
echo "📋 Summary:"
echo "  ✅ System packages updated"
echo "  ✅ AWS CLI installed"
echo "  ✅ Docker installed"
echo "  ✅ Python dependencies installed"
echo "  ✅ TensorFlow with GPU support installed"
echo "  ✅ Training directory created: /home/ubuntu/ai-training"
echo "  ✅ Logs directory created: /var/log/ai-training"
echo "  ✅ Health check script created"
echo "  ✅ Cleanup script created"
echo "  ✅ Monitoring script created"
echo "  ✅ Log rotation configured"
echo "  ✅ Cleanup cron job scheduled"
echo ""
echo "🚀 Your GPU instance is ready for AI training!"
echo ""
echo "📝 Next steps:"
echo "  1. Upload your training code to /home/ubuntu/ai-training/"
echo "  2. Configure your .env file with AWS credentials"
echo "  3. Test the setup with: python3 /home/ubuntu/ai-training/health_check.py"
echo "  4. Start training with your training script"
echo ""
echo "🔧 Useful commands:"
echo "  - Check GPU status: python3 /home/ubuntu/ai-training/health_check.py"
echo "  - Monitor system: python3 /home/ubuntu/ai-training/monitor.py"
echo "  - Clean up: /home/ubuntu/ai-training/cleanup.sh"
echo "  - View logs: tail -f /var/log/ai-training/*.log"
echo ""
echo "Setup completed at: $(date)"