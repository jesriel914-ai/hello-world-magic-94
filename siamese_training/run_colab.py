#!/usr/bin/env python3
# filepath: siamese_training/run_colab.py
"""
Google Colab Startup Script for Siamese Signature Identification
Run this script in Colab to start the API server with Cloudflare tunnel
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print("🚀 SIAMESE SIGNATURE IDENTIFICATION - COLAB SETUP")
    print("="*70 + "\n")

def check_gpu():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No GPU detected - training will be slow")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def install_cloudflared():
    """Install and setup Cloudflare tunnel"""
    print("\n🌐 Setting up Cloudflare tunnel...")
    
    try:
        # Download cloudflared
        if not os.path.exists("cloudflared"):
            print("   Downloading cloudflared...")
            subprocess.run([
                "wget", "-q", "-O", "cloudflared",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            ], check=True)
            
            # Make executable
            os.chmod("cloudflared", 0o755)
        
        print("✅ Cloudflare tunnel ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Cloudflare: {e}")
        return False

def start_flask_server():
    """Start Flask server in background"""
    print("\n🔥 Starting Flask API server...")
    
    try:
        # Start Flask in background
        flask_process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if flask_process.poll() is None:
            print("✅ Flask server started on http://localhost:5000")
            return flask_process
        else:
            print("❌ Flask server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Flask: {e}")
        return None

def start_cloudflare_tunnel():
    """Start Cloudflare tunnel"""
    print("\n🌍 Starting Cloudflare tunnel...")
    print("   This will expose your local server to the internet...")
    print("   Copy the URL that starts with https://")
    print("   Update it in: src/ai-model-siamese/lib/SiameseService.ts (line 9)")
    print("\n" + "="*70 + "\n")
    
    try:
        # Start tunnel (this will run in foreground and print URL)
        subprocess.run([
            "./cloudflared", "tunnel", "--url", "http://localhost:5000"
        ])
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tunnel stopped by user")
    except Exception as e:
        print(f"\n❌ Tunnel error: {e}")

def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Check GPU
    has_gpu = check_gpu()
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return
    
    # Step 3: Setup Cloudflare
    if not install_cloudflared():
        print("\n❌ Setup failed: Could not setup Cloudflare tunnel")
        return
    
    # Step 4: Start Flask server
    flask_process = start_flask_server()
    if flask_process is None:
        print("\n❌ Setup failed: Could not start Flask server")
        return
    
    # Step 5: Start Cloudflare tunnel (blocks until Ctrl+C)
    try:
        start_cloudflare_tunnel()
    finally:
        # Cleanup
        if flask_process:
            print("\n🛑 Stopping Flask server...")
            flask_process.terminate()
            flask_process.wait()
        
        print("\n✅ Cleanup complete")

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run setup
    main()
