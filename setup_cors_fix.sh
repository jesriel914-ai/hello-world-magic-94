#!/bin/bash

echo "🔧 Setting up CORS fix for Siamese Training API..."

# Check if we're in the right directory
if [ ! -f "siamese_training/main.py" ]; then
    echo "❌ Error: siamese_training/main.py not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Found siamese_training/main.py"

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
cd siamese_training

if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt not found. Creating basic requirements..."
    cat > requirements.txt << EOF
flask==2.3.3
flask-cors==4.0.0
tensorflow==2.13.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
scikit-learn==1.3.0
faiss-cpu==1.7.4
EOF
fi

echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔧 CORS configuration has been updated in main.py"
echo ""
echo "🚀 To start the backend server:"
echo "   cd siamese_training"
echo "   python main.py"
echo ""
echo "🌐 To test with ngrok:"
echo "   # In another terminal:"
echo "   ngrok http 5000"
echo ""
echo "📝 Don't forget to update your frontend .env file:"
echo "   VITE_SIAMESE_API_URL=http://localhost:5000"
echo "   # or your ngrok URL:"
echo "   VITE_SIAMESE_API_URL=https://your-ngrok-url.ngrok-free.app"
echo ""
echo "✅ Setup complete! The CORS fix is ready to use."