# Google Colab CORS Test Cell
# Copy and paste this into a new cell in your Colab notebook

import requests
import json

def test_cors_fix(ngrok_url):
    """Test CORS configuration with the ngrok URL"""
    
    print("🔧 Testing CORS configuration...")
    print(f"🌐 Testing URL: {ngrok_url}")
    
    # Test 1: Health check (GET request)
    print("\n1️⃣ Testing GET request (health check)...")
    try:
        headers = {
            'Origin': 'http://localhost:5173',
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        }
        
        response = requests.get(f"{ngrok_url}/api/health", headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ GET request successful!")
            data = response.json()
            print(f"   📊 Service: {data.get('service', 'Unknown')}")
            print(f"   🔧 Version: {data.get('version', 'Unknown')}")
        else:
            print("   ❌ GET request failed!")
            
    except Exception as e:
        print(f"   ❌ GET request error: {e}")
    
    # Test 2: Preflight OPTIONS request
    print("\n2️⃣ Testing OPTIONS request (preflight)...")
    try:
        headers = {
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type, ngrok-skip-browser-warning'
        }
        
        response = requests.options(f"{ngrok_url}/api/train", headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ OPTIONS request successful!")
            cors_headers = {k: v for k, v in response.headers.items() if 'access-control' in k.lower()}
            if cors_headers:
                print("   📋 CORS Headers found:")
                for key, value in cors_headers.items():
                    print(f"      {key}: {value}")
            else:
                print("   ⚠️  No CORS headers found in response")
        else:
            print("   ❌ OPTIONS request failed!")
            
    except Exception as e:
        print(f"   ❌ OPTIONS request error: {e}")
    
    # Test 3: Simulate frontend training request
    print("\n3️⃣ Testing POST request (simulate training)...")
    try:
        headers = {
            'Origin': 'http://localhost:5173',
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        }
        
        # Send a minimal training request (will fail due to missing data, but CORS should work)
        test_data = {
            'student_id': 'test_student',
            'genuine_samples': [],
            'forged_samples': []
        }
        
        response = requests.post(f"{ngrok_url}/api/train", 
                               headers=headers, 
                               json=test_data, 
                               timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code in [200, 400]:  # 400 is expected due to missing data
            print("   ✅ POST request CORS successful!")
            if response.status_code == 400:
                print("   ℹ️  Expected 400 error due to missing training data")
        else:
            print("   ❌ POST request failed!")
            
    except Exception as e:
        print(f"   ❌ POST request error: {e}")
    
    print("\n" + "="*50)
    print("🎯 CORS Test Complete!")
    print("If all tests show ✅, your CORS fix is working!")
    print("="*50)

# Usage: Replace with your actual ngrok URL
# test_cors_fix("https://your-ngrok-url.ngrok-free.app")

print("📋 To use this test:")
print("1. Replace 'your-ngrok-url' with your actual ngrok URL")
print("2. Run: test_cors_fix('https://your-actual-url.ngrok-free.app')")
print("3. Check if all tests show ✅")