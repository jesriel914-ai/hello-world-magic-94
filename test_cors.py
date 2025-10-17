#!/usr/bin/env python3
"""
Test script to verify CORS configuration
"""
import requests
import json

def test_cors_preflight():
    """Test CORS preflight request"""
    url = "https://unfantastic-delmar-incondite.ngrok-free.dev/api/train"
    
    # Test preflight OPTIONS request
    headers = {
        'Origin': 'http://localhost:5173',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type, ngrok-skip-browser-warning'
    }
    
    print("Testing CORS preflight request...")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.options(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"  {key}: {value}")
        
        if response.status_code == 200:
            print("✅ CORS preflight request successful!")
        else:
            print("❌ CORS preflight request failed!")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_cors_actual_request():
    """Test actual POST request with CORS"""
    url = "https://unfantastic-delmar-incondite.ngrok-free.dev/api/health"
    
    headers = {
        'Origin': 'http://localhost:5173',
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
    }
    
    print("\nTesting CORS actual request...")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"  {key}: {value}")
        
        if response.status_code == 200:
            print("✅ CORS actual request successful!")
            print(f"Response: {response.json()}")
        else:
            print("❌ CORS actual request failed!")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_cors_preflight()
    test_cors_actual_request()