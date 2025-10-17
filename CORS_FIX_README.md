# CORS Fix for Siamese Training API

## Problem
The frontend is getting a CORS error when trying to make requests to the Python backend:
```
Access to fetch at 'https://unfantastic-delmar-incondite.ngrok-free.dev/api/train' from origin 'http://localhost:5173' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Solution Applied

### 1. Fixed CORS Configuration in Python Backend

Updated `siamese_training/main.py` with proper CORS handling:

```python
# Enhanced CORS configuration
CORS(app, 
     resources={r"/api/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False,
         "max_age": 3600
     }})

# Add explicit OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization, ngrok-skip-browser-warning")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        response.headers.add('Access-Control-Max-Age', "3600")
        return response

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, ngrok-skip-browser-warning')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response
```

### 2. Key Changes Made

1. **Added explicit OPTIONS handler**: Handles preflight requests properly
2. **Added after_request handler**: Ensures all responses have CORS headers
3. **Imported make_response**: Required for creating custom responses
4. **Comprehensive header support**: Includes all necessary headers for ngrok and browser compatibility

## How to Test the Fix

### Option 1: Test with Local Backend

1. **Start the Python backend locally:**
   ```bash
   cd siamese_training
   pip install -r requirements.txt
   python main.py
   ```

2. **Update frontend environment:**
   Create a `.env` file in the project root:
   ```bash
   VITE_SIAMESE_API_URL=http://localhost:5000
   ```

3. **Start the frontend:**
   ```bash
   npm run dev
   ```

### Option 2: Test with ngrok Backend

1. **Start the Python backend with ngrok:**
   ```bash
   cd siamese_training
   python main.py
   # In another terminal:
   ngrok http 5000
   ```

2. **Update frontend environment:**
   ```bash
   VITE_SIAMESE_API_URL=https://your-ngrok-url.ngrok-free.app
   ```

3. **Test the connection:**
   ```bash
   python3 test_cors.py
   ```

## Verification

The fix addresses the following CORS issues:

1. ✅ **Preflight OPTIONS requests** - Now properly handled
2. ✅ **Access-Control-Allow-Origin header** - Added to all responses
3. ✅ **ngrok-skip-browser-warning header** - Properly allowed
4. ✅ **Content-Type header** - Properly handled
5. ✅ **All HTTP methods** - GET, POST, OPTIONS supported

## Files Modified

- `siamese_training/main.py` - Updated CORS configuration
- `test_cors.py` - Created test script
- `test_cors.html` - Created browser test file

## Next Steps

1. Deploy the updated Python backend to your ngrok URL
2. Update your frontend's `.env` file with the correct API URL
3. Test the training functionality

The CORS error should now be resolved!