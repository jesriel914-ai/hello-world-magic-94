# S3 Upload Backend Server Setup

This backend server provides secure S3 upload functionality for the AI Signature Model Training application.

## Features

- **Secure S3 Upload**: Handles AWS S3 operations server-side to avoid exposing credentials in the browser
- **Model Metadata Management**: Stores and manages model metadata alongside the actual model files
- **REST API**: Provides clean REST endpoints for frontend integration
- **Automatic Startup**: Starts automatically with `npm run dev`
- **Shared Configuration**: Uses existing `.env` file - no separate configuration needed

## Prerequisites

- Node.js 16.0.0 or higher
- AWS Account with S3 access
- AWS S3 bucket created
- AWS credentials already configured in existing `.env` file

## Setup Instructions

### 1. Install Dependencies

```bash
npm install express cors dotenv concurrently
```

### 2. Verify AWS Configuration

Ensure your existing `.env` file contains the AWS credentials (it should already be there):

```env
# AWS S3 Configuration for AI Models
NEXT_PUBLIC_AWS_ACCESS_KEY_ID=your_aws_access_key_here
NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
NEXT_PUBLIC_AWS_REGION=us-east-1
NEXT_PUBLIC_S3_BUCKET=signatureai-uploads

### 3. AWS S3 Bucket Setup

1. Create an S3 bucket in your AWS account
2. Configure bucket permissions to allow write access from your server
3. Ensure the bucket name matches `NEXT_PUBLIC_S3_BUCKET` in your `.env` file

### 4. Start the Application

**Automatic Startup with Frontend**

```bash
# This will start both the frontend (Vite) and backend (S3 server) simultaneously
npm run dev
```

The application will start with:
- Frontend: http://localhost:5173 (or other available port)
- Backend: http://localhost:8001

**Manual Backend Start (if needed)**

```bash
# Start only the backend server
npm run dev:backend
```

### 5. Test the Server

```bash
# Test health endpoint
curl http://localhost:8001/health

# Expected response:
# { "status": "ok", "message": "S3 Upload Server is running" }
```

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and configuration information.

### Upload Model to S3
```
POST /api/upload-model-to-s3
```
Uploads trained AI model and metadata to S3.

**Request Body:**
```json
{
  "modelData": {...},
  "metadata": {
    "version": "1.0",
    "createdAt": "2024-01-01T00:00:00.000Z",
    "environment": "development",
    "modelArchitecture": {...},
    "trainingConfig": {...},
    "performance": {...},
    "classes": [...],
    "storage": {...}
  },
  "studentId": "123",
  "modelType": "individual"
}
```

**Response:**
```json
{
  "success": true,
  "location": "https://bucket.s3.region.amazonaws.com/path/to/model.json",
  "etag": "\"abc123\"",
  "metadata": {...},
  "message": "Model uploaded successfully to S3"
}
```

### Get Trained Models
```
GET /api/trained-models
```
Retrieves list of trained models (placeholder implementation).

### Delete Model
```
DELETE /api/trained-models/:modelId
```
Deletes a model from S3 and database (placeholder implementation).

## S3 File Structure

Models are organized in S3 with the following structure:

```
ai-models/
├── development/
│   ├── individual/
│   │   └── student-{studentId}/
│   │       └── v{version}/
│   │           ├── model_{timestamp}.json
│   │           └── metadata_{timestamp}.json
│   └── global/
│       └── v{version}/
│           ├── model_{timestamp}.json
│           └── metadata_{timestamp}.json
├── staging/
│   └── ... (same structure)
└── production/
    └── ... (same structure)
```

## Integration with Frontend

The backend is already integrated with the frontend through the `AIModelService.ts` file. The `uploadTrainedModelToS3` method now uses the backend API endpoint instead of trying to upload directly from the browser.

## Security Considerations

1. **Environment Variables**: AWS credentials are stored in environment variables and never exposed to the client
2. **Server-Side Operations**: All S3 operations are performed on the server, keeping credentials secure
3. **CORS**: The server is configured to allow requests from your frontend domain
4. **Input Validation**: The server validates incoming requests before processing

## Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   - Ensure your `.env.server` file exists and has the correct AWS credentials
   - Verify that the AWS credentials have the necessary permissions

2. **S3 Access Denied**
   - Check that your AWS credentials have S3 write permissions
   - Verify the S3 bucket name and region are correct
   - Ensure the bucket policy allows write access

3. **Server Won't Start**
   - Check that Node.js is installed and the correct version
   - Verify all dependencies are installed
   - Check for port conflicts (default is 8001)

4. **Frontend Can't Connect**
   - Ensure the backend server is running
   - Check that the frontend is configured to use the correct backend URL (http://localhost:8001)
   - Verify CORS settings allow requests from your frontend domain

### Testing the Server

Test that the server is running correctly:

```bash
curl http://localhost:8001/health
```

You should receive a response like:
```json
{
  "status": "OK",
  "message": "S3 Upload Server is running"
}
```

## Development

### Adding New Endpoints

1. Add the new route handler in `server.js`
2. Implement the business logic
3. Add proper error handling
4. Test the endpoint thoroughly

### Environment-Specific Configurations

You can create different environment files for different stages:

- `.env.server.development` for development
- `.env.server.staging` for staging
- `.env.server.production` for production

Then start the server with the appropriate environment:

```bash
NODE_ENV=production node server.js
```

## License

This backend server is part of the AI Signature Model Training application.
