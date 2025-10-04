const express = require('express');
const cors = require('cors');
const { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command } = require('@aws-sdk/client-s3');
require('dotenv').config();

const app = express();
const PORT = process.env.BACKEND_PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Initialize S3 Client with proper error handling
let s3Client = null;
let BUCKET_NAME = 'signatureai-uploads';

// Check if AWS credentials are available
const hasAwsCredentials = process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID && 
                         process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY;

if (hasAwsCredentials) {
  try {
    s3Client = new S3Client({
      region: process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY,
      },
    });
    BUCKET_NAME = process.env.NEXT_PUBLIC_S3_BUCKET || 'signatureai-uploads';
    console.log('✅ S3 client initialized successfully');
  } catch (error) {
    console.error('❌ Failed to initialize S3 client:', error);
    s3Client = null;
  }
} else {
  console.warn('⚠️ AWS credentials not found. S3 uploads will be disabled.');
  console.warn('   Please set NEXT_PUBLIC_AWS_ACCESS_KEY_ID and NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY');
}

// Helper functions for S3
async function streamToString(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
  });
}

async function streamToBuffer(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks)));
  });
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Backend server is running',
    port: PORT,
    s3: {
      configured: !!s3Client,
      bucket: BUCKET_NAME,
      region: process.env.NEXT_PUBLIC_AWS_REGION
    }
  });
});

// Upload model to S3 - FIXED for 3-file structure
app.post('/api/upload-model-to-s3', async (req, res) => {
  try {
    // Check if S3 is available
    if (!s3Client) {
      return res.status(503).json({
        success: false,
        message: 'S3 service is not available. Please configure AWS credentials in your .env file.',
        error: 'S3_NOT_CONFIGURED',
        instructions: [
          '1. Copy .env.example to .env',
          '2. Add your AWS credentials to the .env file',
          '3. Restart the backend server'
        ]
      });
    }

    const { modelData, metadata, studentId, modelType, isThreeFileFormat } = req.body;

    if (!modelData) {
      return res.status(400).json({
        success: false,
        message: 'Missing modelData'
      });
    }

    // Generate timestamp and folder structure
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const folderPath = `ai-models/${timestamp}`;

    if (isThreeFileFormat) {
      // NEW FORMAT: 3 files (model.json, weights.bin, metadata.json)
      console.log(`📦 Uploading 3-file model to ${folderPath}`);

      // 1. Upload model.json
      const modelJsonKey = `${folderPath}/model.json`;
      await s3Client.send(new PutObjectCommand({
        Bucket: BUCKET_NAME,
        Key: modelJsonKey,
        Body: modelData.modelJson,
        ContentType: 'application/json',
      }));
      console.log(`✅ Uploaded model.json`);

      // 2. Upload weights.bin (decode base64)
      let weightsBase64 = modelData.weightsBin;
      if (weightsBase64.includes(',')) {
        weightsBase64 = weightsBase64.split(',')[1];
      }
      
      const weightsBuffer = Buffer.from(weightsBase64, 'base64');
      const weightsKey = `${folderPath}/weights.bin`;
      
      await s3Client.send(new PutObjectCommand({
        Bucket: BUCKET_NAME,
        Key: weightsKey,
        Body: weightsBuffer,
        ContentType: 'application/octet-stream',
      }));
      console.log(`✅ Uploaded weights.bin (${weightsBuffer.length} bytes)`);

      // 3. Upload metadata.json
      const metadataKey = `${folderPath}/metadata.json`;
      await s3Client.send(new PutObjectCommand({
        Bucket: BUCKET_NAME,
        Key: metadataKey,
        Body: modelData.metadataJson,
        ContentType: 'application/json',
      }));
      console.log(`✅ Uploaded metadata.json`);

      // Return success
      res.json({
        success: true,
        location: `https://${BUCKET_NAME}.s3.${process.env.NEXT_PUBLIC_AWS_REGION}.amazonaws.com/${modelJsonKey}`,
        metadata: {
          storage: {
            location: 's3',
            bucket: BUCKET_NAME,
            region: process.env.NEXT_PUBLIC_AWS_REGION,
            modelKey: modelJsonKey,
            weightsKey: weightsKey,
            metadataKey: metadataKey
          }
        },
        message: 'Model uploaded successfully (3-file format)'
      });

    } else {
      // OLD FORMAT: Not supported
      console.log('⚠️ Old format upload attempt rejected');
      res.status(400).json({
        success: false,
        message: '5-file format is deprecated. Please use 3-file format (isThreeFileFormat=true)'
      });
    }

  } catch (error) {
    console.error('❌ Error uploading model:', error);
    res.status(500).json({
      success: false,
      message: error instanceof Error ? error.message : 'Failed to upload model to S3'
    });
  }
});

// Download model from S3 - FIXED for 3-file structure
app.get('/api/download-model/:modelUuid', async (req, res) => {
  try {
    // Check if S3 is available
    if (!s3Client) {
      return res.status(503).json({
        success: false,
        message: 'S3 service is not available. Please configure AWS credentials in your .env file.',
        error: 'S3_NOT_CONFIGURED'
      });
    }

    const { modelUuid } = req.params;
    console.log(`📥 Downloading model: ${modelUuid}`);

    // List all objects in ai-models/ to find the model
    const listResponse = await s3Client.send(new ListObjectsV2Command({
      Bucket: BUCKET_NAME,
      Prefix: 'ai-models/',
      MaxKeys: 1000
    }));

    if (!listResponse.Contents || listResponse.Contents.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'No models found in S3'
      });
    }

    // Find the model folder (look for metadata.json files)
    const modelFolders = listResponse.Contents
      .filter(obj => obj.Key.endsWith('/metadata.json'))
      .map(obj => obj.Key.replace('/metadata.json', ''));

    if (modelFolders.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'No model metadata found'
      });
    }

    // For now, get the most recent model (last folder)
    const modelFolder = modelFolders[modelFolders.length - 1];
    console.log(`📁 Found model folder: ${modelFolder}`);

    // Download the 3 files
    const modelJsonKey = `${modelFolder}/model.json`;
    const weightsKey = `${modelFolder}/weights.bin`;
    const metadataKey = `${modelFolder}/metadata.json`;

    try {
      // Download model.json
      const modelResponse = await s3Client.send(new GetObjectCommand({
        Bucket: BUCKET_NAME,
        Key: modelJsonKey
      }));
      const modelJson = await streamToString(modelResponse.Body);

      // Download weights.bin
      const weightsResponse = await s3Client.send(new GetObjectCommand({
        Bucket: BUCKET_NAME,
        Key: weightsKey
      }));
      const weightsBuffer = await streamToBuffer(weightsResponse.Body);
      const weightsBase64 = weightsBuffer.toString('base64');

      // Download metadata.json
      const metadataResponse = await s3Client.send(new GetObjectCommand({
        Bucket: BUCKET_NAME,
        Key: metadataKey
      }));
      const metadataJson = await streamToString(metadataResponse.Body);

      // Combine all data
      const combinedData = {
        modelJson: modelJson,
        weightsBin: weightsBase64,
        metadataJson: metadataJson
      };

      res.json({
        success: true,
        data: JSON.stringify(combinedData),
        message: 'Model downloaded successfully (3-file format)'
      });

    } catch (downloadError) {
      console.error('❌ Error downloading model files:', downloadError);
      res.status(500).json({
        success: false,
        message: 'Failed to download model files from S3'
      });
    }

  } catch (error) {
    console.error('❌ Error downloading model:', error);
    res.status(500).json({
      success: false,
      message: error instanceof Error ? error.message : 'Failed to download model from S3'
    });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📡 API endpoints available:`);
  console.log(`   - POST /api/upload-model-to-s3`);
  console.log(`   - GET /api/download-model/:modelUuid`);
  console.log(`   - GET /health`);
  console.log(`🔧 S3 Status: ${s3Client ? '✅ Configured' : '❌ Not configured'}`);
});