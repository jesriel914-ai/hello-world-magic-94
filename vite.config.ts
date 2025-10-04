import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { WebSocketServer } from 'ws';
import express, { Request, Response } from 'express';
import cors from 'cors';
import 'dotenv/config';
import type { ViteDevServer } from 'vite';
import type { IncomingMessage } from 'http';
import type { Duplex } from 'stream';
import type WebSocket from 'ws';
import { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';

// Create Express app for backend endpoints
const backendApp = express();
backendApp.use(cors());
backendApp.use(express.json({ limit: '50mb' }));
backendApp.use(express.urlencoded({ limit: '50mb', extended: true }));

// Initialize S3 Client
const s3Client = new S3Client({
  region: process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY || '',
  },
});

const BUCKET_NAME = process.env.NEXT_PUBLIC_S3_BUCKET || 'signatureai-uploads';

// Helper functions for S3
async function streamToString(stream: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    stream.on('data', (chunk: Buffer) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
  });
}

async function streamToBuffer(stream: any): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    stream.on('data', (chunk: Buffer) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks)));
  });
}

// Health check endpoint
backendApp.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    message: 'Backend is running',
    s3: {
      bucket: BUCKET_NAME,
      region: process.env.NEXT_PUBLIC_AWS_REGION
    }
  });
});

// Upload model to S3 - FIXED for 3-file structure
backendApp.post('/api/upload-model-to-s3', async (req: Request, res: Response) => {
  try {
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
backendApp.get('/api/download-model/:modelUuid', async (req: Request, res: Response) => {
  try {
    const { modelUuid } = req.params;
    console.log(`📥 Downloading model: ${modelUuid}`);

    // List all objects in ai-models/ to find the model
    const listResponse = await s3Client.send(new ListObjectsV2Command({
      Bucket: BUCKET_NAME,
      Prefix: 'ai-models/'
    }));

    // Find the most recent folder containing model.json
    let modelFolder: string | null = null;
    const contents = listResponse.Contents || [];
    
    // Sort by last modified (most recent first)
    contents.sort((a, b) => {
      const dateA = a.LastModified ? a.LastModified.getTime() : 0;
      const dateB = b.LastModified ? b.LastModified.getTime() : 0;
      return dateB - dateA;
    });

    // Find the folder with model.json
    for (const obj of contents) {
      if (obj.Key && obj.Key.endsWith('model.json')) {
        modelFolder = obj.Key.substring(0, obj.Key.lastIndexOf('/'));
        break;
      }
    }

    if (!modelFolder) {
      return res.status(404).json({
        success: false,
        error: 'Model not found in S3'
      });
    }

    console.log(`📂 Found model folder: ${modelFolder}`);

    // Download the 3 files
    const modelJsonKey = `${modelFolder}/model.json`;
    const weightsKey = `${modelFolder}/weights.bin`;
    const metadataKey = `${modelFolder}/metadata.json`;

    // 1. Get model.json
    const modelJsonResponse = await s3Client.send(new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: modelJsonKey
    }));
    const modelJsonContent = await streamToString(modelJsonResponse.Body);

    // 2. Get weights.bin
    const weightsResponse = await s3Client.send(new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: weightsKey
    }));
    const weightsBuffer = await streamToBuffer(weightsResponse.Body);
    const weightsBase64 = `data:application/octet-stream;base64,${weightsBuffer.toString('base64')}`;

    // 3. Get metadata.json
    const metadataResponse = await s3Client.send(new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: metadataKey
    }));
    const metadataContent = await streamToString(metadataResponse.Body);

    // Combine into single response
    const combinedData = {
      modelJson: modelJsonContent,
      weightsBin: weightsBase64,
      metadataJson: metadataContent
    };

    console.log(`✅ Model downloaded successfully`);

    res.json({
      success: true,
      data: JSON.stringify(combinedData),
      message: 'Model downloaded successfully'
    });

  } catch (error) {
    console.error('❌ Error downloading model:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to download model'
    });
  }
});

// Create a custom plugin for WebSocket and backend integration
const customPlugin = () => {
  return {
    name: 'custom-websocket-backend',
    configureServer(server: ViteDevServer) {
      // Create WebSocket server
      const wss = new WebSocketServer({ noServer: true });
      
      // Store connected clients with device information
      interface ConnectedClient {
        ws: WebSocket;
        id: string;
        deviceType: 'desktop' | 'mobile' | 'unknown';
        deviceName: string;
        ipAddress: string;
        connectedAt: number;
        isConnected: boolean;
        connectedTo?: string;
      }
      
      const clients = new Map<string, ConnectedClient>();
      
      // Generate simple device name based on device type
      function generateDeviceName(userAgent: string, deviceType: 'desktop' | 'mobile' | 'unknown'): string {
        if (deviceType === 'desktop') {
          const hostname = process.env.COMPUTERNAME || 
                         process.env.HOSTNAME || 
                         process.env.USERNAME || 
                         'Desktop';
          return `${hostname} - ${getBrowserName(userAgent)}`;
        } else {
          return 'Mobile Device';
        }
      }
      
      function generateClientId(): string {
        return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
      }

      function getBrowserName(userAgent: string): string {
        if (/Chrome/i.test(userAgent) && !/Edg/i.test(userAgent)) return 'Chrome';
        if (/Firefox/i.test(userAgent)) return 'Firefox';
        if (/Safari/i.test(userAgent) && !/Chrome/i.test(userAgent)) return 'Safari';
        if (/Edg/i.test(userAgent)) return 'Edge';
        if (/MSIE|Trident/i.test(userAgent)) return 'Internet Explorer';
        return 'Unknown Browser';
      }

      function broadcastDeviceList(): void {
        const deviceList = Array.from(clients.values()).map(client => ({
          id: client.id,
          deviceType: client.deviceType,
          deviceName: client.deviceName,
          ipAddress: client.ipAddress,
          isConnected: client.isConnected,
          connectedTo: client.connectedTo
        }));
        
        const message = {
          type: 'device-list',
          data: {
            devices: deviceList
          },
          timestamp: Date.now()
        };
        
        clients.forEach((client) => {
          if (client.ws.readyState === 1) {
            try {
              client.ws.send(JSON.stringify(message));
            } catch (error) {
              console.error('❌ Error sending device list:', error);
            }
          }
        });
      }
      
      wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
        const clientId = generateClientId();
        const userAgent = req.headers['user-agent'] || '';
        const ipAddress = req.socket.remoteAddress || 'unknown';
        
        let deviceType: 'desktop' | 'mobile' | 'unknown' = 'unknown';
        
        if (/Mobile|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent) && 
            !/Windows NT/i.test(userAgent)) {
          deviceType = 'mobile';
        } 
        else if (/Windows NT|Macintosh|X11|CrOS/i.test(userAgent) || 
                 (/Linux/i.test(userAgent) && !/Android/i.test(userAgent))) {
          deviceType = 'desktop';
        }
        else if (/Chrome|Firefox|Safari|Edge|MSIE/i.test(userAgent)) {
          deviceType = 'desktop';
        }
        
        const deviceName = generateDeviceName(userAgent, deviceType);
        
        const client: ConnectedClient = {
          ws,
          id: clientId,
          deviceType,
          deviceName,
          ipAddress,
          connectedAt: Date.now(),
          isConnected: false
        };
        
        clients.set(clientId, client);
        
        console.log('🔗 New WebSocket connection:', {
          id: clientId,
          deviceName,
          deviceType,
          ipAddress
        });
        console.log(`📊 Total connected clients: ${clients.size}`);
        
        ws.send(JSON.stringify({
          type: 'device-info',
          data: {
            id: clientId,
            deviceType,
            deviceName,
            ipAddress
          },
          timestamp: Date.now()
        }));
        
        broadcastDeviceList();
        
        ws.on('message', (message: string) => {
          try {
            const data = JSON.parse(message);
            
            if (data.type === 'connection-request') {
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                targetClient.ws.send(JSON.stringify({
                  type: 'connection-request',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName,
                    fromDeviceType: client.deviceType,
                    requestType: 'connect'
                  },
                  timestamp: Date.now()
                }));
              }
            } else if (data.type === 'connection-response') {
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                targetClient.ws.send(JSON.stringify({
                  type: 'connection-response',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName,
                    accepted: data.data.accepted
                  },
                  timestamp: Date.now()
                }));
                
                if (data.data.accepted) {
                  client.isConnected = true;
                  client.connectedTo = targetClientId;
                  targetClient.isConnected = true;
                  targetClient.connectedTo = clientId;
                  
                  broadcastDeviceList();
                }
              }
            } else if (data.type === 'disconnect-request') {
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                targetClient.ws.send(JSON.stringify({
                  type: 'disconnect-request',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName
                  },
                  timestamp: Date.now()
                }));
                
                targetClient.ws.send(JSON.stringify({
                  type: 'disconnected',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName
                  },
                  timestamp: Date.now()
                }));
                
                client.isConnected = false;
                client.connectedTo = undefined;
                targetClient.isConnected = false;
                targetClient.connectedTo = undefined;
                
                console.log(`🔌 Disconnected ${client.deviceName} from ${targetClient.deviceName}`);
                
                broadcastDeviceList();
              }
            } else if (data.type === 'device-info-update') {
              console.log('📱 Received device info update:', data.data);
              
              if (data.data.deviceName) {
                client.deviceName = data.data.deviceName;
                console.log(`✅ Updated device name for ${clientId}: ${client.deviceName}`);
              }
              
              if (data.data.deviceType) {
                client.deviceType = data.data.deviceType;
                console.log(`✅ Updated device type for ${clientId}: ${client.deviceType}`);
              }
              
              broadcastDeviceList();
              
            } else if (data.type === 'preview-update' || data.type === 'prediction-result' || data.type === 'predicting-status' || data.type === 'model-status' || data.type === 'mode-change') {
              if (client.isConnected && client.connectedTo) {
                const targetClient = clients.get(client.connectedTo);
                if (targetClient && targetClient.ws.readyState === 1) {
                  try {
                    targetClient.ws.send(JSON.stringify({
                      ...data,
                      source: client.deviceType,
                      timestamp: Date.now()
                    }));
                    console.log(`📡 Broadcasting ${data.type} from ${client.deviceName} to ${targetClient.deviceName}`);
                  } catch (error) {
                    console.error('❌ Error sending screen sharing message:', error);
                  }
                }
              } else {
                console.log(`⚠️ Ignoring ${data.type} from ${client.deviceName} - no manual connection established`);
              }
            }
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        });
        
        ws.on('close', () => {
          console.log('🔌 Client disconnected:', {
            id: clientId,
            deviceName: client.deviceName
          });
          
          if (client.connectedTo) {
            const connectedClient = clients.get(client.connectedTo);
            if (connectedClient && connectedClient.ws.readyState === 1) {
              const disconnectMessage = {
                type: 'disconnected',
                data: {
                  fromDeviceId: clientId,
                  fromDeviceName: client.deviceName
                },
                timestamp: Date.now()
              };
              
              connectedClient.ws.send(JSON.stringify(disconnectMessage));
              
              connectedClient.isConnected = false;
              connectedClient.connectedTo = undefined;
            }
          }
          
          clients.delete(clientId);
          console.log(`📊 Total connected clients: ${clients.size}`);
          
          broadcastDeviceList();
        });
      });
      
      server.httpServer?.on('upgrade', (req: IncomingMessage, socket: Duplex, head: Buffer) => {
        if (req.url === '/ws') {
          wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
            wss.emit('connection', ws, req);
          });
        }
      });
      
      server.middlewares.use('/api', backendApp);
      
      console.log('✅ WebSocket server configured on /ws');
      console.log('✅ Backend API configured on /api');
      console.log('✅ S3 endpoints ready on /api/upload-model-to-s3 and /api/download-model/:modelUuid');
    }
  };
};

export default defineConfig({
  plugins: [react(), customPlugin()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts: [
      '.ngrok-free.app',
      '.trycloudflare.com'
    ],
    hmr: {
      port: 5173,
      protocol: 'ws',
      host: 'localhost'
    }
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src")
    }
  }
});