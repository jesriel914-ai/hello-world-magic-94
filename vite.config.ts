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

// Create Express app for backend endpoints
const backendApp = express();
backendApp.use(cors());
backendApp.use(express.json({ limit: '50mb' }));
backendApp.use(express.urlencoded({ extended: true }));

// Health check endpoint
backendApp.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'OK', message: 'Backend is running' });
});

// Simple upload endpoint (placeholder)
backendApp.post('/api/upload', (req: Request, res: Response) => {
  res.json({ success: true, message: 'Upload endpoint placeholder' });
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
        connectedTo?: string; // ID of connected device
      }
      
      const clients = new Map<string, ConnectedClient>();
      
      // Generate simple device name based on device type
      function generateDeviceName(userAgent: string, deviceType: 'desktop' | 'mobile' | 'unknown'): string {
        if (deviceType === 'desktop') {
          // Try to get hostname from environment variables
          const hostname = process.env.COMPUTERNAME || 
                         process.env.HOSTNAME || 
                         process.env.USERNAME || 
                         'Desktop';
          return `${hostname} - ${getBrowserName(userAgent)}`;
        } else {
          return 'Mobile Device';
        }
      }
      
      // Helper function to generate unique client ID
      function generateClientId(): string {
        return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
      }

      // Helper function to get browser name
      function getBrowserName(userAgent: string): string {
        if (/Chrome/i.test(userAgent) && !/Edg/i.test(userAgent)) return 'Chrome';
        if (/Firefox/i.test(userAgent)) return 'Firefox';
        if (/Safari/i.test(userAgent) && !/Chrome/i.test(userAgent)) return 'Safari';
        if (/Edg/i.test(userAgent)) return 'Edge';
        if (/MSIE|Trident/i.test(userAgent)) return 'Internet Explorer';
        return 'Unknown Browser';
      }


      // Broadcast device list to all clients
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
        
        // Determine device type - more precise detection
        let deviceType: 'desktop' | 'mobile' | 'unknown' = 'unknown';
        
        // Check for mobile devices first (more specific patterns)
        if (/Mobile|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent) && 
            !/Windows NT/i.test(userAgent)) {
          deviceType = 'mobile';
        } 
        // Check for desktop devices (exclude Android/Linux mobile)
        else if (/Windows NT|Macintosh|X11|CrOS/i.test(userAgent) || 
                 (/Linux/i.test(userAgent) && !/Android/i.test(userAgent))) {
          deviceType = 'desktop';
        }
        // Fallback: if no clear desktop indicators but has browser, assume desktop
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
          ipAddress,
          userAgent: userAgent.substring(0, 100) + '...',
          isMobileDetected: /Mobile|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent) && !/Windows NT/i.test(userAgent),
          isDesktopDetected: /Windows NT|Macintosh|Linux|X11|CrOS/i.test(userAgent)
        });
        console.log(`📊 Total connected clients: ${clients.size}`);
        
        // Send client their device info
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
        
        // Broadcast updated device list
        broadcastDeviceList();
        
        ws.on('message', (message: string) => {
          try {
            const data = JSON.parse(message);
            
            // Handle different message types
            if (data.type === 'connection-request') {
              // Handle connection request
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                // Forward connection request to target device
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
              // Handle connection response
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                // Forward connection response to target device
                targetClient.ws.send(JSON.stringify({
                  type: 'connection-response',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName,
                    accepted: data.data.accepted
                  },
                  timestamp: Date.now()
                }));
                
                // If accepted, update connection status
                if (data.data.accepted) {
                  client.isConnected = true;
                  client.connectedTo = targetClientId;
                  targetClient.isConnected = true;
                  targetClient.connectedTo = clientId;
                  
                  broadcastDeviceList();
                }
              }
            } else if (data.type === 'disconnect-request') {
              // Handle disconnect request
              const targetClientId = data.data.targetDeviceId;
              const targetClient = clients.get(targetClientId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                // Forward disconnect request to target device
                targetClient.ws.send(JSON.stringify({
                  type: 'disconnect-request',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName
                  },
                  timestamp: Date.now()
                }));
                
                // Send disconnected message to target device
                targetClient.ws.send(JSON.stringify({
                  type: 'disconnected',
                  data: {
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName
                  },
                  timestamp: Date.now()
                }));
                
                // Update connection status for both devices
                client.isConnected = false;
                client.connectedTo = undefined;
                targetClient.isConnected = false;
                targetClient.connectedTo = undefined;
                
                console.log(`🔌 Disconnected ${client.deviceName} from ${targetClient.deviceName}`);
                
                broadcastDeviceList();
              }
            } else if (data.type === 'device-info-update') {
              // Handle device info update from mobile device
              console.log('📱 Received device info update:', data.data);
              
              // Update the client's device information
              if (data.data.deviceName) {
                client.deviceName = data.data.deviceName;
                console.log(`✅ Updated device name for ${clientId}: ${client.deviceName}`);
              }
              
              if (data.data.deviceType) {
                client.deviceType = data.data.deviceType;
                console.log(`✅ Updated device type for ${clientId}: ${client.deviceType}`);
              }
              
              // Broadcast updated device list
              broadcastDeviceList();
              
            } else if (data.type === 'preview-update' || data.type === 'prediction-result' || data.type === 'predicting-status' || data.type === 'model-status' || data.type === 'mode-change') {
              // Only broadcast screen sharing messages between manually connected devices
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
          
          // Notify connected device about disconnection
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
              
              // Update connected client status
              connectedClient.isConnected = false;
              connectedClient.connectedTo = undefined;
            }
          }
          
          clients.delete(clientId);
          console.log(`📊 Total connected clients: ${clients.size}`);
          
          // Broadcast updated device list
          broadcastDeviceList();
        });
      });
      
      // Handle WebSocket upgrades
      server.httpServer?.on('upgrade', (req: IncomingMessage, socket: Duplex, head: Buffer) => {
        if (req.url === '/ws') {
          wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
            wss.emit('connection', ws, req);
          });
        }
      });
      
      // Handle backend API requests
      server.middlewares.use('/api', backendApp);
      
      console.log('✅ WebSocket server configured on /ws');
      console.log('✅ Backend API configured on /api');
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
      '.ngrok-free.app', // 👈 allow all ngrok subdomains
      '.trycloudflare.com' // 👈 allow all Cloudflare tunnel subdomains
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
