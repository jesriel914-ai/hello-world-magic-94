import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { WebSocketServer } from 'ws';
import type { ViteDevServer } from 'vite';
import type { IncomingMessage } from 'http';
import type { Duplex } from 'stream';
import type WebSocket from 'ws';

// Create a custom plugin for WebSocket integration
const customPlugin = () => {
  return {
    name: 'custom-websocket',
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
        
        console.log('🔌 New client connected:', {
          id: clientId,
          deviceType: client.deviceType,
          deviceName: client.deviceName,
          ipAddress: client.ipAddress
        });
        
        console.log(`📊 Total connected clients: ${clients.size}`);
        
        // Send initial device list
        broadcastDeviceList();
        
        ws.on('message', (data: Buffer) => {
          try {
            const message = JSON.parse(data.toString());
            console.log(`📨 Message from ${client.deviceName}:`, message.type);
            
            if (message.type === 'connect-to-device') {
              const targetDeviceId = message.data.targetDeviceId;
              const targetClient = clients.get(targetDeviceId);
              
              if (targetClient && targetClient.ws.readyState === 1) {
                // Establish connection
                client.isConnected = true;
                client.connectedTo = targetDeviceId;
                targetClient.isConnected = true;
                targetClient.connectedTo = clientId;
                
                console.log(`🔗 Connection established: ${client.deviceName} ↔ ${targetClient.deviceName}`);
                
                // Notify both clients
                const connectMessage = {
                  type: 'connected',
                  data: {
                    toDeviceId: targetDeviceId,
                    toDeviceName: targetClient.deviceName
                  },
                  timestamp: Date.now()
                };
                
                ws.send(JSON.stringify(connectMessage));
                
                const targetConnectMessage = {
                  type: 'connected',
                  data: {
                    toDeviceId: clientId,
                    toDeviceName: client.deviceName
                  },
                  timestamp: Date.now()
                };
                
                targetClient.ws.send(JSON.stringify(targetConnectMessage));
                
                // Update device list
                broadcastDeviceList();
              } else {
                console.log(`❌ Target device ${targetDeviceId} not found or not available`);
                ws.send(JSON.stringify({
                  type: 'error',
                  data: { message: 'Target device not found or not available' },
                  timestamp: Date.now()
                }));
              }
            } else if (message.type === 'disconnect') {
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
              
              client.isConnected = false;
              client.connectedTo = undefined;
              
              console.log(`🔌 ${client.deviceName} disconnected`);
              broadcastDeviceList();
            } else if (message.type === 'data') {
              // Forward data to connected device
              if (client.connectedTo) {
                const connectedClient = clients.get(client.connectedTo);
                if (connectedClient && connectedClient.ws.readyState === 1) {
                  const forwardMessage = {
                    type: 'data',
                    data: message.data,
                    fromDeviceId: clientId,
                    fromDeviceName: client.deviceName,
                    timestamp: Date.now()
                  };
                  
                  connectedClient.ws.send(JSON.stringify(forwardMessage));
                  console.log(`📤 Data forwarded from ${client.deviceName} to ${connectedClient.deviceName}`);
                }
              } else {
                console.log(`⚠️ ${client.deviceName} tried to send data but is not connected to any device`);
              }
            } else if (message.type === 'ping') {
              // Respond to ping
              ws.send(JSON.stringify({
                type: 'pong',
                timestamp: Date.now()
              }));
            } else {
              console.log(`⚠️ Unknown message type: ${message.type}`);
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
      
      console.log('✅ WebSocket server configured on /ws');
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