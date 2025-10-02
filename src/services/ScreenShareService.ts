import type { PredictionResult } from '../ai/components/AiModeltraining';
import { NetworkValidationService } from './NetworkValidationService';

// WebSocket message types
type WebSocketMessageType = 'preview-update' | 'prediction-result' | 'predicting-status' | 'model-status' | 'mode-change' | 'device-list' | 'connection-request' | 'connection-response' | 'disconnected' | 'disconnect-request' | 'device-info';

interface WebSocketMessage {
  type: WebSocketMessageType;
  data: unknown;
  timestamp: number;
  source: 'desktop' | 'mobile';
}

// Screen share data types
export interface PreviewImageData {
  imageData: string;
  source: 'desktop' | 'mobile';
}

export interface PredictionResultsData {
  results: PredictionResult[];
  source: 'desktop' | 'mobile';
}

export interface IsPredictingData {
  isPredicting: boolean;
  source: 'desktop' | 'mobile';
}

export interface ModelStatusData {
  isModelLoaded: boolean;
  classes: string[];
}

export interface ModeChangeData {
  mode: 'webcam' | 'upload';
  source: 'desktop' | 'mobile';
}

// Device discovery interfaces
export interface DeviceInfo {
  id: string;
  deviceType: 'desktop' | 'mobile' | 'unknown';
  deviceName: string;
  ipAddress: string;
  isConnected: boolean;
  connectedTo?: string;
}

export interface DeviceListData {
  devices: DeviceInfo[];
}

export interface ConnectionRequestData {
  fromDeviceId: string;
  fromDeviceName: string;
  fromDeviceType: 'desktop' | 'mobile' | 'unknown';
  requestType: 'connect';
}

export interface ConnectionResponseData {
  fromDeviceId: string;
  fromDeviceName: string;
  accepted: boolean;
}

export interface DisconnectedData {
  fromDeviceId: string;
  fromDeviceName: string;
}

export interface DeviceInfoData {
  id: string;
  deviceType: 'desktop' | 'mobile' | 'unknown';
  deviceName: string;
  ipAddress: string;
  isConnected?: boolean;
}

export class ScreenShareService {
  private static instance: ScreenShareService;
  private isDesktop: boolean = false;
  private ws: WebSocket | null = null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private isConnecting: boolean = false;
  private hasTriedFallback: boolean = false;
  
  private onPreviewUpdate?: (data: PreviewImageData) => void;
  private onPredictionUpdate?: (data: PredictionResultsData) => void;
  private onPredictingUpdate?: (data: IsPredictingData) => void;
  private onModelStatusUpdate?: (data: ModelStatusData) => void;
  private onConnectionStatus?: (connected: boolean) => void;
  private onModeChange?: (data: ModeChangeData) => void;
  private onDeviceList?: (data: DeviceListData) => void;
  private onConnectionRequest?: (data: ConnectionRequestData) => void;
  private onConnectionResponse?: (data: ConnectionResponseData) => void;
  private onDisconnected?: (data: DisconnectedData) => void;
  
  // Device management
  private currentDevice: DeviceInfo | null = null;
  private availableDevices: DeviceInfo[] = [];
  private connectedDevice: DeviceInfo | null = null;
  private networkValidationService: NetworkValidationService;
  private isNetworkValidated: boolean = false;

  private lastSentTime: number = 0;
  private throttleInterval: number = 300; // Send every 300ms max

  private constructor() {
    this.networkValidationService = NetworkValidationService.getInstance();
  }

  static getInstance(): ScreenShareService {
    if (!ScreenShareService.instance) {
      ScreenShareService.instance = new ScreenShareService();
    }
    return ScreenShareService.instance;
  }

  // Initialize the service
  async initialize(isDesktop: boolean): Promise<void> {
    this.isDesktop = isDesktop;
    
    try {
      // Initialize network validation
      await this.networkValidationService.initialize();
      
      // Check network validation
      if (isDesktop) {
        if (!this.networkValidationService.canActAsServer()) {
          throw new Error('This device cannot act as server - not on local network');
        }
      } else {
        // For mobile, we'll validate during connection
        console.log('📱 Mobile device - network validation will occur during connection');
      }
      
      this.isNetworkValidated = true;
      console.log('✅ Network validation passed for', isDesktop ? 'desktop' : 'mobile');
      
      // Proceed with connection
      await this.connect();
    } catch (error) {
      console.error('❌ Network validation failed:', error);
      throw new Error(`Network validation failed: ${error}`);
    }
  }

  // Connect to WebSocket server
  async connect(): Promise<void> {
    if (this.isConnecting || this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.isConnecting = true;
    this.hasTriedFallback = false; // Reset fallback flag for new connection attempt
    
    // Detect device type for connection strategy
    const isMobileDevice = /Mobile|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Additional check for mobile devices based on screen size and touch support
    const isMobileByScreen = window.innerWidth <= 768 && 'ontouchstart' in window;
    const finalIsMobile = isMobileDevice || isMobileByScreen;
    
    console.log('📱 Device detection:');
    console.log('  - User agent mobile:', isMobileDevice);
    console.log('  - Screen size mobile:', isMobileByScreen);
    console.log('  - Final mobile detection:', finalIsMobile);
    console.log('  - User agent:', navigator.userAgent);
    console.log('  - Screen width:', window.innerWidth);
    
    return new Promise<void>((resolve, reject) => {
      try {
        let hostname, wsUrl;
        
        let useSecure: boolean;
        
        const isCloudflareTunnel = window.location.hostname.includes('.trycloudflare.com') || 
                           window.location.hostname.includes('.cfargotunnel.com');
        
        if (isCloudflareTunnel) {
          // Using Cloudflare Tunnel - try multiple approaches
          const protocol = 'wss';
          const hostname = window.location.hostname;
          
          // Try the same hostname first (single tunnel approach)
          wsUrl = `${protocol}://${hostname}/ws`;
          useSecure = true;
          console.log('☁️ Cloudflare Tunnel detected, trying WebSocket tunnel:', wsUrl);
        } else {
          // Local network connection
          // Use the actual hostname from the browser (works for both localhost and network IPs)
          hostname = window.location.hostname;
          
          wsUrl = ''; // Will be set below
          if (window.location.hostname === 'localhost') {
            console.log('🔄 Replacing localhost with network IP for cross-device connectivity:', hostname);
          }
          
          // For mobile devices, try to use the IP address directly
          if (finalIsMobile && hostname !== 'localhost') {
            console.log('📱 Mobile device detected, using hostname:', hostname);
          } else if (finalIsMobile) {
            console.log('📱 Mobile device on localhost - this might not work for cross-device connection');
            console.log('💡 Try accessing the app via the IP address instead of localhost');
          }
          
          // For mobile devices, prioritize unsecure connection to avoid certificate issues
          // For desktop, try secure first then fallback to unsecure
          useSecure = !finalIsMobile && window.location.protocol === 'https:';
          const protocol = useSecure ? 'wss' : 'ws';
          const port = 5173; // Use unified port for all connections
          wsUrl = `${protocol}://${hostname}:${port}/ws`;
        }
        
        console.log('📱 Device type:', finalIsMobile ? 'Mobile' : 'Desktop');
        console.log('🔗 Connecting to WebSocket:', wsUrl);
        console.log('🔗 Using secure connection:', useSecure);
        
        // Create WebSocket connection
        this.ws = new WebSocket(wsUrl);
        
        // Validate network before establishing connection
        if (this.isNetworkValidated) {
          const networkStatus = this.networkValidationService.getNetworkStatus();
          console.log('🌐 Network status before connection:', networkStatus);
          
          // Check if using Cloudflare Tunnel
          const isCloudflareTunnel = window.location.hostname.includes('.trycloudflare.com') || 
                                     window.location.hostname.includes('.cfargotunnel.com');
          
          // Skip validation for Cloudflare Tunnel (it handles its own security)
          if (!isCloudflareTunnel && (!networkStatus.isLocalNetwork || !networkStatus.isSameNetwork)) {
            throw new Error('Devices are not on the same local network');
          }
        }
        
        // Set up event handlers
        this.ws.onopen = () => {
          console.log('✅ WebSocket connected successfully');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.onConnectionStatus?.(true);
          
          // Send device info after connection
          this.sendDeviceInfo();
          
          resolve();
        };
        
        this.ws.onmessage = (event) => this.handleMessage(event);
        this.ws.onclose = () => this.handleClose();
        
        this.ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          console.error('❌ WebSocket readyState:', this.ws?.readyState);
          console.error('❌ WebSocket URL:', this.ws?.url);
          
          // If secure connection failed and we haven't tried unsecure yet, try fallback
          // Check if it's likely a certificate error (common with self-signed certs)
          if (error.type === 'error') {
            if (finalIsMobile) {
              console.log('💡 Mobile device connection failed - trying unsecure connection automatically');
              console.log('💡 If connection still fails, make sure both devices are on the same WiFi network');
            } else {
              console.log('💡 Certificate error detected - please accept the self-signed certificate');
              console.log(`🔗 Visit https://${hostname}:5173/cert-accept.html to accept the WebSocket server certificate`);
              console.log(`💡 Also try visiting https://${hostname}:5173/ to test if the server is accessible`);
            }
          } else {
            console.log('💡 Connection failed - checking if WebSocket server is running...');
            console.log(`🔗 Try visiting https://${hostname}:5173/ to check if the server is accessible`);
          }
          
          this.isConnecting = false;
          this.onConnectionStatus?.(false);
          
          // Don't reject immediately, allow reconnection attempts
          setTimeout(() => {
            this.attemptReconnect();
          }, 2000);
        };
        
      } catch (error) {
        console.error('❌ WebSocket connection error:', error);
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  // Handle incoming messages
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.handleMessageContent(message);
    } catch (error) {
      console.error('❌ Error parsing WebSocket message:', error);
    }
  }

  // Handle message content
  private handleMessageContent(message: WebSocketMessage): void {
    // Don't process messages from the same device
    if (message.source === (this.isDesktop ? 'desktop' : 'mobile')) {
      return;
    }

    console.log(`📨 Received ${message.type} from ${message.source}`);

    switch (message.type) {
      case 'preview-update':
        this.onPreviewUpdate?.(message.data as PreviewImageData);
        break;
      case 'prediction-result':
        this.onPredictionUpdate?.(message.data as PredictionResultsData);
        break;
      case 'predicting-status':
        this.onPredictingUpdate?.(message.data as IsPredictingData);
        break;
      case 'model-status':
        this.onModelStatusUpdate?.(message.data as ModelStatusData);
        break;
      case 'mode-change':
        this.onModeChange?.(message.data as ModeChangeData);
        break;
      case 'device-list':
        this.onDeviceList?.(message.data as DeviceListData);
        break;
      case 'connection-request':
        this.onConnectionRequest?.(message.data as ConnectionRequestData);
        break;
      case 'connection-response':
        this.onConnectionResponse?.(message.data as ConnectionResponseData);
        break;
      case 'disconnected':
        this.onDisconnected?.(message.data as DisconnectedData);
        break;
      case 'device-info':
        this.handleDeviceInfo(message.data as DeviceInfoData);
        break;
    }
  }

  // Handle connection close
  private handleClose(): void {
    console.log('🔌 WebSocket disconnected');
    this.isConnecting = false;
    this.onConnectionStatus?.(false);
    
    // Attempt to reconnect
    this.attemptReconnect();
  }

  // Attempt to reconnect
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`);
    
    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('❌ Reconnection failed:', error);
      });
    }, this.reconnectDelay * this.reconnectAttempts);
  }

  // Try unsecure WebSocket connection as fallback
  private async tryUnsecureConnection(): Promise<void> {
    if (this.isConnecting || this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    // Check if we're using Cloudflare Tunnel - no unsecure fallback needed
    const isCloudflare = window.location.hostname.includes('.trycloudflare.com') || 
                       window.location.hostname.includes('.cfargotunnel.com');
    
    if (isCloudflare) {
      console.log('☁️ Cloudflare Tunnel detected - no unsecure fallback needed');
      this.isConnecting = false;
      return;
    }

    this.isConnecting = true;
    
    return new Promise<void>((resolve, reject) => {
      try {
        // Use the actual hostname from the browser
        const hostname = window.location.hostname;
        const wsUrl = `ws://${hostname}:5173/ws`;
        console.log('🔗 Connecting to unsecure WebSocket:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          console.log('✅ Unsecure WebSocket connected successfully');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.onConnectionStatus?.(true);
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessageContent(message);
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        };
        
        this.ws.onerror = (error) => {
          console.error('❌ Unsecure WebSocket error:', error);
          this.isConnecting = false;
          this.onConnectionStatus?.(false);
          reject(error);
        };
        
        this.ws.onclose = () => {
          console.log('🔌 Unsecure WebSocket disconnected');
          this.isConnecting = false;
          this.onConnectionStatus?.(false);
          this.attemptReconnect();
        };
        
      } catch (error) {
        console.error('❌ Error creating unsecure WebSocket:', error);
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  // Send message to WebSocket server
  private send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('❌ Error sending message:', error);
      }
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send message');
    }
  }

  // Share preview image (works from both desktop and mobile)
  sharePreviewImage(imageData: string): void {
    const now = Date.now();
    
    // Throttle: only send if 300ms have passed since last send
    if (now - this.lastSentTime < this.throttleInterval) {
      console.log('⏸️ Throttled - skipping frame');
      return;
    }
    
    this.lastSentTime = now;
    console.log('📤 Sending frame to', this.isDesktop ? 'mobile' : 'desktop');
    
    const message: WebSocketMessage = {
      type: 'preview-update',
      data: {
        imageData,
        source: this.isDesktop ? 'desktop' : 'mobile'
      },
      timestamp: now,
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    this.send(message);
  }

  // Share prediction results (desktop only, but mobile can receive)
  sharePredictionResults(results: PredictionResult[]): void {
    if (!this.isDesktop) {
      console.warn('⚠️ Only desktop can share prediction results');
      return;
    }

    const message: WebSocketMessage = {
      type: 'prediction-result',
      data: {
        results,
        source: 'desktop'
      },
      timestamp: Date.now(),
      source: 'desktop'
    };
    
    this.send(message);
  }

  // Share predicting status (works from both desktop and mobile)
  sharePredictingStatus(isPredicting: boolean): void {
    const message: WebSocketMessage = {
      type: 'predicting-status',
      data: {
        isPredicting,
        source: this.isDesktop ? 'desktop' : 'mobile'
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    this.send(message);
  }

  // Share model status (desktop only)
  shareModelStatus(isModelLoaded: boolean, classes: string[]): void {
    if (!this.isDesktop) {
      console.warn('⚠️ Only desktop can share model status');
      return;
    }

    const message: WebSocketMessage = {
      type: 'model-status',
      data: {
        isModelLoaded,
        classes
      },
      timestamp: Date.now(),
      source: 'desktop'
    };
    
    this.send(message);
  }

  // Share mode change (works from both desktop and mobile)
  shareModeChange(mode: 'webcam' | 'upload'): void {
    const message: WebSocketMessage = {
      type: 'mode-change',
      data: {
        mode,
        source: this.isDesktop ? 'desktop' : 'mobile'
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    this.send(message);
  }

  // Event handlers
  onPreviewUpdateHandler(handler: (data: PreviewImageData) => void): void {
    this.onPreviewUpdate = handler;
  }

  onPredictionUpdateHandler(handler: (data: PredictionResultsData) => void): void {
    this.onPredictionUpdate = handler;
  }

  onPredictingUpdateHandler(handler: (data: IsPredictingData) => void): void {
    this.onPredictingUpdate = handler;
  }

  onModelStatusUpdateHandler(handler: (data: ModelStatusData) => void): void {
    this.onModelStatusUpdate = handler;
  }

  onConnectionStatusHandler(handler: (connected: boolean) => void): void {
    this.onConnectionStatus = handler;
  }

  onModeChangeHandler(handler: (data: ModeChangeData) => void): void {
    this.onModeChange = handler;
  }
  
  onDeviceListHandler(handler: (data: DeviceListData) => void): void {
    this.onDeviceList = handler;
  }
  
  onConnectionRequestHandler(handler: (data: ConnectionRequestData) => void): void {
    this.onConnectionRequest = handler;
  }
  
  onConnectionResponseHandler(handler: (data: ConnectionResponseData) => void): void {
    this.onConnectionResponse = handler;
  }
  
  onDisconnectedHandler(handler: (data: DisconnectedData) => void): void {
    this.onDisconnected = handler;
  }
  
  // Device management methods
  getCurrentDevice(): DeviceInfo | null {
    return this.currentDevice;
  }
  
  getAvailableDevices(): DeviceInfo[] {
    return this.availableDevices;
  }
  
  getConnectedDevice(): DeviceInfo | null {
    return this.connectedDevice;
  }
  
  isConnectedToDevice(): boolean {
    return this.connectedDevice !== null;
  }
  
  // Send connection request to another device
  sendConnectionRequest(targetDeviceId: string): void {
    if (!this.currentDevice) {
      console.warn('⚠️ No current device info available');
      return;
    }

    const message: WebSocketMessage = {
      type: 'connection-request',
      data: {
        targetDeviceId,
        fromDeviceId: this.currentDevice.id,
        fromDeviceName: this.currentDevice.deviceName,
        fromDeviceType: this.currentDevice.deviceType,
        requestType: 'connect'
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };

    console.log('🔗 Sending connection request:', message);
    this.send(message);
  }
  
  // Send connection response (accept/reject)
  sendConnectionResponse(targetDeviceId: string, accepted: boolean): void {
    if (!this.currentDevice) {
      console.warn('⚠️ No current device info available');
      return;
    }
    
    const message: WebSocketMessage = {
      type: 'connection-response',
      data: {
        targetDeviceId,
        fromDeviceId: this.currentDevice.id,
        fromDeviceName: this.currentDevice.deviceName,
        accepted
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    console.log('✅ Sending connection response:', message);
    this.send(message);
    
    // If accepted, update connection status
    if (accepted) {
      const targetDevice = this.availableDevices.find(device => device.id === targetDeviceId);
      if (targetDevice) {
        this.connectedDevice = targetDevice;
      }
    }
  }
  
  // Send disconnection request
  sendDisconnectRequest(targetDeviceId: string): void {
    if (!this.currentDevice) {
      console.warn('⚠️ No current device info available');
      return;
    }
    
    const message: WebSocketMessage = {
      type: 'disconnect-request',
      data: {
        targetDeviceId: targetDeviceId,
        fromDeviceId: this.currentDevice.id,
        fromDeviceName: this.currentDevice.deviceName
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    console.log('🔌 Sending disconnect request:', message);
    this.send(message);
    
    // Clear connection status
    this.connectedDevice = null;
  }

  // Check if connected
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // Get actual device name
  private getDeviceName(): string {
    try {
      // Try to get hostname from the browser
      if (typeof window !== 'undefined' && window.location) {
        const hostname = window.location.hostname;
        
        // If it's an IP address, use a more descriptive name
        if (/^\d+\.\d+\.\d+\.\d+$/.test(hostname)) {
          return this.isDesktop ? `Desktop (${hostname})` : `Mobile (${hostname})`;
        }
        
        // If it's localhost, use computer name if available
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
          // Try to get computer name from navigator platform
          const platform = navigator.platform || '';
          const userAgent = navigator.userAgent || '';
          
          // Extract computer name from user agent if available
          const computerNameMatch = userAgent.match(/Windows NT .*?; (.*?);/);
          if (computerNameMatch && computerNameMatch[1]) {
            return `Desktop (${computerNameMatch[1].trim()})`;
          }
          
          // Fallback to platform info
          if (platform) {
            return this.isDesktop ? `Desktop (${platform})` : `Mobile (${platform})`;
          }
        }
        
        // Use hostname for other cases
        return this.isDesktop ? `Desktop (${hostname})` : `Mobile (${hostname})`;
      }
    } catch (error) {
      console.warn('⚠️ Could not get device name, using fallback:', error);
    }
    
    // Ultimate fallback
    return this.isDesktop ? 'Desktop Device' : 'Mobile Device';
  }

  // Save device info to localStorage
  private saveDeviceInfo(): void {
    if (this.currentDevice && typeof window !== 'undefined') {
      try {
        const deviceKey = this.isDesktop ? 'desktop-device-info' : 'mobile-device-info';
        localStorage.setItem(deviceKey, JSON.stringify({
          id: this.currentDevice.id,
          deviceName: this.currentDevice.deviceName,
          deviceType: this.currentDevice.deviceType
        }));
        console.log('💾 Device info saved to localStorage:', this.currentDevice.deviceName);
      } catch (error) {
        console.warn('⚠️ Could not save device info to localStorage:', error);
      }
    }
  }

  // Load device info from localStorage
  private loadDeviceInfo(): DeviceInfo | null {
    if (typeof window !== 'undefined') {
      try {
        const deviceKey = this.isDesktop ? 'desktop-device-info' : 'mobile-device-info';
        const savedInfo = localStorage.getItem(deviceKey);
        if (savedInfo) {
          const parsed = JSON.parse(savedInfo);
          console.log('📂 Device info loaded from localStorage:', parsed.deviceName);
          return {
            ...parsed,
            ipAddress: this.networkValidationService.getDeviceIp(),
            isConnected: true
          };
        }
      } catch (error) {
        console.warn('⚠️ Could not load device info from localStorage:', error);
      }
    }
    return null;
  }

  // Send device info to server
  private sendDeviceInfo(): void {
    if (!this.currentDevice) {
      // Try to load device info from localStorage first
      this.currentDevice = this.loadDeviceInfo();
      
      if (!this.currentDevice) {
        // Create device info if not exists in localStorage
        this.currentDevice = {
          id: `device-${Date.now()}`,
          deviceType: this.isDesktop ? 'desktop' : 'mobile',
          deviceName: this.getDeviceName(),
          ipAddress: this.networkValidationService.getDeviceIp(),
          isConnected: true
        };
        // Save the newly created device info
        this.saveDeviceInfo();
      } else {
        // Update loaded device info with current IP
        this.currentDevice.ipAddress = this.networkValidationService.getDeviceIp();
        this.currentDevice.isConnected = true;
      }
    } else {
      // Update existing device info - preserve server-provided device name
      if (!this.currentDevice.deviceName || this.currentDevice.deviceName.startsWith('Desktop Device') || this.currentDevice.deviceName.startsWith('Mobile Device')) {
        // Only update device name if it's generic or not set
        this.currentDevice.deviceName = this.getDeviceName();
      }
      this.currentDevice.ipAddress = this.networkValidationService.getDeviceIp();
      // Save updated device info
      this.saveDeviceInfo();
    }
    
    const message: WebSocketMessage = {
      type: 'device-list',
      data: {
        devices: [this.currentDevice]
      },
      timestamp: Date.now(),
      source: this.isDesktop ? 'desktop' : 'mobile'
    };
    
    this.send(message);
    console.log('📱 Device info sent:', this.currentDevice);
  }


  // Shutdown and cleanup
  shutdown(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.currentDevice = null;
    this.availableDevices = [];
    this.connectedDevice = null;
    this.onConnectionStatus?.(false);
  }
  
  // Handle device info message
  private handleDeviceInfo(data: DeviceInfoData): void {
    console.log('📱 Device info received from server:', data);
    
    // Update current device with server-provided info
    this.currentDevice = {
      id: data.id,
      deviceType: data.deviceType,
      deviceName: data.deviceName,
      ipAddress: data.ipAddress,
      isConnected: data.isConnected ?? false
    };
    
    // Save the server-provided device info to localStorage
    this.saveDeviceInfo();
    
    console.log('✅ Device info updated from server:', this.currentDevice.deviceName);
  }
  
  // Handle device list message
  private handleDeviceList(data: DeviceListData): void {
    // Filter out current device from available devices
    this.availableDevices = data.devices.filter(device => device.id !== this.currentDevice?.id);
    console.log('📱 Available devices:', this.availableDevices);
  }
  
  // Handle connection request message
  private handleConnectionRequest(data: ConnectionRequestData): void {
    console.log('📨 Connection request received:', data);
  }
  
  // Handle connection response message
  private handleConnectionResponse(data: ConnectionResponseData): void {
    console.log('📨 Connection response received:', data);
    
    if (data.accepted) {
      // Find the connected device from available devices
      const connectedDevice = this.availableDevices.find(device => device.id === data.fromDeviceId);
      if (connectedDevice) {
        this.connectedDevice = connectedDevice;
        console.log('✅ Connected to device:', connectedDevice.deviceName);
      }
    } else {
      console.log('❌ Connection request rejected');
      this.connectedDevice = null;
    }
  }
  
  // Handle disconnected message
  private handleDisconnected(data: DisconnectedData): void {
    console.log('🔌 Disconnected from device:', data.fromDeviceName);
    this.connectedDevice = null;
    
    // Notify UI about disconnection
    this.onConnectionStatus?.(false);
    this.onDisconnected?.(data);
  }
}
