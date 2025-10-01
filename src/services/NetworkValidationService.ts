import {
  isLocalNetworkIp,
  validateLocalConnection,
  getDeviceIpAddress,
  getNetworkInfo,
  sanitizeIpAddress
} from '../utils/networkUtils';

export interface NetworkValidationResult {
  isValid: boolean;
  reason?: string;
  clientIp?: string;
  serverIp?: string;
  networkInfo?: {
    clientNetworkInfo: ReturnType<typeof import('../utils/networkUtils').getNetworkInfo>;
    serverNetworkInfo: ReturnType<typeof import('../utils/networkUtils').getNetworkInfo>;
  };
}

export class NetworkValidationService {
  private static instance: NetworkValidationService;
  private currentDeviceIp: string = '';
  private serverIp: string = '';
  private isInitialized: boolean = false;

  private constructor() {}

  static getInstance(): NetworkValidationService {
    if (!NetworkValidationService.instance) {
      NetworkValidationService.instance = new NetworkValidationService();
    }
    return NetworkValidationService.instance;
  }

  /**
   * Initialize the network validation service
   */
  async initialize(serverIp: string = ''): Promise<void> {
    // Get current device IP
    this.currentDeviceIp = await getDeviceIpAddress();
    
    // Set server IP (use provided or extract from current location)
    this.serverIp = serverIp || this.extractServerIpFromLocation();
    
    this.isInitialized = true;
    
    console.log('🌐 NetworkValidationService initialized:');
    console.log('  - Device IP:', this.currentDeviceIp);
    console.log('  - Server IP:', this.serverIp);
    
    // Check if using Cloudflare Tunnel
    const isCloudflareTunnel = this.serverIp.includes('.trycloudflare.com') || 
                               this.serverIp.includes('.cfargotunnel.com');
    
    if (!isCloudflareTunnel) {
      // Only log network info for non-Cloudflare connections
      const deviceNetworkInfo = getNetworkInfo(this.currentDeviceIp);
      const serverNetworkInfo = getNetworkInfo(this.serverIp);
      
      console.log('  - Device Network Info:', deviceNetworkInfo);
      console.log('  - Server Network Info:', serverNetworkInfo);
    } else {
      console.log('  - Using Cloudflare Tunnel - skipping network info validation');
    }
  }

  /**
   * Validate a WebSocket connection from a client
   */
  validateConnection(clientIp: string, requestOrigin?: string): NetworkValidationResult {
    if (!this.isInitialized) {
      return {
        isValid: false,
        reason: 'Network validation service not initialized'
      };
    }

    // Check if we're using Cloudflare Tunnel - bypass validation
    const isCloudflareTunnel = window.location.hostname.includes('.trycloudflare.com') || 
                               window.location.hostname.includes('.cfargotunnel.com');
    
    if (isCloudflareTunnel) {
      // Cloudflare Tunnel - allow all connections (Cloudflare handles security)
      console.log('☁️ Cloudflare Tunnel detected - bypassing network validation');
      return {
        isValid: true,
        clientIp: clientIp,
        serverIp: this.serverIp,
        networkInfo: {
          clientNetworkInfo: { ip: clientIp, isLocal: false, networkSegment: 'cloudflare', ranges: [] },
          serverNetworkInfo: { ip: this.serverIp, isLocal: false, networkSegment: 'cloudflare', ranges: [] }
        }
      };
    }

    // Sanitize the client IP
    const sanitizedClientIp = sanitizeIpAddress(clientIp);
    if (!sanitizedClientIp) {
      return {
        isValid: false,
        reason: 'Invalid client IP address format'
      };
    }

    // Check if client IP is local network
    if (!isLocalNetworkIp(sanitizedClientIp)) {
      return {
        isValid: false,
        reason: 'Client IP is not on a local network',
        clientIp: sanitizedClientIp,
        serverIp: this.serverIp
      };
    }

    // Check if server IP is local network (allow localhost)
    if (!isLocalNetworkIp(this.serverIp) && this.serverIp !== 'localhost') {
      return {
        isValid: false,
        reason: 'Server IP is not on a local network',
        clientIp: sanitizedClientIp,
        serverIp: this.serverIp
      };
    }

    // If server is localhost, allow any local network client
    if (this.serverIp === 'localhost' || this.serverIp === '127.0.0.1') {
      // Client just needs to be on a local network
      return {
        isValid: true,
        clientIp: sanitizedClientIp,
        serverIp: this.serverIp,
        networkInfo: {
          clientNetworkInfo: getNetworkInfo(sanitizedClientIp),
          serverNetworkInfo: getNetworkInfo(this.serverIp)
        }
      };
    }

    // Check if client and server are on the same network
    if (!validateLocalConnection(sanitizedClientIp, this.serverIp)) {
      return {
        isValid: false,
        reason: 'Client and server are not on the same local network',
        clientIp: sanitizedClientIp,
        serverIp: this.serverIp,
        networkInfo: {
          clientNetworkInfo: getNetworkInfo(sanitizedClientIp),
          serverNetworkInfo: getNetworkInfo(this.serverIp)
        }
      };
    }

    // Connection is valid
    return {
      isValid: true,
      clientIp: sanitizedClientIp,
      serverIp: this.serverIp,
      networkInfo: {
        clientNetworkInfo: getNetworkInfo(sanitizedClientIp),
        serverNetworkInfo: getNetworkInfo(this.serverIp)
      }
    };
  }

  /**
   * Validate WebSocket connection request
   */
  validateWebSocketConnection(clientIp: string, requestOrigin?: string): NetworkValidationResult {
    const result = this.validateConnection(clientIp);
    
    if (!result.isValid) {
      console.warn('🚫 WebSocket connection rejected:', result.reason);
      console.warn('  - Client IP:', clientIp);
      console.warn('  - Server IP:', this.serverIp);
      console.warn('  - Request Origin:', requestOrigin);
    } else {
      console.log('✅ WebSocket connection validated:', {
        clientIp,
        serverIp: this.serverIp,
        origin: requestOrigin
      });
    }
    
    return result;
  }

  /**
   * Validate ScreenShare connection request
   */
  validateScreenShareConnection(clientIp: string, deviceInfo?: {
    id: string;
    deviceType: 'desktop' | 'mobile' | 'unknown';
    deviceName: string;
    ipAddress: string;
    isConnected: boolean;
    connectedTo?: string;
  }): NetworkValidationResult {
    const result = this.validateConnection(clientIp);
    
    if (!result.isValid) {
      console.warn('🚫 ScreenShare connection rejected:', result.reason);
      console.warn('  - Client IP:', clientIp);
      console.warn('  - Server IP:', this.serverIp);
      console.warn('  - Device Info:', deviceInfo);
    } else {
      console.log('✅ ScreenShare connection validated:', {
        clientIp,
        serverIp: this.serverIp,
        deviceInfo
      });
    }
    
    return result;
  }

  /**
   * Check if current device can act as server (desktop host)
   */
  canActAsServer(): boolean {
    if (!this.isInitialized) {
      return false;
    }

    // Allow Cloudflare Tunnel connections
    const isCloudflareTunnel = this.serverIp.includes('.trycloudflare.com') || 
                               this.serverIp.includes('.cfargotunnel.com');
    
    if (isCloudflareTunnel) {
      return true; // Cloudflare Tunnel can act as server
    }

    // Server must be on local network
    return isLocalNetworkIp(this.serverIp);
  }

  /**
   * Check if current device can connect to a server
   */
  canConnectToServer(serverIp: string): NetworkValidationResult {
    if (!this.isInitialized) {
      return {
        isValid: false,
        reason: 'Network validation service not initialized'
      };
    }

    return this.validateConnection(serverIp);
  }

  /**
   * Get current network status
   */
  getNetworkStatus() {
    if (!this.isInitialized) {
      return {
        isInitialized: false,
        message: 'Network validation service not initialized'
      };
    }

    // Check if using Cloudflare Tunnel
    const isCloudflareTunnel = this.serverIp.includes('.trycloudflare.com') || 
                               this.serverIp.includes('.cfargotunnel.com');

    if (isCloudflareTunnel) {
      // Return simplified status for Cloudflare Tunnel
      return {
        isInitialized: true,
        deviceIp: this.currentDeviceIp,
        serverIp: this.serverIp,
        deviceNetworkInfo: { ip: this.currentDeviceIp, isLocal: true, networkSegment: 'local', ranges: [] },
        serverNetworkInfo: { ip: this.serverIp, isLocal: false, networkSegment: 'cloudflare', ranges: [] },
        canActAsServer: true,
        isLocalNetwork: false,
        isSameNetwork: true // Cloudflare Tunnel always allows connection
      };
    }

    const deviceNetworkInfo = getNetworkInfo(this.currentDeviceIp);
    const serverNetworkInfo = getNetworkInfo(this.serverIp);

    return {
      isInitialized: true,
      deviceIp: this.currentDeviceIp,
      serverIp: this.serverIp,
      deviceNetworkInfo,
      serverNetworkInfo,
      canActAsServer: this.canActAsServer(),
      isLocalNetwork: isLocalNetworkIp(this.currentDeviceIp) && isLocalNetworkIp(this.serverIp),
      isSameNetwork: validateLocalConnection(this.currentDeviceIp, this.serverIp)
    };
  }

  /**
   * Extract server IP from current location
   */
  private extractServerIpFromLocation(): string {
    const hostname = window.location.hostname;
    
    // For localhost, try to get the actual network IP from the Vite dev server
    // The Vite server shows the network IP in the console, but we can't access it directly
    // So we'll just use localhost and let the WebSocket handle it
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'localhost';
    }
    
    // Handle IP addresses directly - this is the most common case
    if (hostname.match(/^\d+\.\d+\.\d+\.\d+$/)) {
      return hostname;
    }
    
    // For domain names, return as-is
    return hostname;
  }

  /**
   * Set server IP manually (useful for testing)
   */
  setServerIp(serverIp: string): void {
    this.serverIp = serverIp;
    console.log('🌐 Server IP set to:', serverIp);
  }

  /**
   * Get current device IP
   */
  getDeviceIp(): string {
    return this.currentDeviceIp;
  }

  /**
   * Get server IP
   */
  getServerIp(): string {
    return this.serverIp;
  }

  /**
   * Reset the service
   */
  reset(): void {
    this.currentDeviceIp = '';
    this.serverIp = '';
    this.isInitialized = false;
    console.log('🌐 NetworkValidationService reset');
  }
}
