import * as tf from '@tensorflow/tfjs';
import type { CustomModel, PredictionResult } from '../components/ModelTraining';
import { NetworkValidationService } from './NetworkValidationService';

// WebSocket message types
export interface WebSocketMessage {
  type: 'model-ready' | 'predict' | 'prediction-result' | 'connection-status' | 'error';
  data?: any;
}

export interface ModelReadyMessage {
  classes: string[];
  isModelLoaded: boolean;
}

export interface PredictMessage {
  imageData: string; // base64 encoded image
  requestId: string;
}

export interface PredictionResultMessage {
  predictions: PredictionResult[];
  requestId: string;
}

export interface ConnectionStatusMessage {
  connectedClients: number;
  isDesktop: boolean;
}

export class WebSocketService {
  private static instance: WebSocketService;
  private static sharedModel: CustomModel | null = null; // Static model shared between instances
  private server: any = null; // Simulated server
  private clients: Set<any> = new Set(); // Simulated clients
  private model: CustomModel | null = null;
  private isDesktop: boolean = false;
  private clientConnection: any = null; // WebSocket client connection
  private onPredictionResult?: (predictions: PredictionResult[]) => void;
  private onConnectionStatus?: (status: ConnectionStatusMessage) => void;
  private onError?: (error: string) => void;
  private onModelStatus?: (status: { isModelLoaded: boolean; classes: string[] }) => void;
  private networkValidationService: NetworkValidationService;
  private isNetworkValidated: boolean = false;

  private constructor() {
    this.networkValidationService = NetworkValidationService.getInstance();
  }

  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  // Initialize as desktop (host) - browser compatible
  async initializeAsDesktop(): Promise<void> {
    this.isDesktop = true;
    console.log('🖥️ Initializing WebSocket service as desktop host...');
    
    try {
      // Initialize network validation
      await this.networkValidationService.initialize();
      
      // Check if this device can act as server
      if (!this.networkValidationService.canActAsServer()) {
        throw new Error('This device cannot act as server - not on local network');
      }
      
      this.isNetworkValidated = true;
      console.log('✅ Network validation passed for desktop host');
      
      // In browser environment, we simulate server behavior
      console.log('🚀 Desktop host ready for connections');
      
      // Simulate server object
      this.server = {
        on: (event: string, handler: Function) => {
          // Store handlers for simulation
          if (event === 'connection') {
            (this.server as any).connectionHandler = handler;
          }
        },
        close: () => {
          console.log('🔌 Desktop server closed');
          this.clients.clear();
        }
      };
      
      // Set initial connection status
      if (this.onConnectionStatus) {
        this.onConnectionStatus({
          connectedClients: 0,
          isDesktop: true
        });
      }
      
      console.log('✅ Desktop host initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize desktop host:', error);
      if (this.onError) {
        this.onError(`Network validation failed: ${error}`);
      }
      throw error;
    }
  }

  // Initialize as mobile (client) - browser compatible
  async initializeAsMobile(desktopIP: string = '192.168.254.100'): Promise<void> {
    this.isDesktop = false;
    console.log('📱 Initializing WebSocket service as mobile client...');
    
    try {
      // Initialize network validation
      await this.networkValidationService.initialize(desktopIP);
      
      // Validate connection to desktop
      const validation = this.networkValidationService.validateWebSocketConnection(desktopIP);
      
      if (!validation.isValid) {
        throw new Error(`Network validation failed: ${validation.reason}`);
      }
      
      this.isNetworkValidated = true;
      console.log('✅ Network validation passed for mobile client');
      
      // For demo purposes, simulate connection to desktop
      console.log('🔗 Simulating connection to desktop...');
      
      // Simulate successful connection after a delay
      setTimeout(() => {
        console.log('✅ Connected to desktop WebSocket server');
        
        if (this.onConnectionStatus) {
          this.onConnectionStatus({
            connectedClients: 1,
            isDesktop: false
          });
        }
        
        // Simulate receiving model status
        if (this.onModelStatus) {
          const modelStatus = {
            isModelLoaded: !!WebSocketService.sharedModel,
            classes: WebSocketService.sharedModel ? WebSocketService.sharedModel.getClassLabels() : []
          };
          this.onModelStatus(modelStatus);
          console.log('📱 Model status sent to mobile client:', modelStatus);
        }
      }, 1000);
      
    } catch (error) {
      console.error('❌ Failed to initialize mobile client:', error);
      if (this.onError) {
        this.onError(`Network validation failed: ${error}`);
      }
      throw error;
    }
  }

  // Set the shared model (desktop only)
  setSharedModel(model: CustomModel | null): void {
    if (!this.isDesktop) return;
    
    this.model = model;
    WebSocketService.sharedModel = model; // Set static model for mobile instances
    console.log('🔄 Shared model updated in WebSocket service');
    console.log('🔄 Static model updated for mobile instances');
    
    // Broadcast model status to all connected clients
    this.broadcastModelStatus();
  }

  // Send prediction request (mobile only)
  async sendPredictionRequest(imageData: string): Promise<PredictionResult[]> {
    if (this.isDesktop) {
      throw new Error('Cannot send prediction request from desktop');
    }
    
    return new Promise((resolve, reject) => {
      const requestId = Date.now().toString();
      
      console.log('📤 Sending prediction request to desktop...');
      
      // Simulate sending request to desktop and processing
      setTimeout(async () => {
        try {
          // If desktop has a model, process the image
          if (WebSocketService.sharedModel) {
            console.log('🖥️ Desktop processing image with model...');
            const predictions = await this.processImageForPrediction(imageData);
            console.log('✅ Desktop prediction complete');
            
            // Call the prediction result handler
            if (this.onPredictionResult) {
              this.onPredictionResult(predictions);
            }
            
            resolve(predictions);
          } else {
            // No model available, return error
            const errorResult: PredictionResult[] = [{
              className: 'Error: No model loaded on desktop',
              confidence: 0
            }];
            console.error('❌ No model available in static shared model');
            
            if (this.onPredictionResult) {
              this.onPredictionResult(errorResult);
            }
            
            resolve(errorResult);
          }
        } catch (error) {
          console.error('❌ Desktop prediction failed:', error);
          const errorResult: PredictionResult[] = [{
            className: 'Error: Prediction failed',
            confidence: 0
          }];
          
          if (this.onPredictionResult) {
            this.onPredictionResult(errorResult);
          }
          
          resolve(errorResult);
        }
      }, 1000); // Simulate network delay
    });
  }

  // Handle incoming messages
  private async handleMessage(client: any, message: WebSocketMessage): Promise<void> {
    switch (message.type) {
      case 'predict':
        await this.handlePredictRequest(client, message.data as PredictMessage);
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  // Handle prediction request
  private async handlePredictRequest(client: any, predictMessage: PredictMessage): Promise<void> {
    if (!this.model) {
      this.sendToClient(client, {
        type: 'error',
        data: { message: 'No model available for prediction' }
      });
      return;
    }

    try {
      // Process the image and make predictions
      const predictions = await this.processImageForPrediction(predictMessage.imageData);
      
      this.sendToClient(client, {
        type: 'prediction-result',
        data: {
          predictions,
          requestId: predictMessage.requestId
        }
      });
    } catch (error) {
      console.error('Error processing prediction request:', error);
      this.sendToClient(client, {
        type: 'error',
        data: { message: 'Failed to process prediction request' }
      });
    }
  }

  // Process image for prediction
  private async processImageForPrediction(imageData: string): Promise<PredictionResult[]> {
    // Create image from base64 data
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = async () => {
        try {
          // Create canvas for processing
          const canvas = document.createElement('canvas');
          canvas.width = 224;
          canvas.height = 224;
          const ctx = canvas.getContext('2d');
          
          if (ctx) {
            ctx.drawImage(img, 0, 0, 224, 224);
            
            // Make predictions with the model
            if (WebSocketService.sharedModel) {
              const predictions = await WebSocketService.sharedModel.predict(canvas);
              resolve(predictions);
            } else {
              reject(new Error('No model available'));
            }
          } else {
            reject(new Error('Failed to create canvas context'));
          }
        } catch (error) {
          reject(error);
        }
      };
      
      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };
      
      img.src = imageData;
    });
  }

  // Send message to client
  private sendToClient(client: any, message: WebSocketMessage): void {
    // Simulate sending message to client
    console.log('📤 Sending message to client:', message.type);
    
    if (message.type === 'prediction-result') {
      // Simulate receiving prediction result
      if (this.onPredictionResult) {
        this.onPredictionResult(message.data.predictions);
      }
    }
  }

  // Send message to server
  private sendToServer(message: WebSocketMessage): void {
    if (this.clientConnection && this.clientConnection.readyState === this.clientConnection.OPEN) {
      this.clientConnection.send(JSON.stringify(message));
    } else {
      console.error('❌ No WebSocket connection available or connection not open');
    }
  }

  // Broadcast model status
  private broadcastModelStatus(): void {
    const status = {
      classes: WebSocketService.sharedModel ? WebSocketService.sharedModel.getClassLabels() : [],
      isModelLoaded: !!WebSocketService.sharedModel
    };
    
    // Call model status handler
    if (this.onModelStatus) {
      this.onModelStatus(status);
    }
    
    const message: WebSocketMessage = {
      type: 'model-ready',
      data: status
    };
    
    // Broadcast to all connected clients
    this.clients.forEach(client => {
      this.sendToClient(client, message);
    });
  }

  // Broadcast connection status
  private broadcastConnectionStatus(): void {
    if (this.onConnectionStatus) {
      this.onConnectionStatus({
        connectedClients: this.clients.size,
        isDesktop: true
      });
    }
  }

  // Event handlers
  onPredictionResultHandler(handler: (predictions: PredictionResult[]) => void): void {
    this.onPredictionResult = handler;
  }

  onConnectionStatusHandler(handler: (status: ConnectionStatusMessage) => void): void {
    this.onConnectionStatus = handler;
  }

  onErrorHandler(handler: (error: string) => void): void {
    this.onError = handler;
  }

  onModelStatusHandler(handler: (status: { isModelLoaded: boolean; classes: string[] }) => void): void {
    this.onModelStatus = handler;
  }

  // Shutdown
  shutdown(): void {
    if (this.server) {
      this.server.close();
      this.server = null;
    }
    
    if (this.clientConnection) {
      this.clientConnection.close();
      this.clientConnection = null;
    }
    
    this.clients.clear();
    console.log('🔌 WebSocket service shutdown complete');
  }
}
