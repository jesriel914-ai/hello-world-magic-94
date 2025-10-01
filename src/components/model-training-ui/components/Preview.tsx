import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as tmImage from '@teachablemachine/image';
import * as tf from '@tensorflow/tfjs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { FileImage, Brain, Menu, X, Loader2, Upload, Camera, FolderOpen, Check, Cloud, Download, ChevronDown, List } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { TrainedModel } from '../../ModelTraining';
import type { PredictionResult } from '../../ModelTraining';
import type { CustomModel } from '../../ModelTraining';
import { MobileWebcam } from '../services/mobileWebcam';
import { ConnectionDropdown } from './ConnectionDropdown';
import { ScreenShareService } from '../../../services/ScreenShareService';
import { toast } from '@/hooks/use-toast';

// Helper functions
const getInitialVisibleCount = (preds: PredictionResult[]): number => {
  if (preds.length === 0) return 0;
  if (preds.length === 1) return 1;
  
  const firstConfidence = preds[0].confidence * 100;
  
  // If first prediction is 100%, show only 1
  if (firstConfidence >= 99.9) {
    return 1;
  }
  
  if (preds.length === 2) return 2;
  
  const firstTwoTotal = (preds[0].confidence + preds[1].confidence) * 100;
  const thirdConfidence = preds[2].confidence * 100;
  
  // If first two total 100% or third has negligible confidence (< 1%), show only 2
  if (firstTwoTotal >= 99.9 || thirdConfidence < 1) {
    return 2;
  }
  // Otherwise show 3
  return 3;
};

// Format date in AM PM format
const formatDateTime = (date: Date): string => {
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });
};

interface PreviewProps {
  isMobile: boolean;
  showModels: boolean;
  isLoadingModels: boolean;
  trainedModels: TrainedModel[];
  predictions: PredictionResult[];
  visiblePredictions: number;
  previewImage: string | null;
  isWebcamActive: boolean;
  webcam: tmImage.Webcam | null;
  model: CustomModel | null;
  selectedModel: any;
  models: any[];
  onModelSelect: (model: any) => void;
  onAddModel: () => void;
  onDeleteModel: (model: any) => void;
  onEditModel: (model: any) => void;
  onToggleModels: () => void;
  onShowMorePredictions: () => void;
  onChangeModel: () => void; // Legacy function for backward compatibility
  onCloudModelSelect?: () => void; // Optional: for cloud model selection
  onLocalModelSelect?: () => void; // Optional: for local model selection
  onHandlePreviewFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onGlobalModelSelect: (model: any) => void;
  globalModels: any[];
  onAddGlobalModel: () => void;
  onDeleteGlobalModel: (model: any) => void;
  onEditGlobalModel: (model: any) => void;
  onRefreshGlobalModels: () => void;
  isGlobalModelsLoading: boolean;
  globalModelsError: string | null;
}

export const Preview: React.FC<PreviewProps> = ({
  onToggleModels,
  showModels,
  isMobile = false,
  selectedModel,
  onModelSelect,
  models,
  onAddModel,
  onDeleteModel,
  onEditModel,
  onGlobalModelSelect,
  globalModels,
  onAddGlobalModel,
  onDeleteGlobalModel,
  onEditGlobalModel,
  onRefreshGlobalModels,
  isGlobalModelsLoading,
  globalModelsError,
  model,
  predictions,
  visiblePredictions,
  previewImage,
  isWebcamActive,
  webcam,
  trainedModels,
  isLoadingModels,
  onShowMorePredictions,
  onChangeModel,
  onCloudModelSelect,
  onLocalModelSelect,
  onHandlePreviewFileUpload
}) => {
  const screenShareService = ScreenShareService.getInstance();
  const [activeMode, setActiveMode] = useState<'webcam' | 'upload'>('upload');
  
  const [modelTrainedAt, setModelTrainedAt] = useState<Date | null>(null);
  
  // Real-time sync state
  const [isMobileClient, setIsMobileClient] = useState(false);
  const [isConnected, setIsConnected] = useState(false); // Connection status from new dropdown system
  const [isUsingRemoteModel, setIsUsingRemoteModel] = useState(false);
  const [remotePredictions, setRemotePredictions] = useState<PredictionResult[]>([]);
  const [mobilePreviewImage, setMobilePreviewImage] = useState<string | null>(null);
  const [desktopCameraFeed, setDesktopCameraFeed] = useState<string | null>(null);
  
  // Network validation state
  const [networkError, setNetworkError] = useState<string | null>(null);
  
  // Use remote predictions if available, otherwise use local predictions
  const displayPredictions = isUsingRemoteModel ? remotePredictions : predictions;
  
  // Camera state
  const webcamRef = useRef<HTMLDivElement>(null);
  const mobileWebcam = useRef<MobileWebcam | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  
  // Screen sharing state
  const screenShareIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const screenShareCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // Initialize mobile client if on mobile
  useEffect(() => {
    const initMobileClient = async () => {
      if (isMobile) {
        try {
          console.log('📱 Initializing mobile client for real-time sync...');
          setIsMobileClient(true);
          
          // Initialize ScreenShareService as mobile client
          await screenShareService.initialize(false);
          
          
          screenShareService.onPredictionUpdateHandler((predictionsData) => {
            setIsUsingRemoteModel(true);
            setRemotePredictions(predictionsData.results);
          });
          
          screenShareService.onPreviewUpdateHandler((previewData) => {
            console.log('📱 Preview image received from desktop');
            setMobilePreviewImage(previewData.imageData);
          });
          
          screenShareService.onModeChangeHandler((modeData) => {
            console.log('📱 Mode change received from desktop:', modeData.mode);
            setActiveMode(modeData.mode);
            
            // If switching to upload mode, stop any active camera
            if (modeData.mode === 'upload' && mobileWebcam.current) {
              stopCamera();
            }
          });
          
          screenShareService.onModelStatusUpdateHandler((status) => {
            console.log('📱 Model status received:', status);
            if (status.isModelLoaded) {
              console.log('✅ Model is loaded on desktop with classes:', status.classes);
            } else {
              console.log('⚠️ No model loaded on desktop');
            }
          });
          
          // Note: ScreenShareService doesn't have a separate error handler, errors are logged internally
          
          console.log('✅ Mobile client initialized');
        } catch (error) {
          console.error('❌ Failed to initialize mobile client:', error);
          
          // Check if it's a network validation error
          if (error instanceof Error && error.message.includes('network')) {
            setNetworkError('Network validation failed: Devices must be on the same local network');
          }
        }
      }
    };
    
    initMobileClient();
    
    // Cleanup on unmount
    return () => {
      if (isMobileClient) {
        // ScreenShareService doesn't need explicit shutdown, it handles cleanup automatically
      }
      // Clean up mobile preview image URL
      if (mobilePreviewImage) {
        URL.revokeObjectURL(mobilePreviewImage);
      }
    };
  }, [isMobile, model, screenShareService, isMobileClient, mobilePreviewImage]);

  // Initialize desktop client if on desktop
  useEffect(() => {
    const initDesktopClient = async () => {
      if (!isMobile) {
        try {
          console.log('🖥️ Initializing desktop client for real-time sync...');
          
          // Initialize ScreenShareService as desktop client
          await screenShareService.initialize(true);
          
          screenShareService.onConnectionStatusHandler((connected) => {
            console.log('🖥️ Desktop connection status:', connected ? 'Connected' : 'Disconnected');
            setIsConnected(connected);
          });
          
          screenShareService.onModeChangeHandler((modeData) => {
            console.log('🖥️ Mode change received from mobile:', modeData.mode);
            
            if (modeData.mode === 'webcam') {
              // Desktop should show camera feed from mobile, not switch to webcam mode
              console.log('🖥️ Mobile switched to webcam mode - desktop will show camera feed');
              // Keep desktop in upload mode but show camera feed from mobile
              // The camera feed will be displayed via the preview image from screen sharing
            } else {
              // For upload mode, desktop can switch normally
              console.log('🖥️ Mobile switched to upload mode - desktop switching to upload');
              setActiveMode(modeData.mode);
              // Clear the camera feed when switching to upload mode
              setDesktopCameraFeed(null);
            }
          });
          
          screenShareService.onPreviewUpdateHandler((previewData) => {
            console.log('🖥️ Preview update received from mobile');
            // When mobile is in webcam mode, display the camera feed
            if (previewData.source === 'mobile') {
              // Set the desktop camera feed to show the camera feed from mobile
              console.log('🖥️ Setting mobile camera feed as desktop preview');
              setDesktopCameraFeed(previewData.imageData);
            }
          });
          
          console.log('✅ Desktop client initialized');
        } catch (error) {
          console.error('❌ Failed to initialize desktop client:', error);
          
          // Check if it's a network validation error
          if (error instanceof Error && error.message.includes('network')) {
            setNetworkError('Network validation failed: Desktop must be on a local network');
          }
        }
      }
    };
    
    initDesktopClient();
  }, [isMobile, screenShareService]);

  // Debug function to test mobile detection and camera availability
  const debugMobileCamera = () => {
    console.log('🔍 Mobile Camera Debug:');
    console.log('- isMobile:', isMobile);
    console.log('- navigator.mediaDevices:', !!navigator.mediaDevices);
    console.log('- navigator.mediaDevices.getUserMedia:', !!navigator.mediaDevices?.getUserMedia);
    console.log('- window.location.protocol:', window.location.protocol);
    console.log('- window.location.hostname:', window.location.hostname);
    console.log('- User Agent:', navigator.userAgent);
    console.log('- Screen width:', window.innerWidth);
    console.log('- Screen height:', window.innerHeight);
    console.log('- Touch capability:', 'ontouchstart' in window || navigator.maxTouchPoints > 0);
    
    // Test camera permissions
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.permissions.query({ name: 'camera' as PermissionName })
        .then(permission => {
          console.log('- Camera permission status:', permission.state);
        })
        .catch(err => {
          console.log('- Camera permission check failed:', err);
        });
    }
  };

  // Enhanced webcam click handler
  const handleWebcamClick = () => {
    console.log('📷 Webcam button clicked, current mode:', activeMode, 'isMobile:', isMobile, 'isCameraStarting:', isCameraStarting, 'mobileWebcam.current:', !!mobileWebcam.current);
    
    // Run debug function
    debugMobileCamera();
    
    if (activeMode !== 'webcam') {
      // Switch to webcam mode first
      console.log('🔄 Switching to webcam mode');
      
      if (!isMobile && desktopCameraFeed) {
        // Desktop has active camera feed from mobile, don't switch to webcam mode
        console.log('🖥️ Desktop has active camera feed from mobile, staying in upload mode');
        toast({
          title: 'Mobile Camera Active',
          description: 'Camera feed from mobile is being displayed. No action needed.',
          variant: 'default'
        });
        return;
      }
      
      setActiveMode('webcam');
      screenShareService.shareModeChange('webcam');
      // The camera will start automatically via the useEffect when mode changes
      return;
    }
    
    // Already in webcam mode, toggle camera on/off
    if (isCameraStarting) {
      // Camera is starting, can't stop yet
      console.log('⏳ Camera is starting, ignoring click');
      return;
    }
    
    if (mobileWebcam.current) {
      // Camera is active, stop it
      console.log('🛑 Stopping active camera');
      stopCamera();
    } else if (isMobile) {
      // Camera is inactive, start it (mobile only)
      console.log('🚀 Starting inactive camera on mobile');
      startCamera();
    } else {
      // Desktop - show message that camera is not available
      console.log('💻 Desktop detected - camera not available');
      toast({
        title: 'Camera Not Available',
        description: 'Camera is only available on mobile devices. Please upload an image instead.',
        variant: 'default'
      });
    }
  };

  // handleUploadClick is no longer needed since dropdown handles mode switching

  // Prediction functionality for real-time sync (mobile client)
  const predictWithRemoteModel = useCallback(async (imageElement: HTMLImageElement | HTMLVideoElement) => {
    if (!isConnected || !isMobileClient) {
      throw new Error('Not connected to desktop');
    }
    
    try {
      // Create canvas for processing
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        // Draw the image to canvas for processing
        ctx.drawImage(imageElement, 0, 0, 224, 224);
        
        // Convert to base64 for WebSocket transmission (original quality PNG)
        const imageData = canvas.toDataURL('image/png');
        
        // Send image to desktop for prediction
        screenShareService.sharePreviewImage(imageData);
        
        // Note: Mobile client doesn't get immediate response, waits for desktop to send prediction results
        console.log('📱 Image sent to desktop for prediction');
        
        // Return empty array - actual results will come via WebSocket event
        return [];
      }
      
      throw new Error('Failed to create canvas context');
    } catch (error) {
      console.error('Error in predictWithRemoteModel:', error);
      throw error;
    }
  }, [isConnected, isMobileClient, screenShareService]);

  // Handle preview file upload for local prediction
  const onHandleLocalPreviewFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      console.error('❌ Selected file is not an image:', file.type);
      return;
    }
    
    try {
      const imageUrl = URL.createObjectURL(file);
      const img = new Image();
      
      img.onload = async () => {
        try {
          // Process the image for prediction
          console.log('📷 Processing image for local prediction:', file.name);
          
          // Set the uploaded image as preview
          setMobilePreviewImage(imageUrl);
          
          // If we have a model, make prediction
          if (model) {
            console.log('🤖 Making prediction with local model');
            try {
              const predictions = await model.predict(img as any);
              console.log('✅ Local prediction completed:', predictions);
              
              // You can update the UI with prediction results here
              // For example, setPredictionResults(predictions);
            } catch (predictionError) {
              console.error('❌ Error making local prediction:', predictionError);
            }
          } else {
            console.log('⚠️ No model loaded for local prediction');
          }
          
        } catch (error) {
          console.error('❌ Error processing image:', error);
          URL.revokeObjectURL(imageUrl);
        }
      };
      
      img.onerror = () => {
        console.error('❌ Failed to load image');
        URL.revokeObjectURL(imageUrl);
      };
      
      img.src = imageUrl;
    } catch (error) {
      console.error('❌ Error handling file upload:', error);
    }
  }, [model]);

  // File upload handler
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    // Mobile client: Send images to desktop for prediction
    if (isConnected && isMobileClient) {
      // Process each file
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
          console.warn(`Skipping non-image file: ${file.name}`);
          continue;
        }
        
        try {
          console.log(`📱 Sending image ${i + 1}/${files.length} to desktop for prediction: ${file.name}`);
          
          const imageUrl = URL.createObjectURL(file);
          const img = new Image();
          
          img.onload = async () => {
            try {
              // Create canvas to convert image to base64
              const canvas = document.createElement('canvas');
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext('2d');
              
              if (ctx) {
                ctx.drawImage(img, 0, 0);
                // Send original quality image (PNG format to avoid compression)
                const imageData = canvas.toDataURL('image/png');
                
                // Set preview image for display
                if (i === 0) { // Only set preview for the first image
                  const previewUrl = URL.createObjectURL(file);
                  // Clean up previous mobile preview image URL if exists
                  if (mobilePreviewImage) {
                    URL.revokeObjectURL(mobilePreviewImage);
                  }
                  setMobilePreviewImage(previewUrl);
                  console.log(`📱 Preview image set: ${file.name}`);
                }
                
                // Send image to desktop via WebSocket
                screenShareService.sharePreviewImage(imageData);
                console.log(`✅ Original quality image sent to desktop for prediction: ${file.name}`);
              }
            } catch (error) {
              console.error('Error sending image to desktop:', error);
            } finally {
              // Clean up the object URL
              URL.revokeObjectURL(imageUrl);
            }
          };
          
          img.onerror = () => {
            console.error('❌ Failed to load image');
            URL.revokeObjectURL(imageUrl);
          };
          
          img.src = imageUrl;
        } catch (error) {
          console.error(`Error processing file for desktop prediction: ${file.name}`, error);
        }
      }
    }
    // Desktop/local: Use local prediction
    else {
      console.log('🖥️ Processing files for local prediction on desktop');
      
      // Process each file
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
          console.warn(`Skipping non-image file: ${file.name}`);
          continue;
        }
        
        try {
          console.log(`🖥️ Processing image ${i + 1}/${files.length} for local prediction: ${file.name}`);
          
          const imageUrl = URL.createObjectURL(file);
          const img = new Image();
          
          img.onload = async () => {
            try {
              // Set preview image for display (only for the first image)
              if (i === 0) {
                // Clean up previous preview image URL if exists
                if (mobilePreviewImage) {
                  URL.revokeObjectURL(mobilePreviewImage);
                }
                setMobilePreviewImage(imageUrl);
                console.log(`🖥️ Preview image set: ${file.name}`);
              }
              
              // If we have a model, make prediction
              if (model) {
                console.log('🤖 Making prediction with local model on desktop');
                try {
                  const predictions = await model.predict(img as any);
                  console.log('✅ Desktop local prediction completed:', predictions);
                  
                  // You can update the UI with prediction results here
                  // For example, setPredictionResults(predictions);
                } catch (predictionError) {
                  console.error('❌ Error making desktop local prediction:', predictionError);
                }
              } else {
                console.log('⚠️ No model loaded for desktop local prediction');
              }
              
            } catch (error) {
              console.error('❌ Error processing desktop image:', error);
              // Don't revoke URL here if it's set as preview, it will be cleaned up when next image is loaded
              if (i !== 0) {
                URL.revokeObjectURL(imageUrl);
              }
            }
          };
          
          img.onerror = () => {
            console.error('❌ Failed to load desktop image');
            URL.revokeObjectURL(imageUrl);
          };
          
          img.src = imageUrl;
        } catch (error) {
          console.error(`Error processing file for desktop local prediction: ${file.name}`, error);
        }
      }
    }
  }, [isConnected, isMobileClient, screenShareService, mobilePreviewImage, model]);

  // Screen sharing functionality
  const startScreenSharing = useCallback(() => {
    if (!isMobile || !mobileWebcam.current) {
      console.log('❌ Screen sharing blocked - isMobile:', isMobile, 'mobileWebcam.current:', !!mobileWebcam.current);
      return;
    }
    
    console.log('📤 Starting continuous screen sharing...');
    console.log('- mobileWebcam.current exists:', !!mobileWebcam.current);
    console.log('- mobileWebcam.isCameraActive:', mobileWebcam.current.isCameraActive());
    
    // Clear any existing interval
    if (screenShareIntervalRef.current) {
      clearInterval(screenShareIntervalRef.current);
    }
    
    // Start capturing and sharing frames every 100ms
    screenShareIntervalRef.current = setInterval(() => {
      try {
        const videoElement = mobileWebcam.current?.getVideo();
        console.log('📹 Screen sharing tick - videoElement:', !!videoElement, 'readyState:', videoElement?.readyState);
        
        if (videoElement && videoElement.readyState === 4) { // HAVE_ENOUGH_DATA
          console.log('📹 Video ready, capturing frame...');
          // Capture frame using MobileWebcam's built-in method
          const canvas = mobileWebcam.current?.captureFrame();
          console.log('🖼️ Canvas captured:', !!canvas);
          
          if (canvas) {
            // Get image data URL
            const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
            console.log('📤 Image data URL length:', imageDataUrl.length);
            
            // Share via ScreenShareService
            screenShareService.sharePreviewImage(imageDataUrl);
            console.log('📡 Image shared via ScreenShareService');
            
            // Also update local preview
            setMobilePreviewImage(imageDataUrl);
            console.log('📱 Local preview updated');
          } else {
            console.log('❌ Canvas capture failed');
          }
        } else {
          console.log('⏳ Video not ready yet, readyState:', videoElement?.readyState);
        }
      } catch (error) {
        console.error('❌ Error capturing frame for screen sharing:', error);
      }
    }, 100); // Share 10 frames per second
    
    console.log('✅ Screen sharing interval started');
  }, [isMobile, screenShareService]);

  const stopScreenSharing = useCallback(() => {
    if (screenShareIntervalRef.current) {
      clearInterval(screenShareIntervalRef.current);
      screenShareIntervalRef.current = null;
      console.log('📤 Screen sharing stopped');
    }
  }, []);

  // Camera functionality with enhanced error handling
  const startCamera = useCallback(async () => {
    console.log('📷 startCamera called - isMobile:', isMobile, 'webcamRef.current:', !!webcamRef.current);
    if (!isMobile || !webcamRef.current) {
      console.log('❌ startCamera blocked - isMobile:', isMobile, 'webcamRef.current:', !!webcamRef.current);
      return;
    }
    
    console.log('🚀 Starting camera process...');
    setIsCameraStarting(true);
    setCameraError(null);
    
    try {
      // Check if camera permissions are already granted
      const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
      console.log('📋 Camera permission status:', permissions.state);
      
      if (permissions.state === 'denied') {
        throw new Error('Camera permission denied. Please enable camera permissions in your browser settings and try again.');
      }
      
      // Stop existing camera if any
      if (mobileWebcam.current) {
        console.log('🔄 Stopping existing camera instance');
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
      
      // Clear previous content
      webcamRef.current.innerHTML = '';
      
      // Show loading indicator
      webcamRef.current.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-gray-500">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div>
          <div class="text-sm">Starting camera...</div>
          <div class="text-xs mt-1">Please allow camera access when prompted</div>
        </div>
      `;
      
      // Create new mobile webcam instance with enhanced config
      mobileWebcam.current = new MobileWebcam({
        width: 300,
        height: 300,
        facingMode: 'environment', // Use rear camera by default
        timeout: 20000 // 20 seconds timeout
      });
      
      // Start camera with enhanced error handling
      console.log('📷 Initializing camera...');
      const videoElement = await mobileWebcam.current.start();
      
      // Clear loading indicator and add video
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      // Start screen sharing only after camera is successfully started
      console.log('📡 Starting screen sharing...');
      startScreenSharing();
      console.log('📡 Screen sharing function called');
      
      console.log('✅ Camera started successfully');
    } catch (error) {
      console.error('❌ Error starting camera:', error);
      
      // Provide user-friendly error messages
      let userErrorMessage = 'Failed to start camera';
      
      if (error instanceof Error) {
        if (error.message.includes('permission denied') || error.message.includes('Permission denied')) {
          userErrorMessage = 'Camera permission denied. Please check your browser settings and allow camera access.';
        } else if (error.message.includes('timeout')) {
          userErrorMessage = 'Camera startup timed out. Please check if another app is using the camera and try again.';
        } else if (error.message.includes('not supported')) {
          userErrorMessage = 'Camera not supported on this device or browser.';
        } else if (error.message.includes('in use') || error.message.includes('already in use')) {
          userErrorMessage = 'Camera is already in use by another application. Please close other apps using the camera.';
        } else {
          userErrorMessage = error.message;
        }
      }
      
      setCameraError(userErrorMessage);
      
      // Show error state in webcam area
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `
          <div class="flex flex-col items-center justify-center h-full text-red-500 p-4 text-center">
            <div class="text-lg mb-2">❌ Camera Error</div>
            <div class="text-sm mb-3">${userErrorMessage}</div>
            <div class="text-xs text-gray-500">Troubleshooting tips:</div>
            <div class="text-xs text-gray-500 mt-1">• Check camera permissions</div>
            <div class="text-xs text-gray-500">• Close other camera apps</div>
            <div class="text-xs text-gray-500">• Try refreshing the page</div>
            <div class="text-xs text-gray-500">• Ensure HTTPS connection</div>
          </div>
        `;
      }
      
      // Clean up failed camera instance
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
    } finally {
      setIsCameraStarting(false);
    }
  }, [isMobile, startScreenSharing]);

  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...');
    
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
      mobileWebcam.current = null;
    }
    
    stopScreenSharing();
    
    // Clear camera display area
    if (webcamRef.current) {
      webcamRef.current.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-gray-400">
          <div class="text-lg mb-2">📷</div>
          <div class="text-sm">Camera stopped</div>
          <div class="text-xs mt-1">Click to start camera</div>
        </div>
      `;
    }
    
    console.log('✅ Camera stopped successfully');
  }, [stopScreenSharing]);

  const switchCamera = async () => {
    if (!mobileWebcam.current || !isMobile || !webcamRef.current) return;
    
    console.log('🔄 Switching camera...');
    
    try {
      // Show switching indicator
      webcamRef.current.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-gray-500">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div>
          <div class="text-sm">Switching camera...</div>
        </div>
      `;
      
      // Switch camera
      const videoElement = await mobileWebcam.current.switchCamera();
      
      // Update display
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      console.log('✅ Camera switched successfully');
    } catch (error) {
      console.error('❌ Error switching camera:', error);
      
      let userErrorMessage = 'Failed to switch camera';
      if (error instanceof Error) {
        userErrorMessage = error.message;
      }
      
      setCameraError(userErrorMessage);
      
      // Show error state
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `
          <div class="flex flex-col items-center justify-center h-full text-red-500 p-4 text-center">
            <div class="text-lg mb-2">❌ Camera Switch Error</div>
            <div class="text-sm mb-3">${userErrorMessage}</div>
            <div class="text-xs text-gray-500">The current camera will continue to work</div>
          </div>
        `;
      }
    }
  };

  // Handle webcam mode changes with enhanced error handling
  useEffect(() => {
    console.log('🔄 Webcam mode changed to:', activeMode, 'isMobile:', isMobile);
    
    if (activeMode === 'webcam' && isMobile) {
      // Start camera when switching to webcam mode (mobile only)
      console.log('📱 Starting camera on mobile device');
      startCamera().catch((error) => {
        console.error('❌ Failed to start camera on mode change:', error);
        // Error is already handled in startCamera function
      });
    } else if (activeMode === 'webcam' && !isMobile) {
      // Show desktop message instead of attempting camera start
      console.log('💻 Desktop detected - camera not available');
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `
          <div class="flex flex-col items-center justify-center h-full text-gray-400 p-4 text-center">
            <div class="text-lg mb-2">💻</div>
            <div class="text-sm mb-2">Desktop Mode</div>
            <div class="text-xs">Camera is only available on mobile devices</div>
            <div class="text-xs mt-2">Please use a mobile device or upload an image</div>
          </div>
        `;
      }
    } else {
      // Stop camera when switching away from webcam mode
      stopCamera();
      // Clear camera display area when switching to upload mode
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `
          <div class="flex flex-col items-center justify-center h-full text-gray-400">
            <div class="text-lg mb-2">📁</div>
            <div class="text-sm">Upload Mode</div>
            <div class="text-xs mt-1">Select an image to analyze</div>
          </div>
        `;
      }
    }

    // Cleanup on unmount
    return () => {
      console.log('🧹 Cleaning up camera on unmount');
      stopCamera();
    };
  }, [activeMode, isMobile, startCamera, stopCamera]);

  // Handle main model changes - set training timestamp to current time when main model is available
  useEffect(() => {
    if (model) {
      setModelTrainedAt(new Date());
    } else {
      setModelTrainedAt(null);
    }
  }, [model]);

  // Reset isUsingRemoteModel when local predictions are updated
  useEffect(() => {
    if (predictions.length > 0 && isUsingRemoteModel) {
      console.log('🔄 Local predictions updated, switching from remote to local predictions');
      setIsUsingRemoteModel(false);
    }
  }, [predictions, isUsingRemoteModel]);

  return (
    isMobile ? (
      // Mobile layout - direct page content without any containers
      <>
        {/* Mobile page title and menu - direct page elements */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {showModels ? (
              <>
                <Brain className="w-6 h-6" />
                <span className="text-base">Current Model</span>
              </>
            ) : (
              <>
                <FileImage className="w-6 h-6" />
                <span className="text-base">Scan Results</span>
              </>
            )}
          </div>
          <div className="flex gap-1">
            <ConnectionDropdown isMobile={isMobile} onConnectionStatusChange={setIsConnected} />
            <Button 
              variant="ghost" 
              size="default"
              onClick={onToggleModels}
              className="h-10 w-10 p-0"
              title={ showModels ? "Back to Preview" : "Models"}
            >
              {showModels ? <X className="w-5 h-5" /> : <List className="w-5 h-5" />}
            </Button>
          </div>
        </div>
        
        
        {/* Network error display */}
        {networkError && (
          <div className="text-xs px-3 py-2 rounded-md mb-4 bg-red-100 text-red-800">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span>{networkError}</span>
            </div>
          </div>
        )}
        
        {/* Mobile content - direct page content */}
          {showModels ? (
            // Models View - shows current model and Change button
            <>
              {model ? (
                <div className="space-y-4">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-fit"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Change Model
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem onClick={onCloudModelSelect || onChangeModel}>
                        <Cloud className="mr-2 h-4 w-4" />
                        <span>Cloud</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={onLocalModelSelect || (() => {})}>
                        <FolderOpen className="mr-2 h-4 w-4" />
                        <span>Local</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <div className="p-4 bg-white rounded-lg border shadow-sm">
                  <div className="space-y-3">
                    <div className="text-sm text-gray-900 font-medium">
                      {modelTrainedAt ? formatDateTime(modelTrainedAt) : 'Just now'}
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-gray-700">Trained Students:</div>
                      <div className="space-y-1">
                        {model ? (
                          model.getClassLabels().map((studentName, index) => (
                            <div key={index} className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              {studentName}
                            </div>
                          ))
                        ) : (
                          model.getClassLabels().map((className, index) => (
                            <div key={index} className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                              {className}
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                  </div>
                </div>
              ) : (
                // Show model selection when no model is loaded
                <div className="text-center py-8">
                  <Brain className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <h3 className="text-base font-medium text-gray-500 mb-2">No model loaded</h3>
                  <p className="text-gray-400 text-sm mb-4">Select a model to start making predictions</p>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="outline"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Select Model
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem onClick={onCloudModelSelect || onChangeModel}>
                        <Cloud className="mr-2 h-4 w-4" />
                        <span>Cloud</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={onLocalModelSelect || (() => {})}>
                        <FolderOpen className="mr-2 h-4 w-4" />
                        <span>Local</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}
            </>
          ) : (
            // Preview View - default state
            <>
              <div className="flex flex-row gap-3">
                {/* Mode Selector Dropdown */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="p-2 h-10 w-10">
                      <ChevronDown className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-48">
                    <DropdownMenuItem onClick={() => {
                      setActiveMode('webcam');
                      screenShareService.shareModeChange('webcam');
                    }}>
                      <Camera className="mr-2 h-4 w-4" />
                      <span>Webcam</span>
                      {activeMode === 'webcam' && (
                        <div className="w-2 h-2 bg-green-500 rounded-full ml-auto"></div>
                      )}
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => {
                      setActiveMode('upload');
                      screenShareService.shareModeChange('upload');
                    }}>
                      <FolderOpen className="mr-2 h-4 w-4" />
                      <span>Upload</span>
                      {activeMode === 'upload' && (
                        <div className="w-2 h-2 bg-blue-500 rounded-full ml-auto"></div>
                      )}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                
                {/* Dynamic Action Button */}
                {activeMode === 'webcam' ? (
                  <Button
                    variant={activeMode === 'webcam' ? 'default' : 'outline'}
                    className={`justify-center w-auto py-3 ${activeMode === 'webcam' ? 'bg-green-600 hover:bg-green-700' : ''}`}
                    onClick={handleWebcamClick}
                  >
                    <div className="flex items-center gap-2">
                      {isCameraStarting ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <Camera className="w-5 h-5" />
                      )}
                      <span>{isCameraStarting ? 'Starting...' : 'Webcam'}</span>
                      {activeMode === 'webcam' && !isCameraStarting && (
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse ml-1"></div>
                      )}
                    </div>
                  </Button>
                ) : (
                  <div className="relative w-auto">
                    <input 
                      type="file" 
                      accept="image/*" 
                      multiple
                      className="hidden"
                      id="mobile-file-upload"
                      onChange={handleFileUpload}
                    />
                    <label htmlFor="mobile-file-upload" className="cursor-pointer w-auto">
                      <Button 
                        variant={activeMode === 'upload' ? 'default' : 'outline'}
                        className={`justify-center w-auto py-3 ${activeMode === 'upload' ? 'bg-blue-600 hover:bg-blue-700' : ''}`} 
                        asChild
                      >
                        <span>
                          <FolderOpen className="w-5 h-5 mr-2" />
                          Upload
                          {activeMode === 'upload' && (
                            <div className="w-2 h-2 bg-blue-400 rounded-full ml-2"></div>
                          )}
                        </span>
                      </Button>
                    </label>
                  </div>
                )}
              </div>
              
              
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg min-h-[250px] flex items-center justify-center bg-gray-50 relative transition-colors hover:border-blue-400 z-10"
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.add('border-blue-400', 'bg-blue-50');
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
                  const files = e.dataTransfer.files;
                  if (files.length > 0) {
                    const fileInput = document.getElementById('mobile-file-upload') as HTMLInputElement;
                    if (fileInput) {
                      fileInput.files = files;
                      const event = new Event('change', { bubbles: true });
                      fileInput.dispatchEvent(event);
                    }
                  }
                }}
              >
                {/* Camera display */}
                <div ref={webcamRef} className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden`}></div>
                
                {/* Loading state */}
                {isCameraStarting && (
                  <div className="absolute inset-[2px] flex items-center justify-center bg-gray-50 bg-opacity-75 z-20 rounded-lg overflow-hidden">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                  </div>
                )}
                
                {/* Error state */}
                {cameraError && (
                  <div className="absolute inset-[2px] flex items-center justify-center bg-red-50 z-20 rounded-lg overflow-hidden">
                    <div className="text-center p-4">
                      <div className="text-red-600 font-medium mb-2">Camera Error</div>
                      <div className="text-red-500 text-sm">{cameraError}</div>
                    </div>
                  </div>
                )}
                
                {/* Preview image or placeholder */}
                {activeMode === 'upload' && (
                  (isMobileClient && mobilePreviewImage) || previewImage || desktopCameraFeed ? (
                    <div className="absolute inset-[2px] flex items-center justify-center z-30 rounded-lg overflow-hidden">
                      <img 
                        src={isMobileClient && mobilePreviewImage ? mobilePreviewImage : (previewImage || desktopCameraFeed)} 
                        alt="Preview" 
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  ) : (
                    <div className="text-gray-500 text-center">
                      <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                      Upload an image or use camera
                    </div>
                  )
                )}
              </div>
              
              <div className="space-y-3">
                {/* Desktop connection status indicator */}
                {!isMobile && (
                  <div className="flex items-center gap-2 text-xs px-3 py-2 rounded-md">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                    <span className={isConnected ? 'text-green-700' : 'text-yellow-700'}>
                      {isConnected ? 'Mobile device connected' : 'Waiting for mobile connection...'}
                    </span>
                  </div>
                )}
                
                {/* Remote model indicator */}
                {isUsingRemoteModel && (
                  <div className="flex items-center gap-2 text-xs bg-blue-50 text-blue-700 px-3 py-2 rounded-md">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span>Using model from desktop</span>
                  </div>
                )}
                <h4 className="font-medium">Prediction Results</h4>
                <div className="space-y-2">
                  {displayPredictions.length > 0 ? (
                    <>
                      {/* Top 1 prediction - always static */}
                      {displayPredictions.length > 0 && (
                        <div className="space-y-2 pb-2 border-b border-gray-200">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">{displayPredictions[0].className}</span>
                            <span className="text-sm text-gray-600">{(displayPredictions[0].confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${displayPredictions[0].confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      
                      {/* Scrollable additional predictions */}
                      <div className="max-h-32 overflow-y-auto hide-scrollbar space-y-2">
                        {displayPredictions.slice(1, visiblePredictions > 1 ? visiblePredictions : displayPredictions.length).map((pred, index) => {
                        const actualIndex = index + 1; // Start from 1 (since 0 is the top prediction)
                        const confidence = pred.confidence * 100;
                        
                        // Color coding: 2nd (index 1) = yellow, rest = red
                        const isSecond = actualIndex === 1;
                        const colorClass = isSecond ? 'text-yellow-700' : 'text-red-700';
                        const fillColorClass = isSecond ? 'bg-yellow-500' : 'bg-red-600';
                        
                        return (
                          <div key={actualIndex} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className={`text-sm ${colorClass}`}>{pred.className}</span>
                              <span className={`text-sm ${colorClass}`}>{confidence.toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className={`${fillColorClass} h-2 rounded-full transition-all duration-300`} 
                                style={{ width: `${confidence}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                      </div>
                      
                      {/* Show more button if there are more predictions */}
                      {displayPredictions.length > visiblePredictions && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={onShowMorePredictions}
                          className="w-full"
                        >
                          Show {Math.min(3, displayPredictions.length - visiblePredictions)} More
                        </Button>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-4 text-gray-500">
                      <Brain className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                      <p className="text-sm">No predictions yet. Upload an image or use the camera.</p>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
      </>
    ) : (
      // Desktop layout - keep the card wrapper
      <Card className="h-[605px] flex flex-col">
      <CardHeader className={`${isMobile ? 'pb-3' : 'pb-2'}`}>
        <CardTitle className={`flex items-center justify-between ${isMobile ? 'text-lg' : 'text-base'}`}>
          <div className="flex items-center gap-2">
            {showModels ? (
              <>
                <Brain className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                <span className={isMobile ? 'text-base' : ''}>{ 'Current Model'}</span>
              </>
            ) : (
              <>
                <FileImage className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                <span className={isMobile ? 'text-base' : ''}>{isMobile ? 'Scan Results' : 'Preview'}</span>
              </>
            )}
          </div>
          <div className="flex gap-1">
            <ConnectionDropdown isMobile={isMobile} onConnectionStatusChange={setIsConnected} />
            <Button 
              variant="ghost" 
              size={isMobile ? "default" : "sm"}
              onClick={onToggleModels}
              className={`${isMobile ? 'h-10 w-10' : 'h-8 w-8'} p-0`}
              title={ showModels ? "Back to Preview" : "Models"}
            >
              {showModels ? <X className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'}`} /> : <List className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'}`} />}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className={`${isMobile ? 'flex-1' : 'flex-1 overflow-hidden'} flex flex-col space-y-4`}>
        {/* Network error display */}
        {networkError && (
          <div className="text-xs px-3 py-2 rounded-md bg-red-100 text-red-800">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span>{networkError}</span>
            </div>
          </div>
        )}
        {showModels ? (
          // Models View - shows current model and Change button
          <>
            {model ? (
              <div className="space-y-4">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size={isMobile ? "sm" : "default"}
                      className="w-fit"
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Change Model
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem onClick={onCloudModelSelect || onChangeModel}>
                      <Cloud className="mr-2 h-4 w-4" />
                      <span>Cloud</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={onLocalModelSelect || (() => {})}>
                      <FolderOpen className="mr-2 h-4 w-4" />
                      <span>Local</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <div className={`p-4 bg-white rounded-lg border shadow-sm ${isMobile ? '' : 'p-6'}`}>
                <div className="space-y-3">
                  <div className={`${isMobile ? 'text-sm' : 'text-base'} text-gray-900 font-medium`}>
                    {modelTrainedAt ? formatDateTime(modelTrainedAt) : 'Just now'}
                  </div>
                  <div className="space-y-2">
                    <div className={`${isMobile ? 'text-sm' : 'text-sm'} font-medium text-gray-700`}>Trained Students:</div>
                    <div className="space-y-1">
                      {model ? (
                        model.getClassLabels().map((studentName, index) => (
                          <div key={index} className={`${isMobile ? 'text-sm' : 'text-sm'} text-gray-600 flex items-center gap-2`}>
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            {studentName}
                          </div>
                        ))
                      ) : (
                        <div className={`${isMobile ? 'text-sm' : 'text-sm'} text-gray-500 italic`}>
                          No model loaded
                        </div>
                      )}
                    </div>
                  </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className={`text-center ${isMobile ? 'py-8' : 'py-12'}`}>
                <Brain className={`${isMobile ? 'w-12 h-12' : 'w-16 h-16'} mx-auto mb-4 text-gray-300`} />
                <h3 className={`${isMobile ? 'text-base' : 'text-lg'} font-medium text-gray-500 mb-2`}>No model loaded</h3>
                <p className="text-gray-400 text-sm mb-4">Select a model to start making predictions</p>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size={isMobile ? "sm" : "default"}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Select Model
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem onClick={onCloudModelSelect || onChangeModel}>
                      <Cloud className="mr-2 h-4 w-4" />
                      <span>Cloud</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={onLocalModelSelect || (() => {})}>
                      <FolderOpen className="mr-2 h-4 w-4" />
                      <span>Local</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}
          </>
        ) : (
          // Preview View - default state
          <div className="space-y-4">
            <div className="flex flex-row gap-3">
              {/* Mode Selector Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="p-2 h-10 w-10">
                    <ChevronDown className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-48">
                  <DropdownMenuItem onClick={() => {
                    setActiveMode('webcam');
                    screenShareService.shareModeChange('webcam');
                  }}>
                    <Camera className="mr-2 h-4 w-4" />
                    <span>Webcam</span>
                    {activeMode === 'webcam' && (
                      <div className="w-2 h-2 bg-green-500 rounded-full ml-auto"></div>
                    )}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => {
                    setActiveMode('upload');
                    screenShareService.shareModeChange('upload');
                  }}>
                    <FolderOpen className="mr-2 h-4 w-4" />
                    <span>Upload</span>
                    {activeMode === 'upload' && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full ml-auto"></div>
                    )}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              
              {/* Dynamic Action Button */}
              {activeMode === 'webcam' ? (
                <Button
                  variant={activeMode === 'webcam' ? 'default' : 'outline'}
                  className={`justify-center w-auto py-3 ${activeMode === 'webcam' ? 'bg-green-600 hover:bg-green-700' : ''}`}
                  onClick={handleWebcamClick}
                >
                  <div className="flex items-center gap-2">
                    {isCameraStarting ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Camera className="w-5 h-5" />
                    )}
                    <span>{isCameraStarting ? 'Starting...' : 'Webcam'}</span>
                    {activeMode === 'webcam' && !isCameraStarting && (
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse ml-1"></div>
                    )}
                  </div>
                </Button>
              ) : (
                <div className="relative w-auto">
                  <input 
                    type="file" 
                    accept="image/*" 
                    multiple
                    className="hidden"
                    id="stored-model-file-upload"
                    onChange={onHandlePreviewFileUpload}
                  />
                  <label htmlFor="stored-model-file-upload" className="cursor-pointer w-auto">
                    <Button 
                      variant={activeMode === 'upload' ? 'default' : 'outline'}
                      className={`justify-center w-auto py-3 ${activeMode === 'upload' ? 'bg-blue-600 hover:bg-blue-700' : ''}`} 
                      asChild
                    >
                      <span>
                        <FolderOpen className="w-5 h-5 mr-2" />
                        Upload
                        {activeMode === 'upload' && (
                          <div className="w-2 h-2 bg-blue-400 rounded-full ml-2"></div>
                        )}
                      </span>
                    </Button>
                  </label>
                </div>
              )}
            </div>
            
            <div 
              className={`border-2 border-dashed border-gray-300 rounded-lg min-h-[250px] max-h-[250px] flex items-center justify-center bg-gray-50 relative transition-colors hover:border-blue-400`}
              onDragOver={(e) => {
                e.preventDefault();
                e.currentTarget.classList.add('border-blue-400', 'bg-blue-50');
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                  const fileInput = document.getElementById('stored-model-file-upload') as HTMLInputElement;
                  if (fileInput) {
                    fileInput.files = files;
                    const event = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(event);
                  }
                }
              }}
            >
              {/* Camera display */}
              <div ref={webcamRef} className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden`}></div>
              
              {/* Loading state */}
              {isCameraStarting && (
                <div className="absolute inset-[2px] flex items-center justify-center bg-gray-50 bg-opacity-75 z-20 rounded-lg overflow-hidden">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                </div>
              )}
              
              {/* Error state */}
              {cameraError && (
                <div className="absolute inset-[2px] flex items-center justify-center bg-red-50 z-20 rounded-lg overflow-hidden">
                  <div className="text-center p-4">
                    <div className="text-red-600 font-medium mb-2">Camera Error</div>
                    <div className="text-red-500 text-sm">{cameraError}</div>
                  </div>
                </div>
              )}
              
              {activeMode === 'upload' && (
                ((isMobileClient && mobilePreviewImage) || previewImage || desktopCameraFeed) ? (
                  <div className="absolute inset-[2px] flex items-center justify-center z-30 rounded-lg overflow-hidden">
                    <img 
                      src={isMobileClient && mobilePreviewImage ? mobilePreviewImage : (previewImage || desktopCameraFeed)} 
                      alt="Preview" 
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="text-gray-500 text-center">
                    <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                    Upload an image or use camera
                  </div>
                )
              )}
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium">Prediction Results</h4>
              <div className="space-y-2">
                {predictions.length > 0 ? (
                  <>
                    {/* Top 1 prediction - always static */}
                    <div className="space-y-2 pb-2 border-b border-gray-200">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">{predictions[0].className}</span>
                        <span className="text-sm text-gray-600">{(predictions[0].confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${predictions[0].confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    
                    {/* Scrollable additional predictions */}
                    <div className="max-h-32 overflow-y-auto hide-scrollbar space-y-2">
                      {predictions.slice(1, visiblePredictions > 1 ? visiblePredictions : predictions.length).map((pred, index) => {
                      const actualIndex = index + 1; // Start from 1 (since 0 is the top prediction)
                      const confidence = pred.confidence * 100;
                      
                      // Color coding: 2nd (index 1) = yellow, rest = red
                      const isSecond = actualIndex === 1;
                      const colorClass = isSecond ? 'text-yellow-700' : 'text-red-700';
                      const fillColorClass = isSecond ? 'bg-yellow-500' : 'bg-red-600';
                      
                      return (
                        <div key={actualIndex} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">{pred.className}</span>
                            <span className="text-sm text-gray-600">{confidence.toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className={`${fillColorClass} h-2 rounded-full transition-all duration-300`} 
                              style={{ width: `${confidence}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                    
                    {predictions.length > Math.max(1, visiblePredictions) && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={onShowMorePredictions}
                        className="w-full mt-3"
                      >
                        See More ({predictions.length - Math.max(1, visiblePredictions)} remaining)
                      </Button>
                    )}
                  </>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    Train a model to see predictions
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
    )
  );
};

export default Preview;
