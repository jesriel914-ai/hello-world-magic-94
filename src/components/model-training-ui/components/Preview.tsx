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
import { MobileWebcam, type BoundingBox } from '../services/mobileWebcam';
import { ConnectionDropdown } from './ConnectionDropdown';
import { ScreenShareService } from '../../../services/ScreenShareService';
import { toast } from '@/hooks/use-toast';
import { isCameraLevel } from '@/utils/perspectiveCorrection';

// Helper functions
const getInitialVisibleCount = (preds: PredictionResult[]): number => {
  if (preds.length === 0) return 0;
  if (preds.length === 1) return 1;
  
  const firstConfidence = preds[0].confidence * 100;
  
  if (firstConfidence >= 99.9) {
    return 1;
  }
  
  if (preds.length === 2) return 2;
  
  const firstTwoTotal = (preds[0].confidence + preds[1].confidence) * 100;
  const thirdConfidence = preds[2].confidence * 100;
  
  if (firstTwoTotal >= 99.9 || thirdConfidence < 1) {
    return 2;
  }
  return 3;
};

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
  onChangeModel: () => void;
  onCloudModelSelect?: () => void;
  onLocalModelSelect?: () => void;
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
  const [isConnected, setIsConnected] = useState(false);
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
  
  // Detection state
  const [detectedBoxes, setDetectedBoxes] = useState<BoundingBox[]>([]);
  const [isStabilized, setIsStabilized] = useState(false);
  const [frameQuality, setFrameQuality] = useState<{
    isBlurry: boolean;
    hasGoodLighting: boolean;
  }>({ isBlurry: false, hasGoodLighting: true });
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
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
          
          console.log('✅ Mobile client initialized');
        } catch (error) {
          console.error('❌ Failed to initialize mobile client:', error);
          
          if (error instanceof Error && error.message.includes('network')) {
            setNetworkError('Network validation failed: Devices must be on the same local network');
          }
        }
      }
    };
    
    initMobileClient();
    
    return () => {
      if (isMobileClient) {
        // ScreenShareService doesn't need explicit shutdown
      }
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
          
          await screenShareService.initialize(true);
          
          screenShareService.onConnectionStatusHandler((connected) => {
            console.log('🖥️ Desktop connection status:', connected ? 'Connected' : 'Disconnected');
            setIsConnected(connected);
          });
          
          screenShareService.onModeChangeHandler((modeData) => {
            console.log('🖥️ Mode change received from mobile:', modeData.mode);
            
            if (modeData.mode === 'webcam') {
              console.log('🖥️ Mobile switched to webcam mode - desktop will show camera feed');
            } else {
              console.log('🖥️ Mobile switched to upload mode - desktop switching to upload');
              setActiveMode(modeData.mode);
              setDesktopCameraFeed(null);
            }
          });
          
          screenShareService.onPreviewUpdateHandler((previewData) => {
            console.log('🖥️ Preview update received from mobile');
            if (previewData.source === 'mobile') {
              console.log('🖥️ Setting mobile camera feed as desktop preview');
              setDesktopCameraFeed(previewData.imageData);
            }
          });
          
          console.log('✅ Desktop client initialized');
        } catch (error) {
          console.error('❌ Failed to initialize desktop client:', error);
          
          if (error instanceof Error && error.message.includes('network')) {
            setNetworkError('Network validation failed: Desktop must be on a local network');
          }
        }
      }
    };
    
    initDesktopClient();
  }, [isMobile, screenShareService]);

  // Initialize detection when camera starts
  useEffect(() => {
    if (mobileWebcam.current && activeMode === 'webcam') {
      mobileWebcam.current.initializeDetection().catch(err => {
        console.error('Failed to initialize detection:', err);
      });
    }
  }, [mobileWebcam.current, activeMode]);

  // Detection loop
  useEffect(() => {
    if (!isMobile || activeMode !== 'webcam' || !mobileWebcam.current) {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      setDetectedBoxes([]);
      return;
    }
    
    const runDetection = async () => {
      if (!mobileWebcam.current) return;
      
      try {
        // Detect signatures
        const boxes = await mobileWebcam.current.detectSignatures();
        setDetectedBoxes(boxes);
        
        // Check frame quality
        const isBlurry = mobileWebcam.current.isFrameBlurry();
        const hasGoodLighting = mobileWebcam.current.hasGoodLighting();
        setFrameQuality({ isBlurry, hasGoodLighting });
        
        // Check if camera is level (simplified)
        setIsStabilized(!isBlurry);
        
        // Send cropped signature to PC (only if quality is good and signature detected)
        if (boxes.length > 0 && !isBlurry && hasGoodLighting && isConnected) {
          const croppedCanvas = await mobileWebcam.current.cropActiveSignature();
          if (croppedCanvas) {
            const imageData = croppedCanvas.toDataURL('image/jpeg', 0.8);
            screenShareService.sharePreviewImage(imageData);
          }
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    };
    
    // Run detection every 300ms
    detectionIntervalRef.current = setInterval(runDetection, 300);
    
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    };
  }, [isMobile, activeMode, mobileWebcam.current, isConnected, screenShareService]);

  // Focus point state for visual feedback
  const [focusPoint, setFocusPoint] = useState<{x: number; y: number} | null>(null);

  // Handle box click
  const handleBoxClick = (index: number) => {
    if (!mobileWebcam.current) return;
    
    mobileWebcam.current.setActiveBox(index);
    
    const newBoxes = [...detectedBoxes];
    newBoxes.forEach((box, i) => {
      box.isActive = i === index;
    });
    setDetectedBoxes(newBoxes);
  };

  // Handle camera click/tap for focus and detection guidance
  const handleCameraClick = async (event: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    if (!mobileWebcam.current) return;
    
    const rect = event.currentTarget.getBoundingClientRect();
    let clientX: number, clientY: number;
    
    if ('touches' in event) {
      // Touch event
      if (event.touches.length === 0) return;
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      // Mouse event
      clientX = event.clientX;
      clientY = event.clientY;
    }
    
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    // Convert to normalized coordinates (0-1)
    const normalizedX = x / rect.width;
    const normalizedY = y / rect.height;
    
    // Show visual feedback
    setFocusPoint({ x, y });
    setTimeout(() => setFocusPoint(null), 1000);
    
    // 1. Apply focus at clicked point
    try {
      await mobileWebcam.current.applyFocusPoint(normalizedX, normalizedY);
      console.log('✅ Focus applied');
    } catch (error) {
      console.log('ℹ️ Focus not available on this device');
    }
    
    // 2. Guide detection to this region
    const videoElement = mobileWebcam.current.getVideo();
    if (videoElement) {
      const videoWidth = videoElement.videoWidth || 1280;
      const videoHeight = videoElement.videoHeight || 720;
      
      // Convert screen coordinates to video coordinates
      const videoX = (x / rect.width) * videoWidth;
      const videoY = (y / rect.height) * videoHeight;
      
      // Get the signature detector and set ROI
      const detector = (mobileWebcam.current as any).signatureDetector;
      if (detector && typeof detector.setRegionOfInterest === 'function') {
        detector.setRegionOfInterest(videoX, videoY, 150);
        console.log(`📍 Detection ROI set to (${videoX.toFixed(0)}, ${videoY.toFixed(0)})`);
        
        // Clear ROI after 3 seconds
        setTimeout(() => {
          if (detector && typeof detector.clearRegionOfInterest === 'function') {
            detector.clearRegionOfInterest();
          }
        }, 3000);
      }
    }
  };

  const handleWebcamClick = () => {
    console.log('📷 Webcam button clicked, current mode:', activeMode, 'isMobile:', isMobile);
    
    if (activeMode !== 'webcam') {
      console.log('🔄 Switching to webcam mode');
      
      if (!isMobile && desktopCameraFeed) {
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
      return;
    }
    
    if (isCameraStarting) {
      console.log('⏳ Camera is starting, ignoring click');
      return;
    }
    
    if (mobileWebcam.current) {
      console.log('🛑 Stopping active camera');
      stopCamera();
    } else if (isMobile) {
      console.log('🚀 Starting inactive camera on mobile');
      startCamera();
    } else {
      console.log('💻 Desktop detected - camera not available');
      toast({
        title: 'Camera Not Available',
        description: 'Camera is only available on mobile devices. Please upload an image instead.',
        variant: 'default'
      });
    }
  };

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    if (isConnected && isMobileClient) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        
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
              const canvas = document.createElement('canvas');
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext('2d');
              
              if (ctx) {
                ctx.drawImage(img, 0, 0);
                const imageData = canvas.toDataURL('image/png');
                
                if (i === 0) {
                  const previewUrl = URL.createObjectURL(file);
                  if (mobilePreviewImage) {
                    URL.revokeObjectURL(mobilePreviewImage);
                  }
                  setMobilePreviewImage(previewUrl);
                  console.log(`📱 Preview image set: ${file.name}`);
                }
                
                screenShareService.sharePreviewImage(imageData);
                console.log(`✅ Original quality image sent to desktop for prediction: ${file.name}`);
              }
            } catch (error) {
              console.error('Error sending image to desktop:', error);
            } finally {
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
    } else {
      console.log('🖥️ Processing files for local prediction on desktop');
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        
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
              if (i === 0) {
                if (mobilePreviewImage) {
                  URL.revokeObjectURL(mobilePreviewImage);
                }
                setMobilePreviewImage(imageUrl);
                console.log(`🖥️ Preview image set: ${file.name}`);
              }
              
              if (model) {
                console.log('🤖 Making prediction with local model on desktop');
                try {
                  const predictions = await model.predict(img as any);
                  console.log('✅ Desktop local prediction completed:', predictions);
                } catch (predictionError) {
                  console.error('❌ Error making desktop local prediction:', predictionError);
                }
              } else {
                console.log('⚠️ No model loaded for desktop local prediction');
              }
              
            } catch (error) {
              console.error('❌ Error processing desktop image:', error);
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

  const startCamera = useCallback(async () => {
    console.log('📷 startCamera called - isMobile:', isMobile, 'webcamRef.current:', !!webcamRef.current);
    if (!isMobile || !webcamRef.current) {
      console.log('❌ startCamera blocked');
      return;
    }
    
    console.log('🚀 Starting camera process...');
    setIsCameraStarting(true);
    setCameraError(null);
    
    try {
      const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
      console.log('📋 Camera permission status:', permissions.state);
      
      if (permissions.state === 'denied') {
        throw new Error('Camera permission denied. Please enable camera permissions in your browser settings and try again.');
      }
      
      if (mobileWebcam.current) {
        console.log('🔄 Stopping existing camera instance');
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
      
      webcamRef.current.innerHTML = '';
      
      webcamRef.current.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-gray-500">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div>
          <div class="text-sm">Starting camera...</div>
          <div class="text-xs mt-1">Please allow camera access when prompted</div>
        </div>
      `;
      
      mobileWebcam.current = new MobileWebcam({
        width: 300,
        height: 300,
        facingMode: 'environment',
        timeout: 20000
      });
      
      console.log('📷 Initializing camera...');
      const videoElement = await mobileWebcam.current.start();
      
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      // Start screen sharing after camera is successfully started
      console.log('📡 Starting screen sharing...');
      startScreenSharing();
      console.log('📡 Screen sharing function called');
      
      console.log('✅ Camera started successfully');
    } catch (error) {
      console.error('❌ Error starting camera:', error);
      
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
      
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
    } finally {
      setIsCameraStarting(false);
    }
  }, [isMobile]);

  const startScreenSharing = useCallback(() => {
    if (!isMobile || !mobileWebcam.current) {
      console.log('❌ Screen sharing blocked - isMobile:', isMobile, 'mobileWebcam.current:', !!mobileWebcam.current);
      return;
    }
    
    console.log('📤 Starting continuous screen sharing...');
    
    // Clear any existing interval
    if (screenShareIntervalRef.current) {
      clearInterval(screenShareIntervalRef.current);
    }
    
    // Start capturing and sharing frames every 100ms (10 FPS)
    screenShareIntervalRef.current = setInterval(() => {
      try {
        const videoElement = mobileWebcam.current?.getVideo();
        
        if (videoElement && videoElement.readyState === 4) {
          // Capture frame using MobileWebcam's built-in method
          const canvas = mobileWebcam.current?.captureFrame();
          
          if (canvas) {
            // Get image data URL
            const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
            
            // Share via ScreenShareService (this is the LIVE FEED, separate from detection)
            screenShareService.sharePreviewImage(imageDataUrl);
            
            // Also update local preview
            setMobilePreviewImage(imageDataUrl);
          }
        }
      } catch (error) {
        console.error('❌ Error capturing frame for screen sharing:', error);
      }
    }, 100); // 10 FPS for smooth live feed
    
    console.log('✅ Screen sharing interval started');
  }, [isMobile, screenShareService]);

  const stopScreenSharing = useCallback(() => {
    if (screenShareIntervalRef.current) {
      clearInterval(screenShareIntervalRef.current);
      screenShareIntervalRef.current = null;
      console.log('📤 Screen sharing stopped');
    }
  }, []);

  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...');
    
    stopScreenSharing();
    
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
      mobileWebcam.current = null;
    }
    
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

  useEffect(() => {
    console.log('🔄 Webcam mode changed to:', activeMode, 'isMobile:', isMobile);
    
    if (activeMode === 'webcam' && isMobile) {
      console.log('📱 Starting camera on mobile device');
      startCamera().catch((error) => {
        console.error('❌ Failed to start camera on mode change:', error);
      });
    } else if (activeMode === 'webcam' && !isMobile) {
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
      stopCamera();
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

    return () => {
      console.log('🧹 Cleaning up camera on unmount');
      stopCamera();
    };
  }, [activeMode, isMobile, startCamera, stopCamera]);

  useEffect(() => {
    if (model) {
      setModelTrainedAt(new Date());
    } else {
      setModelTrainedAt(null);
    }
  }, [model]);

  useEffect(() => {
    if (predictions.length > 0 && isUsingRemoteModel) {
      console.log('🔄 Local predictions updated, switching from remote to local predictions');
      setIsUsingRemoteModel(false);
    }
  }, [predictions, isUsingRemoteModel]);

  // Camera display with overlays - UPDATED JSX
  const renderCameraDisplay = () => (
    <div className="relative border-2 border-dashed border-gray-300 rounded-lg min-h-[250px] flex items-center justify-center bg-gray-50">
      {/* Video feed container with tap-to-focus */}
      <div 
        ref={webcamRef} 
        className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden cursor-pointer`}
        onClick={handleCameraClick}
        onTouchStart={handleCameraClick}
      />
      
      {/* Focus point indicator */}
      {focusPoint && activeMode === 'webcam' && (
        <div 
          className="absolute z-30 pointer-events-none"
          style={{
            left: focusPoint.x,
            top: focusPoint.y,
            width: '80px',
            height: '80px',
            border: '3px solid #FFD700',
            borderRadius: '50%',
            transform: 'translate(-50%, -50%)',
            animation: 'focusPulse 0.5s ease-out',
            boxShadow: '0 0 0 2px rgba(0, 0, 0, 0.3), 0 0 20px rgba(255, 215, 0, 0.5)'
          }}
        >
          <style>{`
            @keyframes focusPulse {
              0% {
                transform: translate(-50%, -50%) scale(1.5);
                opacity: 0;
              }
              50% {
                opacity: 1;
              }
              100% {
                transform: translate(-50%, -50%) scale(1);
                opacity: 0.8;
              }
            }
          `}</style>
        </div>
      )}
      
      {/* Bounding boxes overlay - THINNER BORDERS */}
      {activeMode === 'webcam' && detectedBoxes.length > 0 && (
        <div className="absolute inset-[2px] pointer-events-none z-10">
          {detectedBoxes.map((box, index) => (
            <div
              key={index}
              className={`absolute ${box.isActive ? 'border-2 border-yellow-400 shadow-lg' : 'border border-gray-300'} rounded pointer-events-auto cursor-pointer transition-all`}
              style={{
                left: `${(box.x / 300) * 100}%`,
                top: `${(box.y / 300) * 100}%`,
                width: `${(box.width / 300) * 100}%`,
                height: `${(box.height / 300) * 100}%`,
                boxShadow: box.isActive ? '0 0 15px rgba(255, 215, 0, 0.6)' : 'none'
              }}
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering camera click
                handleBoxClick(index);
              }}
            />
          ))}
        </div>
      )}
      
      {/* Stabilizer cross overlay - MUCH THINNER */}
      {activeMode === 'webcam' && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
          {/* Horizontal line - thin */}
          <div 
            className={`absolute ${isStabilized ? 'bg-yellow-400' : 'bg-white/50'} transition-colors`}
            style={{
              width: '40px',
              height: '1px',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          />
          {/* Vertical line - thin */}
          <div 
            className={`absolute ${isStabilized ? 'bg-yellow-400' : 'bg-white/50'} transition-colors`}
            style={{
              width: '1px',
              height: '40px',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          />
        </div>
      )}
      
      {/* Quality warnings */}
      {activeMode === 'webcam' && (
        <>
          {frameQuality.isBlurry && (
            <div className="absolute top-2 left-2 bg-red-500 text-white px-3 py-1.5 rounded text-xs font-medium z-20">
              Too blurry - hold steady
            </div>
          )}
          
          {!frameQuality.hasGoodLighting && (
            <div className="absolute top-10 left-2 bg-orange-500 text-white px-3 py-1.5 rounded text-xs font-medium z-20">
              Poor lighting
            </div>
          )}
          
          {detectedBoxes.length === 0 && !frameQuality.isBlurry && (
            <div className="absolute top-2 left-2 bg-blue-500 text-white px-3 py-1.5 rounded text-xs font-medium z-20">
              Scanning...
            </div>
          )}
        </>
      )}
      
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
      
      {/* Upload mode preview */}
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
  );

  return (
    isMobile ? (
      // Mobile layout
      <>
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
              title={showModels ? "Back to Preview" : "Models"}
            >
              {showModels ? <X className="w-5 h-5" /> : <List className="w-5 h-5" />}
            </Button>
          </div>
        </div>
        
        {networkError && (
          <div className="text-xs px-3 py-2 rounded-md mb-4 bg-red-100 text-red-800">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span>{networkError}</span>
            </div>
          </div>
        )}
        
        {showModels ? (
          <>
            {model ? (
              <div className="space-y-4">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="w-fit">
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
                        {model.getClassLabels().map((studentName, index) => (
                          <div key={index} className="text-sm text-gray-600 flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            {studentName}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Brain className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <h3 className="text-base font-medium text-gray-500 mb-2">No model loaded</h3>
                <p className="text-gray-400 text-sm mb-4">Select a model to start making predictions</p>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">
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
          <>
            <div className="flex flex-row gap-3">
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
            
            {renderCameraDisplay()}
            
            <div className="space-y-3">
              {!isMobile && (
                <div className="flex items-center gap-2 text-xs px-3 py-2 rounded-md">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                  <span className={isConnected ? 'text-green-700' : 'text-yellow-700'}>
                    {isConnected ? 'Mobile device connected' : 'Waiting for mobile connection...'}
                  </span>
                </div>
              )}
              
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
                    
                    <div className="max-h-32 overflow-y-auto hide-scrollbar space-y-2">
                      {displayPredictions.slice(1, visiblePredictions > 1 ? visiblePredictions : displayPredictions.length).map((pred, index) => {
                        const actualIndex = index + 1;
                        const confidence = pred.confidence * 100;
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
      // Desktop layout
      <Card className="h-[605px] flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-base">
            <div className="flex items-center gap-2">
              {showModels ? (
                <>
                  <Brain className="w-5 h-5" />
                  <span>Current Model</span>
                </>
              ) : (
                <>
                  <FileImage className="w-5 h-5" />
                  <span>Preview</span>
                </>
              )}
            </div>
            <div className="flex gap-1">
              <ConnectionDropdown isMobile={isMobile} onConnectionStatusChange={setIsConnected} />
              <Button 
                variant="ghost" 
                size="sm"
                onClick={onToggleModels}
                className="h-8 w-8 p-0"
                title={showModels ? "Back to Preview" : "Models"}
              >
                {showModels ? <X className="w-4 h-4" /> : <List className="w-4 h-4" />}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col space-y-4">
          {networkError && (
            <div className="text-xs px-3 py-2 rounded-md bg-red-100 text-red-800">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-500"></div>
                <span>{networkError}</span>
              </div>
            </div>
          )}
          
          {showModels ? (
            <>
              {model ? (
                <div className="space-y-4">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="default" className="w-fit">
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
                  <div className="p-6 bg-white rounded-lg border shadow-sm">
                    <div className="space-y-3">
                      <div className="text-base text-gray-900 font-medium">
                        {modelTrainedAt ? formatDateTime(modelTrainedAt) : 'Just now'}
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-gray-700">Trained Students:</div>
                        <div className="space-y-1">
                          {model.getClassLabels().map((studentName, index) => (
                            <div key={index} className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              {studentName}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <h3 className="text-lg font-medium text-gray-500 mb-2">No model loaded</h3>
                  <p className="text-gray-400 text-sm mb-4">Select a model to start making predictions</p>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="default">
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
            <div className="space-y-4">
              <div className="flex flex-row gap-3">
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
              
              {renderCameraDisplay()}
              
              <div className="space-y-3">
                <h4 className="font-medium">Prediction Results</h4>
                <div className="space-y-2">
                  {predictions.length > 0 ? (
                    <>
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
                      
                      <div className="max-h-32 overflow-y-auto hide-scrollbar space-y-2">
                        {predictions.slice(1, visiblePredictions > 1 ? visiblePredictions : predictions.length).map((pred, index) => {
                          const actualIndex = index + 1;
                          const confidence = pred.confidence * 100;
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