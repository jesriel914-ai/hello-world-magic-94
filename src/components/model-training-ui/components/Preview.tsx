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
  
  // Screen sharing state
  const screenShareIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Camera prediction state
  const cameraPredictionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Focus point state for tap-to-focus visual feedback
  const [focusPoint, setFocusPoint] = useState<{x: number; y: number} | null>(null);

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

  // Camera prediction loop - runs ML predictions on live camera feed
  useEffect(() => {
    if (!isMobile || activeMode !== 'webcam' || !mobileWebcam.current || !model) {
      if (cameraPredictionIntervalRef.current) {
        clearInterval(cameraPredictionIntervalRef.current);
        cameraPredictionIntervalRef.current = null;
      }
      setIsPredicting(false);
      return;
    }
    
    const runCameraPrediction = async () => {
      if (!mobileWebcam.current || !model) return;
      
      try {
        // Capture current frame
        const canvas = mobileWebcam.current.captureFrame();
        if (!canvas) return;
        
        // Make prediction on current frame
        setIsPredicting(true);
        const predictions = await model.predict(canvas, false);
        const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
        setPredictions(sortedPredictions);
        
        // Share predictions with desktop if connected
        if (isConnected) {
          screenShareService.sharePredictionResults(sortedPredictions);
        }
      } catch (error) {
        console.error('Camera prediction error:', error);
      } finally {
        setIsPredicting(false);
      }
    };
    
    // Run predictions every 500ms (2 FPS) to avoid overwhelming the system
    cameraPredictionIntervalRef.current = setInterval(runCameraPrediction, 500);
    
    return () => {
      if (cameraPredictionIntervalRef.current) {
        clearInterval(cameraPredictionIntervalRef.current);
        cameraPredictionIntervalRef.current = null;
      }
    };
  }, [isMobile, activeMode, model, isConnected, screenShareService]);

  // Handle camera tap for focus
  const handleCameraClick = async (event: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    if (!mobileWebcam.current) return;
    
    const videoElement = mobileWebcam.current.getVideo();
    if (!videoElement) return;
    
    // Get video element's bounding box
    const videoRect = videoElement.getBoundingClientRect();
    
    let clientX: number, clientY: number;
    
    if ('touches' in event) {
      if (event.touches.length === 0) return;
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = event.clientX;
      clientY = event.clientY;
    }
    
    // Check if click is within video bounds
    if (
      clientX < videoRect.left || clientX > videoRect.right ||
      clientY < videoRect.top || clientY > videoRect.bottom
    ) {
      return;
    }
    
    // Calculate position relative to video element
    const x = clientX - videoRect.left;
    const y = clientY - videoRect.top;
    
    // Show visual feedback
    setFocusPoint({ x, y });
    setTimeout(() => setFocusPoint(null), 1000);
    
    // Trigger camera to refocus
    try {
      await mobileWebcam.current.triggerFocus();
      console.log('✅ Focus triggered at', x, y);
    } catch (error) {
      console.log('ℹ️ Focus trigger not available');
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
        if (!file || !file.type.startsWith('image/')) continue;
        
        try {
          console.log(`📱 Sending image ${i + 1}/${files.length} to desktop: ${file.name}`);
          
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
                }
                
                screenShareService.sharePreviewImage(imageData);
                console.log(`✅ Image sent to desktop: ${file.name}`);
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
          console.error(`Error processing file: ${file.name}`, error);
        }
      }
    } else {
      console.log('🖥️ Processing files for local prediction on desktop');
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file || !file.type.startsWith('image/')) continue;
        
        try {
          const imageUrl = URL.createObjectURL(file);
          const img = new Image();
          
          img.onload = async () => {
            try {
              if (i === 0) {
                if (mobilePreviewImage) {
                  URL.revokeObjectURL(mobilePreviewImage);
                }
                setMobilePreviewImage(imageUrl);
              }
              
              if (model) {
                const predictions = await model.predict(img as any);
                console.log('✅ Desktop local prediction completed:', predictions);
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
          console.error(`Error processing file: ${file.name}`, error);
        }
      }
    }
  }, [isConnected, isMobileClient, screenShareService, mobilePreviewImage, model]);

  const startCamera = useCallback(async () => {
    console.log('📷 startCamera called - isMobile:', isMobile);
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
        throw new Error('Camera permission denied. Please enable camera permissions in your browser settings.');
      }
      
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
      
      webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-500"><div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div><div class="text-sm">Starting camera...</div></div>';
      
      // Create mobile webcam with zoom configuration
      mobileWebcam.current = new MobileWebcam({
        width: 300,
        height: 300,
        facingMode: 'environment',
        timeout: 20000,
        zoom: 2.0 // 2x zoom to better frame signatures
      });
      
      console.log('📷 Initializing camera with 2x zoom...');
      const videoElement = await mobileWebcam.current.start();
      
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      // Start screen sharing
      console.log('📡 Starting screen sharing...');
      startScreenSharing();
      
      console.log('✅ Camera started successfully with zoom');
    } catch (error) {
      console.error('❌ Error starting camera:', error);
      
      let userErrorMessage = 'Failed to start camera';
      
      if (error instanceof Error) {
        if (error.message.includes('permission denied')) {
          userErrorMessage = 'Camera permission denied. Please allow camera access.';
        } else if (error.message.includes('timeout')) {
          userErrorMessage = 'Camera startup timed out. Please try again.';
        } else {
          userErrorMessage = error.message;
        }
      }
      
      setCameraError(userErrorMessage);
      
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-red-500 p-4 text-center"><div class="text-lg mb-2">❌ Camera Error</div><div class="text-sm">${userErrorMessage}</div></div>`;
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
    if (!isMobile || !mobileWebcam.current) return;
    
    console.log('📤 Starting continuous screen sharing...');
    
    if (screenShareIntervalRef.current) {
      clearInterval(screenShareIntervalRef.current);
    }
    
    screenShareIntervalRef.current = setInterval(() => {
      try {
        const videoElement = mobileWebcam.current?.getVideo();
        
        if (videoElement && videoElement.readyState === 4) {
          // Use capturePreviewFrame for higher quality screen sharing
          // This captures the ZOOMED portion that the user actually sees (4:3 aspect ratio)
          const canvas = mobileWebcam.current?.capturePreviewFrame(640, 480);
          
          if (canvas) {
            const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
            screenShareService.sharePreviewImage(imageDataUrl);
            setMobilePreviewImage(imageDataUrl);
          }
        }
      } catch (error) {
        console.error('❌ Error capturing frame:', error);
      }
    }, 100);
    
    console.log('✅ Screen sharing interval started (with zoom-aware capture)');
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
      webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400"><div class="text-lg mb-2">📷</div><div class="text-sm">Camera stopped</div></div>';
    }
    
    console.log('✅ Camera stopped successfully');
  }, [stopScreenSharing]);

  useEffect(() => {
    console.log('🔄 Mode changed to:', activeMode);
    
    if (activeMode === 'webcam' && isMobile) {
      startCamera().catch((error) => {
        console.error('❌ Failed to start camera:', error);
      });
    } else if (activeMode === 'webcam' && !isMobile) {
      if (webcamRef.current) {
        webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400 p-4 text-center"><div class="text-lg mb-2">💻</div><div class="text-sm">Desktop Mode</div><div class="text-xs">Camera is only available on mobile</div></div>';
      }
    } else {
      stopCamera();
      if (webcamRef.current) {
        webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400"><div class="text-lg mb-2">📁</div><div class="text-sm">Upload Mode</div></div>';
      }
    }

    return () => {
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
      setIsUsingRemoteModel(false);
    }
  }, [predictions, isUsingRemoteModel]);

  // Camera display with tap-to-focus
  const renderCameraDisplay = () => (
    <div className="relative border-2 border-dashed border-gray-300 rounded-lg min-h-[250px] flex items-center justify-center bg-gray-50">
      {/* Video feed container with tap-to-focus */}
      <div 
        ref={webcamRef} 
        className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden cursor-pointer`}
        onClick={handleCameraClick}
        onTouchStart={handleCameraClick}
        style={{
          WebkitTapHighlightColor: 'transparent'
        }}
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
            border: '2px solid #FFD700',
            borderRadius: '50%',
            transform: 'translate(-50%, -50%)',
            animation: 'focusPulse 0.6s ease-out',
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
                opacity: 0.9;
              }
            }
          `}</style>
        </div>
      )}
      
      {/* Prediction indicator */}
      {activeMode === 'webcam' && isPredicting && (
        <div className="absolute top-2 left-2 bg-blue-500 text-white px-3 py-1.5 rounded text-xs font-medium z-20 flex items-center gap-2">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          Analyzing...
        </div>
      )}
      
      {/* Loading state */}
      {isCameraStarting && (
        <div className="absolute inset-[2px] flex items-center justify-center bg-gray-50 bg-opacity-75 z-20 rounded-lg">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      )}
      
      {/* Error state */}
      {cameraError && (
        <div className="absolute inset-[2px] flex items-center justify-center bg-red-50 z-20 rounded-lg">
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