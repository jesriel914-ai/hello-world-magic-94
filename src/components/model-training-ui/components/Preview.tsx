//filepath: src\components\model-training-ui\components\Preview.tsx
import { predictFromCanvas, forceMemoryCleanup } from '../utils/modelPrediction';
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
import { FileImage, Brain, X, Loader2, Upload, Camera, FolderOpen, Cloud, ChevronDown, List } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { TrainedModel } from '../../ModelTraining';
import type { PredictionResult } from '../../ModelTraining';
import type { CustomModel } from '../../ModelTraining';
import { MobileWebcam } from '../services/mobileWebcam';
import { toast } from '@/hooks/use-toast';

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
  isPredicting: boolean;
  onToggleModels: () => void;
  onShowMorePredictions: () => void;
  onChangeModel: () => void;
  onCloudModelSelect?: () => void;
  onLocalModelSelect?: () => void;
  onHandlePreviewFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

export const Preview: React.FC<PreviewProps> = ({
  onToggleModels,
  showModels,
  isMobile = false,
  model,
  predictions,
  visiblePredictions,
  previewImage,
  isWebcamActive,
  webcam,
  trainedModels,
  isLoadingModels,
  isPredicting,
  onShowMorePredictions,
  onChangeModel,
  onCloudModelSelect,
  onLocalModelSelect,
  onHandlePreviewFileUpload
}) => {
  const [activeMode, setActiveMode] = useState<'webcam' | 'upload'>('upload');
  const [modelTrainedAt, setModelTrainedAt] = useState<Date | null>(null);
  const [localPreviewImage, setLocalPreviewImage] = useState<string | null>(null);
  
  const webcamRef = useRef<HTMLDivElement>(null);
  const mobileWebcam = useRef<MobileWebcam | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [localPredictions, setLocalPredictions] = useState<PredictionResult[]>([]);
  const [isVideoPaused, setIsVideoPaused] = useState(false);
  const cameraPredictionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isCameraReady, setIsCameraReady] = useState(false);

  // Use local predictions if available, otherwise use props predictions
  const displayPredictions = localPredictions.length > 0 ? localPredictions : predictions;

// Update handleWebcamClick to track pause state:
const handleWebcamClick = () => {
  if (!mobileWebcam.current) return;
  
  const videoElement = mobileWebcam.current.getVideo();
  if (!videoElement) return;
  
  if (videoElement.paused) {
    videoElement.play();
    setIsVideoPaused(false);
    console.log('Camera resumed');
  } else {
    videoElement.pause();
    setIsVideoPaused(true);
    console.log('Camera paused');
  }
};

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file || !file.type.startsWith('image/')) continue;
      
      try {
        console.log(`📱 Processing image ${i + 1}/${files.length}: ${file.name}`);
        
        const imageUrl = URL.createObjectURL(file);
        const img = new Image();
        
        img.onload = async () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            
            if (ctx) {
              ctx.drawImage(img, 0, 0, 224, 224);
              
              if (i === 0) {
                if (localPreviewImage) {
                  URL.revokeObjectURL(localPreviewImage);
                }
                setLocalPreviewImage(imageUrl);
              } else {
                URL.revokeObjectURL(imageUrl);
              }
              
              if (model) {
                console.log('🎯 Making prediction...');
                const predictions = await model.predict(canvas, false);
                const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
                console.log('✅ Prediction completed:', sortedPredictions);
                setLocalPredictions(sortedPredictions);
              }
            }
          } catch (error) {
            console.error('❌ Error processing image:', error);
            if (i !== 0) {
              URL.revokeObjectURL(imageUrl);
            }
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
  }, [localPreviewImage, model]);
// Then update your startCamera function to set this state when ready:
const startCamera = useCallback(async () => {
  console.log('📷 startCamera called - isMobile:', isMobile);
  if (!isMobile || !webcamRef.current) {
    console.log('❌ startCamera blocked');
    return;
  }
  
  console.log('🚀 Starting camera process...');
  setIsCameraStarting(true);
  setCameraError(null);
  setIsCameraReady(false); // Reset camera ready state
  
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
    
    mobileWebcam.current = new MobileWebcam({
      width: 300,
      height: 300,
      facingMode: 'environment',
      timeout: 20000,
      zoom: 2.0
    });
    
    console.log('📷 Initializing camera with 2x zoom...');
    const videoElement = await mobileWebcam.current.start();
    
    webcamRef.current.innerHTML = '';
    webcamRef.current.appendChild(videoElement);
    
    setTimeout(() => {
      if (webcamRef.current && mobileWebcam.current) {
        const outerContainer = webcamRef.current.parentElement;
        if (outerContainer) {
          const rect = outerContainer.getBoundingClientRect();
          console.log('📱 Mobile preview container dimensions:', {
            width: rect.width,
            height: rect.height,
            aspectRatio: (rect.width / rect.height).toFixed(2)
          });
          
          mobileWebcam.current.setPreviewDimensions(rect.width, rect.height);
        }
      }
    }, 1000);
    
    // CRITICAL: Set camera ready state AFTER camera is fully started
setIsCameraReady(true);
setIsVideoPaused(false); // Camera starts playing
console.log('✅ Camera started successfully with zoom and marked as ready');
    
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
    setIsCameraReady(false);
    
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

const stopCamera = useCallback(() => {
  console.log('🛑 Stopping camera...');
  
  if (mobileWebcam.current) {
    mobileWebcam.current.stop();
    mobileWebcam.current = null;
  }
  
  setIsCameraReady(false);
  setIsVideoPaused(false); // Reset pause state
  
  if (webcamRef.current) {
    webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400"><div class="text-lg mb-2">📷</div><div class="text-sm">Camera stopped</div></div>';
  }
  
  console.log('✅ Camera stopped successfully');
}, []);

useEffect(() => {
  if (!isMobile || activeMode !== 'webcam' || !isCameraReady || !mobileWebcam.current || !model) {
    if (cameraPredictionIntervalRef.current) {
      clearInterval(cameraPredictionIntervalRef.current);
      cameraPredictionIntervalRef.current = null;
    }
    setLocalPredictions([]);
    console.log('🛑 Camera prediction loop stopped');
    return;
  }
  
  let isRunning = false;
  let consecutiveErrors = 0;
  const MAX_ERRORS = 3;
  let successCount = 0;
  let lastCleanupTime = Date.now();
  const CLEANUP_INTERVAL = 5000; // Cleanup every 5 seconds
  
  const runCameraPrediction = async () => {
    if (isRunning) return;
    if (!mobileWebcam.current || !model) return;
    
    isRunning = true;
    
    try {
      // Check if it's time for periodic memory cleanup
      const now = Date.now();
      if (now - lastCleanupTime > CLEANUP_INTERVAL) {
        forceMemoryCleanup();
        lastCleanupTime = now;
      }
      
      const videoElement = mobileWebcam.current.getVideo();
      if (!videoElement) {
        throw new Error('Video element is null');
      }
      
      if (videoElement.paused || videoElement.ended) {
        throw new Error('Video is paused or ended');
      }
      
      if (videoElement.readyState < 2) {
        throw new Error(`Video not ready (readyState: ${videoElement.readyState})`);
      }
      
      const canvas = mobileWebcam.current.captureFrame();
      if (!canvas) {
        throw new Error('captureFrame() returned null');
      }
      
      if (canvas.width !== 224 || canvas.height !== 224) {
        throw new Error(`Invalid canvas size: ${canvas.width}x${canvas.height}`);
      }
      
      // Prediction now wrapped in tf.tidy() automatically via predictFromCanvas
      const predictions = await predictFromCanvas(model, canvas, false);
      
      if (!predictions || predictions.length === 0) {
        throw new Error('Model returned empty predictions');
      }
      
      consecutiveErrors = 0;
      successCount++;
      
      const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
      
      // Log memory stats every 30 predictions (~9 seconds)
      if (successCount % 30 === 0) {
        const memory = tf.memory();
        console.log(`📊 Memory stats after ${successCount} predictions:`, {
          numTensors: memory.numTensors,
          numBytes: (memory.numBytes / 1024 / 1024).toFixed(2) + ' MB'
        });
      }
      
      setLocalPredictions(sortedPredictions);
      
    } catch (error) {
      consecutiveErrors++;
      console.error(`❌ Camera prediction error #${consecutiveErrors}:`, error);
      
      if (consecutiveErrors >= MAX_ERRORS) {
        console.error('🛑 Too many consecutive errors, stopping predictions');
        
        if (cameraPredictionIntervalRef.current) {
          clearInterval(cameraPredictionIntervalRef.current);
          cameraPredictionIntervalRef.current = null;
        }
        
        toast({
          title: 'Camera Prediction Failed',
          description: 'Unable to process camera feed. Please try restarting the camera.',
          variant: 'destructive',
        });
        
        setLocalPredictions([]);
      }
    } finally {
      isRunning = false;
    }
  };
  
  console.log('🚀 Starting camera prediction loop with memory management');
  runCameraPrediction();
  
  cameraPredictionIntervalRef.current = setInterval(runCameraPrediction, 500);
  
  return () => {
    if (cameraPredictionIntervalRef.current) {
      console.log('🛑 Stopping camera prediction loop');
      clearInterval(cameraPredictionIntervalRef.current);
      cameraPredictionIntervalRef.current = null;
    }
    
    // Final cleanup when stopping
    console.log('🧹 Final memory cleanup');
    forceMemoryCleanup();
  };
}, [isMobile, activeMode, isCameraReady, model, toast]);

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

  const renderCameraDisplay = () => (
    <div className="relative border-2 border-dashed border-gray-300 rounded-lg aspect-video flex items-center justify-center bg-gray-50">
      <div 
        ref={webcamRef} 
        className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden cursor-pointer`}
        onClick={handleWebcamClick}
      />

      {activeMode === 'webcam' && displayPredictions.length > 0 && (
        <div className="absolute top-2 left-2 bg-blue-500 text-white px-3 py-1.5 rounded text-xs font-medium z-20 flex items-center gap-2">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          Analyzing...
        </div>
      )}
      
      {isCameraStarting && (
        <div className="absolute inset-[2px] flex items-center justify-center bg-gray-50 bg-opacity-75 z-20 rounded-lg">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      )}
      
      {cameraError && (
        <div className="absolute inset-[2px] flex items-center justify-center bg-red-50 z-20 rounded-lg">
          <div className="text-center p-4">
            <div className="text-red-600 font-medium mb-2">Camera Error</div>
            <div className="text-red-500 text-sm">{cameraError}</div>
          </div>
        </div>
      )}
      
      {activeMode === 'upload' && (
        (localPreviewImage || previewImage) ? (
          <div className="absolute inset-[2px] flex items-center justify-center z-30 rounded-lg overflow-hidden">
            <img 
              src={localPreviewImage || previewImage} 
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
        
        {showModels ? (
          <>
            {model ? (
              <div className="space-y-4">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="w-fit">
                      <Cloud className="mr-2 h-4 w-4" />
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
                      <Cloud className="mr-2 h-4 w-4" />
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
                  <DropdownMenuItem onClick={() => setActiveMode('webcam')}>
                    <Camera className="mr-2 h-4 w-4" />
                    <span>Webcam</span>
                    {activeMode === 'webcam' && (
                      <div className="w-2 h-2 bg-green-500 rounded-full ml-auto"></div>
                    )}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setActiveMode('upload')}>
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
  disabled={!mobileWebcam.current}
>
  <div className="flex items-center gap-2">
    <Camera className="w-5 h-5" />
    <span>{isVideoPaused ? 'Resume' : 'Pause'}</span>
    {activeMode === 'webcam' && !isVideoPaused && (
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

            {/* TEMPORARY DEBUG BUTTON */}
{activeMode === 'webcam' && mobileWebcam.current && (
  <Button 
    onClick={() => {
      if (mobileWebcam.current) {
        const canvas = mobileWebcam.current.captureFrame();
        if (canvas) {
          const dataUrl = canvas.toDataURL();
          console.log('✅ Test capture successful:', dataUrl.substring(0, 50) + '...');
          const a = document.createElement('a');
          a.href = dataUrl;
          a.download = 'test-capture.png';
          a.click();
        } else {
          console.error('❌ captureFrame returned null');
        }
      }
    }}
    variant="outline"
    size="sm"
    className="w-full"
  >
    🧪 Test Frame Capture
  </Button>
)}

<div className="space-y-3"></div>
            
            <div className="space-y-3">
              <h4 className="font-medium">Prediction Results</h4>
              <div className="space-y-2">
                {displayPredictions.length > 0 ? (
                  <>
                    {displayPredictions[0] && (
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
                    
                    <div className="space-y-2">
                      {displayPredictions.slice(1, 3).map((pred, index) => {
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
                    
                    {displayPredictions.length > 3 && (
                      <div className="text-center text-xs text-gray-500 pt-2">
                        Showing top 3 of {displayPredictions.length} predictions
                      </div>
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
            <Button 
              variant="ghost" 
              size="sm"
              onClick={onToggleModels}
              className="h-8 w-8 p-0"
              title={showModels ? "Back to Preview" : "Models"}
            >
              {showModels ? <X className="w-4 h-4" /> : <List className="w-4 h-4" />}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col space-y-4">
          {showModels ? (
            <>
              {model ? (
                <div className="space-y-4">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="default" className="w-fit">
                        <Cloud className="mr-2 h-4 w-4" />
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
                        <Cloud className="mr-2 h-4 w-4" />
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
                    <DropdownMenuItem onClick={() => setActiveMode('webcam')}>
                      <Camera className="mr-2 h-4 w-4" />
                      <span>Webcam</span>
                      {activeMode === 'webcam' && (
                        <div className="w-2 h-2 bg-green-500 rounded-full ml-auto"></div>
                      )}
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setActiveMode('upload')}>
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
  disabled={!mobileWebcam.current}
>
  <div className="flex items-center gap-2">
    <Camera className="w-5 h-5" />
    <span>{isVideoPaused ? 'Resume' : 'Pause'}</span>
    {activeMode === 'webcam' && !isVideoPaused && (
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
                  {displayPredictions.length > 0 ? (
                    <>
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
                      
                      <div className="space-y-2">
                        {displayPredictions.slice(1, 3).map((pred, index) => {
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
                      
                      {displayPredictions.length > 3 && (
                        <div className="text-center text-xs text-gray-500 pt-2">
                          Showing top 3 of {displayPredictions.length} predictions
                        </div>
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