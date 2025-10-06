//filepath: ai-model-siamese/components/Verification.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { FileImage, X, Loader2, Upload, Camera, FolderOpen, Cloud, ChevronDown, List, Shield, CheckCircle, XCircle } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import { MobileWebcam } from '@/components/model-training-ui/services/mobileWebcam';
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

interface VerificationProps {
  isMobile: boolean;
  showModels: boolean;
  isLoadingModels: boolean;
  previewImage: string | null;
  isWebcamActive: boolean;
  onToggleModels: () => void;
  onChangeModel: () => void;
  onCloudModelSelect?: () => void;
  onLocalModelSelect?: () => void;
  onHandlePreviewFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onVerifySignature?: (signatureData: any) => void;
}

export const Verification: React.FC<VerificationProps> = ({
  onToggleModels,
  showModels,
  isMobile = false,
  previewImage,
  isWebcamActive,
  isLoadingModels,
  onChangeModel,
  onCloudModelSelect,
  onLocalModelSelect,
  onHandlePreviewFileUpload,
  onVerifySignature
}) => {
  const [activeMode, setActiveMode] = useState<'webcam' | 'upload'>('upload');
  const [localPreviewImage, setLocalPreviewImage] = useState<string | null>(null);
  
  const webcamRef = useRef<HTMLDivElement>(null);
  const mobileWebcam = useRef<MobileWebcam | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isVideoPaused, setIsVideoPaused] = useState(false);
  const cameraPredictionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Verification state
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    isVerified: boolean;
    confidence: number;
    studentId?: string;
  } | null>(null);

  const [isCameraReady, setIsCameraReady] = useState(false);

  // Mock verification handler
  const handleVerifySignature = async (signatureData: any) => {
    if (!onVerifySignature) return;
    
    setIsVerifying(true);
    setVerificationResult(null);
    
    try {
      // Simulate verification delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock verification result - always show some result
      const mockResult = {
        isVerified: Math.random() > 0.3, // 70% chance of being verified
        confidence: Math.random() * 0.4 + 0.6, // 60-100% confidence
        studentId: 'mock-student'
      };
      
      setVerificationResult(mockResult);
      onVerifySignature(signatureData);
    } catch (error) {
      console.error('Verification failed:', error);
      setVerificationResult({
        isVerified: false,
        confidence: 0.2, // Show 20% confidence for failed verification
        studentId: 'mock-student'
      });
    } finally {
      setIsVerifying(false);
    }
  };

// Update handleWebcamClick to track pause state:
const handleWebcamClick = () => {
  if (!mobileWebcam.current) return;
  
  if (isVideoPaused) {
    mobileWebcam.current.resume();
    setIsVideoPaused(false);
  } else {
    mobileWebcam.current.pause();
    setIsVideoPaused(true);
  }
};

// Handle file upload for verification
const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  try {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const imageUrl = URL.createObjectURL(file);
      
      if (i === 0) {
        if (localPreviewImage) {
          URL.revokeObjectURL(localPreviewImage);
        }
        setLocalPreviewImage(imageUrl);
      } else {
        URL.revokeObjectURL(imageUrl);
      }
    }
  } catch (error) {
    console.error(`Error processing file: ${file.name}`, error);
  }
}, [localPreviewImage]);

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
  
  try {
    if (!mobileWebcam.current) {
      mobileWebcam.current = new MobileWebcam(webcamRef.current);
    }
    
    await mobileWebcam.current.start();
    console.log('✅ Camera started successfully');
    setIsCameraReady(true);
    setIsVideoPaused(false);
  } catch (error) {
    console.error('❌ Camera start failed:', error);
    setCameraError(error instanceof Error ? error.message : 'Failed to start camera');
  } finally {
    setIsCameraStarting(false);
  }
}, [isMobile]);

const stopCamera = useCallback(() => {
  if (mobileWebcam.current) {
    mobileWebcam.current.stop();
    mobileWebcam.current = null;
    setIsCameraReady(false);
    setIsVideoPaused(false);
  }
}, []);

// Cleanup on unmount
useEffect(() => {
  return () => {
    if (cameraPredictionIntervalRef.current) {
      clearInterval(cameraPredictionIntervalRef.current);
    }
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
    }
    if (localPreviewImage) {
      URL.revokeObjectURL(localPreviewImage);
    }
  };
}, [localPreviewImage]);

// Auto-start camera when switching to webcam mode
useEffect(() => {
  if (activeMode === 'webcam' && isMobile && !isCameraReady && !isCameraStarting) {
    startCamera();
  } else if (activeMode === 'upload' && isCameraReady) {
    stopCamera();
  }
}, [activeMode, isMobile, isCameraReady, isCameraStarting, startCamera, stopCamera]);

const renderCameraDisplay = () => (
  <div className="relative border-2 border-dashed border-gray-300 rounded-lg aspect-video flex items-center justify-center bg-gray-50">
    <div 
      ref={webcamRef} 
      className={`absolute inset-[2px] flex items-center justify-center ${activeMode === 'webcam' ? '' : 'hidden'} z-0 rounded-lg overflow-hidden cursor-pointer`}
      onClick={handleWebcamClick}
    />

    {activeMode === 'webcam' && (
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
              <Shield className="w-6 h-6" />
              <span className="text-base">Current Model</span>
            </>
          ) : (
            <>
              <FileImage className="w-6 h-6" />
              <span className="text-base">Signature Verification</span>
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
          <div className="space-y-4">
            <div className="text-center py-8">
              <Shield className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <h3 className="text-base font-medium text-gray-500 mb-2">No Siamese model loaded</h3>
              <p className="text-gray-400 text-sm mb-4">Select a model to start verification</p>
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
          </div>
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

          {/* Verify Button and Match Display */}
          {(localPreviewImage || previewImage) && (
            <div className="flex items-center justify-between pt-3">
              <div className="flex items-center gap-4">
                <div className="text-sm">
                  <span className="text-gray-600">No Match:</span>
                  <span className="ml-1 font-medium">
                    {verificationResult 
                      ? `${((1 - verificationResult.confidence) * 100).toFixed(0)}%`
                      : '00%'
                    }
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-600">Matched:</span>
                  <span className="ml-1 font-medium">
                    {verificationResult 
                      ? `${(verificationResult.confidence * 100).toFixed(0)}%`
                      : '00%'
                    }
                  </span>
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleVerifySignature({ image: localPreviewImage || previewImage })}
                disabled={isVerifying}
              >
                {isVerifying ? 'Verifying...' : 'Verify'}
              </Button>
            </div>
          )}

<div className="space-y-3"></div>
        </>
      )}
    </>
  ) : (
    <Card className="h-[605px] w-full lg:col-span-1 flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {showModels ? (
              <>
                <Shield className="w-5 h-5" />
                <span>Current Model</span>
              </>
            ) : (
              <>
                <FileImage className="w-5 h-5" />
                <span>Signature Verification</span>
              </>
            )}
          </div>
          <Button 
            variant="ghost" 
            size="default"
            onClick={onToggleModels}
            className="h-8 w-8 p-0"
            title={showModels ? "Back to Preview" : "Models"}
          >
            {showModels ? <X className="w-4 h-4" /> : <List className="w-4 h-4" />}
          </Button>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 overflow-hidden flex flex-col">
        {showModels ? (
          <div className="space-y-4">
            <div className="text-center py-8">
              <Shield className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <h3 className="text-base font-medium text-gray-500 mb-2">No Siamese model loaded</h3>
              <p className="text-gray-400 text-sm mb-4">Select a model to start verification</p>
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
          </div>
        ) : (
          <>
            <div className="flex flex-row gap-3 mb-4">
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

            {/* Verify Button and Match Display */}
            {(localPreviewImage || previewImage) && (
              <div className="flex items-center justify-between pt-3">
                <div className="flex items-center gap-4">
                  <div className="text-sm">
                    <span className="text-gray-600">No Match:</span>
                    <span className="ml-1 font-medium">
                      {verificationResult 
                        ? `${((1 - verificationResult.confidence) * 100).toFixed(0)}%`
                        : '00%'
                      }
                    </span>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-600">Matched:</span>
                    <span className="ml-1 font-medium">
                      {verificationResult 
                        ? `${(verificationResult.confidence * 100).toFixed(0)}%`
                        : '00%'
                      }
                    </span>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleVerifySignature({ image: localPreviewImage || previewImage })}
                  disabled={isVerifying}
                >
                  {isVerifying ? 'Verifying...' : 'Verify'}
                </Button>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
);
};

export default Verification;