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
import { FileImage, X, Loader2, Upload, Camera, FolderOpen, Cloud, ChevronDown, List, CheckCircle, XCircle, User } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import { MobileWebcam } from '@/components/model-training-ui/services/mobileWebcam';
import { toast } from '@/hooks/use-toast';
import { siameseModelService } from '../lib/AIModelService';

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

const Verification: React.FC = () => {
  const isMobile = useMobileDetection();
  const [activeMode, setActiveMode] = useState<'webcam' | 'upload'>('upload');
  const [localPreviewImage, setLocalPreviewImage] = useState<string | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  
  const webcamRef = useRef<HTMLDivElement>(null);
  const mobileWebcam = useRef<MobileWebcam | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isVideoPaused, setIsVideoPaused] = useState(false);
  const cameraPredictionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isCameraReady, setIsCameraReady] = useState(false);

  // Verification state
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    isVerified: boolean;
    confidence: number;
    studentId?: string;
  } | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<string>('');

  // Mock student data for dropdown
  const mockStudents = [
    { id: '1', name: 'John Doe - BSIT 2024-1A' },
    { id: '2', name: 'Jane Smith - BSCS 2024-1B' },
    { id: '3', name: 'Mike Johnson - BSIT 2024-2A' }
  ];

  // Verification handler - calls Python verification pipeline
  const handleVerifySignature = async (signatureData: any) => {
    if (!selectedStudent) {
      alert('Please select a student first!');
      return;
    }
    
    setIsVerifying(true);
    setVerificationResult(null);
    
    try {
      console.log('Starting signature verification...');
      console.log('Student ID:', selectedStudent);
      console.log('Signature data:', signatureData);
      
      // Call the verification service
      const result = await siameseModelService.verifySignature(selectedStudent, signatureData);
      
      console.log('Verification result:', result);
      
      // Convert to our expected format
      const verificationResult = {
        isVerified: result.is_verified,
        confidence: result.confidence,
        studentId: selectedStudent
      };
      
      setVerificationResult(verificationResult);
      
    } catch (error) {
      console.error('Verification failed:', error);
      alert(`Verification failed: ${error}`);
    } finally {
      setIsVerifying(false);
    }
  };

  // Handle preview file upload
  const handlePreviewFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const imageUrl = URL.createObjectURL(file);
    setLocalPreviewImage(imageUrl);
    event.target.value = '';
  };

  // Webcam functions
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

  const startCamera = useCallback(async () => {
    console.log('📷 startCamera called - isMobile:', isMobile);
    if (!isMobile || !webcamRef.current) {
      console.log('❌ startCamera blocked');
      return;
    }
    
    console.log('🚀 Starting camera process...');
    setIsCameraStarting(true);
    setCameraError(null);
    setIsCameraReady(false);
    
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
      
      setIsCameraReady(true);
      setIsVideoPaused(false);
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
    setIsVideoPaused(false);
    
    if (webcamRef.current) {
      webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400"><div class="text-lg mb-2">📷</div><div class="text-sm">Camera stopped</div></div>';
    }
    
    console.log('✅ Camera stopped successfully');
  }, []);

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
          Camera Active
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
            <FileImage className="w-6 h-6" />
            <span className="text-base">Verification</span>
          </div>
        </div>
        
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
                onChange={handlePreviewFileUpload}
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

        {/* Student Selection and Verify Button Row */}
        <div className="flex items-center gap-3 pt-3">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="flex-1 justify-start">
                <User className="w-4 h-4 mr-2" />
                {selectedStudent ? mockStudents.find(s => s.id === selectedStudent)?.name : 'Select Class'}
                <ChevronDown className="w-4 h-4 ml-auto" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-full">
              {mockStudents.map((student) => (
                <DropdownMenuItem
                  key={student.id}
                  onClick={() => setSelectedStudent(student.id)}
                  className="flex items-center gap-2"
                >
                  <User className="w-4 h-4" />
                  {student.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          
          <Button
            onClick={() => handleVerifySignature({ image: localPreviewImage || previewImage })}
            disabled={isVerifying || !selectedStudent}
            className="px-6"
          >
            {isVerifying ? 'Verifying...' : 'Verify'}
          </Button>
        </div>

        {/* Verification Result Display */}
        {verificationResult && (
          <div className="pt-3">
            <div className={`p-3 rounded-lg border ${
              verificationResult.isVerified 
                ? 'bg-green-50 border-green-200' 
                : 'bg-red-50 border-red-200'
            }`}>
              <div className="flex items-center gap-2">
                {verificationResult.isVerified ? (
                  <CheckCircle className="w-5 h-5 text-green-600" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-600" />
                )}
                <div>
                  <p className={`font-medium ${
                    verificationResult.isVerified ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {verificationResult.isVerified ? 'Matched' : 'Not Matched'}
                  </p>
                  <p className={`text-sm ${
                    verificationResult.isVerified ? 'text-green-600' : 'text-red-600'
                  }`}>
                    Confidence: {(verificationResult.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-3"></div>
      </>
    ) : (
      <Card className="h-[605px] flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-base">
            <div className="flex items-center gap-2">
              <FileImage className="w-5 h-5" />
              <span>Verification</span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col space-y-4">
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
                    id="desktop-file-upload"
                    onChange={handlePreviewFileUpload}
                  />
                  <label htmlFor="desktop-file-upload" className="cursor-pointer w-auto">
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

            {/* Student Selection and Verify Button Row */}
            <div className="flex items-center gap-3 pt-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="flex-1 justify-start">
                    <User className="w-4 h-4 mr-2" />
                    {selectedStudent ? mockStudents.find(s => s.id === selectedStudent)?.name : 'Select Class'}
                    <ChevronDown className="w-4 h-4 ml-auto" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-full">
                  {mockStudents.map((student) => (
                    <DropdownMenuItem
                      key={student.id}
                      onClick={() => setSelectedStudent(student.id)}
                      className="flex items-center gap-2"
                    >
                      <User className="w-4 h-4" />
                      {student.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              
              <Button
                onClick={() => handleVerifySignature({ image: localPreviewImage || previewImage })}
                disabled={isVerifying || !selectedStudent}
                className="px-6"
              >
                {isVerifying ? 'Verifying...' : 'Verify'}
              </Button>
            </div>

            {/* Verification Result Display */}
            {verificationResult && (
              <div className="pt-3">
                <div className={`p-3 rounded-lg border ${
                  verificationResult.isVerified 
                    ? 'bg-green-50 border-green-200' 
                    : 'bg-red-50 border-red-200'
                }`}>
                  <div className="flex items-center gap-2">
                    {verificationResult.isVerified ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                    <div>
                      <p className={`font-medium ${
                        verificationResult.isVerified ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {verificationResult.isVerified ? 'Matched' : 'Not Matched'}
                      </p>
                      <p className={`text-sm ${
                        verificationResult.isVerified ? 'text-green-600' : 'text-red-600'
                      }`}>
                        Confidence: {(verificationResult.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )
  );
};

export default Verification;