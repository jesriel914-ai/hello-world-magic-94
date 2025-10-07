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
import { FileImage, X, Loader2, Upload, Camera, ChevronDown, CheckCircle, XCircle, User } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import { MobileWebcam } from '@/components/model-training-ui/services/mobileWebcam';
import { siameseModelService } from '../lib/SiameseAIModelService';

const formatStudentDisplay = (student: any): string => {
  return `${student.student_id} - ${student.firstname} ${student.surname}`;
};

interface VerificationProps {}


const Verification: React.FC<VerificationProps> = () => {
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

  const [isCameraReady, setIsCameraReady] = useState(false);

// Verification state
const [isVerifying, setIsVerifying] = useState(false);
const [verificationResult, setVerificationResult] = useState<{
  isVerified: boolean;
  confidence: number;
  studentId?: string;
} | null>(null);
const [selectedStudent, setSelectedStudent] = useState<string>('');
const [trainedStudents, setTrainedStudents] = useState<Array<{
  student_id: string;
  metadata: any;
}>>([]);
const [isLoadingStudents, setIsLoadingStudents] = useState(true);

  // Capture frame from webcam
  const captureWebcamFrame = useCallback((): string | null => {
    if (!mobileWebcam.current) return null;
    
    const videoElement = mobileWebcam.current.getVideo();
    if (!videoElement || videoElement.paused) return null;
    
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return null;
      
      ctx.drawImage(videoElement, 0, 0, 224, 224);
      return canvas.toDataURL('image/jpeg', 0.9);
    } catch (error) {
      console.error('Error capturing frame:', error);
      return null;
    }
  }, []);

  // Load trained students on mount
useEffect(() => {
  const loadTrainedStudents = async () => {
    setIsLoadingStudents(true);
    try {
      const students = await siameseModelService.listTrainedStudents();
      setTrainedStudents(students);
      console.log('Loaded trained students:', students);
    } catch (error) {
      console.error('Failed to load trained students:', error);
    } finally {
      setIsLoadingStudents(false);
    }
  };
  
  loadTrainedStudents();
}, []);

  // Verification handler - calls Python verification pipeline
  const handleVerifySignature = async () => {
    if (!selectedStudent) {
      alert('Please select a student first!');
      return;
    }
    
    // Determine which image to use
    let signatureImage: string | null = null;
    
    if (activeMode === 'webcam') {
      signatureImage = captureWebcamFrame();
      if (!signatureImage) {
        alert('Failed to capture image from camera. Please try again.');
        return;
      }
      // Set as preview
      setLocalPreviewImage(signatureImage);
    } else {
      signatureImage = localPreviewImage || previewImage;
    }
    
    if (!signatureImage) {
      alert('Please upload an image or capture from camera first!');
      return;
    }
    
    setIsVerifying(true);
    setVerificationResult(null);
    
    try {
      console.log('Starting signature verification...');
      console.log('Student ID:', selectedStudent);
      
      // Call the verification service
      const result = await siameseModelService.verifySignature(selectedStudent, signatureImage);
      
      console.log('Verification result:', result);
      
      // Set the result
      setVerificationResult({
        isVerified: result.is_verified,
        confidence: result.confidence,
        studentId: selectedStudent
      });
      
    } catch (error) {
      console.error('Verification failed:', error);
      alert(`Verification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsVerifying(false);
    }
  };

  // Handle preview file upload
  const handlePreviewFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setLocalPreviewImage(result);
      setVerificationResult(null); // Clear previous result
    };
    reader.readAsDataURL(file);
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
    } else {
      videoElement.pause();
      setIsVideoPaused(true);
    }
  };

  const startCamera = useCallback(async () => {
    if (!isMobile || !webcamRef.current) return;
    
    setIsCameraStarting(true);
    setCameraError(null);
    setIsCameraReady(false);
    
    try {
      const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
      
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
      
      const videoElement = await mobileWebcam.current.start();
      
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      setTimeout(() => {
        if (webcamRef.current && mobileWebcam.current) {
          const outerContainer = webcamRef.current.parentElement;
          if (outerContainer) {
            const rect = outerContainer.getBoundingClientRect();
            mobileWebcam.current.setPreviewDimensions(rect.width, rect.height);
          }
        }
      }, 1000);
      
      setIsCameraReady(true);
      setIsVideoPaused(false);
      
    } catch (error) {
      console.error('Error starting camera:', error);
      
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
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
      mobileWebcam.current = null;
    }
    
    setIsCameraReady(false);
    setIsVideoPaused(false);
    
    if (webcamRef.current) {
      webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-400"><div class="text-lg mb-2">📷</div><div class="text-sm">Camera stopped</div></div>';
    }
  }, []);

  useEffect(() => {
    if (activeMode === 'webcam' && isMobile) {
      startCamera().catch((error) => {
        console.error('Failed to start camera:', error);
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
        <div className="absolute inset-[2px] flex items-center justify-center z-10">
          {localPreviewImage || previewImage ? (
            <div className="relative w-full h-full">
              <img 
                src={localPreviewImage || previewImage || ''} 
                alt="Preview" 
                className="w-full h-full object-contain filter grayscale"
              />
              <button
                onClick={() => {
                  setLocalPreviewImage(null);
                  setPreviewImage(null);
                  setVerificationResult(null);
                }}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <label className="cursor-pointer flex flex-col items-center gap-2">
              <FileImage className="w-12 h-12 text-gray-400" />
              <span className="text-sm text-gray-500">Upload signature</span>
              <input 
                type="file" 
                accept="image/*" 
                onChange={handlePreviewFileUpload}
                className="hidden"
              />
            </label>
          )}
        </div>
      )}
    </div>
  );

  return (
    <Card className="h-[605px] w-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <User className="w-5 h-5" />
            Verification
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto overlay-scrollbar-container space-y-4">
          {/* Mode Selection */}
          <div className="flex gap-2">
            <Button
              variant={activeMode === 'upload' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setActiveMode('upload')}
              className="flex-1"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </Button>
            <Button
              variant={activeMode === 'webcam' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setActiveMode('webcam')}
              className="flex-1"
            >
              <Camera className="w-4 h-4 mr-2" />
              Camera
            </Button>
          </div>

          {/* Camera/Upload Display */}
          {renderCameraDisplay()}

          {/* Student Selection */}
<div className="space-y-2">
  <label className="text-sm font-medium">Select Student</label>
  <DropdownMenu>
    <DropdownMenuTrigger asChild>
      <Button variant="outline" className="w-full justify-between" disabled={isLoadingStudents}>
        <span>
          {isLoadingStudents 
            ? 'Loading students...'
            : selectedStudent 
              ? selectedStudent
              : 'Choose a student...'}
        </span>
        <ChevronDown className="w-4 h-4 ml-2" />
      </Button>
    </DropdownMenuTrigger>
    <DropdownMenuContent className="w-full max-h-[300px] overflow-y-auto">
      {trainedStudents.length === 0 ? (
        <DropdownMenuItem disabled className="text-center text-gray-500">
          No trained models found
        </DropdownMenuItem>
      ) : (
        trainedStudents.map((student, index) => (
          <DropdownMenuItem
            key={index}
            onClick={() => {
              setSelectedStudent(student.student_id);
              setVerificationResult(null);
            }}
          >
            <div className="flex flex-col">
              <span className="font-medium">{student.student_id}</span>
              <span className="text-xs text-gray-500">
                Trained: {new Date(student.metadata.training_date).toLocaleDateString()}
              </span>
            </div>
          </DropdownMenuItem>
        ))
      )}
    </DropdownMenuContent>
  </DropdownMenu>
</div>

          {/* Verification Result */}
          {verificationResult && (
            <div className={`p-4 rounded-lg border-2 ${
              verificationResult.isVerified 
                ? 'bg-green-50 border-green-200' 
                : 'bg-red-50 border-red-200'
            }`}>
              <div className="flex items-center gap-3">
                {verificationResult.isVerified ? (
                  <CheckCircle className="w-8 h-8 text-green-600 flex-shrink-0" />
                ) : (
                  <XCircle className="w-8 h-8 text-red-600 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <div className={`font-semibold text-lg ${
                    verificationResult.isVerified ? 'text-green-900' : 'text-red-900'
                  }`}>
                    {verificationResult.isVerified ? 'VERIFIED' : 'NOT VERIFIED'}
                  </div>
                  <div className="text-sm text-gray-700 mt-1">
                    Confidence: {(verificationResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Verify Button */}
        <div className="border-t pt-4 mt-auto">
          <Button
            onClick={handleVerifySignature}
            disabled={!selectedStudent || (!localPreviewImage && !previewImage && activeMode !== 'webcam') || isVerifying}
            className="w-full"
          >
            {isVerifying ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Verifying...
              </>
            ) : (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                Verify Signature
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default Verification;