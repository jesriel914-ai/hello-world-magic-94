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
import { FileImage, X, Loader2, Upload, Camera, FolderOpen, Cloud, ChevronDown, List, Shield, CheckCircle, XCircle, User } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import { MobileWebcam } from '@/components/model-training-ui/services/mobileWebcam';
import { toast } from '@/hooks/use-toast';
import { siameseModelService } from '../lib/AIModelService';
import type { Student } from '@/types';

// Interfaces
export interface ClassData {
  student: Student | null;
  color: string;
  samples: any[];
  genuineSamples: any[];
  forgedSamples: any[];
}

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
  
  // Verification state
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    isVerified: boolean;
    confidence: number;
    studentId?: string;
  } | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<string>('');

  const [isCameraReady, setIsCameraReady] = useState(false);

  // Mock classes data - in real app this would come from props or context
  const [classes, setClasses] = useState<ClassData[]>([]);

  // Get available classes for dropdown
  const availableClasses = classes.filter(cls => cls.student);

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
    
    if (isVideoPaused) {
      mobileWebcam.current.resume();
      setIsVideoPaused(false);
    } else {
      mobileWebcam.current.pause();
      setIsVideoPaused(true);
    }
  };

  const startWebcam = async () => {
    if (!webcamRef.current) return;
    
    setIsCameraStarting(true);
    setCameraError(null);
    
    try {
      mobileWebcam.current = new MobileWebcam(webcamRef.current, {
        width: 224,
        height: 224,
        facingMode: 'user'
      });
      
      await mobileWebcam.current.setup();
      setIsWebcamActive(true);
      setIsCameraReady(true);
      
    } catch (error) {
      console.error('Camera setup failed:', error);
      setCameraError('Failed to start camera. Please check permissions.');
    } finally {
      setIsCameraStarting(false);
    }
  };

  const stopWebcam = () => {
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
      mobileWebcam.current = null;
    }
    setIsWebcamActive(false);
    setIsCameraReady(false);
    setIsVideoPaused(false);
  };

  const captureFrame = () => {
    if (!mobileWebcam.current) return;
    
    try {
      const imageData = mobileWebcam.current.capture();
      if (imageData) {
        setPreviewImage(imageData);
        setLocalPreviewImage(null);
      }
    } catch (error) {
      console.error('Frame capture failed:', error);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
      }
      if (cameraPredictionIntervalRef.current) {
        clearInterval(cameraPredictionIntervalRef.current);
      }
    };
  }, []);

  // Render camera display
  const renderCameraDisplay = () => {
    if (!isWebcamActive) {
      return (
        <div className="flex flex-col items-center justify-center h-64 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
          <Camera className="w-12 h-12 text-gray-400 mb-4" />
          <p className="text-gray-500 mb-4">Camera not active</p>
          <Button onClick={startWebcam} disabled={isCameraStarting}>
            {isCameraStarting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Starting Camera...
              </>
            ) : (
              <>
                <Camera className="w-4 h-4 mr-2" />
                Start Camera
              </>
            )}
          </Button>
          {cameraError && (
            <p className="text-red-500 text-sm mt-2">{cameraError}</p>
          )}
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="relative">
          <div 
            ref={webcamRef}
            className="w-full h-64 bg-black rounded-lg overflow-hidden"
          />
          {isVideoPaused && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
              <div className="text-white text-center">
                <Pause className="w-8 h-8 mx-auto mb-2" />
                <p>Video Paused</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex gap-2">
          <Button onClick={handleWebcamClick} variant="outline" className="flex-1">
            {isVideoPaused ? 'Resume' : 'Pause'} Video
          </Button>
          <Button onClick={captureFrame} className="flex-1">
            <Camera className="w-4 h-4 mr-2" />
            Capture Frame
          </Button>
          <Button onClick={stopWebcam} variant="destructive">
            <X className="w-4 h-4 mr-2" />
            Stop
          </Button>
        </div>
      </div>
    );
  };

  return (
    <Card className="h-[605px] w-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <FileImage className="w-5 h-5" />
          Verification
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden flex flex-col">
        {isMobile ? (
          <div className="flex flex-col space-y-4">
            {/* Mode Selection */}
            <div className="flex gap-2">
              <Button
                variant={activeMode === 'upload' ? 'default' : 'outline'}
                onClick={() => setActiveMode('upload')}
                className="flex-1"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </Button>
              <Button
                variant={activeMode === 'webcam' ? 'default' : 'outline'}
                onClick={() => setActiveMode('webcam')}
                className="flex-1"
              >
                <Camera className="w-4 h-4 mr-2" />
                Camera
              </Button>
            </div>

            {/* Upload Mode */}
            {activeMode === 'upload' && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-4">Upload signature image</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handlePreviewFileUpload}
                    className="hidden"
                    id="mobile-upload"
                  />
                  <label
                    htmlFor="mobile-upload"
                    className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 cursor-pointer"
                  >
                    Choose File
                  </label>
                </div>
                
                {(localPreviewImage || previewImage) && (
                  <div className="relative">
                    <img
                      src={localPreviewImage || previewImage || ''}
                      alt="Preview"
                      className="w-full h-48 object-contain bg-gray-100 rounded-lg"
                    />
                    <button
                      onClick={() => {
                        setLocalPreviewImage(null);
                        setPreviewImage(null);
                      }}
                      className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Camera Mode */}
            {activeMode === 'webcam' && renderCameraDisplay()}

            {/* Student Selection and Verify Button Row */}
            <div className="flex items-center gap-3 pt-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="flex-1 justify-start">
                    <User className="w-4 h-4 mr-2" />
                    {selectedStudent ? availableClasses.find(cls => cls.student?.student_id === selectedStudent)?.student?.firstname + ' ' + availableClasses.find(cls => cls.student?.student_id === selectedStudent)?.student?.surname : 'Select Class'}
                    <ChevronDown className="w-4 h-4 ml-auto" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-full">
                  {availableClasses.map((cls) => (
                    <DropdownMenuItem
                      key={cls.student?.student_id}
                      onClick={() => setSelectedStudent(cls.student?.student_id || '')}
                      className="flex items-center gap-2"
                    >
                      <User className="w-4 h-4" />
                      {cls.student?.firstname} {cls.student?.surname}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
            {/* Left Column - Image Input */}
            <div className="space-y-4">
              {/* Mode Selection */}
              <div className="flex gap-2">
                <Button
                  variant={activeMode === 'upload' ? 'default' : 'outline'}
                  onClick={() => setActiveMode('upload')}
                  className="flex-1"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Image
                </Button>
                <Button
                  variant={activeMode === 'webcam' ? 'default' : 'outline'}
                  onClick={() => setActiveMode('webcam')}
                  className="flex-1"
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Use Camera
                </Button>
              </div>

              {/* Upload Mode */}
              {activeMode === 'upload' && (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-4">Upload signature image</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handlePreviewFileUpload}
                    className="hidden"
                    id="desktop-upload"
                  />
                  <label
                    htmlFor="desktop-upload"
                    className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 cursor-pointer"
                  >
                    Choose File
                  </label>
                </div>
              )}

              {/* Camera Mode */}
              {activeMode === 'webcam' && renderCameraDisplay()}

              {/* Preview Image */}
              {(localPreviewImage || previewImage) && (
                <div className="relative">
                  <img
                    src={localPreviewImage || previewImage || ''}
                    alt="Preview"
                    className="w-full h-48 object-contain bg-gray-100 rounded-lg"
                  />
                  <button
                    onClick={() => {
                      setLocalPreviewImage(null);
                      setPreviewImage(null);
                    }}
                    className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center"
                  >
                    <X className="w-4 h-4" />
                  </button>
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
                  {selectedStudent ? availableClasses.find(cls => cls.student?.student_id === selectedStudent)?.student?.firstname + ' ' + availableClasses.find(cls => cls.student?.student_id === selectedStudent)?.student?.surname : 'Select Class'}
                  <ChevronDown className="w-4 h-4 ml-auto" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-full">
                {availableClasses.map((cls) => (
                  <DropdownMenuItem
                    key={cls.student?.student_id}
                    onClick={() => setSelectedStudent(cls.student?.student_id || '')}
                    className="flex items-center gap-2"
                  >
                    <User className="w-4 h-4" />
                    {cls.student?.firstname} {cls.student?.surname}
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
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default Verification;