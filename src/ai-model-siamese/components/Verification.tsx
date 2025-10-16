import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FileImage, X, Loader2, Upload, Camera, CheckCircle, XCircle, User, Search } from 'lucide-react';
import { siameseModelService } from '../lib/SiameseAIModelService';

interface VerificationProps {}

const Verification: React.FC<VerificationProps> = () => {
  const [activeMode, setActiveMode] = useState<'upload'>('upload');
  const [localPreviewImage, setLocalPreviewImage] = useState<string | null>(null);
  
  // Classification state (1:N - Automatic identification)
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState<{
    identified: boolean;
    student_id: string | null;
    confidence: number;
    top_matches?: any[];
  } | null>(null);
  
  // Verification state (1:1 - After classification)
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    isVerified: boolean;
    confidence: number;
    studentId?: string;
  } | null>(null);

  // Auto-classify when image is uploaded
  const handlePreviewFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (e) => {
      const result = e.target?.result as string;
      setLocalPreviewImage(result);
      setClassificationResult(null);
      setVerificationResult(null);
      
      // Automatically classify the signature (1:N identification)
      await handleAutoClassify(result);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  // Auto-classify signature (1:N)
  const handleAutoClassify = async (imageData: string) => {
    setIsClassifying(true);
    setClassificationResult(null);
    
    try {
      console.log('🎯 Auto-classifying signature...');
      
      const result = await siameseModelService.classifySignature(imageData, 3);
      
      console.log('Classification result:', result);
      
      setClassificationResult({
        identified: result.identified,
        student_id: result.student_id,
        confidence: result.confidence,
        top_matches: result.top_matches
      });
      
    } catch (error) {
      console.error('Classification failed:', error);
      setClassificationResult({
        identified: false,
        student_id: null,
        confidence: 0,
        top_matches: []
      });
    } finally {
      setIsClassifying(false);
    }
  };

  // Verify signature (1:1) after classification
  const handleVerifySignature = async () => {
    if (!classificationResult?.identified || !classificationResult.student_id) {
      alert('No student identified. Cannot verify.');
      return;
    }
    
    if (!localPreviewImage) {
      alert('No signature image available');
      return;
    }
    
    setIsVerifying(true);
    setVerificationResult(null);
    
    try {
      console.log('🔍 Verifying signature (1:1)...');
      console.log('Student ID:', classificationResult.student_id);
      
      const result = await siameseModelService.verifySignature(
        classificationResult.student_id,
        localPreviewImage
      );
      
      console.log('Verification result:', result);
      
      setVerificationResult({
        isVerified: result.is_verified,
        confidence: result.confidence,
        studentId: classificationResult.student_id
      });
      
    } catch (error) {
      console.error('Verification failed:', error);
      alert(`Verification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsVerifying(false);
    }
  };

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
          
          {/* Upload Area */}
          <div className="relative border-2 border-dashed border-gray-300 rounded-lg aspect-video flex items-center justify-center bg-gray-50">
            {localPreviewImage ? (
              <div className="relative w-full h-full">
                <img 
                  src={localPreviewImage} 
                  alt="Preview" 
                  className="w-full h-full object-contain filter grayscale"
                />
                <button
                  onClick={() => {
                    setLocalPreviewImage(null);
                    setClassificationResult(null);
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
                <span className="text-sm text-gray-500">Upload signature for auto-identification</span>
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handlePreviewFileUpload}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* Auto-Classification Result (1:N) */}
          {isClassifying && (
            <div className="p-4 rounded-lg border-2 border-blue-200 bg-blue-50">
              <div className="flex items-center gap-3">
                <Loader2 className="w-6 h-6 text-blue-600 animate-spin flex-shrink-0" />
                <div className="flex-1">
                  <div className="font-semibold text-blue-900">
                    Identifying signature...
                  </div>
                  <div className="text-sm text-blue-700">
                    Searching through trained students (1:N)
                  </div>
                </div>
              </div>
            </div>
          )}

          {classificationResult && (
            <div className={`p-4 rounded-lg border-2 ${
              classificationResult.identified 
                ? 'bg-blue-50 border-blue-200' 
                : 'bg-yellow-50 border-yellow-200'
            }`}>
              <div className="flex items-center gap-3 mb-3">
                {classificationResult.identified ? (
                  <Search className="w-8 h-8 text-blue-600 flex-shrink-0" />
                ) : (
                  <Search className="w-8 h-8 text-yellow-600 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <div className={`font-semibold text-lg ${
                    classificationResult.identified ? 'text-blue-900' : 'text-yellow-900'
                  }`}>
                    {classificationResult.identified ? 'IDENTIFIED' : 'UNKNOWN SIGNATURE'}
                  </div>
                  {classificationResult.identified && classificationResult.student_id && (
                    <div className="text-sm text-gray-700 mt-1">
                      Student: <span className="font-medium">{classificationResult.student_id}</span>
                    </div>
                  )}
                  <div className="text-sm text-gray-700">
                    Confidence: {(classificationResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Top Matches */}
              {classificationResult.top_matches && classificationResult.top_matches.length > 0 && (
                <div className="border-t pt-3 mt-3">
                  <div className="text-xs font-medium text-gray-600 mb-2">
                    Top Matches:
                  </div>
                  <div className="space-y-1">
                    {classificationResult.top_matches.slice(0, 3).map((match, index) => (
                      <div 
                        key={index}
                        className={`flex items-center justify-between text-sm p-2 rounded ${
                          index === 0 && classificationResult.identified
                            ? 'bg-blue-100 border border-blue-300'
                            : 'bg-white border border-gray-200'
                        }`}
                      >
                        <span className="font-medium">
                          {index + 1}. {match.student_id}
                        </span>
                        <span className="text-gray-600">
                          {(match.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!classificationResult.identified && (
                <div className="border-t pt-3 mt-3">
                  <div className="text-xs text-yellow-700">
                    This signature is not recognized from the trained students.
                    {classificationResult.top_matches && classificationResult.top_matches.length > 0 && (
                      <span> The closest matches are shown above, but confidence is too low for positive identification.</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Verification Result (1:1) - Only after classification */}
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
                  <div className="text-xs text-gray-600 mt-1">
                    1:1 Verification against {verificationResult.studentId}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Explanation */}
          <div className="text-xs text-gray-500 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <div className="font-semibold mb-1">How it works:</div>
            <ol className="list-decimal list-inside space-y-1">
              <li><strong>1:N Classification:</strong> Upload a signature and it automatically identifies the owner from all trained students.</li>
              <li><strong>1:1 Verification:</strong> After identification, click "Verify" to perform detailed signature verification.</li>
              <li>If signature is unknown, it means it doesn't match any trained student.</li>
            </ol>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="border-t pt-4 mt-auto space-y-2">
          {/* Upload Button */}
          <label className="cursor-pointer">
            <Button
              variant="outline"
              className="w-full"
              asChild
            >
              <span>
                <Upload className="w-4 h-4 mr-2" />
                {localPreviewImage ? 'Upload Another Signature' : 'Upload Signature'}
              </span>
            </Button>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handlePreviewFileUpload}
              className="hidden"
            />
          </label>

          {/* Verify Button (Only shown if student identified) */}
          {classificationResult?.identified && (
            <Button
              onClick={handleVerifySignature}
              disabled={isVerifying || !localPreviewImage}
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
                  Verify Signature (1:1)
                </>
              )}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default Verification;