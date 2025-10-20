// filepath: src/ai-model-siamese/components/SignatureIdentification.tsx
/**
 * FIXED Signature Identification Component
 * Returns ONLY the owner (single result)
 * Handles non-signatures (random photos)
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  FileImage, 
  X, 
  Loader2, 
  Upload, 
  Search,
  AlertTriangle,
  CheckCircle,
  XCircle,
  HelpCircle,
  Ban
} from 'lucide-react';
import { siameseService } from '../lib/SiameseService';

interface ClassificationResult {
  identified: boolean;
  student_id: string | null;
  confidence: number;
  distance: number;
  decision: 'ACCEPT' | 'UNCERTAIN' | 'UNKNOWN' | 'NON_SIGNATURE' | 'REJECT';
  message: string;
  threshold_info: {
    accept_threshold: number;
    reject_threshold: number;
    nonsig_threshold: number;
  };
}

const SignatureIdentification: React.FC = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageData = e.target?.result as string;
      setUploadedImage(imageData);
      setResult(null);

      // Auto-classify
      await performClassification(imageData);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  const performClassification = async (imageData: string) => {
    setIsClassifying(true);
    setResult(null);

    try {
      console.log('🎯 Starting signature identification...');
      
      const response = await siameseService.classifySignature(imageData);
      
      console.log('Classification complete:', response);
      setResult(response);

    } catch (error) {
      console.error('Classification failed:', error);
      alert(`Classification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsClassifying(false);
    }
  };

  const clearImage = () => {
    setUploadedImage(null);
    setResult(null);
  };

  const getStatusIcon = () => {
    if (!result) return null;
    
    switch (result.decision) {
      case 'ACCEPT':
        return <CheckCircle className="w-12 h-12 text-green-600" />;
      case 'NON_SIGNATURE':
        return <Ban className="w-12 h-12 text-orange-600" />;
      case 'UNKNOWN':
        return <XCircle className="w-12 h-12 text-red-600" />;
      case 'UNCERTAIN':
        return <HelpCircle className="w-12 h-12 text-yellow-600" />;
      default:
        return <AlertTriangle className="w-12 h-12 text-gray-600" />;
    }
  };

  const getStatusColor = () => {
    if (!result) return '';
    
    switch (result.decision) {
      case 'ACCEPT':
        return 'bg-green-50 border-green-300';
      case 'NON_SIGNATURE':
        return 'bg-orange-50 border-orange-300';
      case 'UNKNOWN':
        return 'bg-red-50 border-red-300';
      case 'UNCERTAIN':
        return 'bg-yellow-50 border-yellow-300';
      default:
        return 'bg-gray-50 border-gray-300';
    }
  };

  const getStatusTitle = () => {
    if (!result) return '';
    
    switch (result.decision) {
      case 'ACCEPT':
        return '✅ OWNER IDENTIFIED';
      case 'NON_SIGNATURE':
        return '🚫 NOT A SIGNATURE';
      case 'UNKNOWN':
        return '❌ UNKNOWN STUDENT';
      case 'UNCERTAIN':
        return '❓ UNCERTAIN';
      default:
        return '⚠️ UNABLE TO IDENTIFY';
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card className="h-[605px] w-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            Signature Identification
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto overlay-scrollbar-container space-y-4">
          
          {/* Upload Area */}
          <div className="relative border-2 border-dashed border-gray-300 rounded-lg aspect-video flex items-center justify-center bg-gray-50 hover:border-blue-400 transition-colors">
            {uploadedImage ? (
              <div className="relative w-full h-full">
                <img 
                  src={uploadedImage} 
                  alt="Uploaded signature" 
                  className="w-full h-full object-contain filter grayscale"
                />
                <button
                  onClick={clearImage}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600 transition-colors shadow-lg"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <label className="cursor-pointer flex flex-col items-center gap-3 p-8">
                <FileImage className="w-16 h-16 text-gray-400" />
                <div className="text-center">
                  <span className="text-sm font-medium text-gray-700 block">
                    Upload signature to identify owner
                  </span>
                  <span className="text-xs text-gray-500 mt-1 block">
                    System will automatically find the owner
                  </span>
                </div>
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* Classification In Progress */}
          {isClassifying && (
            <div className="p-4 rounded-lg border-2 border-blue-200 bg-blue-50 animate-pulse">
              <div className="flex items-center gap-3">
                <Loader2 className="w-6 h-6 text-blue-600 animate-spin flex-shrink-0" />
                <div className="flex-1">
                  <div className="font-semibold text-blue-900">
                    Identifying signature owner...
                  </div>
                  <div className="text-sm text-blue-700 mt-1">
                    Comparing against trained students
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Classification Result */}
          {result && !isClassifying && (
            <div className="space-y-4">
              
              {/* Main Result Card */}
              <div className={`p-6 rounded-lg border-2 ${getStatusColor()}`}>
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className="flex-shrink-0 mt-1">
                    {getStatusIcon()}
                  </div>

                  {/* Result Content */}
                  <div className="flex-1 min-w-0">
                    {/* Status Title */}
                    <div className={`text-2xl font-bold mb-3 ${
                      result.decision === 'ACCEPT' ? 'text-green-700' :
                      result.decision === 'NON_SIGNATURE' ? 'text-orange-700' :
                      result.decision === 'UNKNOWN' ? 'text-red-700' :
                      result.decision === 'UNCERTAIN' ? 'text-yellow-700' :
                      'text-gray-700'
                    }`}>
                      {getStatusTitle()}
                    </div>

                    {/* Student ID (if identified) */}
                    {result.identified && result.student_id && (
                      <div className="mb-4 p-4 bg-white rounded-md border-2 border-green-300 shadow-sm">
                        <div className="text-sm text-gray-600 mb-1 font-medium">
                          Student ID
                        </div>
                        <div className="text-3xl font-bold text-gray-900">
                          {result.student_id}
                        </div>
                      </div>
                    )}

                    {/* Confidence */}
                    <div className="mb-3 p-3 bg-white/70 rounded-md">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700">
                          Confidence Level
                        </span>
                        <span className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-500 ${
                            result.confidence >= 0.8 ? 'bg-green-500' :
                            result.confidence >= 0.6 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Message */}
                    <div className="p-3 bg-white/50 rounded text-sm text-gray-700 leading-relaxed">
                      {result.message}
                    </div>

                    {/* Technical Details (Collapsible) */}
                    <details className="mt-3">
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        Technical Details
                      </summary>
                      <div className="mt-2 p-2 bg-white/50 rounded text-xs text-gray-600 space-y-1">
                        <div>Distance: {result.distance.toFixed(3)}</div>
                        <div>Accept Threshold: {result.threshold_info.accept_threshold.toFixed(3)}</div>
                        <div>Reject Threshold: {result.threshold_info.reject_threshold.toFixed(3)}</div>
                        <div>Non-Signature Threshold: {result.threshold_info.nonsig_threshold.toFixed(3)}</div>
                        <div>Decision: {result.decision}</div>
                      </div>
                    </details>
                  </div>
                </div>
              </div>

              {/* Explanation Based on Result */}
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div className="text-sm text-gray-700">
                  {result.decision === 'ACCEPT' && (
                    <>
                      <div className="font-semibold mb-2">✅ Match Found</div>
                      <p>The signature has been successfully matched to <strong>{result.student_id}</strong>. 
                      The high confidence level indicates this is very likely their genuine signature.</p>
                    </>
                  )}
                  {result.decision === 'NON_SIGNATURE' && (
                    <>
                      <div className="font-semibold mb-2">🚫 Not a Signature</div>
                      <p>The uploaded image does not appear to be a signature. 
                      Please upload a clear image of a signature. Random photos, blank pages, 
                      or non-signature content will be rejected.</p>
                    </>
                  )}
                  {result.decision === 'UNKNOWN' && (
                    <>
                      <div className="font-semibold mb-2">❌ Unknown Student</div>
                      <p>This signature does not match any student in the training database. 
                      The student may not be enrolled yet. Please train their signature first 
                      in the Training Setup tab.</p>
                    </>
                  )}
                  {result.decision === 'UNCERTAIN' && (
                    <>
                      <div className="font-semibold mb-2">❓ Uncertain Match</div>
                      <p>The system cannot confidently identify this signature. 
                      This might be due to poor image quality, an unknown student, 
                      or insufficient training samples.</p>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* How It Works Info */}
          {!uploadedImage && (
            <div className="text-xs text-gray-500 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="font-semibold mb-2 text-gray-700">📋 How it works:</div>
              <ol className="list-decimal list-inside space-y-1.5">
                <li>
                  <strong>Upload:</strong> Click above to upload any signature image
                </li>
                <li>
                  <strong>Auto-Identify:</strong> System finds the owner automatically
                </li>
                <li>
                  <strong>Results:</strong> Shows ONLY the owner (not top 5)
                </li>
                <li>
                  <strong>Smart Detection:</strong> Rejects random photos (non-signatures)
                </li>
                <li>
                  <strong>Unknown Handling:</strong> Detects if student not trained
                </li>
              </ol>
              <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="text-xs text-blue-700">
                  💡 <strong>Tip:</strong> Make sure students are trained first in the Training Setup tab.
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action Button */}
        <div className="border-t pt-4 mt-auto">
          <label className="cursor-pointer block">
            <Button
              variant={uploadedImage ? "outline" : "default"}
              className="w-full"
              asChild
            >
              <span>
                <Upload className="w-4 h-4 mr-2" />
                {uploadedImage ? 'Upload Different Signature' : 'Upload Signature'}
              </span>
            </Button>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload}
              className="hidden"
            />
          </label>
        </div>
      </CardContent>
    </Card>
  );
};

export default SignatureIdentification;