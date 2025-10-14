//filepath: src\components\model-training-ui\components\TrainedModelsForm.tsx
import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Brain, Loader2, Download, Check, List } from 'lucide-react';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { TrainedModel } from '../../ModelTraining';
import JSZip from 'jszip';

interface TrainedModelsFormProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  trainedModels?: TrainedModel[];
  isLoadingModels: boolean;
  onModelSelect?: (model: TrainedModel) => void;
}

const TrainedModelsForm: React.FC<TrainedModelsFormProps> = ({
  open,
  onOpenChange,
  trainedModels,
  isLoadingModels,
  onModelSelect
}) => {
  const isMobile = useMobileDetection();
  
  // Download state
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [downloadErrors, setDownloadErrors] = useState<Record<string, string>>({});
  
  // Expanded models state for showing names
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  
  // Toggle model expansion
  const toggleModelExpansion = (modelId: string) => {
    setExpandedModels(prev => {
      const newSet = new Set(prev);
      if (newSet.has(modelId)) {
        newSet.delete(modelId);
      } else {
        newSet.add(modelId);
      }
      return newSet;
    });
  };

  // Handle model selection
  const handleSelectModel = (model: TrainedModel) => {
    if (onModelSelect) {
      onModelSelect(model);
      onOpenChange(false); // Close the dialog after selection
    }
  };

// Download model handler - FIXED for 3-file structure
const handleDownloadModel = async (modelId: string, studentName: string) => {
  console.log(`🔥 Download requested for modelId: "${modelId}"`);
  
  // Validate modelId
  if (!modelId || typeof modelId !== 'string' || !modelId.trim()) {
    console.error('❌ Invalid modelId provided for download');
    throw new Error('Invalid model ID');
  }
  
  const trimmedModelId = modelId.trim();
  
  if (downloadingModels.has(trimmedModelId)) {
    console.log('⚠️ Model already downloading:', trimmedModelId);
    return;
  }
  
  try {
    // Clear any previous errors
    setDownloadErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[trimmedModelId];
      return newErrors;
    });
    
    // Add to downloading set
    setDownloadingModels(prev => new Set(prev).add(trimmedModelId));
    setDownloadProgress(prev => ({ ...prev, [trimmedModelId]: 0 }));
    
    // Import AIModelService
    const getAIModelService = (await import('@/lib/AIModelService')).default;
    const aiService = getAIModelService();
    
    // Simulate progress
    const progressInterval = setInterval(() => {
      setDownloadProgress(prev => {
        const currentProgress = prev[trimmedModelId] || 0;
        const newProgress = Math.min(currentProgress + 15, 90);
        return { ...prev, [trimmedModelId]: newProgress };
      });
    }, 300);
    
    // Download the model
    console.log(`📡 Downloading model from S3...`);
    const downloadResult = await aiService.downloadModel(trimmedModelId);
    
    clearInterval(progressInterval);
    
    if (!downloadResult.success) {
      throw new Error(downloadResult.error instanceof Error ? downloadResult.error.message : 'Failed to download model');
    }
    
    setDownloadProgress(prev => ({ ...prev, [trimmedModelId]: 100 }));
    
    // Parse the downloaded model data
    console.log('🔄 Processing downloaded model data...');
    
    if (typeof downloadResult.data !== 'string') {
      throw new Error('Downloaded data is not in expected string format');
    }
    
    let combinedData;
    try {
      combinedData = JSON.parse(downloadResult.data);
    } catch (parseError) {
      throw new Error('Failed to parse downloaded model data');
    }
    
    // Validate 3-file structure
    if (!combinedData.modelJson || !combinedData.weightsBin || !combinedData.metadataJson) {
      throw new Error('Downloaded model is missing required files. Expected: modelJson, weightsBin, metadataJson');
    }
    
    console.log('✅ Model has correct 3-file structure');
    
    // Parse metadata to get labels
    const metadata = JSON.parse(combinedData.metadataJson);
    const labels = metadata.labels || [studentName];
    
    // Create enhanced metadata matching export format
    const enhancedMetadata = {
      modelName: metadata.modelName || `model-${Date.now()}`,
      labels: labels,
      imageSize: metadata.imageSize || 224,
      createdAt: metadata.createdAt || new Date().toISOString(),
      userMetadata: {
        student_id: metadata.userMetadata?.student_id || '',
        student_name: metadata.userMetadata?.student_name || studentName,
        sample_count: metadata.userMetadata?.sample_count || 0,
        accuracy: metadata.userMetadata?.accuracy || 0.85,
        training_date: metadata.userMetadata?.training_date || new Date().toISOString(),
        model_architecture: metadata.userMetadata?.model_architecture || 'mobilenet_v1_classifier',
        total_students: metadata.userMetadata?.total_students || labels.length,
        training_summary: metadata.userMetadata?.training_summary || ''
      },
      tfjsVersion: metadata.tfjsVersion || tf.version.tfjs,
      modelType: metadata.modelType || 'image_classification',
      inputShape: metadata.inputShape || [224, 224, 3],
      outputShape: metadata.outputShape || [labels.length]
    };
    
    // Create timestamp for filename
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const zipFileName = `model-${timestamp}.zip`;
    
    // Create ZIP file with 3 files
    console.log('📦 Creating ZIP with 3 files...');
    const zip = new JSZip();
    
    // Add model.json
    zip.file('model.json', combinedData.modelJson);
    
    // Add weights.bin (convert from base64)
    let weightsBlob;
    try {
      const base64Data = combinedData.weightsBin.split(',')[1] || combinedData.weightsBin;
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      weightsBlob = new Blob([bytes], { type: 'application/octet-stream' });
      console.log('✅ Weights blob created, size:', weightsBlob.size, 'bytes');
    } catch (base64Error) {
      console.error('❌ Failed to convert base64 weights:', base64Error);
      throw new Error('Invalid weights data format');
    }
    
    zip.file('weights.bin', weightsBlob);
    
    // Add metadata.json with enhanced metadata
    zip.file('metadata.json', JSON.stringify(enhancedMetadata, null, 2));
    
    // Generate and download ZIP
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    
    const url = URL.createObjectURL(zipBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = zipFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    console.log('✅ Model downloaded successfully:', zipFileName);
    
    // Remove from downloading set after short delay
    setTimeout(() => {
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(trimmedModelId);
        return newSet;
      });
      setDownloadProgress(prev => {
        const newProgress = { ...prev };
        delete newProgress[trimmedModelId];
        return newProgress;
      });
    }, 1000);
    
  } catch (error) {
    console.error('❌ Error downloading model:', error);
    setDownloadErrors(prev => ({
      ...prev,
      [trimmedModelId]: error instanceof Error ? error.message : 'Failed to download model'
    }));
    
    // Clean up on error
    setDownloadingModels(prev => {
      const newSet = new Set(prev);
      newSet.delete(trimmedModelId);
      return newSet;
    });
    setDownloadProgress(prev => {
      const newProgress = { ...prev };
      delete newProgress[trimmedModelId];
      return newProgress;
    });
  }
};

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl" aria-describedby="trained-models-description">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            <span>Trained Models ({trainedModels?.filter(model => model.id && model.id.trim() !== '').length || 0})</span>
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {isLoadingModels ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin mr-3" />
              <span className="text-lg">Loading models...</span>
            </div>
          ) : !trainedModels || trainedModels.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-500 mb-2">No trained models yet</h3>
              <p className="text-gray-400 text-sm">Start by training your first model</p>
            </div>
          ) : (
            <div className="space-y-1 max-h-96 overflow-y-auto hide-scrollbar">
              {/* Filter out models with completely invalid IDs but allow model.json since that's what we have */}
              {trainedModels
                ?.filter(model => model.id && model.id.trim() !== '')
                .map((model, index) => {
                  // Create a unique key using training_date since model.id may be duplicated
                  const uniqueKey = model.training_date || `model-${index}`;
                  const safeModelId = model.id || `model-${index}`;
                  const isDownloading = downloadingModels.has(safeModelId);
                  const progress = downloadProgress[safeModelId] || 0;
                  const hasError = downloadErrors[safeModelId];
                
                return (
                  <div key={uniqueKey} className="flex flex-col p-3 bg-white rounded-lg border border-gray-100 shadow-xs hover:shadow-sm transition-shadow">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">
                          {new Date(model.training_date).toLocaleDateString('en-US', { 
                            year: 'numeric', 
                            month: 'long', 
                            day: 'numeric' 
                          })} {new Date(model.training_date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true })}
                        </div>
                        <div className="text-gray-500 text-sm mt-1">
                          {model.training_metrics?.students?.length || 1} student{((model.training_metrics?.students?.length || 1) !== 1) ? 's' : ''}
                        </div>
                      </div>
                      <div className="text-right ml-4">
                        
                        {/* Action buttons */}
                        <div className="flex items-center gap-1 mt-2">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 w-8 p-0"
                            title="List menu"
                            onClick={() => toggleModelExpansion(uniqueKey)}
                          >
                            <List className="w-4 h-4 text-blue-600" />
                          </Button>
                          {isDownloading ? (
                            <div className="flex items-center gap-2">
                              <Loader2 className="w-3 h-3 animate-spin text-blue-600" />
                              <span className="text-xs text-blue-600 font-medium">{progress}%</span>
                            </div>
                          ) : (
                            <>
                              <Button
                                variant={hasError ? "destructive" : "outline"}
                                size="sm"
                                className={`h-8 w-8 p-0 ${hasError ? 'bg-red-50 hover:bg-red-100 border-red-200' : ''}`}
                                onClick={() => {
                                  handleDownloadModel(model.id, model.student_name);
                                }}
                                title={hasError ? `Error: ${downloadErrors[safeModelId]}` : 'Download model'}
                              >
                                <Download className={`w-4 h-4 ${hasError ? 'text-red-600' : 'text-blue-600'}`} />
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-8 w-8 p-0"
                                title="Select model"
                                onClick={() => handleSelectModel(model)}
                              >
                                <Check className="w-4 h-4 text-blue-600" />
                              </Button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    {/* Names section - shown when expanded */}
                    {expandedModels.has(uniqueKey) && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <div className="text-sm font-medium text-gray-700 mb-2">Students:</div>
                        <div className="space-y-1">
                          {model.training_metrics?.students?.map((student, index) => (
                            <div key={student.id || student.student_id || index} className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              {student.firstname} {student.surname}
                            </div>
                          )) || (
                            <div className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              {model.student_name}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TrainedModelsForm;