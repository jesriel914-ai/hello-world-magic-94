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

  // Download model handler
  const handleDownloadModel = async (modelId: string, studentName: string) => {
    // Debug: Log the modelId being passed to download
    console.log(`🔥 Download requested for modelId: "${modelId}", studentName: "${studentName}"`);
    
    // Enhanced modelId validation
    if (!modelId || typeof modelId !== 'string') {
      console.error('❌ Invalid modelId provided for download:', modelId);
      throw new Error('Invalid model ID: Model ID must be a non-empty string');
    }
    
    const trimmedModelId = modelId.trim();
    if (!trimmedModelId) {
      console.error('❌ Empty modelId provided for download after trimming');
      throw new Error('Invalid model ID: Model ID cannot be empty or whitespace');
    }
    
    // Check for known invalid model IDs (but allow model.json since that's what we have in existing data)
    const invalidIds = ['undefined', 'null', 'NaN', ''];
    if (invalidIds.includes(trimmedModelId)) {
      console.error('❌ Invalid modelId value:', trimmedModelId);
      throw new Error(`Invalid model ID: '${trimmedModelId}' is not a valid model identifier`);
    }
    
    // Check for valid model ID formats (UUID, timestamp, or model_ prefix with timestamp)
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    const timestampRegex = /^model-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$/;
    const modelPrefixRegex = /^model_\d+_[a-z0-9]+$/;
    
    // Allow model.json for backward compatibility with existing data
    if (!uuidRegex.test(trimmedModelId) && !timestampRegex.test(trimmedModelId) && !modelPrefixRegex.test(trimmedModelId) && trimmedModelId !== 'model.json') {
      console.warn('⚠️ Model ID does not match expected format:', trimmedModelId);
    } // Don't throw error here, just warn - allow download to proceed

    if (downloadingModels.has(trimmedModelId)) {
      console.log('⚠️ Model already downloading:', trimmedModelId);
      return;
    }
    
    try {
      // Clear any previous errors for this model
      setDownloadErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[trimmedModelId];
        return newErrors;
      });
      
      // Add to downloading set
      setDownloadingModels(prev => new Set(prev).add(trimmedModelId));
      setDownloadProgress(prev => ({ ...prev, [trimmedModelId]: 0 }));
      
      // Import AIModelService dynamically to avoid server-side import issues
      const getAIModelService = (await import('@/lib/AIModelService')).default;
      const aiService = getAIModelService();
      
      // Simulate progress while downloading (since we can't track actual progress easily)
      const progressInterval = setInterval(() => {
        setDownloadProgress(prev => {
          const currentProgress = prev[trimmedModelId] || 0;
          const newProgress = Math.min(currentProgress + 15, 90);
          return { ...prev, [trimmedModelId]: newProgress };
        });
      }, 300);
      
      // Download the actual model from AIModelService
      console.log(`📡 Calling aiService.downloadModel with modelId: "${trimmedModelId}"`);
      const downloadResult = await aiService.downloadModel(trimmedModelId);
      console.log(`📡 Download result:`, downloadResult);
      
      clearInterval(progressInterval);
      
      if (!downloadResult.success) {
        throw new Error(downloadResult.error instanceof Error ? downloadResult.error.message : 'Failed to download model');
      }
      
      setDownloadProgress(prev => ({ ...prev, [trimmedModelId]: 100 }));
      
      // Parse the downloaded model data
      let modelData;
      let parsedMetadata;
      try {
        // Type guard to ensure downloadResult.data is a string
        if (typeof downloadResult.data !== 'string') {
          throw new Error('Downloaded data is not in expected string format');
        }
        
        // Create a proper wrapper object for the model data
        modelData = {
          modelJson: downloadResult.data, // Store the original JSON string
          weightsBin: downloadResult.weights || '', // Handle weights from server response
          metadataJson: '' // Will be populated below
        };
        
        // Debug: Log weights information
        console.log('📊 Download result weights info:', {
          hasWeights: !!downloadResult.weights,
          weightsLength: downloadResult.weights ? downloadResult.weights.length : 0,
          weightsType: typeof downloadResult.weights
        });
        
        // Parse metadata if it exists in the download result
        if (downloadResult.metadata) {
          parsedMetadata = downloadResult.metadata;
        } else {
          parsedMetadata = null;
        }
      } catch (parseError) {
        // If JSON parsing fails, this might be old format data
        // Handle old format (single file) as fallback
        // Ensure data is treated as string in fallback case
        const dataString = typeof downloadResult.data === 'string' ? downloadResult.data : JSON.stringify(downloadResult.data);
        modelData = {
          modelJson: dataString,
          weightsBin: '',
          metadataJson: JSON.stringify({
            format: 'tensorflowjs',
            generatedBy: 'signature-ai',
            convertedBy: null,
            userMetadata: {
              labels: [studentName || 'Unknown'],
              modelType: 'image_classification',
              inputSize: [224, 224, 3],
              author: 'signature-ai',
              accuracy: 0.85,
              totalSamples: 0,
              trainingTime: 0
            },
            signatures: {
              'default': {
                inputs: [{
                  name: 'input',
                  dtype: 'float32',
                  shape: [1, 224, 224, 3]
                }],
                outputs: [{
                  name: 'output',
                  dtype: 'float32',
                  shape: [1, Math.max((parsedMetadata?.userMetadata?.labels || [studentName || 'Unknown']).length, 1)]
                }]
              }
            }
          })
        };
        parsedMetadata = JSON.parse(modelData.metadataJson);
      }
      
      // Ensure metadata structure matches export format (S3 export structure)
      if (!parsedMetadata || !parsedMetadata.format) {
        // Create comprehensive metadata matching S3 export format
        const labels = parsedMetadata?.userMetadata?.labels || [studentName || 'Unknown'];
        const enhancedMetadata = {
          format: 'tensorflowjs',
          generatedBy: 'signature-ai',
          convertedBy: null,
          userMetadata: {
            labels: labels,
            modelType: 'image_classification',
            inputSize: [224, 224, 3],
            author: 'signature-ai',
            accuracy: parsedMetadata?.userMetadata?.accuracy || 0.85,
            totalSamples: parsedMetadata?.userMetadata?.totalSamples || 0,
            trainingTime: parsedMetadata?.userMetadata?.trainingTime || 0
          },
          signatures: {
            'default': {
              inputs: [{
                name: 'input',
                dtype: 'float32',
                shape: [1, 224, 224, 3]
              }],
              outputs: [{
                name: 'output',
                dtype: 'float32',
                shape: [1, Math.max(labels.length, 1)]
              }]
            }
          }
        };
        
        modelData.metadataJson = JSON.stringify(enhancedMetadata, null, 2);
        parsedMetadata = enhancedMetadata;
      } else {
        // Enhance existing metadata to ensure all required fields are present
        const enhancedMetadata = {
          ...parsedMetadata,
          format: parsedMetadata.format || 'tensorflowjs',
          generatedBy: parsedMetadata.generatedBy || 'signature-ai',
          convertedBy: parsedMetadata.convertedBy !== undefined ? parsedMetadata.convertedBy : null,
          userMetadata: {
            labels: parsedMetadata.userMetadata?.labels || [studentName || 'Unknown'],
            modelType: parsedMetadata.userMetadata?.modelType || 'image_classification',
            inputSize: parsedMetadata.userMetadata?.inputSize || [224, 224, 3],
            author: parsedMetadata.userMetadata?.author || 'signature-ai',
            accuracy: parsedMetadata.userMetadata?.accuracy || 0.85,
            totalSamples: parsedMetadata.userMetadata?.totalSamples || 0,
            trainingTime: parsedMetadata.userMetadata?.trainingTime || 0,
            ...parsedMetadata.userMetadata // Preserve any additional fields
          },
          signatures: parsedMetadata.signatures || {
            'default': {
              inputs: [{
                name: 'input',
                dtype: 'float32',
                shape: [1, 224, 224, 3]
              }],
              outputs: [{
                name: 'output',
                dtype: 'float32',
                shape: [1, Math.max(parsedMetadata.userMetadata?.labels?.length || 1, 1)]
              }]
            }
          }
        };
        
        modelData.metadataJson = JSON.stringify(enhancedMetadata, null, 2);
        parsedMetadata = enhancedMetadata;
      }
      
      // Create date/time filename like local export
      const now = new Date();
      const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const zipFileName = `model-${timestamp}.zip`;
      
      // Create a new zip file with proper structure
      const zip = new JSZip();
      
      // Add model.json
      zip.file('model.json', modelData.modelJson);
      
      // Add weights.bin (convert from base64 or array) - enhanced handling
      let weightsBlob;
      console.log('🔄 Starting weights blob creation...');
      console.log('📊 Input weights data:', {
        hasWeights: !!modelData.weightsBin,
        weightsLength: modelData.weightsBin ? modelData.weightsBin.length : 0,
        weightsType: typeof modelData.weightsBin
      });
      
      if (!modelData.weightsBin || modelData.weightsBin === '') {
        // Create empty weights blob if none provided
        weightsBlob = new Blob([], { type: 'application/octet-stream' });
        console.warn('⚠️ No weights data provided, creating empty weights.bin');
      } else if (typeof modelData.weightsBin === 'string') {
        try {
          // If it's a base64 string, convert to blob
          let base64Data = modelData.weightsBin;
          
          // Handle data URL format (data:application/octet-stream;base64,...)
          if (base64Data.startsWith('data:')) {
            const parts = base64Data.split(',');
            if (parts.length > 1) {
              base64Data = parts[1];
            }
          }
          
          const byteCharacters = atob(base64Data);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          weightsBlob = new Blob([byteArray], { type: 'application/octet-stream' });
          console.log('✅ Successfully converted base64 weights to blob');
          console.log('📊 Final weights blob info:', {
            size: weightsBlob.size,
            type: weightsBlob.type
          });
        } catch (base64Error) {
          console.error('❌ Failed to convert base64 weights:', base64Error);
          throw new Error(`Invalid base64 weights data: ${base64Error.message}`);
        }
      } else if (modelData.weightsBin instanceof Uint8Array) {
        // If it's already a Uint8Array
        weightsBlob = new Blob([modelData.weightsBin], { type: 'application/octet-stream' });
        console.log('✅ Using existing Uint8Array weights');
      } else if (modelData.weightsBin instanceof Blob) {
        // If it's already a Blob
        weightsBlob = modelData.weightsBin;
        console.log('✅ Using existing Blob weights');
      } else {
        console.error('❌ Invalid weights data format:', typeof modelData.weightsBin, modelData.weightsBin);
        throw new Error(`Invalid weights data format: ${typeof modelData.weightsBin}`);
      }
      
      zip.file('weights.bin', weightsBlob);
      
      // Add metadata.json
      zip.file('metadata.json', modelData.metadataJson);
      
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      
      // Create download link
      const url = URL.createObjectURL(zipBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = zipFileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      
      // Download complete - model is now available for user
      
      // Remove from downloading set after a short delay
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
      
      // Clean up download state for failed model
      
      // Remove from downloading set on error
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
        <p id="trained-models-description" className="text-sm text-gray-600">
          View and manage your trained AI models. Download models for offline use or select a model to load for predictions.
        </p>
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
                        {model.accuracy && (
                          <div className="text-green-600 font-semibold">
                            {Math.round(model.accuracy * 100)}%
                          </div>
                        )}
                        
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