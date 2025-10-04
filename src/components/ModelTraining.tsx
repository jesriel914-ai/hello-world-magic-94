//filepath: src\components\ModelTraining.tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tmImage from '@teachablemachine/image';
import JSZip from 'jszip';
import { Progress } from '@/components/ui/progress';
import StudentSelectionModal from './model-training-ui/components/StudentSelectionModal';
import { useStudents } from '@/hooks/use-students';
import { AIModelServiceClass, ModelMetadata, TrainingMetrics } from '@/lib/AIModelService';
import TrainingSetup from './model-training-ui/components/TrainingSetup';
import Preview from './model-training-ui/components/Preview';
import TrainedModelsForm from './model-training-ui/components/TrainedModelsForm';
import ExportConfirmationDialog from './model-training-ui/components/ExportConfirmationDialog';
import { exportModel, exportToS3, exportToLocal } from './model-training-ui/services/ModelExport';
import { loadLocalModel as loadLocalModelUtil, validateImportedModel, disposeImportedModel } from './model-training-ui/services/ModelImport';
import { ScreenShareService } from '../services/ScreenShareService';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { Student } from '@/types';
import { toast } from '@/hooks/use-toast';
import { augmentImage } from './model-training-ui/utils/augmentation';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Cloud, Download, FolderOpen } from 'lucide-react';

interface ModelTrainingProps {
  onModelTrained?: (model: CustomModel) => void;
  onClassification?: (predictions: PredictionResult[]) => void;
}

export interface PredictionResult {
  className: string;
  confidence: number;
}

export interface TrainedModel {
  id: string;
  student_id: string;
  student_name: string;
  sample_count: number;
  accuracy?: number | null;
  training_date: string;
  download_date?: string;
  training_metrics?: TrainingMetrics;
}

export interface CustomModel {
  featureExtractor: tf.LayersModel | null;
  classifier: tf.LayersModel | null;
  getTotalClasses: () => number;
  getClassLabels: () => string[];
  predict: (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => Promise<PredictionResult[]>;
}

export interface ClassData {
  student: Student | null;
  color: string;
  samples: SampleData[];
}

export interface SampleData {
  thumbnail: string;
  timestamp: number;
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({
  onModelTrained,
  onClassification
}) => {
  const isMobile = useMobileDetection();
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploaded, setHasUploaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [hasExportedToCloud, setHasExportedToCloud] = useState(false);
  const [hasDownloadedToPC, setHasDownloadedToPC] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [previousPredictions, setPreviousPredictions] = useState<PredictionResult[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [classes, setClasses] = useState<ClassData[]>([
    { student: null, color: '#FF6B6B', samples: [] }
  ]);
  const [currentClassIndex, setCurrentClassIndex] = useState(0);
  const [newClassName, setNewClassName] = useState('');
  const [model, setModel] = useState<CustomModel | null>(null);
  const [maxPredictions, setMaxPredictions] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingAccuracy, setTrainingAccuracy] = useState<number | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [webcam, setWebcam] = useState<tmImage.Webcam | null>(null);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  
  // Screen share service for real-time sync
  const [screenShareService] = useState(() => ScreenShareService.getInstance());
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  
  // Models display state
  const [showModels, setShowModels] = useState(false);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showTrainedModelsForm, setShowTrainedModelsForm] = useState(false);
  
  // Predictions display state
  const [visiblePredictions, setVisiblePredictions] = useState(3);
  
  // Dialog state management
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogTitle, setDialogTitle] = useState('');
  const [dialogMessage, setDialogMessage] = useState('');
  const [pendingDownloadAction, setPendingDownloadAction] = useState<(() => void) | null>(null);
  
  // Refs
  const localModelFileInputRef = useRef<HTMLInputElement>(null);
  
  const showMorePredictions = () => {
    setVisiblePredictions(prev => Math.min(prev + 1, predictions.length));
  };

  const logMemory = (label: string) => {
    try {
      const mem = tf.memory();
      console.log(`${label} tf.memory():`, mem);
    } catch {}
  };
  
  // Handle local model selection
  const handleLocalModelSelect = () => {
    if (localModelFileInputRef.current) {
      localModelFileInputRef.current.click();
    }
  };
  
  // Load local model from selected folder
  const loadLocalModel = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    try {
      console.log('📁 Loading local model files...');
      
      // Dispose of the current model if it exists to "erase its trace in the browser"
      if (model) {
        console.log('🧹 Disposing current model');
        if (model.featureExtractor) {
          model.featureExtractor.dispose();
        }
        if (model.classifier) {
          model.classifier.dispose();
        }
        setModel(null);
        setIsModelLoaded(false);
      }
      
      // Use the new ModelImport utility to load the model
      const importedModel = await loadLocalModelUtil(files);
      
      // Validate the imported model
      const isValid = validateImportedModel(importedModel);
      if (!isValid) {
        throw new Error('Imported model validation failed. The model structure may be corrupted.');
      }
      
      console.log('✅ Model validation passed');
      
      // Create a CustomModel from the loaded models
      const customModel: CustomModel = {
        featureExtractor: importedModel.featureExtractor,
        classifier: importedModel.classifier,
        getTotalClasses: () => importedModel.metadata.labels.length,
        getClassLabels: () => importedModel.metadata.labels,
        predict: async (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => {
          const { results } = await tf.tidy(async () => {
            const img = tf.browser.fromPixels(image)
              .resizeNearestNeighbor([224, 224])
              .toFloat()
              .div(255.0)
              .expandDims();
            const processedImg = flipped ? img.reverse(1) : img;
            const features = importedModel.featureExtractor.predict(processedImg) as tf.Tensor;
            const predictions = importedModel.classifier.predict(features) as tf.Tensor;
            const predictionArray = await predictions.data();
            const labels = importedModel.metadata.labels;
            const results: PredictionResult[] = [];
            for (let i = 0; i < predictionArray.length; i++) {
              if (i < labels.length) {
                results.push({ className: labels[i], confidence: predictionArray[i] });
              }
            }
            return { results };
          });
          await tf.nextFrame();
          results.sort((a, b) => b.confidence - a.confidence);
          return results;
        }
      };
      
      // Set the new model as active
      setModel(customModel);
      setIsModelLoaded(true);
      
      // Share model status with other devices
      if (!isMobile) {
        screenShareService.shareModelStatus(true, customModel.getClassLabels());
      }
      
      // Update UI state
      setMaxPredictions(importedModel.metadata.labels.length);
      
      // Show success notification
      toast({
        title: 'Local Model Loaded',
        description: `Successfully loaded model with ${importedModel.metadata.labels.length} classes. Accuracy: ${(importedModel.metadata.userMetadata.accuracy * 100).toFixed(2)}%`,
      });
      
      console.log('✅ Local model loading completed successfully');
      
    } catch (error) {
      console.error('Error loading local model:', error);
      toast({
        title: 'Error Loading Model',
        description: error instanceof Error ? error.message : 'Failed to load local model',
        variant: 'destructive',
      });
    } finally {
      // Reset the file input
      if (event.target) {
        event.target.value = '';
      }
    }
  };
  
  
  // Models functions
  const loadTrainedModels = async () => {
    setIsLoadingModels(true);
    try {
      const aiModelService = new AIModelServiceClass();
      const models = await aiModelService.getTrainedModels();
      setTrainedModels(models);
      setIsLoadingModels(false);
    } catch (error) {
      console.error('❌ Error loading trained models:', error);
      setIsLoadingModels(false);
    }
  };

  const toggleModels = () => {
    if (!showModels && trainedModels.length === 0) {
      loadTrainedModels();
    }
    setShowModels(!showModels);
  };

  // Handle model selection from TrainedModelsForm
  const handleModelSelect = async (selectedModel: TrainedModel) => {
    try {
      console.log('🎯 Selected model:', selectedModel);
      
      // Dispose of the current model if it exists to "erase its trace in the browser"
      if (model) {
        console.log('🧹 Disposing current model');
        if (model.featureExtractor) {
          model.featureExtractor.dispose();
        }
        if (model.classifier) {
          model.classifier.dispose();
        }
        setModel(null);
        setIsModelLoaded(false);
      }
      
      // Load the selected model using AIModelService
      const aiModelService = new AIModelServiceClass();
      const loadResult = await aiModelService.loadModel(selectedModel.id);
      
      if (!loadResult.success || !loadResult.model) {
        throw new Error(loadResult.error || 'Failed to load model');
      }
      
      console.log('✅ Model loaded successfully:', loadResult.model);
      
      // Create a CustomModel from the loaded model
      const loadedModelData = loadResult.model;
      const customModel: CustomModel = {
        // For loaded models, we'll create null featureExtractor and classifier
        // since the actual model data is stored differently
        featureExtractor: null,
        classifier: null,
        getTotalClasses: () => loadedModelData.getTotalClasses(),
        getClassLabels: () => loadedModelData.getClassLabels(),
        predict: async (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => {
          // Use the loaded model's predict function
          return loadedModelData.predict(image, flipped);
        }
      };
      
      // Set the new model as active
      setModel(customModel);
      setIsModelLoaded(true);
      
      // Share model status with other devices
      if (!isMobile) {
        screenShareService.shareModelStatus(true, customModel.getClassLabels());
      }
      
      // Update UI state
      setMaxPredictions(loadedModelData.getTotalClasses());
      
      // Show success notification
      toast({
        title: 'Model Loaded',
        description: `Successfully loaded model with ${loadedModelData.getTotalClasses()} students`,
      });
      
      // Call the onModelTrained callback if provided
      if (onModelTrained) {
        onModelTrained(customModel);
      }
      
    } catch (error) {
      console.error('❌ Error loading selected model:', error);
      toast({
        title: 'Error Loading Model',
        description: error instanceof Error ? error.message : 'Failed to load model',
        variant: 'destructive',
      });
    }
  };

  // Handle cloud model selection (opens the trained models form)
  const handleCloudModelSelect = () => {
    setShowTrainedModelsForm(true);
    
    // Always load trained models when opening the form to get the latest models
    loadTrainedModels();
  };
  
  // Legacy function for backward compatibility
  const handleChangeModel = () => {
    handleCloudModelSelect();
  };

  // Helper function for training process
  const formatStudentDisplay = (student: Student): string => {
    return `${student.student_id} - ${student.firstname} ${student.surname}`;
  };

  // Dialog helper functions
  const showDownloadConfirmDialog = (action: () => void) => {
    setDialogTitle('Download Model Again');
    setDialogMessage('This model has already been downloaded to your PC. Do you want to download it again?');
    setPendingDownloadAction(() => action);
    setDialogOpen(true);
  };

  const handleDialogConfirm = () => {
    if (pendingDownloadAction) {
      pendingDownloadAction();
    }
    setDialogOpen(false);
    setPendingDownloadAction(null);
  };

  const handleDialogCancel = () => {
    setDialogOpen(false);
    setPendingDownloadAction(null);
  };

  // Success notification functions
  const showCloudSuccessNotification = () => {
    toast({
      title: 'Upload Successful',
      description: 'Model uploaded successfully to S3!',
    });
  };

  const showDownloadSuccessNotification = () => {
    toast({
      title: 'Download Successful',
      description: 'Model downloaded successfully!',
    });
  };

  const showLocalExportSuccessNotification = (fileName: string) => {
    toast({
      title: 'Export Successful',
      description: `Model exported successfully!\n\nFile downloaded: ${fileName}`,
    });
  };

  const updateClassStudent = (classIndex: number, student: Student | null) => {
    const newClasses = [...classes];
    newClasses[classIndex].student = student;
    setClasses(newClasses);
  };
  
  
  
  const removeClass = (classIndex: number) => {
    if (classes.length <= 1) return;
    
    const newClasses = classes.filter((_, index) => index !== classIndex);
    setClasses(newClasses);
    
    if (currentClassIndex >= newClasses.length) {
      setCurrentClassIndex(newClasses.length - 1);
    }
  };

const addMultipleStudents = (students: Student[], samplesMap?: Map<string, SampleData[]>) => {
  const existingStudentIds = classes.map(cls => cls.student?.id).filter(id => id !== undefined);
  const newStudents = students.filter(student => !existingStudentIds.includes(student.id));
  
  if (newStudents.length === 0) {
    toast({
      title: 'No New Students',
      description: 'All selected students are already added.',
      variant: 'destructive',
    });
    return;
  }
  
  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'];
  const newClasses = newStudents.map((student, index) => ({
    student,
    color: colors[(classes.length + index) % colors.length],
    samples: samplesMap?.get(student.student_id) || []
  }));
  
  setClasses([...classes, ...newClasses]);
  setCurrentClassIndex(classes.length);
  
  const skippedCount = students.length - newStudents.length;
  
  // Show toast notification instead of alert
  if (skippedCount > 0) {
    toast({
      title: 'Students Added',
      description: `Added ${newStudents.length} student${newStudents.length !== 1 ? 's' : ''}. ${skippedCount} student${skippedCount !== 1 ? 's were' : ' was'} already added and skipped.`,
    });
  } else {
    toast({
      title: 'Students Added',
      description: `Successfully added ${newStudents.length} student${newStudents.length !== 1 ? 's' : ''}.`,
    });
  }
};
  // Update class student (replaces updateClassName)
  const updateClassName = (index: number, student: Student | null) => {
    // If student is null (clearing selection), allow it
    if (!student) {
      const newClasses = [...classes];
      newClasses[index].student = null;
      setClasses(newClasses);
      return;
    }
    
    // Check if student is already selected in another class
    const existingClassIndex = classes.findIndex((cls, i) => 
      i !== index && cls.student?.id === student.id
    );
    
    if (existingClassIndex !== -1) {
      alert(`This student is already selected in class ${existingClassIndex + 1}. Please select a different student.`);
      return;
    }
    
    const newClasses = [...classes];
    newClasses[index].student = student;
    setClasses(newClasses);
  };
  

  // Initialize TensorFlow.js
  useEffect(() => {
    const initTF = async () => {
      await tf.ready();
      try {
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('webgl');
        }
      } catch (e) {
        console.warn('Failed to set webgl backend, continuing with default:', e);
      }
      console.log('TensorFlow.js is ready');
      console.log('Backend:', tf.getBackend());
      console.log('Initial tf.memory():', tf.memory());
    };
    initTF();
  }, []);

  // Initialize ScreenShareService
  useEffect(() => {
    console.log('🚀 Initializing ScreenShareService...');
    screenShareService.initialize(!isMobile);
    
    // Set up event handlers for screen share updates
    screenShareService.onPreviewUpdateHandler((data) => {
      console.log('📱 Received preview update from', data.source);
      setPreviewImage(data.imageData);
      
      // If the image is from mobile and we have a model, trigger prediction
      if (data.source === 'mobile' && model) {
        console.log('🎯 Mobile image received, triggering prediction...');
        
        // Create image element from the received image data
        const img = new Image();
        img.onload = async () => {
          try {
            console.log('🖼️ Mobile image loaded, dimensions:', img.width, 'x', img.height);
            
            // Create canvas for processing
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            
            if (ctx) {
              ctx.drawImage(img, 0, 0, 224, 224);
              console.log('✅ Canvas created and mobile image drawn');
              
              // Make predictions
              setIsPredicting(true);
              screenShareService.sharePredictingStatus(true);
              
              try {
                logMemory('Before mobile predict');
                const predictions = await model.predict(canvas, true);
                const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
                console.log('🎯 Mobile image predictions completed:', sortedPredictions);
                setPredictions(sortedPredictions);
                logMemory('After mobile predict');
                
                // Share prediction results with mobile
                screenShareService.sharePredictionResults(sortedPredictions);
                console.log('📤 Mobile prediction results shared via ScreenShareService');
              } catch (error) {
                console.error('❌ Error getting predictions for mobile image:', error);
                setPredictions([]);
              } finally {
                setIsPredicting(false);
                screenShareService.sharePredictingStatus(false);
              }
            } else {
              console.error('❌ Could not get canvas context for mobile image');
            }
          } catch (error) {
            console.error('❌ Error processing mobile image for prediction:', error);
          }
        };
        
        img.onerror = () => {
          console.error('❌ Failed to load mobile image for prediction');
        };
        
        img.src = data.imageData;
      }
    });
    
    screenShareService.onPredictionUpdateHandler((data) => {
      console.log('📱 Received prediction update from', data.source);
      setPredictions(data.results);
    });
    
    screenShareService.onPredictingUpdateHandler((data) => {
      console.log('📱 Received predicting status from', data.source);
      setIsPredicting(data.isPredicting);
    });
    
    screenShareService.onModelStatusUpdateHandler((data) => {
      console.log('📱 Received model status from desktop');
      setIsModelLoaded(data.isModelLoaded);
    });
    
    screenShareService.onConnectionStatusHandler((connected) => {
      console.log('📱 WebSocket connection status:', connected);
      setIsWebSocketConnected(connected);
    });
    
    // Cleanup on unmount
    return () => {
      console.log('🔄 Shutting down ScreenShareService...');
      screenShareService.shutdown();
    };
  }, [isMobile, screenShareService, model]);



  // Handle preview file upload
  const handlePreviewFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📤 File upload handler called');
    const file = event.target.files?.[0];
    if (!file) {
      console.log('❌ No file selected');
      return;
    }

    console.log('📁 File selected:', file.name, file.type, file.size);
    
    try {
      // Create preview image URL directly
      const imageUrl = URL.createObjectURL(file);
      console.log('✅ Created object URL:', imageUrl);
      
      // Set preview image immediately
      setPreviewImage(imageUrl);
      console.log('✅ Set preview image');
      
      // Convert image to base64 for sharing via WebSocket
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64Data = e.target?.result as string;
        console.log('✅ Converted image to base64, length:', base64Data.length);
        
        // Share preview image with other devices via ScreenShareService
        screenShareService.sharePreviewImage(base64Data);
        console.log('📤 Preview image shared via ScreenShareService');
      };
      reader.readAsDataURL(file);
      
      // Create image element for processing
      const img = new Image();
      
      img.onload = async () => {
        try {
          console.log('🖼️ Image loaded, dimensions:', img.width, 'x', img.height);
          
          // Create canvas for processing
          const canvas = document.createElement('canvas');
          canvas.width = 224;
          canvas.height = 224;
          const ctx = canvas.getContext('2d');
          
          if (ctx) {
            ctx.drawImage(img, 0, 0, 224, 224);
            console.log('✅ Canvas created and image drawn');
            
            // If we have a trained model, make predictions
            if (model) {
              console.log('✅ Model found, making predictions...');
              setIsPredicting(true);
              screenShareService.sharePredictingStatus(true);
              
              try {
                logMemory('Before file predict');
                const predictions = await model.predict(canvas, true);
                const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
                console.log('🎯 Predictions completed:', sortedPredictions);
                setPredictions(sortedPredictions);
                logMemory('After file predict');
                
                // Share prediction results with other devices
                screenShareService.sharePredictionResults(sortedPredictions);
                console.log('📤 Prediction results shared via ScreenShareService');
              } catch (error) {
                console.error('❌ Error getting predictions:', error);
                setPredictions([]);
              } finally {
                setIsPredicting(false);
                screenShareService.sharePredictingStatus(false);
              }
            } else {
              console.log('ℹ️ No model available for prediction');
              setPredictions([]);
            }
          } else {
            console.error('❌ Could not get canvas context');
          }
          
          // Clean up the object URL
          URL.revokeObjectURL(imageUrl);
        } catch (error) {
          console.error('❌ Error processing preview file:', error);
        }
      };
      
      img.onerror = () => {
        console.error('❌ Failed to load image');
        URL.revokeObjectURL(imageUrl);
      };
      
      img.src = imageUrl;
    } catch (error) {
      console.error('❌ Error creating object URL:', error);
    }
    
    // Reset file input
    event.target.value = '';
  }, [model, screenShareService]);

  // Handle file upload for specific class
  const handleFileUpload = async (classIndex: number, event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newClasses = [...classes];
    let processedCount = 0;

    try {
      // Process all selected files
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        const img = new Image();
        await new Promise<void>((resolve, reject) => {
          img.onload = async () => {
            try {
              // Create canvas for processing
              const canvas = document.createElement('canvas');
              canvas.width = 224;
              canvas.height = 224;
              const ctx = canvas.getContext('2d');
              
              if (ctx) {
                ctx.drawImage(img, 0, 0, 224, 224);
                
                // Get thumbnail as data URL
                const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
                
                // Add sample to class
                newClasses[classIndex].samples.push({
                  thumbnail,
                  timestamp: Date.now()
                });
                
                processedCount++;
              }
              resolve();
            } catch (error) {
              console.error(`Error processing file ${file.name}:`, error);
              reject(error);
            }
          };
          
          img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
          img.src = URL.createObjectURL(file);
        });
      }
      
      // Update classes after all files are processed
      setClasses(newClasses);
      console.log(`Added ${processedCount} file samples for ${classes[classIndex].student ? formatStudentDisplay(classes[classIndex].student) : 'Unassigned'}`);
      
    } catch (error) {
      console.error('Error processing files:', error);
    }
    
    // Reset file input
    event.target.value = '';
  };









// Training Configuration - Optimized for R5 3400G (Vega 11 iGPU)
const AUGMENTATION_COUNT = 2; // 

const trainModel = async () => {
  const validClasses = classes.filter(cls => cls.samples.length > 0);
  if (validClasses.length < 2) {
    alert('Please add samples for at least 2 classes!');
    return;
  }

  setIsTraining(true);
  setTrainingProgress(0);
  setTrainingStartTime(Date.now());
  
  setHasExportedToCloud(false);
  setHasDownloadedToPC(false);
  
  try {
    console.log(`Starting training with ${AUGMENTATION_COUNT}x augmentation...`);
    logMemory('Before training');
    
    // CRITICAL: Dispose previous model completely
    if (model) {
      console.log('Disposing previous model before training');
      if (model.featureExtractor) {
        model.featureExtractor.dispose();
      }
      if (model.classifier) {
        model.classifier.dispose();
      }
      setModel(null);
      await tf.nextFrame();
      await tf.nextFrame(); // Double wait for complete cleanup
    }
    
    // Force garbage collection
    if (tf.memory().numTensors > 10) {
      console.warn('High tensor count before training:', tf.memory().numTensors);
      await tf.disposeVariables();
      await tf.nextFrame();
    }
    
    // Load MobileNet
    const mobilenet = await tmImage.loadTruncatedMobileNet({
      version: 2,
      alpha: 0.5,
      inputResolution: 224,
      classifierInputSize: 1024
    });
    
    logMemory('After loading MobileNet');
    
    const FEATURE_SIZE = 1280;
    
    // OPTIMIZED classifier for limited VRAM (R5 3400G)
    const classifierModel = tf.sequential({
      layers: [
        // Reduced architecture to fit in iGPU memory
        tf.layers.dense({ 
          units: 256, // 
          activation: 'relu', 
          inputShape: [FEATURE_SIZE],
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }),
        tf.layers.dropout({ rate: 0.5 }),
        
        tf.layers.dense({ 
          units: 128, // 
          activation: 'relu',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }),
        tf.layers.dropout({ rate: 0.4 }),
        
        // Output layer
        tf.layers.dense({ 
          units: validClasses.length, 
          activation: 'softmax' 
        })
      ]
    });

    classifierModel.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    logMemory('After creating classifier');
    
    // CRITICAL: Extract features in batches with immediate cleanup
    console.log('Extracting features with memory-efficient batching...');
    
    const totalSamples = validClasses.reduce((sum, cls) => sum + cls.samples.length, 0);
    const totalWithAug = totalSamples * (AUGMENTATION_COUNT + 1);
    
    // We'll collect features into a single stacked tensor at the end
    const allFeatures: number[][] = [];
    const allLabels: number[] = [];
    
    let processedSamples = 0;
    const CLEANUP_INTERVAL = 10; // Clean up every 10 samples
    
    for (const [classIndex, cls] of validClasses.entries()) {
      console.log(`Processing class ${classIndex + 1}/${validClasses.length}: ${cls.samples.length} samples`);
      
      for (const sample of cls.samples) {
        const img = new Image();
        img.src = sample.thumbnail;
        
        await new Promise<void>((resolve) => {
          img.onload = async () => {
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            
            if (ctx) {
              ctx.drawImage(img, 0, 0, 224, 224);
              
              // Extract features and IMMEDIATELY convert to array (dispose tensor)
              const featuresArray = await tf.tidy(() => {
                const imageTensor = tf.browser.fromPixels(canvas)
                  .resizeNearestNeighbor([224, 224])
                  .toFloat();
                
                const normalized = imageTensor.sub(127.5).div(127.5);
                const batched = normalized.expandDims(0);
                const extracted = mobilenet.predict(batched) as tf.Tensor;
                const squeezed = extracted.squeeze([0]);
                
                // Convert to array synchronously inside tidy
                return squeezed.arraySync() as number[];
              });
              
              allFeatures.push(featuresArray);
              allLabels.push(classIndex);
              
              // Process augmented samples
              for (let aug = 0; aug < AUGMENTATION_COUNT; aug++) {
                const augCanvas = augmentImage(canvas);
                
                const augFeaturesArray = await tf.tidy(() => {
                  const augTensor = tf.browser.fromPixels(augCanvas)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat();
                  const augNormalized = augTensor.sub(127.5).div(127.5);
                  const augBatched = augNormalized.expandDims(0);
                  const augExtracted = mobilenet.predict(augBatched) as tf.Tensor;
                  const augSqueezed = augExtracted.squeeze([0]);
                  
                  return augSqueezed.arraySync() as number[];
                });
                
                allFeatures.push(augFeaturesArray);
                allLabels.push(classIndex);
                
                // Clean up augmented canvas immediately
                augCanvas.remove();
              }
              
              // Clean up original canvas
              canvas.remove();
            }
            
            processedSamples++;
            
            // Update progress (0-75% for feature extraction)
            const progress = Math.min(75, (processedSamples / totalSamples) * 75);
            setTrainingProgress(progress);
            
            // Periodic aggressive cleanup
            if (processedSamples % CLEANUP_INTERVAL === 0) {
              await tf.nextFrame();
              await tf.nextFrame();
              console.log(`✓ Processed ${processedSamples}/${totalSamples} samples | Tensors: ${tf.memory().numTensors}`);
            }
            
            resolve();
          };
          
          img.onerror = () => {
            console.error('Failed to load image');
            processedSamples++;
            resolve();
          };
        });
      }
    }
    
    if (allFeatures.length === 0) {
      throw new Error('No training data extracted');
    }
    
    console.log(`Total training samples: ${allFeatures.length} (${totalSamples} original × ${AUGMENTATION_COUNT + 1})`);
    logMemory('After feature extraction (all in arrays)');
    
    // Now convert arrays to tensors for training (one-time conversion)
    console.log('Converting features to tensors for training...');
    const xs = tf.tensor2d(allFeatures);
    const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), validClasses.length);
    
    // Clear the arrays to free memory
    allFeatures.length = 0;
    allLabels.length = 0;
    
    console.log('Training data shape:', xs.shape);
    console.log('Labels shape:', ys.shape);
    
    logMemory('After tensor conversion');
    
    // Training configuration - optimized for your hardware
    const epochs = 50; 
    const effectiveDatasetSize = xs.shape[0];
    const batchSize = Math.min(8, Math.floor(effectiveDatasetSize / 10));
    
    console.log(`Training for ${epochs} epochs with batch size ${batchSize}...`);
    
    const history = await classifierModel.fit(xs, ys, {
      epochs: epochs,
      batchSize: batchSize,
      validationSplit: 0.1,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (logs) {
            const trainAcc = (logs.acc * 100).toFixed(2);
            const valAcc = logs.val_acc ? (logs.val_acc * 100).toFixed(2) : 'N/A';
            console.log(
              `Epoch ${epoch + 1}/${epochs}: ` +
              `loss=${logs.loss?.toFixed(4)}, ` +
              `acc=${trainAcc}%, ` +
              `val_loss=${logs.val_loss?.toFixed(4)}, ` +
              `val_acc=${valAcc}%`
            );
            
            const progress = 75 + ((epoch + 1) / epochs) * 25;
            setTrainingProgress(progress);
          }
          
          // Memory cleanup every 10 epochs
          if ((epoch + 1) % 10 === 0) {
            await tf.nextFrame();
            logMemory(`After epoch ${epoch + 1}`);
          }
        }
      }
    });
    
    // Get final metrics
    const finalTrainAcc = history.history.acc[history.history.acc.length - 1] as number;
    const finalValAcc = history.history.val_acc 
      ? history.history.val_acc[history.history.val_acc.length - 1] as number
      : finalTrainAcc;
    
    setTrainingAccuracy(finalValAcc);
    
    console.log('═══════════════════════════════════════');
    console.log('Training Summary:');
    console.log(`  Classes: ${validClasses.length}`);
    console.log(`  Original samples: ${totalSamples}`);
    console.log(`  Total samples (with ${AUGMENTATION_COUNT}x aug): ${effectiveDatasetSize}`);
    console.log(`  Final train accuracy: ${(finalTrainAcc * 100).toFixed(2)}%`);
    console.log(`  Final validation accuracy: ${(finalValAcc * 100).toFixed(2)}%`);
    console.log('═══════════════════════════════════════');
    
    // CRITICAL: Clean up training tensors immediately
    xs.dispose();
    ys.dispose();
    
    await tf.nextFrame();
    await tf.nextFrame();
    
    logMemory('After training cleanup');
    
    // Create the custom model
    const customModel: CustomModel = {
      featureExtractor: mobilenet,
      classifier: classifierModel,
      getTotalClasses: () => validClasses.length,
      getClassLabels: () => validClasses.map(cls => 
        cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
      ),
      predict: async (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => {
        const results = await tf.tidy(() => {
          let imageTensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat();
          
          if (flipped) {
            imageTensor = imageTensor.reverse(1);
          }
          
          const normalized = imageTensor.sub(127.5).div(127.5);
          const batched = normalized.expandDims(0);
          const features = mobilenet.predict(batched) as tf.Tensor;
          const predictions = classifierModel.predict(features) as tf.Tensor;
          const predData = predictions.dataSync();
          
          const labels = validClasses.map(cls => 
            cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
          );
          
          const results: PredictionResult[] = [];
          for (let i = 0; i < predData.length; i++) {
            results.push({
              className: labels[i],
              confidence: predData[i]
            });
          }
          
          results.sort((a, b) => b.confidence - a.confidence);
          return results;
        });
        
        await tf.nextFrame();
        return results;
      }
    };
    
    setTrainingProgress(100);
    setModel(customModel);
    setMaxPredictions(validClasses.length);
    setIsModelLoaded(true);
    setHasUploaded(false);
    
    if (!isMobile) {
      screenShareService.shareModelStatus(true, customModel.getClassLabels());
    }
    
    logMemory('After training complete');
    console.log('Model trained successfully!');
    
    if (onModelTrained) {
      onModelTrained(customModel);
    }
    
  } catch (error) {
    console.error('Training error:', error);
    alert('Error training model: ' + error);
  } finally {
    setIsTraining(false);
    setTimeout(() => setTrainingProgress(0), 2000);
  }
};
 











  // Upload trained model to S3
  const uploadModelToS3 = async () => {
    if (!model || !isModelLoaded) {
      alert('Please train a model first before uploading to S3');
      return;
    }
    
    try {
      setIsUploading(true);
      
      // Get the current class student ID
      const currentClass = classes[currentClassIndex];
      const studentId = currentClass?.student?.id;
      
      if (!studentId) {
        alert('Please select a student for the current class before uploading');
        return;
      }
      
      // Create a temporary model ID for the upload (in a real app, this would come from the backend)
      const tempModelId = `model_${Date.now()}`;
      
      // Calculate total sample count from all classes
      const validClasses = classes.filter(cls => cls.samples.length > 0);
      const totalSampleCount = validClasses.reduce((total, cls) => total + cls.samples.length, 0);
      
      // Prepare student information for global model
      const students = validClasses.map(cls => ({
        id: cls.student?.id?.toString() || '',
        student_id: cls.student?.student_id || '',
        firstname: cls.student?.firstname || '',
        surname: cls.student?.surname || '',
        full_name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
      }));
      
      // Prepare training data
      const trainingData = {
        total_sample_count: totalSampleCount,
        student_count: validClasses.length,
        students: students,
        accuracy: trainingAccuracy || 0.85, // Use actual training accuracy if available
        epochs: 50,
        optimizer: 'adam',
        learning_rate: 0.001,
        batch_size: Math.min(16, totalSampleCount),
        training_summary: `Trained on ${validClasses.length} students with ${totalSampleCount} total samples. Final accuracy: ${(trainingAccuracy || 0.85).toFixed(4)}`,
        model_architecture: 'cnn'
      };
      
      // Upload to S3 using AIModelService
      const aiModelService = new AIModelServiceClass();
      const uploadResult = await aiModelService.uploadTrainedModelToS3(
        tempModelId,
        model,
        studentId.toString(),
        trainingData
      );
      
      if (uploadResult.success) {
        setHasExportedToCloud(true); // Mark as exported to cloud
        showCloudSuccessNotification();
      } else {
        throw new Error(uploadResult.message);
      }
      
    } catch (error) {
      console.error('S3 upload error:', error);
      alert('Error uploading model to S3: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsUploading(false);
    }
  };


  // Download trained model to local PC
  const downloadModelToLocal = async () => {
    if (!model || !isModelLoaded) {
      alert('Please train a model first before downloading');
      return;
    }
    if (hasDownloadedToPC) {
      showDownloadConfirmDialog(() => exportToLocalHandler());
      return;
    }
    await exportToLocalHandler();
  };





  // Reset training data
  const resetTraining = () => {
    setClasses(classes.map(cls => ({ ...cls, samples: [] })));
    setModel(null);
    setIsModelLoaded(false);
    setPredictions([]);
    setTrainingProgress(0);
  };

  // Reset model
  const resetModel = () => {
    if (model) {
      try {
        if (model.featureExtractor) model.featureExtractor.dispose();
        if (model.classifier) model.classifier.dispose();
      } catch (e) {
        console.warn('Error disposing model on reset:', e);
      }
    }
    setModel(null);
    setIsModelLoaded(false);
    setPredictions([]);
    setTrainingProgress(0);
    setTrainingAccuracy(null);
  };

  // Dispose models on unmount
  useEffect(() => {
    return () => {
      if (model) {
        try {
          if (model.featureExtractor) model.featureExtractor.dispose();
          if (model.classifier) model.classifier.dispose();
        } catch (e) {
          console.warn('Error disposing model on unmount:', e);
        }
      }
      console.log('Final tf.memory at unmount:', tf.memory());
    };
  }, [model]);

  // Export model
  const exportModelHandler = async () => {
    await exportModel({
      model,
      classes,
      currentClassIndex,
      trainingAccuracy,
      trainingStartTime,
      formatStudentDisplay,
      setIsExporting,
      setIsDownloading,
      setHasExportedToCloud,
      setHasDownloadedToPC,
      showCloudSuccessNotification,
      showLocalExportSuccessNotification
    });
  };

  // Export to S3
  const exportToS3Handler = async () => {
    await exportToS3({
      model,
      classes,
      currentClassIndex,
      trainingAccuracy,
      trainingStartTime,
      formatStudentDisplay,
      setHasExportedToCloud,
      showCloudSuccessNotification
    });
  };

  // Export to local download
  const exportToLocalHandler = async () => {
    await exportToLocal({
      model,
      classes,
      currentClassIndex,
      trainingAccuracy,
      trainingStartTime,
      formatStudentDisplay,
      setIsDownloading,
      setHasDownloadedToPC,
      showLocalExportSuccessNotification
    });
  };

  // Import model
  const importModel = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const metadata = JSON.parse(e.target?.result as string);
        
        // Validate metadata structure
        if (!metadata.classes || !metadata.modelArchitecture) {
          throw new Error('Invalid model file');
        }
        
        // Update classes based on imported metadata
        const importedClasses = metadata.classes.map((cls: {name: string, color?: string}) => ({
          student: null, // Import doesn't have student data, so set to null
          color: cls.color || `#${Math.floor(Math.random()*16777215).toString(16)}`,
          samples: []
        }));
        
        setClasses(importedClasses);
        setCurrentClassIndex(0);
        
        alert('Model metadata imported successfully!\n\nNote: You will need to retrain the model with your samples.');
        
      } catch (error) {
        console.error('Import error:', error);
        alert('Error importing model: Invalid file format');
      }
    };
    
    reader.readAsText(file);
  };

  return (
    <div className="w-full">
      {/* Hidden file input for local model selection */}
      <input
        ref={localModelFileInputRef}
        type="file"
        multiple
        accept=".json,.bin"
        onChange={loadLocalModel}
        className="hidden"
        {...({ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
      />
      {isMobile ? (
        // Mobile layout - Only Preview component with integrated webcam
        <div className="flex flex-col space-y-4">
          {/* Preview component for mobile */}
          <Preview
            isMobile={isMobile}
            showModels={showModels}
            isLoadingModels={isLoadingModels}
            trainedModels={trainedModels}
            predictions={predictions}
            visiblePredictions={visiblePredictions}
            previewImage={previewImage}
            isWebcamActive={isWebcamActive}
            webcam={webcam}
            model={model}
            onToggleModels={toggleModels}
            onShowMorePredictions={showMorePredictions}
            onHandlePreviewFileUpload={handlePreviewFileUpload}
            onChangeModel={handleChangeModel}
            onCloudModelSelect={handleCloudModelSelect}
            onLocalModelSelect={handleLocalModelSelect}
          />
        </div>
      ) : (
        // Desktop layout - All components
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <TrainingSetup
            classes={classes}
            isTraining={isTraining}
            isModelLoaded={isModelLoaded}
            trainingProgress={trainingProgress}
            isUploading={isUploading}
            hasUploaded={hasUploaded}
            isDownloading={isDownloading}
            hasExportedToCloud={hasExportedToCloud}
            hasDownloadedToPC={hasDownloadedToPC}
            onRemoveClass={removeClass}
            onUpdateClassName={updateClassName}
            onAddMultipleStudents={addMultipleStudents}
            onHandleFileUpload={handleFileUpload}
            onTrainModel={trainModel}
            onUploadModelToS3={exportToS3Handler}
            onDownloadModelToLocal={exportToLocalHandler}
            formatStudentDisplay={formatStudentDisplay}
          />
          
          <Preview
            isMobile={isMobile}
            showModels={showModels}
            isLoadingModels={isLoadingModels}
            trainedModels={trainedModels}
            predictions={predictions}
            visiblePredictions={visiblePredictions}
            previewImage={previewImage}
            isWebcamActive={isWebcamActive}
            webcam={webcam}
            model={model}
            onToggleModels={toggleModels}
            onShowMorePredictions={showMorePredictions}
            onHandlePreviewFileUpload={handlePreviewFileUpload}
            onChangeModel={handleChangeModel}
            onCloudModelSelect={handleCloudModelSelect}
            onLocalModelSelect={handleLocalModelSelect}
          />
        </div>
      )}
      
      {/* Export Confirmation Dialog */}
      <ExportConfirmationDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        title={dialogTitle}
        message={dialogMessage}
        onConfirm={handleDialogConfirm}
        onCancel={handleDialogCancel}
      />
      
      {/* Trained Models Form Dialog */}
      <TrainedModelsForm
        open={showTrainedModelsForm}
        onOpenChange={setShowTrainedModelsForm}
        trainedModels={trainedModels}
        isLoadingModels={isLoadingModels}
        onModelSelect={handleModelSelect}
      />
    </div>
  );
};

export default ModelTraining;
