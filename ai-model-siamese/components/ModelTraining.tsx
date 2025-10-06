//filepath: ai-model-siamese/components/ModelTraining.tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Progress } from '@/components/ui/progress';
import StudentSelectionModal from '@/components/model-training-ui/components/StudentSelectionModal';
import { useStudents } from '@/hooks/use-students';
import TrainingSetup from './TrainingSetup';
import Verification from './Verification';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { Student } from '@/types';
import { toast } from '@/hooks/use-toast';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Cloud, Download, FolderOpen } from 'lucide-react';

interface ModelTrainingProps {
  onModelTrained?: (model: any) => void;
  onVerification?: (result: any) => void;
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
}

export interface CustomModel {
  // Siamese model interface - will be implemented later
  predict: (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => Promise<PredictionResult[]>;
}

export interface ClassData {
  student: Student | null;
  color: string;
  samples: SampleData[];
  genuineSamples: SampleData[];
  forgedSamples: SampleData[];
}

export interface SampleData {
  thumbnail: string;
  timestamp: number;
  type?: 'genuine' | 'forged';
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({
  onModelTrained,
  onVerification
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
  const [isExporting, setIsExporting] = useState(false);
  const [classes, setClasses] = useState<ClassData[]>([
    { student: null, color: '#FF6B6B', samples: [], genuineSamples: [], forgedSamples: [] }
  ]);
  const [currentClassIndex, setCurrentClassIndex] = useState(0);
  const [newClassName, setNewClassName] = useState('');
  const [model, setModel] = useState<CustomModel | null>(null);
  const [maxPredictions, setMaxPredictions] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingAccuracy, setTrainingAccuracy] = useState<number | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [webcam, setWebcam] = useState<any>(null);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  
  // Models display state
  const [showModels, setShowModels] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [visiblePredictions, setVisiblePredictions] = useState(3);

  const { students, isLoading: isLoadingStudents } = useStudents();

  // Initialize TensorFlow
  useEffect(() => {
    const initTF = async () => {
      try {
        // Initialize TF for Siamese networks - will be implemented later
        console.log('Initializing TensorFlow for Siamese networks...');
      } catch (error) {
        console.error('Failed to initialize TensorFlow:', error);
      }
    };
    initTF();
  }, []);

  // Handle preview file upload
  const handlePreviewFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📤 File upload handler called');
    const file = event.target.files?.[0];
    if (!file) {
      console.log('❌ No file selected');
      return;
    }
    
    try {
      const imageUrl = URL.createObjectURL(file);
      console.log('✅ Created object URL:', imageUrl);
      
      setPreviewImage(imageUrl);
      console.log('✅ Set preview image');
      
      // Mock prediction for now
      const mockPredictions = [
        { className: 'Mock Student 1', confidence: 0.85 },
        { className: 'Mock Student 2', confidence: 0.72 },
        { className: 'Mock Student 3', confidence: 0.68 }
      ];
      
      console.log('✅ Mock predictions generated:', mockPredictions);
      
    } catch (error) {
      console.error('❌ Error processing file:', error);
    }
    
    event.target.value = '';
  }, []);

  // Handle file upload for specific class
  const handleFileUpload = async (classIndex: number, event: React.ChangeEvent<HTMLInputElement>, type: 'genuine' | 'forged' = 'genuine') => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newClasses = [...classes];
    let processedCount = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        const img = new Image();
        await new Promise<void>((resolve, reject) => {
          img.onload = async () => {
            try {
              const canvas = document.createElement('canvas');
              canvas.width = 224;
              canvas.height = 224;
              const ctx = canvas.getContext('2d');
              
              if (ctx) {
                ctx.drawImage(img, 0, 0, 224, 224);
                const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
                
                const sampleData = {
                  thumbnail,
                  timestamp: Date.now(),
                  type
                };
                
                if (type === 'genuine') {
                  newClasses[classIndex].genuineSamples.push(sampleData);
                } else {
                  newClasses[classIndex].forgedSamples.push(sampleData);
                }
                
                // Also add to main samples array for backward compatibility
                newClasses[classIndex].samples.push(sampleData);
                
                processedCount++;
              }
              resolve();
            } catch (error) {
              reject(error);
            }
          };
          
          img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
          img.src = URL.createObjectURL(file);
        });
      }
      
      setClasses(newClasses);
      console.log(`✅ Added ${processedCount} ${type} samples to class ${classIndex + 1}`);
      
    } catch (error) {
      console.error('❌ Error processing files:', error);
      toast({
        title: "Upload Error",
        description: "Failed to process some files. Please try again.",
        variant: "destructive",
      });
    }
    
    event.target.value = '';
  };

  // Add multiple students
  const addMultipleStudents = (students: Student[], samplesMap?: Map<string, SampleData[]>) => {
    if (students.length === 0) return;
    
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'];
    const newClasses = students.map((student, index) => ({
      student,
      color: colors[(classes.length + index) % colors.length],
      samples: samplesMap?.get(student.student_id) || [],
      genuineSamples: [],
      forgedSamples: []
    }));
    
    setClasses([...classes, ...newClasses]);
    setCurrentClassIndex(classes.length);
    
    toast({
      title: "Students Added",
      description: `Added ${students.length} students to training setup.`,
    });
  };

  // Remove class
  const removeClass = (index: number) => {
    if (classes.length <= 1) {
      toast({
        title: "Cannot Remove Class",
        description: "At least one class is required.",
        variant: "destructive",
      });
      return;
    }
    
    const newClasses = classes.filter((_, i) => i !== index);
    setClasses(newClasses);
    
    if (currentClassIndex >= newClasses.length) {
      setCurrentClassIndex(newClasses.length - 1);
    }
    
    toast({
      title: "Class Removed",
      description: "Class has been removed from training setup.",
    });
  };

  // Update class name (student)
  const updateClassName = (index: number, student: Student) => {
    const newClasses = [...classes];
    newClasses[index].student = student;
    setClasses(newClasses);
    
    toast({
      title: "Student Updated",
      description: `Class ${index + 1} is now assigned to ${student.firstname} ${student.surname}.`,
    });
  };

  // Train model (mock for now)
  const trainModel = async () => {
    const validClasses = classes.filter(cls => cls.student && (cls.genuineSamples.length > 0 || cls.forgedSamples.length > 0));
    
    if (validClasses.length < 1) {
      toast({
        title: "Training Error",
        description: "At least one class with samples is required for training.",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStartTime(Date.now());

    try {
      // Mock training process
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setTrainingProgress(i);
      }

      // Mock model creation
      const mockModel: CustomModel = {
        predict: async (image: HTMLCanvasElement | HTMLVideoElement) => {
          return [
            { className: 'Mock Student 1', confidence: 0.85 },
            { className: 'Mock Student 2', confidence: 0.72 }
          ];
        }
      };

      setModel(mockModel);
      setIsModelLoaded(true);
      setTrainingAccuracy(0.92); // Mock accuracy

      toast({
        title: "Training Complete",
        description: "Siamese model has been trained successfully!",
      });

      if (onModelTrained) {
        onModelTrained(mockModel);
      }

    } catch (error) {
      console.error('Training failed:', error);
      toast({
        title: "Training Failed",
        description: "An error occurred during training. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsTraining(false);
      setTrainingProgress(0);
    }
  };

  // Export handlers (mock for now)
  const exportToS3Handler = async () => {
    setIsUploading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Mock delay
      setHasExportedToCloud(true);
      toast({
        title: "Export Successful",
        description: "Model exported to cloud storage successfully!",
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export model to cloud storage.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const exportToLocalHandler = async () => {
    setIsDownloading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500)); // Mock delay
      setHasDownloadedToPC(true);
      toast({
        title: "Download Complete",
        description: "Model downloaded successfully!",
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "Failed to download model.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  };

  // Toggle models display
  const toggleModels = () => {
    setShowModels(!showModels);
  };

  // Show more predictions
  const showMorePredictions = () => {
    setVisiblePredictions(prev => Math.min(prev + 3, 10));
  };

  // Change model handler
  const handleChangeModel = () => {
    console.log('Change model clicked');
  };

  // Cloud model select
  const handleCloudModelSelect = () => {
    console.log('Cloud model select clicked');
  };

  // Local model select
  const handleLocalModelSelect = () => {
    console.log('Local model select clicked');
  };

  // Format student display
  const formatStudentDisplay = (student: Student): string => {
    return `${student.student_id} - ${student.firstname} ${student.surname}`;
  };

  // Handle verification
  const handleVerification = (result: any) => {
    console.log('Verification result:', result);
    if (onVerification) {
      onVerification(result);
    }
  };

  return (
    <div className="space-y-6">
      <input
        type="file"
        accept="image/*"
        multiple
        onChange={loadLocalModel}
        className="hidden"
        {...({ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
      />
      {isMobile ? (
        <div className="flex flex-col space-y-4">
          <Verification
            isMobile={isMobile}
            showModels={showModels}
            isLoadingModels={isLoadingModels}
            previewImage={previewImage}
            isWebcamActive={isWebcamActive}
            onToggleModels={toggleModels}
            onChangeModel={handleChangeModel}
            onCloudModelSelect={handleCloudModelSelect}
            onLocalModelSelect={handleLocalModelSelect}
            onHandlePreviewFileUpload={handlePreviewFileUpload}
            onVerifySignature={handleVerification}
          />
        </div>
      ) : (
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
          
          <Verification
            isMobile={isMobile}
            showModels={showModels}
            isLoadingModels={isLoadingModels}
            previewImage={previewImage}
            isWebcamActive={isWebcamActive}
            onToggleModels={toggleModels}
            onChangeModel={handleChangeModel}
            onCloudModelSelect={handleCloudModelSelect}
            onLocalModelSelect={handleLocalModelSelect}
            onHandlePreviewFileUpload={handlePreviewFileUpload}
            onVerifySignature={handleVerification}
          />
        </div>
      )}
    </div>
  );
};

export default ModelTraining;