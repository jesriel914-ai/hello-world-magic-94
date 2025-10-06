//filepath: ai-model-siamese/components/ModelTraining.tsx
import React, { useState } from 'react';
import TrainingSetup from './TrainingSetup';
import Verification from './Verification';
import useMobileDetection from '@/hooks/use-mobile-detection';
import type { Student } from '@/types';

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
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  
  // Models display state
  const [showModels, setShowModels] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);

  // Handle preview file upload
  const handlePreviewFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const imageUrl = URL.createObjectURL(file);
    setPreviewImage(imageUrl);
    event.target.value = '';
  };

  // Handle file upload for specific class
  const handleFileUpload = async (classIndex: number, event: React.ChangeEvent<HTMLInputElement>, type: 'genuine' | 'forged' = 'genuine') => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newClasses = [...classes];

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
      
    } catch (error) {
      console.error('Error processing files:', error);
    }
    
    event.target.value = '';
  };

  // Add multiple students
  const addMultipleStudents = (students: Student[], samplesMap?: Map<string, { genuine: SampleData[], forged: SampleData[] }>) => {
    if (students.length === 0) return;
    
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'];
    const newClasses = students.map((student, index) => {
      const studentSamples = samplesMap?.get(student.student_id);
      const allSamples = studentSamples ? [...studentSamples.genuine, ...studentSamples.forged] : [];
      
      return {
        student,
        color: colors[(classes.length + index) % colors.length],
        samples: allSamples,
        genuineSamples: studentSamples?.genuine || [],
        forgedSamples: studentSamples?.forged || []
      };
    });
    
    setClasses([...classes, ...newClasses]);
  };

  // Remove class
  const removeClass = (index: number) => {
    if (classes.length <= 1) return;
    
    const newClasses = classes.filter((_, i) => i !== index);
    setClasses(newClasses);
  };

  // Update class name (student)
  const updateClassName = (index: number, student: Student) => {
    const newClasses = [...classes];
    newClasses[index].student = student;
    setClasses(newClasses);
  };

  // Mock training function
  const trainModel = async () => {
    const validClasses = classes.filter(cls => cls.student && (cls.genuineSamples.length > 0 || cls.forgedSamples.length > 0));
    
    if (validClasses.length < 1) {
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      // Mock training process
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setTrainingProgress(i);
      }

      setIsModelLoaded(true);

    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
      setTrainingProgress(0);
    }
  };

  // Mock export functions
  const exportToS3Handler = async () => {
    setIsUploading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      setHasExportedToCloud(true);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const exportToLocalHandler = async () => {
    setIsDownloading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setHasDownloadedToPC(true);
    } catch (error) {
      console.error('Download failed:', error);
    } finally {
      setIsDownloading(false);
    }
  };

  // Toggle models display
  const toggleModels = () => {
    setShowModels(!showModels);
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