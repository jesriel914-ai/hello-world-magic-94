//filepath: ai-model-siamese/components/TrainingSetup.tsx
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Plus, 
  Trash2, 
  Upload, 
  CheckCircle, 
  Loader2, 
  CloudUpload,
  Download,
  ChevronDown,
  MoreVertical,
  Shield,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Users,
  User
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import StudentSelectionModal from '@/components/model-training-ui/components/StudentSelectionModal';
import BatchUpload from './BatchUpload';
import type { Student } from '@/types';
import { fetchStudents } from '@/lib/supabaseService';
import { siameseModelService } from '../lib/SiameseAIModelService';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

// Interfaces
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

const formatStudentDisplay = (student: Student): string => {
  return `${student.student_id} - ${student.firstname} ${student.surname}`;
};

interface TrainingSetupProps {
  classes: ClassData[];
  setClasses: (classes: ClassData[]) => void;
}

const TrainingSetup: React.FC<TrainingSetupProps> = ({ classes, setClasses }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploaded, setHasUploaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [hasExportedToCloud, setHasExportedToCloud] = useState(false);
  const [hasDownloadedToPC, setHasDownloadedToPC] = useState(false);
  const [batchUploadOpen, setBatchUploadOpen] = useState(false);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentProcessingStudent, setCurrentProcessingStudent] = useState('');
  const [totalFiles, setTotalFiles] = useState(0);
  const [processedFiles, setProcessedFiles] = useState(0);
  const [sampleDisplayMode, setSampleDisplayMode] = useState<'genuine' | 'forged'>('genuine');

  // Load students for batch upload
  const loadStudentsForBatchUpload = async () => {
    try {
      const students = await fetchStudents();
      setAllStudents(students);
      setBatchUploadOpen(true);
    } catch (error) {
      console.error('Error loading students:', error);
      toast({
        title: 'Error',
        description: 'Failed to load students for validation',
        variant: 'destructive',
      });
    }
  };

  // Handle batch upload confirmation
  const handleBatchUploadConfirm = async (validFolders: any[]) => {
    try {
      setBatchUploadOpen(false);
      setIsProcessingUpload(true);
      setUploadProgress(0);
      setProcessedFiles(0);
      
      // Calculate total files
      const total = validFolders.reduce((sum, folder) => sum + folder.genuineFiles.length + folder.forgedFiles.length, 0);
      setTotalFiles(total);
      
      const samplesMap = new Map<string, { genuine: SampleData[], forged: SampleData[] }>();
      let filesProcessed = 0;
      
      for (const folder of validFolders) {
        const { matchedStudent, genuineFiles, forgedFiles } = folder;
        const genuineSamples: SampleData[] = [];
        const forgedSamples: SampleData[] = [];
        
        setCurrentProcessingStudent(formatStudentDisplay(matchedStudent));
        
        // Process genuine files
        for (const file of genuineFiles) {
          try {
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
                    
                    genuineSamples.push({
                      thumbnail,
                      timestamp: Date.now(),
                      type: 'genuine'
                    });
                  }
                  
                  filesProcessed++;
                  setProcessedFiles(filesProcessed);
                  setUploadProgress((filesProcessed / total) * 100);
                  
                  resolve();
                } catch (error) {
                  reject(error);
                }
              };
              
              img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
              img.src = URL.createObjectURL(file);
            });
          } catch (error) {
            console.error('Error processing file:', error);
          }
        }
        
        // Process forged files
        for (const file of forgedFiles) {
          try {
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
                    
                    forgedSamples.push({
                      thumbnail,
                      timestamp: Date.now(),
                      type: 'forged'
                    });
                  }
                  
                  filesProcessed++;
                  setProcessedFiles(filesProcessed);
                  setUploadProgress((filesProcessed / total) * 100);
                  
                  resolve();
                } catch (error) {
                  reject(error);
                }
              };
              
              img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
              img.src = URL.createObjectURL(file);
            });
          } catch (error) {
            console.error('Error processing file:', error);
          }
        }
        
        samplesMap.set(matchedStudent.student_id, { genuine: genuineSamples, forged: forgedSamples });
      }
      
      const students = validFolders.map(folder => folder.matchedStudent);
      addMultipleStudents(students, samplesMap);
      
      setIsProcessingUpload(false);
      setUploadProgress(0);
      setCurrentProcessingStudent('');
      
    } catch (error) {
      console.error('Error processing batch upload:', error);
      setIsProcessingUpload(false);
      toast({
        title: 'Upload Failed',
        description: 'Failed to process batch upload: ' + (error instanceof Error ? error.message : 'Unknown error'),
        variant: 'destructive',
      });
    }
  };

  // Add multiple students
  const addMultipleStudents = (students: Student[], samplesMap?: Map<string, { genuine: SampleData[], forged: SampleData[] }>) => {
    if (students.length === 0) return;
    
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
    const newClasses = newStudents.map((student, index) => {
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
    
    // Remove the first class if it's the placeholder (no student and no samples)
    const shouldRemovePlaceholder = classes.length === 1 && !classes[0].student && 
                                     classes[0].samples.length === 0 && 
                                     classes[0].genuineSamples.length === 0 && 
                                     classes[0].forgedSamples.length === 0;
    
    if (shouldRemovePlaceholder) {
      setClasses(newClasses);
    } else {
      setClasses([...classes, ...newClasses]);
    }
    
    toast({
      title: 'Students Added',
      description: `Added ${newStudents.length} student${newStudents.length !== 1 ? 's' : ''} successfully.`,
    });
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

  // Training function - calls Python training pipeline
  const trainModel = async () => {
    const validClasses = classes.filter(cls => cls.student && (cls.genuineSamples.length > 0 || cls.forgedSamples.length > 0));
    
    if (validClasses.length < 1) {
      toast({
        title: 'Cannot Train',
        description: 'Please add students and upload samples before training!',
        variant: 'destructive',
      });
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      console.log('Starting Siamese training for all students...');
      
      // Train model for each student
      for (let i = 0; i < validClasses.length; i++) {
        const cls = validClasses[i];
        const studentId = cls.student?.student_id || `student_${i}`;
        
        console.log(`Training model for student: ${studentId}`);
        
        // Call the Siamese training service
        const metadata = await siameseModelService.trainModel(
          studentId,
          cls.genuineSamples,
          cls.forgedSamples
        );
        
        console.log('Training completed for student:', metadata);
        
        // Update progress
        const progress = ((i + 1) / validClasses.length) * 100;
        setTrainingProgress(progress);
      }

      setIsModelLoaded(true);
      console.log('All models trained successfully!');

    } catch (error) {
      console.error('Training failed:', error);
      toast({
        title: 'Training Failed',
        description: `Training failed: ${error}`,
        variant: 'destructive',
      });
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

  return (
    <Card className="h-[605px] w-full lg:col-span-2 flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <User className="w-5 h-5" />
            Training Setup
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <MoreVertical className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={loadStudentsForBatchUpload}>
                Batch Upload
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto overlay-scrollbar-container space-y-4 pb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {classes.map((cls, index) => (
              <Card key={index} className="border-2 border-gray-200">
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 flex-1">
                      <span className="text-sm font-medium text-gray-600 min-w-[20px]">
                        {index + 1}.
                      </span>
                      <div className="font-bold text-base">
                        {cls.student ? formatStudentDisplay(cls.student) : 'Select a student...'}
                      </div>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          className="h-8 w-8 p-0"
                        >
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <StudentSelectionModal
                          selectedStudent={cls.student}
                          excludeStudents={classes.filter(c => c.student && c.student?.student_id !== cls.student?.student_id).map(c => c.student!).filter(Boolean)}
                          selectionContext="classCard"
                          onStudentSelect={(student) => updateClassName(index, student)}
                          trigger={
                            <DropdownMenuItem onSelect={(e) => e.preventDefault()}>
                              Change Student
                            </DropdownMenuItem>
                          }
                        />
                        <DropdownMenuItem onClick={() => removeClass(index)}>
                          Delete Class
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  
                  <div className="border-t border-gray-200 my-3"></div>
                  
                  <div className="space-y-3">
                    {/* Upload Buttons */}
                    <div className="flex items-center justify-between">
                      <div className="flex gap-2">
                        <label className="cursor-pointer">
                          <Button variant="secondary" size="sm" className="bg-green-100 hover:bg-green-200 text-green-800 border-green-200" asChild>
                            <span>
                              <Upload className="w-4 h-4 mr-1" />
                              Genuine
                            </span>
                          </Button>
                          <input 
                            type="file" 
                            accept="image/*" 
                            multiple
                            onChange={(e) => handleFileUpload(index, e, 'genuine')} 
                            className="hidden"
                          />
                        </label>
                        <label className="cursor-pointer">
                          <Button variant="secondary" size="sm" className="bg-red-100 hover:bg-red-200 text-red-800 border-red-200" asChild>
                            <span>
                              <Upload className="w-4 h-4 mr-1" />
                              Forged
                            </span>
                          </Button>
                          <input 
                            type="file" 
                            accept="image/*" 
                            multiple
                            onChange={(e) => handleFileUpload(index, e, 'forged')} 
                            className="hidden"
                          />
                        </label>
                      </div>
                      <div className="text-sm text-gray-600">
                        <div>Genuine: {cls.genuineSamples.length}</div>
                        <div>Forged: {cls.forgedSamples.length}</div>
                      </div>
                    </div>
                    
                    {/* Sample Display */}
                    <div className="border-solid border-gray-300 rounded-lg p-1 min-h-[100px] bg-gray-50 overflow-x-auto overlay-scrollbar-container border-[0.5px]">
                      {cls.samples.length > 0 ? (
                        <div className="flex gap-1">
                          {cls.samples.map((sample, sampleIndex) => (
                            <div 
                              key={sampleIndex} 
                              className="flex-shrink-0 w-16 h-16 border border-gray-300 rounded overflow-hidden relative group"
                            >
                              <img 
                                src={sample.thumbnail} 
                                alt={`${cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'} sample ${sampleIndex + 1}`} 
                                className="w-full h-full object-cover filter grayscale"
                              />
                              <div className="absolute top-0 left-0 bg-black bg-opacity-50 text-white text-xs px-1 rounded-br">
                                {sample.type === 'genuine' ? 'G' : 'F'}
                              </div>
                              <button
                                className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity transform scale-90 group-hover:scale-100"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newClasses = [...classes];
                                  newClasses[index].samples.splice(sampleIndex, 1);
                                  if (sample.type === 'genuine') {
                                    const genuineIndex = newClasses[index].genuineSamples.findIndex(s => s.thumbnail === sample.thumbnail);
                                    if (genuineIndex !== -1) {
                                      newClasses[index].genuineSamples.splice(genuineIndex, 1);
                                    }
                                  } else {
                                    const forgedIndex = newClasses[index].forgedSamples.findIndex(s => s.thumbnail === sample.thumbnail);
                                    if (forgedIndex !== -1) {
                                      newClasses[index].forgedSamples.splice(forgedIndex, 1);
                                    }
                                  }
                                  setClasses(newClasses);
                                }}
                                title="Remove sample"
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full min-h-[92px]">
                          <div className="text-gray-500 text-sm">
                            No samples yet
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <StudentSelectionModal
              mode="multiple"
              excludeStudents={classes.map(c => c.student).filter(Boolean)}
              onStudentsSelect={addMultipleStudents}
              trigger={
                <Button 
                  variant="outline" 
                  className="flex-1 border-2 border-dashed border-gray-300 hover:border-green-500 hover:text-green-500 h-auto p-3"
                  disabled={classes.length >= 50}
                >
                  <Plus className="w-5 h-5 mr-2" />
                  Add Students
                </Button>
              }
            />
          </div>
        </div>
        
        <div className="border-t pt-6 mt-auto">
          <div className="text-sm mb-4">
            {isModelLoaded ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="w-4 h-4" />
                Model Trained
              </div>
            ) : isTraining ? (
              <div className="flex items-center gap-2 text-yellow-600">
                <Loader2 className="w-4 h-4 animate-spin" />
                Training...
              </div>
            ) : (
              <div className="text-gray-600">
                Ready to train
              </div>
            )}
          </div>
          
          {isTraining && (
            <div className="space-y-2 mb-4">
              <Progress value={trainingProgress} className="w-full" />
              <div className="text-sm text-gray-600 text-center">
                {trainingProgress.toFixed(0)}% complete
              </div>
            </div>
          )}
          
          <div className="flex gap-3">
            <Button 
              onClick={trainModel}
              disabled={classes.filter(cls => cls.samples.length > 0).length < 1 || isTraining}
              className="flex-1"
            >
              <User className="w-4 h-4 mr-2" />
              {isTraining ? 'Training...' : 'Train Model'}
            </Button>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  disabled={!isModelLoaded || isUploading || isDownloading}
                  className="flex-1"
                  variant="outline"
                  title="Export trained model"
                >
                  {isUploading || isDownloading ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Upload className="w-4 h-4 mr-2" />
                  )}
                  {isUploading ? 'Exporting...' : isDownloading ? 'Preparing...' : 'Export Model'}
                  <ChevronDown className="w-4 h-4 ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem 
                  onClick={exportToS3Handler}
                  disabled={isUploading || isDownloading || hasExportedToCloud}
                  className="flex items-center gap-2"
                >
                  {hasExportedToCloud ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : (
                    <CloudUpload className="w-4 h-4" />
                  )}
                  <span>{hasExportedToCloud ? 'Exported to Cloud' : 'Cloud Storage'}</span>
                </DropdownMenuItem>
                <DropdownMenuItem 
                  onClick={exportToLocalHandler}
                  disabled={isUploading || isDownloading}
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          
        </div>
      </CardContent>

      <BatchUpload
        open={batchUploadOpen}
        onOpenChange={setBatchUploadOpen}
        onConfirm={handleBatchUploadConfirm}
        students={allStudents}
      />

      {/* Processing Upload Progress Dialog */}
      <Dialog open={isProcessingUpload} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md" hideCloseButton>
          <DialogHeader>
            <DialogTitle>Processing Batch Upload</DialogTitle>
            <DialogDescription>
              Creating classes and uploading signatures...
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            </div>
            {currentProcessingStudent && (
              <p className="text-sm text-center text-gray-600">
                Processing: {currentProcessingStudent}
              </p>
            )}
            <div className="space-y-2">
              <Progress value={uploadProgress} className="w-full" />
              <p className="text-xs text-center text-gray-500">
                {processedFiles} / {totalFiles} files ({uploadProgress.toFixed(0)}%)
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
};

export default TrainingSetup;