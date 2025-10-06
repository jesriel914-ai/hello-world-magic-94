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
import type { ClassData } from './ModelTraining';
import { fetchStudents } from '@/lib/supabaseService';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface TrainingSetupProps {
  classes: ClassData[];
  isTraining: boolean;
  isModelLoaded: boolean;
  trainingProgress: number;
  isUploading: boolean;
  hasUploaded: boolean;
  isDownloading: boolean;
  hasExportedToCloud: boolean;
  hasDownloadedToPC: boolean;
  onRemoveClass: (index: number) => void;
  onUpdateClassName: (index: number, student: Student) => void;
  onAddMultipleStudents: (students: Student[], samplesMap?: Map<string, any[]>) => void;
  onTrainModel: () => void;
  onUploadModelToS3: () => void;
  onDownloadModelToLocal: () => void;
  onHandleFileUpload: (classIndex: number, event: React.ChangeEvent<HTMLInputElement>, type: 'genuine' | 'forged') => void;
  formatStudentDisplay: (student: Student) => string;
}

const formatStudentDisplay = (student: Student): string => {
  return `${student.student_id} - ${student.firstname} ${student.surname}`;
};

const TrainingSetup: React.FC<TrainingSetupProps> = ({
  classes,
  isTraining,
  isModelLoaded,
  trainingProgress,
  isUploading,
  hasUploaded,
  isDownloading,
  hasExportedToCloud,
  hasDownloadedToPC,
  onRemoveClass,
  onUpdateClassName,
  onAddMultipleStudents,
  onHandleFileUpload,
  onTrainModel,
  onUploadModelToS3,
  onDownloadModelToLocal,
  formatStudentDisplay
}) => {
  const [batchUploadOpen, setBatchUploadOpen] = useState(false);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentProcessingStudent, setCurrentProcessingStudent] = useState('');
  const [totalFiles, setTotalFiles] = useState(0);
  const [processedFiles, setProcessedFiles] = useState(0);
  const [sampleDisplayMode, setSampleDisplayMode] = useState<'genuine' | 'forged'>('genuine');

  const loadStudentsForBatchUpload = async () => {
    try {
      const students = await fetchStudents();
      setAllStudents(students);
      setBatchUploadOpen(true);
    } catch (error) {
      console.error('Error loading students:', error);
      alert('Failed to load students for validation');
    }
  };

  const handleBatchUploadConfirm = async (validFolders: any[]) => {
    try {
      setBatchUploadOpen(false);
      setIsProcessingUpload(true);
      setUploadProgress(0);
      setProcessedFiles(0);
      
      // Calculate total files
      const total = validFolders.reduce((sum, folder) => sum + folder.genuineFiles.length + folder.forgedFiles.length, 0);
      setTotalFiles(total);
      
      const samplesMap = new Map<string, { genuine: any[], forged: any[] }>();
      let filesProcessed = 0;
      
      for (const folder of validFolders) {
        const { matchedStudent, genuineFiles, forgedFiles } = folder;
        const genuineSamples: any[] = [];
        const forgedSamples: any[] = [];
        
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
            console.error('Error processing genuine file:', error);
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
            console.error('Error processing forged file:', error);
          }
        }
        
        samplesMap.set(matchedStudent.student_id, { 
          genuine: genuineSamples, 
          forged: forgedSamples 
        });
      }
      
      const students = validFolders.map(folder => folder.matchedStudent);
      onAddMultipleStudents(students, samplesMap);
      
      setIsProcessingUpload(false);
      setUploadProgress(0);
      setCurrentProcessingStudent('');
      
    } catch (error) {
      console.error('Error processing batch upload:', error);
      setIsProcessingUpload(false);
      alert('Failed to process batch upload: ' + (error instanceof Error ? error.message : 'Unknown error'));
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
                          onStudentSelect={(student) => onUpdateClassName(index, student)}
                          trigger={
                            <DropdownMenuItem onSelect={(e) => e.preventDefault()}>
                              Change Student
                            </DropdownMenuItem>
                          }
                        />
                        <DropdownMenuItem onClick={() => onRemoveClass(index)}>
                          Delete Class
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  
                  <div className="border-t border-gray-200 my-3"></div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex gap-2">
                        <label className="cursor-pointer">
                          <Button variant="secondary" size="sm" className="bg-green-100 hover:bg-green-200 text-green-800 border-green-200" asChild>
                            <span>
                              <Upload className="w-3 h-3 mr-1" />
                              Genuine
                            </span>
                          </Button>
                          <input 
                            type="file" 
                            accept="image/*" 
                            multiple
                            onChange={(e) => onHandleFileUpload(index, e, 'genuine')} 
                            className="hidden"
                          />
                        </label>
                        <label className="cursor-pointer">
                          <Button variant="secondary" size="sm" className="bg-red-100 hover:bg-red-200 text-red-800 border-red-200" asChild>
                            <span>
                              <Upload className="w-3 h-3 mr-1" />
                              Forged
                            </span>
                          </Button>
                          <input 
                            type="file" 
                            accept="image/*" 
                            multiple
                            onChange={(e) => onHandleFileUpload(index, e, 'forged')} 
                            className="hidden"
                          />
                        </label>
                      </div>
                      <div className="text-sm text-gray-600">
                        <div>Genuine: {cls.genuineSamples?.length || 0}</div>
                        <div>Forged: {cls.forgedSamples?.length || 0}</div>
                      </div>
                    </div>
                    
                    <div className="border-solid border-gray-300 rounded-lg p-1 min-h-[100px] bg-gray-50 overflow-x-auto overlay-scrollbar-container border-[0.5px] relative group">
                      {/* Sample Type Navigation */}
                      {(cls.genuineSamples?.length > 0 || cls.forgedSamples?.length > 0) && (
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1 z-10">
                          <button
                            onClick={() => setSampleDisplayMode('genuine')}
                            className={`p-1 rounded ${
                              sampleDisplayMode === 'genuine' 
                                ? 'bg-green-500 text-white' 
                                : 'bg-white text-gray-600 hover:bg-gray-100'
                            }`}
                            title="Show genuine samples"
                          >
                            <Upload className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => setSampleDisplayMode('forged')}
                            className={`p-1 rounded ${
                              sampleDisplayMode === 'forged' 
                                ? 'bg-red-500 text-white' 
                                : 'bg-white text-gray-600 hover:bg-gray-100'
                            }`}
                            title="Show forged samples"
                          >
                            <AlertTriangle className="w-3 h-3" />
                          </button>
                        </div>
                      )}
                      
                      {/* Sample Display */}
                      {(() => {
                        const currentSamples = sampleDisplayMode === 'genuine' 
                          ? (cls.genuineSamples || []) 
                          : (cls.forgedSamples || []);
                        const sampleType = sampleDisplayMode === 'genuine' ? 'genuine' : 'forged';
                        
                        return currentSamples.length > 0 ? (
                          <div className="flex gap-1">
                            {currentSamples.map((sample, sampleIndex) => (
                              <div 
                                key={sampleIndex} 
                                className="flex-shrink-0 w-16 h-16 border border-gray-300 rounded overflow-hidden relative group"
                              >
                                <img 
                                  src={sample.thumbnail} 
                                  alt={`${cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'} ${sampleType} sample ${sampleIndex + 1}`} 
                                  className="w-full h-full object-cover filter grayscale"
                                />
                                <div className={`absolute top-0 left-0 px-1 py-0.5 text-xs font-medium ${
                                  sampleType === 'genuine' 
                                    ? 'bg-green-500 text-white' 
                                    : 'bg-red-500 text-white'
                                }`}>
                                  {sampleType === 'genuine' ? 'G' : 'F'}
                                </div>
                                <button
                                  className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity transform scale-90 group-hover:scale-100"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    const newClasses = [...classes];
                                    if (sampleType === 'genuine') {
                                      newClasses[index].genuineSamples.splice(sampleIndex, 1);
                                    } else {
                                      newClasses[index].forgedSamples.splice(sampleIndex, 1);
                                    }
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
                              No {sampleType} samples yet
                            </div>
                          </div>
                        );
                      })()}
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
              onStudentsSelect={onAddMultipleStudents}
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
              onClick={onTrainModel}
              disabled={classes.filter(cls => cls.samples.length > 0).length < 2 || isTraining}
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
                  onClick={onUploadModelToS3}
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
                  onClick={onDownloadModelToLocal}
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