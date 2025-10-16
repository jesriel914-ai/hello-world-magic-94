import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { 
  Upload, 
  ChevronDown, 
  Loader2, 
  CheckCircle, 
  AlertTriangle,
  Trash2,
  Plus,
  RefreshCw,
  X
} from 'lucide-react';
import { siameseModelService } from '../lib/SiameseAIModelService';

interface SampleData {
  thumbnail: string;
  timestamp: number;
  type?: 'genuine' | 'forged';
}

interface IncrementalLearningDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

const IncrementalLearningDialog: React.FC<IncrementalLearningDialogProps> = ({ 
  isOpen, 
  onOpenChange 
}) => {
  const [trainedStudents, setTrainedStudents] = useState<Array<{
    student_id: string;
    metadata: any;
  }>>([]);
  const [isLoadingStudents, setIsLoadingStudents] = useState(true);
  const [selectedStudent, setSelectedStudent] = useState<string>('');
  
  const [newGenuineSamples, setNewGenuineSamples] = useState<SampleData[]>([]);
  const [newForgedSamples, setNewForgedSamples] = useState<SampleData[]>([]);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMessage, setProcessingMessage] = useState('');
  const [isCheckingRetraining, setIsCheckingRetraining] = useState(false);
  const [retrainingCheck, setRetrainingCheck] = useState<{
    needs_retraining: boolean;
    reason?: string;
    recommendation?: string;
  } | null>(null);

  // Load trained students when dialog opens
  useEffect(() => {
    if (isOpen) {
      const loadTrainedStudents = async () => {
        setIsLoadingStudents(true);
        try {
          const students = await siameseModelService.listTrainedStudents();
          setTrainedStudents(students);
        } catch (error) {
          console.error('Failed to load trained students:', error);
        } finally {
          setIsLoadingStudents(false);
        }
      };
      
      loadTrainedStudents();
    }
  }, [isOpen]);

  // Reset form when dialog closes
  useEffect(() => {
    if (!isOpen) {
      setSelectedStudent('');
      setNewGenuineSamples([]);
      setNewForgedSamples([]);
      setRetrainingCheck(null);
      setIsProcessing(false);
      setProcessingMessage('');
    }
  }, [isOpen]);

  // Check if retraining is needed when samples change
  useEffect(() => {
    if (selectedStudent && (newGenuineSamples.length > 0 || newForgedSamples.length > 0)) {
      checkIfRetrainingNeeded();
    } else {
      setRetrainingCheck(null);
    }
  }, [selectedStudent, newGenuineSamples.length, newForgedSamples.length]);

  const checkIfRetrainingNeeded = async () => {
    if (!selectedStudent) return;
    
    setIsCheckingRetraining(true);
    try {
      const totalNewSamples = newGenuineSamples.length + newForgedSamples.length;
      const check = await siameseModelService.checkIncrementalLearning(
        selectedStudent,
        totalNewSamples
      );
      setRetrainingCheck(check);
    } catch (error) {
      console.error('Failed to check retraining:', error);
    } finally {
      setIsCheckingRetraining(false);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    type: 'genuine' | 'forged'
  ) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newSamples: SampleData[] = [];

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
                
                newSamples.push({
                  thumbnail,
                  timestamp: Date.now(),
                  type
                });
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
      
      if (type === 'genuine') {
        setNewGenuineSamples([...newGenuineSamples, ...newSamples]);
      } else {
        setNewForgedSamples([...newForgedSamples, ...newSamples]);
      }
      
    } catch (error) {
      console.error('Error processing files:', error);
      alert('Failed to process some images. Please try again.');
    }
    
    event.target.value = '';
  };

  const removeSample = (index: number, type: 'genuine' | 'forged') => {
    if (type === 'genuine') {
      setNewGenuineSamples(newGenuineSamples.filter((_, i) => i !== index));
    } else {
      setNewForgedSamples(newForgedSamples.filter((_, i) => i !== index));
    }
  };

  const handleAddSamples = async () => {
    if (!selectedStudent) {
      alert('Please select a student first!');
      return;
    }

    if (newGenuineSamples.length === 0 && newForgedSamples.length === 0) {
      alert('Please upload at least one sample!');
      return;
    }

    // Check if retraining is recommended
    if (retrainingCheck?.needs_retraining) {
      const confirm = window.confirm(
        `Full retraining is recommended:\n\n${retrainingCheck.reason}\n\n${retrainingCheck.recommendation}\n\nDo you want to proceed with incremental learning anyway?`
      );
      if (!confirm) return;
    }

    setIsProcessing(true);
    
    try {
      // Add genuine samples
      if (newGenuineSamples.length > 0) {
        setProcessingMessage(`Adding ${newGenuineSamples.length} genuine samples...`);
        await siameseModelService.addGenuineSamples(
          selectedStudent,
          newGenuineSamples,
          true // update threshold
        );
        console.log('✅ Genuine samples added successfully');
      }

      // Add forged samples
      if (newForgedSamples.length > 0) {
        setProcessingMessage(`Adding ${newForgedSamples.length} forged samples...`);
        await siameseModelService.addForgedSamples(
          selectedStudent,
          newForgedSamples
        );
        console.log('✅ Forged samples added successfully');
      }

      // Success
      alert(`Successfully added samples via incremental learning!\n\nGenuine: ${newGenuineSamples.length}\nForged: ${newForgedSamples.length}`);
      
      // Close dialog
      onOpenChange(false);
      
    } catch (error) {
      console.error('Failed to add samples:', error);
      alert(`Failed to add samples:\n\n${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
      setProcessingMessage('');
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5" />
            Incremental Learning
          </DialogTitle>
          <DialogDescription>
            Add new signature samples to an existing student model without retraining from scratch.
          </DialogDescription>
        </DialogHeader>
        
        <div className="flex-1 overflow-y-auto space-y-6">
          {/* Info Box */}
          <div className="text-sm text-blue-700 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="font-semibold mb-2">💡 What is Incremental Learning?</div>
            <p>
              Add new signature samples to an existing student model <strong>without retraining from scratch</strong>.
              The model updates its knowledge efficiently, saving time and computational resources.
            </p>
          </div>

          {/* Student Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Select Student</label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="w-full justify-between" disabled={isLoadingStudents}>
                  <span>
                    {isLoadingStudents 
                      ? 'Loading students...'
                      : selectedStudent 
                        ? selectedStudent
                        : 'Choose a student...'}
                  </span>
                  <ChevronDown className="w-4 h-4 ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-full max-h-[300px] overflow-y-auto">
                {trainedStudents.length === 0 ? (
                  <DropdownMenuItem disabled className="text-center text-gray-500">
                    No trained models found
                  </DropdownMenuItem>
                ) : (
                  trainedStudents.map((student, index) => (
                    <DropdownMenuItem
                      key={index}
                      onClick={() => {
                        setSelectedStudent(student.student_id);
                        setNewGenuineSamples([]);
                        setNewForgedSamples([]);
                        setRetrainingCheck(null);
                      }}
                    >
                      <div className="flex flex-col">
                        <span className="font-medium">{student.student_id}</span>
                        <span className="text-xs text-gray-500">
                          {student.metadata.genuine_samples} genuine, {student.metadata.forged_samples} forged
                        </span>
                      </div>
                    </DropdownMenuItem>
                  ))
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Retraining Check Warning */}
          {retrainingCheck?.needs_retraining && (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="font-semibold text-yellow-900 text-sm">
                    Full Retraining Recommended
                  </div>
                  <div className="text-sm text-yellow-700 mt-1">
                    {retrainingCheck.reason}
                  </div>
                  <div className="text-sm text-yellow-600 mt-1 italic">
                    {retrainingCheck.recommendation}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Upload Buttons */}
          {selectedStudent && (
            <div className="space-y-4">
              <div className="flex gap-3">
                <label className="flex-1">
                  <Button 
                    variant="secondary" 
                    size="sm" 
                    className="w-full bg-green-100 hover:bg-green-200 text-green-800 border-green-200"
                    asChild
                  >
                    <span>
                      <Plus className="w-4 h-4 mr-2" />
                      Add Genuine Samples
                    </span>
                  </Button>
                  <input 
                    type="file" 
                    accept="image/*" 
                    multiple
                    onChange={(e) => handleFileUpload(e, 'genuine')}
                    className="hidden"
                  />
                </label>
                
                <label className="flex-1">
                  <Button 
                    variant="secondary" 
                    size="sm" 
                    className="w-full bg-red-100 hover:bg-red-200 text-red-800 border-red-200"
                    asChild
                  >
                    <span>
                      <Plus className="w-4 h-4 mr-2" />
                      Add Forged Samples
                    </span>
                  </Button>
                  <input 
                    type="file" 
                    accept="image/*" 
                    multiple
                    onChange={(e) => handleFileUpload(e, 'forged')}
                    className="hidden"
                  />
                </label>
              </div>

              {/* Sample Counts */}
              <div className="flex justify-between text-sm text-gray-600">
                <span>New Genuine: {newGenuineSamples.length}</span>
                <span>New Forged: {newForgedSamples.length}</span>
              </div>
            </div>
          )}

          {/* Sample Preview */}
          {(newGenuineSamples.length > 0 || newForgedSamples.length > 0) && (
            <div className="space-y-4">
              {/* Genuine Samples */}
              {newGenuineSamples.length > 0 && (
                <div>
                  <div className="text-sm font-medium text-green-700 mb-3">
                    Genuine Samples ({newGenuineSamples.length})
                  </div>
                  <div className="border rounded-lg p-3 bg-green-50 overflow-x-auto">
                    <div className="flex gap-2">
                      {newGenuineSamples.map((sample, index) => (
                        <div 
                          key={index}
                          className="flex-shrink-0 w-20 h-20 border border-gray-300 rounded overflow-hidden relative group"
                        >
                          <img 
                            src={sample.thumbnail}
                            alt={`Genuine ${index + 1}`}
                            className="w-full h-full object-cover filter grayscale"
                          />
                          <button
                            className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => removeSample(index, 'genuine')}
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Forged Samples */}
              {newForgedSamples.length > 0 && (
                <div>
                  <div className="text-sm font-medium text-red-700 mb-3">
                    Forged Samples ({newForgedSamples.length})
                  </div>
                  <div className="border rounded-lg p-3 bg-red-50 overflow-x-auto">
                    <div className="flex gap-2">
                      {newForgedSamples.map((sample, index) => (
                        <div 
                          key={index}
                          className="flex-shrink-0 w-20 h-20 border border-gray-300 rounded overflow-hidden relative group"
                        >
                          <img 
                            src={sample.thumbnail}
                            alt={`Forged ${index + 1}`}
                            className="w-full h-full object-cover filter grayscale"
                          />
                          <button
                            className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => removeSample(index, 'forged')}
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="border-t pt-4 flex gap-3">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isProcessing}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            onClick={handleAddSamples}
            disabled={!selectedStudent || (newGenuineSamples.length === 0 && newForgedSamples.length === 0) || isProcessing}
            className="flex-1"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                {processingMessage || 'Processing...'}
              </>
            ) : (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                Add Samples (Incremental)
              </>
            )}
          </Button>
        </div>
        
        {isProcessing && (
          <div className="text-sm text-gray-500 text-center mt-2">
            This won't retrain from scratch - only updating with new data
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default IncrementalLearningDialog;