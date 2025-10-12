import React, { useEffect, useState, useCallback } from 'react';
import Layout from '@/components/Layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/components/ui/use-toast';
import { 
  Upload, 
  Brain, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Loader2,
  User,
  Trash2,
  X,
  Camera,
  Zap,
  Target,
  Users,
  TrendingUp,
  RefreshCw
} from 'lucide-react';
import ModelTraining from '@/components/ModelTraining';
import getAIModelService from '@/lib/AIModelService';
import { fetchStudents } from '@/lib/supabaseService';
import { formatStudentDisplay } from '@/lib/utils';
import type { Student } from '@/types';

type SimpleTrainedModel = {
  id: string;
  student_id: string;
  student_name: string;
  training_date: string;
  accuracy?: number;
  sample_count: number;
};

const AISignatureModelTraining = () => {
  const { toast } = useToast();
  
  // Simplified Training State
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [signatureFiles, setSignatureFiles] = useState<File[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<{
    success: boolean;
    message: string;
    accuracy?: number;
    training_time?: number;
  } | null>(null);
  
  // Classification State
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [verificationPreview, setVerificationPreview] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState<{
    success: boolean;
    student_id?: string;
    student_name?: string;
    confidence?: number;
    message?: string;
  } | null>(null);
  
  const [trainedModels, setTrainedModels] = useState<SimpleTrainedModel[]>([]);
  const [students, setStudents] = useState<Student[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // AI Model Training State
  const [aiModelClasses, setAiModelClasses] = useState<string[]>([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isAiModelTraining, setIsAiModelTraining] = useState(false);
  const [useWebcam, setUseWebcam] = useState(false);

  // Remove TabsList and make AI Model Training always visible
  const loadStudents = useCallback(async () => {
    try {
      const studentsData = await fetchStudents();
      setStudents(studentsData);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load students",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [setStudents, toast]);

  const loadTrainedModels = useCallback(async () => {
    try {
      const aiModelService = getAIModelService();
      const models = await aiModelService.getTrainedModels();
      setTrainedModels(models);
    } catch (error) {
      console.error('Error loading trained models:', error);
    }
  }, [setTrainedModels]);

  // Load students and trained models on component mount
  useEffect(() => {
    loadStudents();
    loadTrainedModels();
  }, [loadStudents, loadTrainedModels]);

  const handleSignatureFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setSignatureFiles(prev => [...prev, ...files]);
  };

  const removeSignatureFile = (index: number) => {
    setSignatureFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleTrainModel = async () => {
    if (!selectedStudent || signatureFiles.length < 3) {
      toast({
        title: "Cannot Train Model",
        description: "Please select a student and upload at least 3 signature samples",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setTrainingResult(null);

    try {
      const aiModelService = getAIModelService();
      const result = await aiModelService.trainModel(
        selectedStudent.student_id,
        `${selectedStudent.firstname} ${selectedStudent.surname}`,
        signatureFiles
      );

      setTrainingResult(result);

      if (result.success) {
        toast({
          title: "Training Completed",
          description: `Model trained successfully with ${result.accuracy ? Math.round(result.accuracy * 100) : '?'}% accuracy in ${result.training_time || '?'} seconds`
        });

        // Reset form
        setSelectedStudent(null);
        setSignatureFiles([]);
        
        // Reload models list
        await loadTrainedModels();
      } else {
        toast({
          title: "Training Failed",
          description: result.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Training error:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred during training",
        variant: "destructive",
      });
    } finally {
      setIsTraining(false);
    }
  };

  const handleVerificationFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVerificationFile(file);
      setVerificationPreview(URL.createObjectURL(file));
      setClassificationResult(null);
    }
  };

  const handleClassifySignature = async () => {
    if (!verificationFile) {
      toast({
        title: "Error",
        description: "Please upload a signature image to classify",
        variant: "destructive",
      });
      return;
    }

    setIsVerifying(true);
    setClassificationResult(null);

    try {
      const aiModelService = getAIModelService();
      const result = await aiModelService.classifySignature(verificationFile);
      setClassificationResult(result);

      if (result.success) {
        toast({
          title: "Classification Complete",
          description: result.student_name
            ? `Signature belongs to: ${result.student_name} (${Math.round((result.confidence || 0) * 100)}% confidence)`
            : "No match found",
        });
      } else {
        toast({
          title: "Classification Failed",
          description: result.message || "Failed to classify signature",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Classification error:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred during classification",
        variant: "destructive",
      });
    } finally {
      setIsVerifying(false);
    }
  };

  const handleDeleteModel = async (modelId: string | number) => {
    try {
      const aiModelService = getAIModelService();
      await aiModelService.deleteModel(String(modelId));
      toast({
        title: "Model Deleted",
        description: "Trained model has been deleted successfully",
      });
      await loadTrainedModels();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete model",
        variant: "destructive",
      });
    }
  };

  // AI Model Training Interface Functions
  // Note: Training and classification are now handled directly by the AI Model Training component
  // These functions are no longer needed as the component handles everything client-side

  // Get model statistics
  const getModelStats = () => {
    const totalModels = trainedModels.length;
    const avgAccuracy = trainedModels.length > 0 
      ? trainedModels.reduce((sum, m) => sum + (m.accuracy || 0), 0) / trainedModels.length 
      : 0;
    
    return { totalModels, avgAccuracy };
  };

  const stats = getModelStats();

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <Layout>
      <div className="md:px-6 md:py-4">
        <div className="mb-3">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
            <div className="hidden md:block">
              <h1 className="text-2xl font-bold text-education-navy uppercase">Model Training</h1>
            </div>
          </div>
        </div>
        
        {/* Space between page title and card */}
        <div className="mb-4"></div>


        {/* Main Content */}
        <div className="w-full">

          {/* AI Model Training */}
          <div className="space-y-6">
            <ModelTraining
              onModelTrained={(model) => {
                console.log('Model trained:', model);
                toast({
                  title: "Model Trained",
                  description: "AI model has been trained successfully",
                });
              }}
              onClassification={(predictions) => {
                console.log('Classification results:', predictions);
              }}
            />
          </div>

        </div>
      </div>
    </Layout>
  );
};

export default AISignatureModelTraining;
