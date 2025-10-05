//filepath: src\lib\AIModelService.ts

import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { supabase } from '@/integrations/supabase/client';
import * as tf from '@tensorflow/tfjs';

// Extend Window interface to include ENV property
declare global {
  interface Window {
    ENV?: {
      NEXT_PUBLIC_AI_SERVICE_URL?: string;
      NEXT_PUBLIC_API_BASE_URL?: string;
    };
  }
}

// Function to get AI service URL with fallback for browser environment
// Function to get AI service URL with fallback for browser environment
// Function to get AI service URL with fallback for browser environment
const getAIServiceUrl = () => {
  if (typeof window !== 'undefined') {
    const currentHost = window.location.hostname;
    
    // If accessing via Cloudflare tunnel, use the same domain
    if (currentHost.includes('.trycloudflare.com') || currentHost.includes('.cfargotunnel.com')) {
      // Use the current Cloudflare URL with /api prefix
      const protocol = window.location.protocol; // https
      const host = window.location.host; // includes port if any
      return `${protocol}//${host}`;
    }
    
    // For localhost/local IP, use environment variable or default
    const baseUrl = import.meta.env.VITE_AI_BASE_URL as string;
    if (baseUrl) {
      return baseUrl.replace(/\/$/, '');
    }
    
    // Fallback to localhost:8000
    return 'http://localhost:8000';
  }
  return 'http://localhost:8000';
};

const AI_SERVICE_URL = getAIServiceUrl();

console.log('🔧 AI Service URL configured as:', AI_SERVICE_URL);

// TypeScript interfaces for model topology
interface ModelLayer {
  class_name: string;
  config?: {
    units?: number;
    activation?: string;
    use_bias?: boolean;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

interface ModelTopology {
  class_name: string;
  config?: {
    layers?: ModelLayer[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

// Helper function to fix model topology for TensorFlow.js compatibility
const fixModelTopology = (topology: ModelTopology): ModelTopology => {
  console.log('🔧 Fixing model topology for TensorFlow.js compatibility during import...');
  
  // Create a deep copy of the topology
  const fixedTopology = JSON.parse(JSON.stringify(topology));
  
  // Fix the main model class_name
  if (fixedTopology.class_name === 'Sequential' || fixedTopology.class_name === 'Model') {
    fixedTopology.class_name = 'Functional';
    console.log('✅ Changed model class_name to "Functional"');
  }
  
  // Fix layer types in the config
  if (fixedTopology.config && fixedTopology.config.layers) {
    fixedTopology.config.layers = fixedTopology.config.layers.map((layer: ModelLayer) => {
      if (layer.class_name === 'Sequential' || layer.class_name === 'Model') {
        console.log(`🔧 Converting layer ${layer.class_name} to Dense`);
        return {
          class_name: 'Dense',
          config: {
            units: layer.config?.units || 128,
            activation: layer.config?.activation || 'relu',
            use_bias: layer.config?.use_bias !== false
          }
        };
      }
      return layer;
    });
  }
  
  return fixedTopology;
};

// S3 client will be initialized in constructor to avoid process.env access at module level
let s3Client: S3Client;

interface SimpleTrainingResult {
  success: boolean;
  message: string;
  accuracy?: number;
  training_time?: number;
}

interface SimpleClassificationResult {
  success: boolean;
  student_id?: string;
  student_name?: string;
  confidence?: number;
  message?: string;
  error?: string;
}

// Interface for training metrics to replace 'any' types
// Interface for prediction results
interface PredictionResult {
  className: string;
  confidence: number;
}

// Interface for loaded model
interface LoadedModel {
  modelId: string;
  trainingDate: string;
  accuracy?: number;
  sampleCount: number;
  studentCount?: number;
  getClassLabels: () => string[];
  getTotalClasses: () => number;
  predict: (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => Promise<PredictionResult[]>;
  dispose: () => void;
}

export interface TrainingMetrics {
  students?: Array<{
    id?: string;
    student_id?: string;
    firstname: string;
    surname: string;
    full_name?: string;
  }>;
  total_sample_count?: number;
  student_count?: number;
  [key: string]: unknown; // Allow additional properties
}

export interface ModelMetadata {
  version: string;
  createdAt: string;
  environment?: 'production' | 'staging' | 'development';
  modelArchitecture: {
    featureExtractor: string;
    classifier: string;
    inputShape: number[];
    outputShape: number[];
    featureSize?: number[];
  };
  modelStructure?: {
    hasMobileNet?: boolean;
    hasClassifier?: boolean;
    inputSize?: number[];
    featureSize?: number[];
    outputSize?: number;
    labels?: string[];
  };
  trainingConfig: {
    epochs: number;
    optimizer: string;
    learningRate: number;
    batchSize?: number;
    augmentationTypes: number;
    totalSamples: number;
  };
  performance?: {
    finalAccuracy?: number;
    finalLoss?: number;
    trainingTime?: number;
  };
  classes: Array<{
    name: string;
    color: string;
    sampleCount: number;
    studentId?: string;
  }>;
  storage: {
    location: 's3';
    bucket: string;
    region: string;
    modelKey: string;
    metadataKey: string;
    weightsKey?: string;
  };
}

export interface S3UploadResult {
  success: boolean;
  location?: string;
  etag?: string;
  error?: Error | unknown;
}

export interface S3DownloadResult {
  success: boolean;
  data?: unknown;
  weights?: string; // Base64 encoded weights data
  metadata?: Record<string, string>;
  error?: Error | unknown;
}

class AIModelServiceClass {
  private baseUrl = AI_SERVICE_URL;
  private bucketName: string;
  private publicBaseUrl: string;

  constructor() {
    console.log('AIModelService initialized with baseUrl:', this.baseUrl);
    // Only initialize S3 client if process is available (not in browser)
    if (typeof process !== 'undefined' && process.env) {
      this.bucketName = process.env.NEXT_PUBLIC_S3_BUCKET || 'signatureai-uploads';
      this.publicBaseUrl = process.env.NEXT_PUBLIC_S3_PUBLIC_BASE_URL || 
        `https://${this.bucketName}.s3.${process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1'}.amazonaws.com`;
      
      // Initialize S3 client with your existing configuration
      s3Client = new S3Client({
        region: process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1',
        credentials: {
          accessKeyId: process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY!,
        },
      });
    } else {
      // Fallback values for browser environment
      this.bucketName = 'signatureai-uploads';
      this.publicBaseUrl = 'https://signatureai-uploads.s3.us-east-1.amazonaws.com';
      console.warn('S3 client not initialized in browser environment. S3 uploads will not work without a backend API.');
    }
  }

  private getUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  /**
   * Upload AI model to S3 with organized folder structure
   */
  async uploadModel(
    modelData: unknown,
    metadata: ModelMetadata,
    environment: 'production' | 'staging' | 'development' = 'production',
    studentId?: string
  ): Promise<S3UploadResult> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const version = metadata.version;
      
      // Generate S3 keys with date-time folder
      const folderName = timestamp.slice(0, -5); // Remove milliseconds for cleaner folder name
      const modelKey = `ai-models/${folderName}/model.json`;
      const weightsKey = `ai-models/${folderName}/weights.bin`;
      const metadataKey = `ai-models/${folderName}/metadata.json`;
      
      // Check if S3 client is available (server-side only)
      if (!s3Client) {
        throw new Error('S3 upload is not supported in browser environment. Please use a backend API for S3 operations.');
      }

      // Handle new model data format with real TensorFlow.js files
      if (typeof modelData === 'object' && modelData !== null && 'modelJson' in modelData) {
        const typedModelData = modelData as {
          modelJson: string;
          weightsBin: Blob;
          metadataJson: string;
        };

        // Upload model.json
        const modelCommand = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: modelKey,
          Body: typedModelData.modelJson,
          ContentType: 'application/json',
        });
        
        const modelResult = await s3Client.send(modelCommand);

        // Upload weights.bin
        const weightsCommand = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: weightsKey,
          Body: typedModelData.weightsBin,
          ContentType: 'application/octet-stream',
        });
        
        const weightsResult = await s3Client.send(weightsCommand);

        // Update metadata with storage information
        metadata.storage = {
          location: 's3',
          bucket: this.bucketName,
          region: process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1',
          modelKey,
          weightsKey,
          metadataKey
        };

        // Upload metadata
        const metadataCommand = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: metadataKey,
          Body: typedModelData.metadataJson,
          ContentType: 'application/json',
        });

        const metadataResult = await s3Client.send(metadataCommand);

        return {
          success: true,
          location: `${this.publicBaseUrl}/${modelKey}`,
          etag: modelResult.ETag
        };
      } else {
        // Fallback to old format for backward compatibility
        const modelCommand = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: modelKey,
          Body: JSON.stringify(modelData),
          ContentType: 'application/json',
        });
        
        const modelResult = await s3Client.send(modelCommand);

        // Update metadata with storage information
        metadata.storage = {
          location: 's3',
          bucket: this.bucketName,
          region: process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1',
          modelKey,
          metadataKey
        };

        // Upload metadata
        const metadataCommand = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: metadataKey,
          Body: JSON.stringify(metadata, null, 2),
          ContentType: 'application/json',
        });

        const metadataResult = await s3Client.send(metadataCommand);

        return {
          success: true,
          location: `${this.publicBaseUrl}/${modelKey}`,
          etag: modelResult.ETag
        };
      }
    } catch (error) {
      console.error('S3 upload error:', error);
      return {
        success: false,
        error
      };
    }
  }

  /**
   * Get public URL for a model
   */
  getPublicUrl(key: string): string {
    return `${this.publicBaseUrl}/${key}`;
  }

  // Health check
  async healthCheck() {
    const res = await fetch(this.getUrl('/health'));
    if (!res.ok) throw new Error('AI service not available');
    return res.json();
  }

  // Preview generation for images
  async getPreviewURL(file: File): Promise<string> {
    if (file.type.startsWith('image/') && file.type !== 'image/tiff' &&
        !file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
      return URL.createObjectURL(file);
    }

    // For non-image files, return a placeholder
    return '/placeholder-signature.png';
  }

  // Simple training - just like AI Model Training
  async trainModel(
    studentId: string,
    studentName: string,
    genuineFiles: File[]
  ): Promise<SimpleTrainingResult> {
    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      formData.append('student_name', studentName);

      // Add genuine signature files
      for (const file of genuineFiles) {
        formData.append('signature_files', file);
      }

      const response = await fetch(`${this.baseUrl}/api/simple-train`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Training failed: ${response.status}`);
      }

      const result = await response.json();

      return {
        success: true,
        message: result.message || 'Model trained successfully',
        accuracy: result.accuracy,
        training_time: result.training_time
      };
    } catch (error) {
      console.error('Simple training error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Training failed'
      };
    }
  }

  // Simple classification - identify the owner
  async classifySignature(imageFile: File): Promise<SimpleClassificationResult> {
    try {
      const formData = new FormData();
      formData.append('signature_image', imageFile);

      const response = await fetch(`${this.baseUrl}/api/simple-classify`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Classification failed: ${response.status}`);
      }

      const result = await response.json();

      return {
        success: true,
        student_id: result.student_id,
        student_name: result.student_name,
        confidence: result.confidence,
        message: result.message || 'Classification completed'
      };
    } catch (error) {
      console.error('Simple classification error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Classification failed'
      };
    }
  }

  // Get all trained models
  async getTrainedModels(): Promise<Array<{
    id: string;
    student_id: string;
    student_name: string;
    training_date: string;
    accuracy?: number;
    sample_count: number;
    training_metrics?: TrainingMetrics;
  }>> {
    try {
      const query = supabase
        .from('global_trained_models')
        .select(`
          model_uuid,
          training_date,
          accuracy,
          sample_count,
          student_count,
          training_metrics
        `)
        .order('training_date', { ascending: false });
      
      const { data: models, error } = await query;

      if (error) {
        throw new Error(`Failed to get models: ${error.message}`);
      }

      // Transform the data to match the expected format
      return models?.map(model => {
        // For global models, extract student names from training_metrics if available
        let studentName = 'Global Model';
        if (model.training_metrics && typeof model.training_metrics === 'object') {
          const trainingMetrics = model.training_metrics as TrainingMetrics;
          if (trainingMetrics.students && Array.isArray(trainingMetrics.students)) {
            studentName = trainingMetrics.students.map((s) => `${s.firstname} ${s.surname}`).join(', ');
          }
        }
        
        return {
          id: model.model_uuid, // Use model_uuid as the ID for downloads
          student_id: 'global',
          student_name: studentName,
          training_date: model.training_date || new Date().toISOString(),
          accuracy: model.accuracy,
          sample_count: model.sample_count || 0,
          training_metrics: model.training_metrics as TrainingMetrics
        };
      }) || [];
    } catch (error) {
      console.error('Error getting trained models:', error);
      return [];
    }
  }

  // Delete a trained model
  async deleteModel(modelId: string): Promise<{ success: boolean; message: string }> {
    try {
      const { error } = await supabase
        .from('global_trained_models')
        .delete()
        .eq('model_uuid', modelId);

      if (error) {
        throw new Error(`Failed to delete model: ${error.message}`);
      }

      return {
        success: true,
        message: 'Model deleted successfully'
      };
    } catch (error) {
      console.error('Error deleting model:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Failed to delete model'
      };
    }
  }

/**
 * Upload a trained model to S3 and update database record
 * FIXED: Uses correct 3-file structure (model.json, weights.bin, metadata.json)
 */
async uploadTrainedModelToS3(
  modelId: string,
  modelData: unknown,
  studentId?: string,
  trainingData?: {
    total_sample_count?: number;
    student_count?: number;
    students?: Array<{
      id: string;
      student_id: string;
      firstname: string;
      surname: string;
      full_name: string;
    }>;
    accuracy?: number;
    epochs?: number;
    optimizer?: string;
    learning_rate?: number;
    batch_size?: number;
    training_summary?: string;
    model_architecture?: string;
  }
): Promise<{ success: boolean; message: string; s3Location?: string }> {
  try {
    console.log('🔄 Starting model upload to S3 with CORRECT 3-file structure...');
    
    // Check for duplicates
    const { data: existingModel, error: checkError } = await supabase
      .from('global_trained_models')
      .select('id, model_path, created_at')
      .eq('status', 'completed')
      .order('created_at', { ascending: false })
      .limit(1);
    
    if (checkError) {
      console.warn('Error checking for existing model:', checkError);
    } else if (existingModel && existingModel.length > 0) {
      const timeDiff = Date.now() - new Date(existingModel[0].created_at).getTime();
      if (timeDiff < 5000) { // Less than 5 seconds ago
        return {
          success: false,
          message: 'This model was just uploaded. Please wait before uploading again.'
        };
      }
    }
    
    // Validate modelData structure - should have 3 files
    if (typeof modelData !== 'object' || modelData === null) {
      throw new Error('Invalid model data format');
    }
    
    const typedModelData = modelData as {
      modelJson: string;
      weightsBin: string; // Base64
      metadataJson: string;
    };
    
    if (!typedModelData.modelJson || !typedModelData.weightsBin || !typedModelData.metadataJson) {
      throw new Error('Model data must contain modelJson, weightsBin, and metadataJson');
    }
    
    console.log('✅ Model data validated - has 3 required files');
    
    // Parse the JSON strings to validate them
    let parsedModelJson, parsedMetadata;
    try {
      parsedModelJson = JSON.parse(typedModelData.modelJson);
      parsedMetadata = JSON.parse(typedModelData.metadataJson);
    } catch (parseError) {
      throw new Error('Invalid JSON in model files');
    }
    
    const metadataToUpload = JSON.parse(typedModelData.metadataJson);
console.log('📤 Metadata being uploaded - labels:', metadataToUpload.labels);
console.log('📤 Metadata being uploaded - full:', metadataToUpload);
    // Create model metadata for database
    const modelDetails = {
      sample_count: trainingData?.total_sample_count || 0,
      total_sample_count: trainingData?.total_sample_count || 0,
      student_count: trainingData?.student_count || 1,
      students: trainingData?.students || [],
      accuracy: trainingData?.accuracy || 0.85,
      epochs: trainingData?.epochs || 50,
      optimizer: trainingData?.optimizer || 'adam',
      learning_rate: trainingData?.learning_rate || 0.001,
      batch_size: trainingData?.batch_size || 16,
      training_summary: trainingData?.training_summary,
      model_architecture: trainingData?.model_architecture || 'mobilenet_v1_classifier'
    };
    
    // Use the backend API for S3 upload
    const response = await fetch(`${this.baseUrl}/api/upload-model-to-s3`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        modelData: typedModelData,
        metadata: parsedMetadata,
        studentId,
        modelType: 'global',
        isThreeFileFormat: true // Flag to tell backend this is correct format
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      
      // Handle specific S3 configuration errors
      if (errorData.error === 'S3_NOT_CONFIGURED') {
        throw new Error(`S3 is not configured: ${errorData.message}. Please check your AWS credentials.`);
      }
      
      throw new Error(errorData.message || `Failed to upload to S3: ${response.status}`);
    }
    
    const uploadResult = await response.json();
    
    if (!uploadResult.success) {
      throw new Error(uploadResult.message || 'Failed to upload to S3');
    }
    
    console.log('✅ S3 upload successful:', uploadResult.location);
    
    // Create database record
    const { data: insertData, error: insertError } = await supabase
      .from('global_trained_models')
      .insert({
        model_path: uploadResult.location,
        s3_key: uploadResult.metadata?.storage?.modelKey || '',
        model_uuid: `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        status: 'completed',
        sample_count: modelDetails.total_sample_count,
        student_count: modelDetails.student_count,
        accuracy: modelDetails.accuracy,
        training_metrics: {
          epochs: modelDetails.epochs,
          optimizer: modelDetails.optimizer,
          learningRate: modelDetails.learning_rate,
          batchSize: modelDetails.batch_size,
          students: modelDetails.students,
          training_summary: modelDetails.training_summary,
          model_architecture: modelDetails.model_architecture
        }
      })
      .select()
      .single();
    
    if (insertError) {
      console.warn('Model uploaded to S3 but failed to create database record:', insertError);
      return {
        success: true,
        message: 'Model uploaded to S3 successfully, but database record creation failed',
        s3Location: uploadResult.location
      };
    }
    
    console.log('✅ Database record created');
    
    return {
      success: true,
      message: 'Model uploaded to S3 successfully and database record created',
      s3Location: uploadResult.location
    };
    
  } catch (error) {
    console.error('❌ Error uploading model to S3:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Failed to upload model to S3'
    };
  }
}

/**
 * Download a trained model from S3
 * FIXED: Expects correct 3-file structure (model.json, weights.bin, metadata.json)
 */
async downloadModel(modelId: string): Promise<S3DownloadResult> {
  try {
    console.log(`🔄 Downloading model: ${modelId}`);
    
    // Get model information from database
    const { data: modelData, error: modelError } = await supabase
      .from('global_trained_models')
      .select('*')
      .eq('model_uuid', modelId)
      .single();
    
    if (modelError || !modelData) {
      throw new Error(`Model not found: ${modelId}`);
    }
    
    console.log('✅ Model data found in database');
    
    // Try backend proxy download (handles CORS and S3 access)
    try {
      const backendUrl = `${this.baseUrl}/api/download-model/${modelData.model_uuid}`;
      console.log('🔄 Downloading via backend proxy:', backendUrl);
      
      const response = await fetch(backendUrl);
      
      if (!response.ok) {
        throw new Error(`Backend proxy error: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Backend proxy failed to download model');
      }
      
      console.log('📥 Downloaded metadata - labels:', JSON.parse(result.data).metadataJson);
      console.log('✅ Backend proxy download successful');
      
      // Validate that we have the 3 required files
      if (!result.data || typeof result.data !== 'string') {
        throw new Error('Invalid model data received from backend');
      }
      
      // Parse the combined data
      let combinedData;
      try {
        combinedData = JSON.parse(result.data);
      } catch (parseError) {
        throw new Error('Failed to parse model data from backend');
      }
      
      // Validate structure - should have model.json, weights.bin, metadata.json
      if (!combinedData.modelJson || !combinedData.weightsBin || !combinedData.metadataJson) {
        throw new Error('Downloaded model is missing required files. Expected: modelJson, weightsBin, metadataJson');
      }
      
      console.log('✅ Model has correct 3-file structure');
      
      return {
        success: true,
        data: result.data,
        weights: result.weights,
        metadata: {
          modelId: modelData.model_uuid,
          studentCount: modelData.student_count?.toString() || '0',
          sampleCount: modelData.sample_count?.toString() || '0',
          accuracy: modelData.accuracy?.toString() || '0',
          trainingDate: modelData.created_at,
          s3Key: modelData.s3_key,
          modelPath: modelData.model_path
        }
      };
      
    } catch (backendError) {
      console.error('❌ Backend proxy download failed:', backendError);
      throw new Error(`Failed to download model: ${backendError instanceof Error ? backendError.message : 'Unknown error'}`);
    }
    
  } catch (error) {
    console.error('❌ Error downloading model:', error);
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Failed to download model')
    };
  }
}

  // Get public download URL for a model (for direct browser downloads)
  getModelDownloadUrl(modelId: string): string | null {
    // This would typically generate a pre-signed URL for S3 downloads
    // For now, return the public model path if available
    // In a real implementation, you'd generate temporary signed URLs
    return null; // Placeholder - implement based on your S3 setup
  }

  // Check if a model exists and is available for download
  async checkModelExists(modelId: string): Promise<boolean> {
    try {
      const { data, error } = await supabase
        .from('global_trained_models')
        .select('model_uuid, model_path, s3_key')
        .eq('model_uuid', modelId)
        .single();
      
      if (error || !data) {
        return false;
      }
      
      // Check if the model is accessible
      if (data.model_path) {
        try {
          const response = await fetch(data.model_path, { method: 'HEAD' });
          return response.ok;
        } catch {
          return false;
        }
      }
      
      return true;
      
    } catch (error) {
      console.error('Error checking model existence:', error);
      return false;
    }
  }


/**
 * Load a trained model and convert it to CustomModel format
 * FIXED: Uses new ModelImport service for Teachable Machine models
 * 
 * Replace the existing loadModel function in AIModelService.ts with this:
 */
async loadModel(modelId: string): Promise<{
  success: boolean;
  model?: LoadedModel;
  error?: string;
}> {
  try {
    console.log('🔄 Loading model:', modelId);
    
    // Download the model
    const downloadResult = await this.downloadModel(modelId);
    
    if (!downloadResult.success) {
      throw new Error((downloadResult.error as Error)?.message || 'Failed to download model');
    }
    
    // Parse the downloaded data
    const modelContent = downloadResult.data;
    if (typeof modelContent !== 'string') {
      throw new Error('Downloaded data is not in expected string format');
    }
    
    console.log('✅ Model data downloaded');
    
    // Use the new ModelImport service to load the model
    const { loadModelFromDownload } = await import('@/components/model-training-ui/services/ModelImport');
    const importedModel = await loadModelFromDownload(modelContent);
    
    console.log('✅ Model imported successfully');
    console.log('📋 Model labels:', importedModel.metadata.labels);
    
    // Create CustomModel interface with proper prediction function
    const customModel: LoadedModel = {
      modelId: modelId,
      trainingDate: importedModel.metadata.userMetadata?.training_date || new Date().toISOString(),
      accuracy: importedModel.metadata.userMetadata?.accuracy || 0.85,
      sampleCount: importedModel.metadata.userMetadata?.sample_count || 0,
      studentCount: importedModel.metadata.userMetadata?.total_students || importedModel.metadata.labels.length,
      
      getClassLabels: () => importedModel.metadata.labels,
      getTotalClasses: () => importedModel.metadata.labels.length,
      
      predict: async (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => {
        console.log('🔮 Making prediction with loaded model');
  
        // DON'T use tf.tidy with async - do memory management manually
        let imageTensor: tf.Tensor | null = null;
        let normalized: tf.Tensor | null = null;
        let batched: tf.Tensor | null = null;
        let features: tf.Tensor | null = null;
        let predictions: tf.Tensor | null = null;
  
        try {
          // Preprocess image
          imageTensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat();
    
          if (flipped) {
            const flippedTensor = imageTensor.reverse(1);
            imageTensor.dispose();
            imageTensor = flippedTensor;
          }
    
          // Normalize for MobileNet [-1, 1]
          normalized = imageTensor.sub(127.5).div(127.5);
          batched = normalized.expandDims(0);
    
          // Extract features using MobileNet
          features = importedModel.featureExtractor.predict(batched) as tf.Tensor;
    
          // Run through classifier
          predictions = importedModel.classifier.predict(features) as tf.Tensor;
          const predictionData = await predictions.data();
    
          // Convert to results
          const results: PredictionResult[] = Array.from(predictionData).map((confidence, index) => ({
            className: importedModel.metadata.labels[index] || `Class ${index}`,
            confidence: Number(confidence)
          }));
    
          // Sort by confidence
          results.sort((a, b) => b.confidence - a.confidence);
    
          return results;
    
        } finally {
          // Clean up all tensors
          if (imageTensor) imageTensor.dispose();
          if (normalized) normalized.dispose();
          if (batched) batched.dispose();
          if (features) features.dispose();
          if (predictions) predictions.dispose();
        }
      }
      
      
    };
    
    console.log('✅ Model loaded and ready for predictions');
    
    return {
      success: true,
      model: customModel
    };
    
  } catch (error) {
    console.error('❌ Error loading model:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to load model'
    };
  }
}

}

class AIModelServiceSingleton {
  private static instance: AIModelServiceClass | null = null;
  
  static getInstance(): AIModelServiceClass {
    if (!this.instance) {
      this.instance = new AIModelServiceClass();
    }
    return this.instance;
  }
}

export function getAIModelService(): AIModelServiceClass {
  return AIModelServiceSingleton.getInstance();
}

export default getAIModelService;