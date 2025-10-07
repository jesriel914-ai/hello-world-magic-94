//filepath: src/ai-model-siamese/lib/SiameseAIModelService.ts

// Siamese Model Service - Backend Integration
// Connects React frontend to Flask training API

export interface SiameseModelMetadata {
  id?: string;
  student_id: string;
  student_name?: string;
  model_type: 'siamese';
  training_date: string;
  sample_count: number;
  genuine_samples: number;
  forged_samples: number;
  epochs_trained: number;
  training_time_seconds: number;
  final_val_accuracy: number;
  final_val_loss: number;
  threshold: number;
  img_size: number[];
}

export interface SiameseTrainingMetrics {
  training_loss: number;
  validation_loss: number;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  training_time: number;
  epochs: number;
}

export interface SiameseVerificationResult {
  is_verified: boolean;
  confidence: number;
  similarity_score: number;
  threshold_used: number;
  student_id: string;
  student_name?: string;
}

// Configuration - Use Vite's import.meta.env instead of process.env
const API_BASE_URL = import.meta.env.VITE_SIAMESE_API_URL || 'http://localhost:5000';

export class SiameseModelService {
  private static instance: SiameseModelService;
  private apiUrl: string;

  constructor() {
    this.apiUrl = API_BASE_URL;
    console.log('🔧 Siamese Service URL configured as:', this.apiUrl);
  }

  static getInstance(): SiameseModelService {
    if (!SiameseModelService.instance) {
      SiameseModelService.instance = new SiameseModelService();
    }
    return SiameseModelService.instance;
  }

  /**
   * Train Siamese model for a student
   * @param studentId - Student identifier
   * @param genuineSamples - Array of genuine signature samples (base64 data URLs)
   * @param forgedSamples - Optional array of forged signature samples
   * @returns Training metadata
   */
  async trainModel(
    studentId: string,
    genuineSamples: any[],
    forgedSamples: any[] = []
  ): Promise<SiameseModelMetadata> {
    console.log(`🎓 Training Siamese model for student: ${studentId}`);
    console.log(`   Genuine samples: ${genuineSamples.length}`);
    console.log(`   Forged samples: ${forgedSamples.length}`);
    
    try {
      // Validate inputs
      if (genuineSamples.length < 2) {
        throw new Error('At least 2 genuine samples are required for training');
      }
      
      // Extract base64 thumbnails
      const genuineBase64 = genuineSamples.map(sample => sample.thumbnail);
      const forgedBase64 = forgedSamples.length > 0 ? forgedSamples.map(sample => sample.thumbnail) : [];
      
      console.log('📡 Sending training request to:', `${this.apiUrl}/api/train`);
      
      // Call Flask API
      const response = await fetch(`${this.apiUrl}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          student_id: studentId,
          genuine_samples: genuineBase64,
          forged_samples: forgedBase64.length > 0 ? forgedBase64 : undefined,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Training failed');
      }
      
      const data = await response.json();
      console.log('✅ Training completed:', data.metadata);
      
      return data.metadata as SiameseModelMetadata;
      
    } catch (error) {
      console.error('❌ Training error:', error);
      throw new Error(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Verify signature against student's trained model
   * @param studentId - Student identifier
   * @param signatureData - Signature image data (base64 or object with image property)
   * @returns Verification result
   */
  async verifySignature(
    studentId: string,
    signatureData: any
  ): Promise<SiameseVerificationResult> {
    console.log(`🔍 Verifying signature for student: ${studentId}`);
    
    try {
      // Extract base64 image
      let signatureBase64: string;
      
      if (typeof signatureData === 'string') {
        signatureBase64 = signatureData;
      } else if (signatureData.image) {
        signatureBase64 = signatureData.image;
      } else if (signatureData.thumbnail) {
        signatureBase64 = signatureData.thumbnail;
      } else {
        throw new Error('Invalid signature data format');
      }
      
      if (!signatureBase64) {
        throw new Error('No signature image provided');
      }
      
      console.log('📡 Sending verification request to:', `${this.apiUrl}/api/verify`);
      
      // Call Flask API
      const response = await fetch(`${this.apiUrl}/api/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          student_id: studentId,
          signature_image: signatureBase64,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Verification failed');
      }
      
      const data = await response.json();
      console.log('✅ Verification result:', data.result);
      
      return data.result as SiameseVerificationResult;
      
    } catch (error) {
      console.error('❌ Verification error:', error);
      throw new Error(`Verification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Check if a model exists for a student
   * @param studentId - Student identifier
   * @returns Model status and metadata
   */
  async checkModelStatus(studentId: string): Promise<{
    exists: boolean;
    metadata?: SiameseModelMetadata;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/model/status/${studentId}`);
      
      if (!response.ok) {
        throw new Error('Failed to check model status');
      }
      
      const data = await response.json();
      return data;
      
    } catch (error) {
      console.error('Error checking model status:', error);
      return { exists: false };
    }
  }

  /**
   * Health check for the API
   * @returns API health status
   */
  async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/health`);
      
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      
      return await response.json();
      
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
 * List all students with trained models
 * @returns Array of students with trained models and metadata
 */
async listTrainedStudents(): Promise<Array<{
  student_id: string;
  metadata: SiameseModelMetadata;
}>> {
  try {
    console.log('📋 Fetching list of trained students...');
    
    const response = await fetch(`${this.apiUrl}/api/models/list`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch trained students');
    }
    
    const data = await response.json();
    console.log(`✅ Found ${data.students.length} trained students`);
    
    return data.students;
    
  } catch (error) {
    console.error('Error fetching trained students:', error);
    return [];
  }
}

  // Placeholder methods for future TensorFlow.js conversion
  async loadModel(studentId: string): Promise<any> {
    console.log('Model loading from browser not yet implemented');
    throw new Error('Model loading from browser not yet implemented');
  }

  async saveModel(studentId: string, model: any): Promise<void> {
    console.log('Model saving from browser not yet implemented');
    throw new Error('Model saving from browser not yet implemented');
  }

  async exportModel(studentId: string): Promise<any> {
    console.log('Model export to TensorFlow.js not yet implemented');
    throw new Error('Model export to TensorFlow.js not yet implemented');
  }

  async importModel(modelData: any): Promise<void> {
    console.log('Model import not yet implemented');
    throw new Error('Model import not yet implemented');
  }
}

// Export singleton instance
export const siameseModelService = SiameseModelService.getInstance();