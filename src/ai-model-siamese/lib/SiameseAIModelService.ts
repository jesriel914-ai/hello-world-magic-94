//filepath: src/ai-model-siamese/lib/SiameseAIModelService.ts

// Siamese Model Service - Backend Integration
// UPDATED: Compatible with signature isolation preprocessing

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
  // NEW: Signature isolation metadata
  preprocessing?: string;  // 'signature_extraction' or 'legacy'
  background_invariant?: boolean;  // true if using isolation
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
  min_distance: number;  // NEW: Distance metrics
  average_distance: number;
  max_distance?: number;
  std_distance?: number;
  similarity_score?: number;  // Optional legacy support
  threshold_used: number;
  student_id: string;
  student_name?: string;
  num_references?: number;  // NEW: How many reference samples
  model_accuracy?: number;  // NEW: Model training accuracy
  model_precision?: number;
  model_recall?: number;
  preprocessing?: string;  // NEW: Which preprocessing was used
}


const getSiameseServiceUrl = () => {
  if (typeof window !== 'undefined') {
    const currentHost = window.location.hostname;
    
    // If accessing via Cloudflare tunnel
    if (currentHost.includes('.trycloudflare.com') || currentHost.includes('.cfargotunnel.com')) {
      // Option A: Separate siamese tunnel
      const envUrl = import.meta.env.VITE_SIAMESE_API_URL as string;
      if (envUrl && (envUrl.includes('.trycloudflare.com') || envUrl.includes('ngrok'))) {
        return envUrl;
      }
      
      // Option B: Same server as main app
      const protocol = window.location.protocol;
      const host = window.location.host;
      return `${protocol}//${host}`;
    }
    
    // For localhost or ngrok, use environment variable
    const baseUrl = import.meta.env.VITE_SIAMESE_API_URL as string;
    if (baseUrl) {
      return baseUrl;
    }
    
    return 'http://localhost:5000';
  }
  return 'http://localhost:5000';
};


const API_BASE_URL = getSiameseServiceUrl();

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
   * NOW with signature isolation preprocessing
   * 
   * @param studentId - Student identifier
   * @param genuineSamples - Array of genuine signature samples (base64 data URLs)
   * @param forgedSamples - Array of forged signature samples (minimum 5 required)
   * @returns Training metadata including preprocessing info
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
      
      if (forgedSamples.length < 5) {
        throw new Error(
          `At least 5 forged samples are required for training.\n` +
          `Current: ${forgedSamples.length} forged samples\n` +
          `Recommended: 20+ forged samples for best accuracy`
        );
      }
      
      // Extract base64 thumbnails
      const genuineBase64 = genuineSamples.map(sample => sample.thumbnail);
      const forgedBase64 = forgedSamples.map(sample => sample.thumbnail);
      
      console.log('📡 Sending training request to:', `${this.apiUrl}/api/train`);
      console.log('   Using signature isolation preprocessing');
      
      // Call Flask API
      const response = await fetch(`${this.apiUrl}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',  // Bypass ngrok browser warning
        },
        body: JSON.stringify({
          student_id: studentId,
          genuine_samples: genuineBase64,
          forged_samples: forgedBase64,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Training failed');
      }
      
      const data = await response.json();
      console.log('✅ Training completed:', data.metadata);
      
      // Log preprocessing info
      if (data.metadata.preprocessing === 'signature_extraction') {
        console.log('✅ Model trained with signature isolation (background-invariant)');
      }
      
      return data.metadata as SiameseModelMetadata;
      
    } catch (error) {
      console.error('❌ Training error:', error);
      throw new Error(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Verify signature against student's trained model
   * Works with ANY camera quality, lighting, or background!
   * 
   * @param studentId - Student identifier
   * @param signatureData - Signature image data (base64 or object with image property)
   * @returns Verification result with detailed metrics
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
      console.log('   Backend will apply signature isolation automatically');
      
      // Call Flask API
      const response = await fetch(`${this.apiUrl}/api/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',  // Bypass ngrok browser warning
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
      
      // Log preprocessing info
      if (data.result.preprocessing === 'signature_extraction') {
        console.log('✅ Verified using signature isolation (background ignored)');
      } else {
        console.warn('⚠️  Using legacy preprocessing - consider retraining model');
      }
      
      // Log detailed metrics
      console.log('📊 Verification metrics:');
      console.log(`   Min distance: ${data.result.min_distance?.toFixed(4)}`);
      console.log(`   Avg distance: ${data.result.average_distance?.toFixed(4)}`);
      console.log(`   Confidence: ${(data.result.confidence * 100).toFixed(1)}%`);
      console.log(`   References used: ${data.result.num_references}`);
      
      return data.result as SiameseVerificationResult;
      
    } catch (error) {
      console.error('❌ Verification error:', error);
      throw new Error(`Verification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Check if a model exists for a student
   * Also returns preprocessing method used
   * 
   * @param studentId - Student identifier
   * @returns Model status and metadata
   */
  async checkModelStatus(studentId: string): Promise<{
    exists: boolean;
    metadata?: SiameseModelMetadata;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/model/status/${studentId}`, {
        headers: {
          'ngrok-skip-browser-warning': 'true',
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to check model status');
      }
      
      const data = await response.json();
      
      // Log preprocessing info if model exists
      if (data.exists && data.metadata) {
        if (data.metadata.preprocessing === 'signature_extraction') {
          console.log(`✅ Model for ${studentId} uses signature isolation`);
        } else {
          console.log(`⚠️  Model for ${studentId} uses legacy preprocessing`);
        }
      }
      
      return data;
      
    } catch (error) {
      console.error('Error checking model status:', error);
      return { exists: false };
    }
  }

  /**
   * Health check for the API
   * @returns API health status including GPU availability
   */
  async healthCheck(): Promise<{ 
    status: string; 
    service: string; 
    version: string;
    gpu_available?: boolean;
    tensorflow_version?: string;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/health`, {
        headers: {
          'ngrok-skip-browser-warning': 'true',
        }
      });
      
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      
      const health = await response.json();
      
      console.log('🏥 API Health Check:');
      console.log(`   Status: ${health.status}`);
      console.log(`   Service: ${health.service}`);
      console.log(`   Version: ${health.version}`);
      if (health.gpu_available !== undefined) {
        console.log(`   GPU: ${health.gpu_available ? '✅ Available' : '❌ Not available'}`);
      }
      if (health.tensorflow_version) {
        console.log(`   TensorFlow: ${health.tensorflow_version}`);
      }
      
      return health;
      
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
   * List all students with trained models
   * Shows which models use signature isolation
   * @returns Array of students with trained models and metadata
   */
  async listTrainedStudents(): Promise<Array<{
    student_id: string;
    metadata: SiameseModelMetadata;
  }>> {
    try {
      console.log('📋 Fetching list of trained students...');
      
      const response = await fetch(`${this.apiUrl}/api/models/list`, {
        headers: {
          'ngrok-skip-browser-warning': 'true',
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch trained students');
      }
      
      const data = await response.json();
      console.log(`✅ Found ${data.students.length} trained students`);
      
      // Log preprocessing info for each model
      const isolatedCount = data.students.filter(
        (s: any) => s.metadata.preprocessing === 'signature_extraction'
      ).length;
      const legacyCount = data.students.length - isolatedCount;
      
      if (isolatedCount > 0) {
        console.log(`   ${isolatedCount} models with signature isolation ✅`);
      }
      if (legacyCount > 0) {
        console.log(`   ${legacyCount} models with legacy preprocessing ⚠️`);
      }
      
      return data.students;
      
    } catch (error) {
      console.error('Error fetching trained students:', error);
      return [];
    }
  }

  /**
   * Batch train multiple students
   * NEW: Efficient batch training with memory cleanup between students
   * 
   * @param students - Array of student training data
   * @returns Batch training results
   */
  async batchTrainModels(students: Array<{
    student_id: string;
    genuine_samples: any[];
    forged_samples: any[];
  }>): Promise<{
    success: boolean;
    total: number;
    succeeded: number;
    failed: number;
    results: any[];
    errors: any[];
  }> {
    console.log(`🎓 Batch training ${students.length} students...`);
    
    try {
      // Validate all students first
      for (const student of students) {
        if (student.genuine_samples.length < 2) {
          throw new Error(`Student ${student.student_id}: At least 2 genuine samples required`);
        }
        if (student.forged_samples.length < 5) {
          throw new Error(`Student ${student.student_id}: At least 5 forged samples required`);
        }
      }
      
      // Prepare batch data
      const batchData = students.map(student => ({
        student_id: student.student_id,
        genuine_samples: student.genuine_samples.map(s => s.thumbnail),
        forged_samples: student.forged_samples.map(s => s.thumbnail)
      }));
      
      console.log('📡 Sending batch training request...');
      
      const response = await fetch(`${this.apiUrl}/api/train/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          students: batchData
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Batch training failed');
      }
      
      const result = await response.json();
      
      console.log(`✅ Batch training completed:`);
      console.log(`   Total: ${result.total}`);
      console.log(`   Succeeded: ${result.succeeded}`);
      console.log(`   Failed: ${result.failed}`);
      
      return result;
      
    } catch (error) {
      console.error('❌ Batch training error:', error);
      throw new Error(`Batch training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get detailed verification metrics
   * Useful for debugging or showing detailed results to users
   * 
   * @param verificationResult - Result from verifySignature()
   * @returns Human-readable interpretation
   */
  interpretVerificationResult(verificationResult: SiameseVerificationResult): {
    decision: string;
    confidenceLevel: string;
    explanation: string;
    technicalDetails: string;
  } {
    const { is_verified, confidence, min_distance, threshold_used } = verificationResult;
    
    let decision: string;
    let confidenceLevel: string;
    let explanation: string;
    
    if (is_verified) {
      decision = 'VERIFIED ✅';
      
      if (confidence > 0.9) {
        confidenceLevel = 'Very High';
        explanation = 'Signature strongly matches the trained references. Very likely genuine.';
      } else if (confidence > 0.75) {
        confidenceLevel = 'High';
        explanation = 'Signature matches the trained references well. Likely genuine.';
      } else if (confidence > 0.6) {
        confidenceLevel = 'Moderate';
        explanation = 'Signature matches but with some variations. Consider manual review.';
      } else {
        confidenceLevel = 'Low';
        explanation = 'Signature barely passes threshold. Recommend manual verification.';
      }
    } else {
      decision = 'NOT VERIFIED ❌';
      
      if (min_distance > threshold_used * 1.5) {
        confidenceLevel = 'High Rejection';
        explanation = 'Signature significantly differs from trained references. Very likely forged.';
      } else if (min_distance > threshold_used * 1.2) {
        confidenceLevel = 'Moderate Rejection';
        explanation = 'Signature differs from trained references. Likely forged or different person.';
      } else {
        confidenceLevel = 'Borderline Rejection';
        explanation = 'Signature is close to threshold. May be genuine with poor capture quality or slight variation.';
      }
    }
    
    const technicalDetails = `
Min Distance: ${min_distance?.toFixed(4) || 'N/A'}
Avg Distance: ${verificationResult.average_distance?.toFixed(4) || 'N/A'}
Threshold: ${threshold_used?.toFixed(4) || 'N/A'}
Confidence: ${(confidence * 100).toFixed(1)}%
References: ${verificationResult.num_references || 'N/A'} samples
Model Accuracy: ${verificationResult.model_accuracy ? (verificationResult.model_accuracy * 100).toFixed(1) + '%' : 'N/A'}
Preprocessing: ${verificationResult.preprocessing === 'signature_extraction' ? 'Signature Isolation ✅' : 'Legacy ⚠️'}
    `.trim();
    
    return {
      decision,
      confidenceLevel,
      explanation,
      technicalDetails
    };
  }

  /**
   * Check if a model needs retraining (legacy preprocessing)
   * @param studentId - Student identifier
   * @returns Whether model should be retrained
   */
  async shouldRetrainModel(studentId: string): Promise<{
    shouldRetrain: boolean;
    reason?: string;
  }> {
    try {
      const status = await this.checkModelStatus(studentId);
      
      if (!status.exists) {
        return {
          shouldRetrain: true,
          reason: 'No model exists for this student'
        };
      }
      
      const metadata = status.metadata!;
      
      // Check if using legacy preprocessing
      if (metadata.preprocessing !== 'signature_extraction') {
        return {
          shouldRetrain: true,
          reason: 'Model uses legacy preprocessing. Retrain for background-invariant verification.'
        };
      }
      
      // Check if model has low accuracy
      if (metadata.final_val_accuracy && metadata.final_val_accuracy < 0.85) {
        return {
          shouldRetrain: true,
          reason: `Model accuracy is ${(metadata.final_val_accuracy * 100).toFixed(1)}%. Consider retraining with more samples.`
        };
      }
      
      // Check if insufficient samples
      if (metadata.forged_samples < 20) {
        return {
          shouldRetrain: true,
          reason: `Only ${metadata.forged_samples} forged samples. Recommended: 20+ for better accuracy.`
        };
      }
      
      return {
        shouldRetrain: false
      };
      
    } catch (error) {
      console.error('Error checking if model needs retraining:', error);
      return {
        shouldRetrain: false
      };
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