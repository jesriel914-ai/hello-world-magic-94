//filepath: src/ai-model-siamese/lib/SiameseAIModelService.ts

// Siamese Model Service - Backend Integration
// UPDATED: Now with 1:N Classification and Incremental Learning

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
  preprocessing?: string;
  background_invariant?: boolean;
  // NEW: Incremental learning metadata
  total_reference_embeddings?: number;
  last_incremental_update?: string;
  incremental_updates?: number;
}

export interface SiameseVerificationResult {
  is_verified: boolean;
  confidence: number;
  min_distance: number;
  average_distance: number;
  max_distance?: number;
  std_distance?: number;
  similarity_score?: number;
  threshold_used: number;
  student_id: string;
  student_name?: string;
  num_references?: number;
  model_accuracy?: number;
  model_precision?: number;
  model_recall?: number;
  preprocessing?: string;
}

// NEW: Classification result (1:N)
export interface SiameseClassificationResult {
  identified: boolean;
  student_id: string | null;
  confidence: number;
  distance?: number;
  top_matches: Array<{
    student_id: string;
    distance: number;
    confidence: number;
    metadata?: any;
  }>;
  error?: string;
}

// NEW: Incremental learning check result
export interface IncrementalLearningCheck {
  needs_retraining: boolean;
  reason?: string;
  recommendation?: string;
}

const getSiameseServiceUrl = () => {
  if (typeof window !== 'undefined') {
    const currentHost = window.location.hostname;
    
    if (currentHost.includes('.trycloudflare.com') || currentHost.includes('.cfargotunnel.com')) {
      const envUrl = import.meta.env.VITE_SIAMESE_API_URL as string;
      if (envUrl && (envUrl.includes('.trycloudflare.com') || envUrl.includes('ngrok'))) {
        return envUrl;
      }
      
      const protocol = window.location.protocol;
      const host = window.location.host;
      return `${protocol}//${host}`;
    }
    
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

  // ============================================================================
  // EXISTING METHODS (Training & 1:1 Verification)
  // ============================================================================

  async trainModel(
    studentId: string,
    genuineSamples: any[],
    forgedSamples: any[] = []
  ): Promise<SiameseModelMetadata> {
    console.log(`🎓 Training Siamese model for student: ${studentId}`);
    console.log(`   Genuine samples: ${genuineSamples.length}`);
    console.log(`   Forged samples: ${forgedSamples.length}`);
    
    try {
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
      
      const genuineBase64 = genuineSamples.map(sample => sample.thumbnail);
      const forgedBase64 = forgedSamples.map(sample => sample.thumbnail);
      
      console.log('📡 Sending training request to:', `${this.apiUrl}/api/train`);
      
      const response = await fetch(`${this.apiUrl}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
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
      
      return data.metadata as SiameseModelMetadata;
      
    } catch (error) {
      console.error('❌ Training error:', error);
      throw new Error(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async verifySignature(
    studentId: string,
    signatureData: any
  ): Promise<SiameseVerificationResult> {
    console.log(`🔍 Verifying signature for student: ${studentId}`);
    
    try {
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
      
      const response = await fetch(`${this.apiUrl}/api/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
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

  // ============================================================================
  // NEW METHODS - Classification (1:N)
  // ============================================================================

  /**
   * NEW: Classify signature to automatically identify owner (1:N)
   * Works without selecting student first
   * 
   * @param signatureData - Signature image (base64 or object)
   * @param topK - Number of top matches to return (default: 3)
   * @returns Classification result with identified student or "unknown"
   */
  async classifySignature(
    signatureData: any,
    topK: number = 3
  ): Promise<SiameseClassificationResult> {
    console.log(`🎯 Classifying signature (1:N identification)...`);
    
    try {
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
      
      console.log('📡 Sending classification request to:', `${this.apiUrl}/api/classify`);
      
      const response = await fetch(`${this.apiUrl}/api/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          signature_image: signatureBase64,
          top_k: topK,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Classification failed');
      }
      
      const data = await response.json();
      const result = data.result as SiameseClassificationResult;
      
      if (result.identified) {
        console.log(`✅ Signature identified: ${result.student_id}`);
        console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
      } else {
        console.log(`❓ Unknown signature (no match found)`);
      }
      
      if (result.top_matches && result.top_matches.length > 0) {
        console.log(`📊 Top matches:`);
        result.top_matches.slice(0, 3).forEach((match, i) => {
          console.log(`   ${i + 1}. ${match.student_id}: ${(match.confidence * 100).toFixed(1)}%`);
        });
      }
      
      return result;
      
    } catch (error) {
      console.error('❌ Classification error:', error);
      throw new Error(`Classification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * NEW: Real-time classification for webcam frames
   * Optimized for speed, use for live camera feed
   * 
   * @param frameData - Webcam frame (base64)
   * @returns Classification result
   */
  async classifyRealtimeFrame(
    frameData: string
  ): Promise<SiameseClassificationResult> {
    try {
      const response = await fetch(`${this.apiUrl}/api/classify/realtime`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          frame: frameData,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Real-time classification failed');
      }
      
      const data = await response.json();
      return data.result as SiameseClassificationResult;
      
    } catch (error) {
      console.error('❌ Real-time classification error:', error);
      return {
        identified: false,
        student_id: null,
        confidence: 0,
        top_matches: [],
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * NEW: Rebuild classification database
   * Use after training new models or when database is corrupted
   */
  async rebuildClassifier(): Promise<void> {
    console.log('🔄 Rebuilding classification database...');
    
    try {
      const response = await fetch(`${this.apiUrl}/api/classifier/rebuild`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to rebuild classifier');
      }
      
      console.log('✅ Classification database rebuilt successfully');
      
    } catch (error) {
      console.error('❌ Failed to rebuild classifier:', error);
      throw error;
    }
  }

  // ============================================================================
  // NEW METHODS - Incremental Learning
  // ============================================================================

  /**
   * NEW: Add new genuine samples to existing student model
   * WITHOUT retraining from scratch
   * 
   * @param studentId - Student identifier
   * @param newSamples - Array of new genuine signature samples
   * @param updateThreshold - Whether to recalculate verification threshold
   * @returns Updated metadata
   */
  async addGenuineSamples(
    studentId: string,
    newSamples: any[],
    updateThreshold: boolean = true
  ): Promise<SiameseModelMetadata> {
    console.log(`🔄 Adding ${newSamples.length} genuine samples for ${studentId} (incremental learning)`);
    
    try {
      if (newSamples.length === 0) {
        throw new Error('No new samples provided');
      }
      
      const samplesBase64 = newSamples.map(sample => sample.thumbnail);
      
      console.log('📡 Sending incremental learning request...');
      
      const response = await fetch(`${this.apiUrl}/api/incremental/add-genuine`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          student_id: studentId,
          new_samples: samplesBase64,
          update_threshold: updateThreshold,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        
        // Check if full retraining is needed
        if (error.needs_retraining) {
          throw new Error(
            `Full retraining recommended:\n${error.reason}\n\n${error.recommendation}`
          );
        }
        
        throw new Error(error.error || 'Failed to add samples');
      }
      
      const data = await response.json();
      console.log('✅ Samples added successfully (incremental)');
      console.log('   Total embeddings:', data.metadata.total_reference_embeddings);
      
      return data.metadata as SiameseModelMetadata;
      
    } catch (error) {
      console.error('❌ Incremental learning error:', error);
      throw error;
    }
  }

  /**
   * NEW: Add new forged samples to existing student model
   * 
   * @param studentId - Student identifier
   * @param newSamples - Array of new forged signature samples
   * @returns Updated metadata
   */
  async addForgedSamples(
    studentId: string,
    newSamples: any[]
  ): Promise<SiameseModelMetadata> {
    console.log(`🔄 Adding ${newSamples.length} forged samples for ${studentId}`);
    
    try {
      if (newSamples.length === 0) {
        throw new Error('No new samples provided');
      }
      
      const samplesBase64 = newSamples.map(sample => sample.thumbnail);
      
      const response = await fetch(`${this.apiUrl}/api/incremental/add-forged`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          student_id: studentId,
          new_samples: samplesBase64,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to add forged samples');
      }
      
      const data = await response.json();
      console.log('✅ Forged samples added successfully');
      
      return data.metadata as SiameseModelMetadata;
      
    } catch (error) {
      console.error('❌ Failed to add forged samples:', error);
      throw error;
    }
  }

  /**
   * NEW: Check if incremental learning is suitable or full retraining is needed
   * 
   * @param studentId - Student identifier
   * @param newSampleCount - Number of new samples to add
   * @returns Check result with recommendation
   */
  async checkIncrementalLearning(
    studentId: string,
    newSampleCount: number
  ): Promise<IncrementalLearningCheck> {
    try {
      const response = await fetch(`${this.apiUrl}/api/incremental/check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          student_id: studentId,
          new_sample_count: newSampleCount,
        }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Check failed');
      }
      
      const data = await response.json();
      
      if (data.needs_retraining) {
        console.log('⚠️  Full retraining recommended');
        console.log(`   Reason: ${data.reason}`);
      } else {
        console.log('✅ Incremental learning is suitable');
      }
      
      return data as IncrementalLearningCheck;
      
    } catch (error) {
      console.error('❌ Check failed:', error);
      throw error;
    }
  }

  // ============================================================================
  // EXISTING METHODS (Management)
  // ============================================================================

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

  async healthCheck(): Promise<{ 
    status: string; 
    service: string; 
    version: string;
    gpu_available?: boolean;
    tensorflow_version?: string;
    features?: {
      training: boolean;
      verification_1to1: boolean;
      classification_1toN: boolean;
      incremental_learning: boolean;
    };
    classifier?: {
      ready: boolean;
      num_students: number;
      num_embeddings: number;
    };
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
      console.log(`   Version: ${health.version}`);
      console.log(`   GPU: ${health.gpu_available ? '✅ Available' : '❌ Not available'}`);
      
      if (health.features) {
        console.log(`   Features:`);
        console.log(`     Training: ${health.features.training ? '✅' : '❌'}`);
        console.log(`     1:1 Verification: ${health.features.verification_1to1 ? '✅' : '❌'}`);
        console.log(`     1:N Classification: ${health.features.classification_1toN ? '✅' : '❌'}`);
        console.log(`     Incremental Learning: ${health.features.incremental_learning ? '✅' : '❌'}`);
      }
      
      if (health.classifier) {
        console.log(`   Classifier:`);
        console.log(`     Ready: ${health.classifier.ready ? '✅' : '❌'}`);
        console.log(`     Students: ${health.classifier.num_students}`);
        console.log(`     Embeddings: ${health.classifier.num_embeddings}`);
      }
      
      return health;
      
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

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
      
      return data.students;
      
    } catch (error) {
      console.error('Error fetching trained students:', error);
      return [];
    }
  }

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
      for (const student of students) {
        if (student.genuine_samples.length < 2) {
          throw new Error(`Student ${student.student_id}: At least 2 genuine samples required`);
        }
        if (student.forged_samples.length < 5) {
          throw new Error(`Student ${student.student_id}: At least 5 forged samples required`);
        }
      }
      
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
}

// Export singleton instance
export const siameseModelService = SiameseModelService.getInstance();