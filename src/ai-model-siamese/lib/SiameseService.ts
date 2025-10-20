// filepath: src/ai-model-siamese/lib/SiameseService.ts
/**
 * FIXED Siamese Network Service
 * - Single result classification (returns only owner, not top 5)
 * - Non-signature detection
 * - Batch training support
 */

const API_BASE_URL = "https://addressing-connectors-moisture-twice.trycloudflare.com";

export interface TrainingMetadata {
  student_id: string;
  genuine_count: number;
  forged_count: number;
  total_embeddings: number;
  training_date: string;
}

export interface ClassificationResult {
  identified: boolean;
  student_id: string | null;
  confidence: number;
  distance: number;
  decision: 'ACCEPT' | 'UNCERTAIN' | 'UNKNOWN' | 'NON_SIGNATURE' | 'REJECT';
  message: string;
  threshold_info: {
    accept_threshold: number;
    reject_threshold: number;
    nonsig_threshold: number;
  };
}

export interface SampleData {
  thumbnail: string;
  timestamp: number;
  type?: 'genuine' | 'forged';
}

export interface TrainingStatus {
  is_training: boolean;
  progress: number;
  current_student: string | null;
  total_students: number;
  completed_students: number;
  error: string | null;
  start_time: number | null;
}

export interface BatchTrainingRequest {
  students: Array<{
    student_id: string;
    genuine_samples: string[];
    forged_samples: string[];
  }>;
  epochs?: number;
  batch_size?: number;
}

class SiameseService {
  private baseUrl: string;
  private statusPollInterval: number = 2000; // Poll every 2 seconds

  constructor() {
    this.baseUrl = API_BASE_URL;
    console.log('🔧 Siamese Service initialized:', this.baseUrl);
  }

  /**
   * Check if the API server is online
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ API Health:', data);
        return true;
      }
      return false;
    } catch (error) {
      console.error('❌ API health check failed:', error);
      return false;
    }
  }

  /**
   * Get training status (for polling during batch training)
   */
  async getTrainingStatus(): Promise<TrainingStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/training_status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Failed to get training status');
      }

      return await response.json();
    } catch (error) {
      console.error('❌ Failed to get training status:', error);
      return {
        is_training: false,
        progress: 0,
        current_student: null,
        total_students: 0,
        completed_students: 0,
        error: 'Failed to fetch status',
        start_time: null
      };
    }
  }

  /**
   * Batch train multiple students at once
   * All data sent in one request - training continues even if frontend disconnects
   */
  async trainBatch(
    students: Array<{
      studentId: string;
      genuineSamples: SampleData[];
      forgedSamples: SampleData[];
    }>,
    epochs: number = 30,
    batchSize: number = 96
  ): Promise<{ success: boolean; message: string; total_students: number }> {
    try {
      console.log(`🔥 Batch training ${students.length} students...`);

      // Convert to API format
      const requestData: BatchTrainingRequest = {
        students: students.map(student => ({
          student_id: student.studentId,
          genuine_samples: student.genuineSamples.map(s => s.thumbnail),
          forged_samples: student.forgedSamples.map(s => s.thumbnail)
        })),
        epochs,
        batch_size: batchSize
      };

      const response = await fetch(`${this.baseUrl}/train_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Training failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Batch training started:', result);

      return result;

    } catch (error) {
      console.error('❌ Batch training failed:', error);
      throw error;
    }
  }

  /**
   * Train with progress monitoring
   * Returns a promise that resolves when training completes
   * Calls onProgress callback during training
   */
  async trainBatchWithProgress(
    students: Array<{
      studentId: string;
      genuineSamples: SampleData[];
      forgedSamples: SampleData[];
    }>,
    onProgress?: (status: TrainingStatus) => void
  ): Promise<TrainingMetadata[]> {
    // Start batch training
    await this.trainBatch(students);

    // Poll for status until training completes
    return new Promise((resolve, reject) => {
      const pollInterval = setInterval(async () => {
        try {
          const status = await this.getTrainingStatus();

          // Call progress callback
          if (onProgress) {
            onProgress(status);
          }

          // Check if training complete
          if (!status.is_training) {
            clearInterval(pollInterval);

            if (status.error) {
              reject(new Error(status.error));
            } else {
              // Training complete - return metadata
              const metadata: TrainingMetadata[] = students.map(s => ({
                student_id: s.studentId,
                genuine_count: s.genuineSamples.length,
                forged_count: s.forgedSamples.length,
                total_embeddings: 0, // Will be updated by model
                training_date: new Date().toISOString()
              }));
              resolve(metadata);
            }
          }
        } catch (error) {
          clearInterval(pollInterval);
          reject(error);
        }
      }, this.statusPollInterval);
    });
  }

  /**
   * Classify signature - Returns ONLY 1 result (owner or unknown)
   * NO top_k parameter - system finds the single best match
   */
  async classifySignature(signatureImage: string): Promise<ClassificationResult> {
    try {
      console.log('🔍 Classifying signature (finding owner)...');

      const response = await fetch(`${this.baseUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: signatureImage
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Classification failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Classification result:', result);

      // Return the single result
      return {
        identified: result.identified,
        student_id: result.student_id,
        confidence: result.confidence,
        distance: result.distance,
        decision: result.decision,
        message: result.message,
        threshold_info: result.threshold_info
      };

    } catch (error) {
      console.error('❌ Classification failed:', error);
      throw error;
    }
  }

  /**
   * Get model status and statistics
   */
  async getModelStatus(): Promise<{
    is_trained: boolean;
    total_students: number;
    total_embeddings: number;
    last_updated: string | null;
    architecture: string;
    returns_single_result: boolean;
    nonsignature_detection: boolean;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Failed to get model status');
      }

      return await response.json();
    } catch (error) {
      console.error('❌ Failed to get model status:', error);
      return {
        is_trained: false,
        total_students: 0,
        total_embeddings: 0,
        last_updated: null,
        architecture: 'unknown',
        returns_single_result: false,
        nonsignature_detection: false
      };
    }
  }

  /**
   * Delete a student from the model
   */
  async deleteStudent(studentId: string): Promise<void> {
    try {
      console.log(`🗑️  Deleting student: ${studentId}`);

      const response = await fetch(`${this.baseUrl}/delete_student`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: studentId })
      });

      if (!response.ok) {
        throw new Error('Failed to delete student');
      }

      console.log('✅ Student deleted successfully');
    } catch (error) {
      console.error('❌ Failed to delete student:', error);
      throw error;
    }
  }

  /**
   * Export model to cloud (S3)
   */
  async exportToCloud(): Promise<{ success: boolean; url?: string }> {
    try {
      console.log('☁️  Exporting model to cloud...');

      const response = await fetch(`${this.baseUrl}/export_cloud`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Failed to export to cloud');
      }

      const result = await response.json();
      console.log('✅ Model exported to cloud');
      return result;
    } catch (error) {
      console.error('❌ Cloud export failed:', error);
      throw error;
    }
  }

  /**
   * Download model to local PC
   */
  async downloadModel(): Promise<Blob> {
    try {
      console.log('💾 Downloading model...');

      const response = await fetch(`${this.baseUrl}/download_model`, {
        method: 'GET'
      });

      if (!response.ok) {
        throw new Error('Failed to download model');
      }

      const blob = await response.blob();
      console.log('✅ Model downloaded');
      return blob;
    } catch (error) {
      console.error('❌ Model download failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const siameseService = new SiameseService();