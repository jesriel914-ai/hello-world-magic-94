// filepath: src/ai-model-siamese/lib/SiameseService.ts
/**
 * Siamese Network Service - Client for Python backend
 * Handles communication with Flask API server via Cloudflare tunnel
 */

const API_BASE_URL = "https://closes-prince-such-floyd.trycloudflare.com";

export interface TrainingStatus {
  is_training: boolean;
  progress: number;
  current_student: string | null;
  total_students: number;
  completed_students: number;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
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

export interface VerificationResult {
  verified: boolean;
  student_id: string;
  distance: number;
  confidence: number;
  message: string;
}

export interface StudentData {
  studentId: string;
  genuineSamples: Array<{
    thumbnail: string;
    timestamp: number;
  }>;
}

class SiameseService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Update API base URL (for when Cloudflare tunnel URL changes)
   */
  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  /**
   * Health check - verify API is online
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      return data.status === 'online';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Train model with batch of students (background training)
   */
  async trainBatch(students: StudentData[]): Promise<{
    message: string;
    total_students: number;
    status: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/train/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ students }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Training failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Train batch failed:', error);
      throw error;
    }
  }

  /**
   * Get current training status
   */
  async getTrainingStatus(): Promise<TrainingStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/train/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to get training status');
      }

      return await response.json();
    } catch (error) {
      console.error('Get training status failed:', error);
      throw error;
    }
  }

  /**
   * Classify/identify signature owner
   */
  async classifySignature(imageBase64: string): Promise<ClassificationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Classification failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Classify signature failed:', error);
      throw error;
    }
  }

  /**
   * Verify if signature belongs to claimed student (1:1 verification)
   */
  async verifySignature(
    imageBase64: string,
    studentId: string
  ): Promise<VerificationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          studentId: studentId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Verification failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Verify signature failed:', error);
      throw error;
    }
  }

  /**
   * Export model to cloud storage
   */
  async exportToCloud(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/model/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Export failed');
      }
    } catch (error) {
      console.error('Export to cloud failed:', error);
      throw error;
    }
  }

  /**
   * Download model as ZIP file
   */
  async downloadModel(): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseUrl}/api/model/download`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error('Download failed');
      }

      return await response.blob();
    } catch (error) {
      console.error('Download model failed:', error);
      throw error;
    }
  }

  /**
   * Get list of trained students
   */
  async getTrainedStudents(): Promise<{
    students: string[];
    total: number;
    last_updated: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/students`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to get trained students');
      }

      return await response.json();
    } catch (error) {
      console.error('Get trained students failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const siameseService = new SiameseService();
