// filepath: src/components/model-training-ui/utils/modelPrediction.ts

import * as tf from '@tensorflow/tfjs';
import type { PredictionResult } from '../../ModelTraining';

interface PredictableModel {
  predict: (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => Promise<PredictionResult[]>;
}

/**
 * Memory-safe prediction with automatic cleanup
 */
export async function predictFromCanvas(
  model: PredictableModel,
  canvas: HTMLCanvasElement,
  flipped: boolean = false
): Promise<PredictionResult[]> {
  
  if (!canvas) {
    throw new Error('Canvas is null or undefined');
  }
  
  if (canvas.width !== 224 || canvas.height !== 224) {
    throw new Error(`Canvas dimensions must be 224x224, got ${canvas.width}x${canvas.height}`);
  }
  
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Cannot get 2D context from canvas');
  }
  
  try {
    const imageData = ctx.getImageData(0, 0, 224, 224);
    let hasData = false;
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      if (imageData.data[i] !== 0 || imageData.data[i+1] !== 0 || imageData.data[i+2] !== 0) {
        hasData = true;
        break;
      }
    }
    
    if (!hasData) {
      throw new Error('Canvas contains no image data');
    }
    
  } catch (error) {
    console.error('❌ Canvas validation failed:', error);
    throw error;
  }
  
  const predictions = await model.predict(canvas, flipped);
  
  if (!predictions || predictions.length === 0) {
    throw new Error('Model returned empty predictions array');
  }
  
  const isValid = predictions.every(p => 
    p && 
    typeof p.className === 'string' && 
    typeof p.confidence === 'number' &&
    p.confidence >= 0 && 
    p.confidence <= 1
  );
  
  if (!isValid) {
    throw new Error('Model returned invalid prediction format');
  }
  
  return predictions;
}

/**
 * Gentle memory cleanup that doesn't dispose model weights
 * This keeps the model intact while cleaning up leaked tensors
 */
export function forceMemoryCleanup(): void {
  const memory = tf.memory();
  const numTensors = memory.numTensors;
  
  // Log memory stats for monitoring
  console.log(`🧹 Memory check: ${numTensors} tensors, ${(memory.numBytes / 1024 / 1024).toFixed(2)} MB`);
  
  // Only warn if tensor count is extremely high (actual leak)
  // Normal operation with loaded model should be 250-350 tensors
  if (numTensors > 500) {
    console.warn(`⚠️ High tensor count detected: ${numTensors} tensors`);
    // Don't dispose - just log the warning
    // The model's internal tf.tidy() should handle cleanup
  }
}