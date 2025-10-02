import type { BoundingBox } from './signatureDetection';

/**
 * Apply perspective correction to make signature look flat
 * Like "document mode" in phone cameras
 */
export class PerspectiveCorrector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  
  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }
  
  /**
   * Apply perspective correction to signature region
   */
  async correctPerspective(
    sourceCanvas: HTMLCanvasElement,
    box: BoundingBox
  ): Promise<HTMLCanvasElement> {
    // For now, return a simple crop with resize
    // Full OpenCV.js implementation would do warpPerspective transform
    
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = 224;
    outputCanvas.height = 224;
    const ctx = outputCanvas.getContext('2d')!;
    
    // Draw with slight sharpening
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(sourceCanvas, 0, 0, 224, 224);
    
    // Apply basic contrast enhancement
    this.enhanceContrast(ctx);
    
    return outputCanvas;
  }
  
  /**
   * Enhance contrast for better signature visibility
   */
  private enhanceContrast(ctx: CanvasRenderingContext2D): void {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const data = imageData.data;
    
    const factor = 1.2; // Contrast factor
    
    for (let i = 0; i < data.length; i += 4) {
      data[i] = this.clamp((data[i] - 128) * factor + 128);
      data[i + 1] = this.clamp((data[i + 1] - 128) * factor + 128);
      data[i + 2] = this.clamp((data[i + 2] - 128) * factor + 128);
    }
    
    ctx.putImageData(imageData, 0, 0);
  }
  
  /**
   * Clamp value between 0 and 255
   */
  private clamp(value: number): number {
    return Math.max(0, Math.min(255, value));
  }
}

/**
 * Check if camera is level (for stabilizer overlay)
 */
export function isCameraLevel(accelerationX?: number, accelerationY?: number): boolean {
  if (typeof accelerationX === 'undefined' || typeof accelerationY === 'undefined') {
    // Fallback if no accelerometer data
    return true;
  }
  
  // Check if device is within 5 degrees of level
  const threshold = 0.5;
  return Math.abs(accelerationX) < threshold && Math.abs(accelerationY) < threshold;
}