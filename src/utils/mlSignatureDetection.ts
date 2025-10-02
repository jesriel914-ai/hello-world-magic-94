/**
 * ML-Based Signature Detection
 * Uses TensorFlow.js COCO-SSD for intelligent object detection
 * to pre-filter and focus on paper/document regions
 */

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import type { BoundingBox } from './signatureDetection';

export class MLSignatureDetector {
  private model: cocoSsd.ObjectDetection | null = null;
  private isInitializing: boolean = false;
  private initializationPromise: Promise<void> | null = null;

  /**
   * Initialize the COCO-SSD model
   */
  async initialize(): Promise<void> {
    if (this.model) return;
    
    if (this.isInitializing && this.initializationPromise) {
      return this.initializationPromise;
    }

    this.isInitializing = true;
    this.initializationPromise = this._initialize();
    
    try {
      await this.initializationPromise;
    } finally {
      this.isInitializing = false;
    }
  }

  private async _initialize(): Promise<void> {
    console.log('📦 Loading COCO-SSD model for intelligent detection...');
    
    try {
      // Use MobileNet V2 for better mobile performance
      this.model = await cocoSsd.load({
        base: 'mobilenet_v2'
      });
      
      console.log('✅ COCO-SSD model loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load COCO-SSD model:', error);
      throw error;
    }
  }

  /**
   * Detect regions that might contain paper/documents
   * This helps filter out random objects and focus on likely signature areas
   */
  async detectPaperRegions(video: HTMLVideoElement): Promise<BoundingBox[]> {
    if (!this.model) {
      console.warn('⚠️ Model not initialized');
      return [];
    }

    try {
      const predictions = await this.model.detect(video);
      
      // Filter for objects that commonly have signatures:
      // - book (documents, papers)
      // - laptop (screen with documents)
      // - cell phone (showing document photos)
      // - person (holding paper)
      const relevantClasses = ['book', 'laptop', 'cell phone', 'person'];
      
      const paperRegions: BoundingBox[] = predictions
        .filter(p => relevantClasses.includes(p.class))
        .map(p => ({
          x: p.bbox[0],
          y: p.bbox[1],
          width: p.bbox[2],
          height: p.bbox[3],
          confidence: p.score,
          isActive: false
        }));
      
      console.log(`🔍 ML detected ${paperRegions.length} potential document regions`);
      
      return paperRegions;
    } catch (error) {
      console.error('❌ Error detecting paper regions:', error);
      return [];
    }
  }

  /**
   * Check if a detected box is within or overlaps with ML-detected paper regions
   * This helps validate that our edge-detected signatures are on actual paper
   */
  isWithinPaperRegion(box: BoundingBox, paperRegions: BoundingBox[]): boolean {
    if (paperRegions.length === 0) {
      // If no paper detected, allow all detections (fallback)
      return true;
    }

    return paperRegions.some(region => {
      // Check if box overlaps with paper region
      return this.boxesOverlap(box, region);
    });
  }

  /**
   * Check if two boxes overlap
   */
  private boxesOverlap(a: BoundingBox, b: BoundingBox): boolean {
    return !(
      a.x + a.width < b.x ||
      b.x + b.width < a.x ||
      a.y + a.height < b.y ||
      b.y + b.height < a.y
    );
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}
