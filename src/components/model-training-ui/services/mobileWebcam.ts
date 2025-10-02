import { SignatureDetector, type BoundingBox } from '@/utils/signatureDetection';
import { PerspectiveCorrector } from '@/utils/perspectiveCorrection';
export type { BoundingBox } from '@/utils/signatureDetection';

export interface MobileWebcamConfig {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  timeout?: number;
}

export class MobileWebcam {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private config: MobileWebcamConfig;
  private isActive: boolean = false;
  private startTimeout: NodeJS.Timeout | null = null;
  private signatureDetector: SignatureDetector | null = null;
  private perspectiveCorrector: PerspectiveCorrector | null = null;
  private detectedBoxes: BoundingBox[] = [];

  constructor(config: MobileWebcamConfig = {}) {
    this.config = {
      width: config.width || 300,
      height: config.height || 300,
      facingMode: config.facingMode || 'environment',
      timeout: config.timeout || 15000 // 15 seconds timeout
    };
  }

  /**
   * Initialize and start the mobile camera with enhanced error handling and fallbacks
   */
  public async start(): Promise<HTMLVideoElement> {
    try {
      console.log('📱 Starting mobile camera with config:', this.config);
      
      // Check if camera is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported on this device');
      }

      // Check if we're in secure context (HTTPS required for mobile cameras)
      if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        console.warn('⚠️ Camera access requires HTTPS on mobile devices');
      }

      // Create video element
      this.video = document.createElement('video');
      this.video.width = this.config.width || 300;
      this.video.height = this.config.height || 300;
      this.video.autoplay = true;
      this.video.playsInline = true; // Important for mobile
      this.video.muted = true;
      this.video.style.objectFit = 'cover';
      this.video.style.width = '100%';
      this.video.style.height = '100%';

      // Setup timeout mechanism
      const timeoutPromise = new Promise<never>((_, reject) => {
        this.startTimeout = setTimeout(() => {
          reject(new Error('Camera startup timeout - please check permissions and try again'));
        }, this.config.timeout);
      });

      // Try to start camera with fallback constraints
      const cameraPromise = this.startCameraWithFallbacks();
      
      // Race between camera start and timeout
      const videoElement = await Promise.race([cameraPromise, timeoutPromise]);
      
      // Clear timeout on success
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }

      // Create canvas for capturing frames
      this.canvas = document.createElement('canvas');
      this.canvas.width = this.config.width || 300;
      this.canvas.height = this.config.height || 300;

      this.isActive = true;
      console.log('✅ Mobile camera started successfully');
      return videoElement;

    } catch (error) {
      // Cleanup on error
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }
      
      this.cleanup();
      console.error('❌ Error starting mobile camera:', error);
      throw error;
    }
  }

  /**
   * Try to start camera with multiple constraint fallbacks
   */
  private async startCameraWithFallbacks(): Promise<HTMLVideoElement> {
    if (!this.video) {
      throw new Error('Video element not created');
    }

    // Define constraint sets in order of preference
    const constraintSets: MediaStreamConstraints[] = [
      // Primary: High-quality back camera with continuous autofocus for document scanning
      {
        video: {
          width: { ideal: 1280 }, // Higher res for better text/signature detection
          height: { ideal: 720 },
          facingMode: this.config.facingMode,
          // @ts-ignore - Advanced constraints for better document scanning
          focusMode: 'continuous',
          focusDistance: { ideal: 0.3 }, // ~30cm optimal for documents
          advanced: [{
            focusMode: 'continuous',
            torch: false
          }]
        }
      },
      // Fallback 1: Standard resolution with autofocus
      {
        video: {
          width: { ideal: this.config.width },
          height: { ideal: this.config.height },
          facingMode: this.config.facingMode,
          // @ts-ignore
          focusMode: 'continuous'
        }
      },
      // Fallback 2: Same facing mode, any resolution, basic autofocus
      {
        video: {
          facingMode: this.config.facingMode
        }
      },
      // Fallback 3: Switch facing mode with focus
      {
        video: {
          width: { ideal: this.config.width },
          height: { ideal: this.config.height },
          facingMode: this.config.facingMode === 'environment' ? 'user' : 'environment',
          // @ts-ignore
          focusMode: 'continuous'
        }
      },
      // Fallback 4: Any camera, any resolution (last resort)
      {
        video: true
      }
    ];

    let lastError: Error | null = null;

    // Try each constraint set
    for (let i = 0; i < constraintSets.length; i++) {
      const constraints = constraintSets[i];
      console.log(`📷 Attempting camera with constraints set ${i + 1}:`, constraints);
      
      try {
        // Clean up any existing stream
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }

        // Get camera stream
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.video.srcObject = this.stream;

        // Wait for video to be ready
        await this.waitForVideoReady();
        
        console.log(`✅ Camera started successfully with constraints set ${i + 1}`);
        return this.video;

      } catch (error) {
        lastError = error as Error;
        console.warn(`⚠️ Constraints set ${i + 1} failed:`, error);
        
        // Clean up failed attempt
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }
        
        if (this.video) {
          this.video.srcObject = null;
        }
      }
    }

    // All constraint sets failed
    throw lastError || new Error('All camera constraint sets failed');
  }

  /**
   * Wait for video element to be ready with enhanced error handling
   */
  private async waitForVideoReady(): Promise<void> {
    if (!this.video) {
      throw new Error('Video element not created');
    }

    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Video metadata load timeout'));
      }, 10000); // 10 seconds timeout for video metadata

      const cleanup = () => {
        clearTimeout(timeout);
        if (this.video) {
          this.video.onloadedmetadata = null;
          this.video.onerror = null;
        }
      };

      this.video!.onloadedmetadata = () => {
        cleanup();
        this.video!.play()
          .then(() => {
            console.log('🎬 Video playback started');
            resolve();
          })
          .catch((playError) => {
            console.error('❌ Video playback failed:', playError);
            reject(new Error(`Video playback failed: ${playError.message}`));
          });
      };

      this.video!.onerror = (event) => {
        cleanup();
        console.error('❌ Video element error:', event);
        reject(new Error('Video element error - camera may be in use by another application'));
      };
    });
  }

  /**
   * Stop the mobile camera with enhanced cleanup
   */
  public stop(): void {
    console.log('🛑 Stopping mobile camera');
    this.cleanup();
  }

  /**
   * Comprehensive cleanup method
   */
  private cleanup(): void {
    try {
      // Clear any pending timeout
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }

      // Stop video tracks
      if (this.stream) {
        this.stream.getTracks().forEach(track => {
          track.stop();
          console.log('📹 Stopped video track:', track.kind, track.label);
        });
        this.stream = null;
      }

      // Clear video element
      if (this.video) {
        this.video.srcObject = null;
        this.video.onloadedmetadata = null;
        this.video.onerror = null;
        this.video = null;
      }

      this.isActive = false;
      console.log('✅ Mobile camera cleanup completed');
    } catch (error) {
      console.error('❌ Error during camera cleanup:', error);
    }
  }

  /**
   * Get the current video element
   */
  public getVideo(): HTMLVideoElement | null {
    return this.video;
  }

  /**
   * Get the canvas element for frame capture
   */
  public getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }

  /**
   * Capture current frame as canvas
   */
  public captureFrame(): HTMLCanvasElement | null {
    if (!this.video || !this.canvas || !this.video.videoWidth || !this.video.videoHeight) {
      return null;
    }

    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    // Draw current video frame to canvas
    ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    return this.canvas;
  }

  /**
   * Check if camera is active
   */
  public isCameraActive(): boolean {
    return this.isActive;
  }

  /**
   * Switch between front and rear camera
   */
  public async switchCamera(): Promise<HTMLVideoElement> {
    if (!this.isActive) {
      throw new Error('Camera is not active');
    }

    // Stop current camera
    this.stop();

    // Switch facing mode
    this.config.facingMode = this.config.facingMode === 'user' ? 'environment' : 'user';

    // Start camera with new facing mode
    return await this.start();
  }

  /**
   * Get current facing mode
   */
  public getFacingMode(): 'user' | 'environment' | undefined {
    return this.config.facingMode;
  }

    /**
   * Initialize signature detection
   */
    public async initializeDetection(): Promise<void> {
      this.signatureDetector = new SignatureDetector();
      this.perspectiveCorrector = new PerspectiveCorrector();
      console.log('✅ Signature detection initialized');
    }
    
    /**
     * Detect signatures in current frame
     */
    public async detectSignatures(): Promise<BoundingBox[]> {
      if (!this.video || !this.signatureDetector) {
        return [];
      }
      
      try {
        this.detectedBoxes = await this.signatureDetector.detectSignatures(this.video);
        return this.detectedBoxes;
      } catch (error) {
        console.error('Error detecting signatures:', error);
        return [];
      }
    }
    
    /**
     * Get currently detected boxes
     */
    public getDetectedBoxes(): BoundingBox[] {
      return this.detectedBoxes;
    }
    
    /**
     * Set active box (yellow box)
     */
    public setActiveBox(index: number): void {
      this.detectedBoxes.forEach((box, i) => {
        box.isActive = i === index;
      });
    }
    
    /**
     * Get the active (yellow) box
     */
    public getActiveBox(): BoundingBox | null {
      return this.detectedBoxes.find(box => box.isActive) || null;
    }
    
    /**
     * Crop active signature with perspective correction
     */
    public async cropActiveSignature(): Promise<HTMLCanvasElement | null> {
      const activeBox = this.getActiveBox();
      if (!activeBox || !this.video || !this.perspectiveCorrector) {
        return null;
      }
      
      try {
        // Create temp canvas with the cropped region
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = activeBox.width;
        tempCanvas.height = activeBox.height;
        const ctx = tempCanvas.getContext('2d');
        
        if (!ctx) return null;
        
        // Draw the cropped region
        ctx.drawImage(
          this.video,
          activeBox.x, activeBox.y, activeBox.width, activeBox.height,
          0, 0, tempCanvas.width, tempCanvas.height
        );
        
        // Apply perspective correction and resize to 224x224
        const correctedCanvas = await this.perspectiveCorrector.correctPerspective(
          tempCanvas,
          activeBox
        );
        
        return correctedCanvas;
      } catch (error) {
        console.error('Error cropping signature:', error);
        return null;
      }
    }
    
    /**
     * Check if current frame is too blurry
     */
    public isFrameBlurry(threshold: number = 100): boolean {
      if (!this.video) return true;
      
      const canvas = this.captureFrame();
      if (!canvas) return true;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return true;
      
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const variance = this.calculateLaplacianVariance(imageData);
      
      return variance < threshold;
    }
    
    /**
     * Calculate blur metric (Laplacian variance)
     */
    private calculateLaplacianVariance(imageData: ImageData): number {
      const data = imageData.data;
      const width = imageData.width;
      const height = imageData.height;
      
      let sum = 0;
      let count = 0;
      
      // Simple edge detection approximation
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const idx = (y * width + x) * 4;
          const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          
          const topIdx = ((y - 1) * width + x) * 4;
          const topGray = (data[topIdx] + data[topIdx + 1] + data[topIdx + 2]) / 3;
          
          const leftIdx = (y * width + (x - 1)) * 4;
          const leftGray = (data[leftIdx] + data[leftIdx + 1] + data[leftIdx + 2]) / 3;
          
          const edgeStrength = Math.abs(gray - topGray) + Math.abs(gray - leftGray);
          sum += edgeStrength * edgeStrength;
          count++;
        }
      }
      
      return count > 0 ? sum / count : 0;
    }
    
  /**
   * Check if lighting is adequate
   */
  public hasGoodLighting(): boolean {
    if (!this.video) return false;
    
    const canvas = this.captureFrame();
    if (!canvas) return false;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return false;
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
    }
    
    const avgBrightness = sum / (data.length / 4);
    
    // Good range: 50-200 (not too dark, not overexposed)
    return avgBrightness > 50 && avgBrightness < 200;
  }
  
  /**
   * Trigger camera to refocus (single-shot focus)
   * This works on most mobile devices by triggering a one-time focus,
   * then returning to continuous autofocus
   */
  public async triggerFocus(): Promise<boolean> {
    if (!this.stream) {
      console.warn('⚠️ No active stream to trigger focus');
      return false;
    }
    
    const videoTrack = this.stream.getVideoTracks()[0];
    if (!videoTrack) {
      console.warn('⚠️ No video track available');
      return false;
    }
    
    try {
      const capabilities = videoTrack.getCapabilities();
      console.log('📷 Focus capabilities:', capabilities);
      
      // Try single-shot focus (works on most mobile devices)
      // @ts-ignore - Advanced camera API
      if (capabilities.focusMode && Array.isArray(capabilities.focusMode) && 
          capabilities.focusMode.includes('single-shot')) {
        
        await videoTrack.applyConstraints({
          advanced: [{
            // @ts-ignore
            focusMode: 'single-shot'
          }]
        });
        
        console.log('✅ Single-shot focus triggered');
        
        // Return to continuous autofocus after 1.5 seconds
        setTimeout(async () => {
          try {
            await videoTrack.applyConstraints({
              advanced: [{
                // @ts-ignore
                focusMode: 'continuous'
              }]
            });
            console.log('🔄 Returned to continuous autofocus');
          } catch (err) {
            console.log('ℹ️ Could not return to continuous mode');
          }
        }, 1500);
        
        return true;
      } else {
        console.log('ℹ️ Single-shot focus not supported, staying with continuous');
        return false;
      }
    } catch (error) {
      console.warn('⚠️ Could not trigger focus:', error);
      return false;
    }
  }
  
  /**
   * Toggle torch/flash (if available)
   * Useful for low-light signature scanning
   */
  public async toggleTorch(enabled: boolean): Promise<boolean> {
    if (!this.stream) {
      console.warn('⚠️ No active stream to toggle torch');
      return false;
    }
    
    const videoTrack = this.stream.getVideoTracks()[0];
    if (!videoTrack) return false;
    
    try {
      const capabilities = videoTrack.getCapabilities();
      
      // @ts-ignore - Torch capability
      if ('torch' in capabilities && capabilities.torch) {
        await videoTrack.applyConstraints({
          // @ts-ignore
          advanced: [{ torch: enabled }]
        });
        console.log(`💡 Torch ${enabled ? 'enabled' : 'disabled'}`);
        return true;
      } else {
        console.log('ℹ️ Torch not supported on this device');
        return false;
      }
    } catch (error) {
      console.warn('⚠️ Could not toggle torch:', error);
      return false;
    }
  }
  
}