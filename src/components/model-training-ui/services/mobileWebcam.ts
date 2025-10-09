//filepath: src/components/model-training-ui/services/mobileWebcam.ts

export interface MobileWebcamConfig {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  timeout?: number;
  zoom?: number;
}

export class MobileWebcam {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private config: MobileWebcamConfig;
  private isActive: boolean = false;
  private startTimeout: NodeJS.Timeout | null = null;
  
  private previewWidth: number = 640;
  private previewHeight: number = 360;

  constructor(config: MobileWebcamConfig = {}) {
    this.config = {
      width: config.width || 300,
      height: config.height || 300,
      facingMode: config.facingMode || 'environment',
      timeout: config.timeout || 15000,
      zoom: config.zoom || 2.0
    };
  }

  public async start(): Promise<HTMLVideoElement> {
    try {
      console.log('📱 Starting mobile camera with config:', this.config);
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported on this device');
      }

      if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        console.warn('⚠️ Camera access requires HTTPS on mobile devices');
      }

      this.video = document.createElement('video');
      this.video.width = this.config.width || 300;
      this.video.height = this.config.height || 300;
      this.video.autoplay = true;
      this.video.playsInline = true;
      this.video.muted = true;
      this.video.style.objectFit = 'cover';
      this.video.style.width = '100%';
      this.video.style.height = '100%';

      const timeoutPromise = new Promise<never>((_, reject) => {
        this.startTimeout = setTimeout(() => {
          reject(new Error('Camera startup timeout'));
        }, this.config.timeout);
      });

      const cameraPromise = this.startCameraWithFallbacks();
      const videoElement = await Promise.race([cameraPromise, timeoutPromise]);
      
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }

      this.canvas = document.createElement('canvas');
      this.canvas.width = this.config.width || 300;
      this.canvas.height = this.config.height || 300;

      this.isActive = true;
      console.log('✅ Mobile camera started with zoom');
      return videoElement;

    } catch (error) {
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }
      
      this.cleanup();
      console.error('❌ Error starting mobile camera:', error);
      throw error;
    }
  }

  private async startCameraWithFallbacks(): Promise<HTMLVideoElement> {
    if (!this.video) {
      throw new Error('Video element not created');
    }

    const constraintSets: MediaStreamConstraints[] = [
      {
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: this.config.facingMode,
          // @ts-ignore
          zoom: this.config.zoom,
          focusMode: 'continuous',
          focusDistance: { ideal: 0.15 },
          imageStabilization: true,
          videoStabilization: true,
          advanced: [{
            focusMode: 'continuous',
            zoom: this.config.zoom
          }]
        }
      },
      {
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: this.config.facingMode,
          // @ts-ignore
          zoom: this.config.zoom,
          focusMode: 'continuous',
          focusDistance: { ideal: 0.15 },
          advanced: [{
            focusMode: 'continuous',
            zoom: this.config.zoom
          }]
        }
      },
      {
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: this.config.facingMode,
          // @ts-ignore
          focusMode: 'continuous',
          focusDistance: { ideal: 0.15 }
        }
      },
      {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: this.config.facingMode,
          // @ts-ignore
          focusMode: 'continuous'
        }
      },
      {
        video: {
          facingMode: this.config.facingMode
        }
      },
      {
        video: true
      }
    ];

    let lastError: Error | null = null;

    for (let i = 0; i < constraintSets.length; i++) {
      const constraints = constraintSets[i];
      console.log(`📷 Attempting camera constraints set ${i + 1}`);
      
      try {
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }

        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.video.srcObject = this.stream;

        await this.waitForVideoReady();
        
        if (this.config.zoom && this.config.zoom > 1) {
          await this.applyZoom(this.config.zoom);
        }
        
        console.log(`✅ Camera started with constraints set ${i + 1}`);
        return this.video;

      } catch (error) {
        lastError = error as Error;
        console.warn(`⚠️ Constraints set ${i + 1} failed:`, error);
        
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }
        
        if (this.video) {
          this.video.srcObject = null;
        }
      }
    }

    throw lastError || new Error('All camera constraint sets failed');
  }

  private async applyZoom(zoomLevel: number): Promise<boolean> {
    if (!this.stream) return false;
    
    const videoTrack = this.stream.getVideoTracks()[0];
    if (!videoTrack) return false;
    
    try {
      const capabilities = videoTrack.getCapabilities();
      
      // @ts-ignore
      if ('zoom' in capabilities && capabilities.zoom) {
        // @ts-ignore
        const { min, max } = capabilities.zoom;
        const clampedZoom = Math.max(min, Math.min(max, zoomLevel));
        
        await videoTrack.applyConstraints({
          // @ts-ignore
          advanced: [{ zoom: clampedZoom }]
        });
        
        console.log(`📷 Applied zoom: ${clampedZoom}x (requested: ${zoomLevel}x, range: ${min}-${max})`);
        return true;
      } else {
        console.log('ℹ️ Optical zoom not supported, camera will use digital zoom via CSS');
        
        if (this.video) {
          this.video.style.transform = `scale(${zoomLevel})`;
          this.video.style.transformOrigin = 'center center';
        }
        return false;
      }
    } catch (error) {
      console.warn('⚠️ Could not apply zoom:', error);
      return false;
    }
  }

  private async waitForVideoReady(): Promise<void> {
    if (!this.video) {
      throw new Error('Video element not created');
    }

    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Video metadata load timeout'));
      }, 10000);

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
        reject(new Error('Video element error'));
      };
    });
  }

  public stop(): void {
    console.log('🛑 Stopping mobile camera');
    this.cleanup();
  }

  private cleanup(): void {
    try {
      if (this.startTimeout) {
        clearTimeout(this.startTimeout);
        this.startTimeout = null;
      }

      if (this.stream) {
        this.stream.getTracks().forEach(track => {
          track.stop();
          console.log('📹 Stopped video track:', track.kind, track.label);
        });
        this.stream = null;
      }

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

  public getVideo(): HTMLVideoElement | null {
    return this.video;
  }

  public getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }

// Replace the captureFrame method in mobileWebcam.ts (around line 312)

/**
 * CRITICAL: Capture frame with 16:9 crop from portrait video
 * This ensures the model sees EXACTLY what the user sees in the UI
 */
public captureFrame(): HTMLCanvasElement | null {
  if (!this.video || !this.canvas || !this.video.videoWidth || !this.video.videoHeight) {
    console.warn('⚠️ captureFrame: Video or canvas not ready', {
      hasVideo: !!this.video,
      hasCanvas: !!this.canvas,
      videoWidth: this.video?.videoWidth,
      videoHeight: this.video?.videoHeight
    });
    return null;
  }

  const ctx = this.canvas.getContext('2d');
  if (!ctx) {
    console.error('❌ captureFrame: Could not get canvas context');
    return null;
  }

  const videoWidth = this.video.videoWidth;
  const videoHeight = this.video.videoHeight;
  
  // Calculate the centered 16:9 crop region
  const targetAspect = 16 / 9;
  const videoAspect = videoWidth / videoHeight;
  
  let cropWidth: number;
  let cropHeight: number;
  let cropX: number;
  let cropY: number;
  
  if (videoAspect > targetAspect) {
    // Video is wider than 16:9 (shouldn't happen in portrait mode)
    cropHeight = videoHeight;
    cropWidth = cropHeight * targetAspect;
    cropX = (videoWidth - cropWidth) / 2;
    cropY = 0;
  } else {
    // Video is taller than 16:9 (portrait mode - expected case)
    // This crops the top and bottom, keeping the center 16:9 portion
    cropWidth = videoWidth;
    cropHeight = cropWidth / targetAspect;
    cropX = 0;
    cropY = (videoHeight - cropHeight) / 2;
  }

  // Set canvas to ML input size (224x224 for MobileNet)
  this.canvas.width = 224;
  this.canvas.height = 224;
  
  // Clear any previous content
  ctx.clearRect(0, 0, 224, 224);
  
  try {
    // Draw ONLY the cropped 16:9 viewport scaled to 224x224
    ctx.drawImage(
      this.video,
      cropX, cropY, cropWidth, cropHeight,  // Source: centered 16:9 crop
      0, 0, 224, 224                         // Destination: 224x224 for ML
    );
    
    console.log(`📸 Captured frame - Source crop: ${cropWidth.toFixed(0)}x${cropHeight.toFixed(0)} at (${cropX.toFixed(0)}, ${cropY.toFixed(0)}) -> ML: 224x224`);
    
    // Verify the canvas has valid image data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const hasData = imageData.data.some(byte => byte !== 0);
    
    if (!hasData) {
      console.error('❌ captureFrame: Canvas is empty (all zeros)');
      return null;
    }
    
    return this.canvas;
    
  } catch (error) {
    console.error('❌ captureFrame: Error drawing to canvas:', error);
    return null;
  }
}

  /**
   * Capture preview frame for display (not used in cleaned-up version)
   */
  public capturePreviewFrame(): HTMLCanvasElement | null {
    if (!this.video || !this.video.videoWidth || !this.video.videoHeight) {
      return null;
    }

    const videoWidth = this.video.videoWidth;
    const videoHeight = this.video.videoHeight;
    
    const targetAspect = 16 / 9;
    const videoAspect = videoWidth / videoHeight;
    
    let cropWidth: number;
    let cropHeight: number;
    let cropX: number;
    let cropY: number;
    
    if (videoAspect > targetAspect) {
      cropHeight = videoHeight;
      cropWidth = cropHeight * targetAspect;
      cropX = (videoWidth - cropWidth) / 2;
      cropY = 0;
    } else {
      cropWidth = videoWidth;
      cropHeight = cropWidth / targetAspect;
      cropX = 0;
      cropY = (videoHeight - cropHeight) / 2;
    }
    
    const previewCanvas = document.createElement('canvas');
    previewCanvas.width = cropWidth;
    previewCanvas.height = cropHeight;
    
    const ctx = previewCanvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    ctx.drawImage(
      this.video,
      cropX, cropY, cropWidth, cropHeight,
      0, 0, cropWidth, cropHeight
    );
    
    return previewCanvas;
  }

  public isCameraActive(): boolean {
    return this.isActive;
  }

  public async switchCamera(): Promise<HTMLVideoElement> {
    if (!this.isActive) {
      throw new Error('Camera is not active');
    }

    this.stop();
    this.config.facingMode = this.config.facingMode === 'user' ? 'environment' : 'user';
    return await this.start();
  }

  public getFacingMode(): 'user' | 'environment' | undefined {
    return this.config.facingMode;
  }

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
      
      // @ts-ignore
      if (capabilities.focusMode && Array.isArray(capabilities.focusMode)) {
        // @ts-ignore
        if (capabilities.focusMode.includes('single-shot')) {
          await videoTrack.applyConstraints({
            advanced: [{
              // @ts-ignore
              focusMode: 'single-shot'
            }]
          });
          
          console.log('✅ Single-shot focus triggered');
          
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
          }, 1000);
          
          return true;
        } else {
          console.log('ℹ️ Single-shot focus not supported');
          return false;
        }
      } else {
        console.log('ℹ️ Focus mode control not available');
        return false;
      }
    } catch (error) {
      console.warn('⚠️ Could not trigger focus:', error);
      return false;
    }
  }

  public async toggleTorch(enabled: boolean): Promise<boolean> {
    if (!this.stream) {
      console.warn('⚠️ No active stream to toggle torch');
      return false;
    }
    
    const videoTrack = this.stream.getVideoTracks()[0];
    if (!videoTrack) return false;
    
    try {
      const capabilities = videoTrack.getCapabilities();
      
      // @ts-ignore
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
  
  public async getCurrentZoom(): Promise<number> {
    if (!this.stream) return 1.0;
    
    const videoTrack = this.stream.getVideoTracks()[0];
    if (!videoTrack) return 1.0;
    
    try {
      const settings = videoTrack.getSettings();
      // @ts-ignore
      return settings.zoom || 1.0;
    } catch (error) {
      return 1.0;
    }
  }
  
  public async setZoom(zoomLevel: number): Promise<boolean> {
    this.config.zoom = zoomLevel;
    return await this.applyZoom(zoomLevel);
  }

  public setPreviewDimensions(width: number, height: number): void {
    this.previewWidth = width;
    this.previewHeight = height;
    console.log(`📱 Preview dimensions set: ${width}x${height} (aspect ratio: ${(width/height).toFixed(2)})`);
  }

  public getPreviewDimensions(): { width: number; height: number } {
    return { width: this.previewWidth, height: this.previewHeight };
  }
}