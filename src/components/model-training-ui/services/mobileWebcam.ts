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
      // Primary: Specific facing mode with ideal resolution
      {
        video: {
          width: { ideal: this.config.width },
          height: { ideal: this.config.height },
          facingMode: this.config.facingMode
        }
      },
      // Fallback 1: Same facing mode, any resolution
      {
        video: {
          facingMode: this.config.facingMode
        }
      },
      // Fallback 2: Switch facing mode
      {
        video: {
          width: { ideal: this.config.width },
          height: { ideal: this.config.height },
          facingMode: this.config.facingMode === 'environment' ? 'user' : 'environment'
        }
      },
      // Fallback 3: Any camera, any resolution
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
}