/**
 * filepath: src/components/model-training-ui/utils/augmentation.ts
 * IMPROVED MOBILE SIGNATURE AUGMENTATION
 * Trained on: iPhone XR captures
 * Predicts on: Various mobile devices (Android/iOS, different cameras)
 * 
 * Key improvements:
 * 1. Stronger geometric variations (rotation, perspective)
 * 2. More aggressive lighting/color changes (cross-device sensors)
 * 3. Camera quality simulation (blur, noise, compression)
 * 4. Real-world capture conditions (shadows, angles, paper types)
 */
export const augmentImage = (canvas: HTMLCanvasElement): HTMLCanvasElement => {
  const augmentedCanvas = document.createElement('canvas');
  augmentedCanvas.width = canvas.width;
  augmentedCanvas.height = canvas.height;
  const ctx = augmentedCanvas.getContext('2d');
  
  if (!ctx) return canvas;
  
  ctx.clearRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
  
  // ============================================
  // STRATEGY: Mix multiple augmentations
  // Real-world photos have MULTIPLE variations at once
  // ============================================
  
  // Always apply base geometric transform (phone angle)
  applyGeometricTransform(ctx, canvas);
  
  // Then add ONE random condition (40% each)
  const conditionType = Math.random();
  
  if (conditionType < 0.4) {
    // 40% - Lighting variations
    applyLightingConditions(ctx, canvas);
  } else if (conditionType < 0.8) {
    // 40% - Camera quality variations
    applyCameraQuality(ctx, canvas);
  } else {
    // 20% - Extreme conditions (underexposed, overexposed, motion blur)
    applyExtremeConditions(ctx, canvas);
  }
  
  return augmentedCanvas;
};

/**
 * GEOMETRIC TRANSFORM - Always applied
 * Simulates: Phone not perfectly parallel to paper
 */
function applyGeometricTransform(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  ctx.save();
  
  // STRONGER rotation (±18°) - people don't hold phones perfectly straight
  const rotation = (Math.random() * 36 - 18) * Math.PI / 180; // Was ±12°
  
  // STRONGER perspective skew (±12°) - viewing angle matters
  const skewX = (Math.random() * 24 - 12) * Math.PI / 180; // Was ±8°
  const skewY = (Math.random() * 24 - 12) * Math.PI / 180;
  
  // Moderate cropping (25% max) - sometimes signatures are partially out of frame
  const cropAmount = Math.random() * 0.25; // Was 0.15
  const applyCrop = Math.random() > 0.5; // 50% chance (was 35%)
  
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate(rotation);
  ctx.transform(1, Math.tan(skewY), Math.tan(skewX), 1, 0, 0);
  ctx.translate(-canvas.width / 2, -canvas.height / 2);
  
  if (applyCrop) {
    const cropSide = Math.floor(Math.random() * 4);
    let sx = 0, sy = 0, sw = canvas.width, sh = canvas.height;
    
    switch (cropSide) {
      case 0: sy = canvas.height * cropAmount; sh = canvas.height - sy; break;
      case 1: sw = canvas.width * (1 - cropAmount); break;
      case 2: sh = canvas.height * (1 - cropAmount); break;
      case 3: sx = canvas.width * cropAmount; sw = canvas.width - sx; break;
    }
    
    const dx = (canvas.width - sw) / 2;
    const dy = (canvas.height - sh) / 2;
    ctx.drawImage(canvas, sx, sy, sw, sh, dx, dy, sw, sh);
  } else {
    ctx.drawImage(canvas, 0, 0);
  }
  
  ctx.restore();
}

/**
 * LIGHTING CONDITIONS - 40%
 * Simulates: Different lighting (indoor/outdoor, shadows, reflections)
 * CRITICAL for cross-device: Camera sensors react differently to light
 */
function applyLightingConditions(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  // WIDER brightness range (0.6 to 1.4) - Android cameras often darker/brighter
  const brightness = 0.6 + Math.random() * 0.8; // Was 0.75-1.25
  
  // WIDER contrast (0.75 to 1.3) - budget phones have poor dynamic range
  const contrast = 0.75 + Math.random() * 0.55; // Was 0.85-1.15
  
  // STRONGER color temperature shifts
  const tempType = Math.random();
  let hueRotate = 0;
  let saturate = 1;
  
  if (tempType < 0.35) { 
    // Warm indoor (yellowish) - LED/incandescent lights
    hueRotate = 5 + Math.random() * 15; // Was 0-10
    saturate = 0.9 + Math.random() * 0.2;
  } else if (tempType < 0.70) { 
    // Cool outdoor (bluish) - daylight
    hueRotate = -(5 + Math.random() * 15); // Was 0-10
    saturate = 0.85 + Math.random() * 0.25;
  } else {
    // Neutral but desaturated (fluorescent/cloudy)
    saturate = 0.7 + Math.random() * 0.3;
  }
  
  ctx.filter = `brightness(${brightness}) contrast(${contrast}) hue-rotate(${hueRotate}deg) saturate(${saturate})`;
  ctx.drawImage(canvas, 0, 0);
  
  // STRONGER shadow gradient (40% chance)
  if (Math.random() > 0.6) {
    const shadowDir = Math.floor(Math.random() * 4);
    const shadowIntensity = 0.15 + Math.random() * 0.25; // Was 0.08-0.20
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    
    switch (shadowDir) {
      case 0: // Top shadow
        gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
        gradient.addColorStop(0.6, `rgba(0,0,0,${shadowIntensity * 0.3})`);
        gradient.addColorStop(1, 'rgba(0,0,0,0)');
        break;
      case 1: // Right shadow
        gradient.addColorStop(0, 'rgba(0,0,0,0)');
        gradient.addColorStop(0.4, `rgba(0,0,0,${shadowIntensity * 0.3})`);
        gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
        break;
      case 2: // Bottom shadow
        gradient.addColorStop(0, 'rgba(0,0,0,0)');
        gradient.addColorStop(0.4, `rgba(0,0,0,${shadowIntensity * 0.3})`);
        gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
        break;
      case 3: // Left shadow
        gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
        gradient.addColorStop(0.6, `rgba(0,0,0,${shadowIntensity * 0.3})`);
        gradient.addColorStop(1, 'rgba(0,0,0,0)');
        break;
    }
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  
  // STRONGER glare/reflection (25% chance)
  if (Math.random() > 0.75) {
    const glareX = Math.random() * canvas.width;
    const glareY = Math.random() * canvas.height;
    const glareSize = 60 + Math.random() * 80; // Was 40-90
    const glareGradient = ctx.createRadialGradient(glareX, glareY, 0, glareX, glareY, glareSize);
    glareGradient.addColorStop(0, 'rgba(255,255,255,0.4)'); // Was 0.25
    glareGradient.addColorStop(0.5, 'rgba(255,255,255,0.2)'); // Was 0.1
    glareGradient.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = glareGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
}

/**
 * CAMERA QUALITY - 40%
 * Simulates: Different phone cameras (premium vs budget)
 * CRITICAL: iPhone XR is high-quality, but prediction devices vary widely
 */
function applyCameraQuality(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  const qualityType = Math.random();
  
  if (qualityType < 0.4) {
    // 40% - Motion blur (hand shake, moving paper)
    const blurAmount = 0.5 + Math.random() * 1.5; // Was 0.3-0.8
    ctx.filter = `blur(${blurAmount}px)`;
    ctx.drawImage(canvas, 0, 0);
    
  } else if (qualityType < 0.7) {
    // 30% - Camera noise (low-light, budget sensors)
    ctx.drawImage(canvas, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const noiseLevel = 8 + Math.random() * 12; // Was 3-8 (MUCH stronger)
    
    // Apply noise to ALL pixels (not just background)
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * noiseLevel;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    
    ctx.putImageData(imageData, 0, 0);
    
  } else {
    // 30% - JPEG compression artifacts (budget phones, WhatsApp sharing)
    ctx.drawImage(canvas, 0, 0);
    
    // Simulate by adding subtle block artifacts
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const blockSize = 8;
    
    for (let y = 0; y < canvas.height; y += blockSize) {
      for (let x = 0; x < canvas.width; x += blockSize) {
        // Average color in 8x8 block
        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        
        for (let by = 0; by < blockSize && y + by < canvas.height; by++) {
          for (let bx = 0; bx < blockSize && x + bx < canvas.width; bx++) {
            const i = ((y + by) * canvas.width + (x + bx)) * 4;
            sumR += data[i];
            sumG += data[i + 1];
            sumB += data[i + 2];
            count++;
          }
        }
        
        const avgR = sumR / count;
        const avgG = sumG / count;
        const avgB = sumB / count;
        
        // Slightly push pixels toward average (compression effect)
        const compressionStrength = 0.15; // 15% toward block average
        
        for (let by = 0; by < blockSize && y + by < canvas.height; by++) {
          for (let bx = 0; bx < blockSize && x + bx < canvas.width; bx++) {
            const i = ((y + by) * canvas.width + (x + bx)) * 4;
            data[i] = data[i] * (1 - compressionStrength) + avgR * compressionStrength;
            data[i + 1] = data[i + 1] * (1 - compressionStrength) + avgG * compressionStrength;
            data[i + 2] = data[i + 2] * (1 - compressionStrength) + avgB * compressionStrength;
          }
        }
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
  }
}

/**
 * EXTREME CONDITIONS - 20%
 * Simulates: Worst-case real-world scenarios
 */
function applyExtremeConditions(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  const extremeType = Math.random();
  
  if (extremeType < 0.4) {
    // 40% - Severely underexposed (dark room, backlit)
    ctx.filter = 'brightness(0.4) contrast(1.3)';
    ctx.drawImage(canvas, 0, 0);
    
  } else if (extremeType < 0.7) {
    // 30% - Severely overexposed (bright window, flash)
    ctx.filter = 'brightness(1.6) contrast(0.7)';
    ctx.drawImage(canvas, 0, 0);
    
  } else {
    // 30% - Strong motion blur (quick capture)
    ctx.filter = 'blur(2px)';
    
    // Apply directional blur
    ctx.save();
    ctx.globalAlpha = 0.3;
    const angle = Math.random() * Math.PI * 2;
    const distance = 2 + Math.random() * 3;
    
    for (let i = 1; i <= 5; i++) {
      const offset = (distance / 5) * i;
      const dx = Math.cos(angle) * offset;
      const dy = Math.sin(angle) * offset;
      ctx.drawImage(canvas, dx, dy);
    }
    
    ctx.restore();
    ctx.globalAlpha = 1;
    ctx.drawImage(canvas, 0, 0);
  }
  
  // Add paper texture variations (different paper types)
  if (Math.random() > 0.5) {
    const paperType = Math.floor(Math.random() * 5);
    ctx.globalCompositeOperation = 'multiply';
    
    switch (paperType) {
      case 0: // Bright white
        ctx.fillStyle = '#ffffff';
        break;
      case 1: // Cream/off-white
        ctx.fillStyle = '#faf8f3';
        break;
      case 2: // Light gray (old/recycled)
        ctx.fillStyle = '#f0f0f0';
        break;
      case 3: // Yellowish (aged)
        ctx.fillStyle = '#fef9e7';
        break;
      case 4: // Slightly blue (bright white LED)
        ctx.fillStyle = '#f5f7fa';
        break;
    }
    
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over';
  }
}