/**
 * Mobile signature scanning augmentation - Preserves signature features
 * while simulating real-world iPhone XR capture conditions
 */
export const augmentImage = (canvas: HTMLCanvasElement): HTMLCanvasElement => {
  const augmentedCanvas = document.createElement('canvas');
  augmentedCanvas.width = canvas.width;
  augmentedCanvas.height = canvas.height;
  const ctx = augmentedCanvas.getContext('2d');
  
  if (!ctx) return canvas;
  
  ctx.clearRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
  
  // Augmentation strategy weights - focused on mobile camera capture realism
  const augType = Math.random();
  
  if (augType < 0.35) {
    // ========================================
    // GEOMETRIC VARIATION (35% - Most Common)
    // ========================================
    // Simulates: Phone not held perfectly parallel, slight tilting
    
    ctx.save();
    
    // Moderate rotation (-12° to +12°) - realistic hand-held variation
    const rotation = (Math.random() * 24 - 12) * Math.PI / 180;
    
    // Slight perspective skew (max 8°) - preserve signature shape
    const skewX = (Math.random() * 16 - 8) * Math.PI / 180;
    const skewY = (Math.random() * 16 - 8) * Math.PI / 180;
    
    // Minimal cropping (max 15%) - signatures usually fit in frame
    const cropAmount = Math.random() * 0.15;
    const applyCrop = Math.random() > 0.65; // 35% chance
    
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
    
  } else if (augType < 0.65) {
    // ========================================
    // DISTANCE & HAND SHAKE (30%)
    // ========================================
    // Simulates: User moving phone closer/farther, hand tremor
    
    ctx.save();
    
    // Distance scaling (0.75x to 1.35x) - realistic range
    const distance = 0.75 + Math.random() * 0.6;
    
    // Hand shake - small random offset (realistic tremor)
    const shakeX = (Math.random() - 0.5) * 4;
    const shakeY = (Math.random() - 0.5) * 4;
    const microRotate = (Math.random() - 0.5) * 0.04;
    
    // Very light motion blur (50% chance, minimal)
    const applyBlur = Math.random() > 0.5;
    if (applyBlur) {
      const blurAmount = 0.3 + Math.random() * 0.5; // 0.3-0.8px only
      ctx.filter = `blur(${blurAmount}px)`;
    }
    
    ctx.translate(shakeX, shakeY);
    ctx.rotate(microRotate);
    
    const scaledW = canvas.width * distance;
    const scaledH = canvas.height * distance;
    const offsetX = (canvas.width - scaledW) / 2;
    const offsetY = (canvas.height - scaledH) / 2;
    
    ctx.drawImage(canvas, offsetX, offsetY, scaledW, scaledH);
    ctx.restore();
    
  } else if (augType < 0.90) {
    // ========================================
    // LIGHTING CONDITIONS (25%)
    // ========================================
    // Simulates: Indoor/outdoor lighting, shadows, exposure changes
    
    // Brightness (0.75 to 1.25) - realistic iPhone XR auto-exposure
    const brightness = 0.75 + Math.random() * 0.5;
    
    // Contrast (0.85 to 1.15) - preserve stroke visibility
    const contrast = 0.85 + Math.random() * 0.3;
    
    // Color temperature - warm indoor vs cool outdoor
    const tempType = Math.random();
    let hueRotate = 0;
    let saturate = 1;
    
    if (tempType < 0.35) { // Warm indoor (yellowish)
      hueRotate = Math.random() * 10;
      saturate = 0.95 + Math.random() * 0.1;
    } else if (tempType < 0.70) { // Cool outdoor (bluish)
      hueRotate = -(Math.random() * 10);
      saturate = 0.9 + Math.random() * 0.15;
    }
    // else: neutral lighting
    
    ctx.filter = `brightness(${brightness}) contrast(${contrast}) hue-rotate(${hueRotate}deg) saturate(${saturate})`;
    ctx.drawImage(canvas, 0, 0);
    
    // Subtle shadow gradient (25% chance) - hand or paper shadow
    if (Math.random() > 0.75) {
      const shadowDir = Math.floor(Math.random() * 4);
      const shadowIntensity = 0.08 + Math.random() * 0.12; // Very subtle
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      
      switch (shadowDir) {
        case 0: // Top shadow
          gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
          gradient.addColorStop(0.5, `rgba(0,0,0,${shadowIntensity * 0.4})`);
          gradient.addColorStop(1, 'rgba(0,0,0,0)');
          break;
        case 1: // Right shadow
          gradient.addColorStop(0, 'rgba(0,0,0,0)');
          gradient.addColorStop(0.5, `rgba(0,0,0,${shadowIntensity * 0.4})`);
          gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
          break;
        case 2: // Bottom shadow
          gradient.addColorStop(0, 'rgba(0,0,0,0)');
          gradient.addColorStop(0.5, `rgba(0,0,0,${shadowIntensity * 0.4})`);
          gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
          break;
        case 3: // Left shadow
          gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
          gradient.addColorStop(0.5, `rgba(0,0,0,${shadowIntensity * 0.4})`);
          gradient.addColorStop(1, 'rgba(0,0,0,0)');
          break;
      }
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Mild glare (15% chance) - paper reflection
    if (Math.random() > 0.85) {
      const glareX = Math.random() * canvas.width;
      const glareY = Math.random() * canvas.height;
      const glareSize = 40 + Math.random() * 50;
      const glareGradient = ctx.createRadialGradient(glareX, glareY, 0, glareX, glareY, glareSize);
      glareGradient.addColorStop(0, 'rgba(255,255,255,0.25)');
      glareGradient.addColorStop(0.5, 'rgba(255,255,255,0.1)');
      glareGradient.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = glareGradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
  } else {
    // ========================================
    // PAPER/BACKGROUND (10%)
    // ========================================
    // Simulates: Different paper types user might sign on
    
    const paperType = Math.floor(Math.random() * 4);
    
    switch (paperType) {
      case 0: // Plain white
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        break;
      case 1: // Off-white/cream
        ctx.fillStyle = '#fafaf5';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        break;
      case 2: // Light beige
        ctx.fillStyle = '#f5f5dc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        break;
      case 3: // Very light gray (photocopied paper)
        ctx.fillStyle = '#fafafa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        break;
    }
    
    // Draw signature on top
    ctx.drawImage(canvas, 0, 0);
    
    // Very subtle noise (paper texture) - 40% chance
    if (Math.random() > 0.6) {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const noiseLevel = 3 + Math.random() * 5; // Very minimal noise
      
      for (let i = 0; i < data.length; i += 4) {
        // Only add noise to non-signature areas (bright pixels)
        if (data[i] > 200 && data[i + 1] > 200 && data[i + 2] > 200) {
          const noise = (Math.random() - 0.5) * noiseLevel;
          data[i] = Math.max(0, Math.min(255, data[i] + noise));
          data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
          data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
    }
  }
  
  return augmentedCanvas;
};