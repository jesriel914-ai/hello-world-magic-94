/**
 * Comprehensive canvas-based image augmentation for real-world mobile signature scanning
 * @param canvas The input canvas containing the signature/image to augment
 * @returns A new canvas with the augmented image
 */
/**
 * Refactored canvas-based image augmentation for real-world mobile signature scanning
 * Merged related augmentations into parametric groups for better maintainability
 */
export const augmentImage = (canvas: HTMLCanvasElement): HTMLCanvasElement => {
  const augmentedCanvas = document.createElement('canvas');
  augmentedCanvas.width = canvas.width;
  augmentedCanvas.height = canvas.height;
  const ctx = augmentedCanvas.getContext('2d');
  
  if (!ctx) return canvas;
  
  ctx.clearRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
  
  // Simplified augmentation groups with weights favoring critical scenarios
  const augmentationWeights = [
    0.28, // 0: Geometric transforms (rotation, perspective, crop) - CRITICAL
    0.25, // 1: Focus & motion (blur, shake, distance) - CRITICAL
    0.22, // 2: Lighting & ink (brightness, color temp, pen variation) - CRITICAL
    0.15, // 3: Background & context (paper types, document elements)
    0.10  // 4: Camera artifacts (noise, compression, resolution)
  ];
  
  // Select augmentation type based on weights
  const totalWeight = augmentationWeights.reduce((a, b) => a + b, 0);
  const normalized = augmentationWeights.map(w => w / totalWeight);
  const rand = Math.random();
  let sum = 0;
  let augmentationType = 0;
  
  for (let i = 0; i < normalized.length; i++) {
    sum += normalized[i];
    if (rand <= sum) {
      augmentationType = i;
      break;
    }
  }
  
  switch (augmentationType) {
    case 0: { // GEOMETRIC TRANSFORMS - Rotation, Perspective, Cropping
      ctx.save();
      
      // Random rotation (-15° to +15°)
      const rotation = (Math.random() * 30 - 15) * Math.PI / 180;
      
      // Random perspective skew
      const skewX = (Math.random() * 20 - 10) * Math.PI / 180;
      const skewY = (Math.random() * 20 - 10) * Math.PI / 180;
      
      // Random cropping (simulate partial signature detection)
      const cropAmount = Math.random() * 0.25; // 0-25% crop
      const applyCrop = Math.random() > 0.6; // 40% chance
      
      // Apply transformations
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
      break;
    }
    
    case 1: { // FOCUS & MOTION - Blur, Distance, Hand Shake
      // Distance scaling (0.6x to 1.8x)
      const distance = Math.random();
      const scale = distance < 0.5 ? 
        distance * 1.6 + 0.6 : // 0.6 to 1.4
        (distance - 0.5) * 0.8 + 1.4; // 1.4 to 1.8
      
      // Blur type selection
      const blurType = Math.random();
      let blurFilter = '';
      
      if (blurType < 0.4) { // 40% - Focus blur (depth of field)
        const blurAmount = Math.random() * 1.5 + 0.5;
        blurFilter = `blur(${blurAmount}px)`;
      } else if (blurType < 0.7) { // 30% - Motion blur
        const motionBlur = Math.random() * 1.2;
        blurFilter = `blur(${motionBlur}px)`;
      }
      // 30% - No blur (sharp image)
      
      // Hand shake (micro-movements)
      const applyShake = Math.random() > 0.5;
      
      ctx.save();
      
      if (applyShake) {
        const shakeX = (Math.random() - 0.5) * 3;
        const shakeY = (Math.random() - 0.5) * 3;
        const microRotate = (Math.random() - 0.5) * 0.03;
        ctx.translate(shakeX, shakeY);
        ctx.rotate(microRotate);
      }
      
      // Apply scaling
      const scaledW = canvas.width * scale;
      const scaledH = canvas.height * scale;
      const offsetX = (canvas.width - scaledW) / 2;
      const offsetY = (canvas.height - scaledH) / 2;
      
      if (blurFilter) ctx.filter = blurFilter;
      ctx.drawImage(canvas, offsetX, offsetY, scaledW, scaledH);
      
      ctx.restore();
      break;
    }
    
    case 2: { // LIGHTING & INK - Brightness, Color Temperature, Pen Variation
      // Brightness variation (0.65 to 1.35)
      const brightness = Math.random() * 0.7 + 0.65;
      
      // Contrast variation (0.7 to 1.3)
      const contrast = Math.random() * 0.6 + 0.7;
      
      // Color temperature shift (warm/cool lighting)
      const tempShift = Math.random();
      let hueRotate = 0;
      let saturate = 1;
      
      if (tempShift < 0.3) { // 30% - Warm (yellowish indoor)
        hueRotate = Math.random() * 15;
        saturate = 0.9 + Math.random() * 0.2;
      } else if (tempShift < 0.6) { // 30% - Cool (bluish daylight)
        hueRotate = -(Math.random() * 15);
        saturate = 0.85 + Math.random() * 0.25;
      }
      // 40% - Neutral
      
      // Apply filters
      ctx.filter = `brightness(${brightness}) contrast(${contrast}) hue-rotate(${hueRotate}deg) saturate(${saturate})`;
      ctx.drawImage(canvas, 0, 0);
      
      // Add shadow/lighting gradient (30% chance)
      if (Math.random() > 0.7) {
        const shadowDir = Math.floor(Math.random() * 4);
        const shadowIntensity = Math.random() * 0.25 + 0.1;
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        
        switch (shadowDir) {
          case 0: // Top
            gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
            gradient.addColorStop(0.6, `rgba(0,0,0,${shadowIntensity * 0.3})`);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            break;
          case 1: // Right
            gradient.addColorStop(0, 'rgba(0,0,0,0)');
            gradient.addColorStop(0.4, `rgba(0,0,0,${shadowIntensity * 0.3})`);
            gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
            break;
          case 2: // Bottom
            gradient.addColorStop(0, 'rgba(0,0,0,0)');
            gradient.addColorStop(0.4, `rgba(0,0,0,${shadowIntensity * 0.3})`);
            gradient.addColorStop(1, `rgba(0,0,0,${shadowIntensity})`);
            break;
          case 3: // Left
            gradient.addColorStop(0, `rgba(0,0,0,${shadowIntensity})`);
            gradient.addColorStop(0.6, `rgba(0,0,0,${shadowIntensity * 0.3})`);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            break;
        }
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      
      // Add glare/overexposure spots (20% chance)
      if (Math.random() > 0.8) {
        const glareX = Math.random() * canvas.width;
        const glareY = Math.random() * canvas.height;
        const glareSize = Math.random() * 80 + 40;
        const glareGradient = ctx.createRadialGradient(glareX, glareY, 0, glareX, glareY, glareSize);
        glareGradient.addColorStop(0, 'rgba(255,255,255,0.4)');
        glareGradient.addColorStop(0.5, 'rgba(255,255,255,0.15)');
        glareGradient.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = glareGradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      
      break;
    }
    
    case 3: { // BACKGROUND & CONTEXT - Paper Types, Document Elements
      // Select paper type
      const paperType = Math.floor(Math.random() * 6);
      
      switch (paperType) {
        case 0: // Plain white
          ctx.fillStyle = '#ffffff';
          break;
        case 1: // Yellow pad with lines
          ctx.fillStyle = '#ffffe0';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = '#d8d8b8';
          ctx.lineWidth = 1;
          for (let y = 20; y < canvas.height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
          }
          break;
        case 2: // Beige
          ctx.fillStyle = '#f5f5dc';
          break;
        case 3: // Off-white (slightly gray)
          ctx.fillStyle = '#fafafa';
          break;
        case 4: // Light yellow
          ctx.fillStyle = '#fffacd';
          break;
        case 5: // Notebook paper
          ctx.fillStyle = '#ffffcc';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          // Red margin line
          ctx.strokeStyle = '#ff8888';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(30, 0);
          ctx.lineTo(30, canvas.height);
          ctx.stroke();
          // Horizontal lines
          ctx.strokeStyle = '#d8d8d8';
          ctx.lineWidth = 1;
          for (let y = 20; y < canvas.height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
          }
          break;
      }
      
      if (paperType !== 1 && paperType !== 5) {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      
      // Add document context elements (40% chance)
      if (Math.random() > 0.6) {
        ctx.fillStyle = '#888888';
        ctx.font = '10px Arial';
        
        const contextType = Math.floor(Math.random() * 3);
        switch (contextType) {
          case 0: // Form labels
            ctx.fillText('Name: _______________', 15, 20);
            ctx.fillText('Date: _______________', 15, 35);
            break;
          case 1: // Contract lines
            ctx.strokeStyle = '#cccccc';
            ctx.lineWidth = 1;
            for (let i = 0; i < 5; i++) {
              const y = 15 + i * 25;
              ctx.beginPath();
              ctx.moveTo(15, y);
              ctx.lineTo(canvas.width - 15, y);
              ctx.stroke();
            }
            break;
          case 2: // Text snippets
            ctx.fillText('Lorem ipsum dolor sit...', 10, 15);
            ctx.fillText('Agreement No: _____', 10, 30);
            break;
        }
      }
      
      // Draw signature on top
      ctx.drawImage(canvas, 0, 0);
      
      // Add subtle distractors (30% chance)
      if (Math.random() > 0.7) {
        const numDistractors = Math.floor(Math.random() * 2) + 1;
        for (let i = 0; i < numDistractors; i++) {
          ctx.save();
          ctx.globalAlpha = 0.2;
          ctx.strokeStyle = '#cccccc';
          ctx.lineWidth = 1;
          const x = Math.random() * canvas.width * 0.6;
          const y = Math.random() * canvas.height * 0.6;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.quadraticCurveTo(x + 20, y + 10, x + 40, y + 5);
          ctx.stroke();
          ctx.restore();
        }
      }
      
      break;
    }
    
    case 4: { // CAMERA ARTIFACTS - Noise, Compression, Resolution
      // First draw original
      ctx.drawImage(canvas, 0, 0);
      
      const artifactType = Math.random();
      
      if (artifactType < 0.4) { // 40% - Sensor noise
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        const noiseLevel = Math.random() * 20 + 10;
        
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] < 10) continue;
          const noise = (Math.random() - 0.5) * noiseLevel;
          data[i] = Math.max(0, Math.min(255, data[i] + noise));
          data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
          data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
        }
        
        ctx.putImageData(imageData, 0, 0);
        
      } else if (artifactType < 0.7) { // 30% - Resolution degradation
        const scaleFactor = 0.4 + Math.random() * 0.3; // 0.4-0.7x
        const tmp = document.createElement('canvas');
        tmp.width = Math.max(8, Math.floor(canvas.width * scaleFactor));
        tmp.height = Math.max(8, Math.floor(canvas.height * scaleFactor));
        const tctx = tmp.getContext('2d');
        
        if (tctx) {
          tctx.drawImage(canvas, 0, 0, tmp.width, tmp.height);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
        }
        
      } else { // 30% - Compression artifacts
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        const blockSize = 8;
        
        for (let by = 0; by < canvas.height; by += blockSize) {
          for (let bx = 0; bx < canvas.width; bx += blockSize) {
            let r = 0, g = 0, b = 0, count = 0;
            
            for (let y = 0; y < blockSize && by + y < canvas.height; y++) {
              for (let x = 0; x < blockSize && bx + x < canvas.width; x++) {
                const i = ((by + y) * canvas.width + (bx + x)) * 4;
                r += data[i];
                g += data[i + 1];
                b += data[i + 2];
                count++;
              }
            }
            
            if (count > 0) {
              r = Math.floor(r / count);
              g = Math.floor(g / count);
              b = Math.floor(b / count);
              
              for (let y = 0; y < blockSize && by + y < canvas.height; y++) {
                for (let x = 0; x < blockSize && bx + x < canvas.width; x++) {
                  const i = ((by + y) * canvas.width + (bx + x)) * 4;
                  const noise = (Math.random() - 0.5) * 15;
                  data[i] = Math.max(0, Math.min(255, r + noise));
                  data[i + 1] = Math.max(0, Math.min(255, g + noise));
                  data[i + 2] = Math.max(0, Math.min(255, b + noise));
                }
              }
            }
          }
        }
        
        ctx.putImageData(imageData, 0, 0);
      }
      
      break;
    }
  }
  
  return augmentedCanvas;
};