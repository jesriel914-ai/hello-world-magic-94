/**
 * Comprehensive canvas-based image augmentation for real-world mobile signature scanning
 * @param canvas The input canvas containing the signature/image to augment
 * @returns A new canvas with the augmented image
 */
export const augmentImage = (canvas: HTMLCanvasElement): HTMLCanvasElement => {
  const augmentedCanvas = document.createElement('canvas');
  augmentedCanvas.width = canvas.width;
  augmentedCanvas.height = canvas.height;
  const ctx = augmentedCanvas.getContext('2d');
  
  if (!ctx) return canvas;
  
  // Clear canvas
  ctx.clearRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
  
  // Apply comprehensive augmentations for all real-world mobile scanning challenges
  // Using a weighted distribution to favor more common scenarios
  // Weights must cover all switch cases (0..14)
  const augmentationWeights = [
    0.14, // 0: Distance + focus
    0.14, // 1: Angle variation
    0.12, // 2: Paper background
    0.10, // 3: Motion blur
    0.10, // 4: Lighting variation
    0.07, // 5: Pen characteristics
    0.07, // 6: Perspective distortion
    0.07, // 7: Partial signature
    0.06, // 8: Document context
    0.05, // 9: Camera artifacts
    0.03, // 10: Hand shake + shadows
    0.02, // 11: Lens distortion + perspective
    0.02, // 12: Resolution/quality degradation (new)
    0.01, // 13: Variable focus/depth of field
    0.02  // 14: Video compression artifacts
  ];

  // Normalize and select augmentation based on weights
  const totalWeight = augmentationWeights.reduce((a, b) => a + b, 0);
  const normalized = augmentationWeights.map(w => w / totalWeight);
  const rand = Math.random();
  let sum = 0;
  let augmentationType = normalized.length - 1;
  for (let i = 0; i < normalized.length; i++) {
    sum += normalized[i];
    if (rand <= sum) {
      augmentationType = i;
      break;
    }
  }
  
  switch (augmentationType) {
    case 0: { // Enhanced Distance and Focus Simulation
      // Extended range with non-linear distribution
      const distance = Math.random();
      const scale = distance < 0.6 ? 
        distance * 1.5 + 0.4 :  // 0.4 to 1.3
        (distance - 0.6) * 1.75 + 1.3; // 1.3 to 2.0
      
      // Dynamic focus point (somewhere on the signature)
      const focusPoint = {
        x: canvas.width * (0.3 + Math.random() * 0.4),
        y: canvas.height * (0.3 + Math.random() * 0.4)
      };
      
      // Create a temporary canvas for depth effects
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) break;
      
      // Draw scaled image
      const scaledWidth = canvas.width * scale;
      const scaledHeight = canvas.height * scale;
      const offsetX = (canvas.width - scaledWidth) / 2;
      const offsetY = (canvas.height - scaledHeight) / 2;
      
      tempCtx.drawImage(canvas, offsetX, offsetY, scaledWidth, scaledHeight);
      
      // Apply depth of field effect
      const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const data = imageData.data;
      const focusRadius = Math.min(canvas.width, canvas.height) * (0.2 + Math.random() * 0.3);
      const maxBlur = 1.5 + Math.random() * 1.5; // Max blur amount
      
      // Simple depth-based blur (in practice, use a proper blur algorithm for production)
      for (let y = 0; y < tempCanvas.height; y += 2) {
        for (let x = 0; x < tempCanvas.width; x += 2) {
          // Calculate distance from focus point
          const dist = Math.sqrt(Math.pow(x - focusPoint.x, 2) + Math.pow(y - focusPoint.y, 2));
          // Calculate blur amount based on distance from focus point
          const blurAmount = Math.min(maxBlur, Math.max(0, (dist - focusRadius) / 20));
          
          if (blurAmount > 0.3) {
            const i = (y * tempCanvas.width + x) * 4;
            // Simple box blur (in practice, use a better blur algorithm)
            const r = (data[i] + data[i + 4] + data[i + tempCanvas.width * 4] + data[i + tempCanvas.width * 4 + 4]) / 4;
            const g = (data[i + 1] + data[i + 5] + data[i + tempCanvas.width * 4 + 1] + data[i + tempCanvas.width * 4 + 5]) / 4;
            const b = (data[i + 2] + data[i + 6] + data[i + tempCanvas.width * 4 + 2] + data[i + tempCanvas.width * 4 + 6]) / 4;
            
            // Apply the blurred color
            data[i] = r; data[i + 1] = g; data[i + 2] = b;
          }
        }
      }
      
      tempCtx.putImageData(imageData, 0, 0);
      
      // Draw the result to the main canvas
      ctx.drawImage(tempCanvas, 0, 0);
      
      // Add subtle camera focus effect (highlight in-focus area)
      if (Math.random() > 0.7) { // 30% chance to show focus effect
        ctx.save();
        ctx.strokeStyle = 'rgba(100, 255, 100, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(focusPoint.x, focusPoint.y, focusRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      
      break;
    }
      
    case 1: { // Angle variation (SECOND MOST IMPORTANT)
      ctx.save();
      const angle = (Math.random() * 30 - 15) * Math.PI / 180; // -15 to +15 degrees
      ctx.translate(augmentedCanvas.width / 2, augmentedCanvas.height / 2);
      ctx.rotate(angle);
      ctx.translate(-augmentedCanvas.width / 2, -augmentedCanvas.height / 2);
      ctx.drawImage(canvas, 0, 0);
      ctx.restore();
      break;
    }
      
    case 2: { // Paper background variation (IMPORTANT)
      // Different paper types: white, yellow pad (with lines), brownish paper, etc.
      const paperType = Math.floor(Math.random() * 6);
      
      switch (paperType) {
        case 0: // Plain white paper
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          break;
        case 1: { // Yellow pad with horizontal lines
          ctx.fillStyle = '#ffffe0';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          // Add horizontal lines
          {
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            const lineSpacing = 20;
            for (let y = lineSpacing; y < augmentedCanvas.height; y += lineSpacing) {
              ctx.beginPath();
              ctx.moveTo(0, y);
              ctx.lineTo(augmentedCanvas.width, y);
              ctx.stroke();
            }
          }
          break;
        }
        case 2: // Light beige
          ctx.fillStyle = '#f5f5dc';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          break;
        case 3: // Brownish paper
          ctx.fillStyle = '#d2b48c';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          break;
        case 4: // Off-white
          ctx.fillStyle = '#fafafa';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          break;
        case 5: // Light yellow
          ctx.fillStyle = '#fffacd';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          break;
      }
      ctx.drawImage(canvas, 0, 0);
      break;
    }
      
    case 3: { // Motion blur simulation (IMPORTANT)
      const blurAmount = Math.random() * 1.5; // 0 to 1.5px blur
      ctx.filter = `blur(${blurAmount}px)`;
      ctx.drawImage(canvas, 0, 0);
      break;
    }
      
    case 4: { // Lighting variation (IMPORTANT)
      const brightness = Math.random() * 0.6 + 0.7; // 0.7 to 1.3
      const contrast = Math.random() * 0.5 + 0.75; // 0.75 to 1.25
      ctx.filter = `brightness(${brightness}) contrast(${contrast})`;
      ctx.drawImage(canvas, 0, 0);
      break;
    }
      
    case 5: { // Pen characteristics simulation (IMPORTANT)
      // Simulate different pen colors: black or blue tint
      const penTypes = [
        { contrast: 1.2, brightness: 0.9 }, // Bold black pen
        { contrast: 1.1, brightness: 0.95 }  // Blue pen tint
      ];
      const penType = penTypes[Math.floor(Math.random() * penTypes.length)];
      ctx.filter = `brightness(${penType.brightness}) contrast(${penType.contrast})`;
      ctx.drawImage(canvas, 0, 0);
      break;
    }
      
    case 6: { // Perspective distortion (CRITICAL for mobile scanning)
      // Simulate viewing signature at an angle (trapezoidal distortion)
      ctx.save();
      
      // Calculate perspective distortion parameters
      const maxSkew = 25; // Maximum skew in degrees
      const skewX = (Math.random() * maxSkew - maxSkew/2) * Math.PI / 180; // -12.5 to +12.5 degrees
      const skewY = (Math.random() * maxSkew - maxSkew/2) * Math.PI / 180; // -12.5 to +12.5 degrees
      
      // Apply perspective transformation using skew
      ctx.translate(augmentedCanvas.width / 2, augmentedCanvas.height / 2);
      ctx.transform(1, Math.tan(skewY), Math.tan(skewX), 1, 0, 0);
      ctx.translate(-augmentedCanvas.width / 2, -augmentedCanvas.height / 2);
      
      ctx.drawImage(canvas, 0, 0);
      ctx.restore();
      break;
    }
      
    case 7: { // Partial signature/cropping (VERY IMPORTANT for real-time detection)
      // Simulate real-time bounding box catching partial signatures
      const cropAmount = Math.random() * 0.3 + 0.1; // 10% to 40% crop
      const cropSide = Math.floor(Math.random() * 4); // 0: top, 1: right, 2: bottom, 3: left
      
      let sourceX = 0, sourceY = 0, sourceWidth = canvas.width, sourceHeight = canvas.height;
      
      switch (cropSide) {
        case 0: // Crop top
          sourceY = canvas.height * cropAmount;
          sourceHeight = canvas.height - sourceY;
          break;
        case 1: // Crop right
          sourceWidth = canvas.width * (1 - cropAmount);
          break;
        case 2: // Crop bottom
          sourceHeight = canvas.height * (1 - cropAmount);
          break;
        case 3: // Crop left
          sourceX = canvas.width * cropAmount;
          sourceWidth = canvas.width - sourceX;
          break;
      }
      
      // Center the cropped signature
      const destX = (augmentedCanvas.width - sourceWidth) / 2;
      const destY = (augmentedCanvas.height - sourceHeight) / 2;
      
      ctx.drawImage(canvas, sourceX, sourceY, sourceWidth, sourceHeight, destX, destY, sourceWidth, sourceHeight);
      break;
    }
      
    case 8: { // Document Context + Multi-signature Interference (MERGED - IMPORTANT for real-world scanning)
      // Simulate signatures on documents with text, forms, AND potential signature distractors
      const documentType = Math.floor(Math.random() * 4);
      
      // Draw background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
      
      switch (documentType) {
        case 0: { // Form with lines
          ctx.strokeStyle = '#cccccc';
          ctx.lineWidth = 1;
          const lineSpacing = 15;
          for (let y = 10; y < augmentedCanvas.height - 10; y += lineSpacing) {
            ctx.beginPath();
            ctx.moveTo(10, y);
            ctx.lineTo(augmentedCanvas.width - 10, y);
            ctx.stroke();
          }
          // Add some form text
          ctx.fillStyle = '#666666';
          ctx.font = '12px Arial';
          ctx.fillText('Name: _______________', 20, 25);
          ctx.fillText('Date: _______________', 20, 40);
          ctx.fillText('Signature: __________', 20, 55);
          break;
        }
        case 1: { // Document with paragraph text
          ctx.fillStyle = '#666666';
          ctx.font = '10px Arial';
          const text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ';
          const words = text.split(' ');
          let line = '';
          let y = 15;
          
          for (let i = 0; i < words.length; i++) {
            const testLine = line + words[i] + ' ';
            const metrics = ctx.measureText(testLine);
            
            if (metrics.width > augmentedCanvas.width - 20 && i > 0) {
              ctx.fillText(line, 10, y);
              line = words[i] + ' ';
              y += 12;
            } else {
              line = testLine;
            }
          }
          ctx.fillText(line, 10, y);
          break;
        }
        case 2: { // Contract-style document
          ctx.strokeStyle = '#cccccc';
          ctx.lineWidth = 1;
          // Draw contract lines
          for (let i = 0; i < 8; i++) {
            const y = 15 + i * 20;
            ctx.beginPath();
            ctx.moveTo(15, y);
            ctx.lineTo(augmentedCanvas.width - 15, y);
            ctx.stroke();
          }
          // Add contract labels
          ctx.fillStyle = '#666666';
          ctx.font = '10px Arial';
          ctx.fillText('Agreement No:', 20, 12);
          ctx.fillText('Party A:', 20, 32);
          ctx.fillText('Party B:', 20, 52);
          ctx.fillText('Authorized Signature:', 20, 152);
          break;
        }
        case 3: { // Notebook paper
          ctx.fillStyle = '#ffffcc';
          ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          ctx.strokeStyle = '#ff6666';
          ctx.lineWidth = 1;
          // Vertical red line
          ctx.beginPath();
          ctx.moveTo(30, 0);
          ctx.lineTo(30, augmentedCanvas.height);
          ctx.stroke();
          // Horizontal lines
          ctx.strokeStyle = '#cccccc';
          const lineSpacing = 20;
          for (let y = lineSpacing; y < augmentedCanvas.height; y += lineSpacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(augmentedCanvas.width, y);
            ctx.stroke();
          }
          break;
        }
      }
      
      // Add signature distractors (merged from case 17)
      const numDistractors = Math.floor(Math.random() * 2) + 1; // 1-2 distractors only
      
      for (let i = 0; i < numDistractors; i++) {
        const x = Math.random() * augmentedCanvas.width * 0.6;
        const y = Math.random() * augmentedCanvas.height * 0.6;
        const scale = Math.random() * 0.3 + 0.2; // 20-50% scale
        
        ctx.save();
        ctx.globalAlpha = 0.25;
        ctx.translate(x, y);
        ctx.scale(scale, scale);
        ctx.rotate(Math.random() * 0.3 - 0.15);
        
        // Draw simple scribble-like distractor
        ctx.strokeStyle = '#dddddd';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(15, 8, 30, 4);
        ctx.quadraticCurveTo(45, 0, 60, 10);
        ctx.stroke();
        
        ctx.restore();
      }
      
      // Draw the signature on top of the background
      ctx.drawImage(canvas, 0, 0);
      break;
    }
      
    case 9: { // Enhanced Mobile Camera Artifacts
      // First draw the original image
      ctx.drawImage(canvas, 0, 0);
      
      // Simulate different camera sensors (phone models)
      const cameraProfiles = [
        { noise: 0.3, sharpness: 0.8, colorShift: 0.1 }, // Low-end phone
        { noise: 0.15, sharpness: 0.9, colorShift: 0.05 }, // Mid-range phone
        { noise: 0.05, sharpness: 1.0, colorShift: 0.02 }  // High-end phone
      ];
      
      const profile = cameraProfiles[Math.floor(Math.random() * cameraProfiles.length)];
      const isLowLight = Math.random() > 0.7; // 30% chance of low-light
      
      // Get image data
      const imageData = ctx.getImageData(0, 0, augmentedCanvas.width, augmentedCanvas.height);
      const data = imageData.data;
      
      // Apply camera profile and lighting conditions
      for (let i = 0; i < data.length; i += 4) {
        // Skip transparent pixels
        if (data[i + 3] < 10) continue;
        
        // Apply color shift (subtle camera-specific color cast)
        const rShift = (Math.random() - 0.5) * profile.colorShift * 50;
        const bShift = (Math.random() - 0.5) * profile.colorShift * 50;
        
        // Base noise level from camera profile
        let noise = (Math.random() - 0.5) * profile.noise * 40;
        
        // Apply lighting flicker (subtle brightness variation)
        const flickerIntensity = Math.random() * 0.2 + 0.9; // 90-110% brightness range
        const flickerPattern = 0.5 + Math.sin(Date.now() * 0.005) * 0.1; // Slow, subtle flicker
        const brightness = flickerIntensity * flickerPattern;
        
        // Increase noise in low light
        if (isLowLight) {
          noise *= 2.5;
          // Reduce brightness and contrast in low light
          const lowLightBrightness = 0.8 + Math.random() * 0.2; // 0.8-1.0
          const contrast = 0.7 + Math.random() * 0.3;   // 0.7-1.0
          
          // Apply brightness, contrast and noise
          data[i] = Math.max(0, Math.min(255, 
            ((data[i] / 255 - 0.5) * contrast + 0.5) * 255 * lowLightBrightness * brightness + noise + rShift));
          data[i + 1] = Math.max(0, Math.min(255, 
            ((data[i + 1] / 255 - 0.5) * contrast + 0.5) * 255 * lowLightBrightness * brightness + noise));
          data[i + 2] = Math.max(0, Math.min(255, 
            ((data[i + 2] / 255 - 0.5) * contrast + 0.5) * 255 * lowLightBrightness * brightness + noise + bShift));
        } else {
          // Normal lighting - just add noise and flicker
          data[i] = Math.max(0, Math.min(255, data[i] * brightness + noise + rShift));
          data[i + 1] = Math.max(0, Math.min(255, data[i + 1] * brightness + noise));
          data[i + 2] = Math.max(0, Math.min(255, data[i + 2] * brightness + noise + bShift));
        }
        
        // Apply sharpness (simulate different camera processing)
        if (i > 4 && i < data.length - 4 && Math.random() > profile.sharpness) {
          // Slight blur effect for lower sharpness - only apply to non-edge pixels
          data[i] = data[i] * 0.8 + data[i - 4] * 0.1 + data[i + 4] * 0.1;
          data[i + 1] = data[i + 1] * 0.8 + data[i - 3] * 0.1 + data[i + 5] * 0.1;
          data[i + 2] = data[i + 2] * 0.8 + data[i - 2] * 0.1 + data[i + 6] * 0.1;
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
      break;
    }
      
    case 10: { // Mobile Device Handling - Hand Shake + Shadow Effects (MERGED - IMPORTANT for mobile scanning)
      // Simulate small rapid movements AND shadow effects during scanning
      ctx.save();
      
      // Apply hand shake/tremor
      const tremorIntensity = 2; // Maximum 2 pixel displacement
      const steps = 5; // Number of tremor steps
      
      for (let step = 0; step < steps; step++) {
        const offsetX = (Math.random() - 0.5) * tremorIntensity;
        const offsetY = (Math.random() - 0.5) * tremorIntensity;
        const microRotation = (Math.random() - 0.5) * 0.02; // Very small rotation
        
        ctx.translate(offsetX, offsetY);
        ctx.rotate(microRotation);
      }
      
      ctx.drawImage(canvas, 0, 0);
      ctx.restore();
      
      // Apply shadow effects (merged from case 11)
      const shadowDirection = Math.floor(Math.random() * 4); // 0: top, 1: right, 2: bottom, 3: left
      const shadowIntensity = Math.random() * 0.3 + 0.1; // 0.1 to 0.4 intensity
      
      const gradient = ctx.createLinearGradient(0, 0, augmentedCanvas.width, augmentedCanvas.height);
      
      switch (shadowDirection) {
        case 0: // Top shadow
          gradient.addColorStop(0, `rgba(0, 0, 0, ${shadowIntensity})`);
          gradient.addColorStop(0.5, `rgba(0, 0, 0, ${shadowIntensity * 0.5})`);
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          break;
        case 1: // Right shadow
          gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
          gradient.addColorStop(0.5, `rgba(0, 0, 0, ${shadowIntensity * 0.5})`);
          gradient.addColorStop(1, `rgba(0, 0, 0, ${shadowIntensity})`);
          break;
        case 2: // Bottom shadow
          gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
          gradient.addColorStop(0.5, `rgba(0, 0, 0, ${shadowIntensity * 0.5})`);
          gradient.addColorStop(1, `rgba(0, 0, 0, ${shadowIntensity})`);
          break;
        case 3: // Left shadow
          gradient.addColorStop(0, `rgba(0, 0, 0, ${shadowIntensity})`);
          gradient.addColorStop(0.5, `rgba(0, 0, 0, ${shadowIntensity * 0.5})`);
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          break;
      }
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
      break;
    }
      
    case 11: { // Mobile Perspective & Lens Effects - Perspective + Lens Distortion (MERGED - CRITICAL for mobile scanning)
      // Simulate both perspective distortion AND lens distortion (merged from cases 6 and 12)
      ctx.save();
      
      // Apply perspective distortion (from case 6)
      const maxSkew = 25; // Maximum skew in degrees
      const skewX = (Math.random() * maxSkew - maxSkew/2) * Math.PI / 180; // -12.5 to +12.5 degrees
      const skewY = (Math.random() * maxSkew - maxSkew/2) * Math.PI / 180; // -12.5 to +12.5 degrees
      
      // Apply perspective transformation using skew
      ctx.translate(augmentedCanvas.width / 2, augmentedCanvas.height / 2);
      ctx.transform(1, Math.tan(skewY), Math.tan(skewX), 1, 0, 0);
      ctx.translate(-augmentedCanvas.width / 2, -augmentedCanvas.height / 2);
      
      // Apply lens distortion (from case 12)
      const distortionType = Math.random() > 0.5 ? 'barrel' : 'pincushion';
      const distortionStrength = Math.random() * 0.1 + 0.95; // 0.95 to 1.05
      
      const centerX = augmentedCanvas.width / 2;
      const centerY = augmentedCanvas.height / 2;
      
      ctx.translate(centerX, centerY);
      ctx.scale(distortionType === 'barrel' ? distortionStrength : 1 / distortionStrength, 
                 distortionType === 'barrel' ? 1 / distortionStrength : distortionStrength);
      ctx.translate(-centerX, -centerY);
      
      ctx.drawImage(canvas, 0, 0);
      ctx.restore();
      break;
    }
      
    case 12: { // Resolution and quality degradation (CRITICAL for low-end cameras)
      // Downscale then upscale to simulate low resolution and compression
      const scaleFactor = 0.3 + Math.random() * 0.4; // 0.3 - 0.7
      const w = Math.max(8, Math.floor(augmentedCanvas.width * scaleFactor));
      const h = Math.max(8, Math.floor(augmentedCanvas.height * scaleFactor));

      const tmp = document.createElement('canvas');
      tmp.width = w;
      tmp.height = h;
      const tctx = tmp.getContext('2d');
      if (!tctx) { ctx.drawImage(canvas, 0, 0); break; }

      // Draw downscaled
      tctx.imageSmoothingEnabled = true;
      tctx.drawImage(canvas, 0, 0, w, h);

      // Optionally add light noise
      if (Math.random() > 0.5) {
        const id = tctx.getImageData(0, 0, w, h);
        const d = id.data;
        const noiseAmp = 8 + Math.random() * 12; // 8-20
        for (let i = 0; i < d.length; i += 4) {
          const n = (Math.random() - 0.5) * noiseAmp;
          d[i] = Math.max(0, Math.min(255, d[i] + n));
          d[i+1] = Math.max(0, Math.min(255, d[i+1] + n));
          d[i+2] = Math.max(0, Math.min(255, d[i+2] + n));
        }
        tctx.putImageData(id, 0, 0);
      }

      // Upscale back with pixelation
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, w, h, 0, 0, augmentedCanvas.width, augmentedCanvas.height);

      // Simulate JPEG compression by drawing toDataURL with low quality then back
      if (Math.random() > 0.5) {
        const q = 0.4 + Math.random() * 0.3; // 0.4 - 0.7
        const jpeg = augmentedCanvas.toDataURL('image/jpeg', q);
        const img = new Image();
        img.onload = () => {
          ctx.clearRect(0, 0, augmentedCanvas.width, augmentedCanvas.height);
          ctx.drawImage(img, 0, 0);
        };
        img.src = jpeg;
      }
      break;
    }

    case 13: { // Variable focus/depth of field (IMPORTANT for mobile cameras)
      // Simulate parts of signature being in/out of focus
      
      // First draw the signature
      ctx.drawImage(canvas, 0, 0);
      
      // Create selective blur effect
      const focusRegion = Math.floor(Math.random() * 3); // 0: center, 1: top, 2: bottom
      const blurAmount = Math.random() * 2 + 1; // 1 to 3px blur
      
      // Create a mask for the focus region
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = augmentedCanvas.width;
      maskCanvas.height = augmentedCanvas.height;
      const maskCtx = maskCanvas.getContext('2d');
      
      if (maskCtx) {
        maskCtx.fillStyle = 'rgba(0, 0, 0, 1)';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        
        // Create clear focus region
        maskCtx.globalCompositeOperation = 'destination-out';
        maskCtx.fillStyle = 'rgba(255, 255, 255, 1)';
        
        switch (focusRegion) {
          case 0: // Center focus
            maskCtx.fillRect(augmentedCanvas.width * 0.25, augmentedCanvas.height * 0.25, 
                           augmentedCanvas.width * 0.5, augmentedCanvas.height * 0.5);
            break;
          case 1: // Top focus
            maskCtx.fillRect(0, 0, augmentedCanvas.width, augmentedCanvas.height * 0.6);
            break;
          case 2: // Bottom focus
            maskCtx.fillRect(0, augmentedCanvas.height * 0.4, augmentedCanvas.width, augmentedCanvas.height * 0.6);
            break;
        }
        
        // Apply blur to the masked regions
        ctx.save();
        ctx.filter = `blur(${blurAmount}px)`;
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.restore();
      }
      
      break;
    }
      
    case 14: { // Video compression artifacts (CRITICAL for mobile real-time detection)
      // Simulate H.264 block artifacts and quantization noise with WebGL memory optimization
      // Use smaller blocks and optimized processing to avoid memory overload
      const blockSize = 8; // Fixed 8px blocks for efficiency
      const compressionLevel = Math.random() * 0.2 + 0.1; // 10-30% compression (reduced range)
      
      // Draw original first
      ctx.drawImage(canvas, 0, 0);
      
      // Apply blocky compression artifacts with memory-efficient approach
      const imageData = ctx.getImageData(0, 0, augmentedCanvas.width, augmentedCanvas.height);
      const data = imageData.data;
      
      // Process in larger chunks for better performance
      const chunkSize = 32; // Process 32x32 chunks at a time
      
      for (let chunkY = 0; chunkY < augmentedCanvas.height; chunkY += chunkSize) {
        for (let chunkX = 0; chunkX < augmentedCanvas.width; chunkX += chunkSize) {
          // Process blocks within this chunk
          for (let blockY = chunkY; blockY < Math.min(chunkY + chunkSize, augmentedCanvas.height); blockY += blockSize) {
            for (let blockX = chunkX; blockX < Math.min(chunkX + chunkSize, augmentedCanvas.width); blockX += blockSize) {
              // Average block colors (quantization)
              let r = 0, g = 0, b = 0, count = 0;
              
              for (let by = 0; by < blockSize && blockY + by < augmentedCanvas.height; by++) {
                for (let bx = 0; bx < blockSize && blockX + bx < augmentedCanvas.width; bx++) {
                  const i = ((blockY + by) * augmentedCanvas.width + (blockX + bx)) * 4;
                  r += data[i];
                  g += data[i + 1];
                  b += data[i + 2];
                  count++;
                }
              }
              
              if (count > 0) {
                // Apply averaged color to entire block
                r = Math.floor(r / count);
                g = Math.floor(g / count);
                b = Math.floor(b / count);
                
                for (let by = 0; by < blockSize && blockY + by < augmentedCanvas.height; by++) {
                  for (let bx = 0; bx < blockSize && blockX + bx < augmentedCanvas.width; bx++) {
                    const i = ((blockY + by) * augmentedCanvas.width + (blockX + bx)) * 4;
                    // Reduced noise intensity for memory efficiency
                    const noise = (Math.random() - 0.5) * compressionLevel * 30;
                    data[i] = Math.max(0, Math.min(255, r + noise));
                    data[i + 1] = Math.max(0, Math.min(255, g + noise));
                    data[i + 2] = Math.max(0, Math.min(255, b + noise));
                  }
                }
              }
            }
          }
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
      break;
    }
  }
  
  return augmentedCanvas;
};




