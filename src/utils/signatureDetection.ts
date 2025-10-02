export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  isActive: boolean;
}

export class SignatureDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lastDetectionTime: number = 0;
  private detectionInterval: number = 300; // Detect every 300ms
  private lastValidBoxes: BoundingBox[] = []; // Cache last valid detection
  private boxHistory: BoundingBox[][] = []; // History for temporal smoothing
  private readonly HISTORY_SIZE = 5;
  private roi: { x: number; y: number; radius: number } | null = null; // Region of interest for user guidance
  
  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }
  
  /**
   * Set region of interest when user taps on screen
   * This guides detection to prioritize the tapped area
   */
  public setRegionOfInterest(x: number, y: number, radius: number = 150): void {
    this.roi = { x, y, radius };
    console.log(`📍 ROI set at (${x}, ${y}) with radius ${radius}`);
  }
  
  /**
   * Clear region of interest
   */
  public clearRegionOfInterest(): void {
    this.roi = null;
  }
  
  /**
   * Detect signatures using improved edge-based detection
   */
  async detectSignatures(videoElement: HTMLVideoElement): Promise<BoundingBox[]> {
    const now = Date.now();
    if (now - this.lastDetectionTime < this.detectionInterval) {
      return this.lastValidBoxes; // Return cached result to avoid flicker
    }
    this.lastDetectionTime = now;

    this.canvas.width = videoElement.videoWidth;
    this.canvas.height = videoElement.videoHeight;
    
    if (this.canvas.width === 0 || this.canvas.height === 0) {
      return [];
    }
    
    this.ctx.drawImage(videoElement, 0, 0);
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    
    // Apply edge detection to find ink strokes
    const edges = this.detectEdges(imageData);
    
    // Also detect ink-specific regions (blue/black ink)
    const inkMask = this.detectInkRegions(imageData);
    
    // Combine edge and ink detection for better accuracy
    const combinedMask = this.combineDetectionMasks(edges, inkMask);
    
    const boxes = this.findSignatureRegions(combinedMask, this.canvas.width, this.canvas.height);
    const mergedBoxes = this.mergeNearbyBoxes(boxes);
    
    // Filter out noise and apply signature-specific validation
    const validBoxes = mergedBoxes.filter(box => {
      const area = box.width * box.height;
      const minArea = 2000; // ~45x45 pixels minimum
      const maxArea = (this.canvas.width * this.canvas.height) * 0.8; // Max 80% of frame
      
      if (area <= minArea || area >= maxArea) return false;
      
      // Apply signature-specific heuristics
      return this.isLikelySignature(box, combinedMask);
    });
    
    // Apply ROI-based confidence boost if user tapped
    if (this.roi && validBoxes.length > 0) {
      validBoxes.forEach(box => {
        const centerX = box.x + box.width / 2;
        const centerY = box.y + box.height / 2;
        const distance = Math.sqrt(
          Math.pow(centerX - this.roi!.x, 2) + 
          Math.pow(centerY - this.roi!.y, 2)
        );
        
        if (distance < this.roi.radius) {
          box.confidence *= 1.5; // Boost confidence for boxes near tap
        }
      });
      
      // Sort by confidence
      validBoxes.sort((a, b) => b.confidence - a.confidence);
    }
    
    // Add margin and mark first as active
    let finalBoxes = validBoxes.map((box, index) => ({
      ...this.addMargin(box, 0.08),
      isActive: index === 0
    }));
    
    // Apply temporal smoothing to reduce jitter
    finalBoxes = this.smoothBoxes(finalBoxes);
    
    // Cache result
    if (finalBoxes.length > 0) {
      this.lastValidBoxes = finalBoxes;
    }
    
    return finalBoxes;
  }
  
  /**
   * Detect edges using Sobel operator (finds ink strokes)
   */
  private detectEdges(imageData: ImageData): Uint8ClampedArray {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const edges = new Uint8ClampedArray(width * height);
    
    // Convert to grayscale first
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      gray[i / 4] = avg;
    }
    
    // Sobel edge detection
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        
        // Sobel kernels
        const gx = 
          -gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
          -2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)] +
          -gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];
        
        const gy = 
          -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
          gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];
        
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        edges[idx] = magnitude > 50 ? 255 : 0; // Threshold for ink detection
      }
    }
    
    return edges;
  }
  
  /**
   * Find regions with high edge density (likely signatures)
   */
  private findSignatureRegions(edges: Uint8ClampedArray, width: number, height: number): BoundingBox[] {
    const boxes: BoundingBox[] = [];
    const cellSize = 20; // Divide frame into 20x20 pixel cells
    const cols = Math.floor(width / cellSize);
    const rows = Math.floor(height / cellSize);
    
    // Calculate edge density per cell
    const densityMap: number[][] = [];
    for (let r = 0; r < rows; r++) {
      densityMap[r] = [];
      for (let c = 0; c < cols; c++) {
        let edgeCount = 0;
        
        // Count edges in this cell
        for (let dy = 0; dy < cellSize; dy++) {
          for (let dx = 0; dx < cellSize; dx++) {
            const x = c * cellSize + dx;
            const y = r * cellSize + dy;
            if (x < width && y < height) {
              const idx = y * width + x;
              if (edges[idx] > 0) edgeCount++;
            }
          }
        }
        
        densityMap[r][c] = edgeCount;
      }
    }
    
    // Find connected regions with high edge density
    const visited = new Set<string>();
    const minDensity = 30; // Minimum edges per cell to consider
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const key = `${r},${c}`;
        if (visited.has(key) || densityMap[r][c] < minDensity) continue;
        
        const region = this.floodFillRegion(densityMap, r, c, rows, cols, minDensity, visited);
        
        if (region.cells.length >= 4) { // At least 4 cells = ~40x40 pixels
          boxes.push({
            x: region.minC * cellSize,
            y: region.minR * cellSize,
            width: (region.maxC - region.minC + 1) * cellSize,
            height: (region.maxR - region.minR + 1) * cellSize,
            confidence: 0.8,
            isActive: false
          });
        }
      }
    }
    
    return boxes;
  }
  
  /**
   * Flood fill to find connected high-density regions
   */
  private floodFillRegion(
    densityMap: number[][],
    startR: number,
    startC: number,
    rows: number,
    cols: number,
    minDensity: number,
    visited: Set<string>
  ): { cells: string[]; minR: number; maxR: number; minC: number; maxC: number } {
    const stack = [[startR, startC]];
    const cells: string[] = [];
    let minR = startR, maxR = startR, minC = startC, maxC = startC;
    
    while (stack.length > 0) {
      const [r, c] = stack.pop()!;
      const key = `${r},${c}`;
      
      if (r < 0 || r >= rows || c < 0 || c >= cols || visited.has(key)) continue;
      if (densityMap[r][c] < minDensity) continue;
      
      visited.add(key);
      cells.push(key);
      
      minR = Math.min(minR, r);
      maxR = Math.max(maxR, r);
      minC = Math.min(minC, c);
      maxC = Math.max(maxC, c);
      
      // Check 4 neighbors
      stack.push([r + 1, c]);
      stack.push([r - 1, c]);
      stack.push([r, c + 1]);
      stack.push([r, c - 1]);
    }
    
    return { cells, minR, maxR, minC, maxC };
  }
  
  /**
   * Merge overlapping boxes
   */
  private mergeNearbyBoxes(boxes: BoundingBox[]): BoundingBox[] {
    if (boxes.length <= 1) return boxes;
    
    const merged: BoundingBox[] = [];
    const used = new Set<number>();
    
    for (let i = 0; i < boxes.length; i++) {
      if (used.has(i)) continue;
      
      let current = boxes[i];
      used.add(i);
      
      for (let j = i + 1; j < boxes.length; j++) {
        if (used.has(j)) continue;
        
        if (this.boxesOverlap(current, boxes[j])) {
          current = this.mergeBoxes(current, boxes[j]);
          used.add(j);
        }
      }
      
      merged.push(current);
    }
    
    return merged;
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
   * Merge two boxes
   */
  private mergeBoxes(a: BoundingBox, b: BoundingBox): BoundingBox {
    const minX = Math.min(a.x, b.x);
    const minY = Math.min(a.y, b.y);
    const maxX = Math.max(a.x + a.width, b.x + b.width);
    const maxY = Math.max(a.y + a.height, b.y + b.height);
    
    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
      confidence: Math.max(a.confidence, b.confidence),
      isActive: a.isActive || b.isActive
    };
  }
  
  /**
   * Add margin to prevent edge cropping
   */
  private addMargin(box: BoundingBox, marginPercent: number): BoundingBox {
    const marginX = box.width * marginPercent;
    const marginY = box.height * marginPercent;
    
    return {
      ...box,
      x: Math.max(0, box.x - marginX),
      y: Math.max(0, box.y - marginY),
      width: box.width + (marginX * 2),
      height: box.height + (marginY * 2)
    };
  }
  
  /**
   * Detect ink regions specifically (blue/black ink detection)
   */
  private detectInkRegions(imageData: ImageData): Uint8ClampedArray {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const inkMask = new Uint8ClampedArray(width * height);
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Detect blue ink (blue > red and blue > green, moderately saturated)
      const isBlueInk = b > r + 20 && b > g + 10 && b > 80 && b < 200;
      
      // Detect black ink (low RGB values across all channels)
      const isBlackInk = r < 80 && g < 80 && b < 80;
      
      // Detect dark blue/navy ink (common in professional signatures)
      const isDarkBlueInk = b > r && b > g && r < 100 && g < 100 && b < 150;
      
      if (isBlueInk || isBlackInk || isDarkBlueInk) {
        inkMask[i / 4] = 255;
      }
    }
    
    return inkMask;
  }
  
  /**
   * Combine edge detection and ink detection for better accuracy
   */
  private combineDetectionMasks(edges: Uint8ClampedArray, inkMask: Uint8ClampedArray): Uint8ClampedArray {
    const combined = new Uint8ClampedArray(edges.length);
    
    for (let i = 0; i < edges.length; i++) {
      // A pixel is considered part of signature if:
      // 1. It has strong edges (likely ink stroke border)
      // 2. OR it's ink-colored (likely ink fill)
      // Combining both gives us better detection
      combined[i] = Math.max(edges[i], inkMask[i]);
    }
    
    return combined;
  }
  
  /**
   * Check if a detected box is likely a signature using heuristics
   */
  private isLikelySignature(box: BoundingBox, mask: Uint8ClampedArray): boolean {
    // Heuristic 1: Aspect ratio check
    // Signatures are typically wider than tall (ratio between 1.5 and 6)
    const aspectRatio = box.width / box.height;
    if (aspectRatio < 1.2 || aspectRatio > 8) {
      return false; // Too square or too elongated
    }
    
    // Heuristic 2: Edge density check
    // Signatures have moderate edge density (not too dense, not too sparse)
    const edgeDensity = this.calculateEdgeDensityInBox(box, mask);
    if (edgeDensity < 0.1 || edgeDensity > 0.7) {
      return false; // Too sparse or too dense
    }
    
    // Heuristic 3: Position check
    // Signatures rarely appear at the very top of the frame
    const relativeY = box.y / this.canvas.height;
    if (relativeY < 0.1) {
      return false; // Unlikely to be signature at top 10% of frame
    }
    
    // Heuristic 4: Size relative to frame
    // Signatures typically take up 5-40% of the frame width
    const relativeWidth = box.width / this.canvas.width;
    if (relativeWidth < 0.05 || relativeWidth > 0.6) {
      return false; // Too small or too large relative to frame
    }
    
    return true;
  }
  
  /**
   * Calculate edge density within a bounding box
   */
  private calculateEdgeDensityInBox(box: BoundingBox, mask: Uint8ClampedArray): number {
    let edgeCount = 0;
    let totalPixels = 0;
    
    for (let y = Math.floor(box.y); y < Math.min(box.y + box.height, this.canvas.height); y++) {
      for (let x = Math.floor(box.x); x < Math.min(box.x + box.width, this.canvas.width); x++) {
        const idx = y * this.canvas.width + x;
        if (mask[idx] > 0) edgeCount++;
        totalPixels++;
      }
    }
    
    return totalPixels > 0 ? edgeCount / totalPixels : 0;
  }
  
  /**
   * Apply temporal smoothing to reduce box jitter
   */
  private smoothBoxes(currentBoxes: BoundingBox[]): BoundingBox[] {
    if (currentBoxes.length === 0) return currentBoxes;
    
    this.boxHistory.push(currentBoxes);
    if (this.boxHistory.length > this.HISTORY_SIZE) {
      this.boxHistory.shift();
    }
    
    // Need at least 3 frames for meaningful smoothing
    if (this.boxHistory.length < 3) {
      return currentBoxes;
    }
    
    // Average box positions over time
    return currentBoxes.map((box, index) => {
      const historicalBoxes = this.boxHistory
        .map(h => h[index])
        .filter(Boolean);
      
      if (historicalBoxes.length === 0) return box;
      
      const avgX = historicalBoxes.reduce((sum, b) => sum + b.x, 0) / historicalBoxes.length;
      const avgY = historicalBoxes.reduce((sum, b) => sum + b.y, 0) / historicalBoxes.length;
      const avgW = historicalBoxes.reduce((sum, b) => sum + b.width, 0) / historicalBoxes.length;
      const avgH = historicalBoxes.reduce((sum, b) => sum + b.height, 0) / historicalBoxes.length;
      
      return {
        ...box,
        x: avgX,
        y: avgY,
        width: avgW,
        height: avgH
      };
    });
  }
}