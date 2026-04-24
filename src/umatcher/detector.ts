/**
 * UDetector - pyramid sliding-window detection built on top of UMatcher.
 * TypeScript port of the UMatcher reference detector, preserving identical
 * semantics (center-cropped template, pyramid scales, NMS, thresholds).
 */

import type { Bbox, CxCyWh, PyramidScales } from "./types.js";
import { UMatcher } from "./matcher.js";
import { centerCrop, imageDataToTensor, resize } from "./image.js";
import { nms } from "./nms.js";

export interface DetectionResult {
  boxes: Bbox[];
  scores: number[];
}

export interface DetectOptions {
  threshold?: number;
  pyramid?: PyramidScales;
  overlap?: number;
  /** IoU threshold for NMS. Defaults to 0.5 (matches the reference code). */
  nmsIou?: number;
}

const DEFAULT_PYRAMID: PyramidScales = [0.7, 1.0, 1.3];

export class UDetector {
  readonly matcher: UMatcher;
  templateEmbedding: Float32Array | null = null;
  /** Image -> template preprocessing scale factor, set by `setTemplate()`. */
  scaleFactor = 1;
  /** The cropped template image, for display / debugging. */
  templateImage: ImageData | null = null;
  private templateCount = 0;

  constructor(matcher: UMatcher) {
    this.matcher = matcher;
  }

  get featSz(): number {
    return Math.floor(this.matcher.cfg.searchSize / this.matcher.cfg.stride);
  }

  /**
   * Set the template from a source image + bbox around the object.
   *
   * `bbox` is [cx, cy, w, h] in pixels of the source image. This mirrors
   * the reference `set_template`: center-crop a square region around the
   * bbox (scaled by templateScale), resize it to templateSize, run the
   * template branch, and store the L2-normalised embedding.
   */
  async setTemplate(image: ImageData, bbox: CxCyWh): Promise<void> {
    this.templateEmbedding = null;
    this.templateCount = 0;
    await this.addTemplate(image, bbox);
  }

  /**
   * Add another template view and average it with the existing embedding.
   * This implements the upstream few-shot recommendation: extract multiple
   * template embeddings, sum them, then normalise back to unit length.
   */
  async addTemplate(image: ImageData, bbox: CxCyWh): Promise<void> {
    const { templateSize, templateScale } = this.matcher.cfg;
    const cropped = centerCrop(image, bbox, templateScale);
    const cropScaleFactor = templateSize / cropped.width;
    const resized = resize(cropped, templateSize, templateSize);
    this.templateImage = resized;
    const tensor = imageDataToTensor(resized);
    const embedding = await this.matcher.embedTemplate(tensor);
    this.templateEmbedding = this.templateEmbedding
      ? normalizeSum(this.templateEmbedding, embedding)
      : embedding;
    this.scaleFactor =
      this.templateCount === 0
        ? cropScaleFactor
        : (this.scaleFactor * this.templateCount + cropScaleFactor) /
          (this.templateCount + 1);
    this.templateCount++;
  }

  /** Use a precomputed template embedding, useful for classification/few-shot UIs. */
  setTemplateEmbedding(embedding: Float32Array): void {
    const expected = this.matcher.cfg.embeddingDim;
    if (embedding.length !== expected) {
      throw new Error(
        `Template embedding has wrong length: expected ${expected}, got ${embedding.length}`,
      );
    }
    this.templateEmbedding = l2Normalize(embedding);
    this.templateCount = 1;
  }

  getTemplateEmbedding(): Float32Array | null {
    return this.templateEmbedding ? new Float32Array(this.templateEmbedding) : null;
  }

  /**
   * Run detection on a full-resolution image using the pyramid sliding-window
   * scheme from the reference implementation.
   */
  async detect(
    image: ImageData,
    opts: DetectOptions = {},
  ): Promise<DetectionResult> {
    if (!this.templateEmbedding) {
      throw new Error("Call setTemplate() before detect()");
    }
    const threshold = opts.threshold ?? 0.5;
    const pyramid = opts.pyramid ?? DEFAULT_PYRAMID;
    const overlap = opts.overlap ?? 0.5;
    const iouThreshold = opts.nmsIou ?? 0.5;
    const { searchSize } = this.matcher.cfg;

    const originalW = image.width;
    const originalH = image.height;
    const allBoxes: Bbox[] = [];
    const allScores: number[] = [];

    for (const baseScale of pyramid) {
      const scale = this.scaleFactor * baseScale;
      let scaledW = Math.round(originalW * scale);
      let scaledH = Math.round(originalH * scale);

      // Skip when image shrinks below a single window in both dims.
      if (scaledW < searchSize && scaledH < searchSize) continue;

      let scaledCanvas = imageDataToCanvas(resize(image, scaledW, scaledH));

      const padW = Math.max(0, searchSize - scaledW);
      const padH = Math.max(0, searchSize - scaledH);
      if (padW > 0 || padH > 0) {
        scaledCanvas = padCanvas(scaledCanvas, scaledW + padW, scaledH + padH);
        scaledW += padW;
        scaledH += padH;
      }

      const step = Math.max(1, Math.round(searchSize * (1 - overlap)));

      const xStarts = buildStarts(scaledW, searchSize, step);
      const yStarts = buildStarts(scaledH, searchSize, step);
      const cropper = createCropper(searchSize, searchSize);

      for (const x of xStarts) {
        for (const y of yStarts) {
          const window = cropper(scaledCanvas, x, y);
          const { boxes, scores } = await this.searchWindow(window);
          for (let i = 0; i < boxes.length; i++) {
            const [cx, cy, w, h] = boxes[i];
            // Translate window-local coords to the resized-image coords,
            // then back to original image coords.
            const gcx = (cx + x) / scale;
            const gcy = (cy + y) / scale;
            const gw = w / scale;
            const gh = h / scale;
            const x1 = clamp(gcx - gw / 2, 0, originalW);
            const y1 = clamp(gcy - gh / 2, 0, originalH);
            const x2 = clamp(gcx + gw / 2, 0, originalW);
            const y2 = clamp(gcy + gh / 2, 0, originalH);
            if (x2 <= x1 || y2 <= y1) continue;
            allBoxes.push([x1, y1, x2, y2]);
            allScores.push(scores[i]);
          }
        }
      }
    }

    const kept = nms(allBoxes, allScores, threshold, iouThreshold);
    return {
      boxes: kept.map((i) => allBoxes[i]),
      scores: kept.map((i) => allScores[i]),
    };
  }

  /** Run a single window search. Returns raw [cx, cy, w, h] in window pixels. */
  async searchWindow(window: ImageData): Promise<{
    boxes: [number, number, number, number][];
    scores: number[];
  }> {
    if (!this.templateEmbedding) {
      throw new Error("Call setTemplate() before searchWindow()");
    }
    const { searchSize } = this.matcher.cfg;
    const tensor = imageDataToTensor(window);
    const out = await this.matcher.runSearch(tensor, this.templateEmbedding);
    return decodeHeatmap(out, searchSize, this.featSz, this.matcher.cfg.stride, 0.1);
  }
}

function normalizeSum(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new Error(`Embedding length mismatch: ${a.length} vs ${b.length}`);
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i] + b[i];
  return l2Normalize(out);
}

function l2Normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum) + 1e-12;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function imageDataToCanvas(img: ImageData): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire 2D canvas context");
  ctx.putImageData(img, 0, 0);
  return canvas;
}

function padCanvas(source: HTMLCanvasElement, w: number, h: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire 2D canvas context");
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(source, 0, 0);
  return canvas;
}

function createCropper(w: number, h: number): (
  source: HTMLCanvasElement,
  x: number,
  y: number,
) => ImageData {
  const dst = document.createElement("canvas");
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext("2d", { willReadFrequently: true });
  if (!dctx) throw new Error("Could not acquire 2D canvas context");
  return (source, x, y) => {
    dctx.drawImage(source, x, y, w, h, 0, 0, w, h);
    return dctx.getImageData(0, 0, w, h);
  };
}

function buildStarts(total: number, size: number, step: number): number[] {
  const starts: number[] = [];
  let cur = 0;
  while (cur <= total - size) {
    starts.push(cur);
    cur += step;
  }
  if (starts.length > 0 && starts[starts.length - 1] + size < total) {
    starts.push(total - size);
  } else if (starts.length === 0 && total >= size) {
    starts.push(0);
  }
  return starts;
}

/**
 * Decode the UMatcher heatmaps. Returns unscaled [cx, cy, w, h] in window
 * pixels (multiplied by searchSize so w/h are absolute window pixels).
 */
function decodeHeatmap(
  out: { scoreMap: Float32Array; scaleMap: Float32Array; offsetMap: Float32Array },
  searchSize: number,
  featSz: number,
  stride: number,
  detectThreshold: number,
): {
  boxes: [number, number, number, number][];
  scores: number[];
} {
  const { scoreMap, scaleMap, offsetMap } = out;
  const plane = featSz * featSz;
  const boxes: [number, number, number, number][] = [];
  const scores: number[] = [];
  for (let idx = 0; idx < plane; idx++) {
    const s = scoreMap[idx];
    if (s <= detectThreshold) continue;
    // scale_map: [1, 2, H, W] -> (w, h)
    const w = scaleMap[idx];
    const h = scaleMap[idx + plane];
    // offset_map: [1, 2, H, W] -> (x, y)
    const ox = offsetMap[idx];
    const oy = offsetMap[idx + plane];
    const iy = Math.floor(idx / featSz);
    const ix = idx % featSz;
    const cx = (ix + ox) / stride;
    const cy = (iy + oy) / stride;
    boxes.push([cx * searchSize, cy * searchSize, w * searchSize, h * searchSize]);
    scores.push(s);
  }
  return { boxes, scores };
}
