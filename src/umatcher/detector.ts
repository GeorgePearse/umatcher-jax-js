/**
 * UDetector - pyramid sliding-window detection built on top of UMatcher.
 * Port of `lib/detector/udetector.py` with identical semantics.
 */

import type { Bbox, PyramidScales } from "./types.js";
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
  /** IoU threshold for NMS. Defaults to 0.5 (matches the Python code). */
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

  constructor(matcher: UMatcher) {
    this.matcher = matcher;
  }

  get featSz(): number {
    return Math.floor(this.matcher.cfg.searchSize / this.matcher.cfg.stride);
  }

  /**
   * Set the template from a source image + bbox around the object.
   *
   * `bbox` is [x1, y1, x2, y2] in pixels of the source image. This mirrors
   * the Python `set_template()`: it center-crops a square region around the
   * bbox (scaled by templateScale), resizes it to templateSize, runs the
   * template branch, and stores the L2-normalised embedding.
   */
  async setTemplate(image: ImageData, bbox: Bbox): Promise<void> {
    const { templateSize, templateScale } = this.matcher.cfg;
    const cropped = centerCrop(image, bbox, templateScale);
    this.scaleFactor = templateSize / cropped.width;
    const resized = resize(cropped, templateSize, templateSize);
    this.templateImage = resized;
    const tensor = imageDataToTensor(resized);
    this.templateEmbedding = await this.matcher.embedTemplate(tensor);
  }

  /**
   * Run detection on a full-resolution image using the pyramid sliding-window
   * scheme from the Python reference.
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

      // Skip when image shrinks below a single window in both dims, matching Python.
      if (scaledW < searchSize && scaledH < searchSize) continue;

      let scaled = resize(image, scaledW, scaledH);

      const padW = Math.max(0, searchSize - scaledW);
      const padH = Math.max(0, searchSize - scaledH);
      if (padW > 0 || padH > 0) {
        scaled = padImage(scaled, scaledW + padW, scaledH + padH);
        scaledW += padW;
        scaledH += padH;
      }

      const step = Math.max(1, Math.round(searchSize * (1 - overlap)));

      const xStarts = buildStarts(scaledW, searchSize, step);
      const yStarts = buildStarts(scaledH, searchSize, step);

      for (const x of xStarts) {
        for (const y of yStarts) {
          const window = cropRegion(scaled, x, y, searchSize, searchSize);
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

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function padImage(img: ImageData, w: number, h: number): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire 2D canvas context");
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, w, h);
  const src = document.createElement("canvas");
  src.width = img.width;
  src.height = img.height;
  const sctx = src.getContext("2d");
  if (!sctx) throw new Error("Could not acquire 2D canvas context");
  sctx.putImageData(img, 0, 0);
  ctx.drawImage(src, 0, 0);
  return ctx.getImageData(0, 0, w, h);
}

function cropRegion(
  img: ImageData,
  x: number,
  y: number,
  w: number,
  h: number,
): ImageData {
  const src = document.createElement("canvas");
  src.width = img.width;
  src.height = img.height;
  const sctx = src.getContext("2d");
  if (!sctx) throw new Error("Could not acquire 2D canvas context");
  sctx.putImageData(img, 0, 0);
  const dst = document.createElement("canvas");
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext("2d");
  if (!dctx) throw new Error("Could not acquire 2D canvas context");
  dctx.drawImage(src, x, y, w, h, 0, 0, w, h);
  return dctx.getImageData(0, 0, w, h);
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
