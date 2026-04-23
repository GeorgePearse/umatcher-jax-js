/**
 * Image utilities - all pure JS/Canvas, no server required.
 *
 * These mirror the few image helpers we need for running inference
 * (center_crop, resize) plus a convenience for converting from
 * HTMLImageElement / HTMLVideoElement / Canvas to tensor-ready Float32.
 */

import type { Bbox } from "./types.js";

/**
 * Convert any browser image source into an ImageData at its native resolution.
 */
export function imageDataFromImage(
  source: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap,
): ImageData {
  const w = "videoWidth" in source ? source.videoWidth : source.width;
  const h = "videoHeight" in source ? source.videoHeight : source.height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not acquire 2D canvas context");
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Convert an ImageData into a CHW Float32Array normalised to [0, 1].
 *
 * Layout matches the reference preprocessing: transpose HWC -> CHW and scale
 * by 1/255. The UMatcher model expects RGB input; since ImageData is RGBA
 * the alpha channel is dropped.
 */
export function imageDataToTensor(img: ImageData): Float32Array {
  const { data, width, height } = img;
  const out = new Float32Array(3 * width * height);
  const plane = width * height;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = data[i] / 255.0; // R
    out[j + plane] = data[i + 1] / 255.0; // G
    out[j + 2 * plane] = data[i + 2] / 255.0; // B
  }
  return out;
}

/**
 * Center-crop a region around a bbox, mirroring the reference `center_crop`.
 *
 * The bbox is in [x1, y1, x2, y2] form. The crop is a square whose side length
 * equals `max(w, h) * scale`, centered on the bbox center. Regions outside the
 * source image are padded with zeros (black).
 */
export function centerCrop(img: ImageData, bbox: Bbox, scale: number): ImageData {
  const [x1, y1, x2, y2] = bbox;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const side = Math.max(x2 - x1, y2 - y1) * scale;
  const half = side / 2;

  const size = Math.max(1, Math.round(side));
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire 2D canvas context");

  // Put the source image onto a temp canvas so we can draw a region out of it.
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = img.width;
  srcCanvas.height = img.height;
  const srcCtx = srcCanvas.getContext("2d");
  if (!srcCtx) throw new Error("Could not acquire 2D canvas context");
  srcCtx.putImageData(img, 0, 0);

  const sx = cx - half;
  const sy = cy - half;
  // drawImage gracefully clips when coords are out of bounds, leaving zero alpha.
  // We fill black first so the padding is RGB=(0,0,0) rather than transparent.
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(srcCanvas, sx, sy, side, side, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

/** Resize an ImageData using the browser's native bilinear filter. */
export function resize(img: ImageData, w: number, h: number): ImageData {
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
  dctx.imageSmoothingEnabled = true;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, w, h);
  return dctx.getImageData(0, 0, w, h);
}

/**
 * Draw bounding boxes onto a canvas. Convenience helper used by the demo.
 */
export function drawBoxes(
  canvas: HTMLCanvasElement,
  boxes: Bbox[],
  scores: number[],
  options: { color?: string; lineWidth?: number; showScores?: boolean } = {},
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.strokeStyle = options.color ?? "#22c55e";
  ctx.lineWidth = options.lineWidth ?? 2;
  ctx.font = "14px system-ui, sans-serif";
  ctx.fillStyle = options.color ?? "#22c55e";
  for (let i = 0; i < boxes.length; i++) {
    const [x1, y1, x2, y2] = boxes[i];
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (options.showScores !== false) {
      const label = scores[i].toFixed(2);
      ctx.fillText(label, x1 + 2, Math.max(12, y1 - 4));
    }
  }
  ctx.restore();
}
