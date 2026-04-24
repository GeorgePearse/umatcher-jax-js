/**
 * Image utilities - all pure JS/Canvas, no server required.
 *
 * These mirror the few image helpers we need for running inference
 * (center_crop, resize) plus a convenience for converting from
 * HTMLImageElement / HTMLVideoElement / Canvas to tensor-ready Float32.
 */

import type { Bbox, CxCyWh } from "./types.js";

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
 * The bbox is in upstream UMatcher's [cx, cy, w, h] form. The crop is a square
 * whose side length equals `sqrt(w * h) * scale`, centered on the bbox center.
 * Regions outside the source image are padded with replicated edge pixels, as
 * OpenCV's `BORDER_REPLICATE` does in the Python and C++ demos.
 */
export function centerCrop(img: ImageData, bbox: CxCyWh, scale: number): ImageData {
  const [rawCx, rawCy, rawW, rawH] = bbox;
  const cx = Math.trunc(rawCx);
  const cy = Math.trunc(rawCy);
  const bw = Math.max(1, Math.trunc(rawW));
  const bh = Math.max(1, Math.trunc(rawH));
  const size = Math.max(1, Math.trunc(Math.sqrt(bw * bh) * scale));
  const x1 = cx - Math.floor(size / 2);
  const y1 = cy - Math.floor(size / 2);

  const out = new ImageData(size, size);
  const src = img.data;
  const dst = out.data;
  const srcW = img.width;
  const srcH = img.height;

  for (let oy = 0; oy < size; oy++) {
    const sy = clampInt(y1 + oy, 0, srcH - 1);
    for (let ox = 0; ox < size; ox++) {
      const sx = clampInt(x1 + ox, 0, srcW - 1);
      const si = (sy * srcW + sx) * 4;
      const di = (oy * size + ox) * 4;
      dst[di] = src[si];
      dst[di + 1] = src[si + 1];
      dst[di + 2] = src[si + 2];
      dst[di + 3] = src[si + 3];
    }
  }

  return out;
}

/** Convenience wrapper for callers that have a corner-format [x1, y1, x2, y2]. */
export function centerCropBbox(img: ImageData, bbox: Bbox, scale: number): ImageData {
  const [x1, y1, x2, y2] = bbox;
  return centerCrop(img, [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], scale);
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

function clampInt(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
