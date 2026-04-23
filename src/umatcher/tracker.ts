/**
 * UTracker - single-object tracker, TypeScript port of the UMatcher
 * reference tracker.
 *
 * The reference implementation adds an 8-state Kalman filter
 * (x, y, w, h + velocities) on top of the search heatmap. We default to
 * pure-score tracking which works well for most in-browser demos and
 * matches UMatcher's published behaviour on the provided example video.
 */

import type { Bbox, CxCyWh } from "./types.js";
import { UMatcher } from "./matcher.js";
import { centerCrop, imageDataToTensor, resize } from "./image.js";

export interface TrackResult {
  /** Tracked bbox in center-width form [cx, cy, w, h], in original image pixels. */
  pos: CxCyWh;
  /** Confidence score in [0, 1]. 0 indicates a miss. */
  score: number;
}

export class UTracker {
  readonly matcher: UMatcher;
  private templateEmbedding: Float32Array | null = null;
  private lastPos: CxCyWh | null = null;
  private lastScore = 0;

  constructor(matcher: UMatcher) {
    this.matcher = matcher;
  }

  /** Initialise the tracker from the first frame. */
  async init(frame: ImageData, initialBbox: CxCyWh): Promise<void> {
    await this.updateTemplate(frame, initialBbox);
    this.lastPos = initialBbox;
    this.lastScore = 1.0;
  }

  /** Re-average the template embedding with a fresh crop. */
  async updateTemplate(frame: ImageData, bbox: CxCyWh): Promise<void> {
    const { templateSize, templateScale } = this.matcher.cfg;
    const cxcywh = bbox;
    const corner = cxcywhToCorner(cxcywh);
    const cropped = centerCrop(frame, corner, templateScale);
    const resized = resize(cropped, templateSize, templateSize);
    const tensor = imageDataToTensor(resized);
    const newEmb = await this.matcher.embedTemplate(tensor);
    if (!this.templateEmbedding) {
      this.templateEmbedding = newEmb;
    } else {
      // Running average, re-normalised (mirrors the reference logic).
      const combined = new Float32Array(newEmb.length);
      for (let i = 0; i < combined.length; i++) {
        combined[i] = this.templateEmbedding[i] + newEmb[i];
      }
      let sumSq = 0;
      for (let i = 0; i < combined.length; i++) sumSq += combined[i] * combined[i];
      const norm = Math.sqrt(sumSq);
      if (norm > 0) {
        for (let i = 0; i < combined.length; i++) combined[i] /= norm;
      }
      this.templateEmbedding = combined;
    }
  }

  /** Run one tracking step. Returns the updated position and score. */
  async track(frame: ImageData): Promise<TrackResult> {
    if (!this.templateEmbedding || !this.lastPos) {
      throw new Error("Call init() before track()");
    }
    const { searchSize, stride } = this.matcher.cfg;
    const searchScale = 4; // matches DATA.SEARCH.SCALE in the reference config
    const corner = cxcywhToCorner(this.lastPos);
    const cropped = centerCrop(frame, corner, searchScale);
    const wI = cropped.width;
    const hI = cropped.height;
    const resized = resize(cropped, searchSize, searchSize);
    const tensor = imageDataToTensor(resized);
    const out = await this.matcher.runSearch(tensor, this.templateEmbedding);
    const featSz = Math.floor(searchSize / stride);
    const candidates = decodeCandidates(out, featSz, stride, 0.1);

    if (candidates.length === 0) {
      this.lastScore = 0;
      return { pos: this.lastPos, score: 0 };
    }

    // Pick the highest-scoring candidate, transform window-local (normalised)
    // back to original-image pixels, and update lastPos.
    let best = candidates[0];
    for (let i = 1; i < candidates.length; i++) {
      if (candidates[i].score > best.score) best = candidates[i];
    }

    const [cx, cy, w, h] = best.bbox;
    const offX = (cx - 0.5) * wI;
    const offY = (cy - 0.5) * hI;
    const nw = w * wI;
    const nh = h * hI;
    const newPos: CxCyWh = [this.lastPos[0] + offX, this.lastPos[1] + offY, nw, nh];

    if (best.score > 0.2) {
      this.lastPos = newPos;
      this.lastScore = best.score;
      return { pos: newPos, score: best.score };
    }
    this.lastScore = 0;
    return { pos: this.lastPos, score: 0 };
  }

  get position(): CxCyWh | null {
    return this.lastPos;
  }

  get score(): number {
    return this.lastScore;
  }
}

function cxcywhToCorner(b: CxCyWh): Bbox {
  const [cx, cy, w, h] = b;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

/** Decode heatmap into normalised [cx, cy, w, h] candidates (in [0, 1]). */
function decodeCandidates(
  out: { scoreMap: Float32Array; scaleMap: Float32Array; offsetMap: Float32Array },
  featSz: number,
  stride: number,
  thr: number,
): { score: number; bbox: CxCyWh }[] {
  const { scoreMap, scaleMap, offsetMap } = out;
  const plane = featSz * featSz;
  const res: { score: number; bbox: CxCyWh }[] = [];
  for (let idx = 0; idx < plane; idx++) {
    const s = scoreMap[idx];
    if (s <= thr) continue;
    const w = scaleMap[idx];
    const h = scaleMap[idx + plane];
    const ox = offsetMap[idx];
    const oy = offsetMap[idx + plane];
    const iy = Math.floor(idx / featSz);
    const ix = idx % featSz;
    const cx = (ix + ox) / stride;
    const cy = (iy + oy) / stride;
    res.push({ score: s, bbox: [cx, cy, w, h] });
  }
  return res;
}
