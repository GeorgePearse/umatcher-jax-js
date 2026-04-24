/**
 * UTracker - single-object tracker, TypeScript port of the UMatcher
 * reference tracker.
 *
 * The reference implementation adds an 8-state Kalman filter
 * (x, y, w, h + velocities) on top of the search heatmap; this port mirrors
 * that KF-IoU fusion in TypeScript.
 */

import type { CxCyWh } from "./types.js";
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
  private readonly kf = new KalmanFilter8x4();
  private successCount = 0;
  private readonly alphaKf = 0.5;
  private readonly tauKf = 10;
  lastCandidates: { score: number; bbox: CxCyWh }[] = [];

  constructor(matcher: UMatcher) {
    this.matcher = matcher;
  }

  /** Initialise the tracker from the first frame. */
  async init(frame: ImageData, initialBbox: CxCyWh): Promise<void> {
    await this.updateTemplate(frame, initialBbox);
    this.lastPos = initialBbox;
    this.lastScore = 1.0;
    this.successCount = 0;
    this.kf.init(initialBbox);
  }

  /** Re-average the template embedding with a fresh crop. */
  async updateTemplate(frame: ImageData, bbox: CxCyWh): Promise<void> {
    const { templateSize, templateScale } = this.matcher.cfg;
    const cropped = centerCrop(frame, bbox, templateScale);
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
    const cropped = centerCrop(frame, this.lastPos, searchScale);
    const wI = cropped.width;
    const hI = cropped.height;
    const resized = resize(cropped, searchSize, searchSize);
    const tensor = imageDataToTensor(resized);
    const out = await this.matcher.runSearch(tensor, this.templateEmbedding);
    const featSz = Math.floor(searchSize / stride);
    const candidates = decodeCandidates(out, featSz, stride, 0.1);

    if (candidates.length === 0) {
      this.lastCandidates = [];
      this.successCount = 0;
      this.lastScore = 0;
      return { pos: this.lastPos, score: 0 };
    }

    this.lastCandidates = candidates.map(({ score, bbox }) => ({
      score,
      bbox: translateCandidate(bbox, this.lastPos as CxCyWh, wI, hI),
    }));

    const matched = this.matchPos(this.lastCandidates);
    this.lastPos = matched.pos;
    this.lastScore = matched.score;
    return matched;
  }

  private matchPos(detections: { score: number; bbox: CxCyWh }[]): TrackResult {
    if (!this.lastPos || detections.length === 0) {
      this.successCount = 0;
      return { pos: this.lastPos as CxCyWh, score: 0 };
    }

    const useKf = this.successCount >= this.tauKf;
    const prediction = useKf ? this.kf.predict() : null;
    let bestScore = -1;
    let best = detections[0];

    for (const det of detections) {
      const combined =
        prediction === null
          ? det.score
          : this.alphaKf * cxcywhIou(prediction, det.bbox) +
            (1 - this.alphaKf) * det.score;
      if (combined > bestScore) {
        bestScore = combined;
        best = det;
      }
    }

    if (bestScore > 0.2) {
      this.successCount++;
      this.kf.correct(best.bbox);
      return { pos: best.bbox, score: bestScore };
    }

    this.successCount = 0;
    return { pos: this.lastPos, score: 0 };
  }

  get position(): CxCyWh | null {
    return this.lastPos;
  }

  get score(): number {
    return this.lastScore;
  }
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

function translateCandidate(bbox: CxCyWh, lastPos: CxCyWh, wI: number, hI: number): CxCyWh {
  const [cx, cy, w, h] = bbox;
  return [
    lastPos[0] + (cx - 0.5) * wI,
    lastPos[1] + (cy - 0.5) * hI,
    w * wI,
    h * hI,
  ];
}

function cxcywhIou(a: CxCyWh, b: CxCyWh): number {
  const ax1 = a[0] - a[2] / 2;
  const ay1 = a[1] - a[3] / 2;
  const ax2 = a[0] + a[2] / 2;
  const ay2 = a[1] + a[3] / 2;
  const bx1 = b[0] - b[2] / 2;
  const by1 = b[1] - b[3] / 2;
  const bx2 = b[0] + b[2] / 2;
  const by2 = b[1] + b[3] / 2;
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

class KalmanFilter8x4 {
  private x = new Float32Array(8);
  private p = identity(8, 1);
  private readonly q = identity(8, 1e-2);
  private readonly r = identity(4, 1e-1);

  init(measurement: CxCyWh): void {
    this.x.fill(0);
    this.x[0] = measurement[0];
    this.x[1] = measurement[1];
    this.x[2] = measurement[2];
    this.x[3] = measurement[3];
    this.p = identity(8, 1);
  }

  predict(): CxCyWh {
    for (let i = 0; i < 4; i++) this.x[i] += this.x[i + 4];

    const fp = new Float32Array(64);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        fp[r * 8 + c] = this.p[r * 8 + c] + (r < 4 ? this.p[(r + 4) * 8 + c] : 0);
      }
    }

    const nextP = new Float32Array(64);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        nextP[r * 8 + c] = fp[r * 8 + c] + (c < 4 ? fp[r * 8 + c + 4] : 0) + this.q[r * 8 + c];
      }
    }
    this.p = nextP;
    return [this.x[0], this.x[1], this.x[2], this.x[3]];
  }

  correct(measurement: CxCyWh): void {
    const s = new Float32Array(16);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        s[r * 4 + c] = this.p[r * 8 + c] + this.r[r * 4 + c];
      }
    }
    const sInv = invert4(s);

    const k = new Float32Array(32);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let i = 0; i < 4; i++) sum += this.p[r * 8 + i] * sInv[i * 4 + c];
        k[r * 4 + c] = sum;
      }
    }

    const y = new Float32Array(4);
    for (let i = 0; i < 4; i++) y[i] = measurement[i] - this.x[i];
    for (let r = 0; r < 8; r++) {
      let delta = 0;
      for (let c = 0; c < 4; c++) delta += k[r * 4 + c] * y[c];
      this.x[r] += delta;
    }

    const nextP = new Float32Array(64);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        let sum = this.p[r * 8 + c];
        for (let i = 0; i < 4; i++) sum -= k[r * 4 + i] * this.p[i * 8 + c];
        nextP[r * 8 + c] = sum;
      }
    }
    this.p = nextP;
  }
}

function identity(size: number, value: number): Float32Array {
  const out = new Float32Array(size * size);
  for (let i = 0; i < size; i++) out[i * size + i] = value;
  return out;
}

function invert4(m: Float32Array): Float32Array {
  const a = new Float64Array(32);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) a[r * 8 + c] = m[r * 4 + c];
    a[r * 8 + 4 + r] = 1;
  }

  for (let col = 0; col < 4; col++) {
    let pivot = col;
    for (let r = col + 1; r < 4; r++) {
      if (Math.abs(a[r * 8 + col]) > Math.abs(a[pivot * 8 + col])) pivot = r;
    }
    if (Math.abs(a[pivot * 8 + col]) < 1e-12) {
      throw new Error("Kalman correction matrix is singular");
    }
    if (pivot !== col) {
      for (let c = 0; c < 8; c++) {
        const tmp = a[col * 8 + c];
        a[col * 8 + c] = a[pivot * 8 + c];
        a[pivot * 8 + c] = tmp;
      }
    }
    const div = a[col * 8 + col];
    for (let c = 0; c < 8; c++) a[col * 8 + c] /= div;
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const factor = a[r * 8 + col];
      for (let c = 0; c < 8; c++) a[r * 8 + c] -= factor * a[col * 8 + c];
    }
  }

  const inv = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) inv[r * 4 + c] = a[r * 8 + 4 + c];
  }
  return inv;
}
