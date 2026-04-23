/**
 * Non-maximum suppression, matching the behaviour of OpenCV's NMSBoxes as
 * used by UMatcher's reference detector.
 */

import type { Bbox } from "./types.js";

/** IoU of two [x1, y1, x2, y2] boxes. */
export function iou(a: Bbox, b: Bbox): number {
  const xx1 = Math.max(a[0], b[0]);
  const yy1 = Math.max(a[1], b[1]);
  const xx2 = Math.min(a[2], b[2]);
  const yy2 = Math.min(a[3], b[3]);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

/**
 * Greedy NMS. Returns the indices (into `boxes`) that survive, sorted by score
 * descending. Boxes with score < `scoreThreshold` are filtered before NMS.
 */
export function nms(
  boxes: Bbox[],
  scores: number[],
  scoreThreshold: number,
  iouThreshold = 0.5,
): number[] {
  if (boxes.length === 0) return [];
  const indices: number[] = [];
  for (let i = 0; i < boxes.length; i++) {
    if (scores[i] >= scoreThreshold) indices.push(i);
  }
  indices.sort((i, j) => scores[j] - scores[i]);

  const keep: number[] = [];
  const suppressed = new Uint8Array(boxes.length);
  for (const i of indices) {
    if (suppressed[i]) continue;
    keep.push(i);
    for (const j of indices) {
      if (j === i || suppressed[j]) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) suppressed[j] = 1;
    }
  }
  return keep;
}
