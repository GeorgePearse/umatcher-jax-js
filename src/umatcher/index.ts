/**
 * UMatcher in jax-js - in-browser template matching.
 *
 * Public entry point. Everything here runs entirely client-side using WebGPU
 * or Wasm through jax-js; there is no server-side component.
 */

export { UMatcher, buildUMatcher } from "./matcher.js";
export { UDetector, type DetectionResult } from "./detector.js";
export { UTracker, type TrackResult } from "./tracker.js";
export {
  imageDataFromImage,
  imageDataToTensor,
  centerCrop,
  resize,
  drawBoxes,
} from "./image.js";
export { nms, iou } from "./nms.js";
export type { UMatcherConfig, Bbox, PyramidScales, Rgb } from "./types.js";
