/**
 * Shared types for UMatcher.
 */

/** Bounding box in corner format: [x1, y1, x2, y2] in pixel coordinates. */
export type Bbox = [number, number, number, number];

/** Center-width bounding box: [cx, cy, w, h]. */
export type CxCyWh = [number, number, number, number];

export type Rgb = [number, number, number];

export type PyramidScales = number[];

export interface UMatcherConfig {
  /** Input size expected by the template branch ONNX model (side length). */
  templateSize: number;
  /** Input size expected by the search branch ONNX model (side length). */
  searchSize: number;
  /**
   * Scale factor used when cropping the template from the reference image.
   * Matches `DATA.TEMPLATE.SCALE` from the UMatcher config (default 2).
   */
  templateScale: number;
  /**
   * Stride of the feature map (defaults to 16 for the mobileone backbone).
   */
  stride: number;
  /**
   * Dimensionality of the template embedding produced by the template branch.
   */
  embeddingDim: number;
  /**
   * URL (absolute or relative) to the fused template-branch ONNX model.
   */
  templateBranchUrl: string;
  /**
   * URL (absolute or relative) to the fused search-branch ONNX model.
   */
  searchBranchUrl: string;
  /**
   * Which jax-js device to run on. Defaults to the fastest available.
   */
  device?: "webgpu" | "wasm" | "webgl" | "cpu";
}

export const DEFAULT_CONFIG: Readonly<
  Omit<UMatcherConfig, "templateBranchUrl" | "searchBranchUrl">
> = Object.freeze({
  templateSize: 128,
  searchSize: 256,
  templateScale: 2,
  stride: 16,
  embeddingDim: 128,
  device: undefined,
});
