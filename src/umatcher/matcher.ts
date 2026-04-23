/**
 * UMatcher - loads the ONNX template and search branches and exposes the
 * low-level `embedTemplate()` / `runSearch()` primitives used by the detector
 * and tracker.
 *
 * The model runs entirely in the browser via jax-js (WebGPU or Wasm).
 */

import { numpy as np, defaultDevice, init as initJax } from "@jax-js/jax";
import type { Array as JArray } from "@jax-js/jax";
import { ONNXModel } from "@jax-js/onnx";

import type { UMatcherConfig } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";

export interface SearchOutputs {
  /** Score map, shape [1, 1, feat_sz, feat_sz]. Values in [0, 1]. */
  scoreMap: Float32Array;
  /** Scale map, shape [1, 2, feat_sz, feat_sz]. Channels are (w, h). */
  scaleMap: Float32Array;
  /** Offset map, shape [1, 2, feat_sz, feat_sz]. Channels are (x, y). */
  offsetMap: Float32Array;
  /** The feature-map side length (search_size / stride). */
  featSz: number;
}

let jaxInitialized: Promise<void> | null = null;

/**
 * Kick off jax-js device initialization and pick the best available backend.
 * Safe to call multiple times - returns the same promise.
 *
 * Device preference order when none is explicitly requested:
 *   1. `webgpu`  - fastest and supports all ops we need.
 *   2. `webgl`   - broad op coverage (including fp16 intermediates).
 *   3. `wasm`    - last resort; jax-js's wasm backend has limited ops so
 *                  some large models may fail with "not supported in wasm"
 *                  errors.
 */
function ensureJaxInit(preferred?: UMatcherConfig["device"]): Promise<void> {
  if (jaxInitialized) return jaxInitialized;
  jaxInitialized = (async () => {
    const devices = await initJax();
    // eslint-disable-next-line no-console
    console.log("[UMatcher] jax-js available devices:", devices);
    if (preferred && devices.includes(preferred)) {
      defaultDevice(preferred);
      return;
    }
    for (const candidate of ["webgpu", "webgl", "wasm"] as const) {
      if (devices.includes(candidate)) {
        // eslint-disable-next-line no-console
        console.log("[UMatcher] using backend:", candidate);
        defaultDevice(candidate);
        return;
      }
    }
  })();
  return jaxInitialized;
}

export class UMatcher {
  readonly cfg: UMatcherConfig;
  private templateModel: ONNXModel | null = null;
  private searchModel: ONNXModel | null = null;

  constructor(cfg: UMatcherConfig) {
    this.cfg = cfg;
  }

  /** Download and compile both ONNX branches. Safe to `await` multiple times. */
  async load(
    onProgress?: (info: { label: string; done: number; total: number }) => void,
  ): Promise<void> {
    await ensureJaxInit(this.cfg.device);
    if (this.templateModel && this.searchModel) return;

    onProgress?.({ label: "template branch", done: 0, total: 2 });
    const templateBytes = await fetchBytes(this.cfg.templateBranchUrl);
    this.templateModel = new ONNXModel(templateBytes);

    onProgress?.({ label: "search branch", done: 1, total: 2 });
    const searchBytes = await fetchBytes(this.cfg.searchBranchUrl);
    this.searchModel = new ONNXModel(searchBytes);

    onProgress?.({ label: "ready", done: 2, total: 2 });
  }

  /** Release model weights when you're done. */
  dispose(): void {
    this.templateModel?.dispose();
    this.searchModel?.dispose();
    this.templateModel = null;
    this.searchModel = null;
  }

  /**
   * Run the template branch. `tensor` is a CHW Float32Array of length
   * 3 * templateSize * templateSize normalised to [0, 1].
   *
   * Returns the L2-normalised embedding as a flat Float32Array of length
   * `embeddingDim`.
   */
  async embedTemplate(tensor: Float32Array): Promise<Float32Array> {
    const model = this.requireTemplate();
    const { templateSize, embeddingDim } = this.cfg;
    const expected = 3 * templateSize * templateSize;
    if (tensor.length !== expected) {
      throw new Error(
        `Template tensor has wrong length: expected ${expected}, got ${tensor.length}`,
      );
    }
    const input = np.array(tensor).reshape([1, 3, templateSize, templateSize]);
    const outputs = model.run({ template_img: input });
    const emb = firstOutput(outputs);
    const data = await emb.data();
    // L2 normalise to match the reference detector's behaviour.
    return l2Normalize(new Float32Array(data.buffer, data.byteOffset, embeddingDim));
  }

  /**
   * Run the search branch. `tensor` is a CHW Float32Array of length
   * 3 * searchSize * searchSize, and `templateEmbedding` is the flat vector
   * returned by `embedTemplate()`.
   */
  async runSearch(
    tensor: Float32Array,
    templateEmbedding: Float32Array,
  ): Promise<SearchOutputs> {
    const model = this.requireSearch();
    const { searchSize, embeddingDim, stride } = this.cfg;
    const expected = 3 * searchSize * searchSize;
    if (tensor.length !== expected) {
      throw new Error(
        `Search tensor has wrong length: expected ${expected}, got ${tensor.length}`,
      );
    }
    if (templateEmbedding.length !== embeddingDim) {
      throw new Error(
        `Template embedding has wrong length: expected ${embeddingDim}, got ${templateEmbedding.length}`,
      );
    }
    const searchInput = np
      .array(tensor)
      .reshape([1, 3, searchSize, searchSize]);
    // The ONNX search branch expects template_embedding as [1, embeddingDim, 1, 1].
    const embInput = np
      .array(templateEmbedding)
      .reshape([1, embeddingDim, 1, 1]);

    const outputs = model.run({
      search_img: searchInput,
      template_embedding: embInput,
    });

    const [score, scale, offset] = ["score_map", "scale_map", "offset_map"].map(
      (name) => findOutput(outputs, name) ?? firstOutput(outputs, name),
    );

    const [scoreData, scaleData, offsetData] = await Promise.all([
      score.data(),
      scale.data(),
      offset.data(),
    ]);

    return {
      scoreMap: toFloat32(scoreData),
      scaleMap: toFloat32(scaleData),
      offsetMap: toFloat32(offsetData),
      featSz: Math.floor(searchSize / stride),
    };
  }

  private requireTemplate(): ONNXModel {
    if (!this.templateModel) {
      throw new Error("UMatcher: call await .load() before running inference");
    }
    return this.templateModel;
  }

  private requireSearch(): ONNXModel {
    if (!this.searchModel) {
      throw new Error("UMatcher: call await .load() before running inference");
    }
    return this.searchModel;
  }
}

export function buildUMatcher(
  cfg: Partial<UMatcherConfig> &
    Pick<UMatcherConfig, "templateBranchUrl" | "searchBranchUrl">,
): UMatcher {
  return new UMatcher({ ...DEFAULT_CONFIG, ...cfg });
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
  }
  // `.bytes()` is newer; fall back to arrayBuffer for broader compat.
  if (typeof (resp as unknown as { bytes?: () => Promise<Uint8Array> }).bytes === "function") {
    return await (resp as unknown as { bytes: () => Promise<Uint8Array> }).bytes();
  }
  return new Uint8Array(await resp.arrayBuffer());
}

function firstOutput(
  outputs: Record<string, JArray>,
  preferredName?: string,
): JArray {
  if (preferredName && outputs[preferredName]) return outputs[preferredName];
  const keys = Object.keys(outputs);
  if (keys.length === 0) {
    throw new Error("ONNX model returned no outputs");
  }
  return outputs[keys[0]];
}

function findOutput(
  outputs: Record<string, JArray>,
  name: string,
): JArray | undefined {
  return outputs[name];
}

function toFloat32(data: ArrayBufferView): Float32Array {
  if (data instanceof Float32Array) return data;
  // jax-js returns TypedArrays; for fp16 we'd need conversion, but the exported
  // umatcher ONNX is fp32 by default. If the model is fp16 the user should
  // rebuild with --half=false, or we'd cast here.
  return new Float32Array(
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength / 4,
  );
}

function l2Normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum) + 1e-12;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}
