/**
 * Dynamic loader for `@jax-js/onnx`.
 *
 * That package is not yet published to npm (as of jax-js v0.1.11) but is
 * available as a pre-built ESM bundle from esm.sh. This module provides a
 * minimal, typed wrapper so the rest of the library can import from a stable
 * location while the upstream package catches up.
 *
 * Override `ONNX_MODULE_URL` (or set it on `globalThis.__UMATCHER_ONNX_URL__`)
 * to point at a different build - for example a locally vendored bundle or a
 * specific commit SHA on esm.sh (e.g. `"https://esm.sh/gh/ekzhang/jax-js@main/packages/onnx"`).
 */

import type { Array as JArray } from "@jax-js/jax";

/** Minimal surface of `ONNXModel` that we rely on. */
export interface IOnnxModel {
  run(inputs: Record<string, JArray>): Record<string, JArray>;
  dispose(): void;
}

export interface IOnnxModelCtor {
  new (bytes: Uint8Array): IOnnxModel;
}

declare global {
  // Allow consumers to set a custom URL before loading.
  // eslint-disable-next-line no-var
  var __UMATCHER_ONNX_URL__: string | undefined;
}

const DEFAULT_ONNX_MODULE_URL =
  "https://esm.sh/gh/ekzhang/jax-js/packages/onnx?bundle";

let cached: Promise<IOnnxModelCtor> | null = null;

/**
 * Load (and memoise) the `ONNXModel` constructor. First tries a plain ESM
 * `import()` (so bundlers can resolve a vendored copy), and falls back to
 * esm.sh if that fails.
 */
export async function loadOnnxModel(): Promise<IOnnxModelCtor> {
  if (cached) return cached;
  cached = (async () => {
    const url =
      (typeof globalThis !== "undefined" && globalThis.__UMATCHER_ONNX_URL__) ||
      DEFAULT_ONNX_MODULE_URL;
    try {
      const mod = (await import(/* @vite-ignore */ url)) as {
        ONNXModel: IOnnxModelCtor;
      };
      if (!mod.ONNXModel) {
        throw new Error(
          `Module at ${url} does not export ONNXModel; got keys: ${Object.keys(mod).join(", ")}`,
        );
      }
      return mod.ONNXModel;
    } catch (err) {
      throw new Error(
        `UMatcher could not load @jax-js/onnx from "${url}". ` +
          `You can override the URL by setting globalThis.__UMATCHER_ONNX_URL__ ` +
          `before importing UMatcher, or by vendoring the package locally. ` +
          `Underlying error: ${(err as Error).message}`,
      );
    }
  })();
  return cached;
}
