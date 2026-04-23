/**
 * Local type stubs for `@jax-js/onnx`.
 *
 * The package isn't published to npm yet (jax-js v0.1.11), so we alias the
 * specifier to the esm.sh bundle in `vite.config.ts`. These stubs give us
 * IDE / type-check support while we wait for a proper npm release.
 */
declare module "@jax-js/onnx" {
  import type { Array as JArray } from "@jax-js/jax";

  export interface ONNXRunOptions {
    debugStats?: string[];
    verbose?: boolean;
    additionalOutputs?: string[];
  }

  export class ONNXModel {
    constructor(modelBytes: Uint8Array);
    run(
      inputs: Record<string, JArray>,
      options?: ONNXRunOptions,
    ): Record<string, JArray>;
    dispose(): void;
  }
}
