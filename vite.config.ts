import { defineConfig } from "vite";
import { resolve } from "node:path";

// The demo lives in src/demo and shares the library code in src/umatcher.
export default defineConfig({
  root: "src/demo",
  publicDir: resolve(__dirname, "public"),
  resolve: {
    alias: {
      "@umatcher": resolve(__dirname, "src/umatcher"),
    },
  },
  server: {
    port: 5173,
    // Cross-origin isolation enables SharedArrayBuffer (required for multi-threaded Wasm in jax-js).
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      allow: [resolve(__dirname)],
    },
  },
  preview: {
    port: 4173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    outDir: resolve(__dirname, "dist-demo"),
    emptyOutDir: true,
    target: "es2022",
    rollupOptions: {
      input: {
        index: resolve(__dirname, "src/demo/index.html"),
      },
    },
  },
});
