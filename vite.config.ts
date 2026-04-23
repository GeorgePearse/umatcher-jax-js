import { defineConfig, type Plugin } from "vite";
import { resolve } from "node:path";

/**
 * Resolve `@jax-js/onnx` to our vendored bundle so it shares a single
 * `@jax-js/jax` instance (from node_modules) with the rest of the app.
 * The vendored file is a minor rewrite of the upstream esm.sh bundle with
 * its esm.sh-relative imports swapped for bare npm specifiers.
 */
function jaxOnnxVendor(): Plugin {
  const vendored = resolve(__dirname, "src/umatcher/vendor/jax-onnx.mjs");
  return {
    name: "jax-onnx-vendor",
    enforce: "pre",
    resolveId(id) {
      if (id === "@jax-js/onnx") return vendored;
      return null;
    },
  };
}

export default defineConfig({
  root: "src/demo",
  publicDir: resolve(__dirname, "public"),
  plugins: [jaxOnnxVendor()],
  resolve: {
    alias: {
      "@umatcher": resolve(__dirname, "src/umatcher"),
    },
  },
  server: {
    port: 5173,
    fs: {
      allow: [resolve(__dirname)],
    },
  },
  preview: {
    port: 4173,
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
