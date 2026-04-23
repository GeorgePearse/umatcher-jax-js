# umatcher-jax-js

[UMatcher](https://github.com/aemior/UMatcher) - a 4M-parameter template
matching model - rewritten as a pure TypeScript library that runs entirely in
the browser on top of [jax-js](https://github.com/ekzhang/jax-js). No server,
no backend, no Python anywhere in the pipeline: templates, images, frames,
and model weights stay on the user's device, and inference runs on WebGPU
(with a Wasm fallback) via jax-js.

- **What**: detector + tracker + image preprocessing + NMS, all in TypeScript.
- **Where does inference happen?** In your browser.
- **Setup**: `npm install && npm run dev`. Model weights are committed in
  `public/models/` so the demo works immediately.

## What's inside

```
src/
  umatcher/        # Library (published as a TS package)
    matcher.ts     # ONNX model wrapper (template & search branches)
    detector.ts    # Pyramid sliding-window detector
    tracker.ts     # Single-object tracker
    image.ts       # center_crop / resize / tensor helpers
    nms.ts         # Greedy NMS
    types.ts       # Shared types and defaults
    onnx-loader.ts # Dynamic @jax-js/onnx loader (via esm.sh)
    index.ts       # Public API
  demo/            # Vite static site
    main.ts
    samples.ts     # Built-in sample image/video presets
    index.html
    style.css
public/
  models/          # Pre-built ONNX weights (template + search branches)
  samples/         # Sample media (test_*.png, template_3.png, girl_dance.mp4)
```

## Quickstart

```bash
npm install
npm run dev
# open http://localhost:5173
```

The demo auto-loads the pre-built ONNX weights from `public/models/` and
ships with the same sample images and video as the upstream UMatcher demo:

- `test_1.png` / `test_2.png` / `test_3.png` / `test_4.png` (detection)
- `template_3.png` (one-shot cross-image detection)
- `girl_dance.mp4` (single-object tracking)

Each tab (Detection / Tracking) has a preset bar at the top so you can
reproduce the upstream examples with a single click. The preset ROIs match
the upstream defaults exactly (e.g. `[110, 233, 52, 99]` cxcywh on
`test_1.png`, `[547, 188, 43, 57]` on `girl_dance.mp4`) so the numerical
output is directly comparable.

## Using the library

```ts
import {
  buildUMatcher,
  UDetector,
  UTracker,
} from "umatcher-jax-js";

const matcher = buildUMatcher({
  templateBranchUrl: "/models/template_branch.onnx",
  searchBranchUrl: "/models/search_branch.onnx",
});
await matcher.load();

// --- Detection ---
const detector = new UDetector(matcher);
await detector.setTemplate(refImageData, [x1, y1, x2, y2]);
const { boxes, scores } = await detector.detect(searchImageData, {
  threshold: 0.3,
  pyramid: [0.7, 1.0, 1.3],
});

// --- Single-object tracking ---
const tracker = new UTracker(matcher);
await tracker.init(firstFrame, [cx, cy, w, h]);
for await (const frame of frames) {
  const { pos, score } = await tracker.track(frame);
  if (score > 0) drawBox(frame, pos);
}
```

## Why ONNX?

jax-js ships `@jax-js/onnx` which lowers ONNX graphs onto its WebGPU/Wasm
compiler. Using it as the model format keeps this port compact (<1k LOC)
and preserves bit-level agreement with the reference implementation: we
only reimplement the pre/post processing in TypeScript.

## Runtime dependencies

- [`@jax-js/jax`](https://www.npmjs.com/package/@jax-js/jax) - pulled from
  npm as a standard dependency.
- `@jax-js/onnx` - not yet published to npm, so we load the pre-built ESM
  bundle from esm.sh at runtime. Override with
  `globalThis.__UMATCHER_ONNX_URL__` before importing UMatcher if you need
  to pin a specific commit or vendor the bundle locally.

## Backend selection

jax-js auto-picks the fastest available device. To force a backend, pass
`device: "webgpu"` (or `"wasm"` / `"webgl"`) to `buildUMatcher`.

## Browser support

Anywhere jax-js works - Chrome/Edge, Firefox, Safari, iOS 26+, Chrome for
Android, and Deno. WebGPU gives the best performance; the Wasm fallback
works everywhere.

## Licence

MIT. The bundled ONNX model weights are copyright the UMatcher authors and
distributed under the [upstream licence](https://github.com/aemior/UMatcher).
