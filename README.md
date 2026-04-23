# umatcher-jax-js

[UMatcher](https://github.com/aemior/UMatcher) - a 4M-parameter template
matching model - ported to [jax-js](https://github.com/ekzhang/jax-js) so it
can be deployed **exclusively in the frontend**. No servers, no uploads:
templates, images, and frames stay on the user's device and inference runs on
WebGPU (with a Wasm fallback) via jax-js.

- **Model origin**: [aemior/UMatcher](https://github.com/aemior/UMatcher),
  UNet-based SiamFC-style template matcher.
- **Runtime**: [@jax-js/jax](https://www.npmjs.com/package/@jax-js/jax) from
  npm + [@jax-js/onnx](https://github.com/ekzhang/jax-js/tree/main/packages/onnx)
  loaded dynamically from esm.sh (until the package is published on npm). You
  can override the URL by setting `globalThis.__UMATCHER_ONNX_URL__` before
  importing UMatcher - handy for vendoring a specific commit or a local build.
- **Where does inference happen?** In your browser.

## What's inside

```
src/
  umatcher/        # Re-usable TypeScript library
    matcher.ts     # ONNX model wrapper (template & search branches)
    detector.ts    # Pyramid sliding-window UDetector port
    tracker.ts     # Single-object UTracker port
    image.ts       # center_crop, resize, CHW/Float32 tensor helpers
    nms.ts         # Greedy NMS (matches cv2.dnn.NMSBoxes behaviour)
    types.ts       # Shared types and defaults
    index.ts       # Public API
  demo/            # Vite static site showcasing the detector
    main.ts
    index.html
    style.css
scripts/
  export_umatcher_onnx.sh   # Clones upstream + runs export_onnx.py
public/models/     # Put template_branch.onnx / search_branch.onnx here
```

## Quickstart

```bash
pnpm install         # or npm install / yarn
pnpm run dev         # starts the Vite demo on http://localhost:5173
```

The demo ships with the same sample images and video used by the upstream
Python demos:

- `test_1.png` / `test_2.png` / `test_3.png` / `test_4.png` (detection)
- `template_3.png` (one-shot detection)
- `girl_dance.mp4` (single-object tracking)

They live under `public/samples/` and are exposed as one-click presets at
the top of each demo tab (Detection / Tracking). The preset ROIs match the
upstream defaults exactly (e.g. `[110, 233, 52, 99]` for `test_1.png`,
`[547, 188, 43, 57]` for `girl_dance.mp4`) so results are directly
comparable to the Python reference implementation.

Before the demo can do anything useful you need to export the ONNX weights:

```bash
# One-liner: clones aemior/UMatcher, runs upstream export_onnx.py,
# writes the two .onnx files into public/models/
./scripts/export_umatcher_onnx.sh /path/to/best.pth
```

If you'd rather do it by hand, follow the
[upstream instructions](https://github.com/aemior/UMatcher#train) and copy the
resulting `template_branch.onnx` and `search_branch.onnx` into
`public/models/`.

## Using the library

```ts
import {
  buildUMatcher,
  UDetector,
} from "umatcher-jax-js";

const matcher = buildUMatcher({
  templateBranchUrl: "/models/template_branch.onnx",
  searchBranchUrl: "/models/search_branch.onnx",
});
await matcher.load();

const detector = new UDetector(matcher);
await detector.setTemplate(refImageData, [x1, y1, x2, y2]);

const { boxes, scores } = await detector.detect(searchImageData, {
  threshold: 0.5,
  pyramid: [0.7, 1.0, 1.3],
});
```

For single-object tracking use `UTracker` instead:

```ts
import { UTracker } from "umatcher-jax-js";

const tracker = new UTracker(matcher);
await tracker.init(firstFrame, [cx, cy, w, h]);

for await (const frame of frames) {
  const { pos, score } = await tracker.track(frame);
  if (score > 0) drawBox(frame, pos);
}
```

## Why ONNX instead of rewriting the model?

UMatcher's upstream already exports clean ONNX branches, and jax-js ships
`@jax-js/onnx` which lowers ONNX graphs onto its WebGPU/Wasm compiler. This
keeps the port small (<1k LOC) and preserves bit-level agreement with the
reference Python implementation: the only code that changes is the pre/post
processing, which we re-implement in TypeScript.

## Backend selection

jax-js auto-picks the fastest available device. To force a backend, pass
`device: "webgpu"` (or `"wasm"` / `"webgl"`) to `buildUMatcher`.

## Browser support

Anywhere jax-js works - Chrome/Edge, Firefox, Safari, iOS 26+, Chrome for
Android, and Deno. WebGPU gives the best performance; the Wasm fallback
works everywhere.

## Licence

MIT. Model weights and original training code are copyright the UMatcher
authors and distributed under the
[upstream licence](https://github.com/aemior/UMatcher).
