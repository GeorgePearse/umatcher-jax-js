/**
 * Detection demo for UMatcher-in-jax-js. Pick a repo sample image, draw a box,
 * and render detection results back onto that same image.
 */

import {
  buildUMatcher,
  UDetector,
  drawBoxes,
  type Bbox,
  type CxCyWh,
} from "@umatcher";

import { IMAGE_SAMPLES } from "./samples.js";

type DetectionState = {
  image: HTMLImageElement | null;
  imageData: ImageData | null;
  bbox: Bbox | null; // in natural coordinates of image
  boxes: Bbox[];
  scores: number[];
};

const state: DetectionState = {
  image: null,
  imageData: null,
  bbox: null,
  boxes: [],
  scores: [],
};
let detectionRunId = 0;
const DETECTION_THRESHOLD = 0.3;
const DETECTION_PYRAMID = [1.0];
const DETECTION_OVERLAP = 0;
const DETECTION_MAX_SEARCH_SIDE = 512;

// Model URLs are relative to the dev server's public root.
const MODEL_BASE = "/models";
// Override the backend via ?device=webgl / wasm / webgpu / cpu query param
// (useful for testing on hardware that doesn't expose WebGPU).
const deviceParam = new URLSearchParams(location.search).get("device");
const device = (
  deviceParam === "webgpu" ||
  deviceParam === "webgl" ||
  deviceParam === "wasm" ||
  deviceParam === "cpu"
    ? deviceParam
    : undefined
) as "webgpu" | "webgl" | "wasm" | "cpu" | undefined;
const matcher = buildUMatcher({
  templateBranchUrl: `${MODEL_BASE}/template_branch.onnx`,
  searchBranchUrl: `${MODEL_BASE}/search_branch.onnx`,
  device,
});

const detector = new UDetector(matcher);

// ---------------------- Detection ----------------------
const backendLine = byId<HTMLParagraphElement>("backend-line");
const imageSelect = byId<HTMLSelectElement>("imageSelect");
const imageCanvas = byId<HTMLCanvasElement>("imageCanvas");
const statusEl = byId<HTMLParagraphElement>("status");

populateImageSelect();
imageSelect.addEventListener("change", () => {
  void loadSelectedImage();
});

wireBboxSelection();
void loadSelectedImage();

// Kick off model loading up front so the user doesn't wait on the first click.
const matcherReady = (async () => {
  try {
    setStatus("");
    backendLine.textContent = "Initialising jax-js...";
    await matcher.load((info: { label: string; done: number; total: number }) => {
      backendLine.textContent = `Loading ${info.label}... (${info.done}/${info.total})`;
    });
    backendLine.textContent = "jax-js ready - running on WebGPU (or Wasm fallback).";
  } catch (err) {
    backendLine.textContent = `Could not load models: ${(err as Error).message}. See README for export instructions.`;
    backendLine.style.color = "#fca5a5";
  }
})();

function populateImageSelect(): void {
  imageSelect.replaceChildren(
    ...IMAGE_SAMPLES.map((sample) => {
      const option = document.createElement("option");
      option.value = sample.url;
      option.textContent = sample.label;
      return option;
    }),
  );
}

async function loadSelectedImage(): Promise<void> {
  const sample = IMAGE_SAMPLES.find((item) => item.url === imageSelect.value);
  if (!sample) return;
  const runId = ++detectionRunId;
  state.image = null;
  state.imageData = null;
  state.bbox = null;
  state.boxes = [];
  state.scores = [];
  clearImageCanvas();
  setStatus(`Loading ${sample.label}...`);
  try {
    const image = await loadImage(sample.url);
    if (runId !== detectionRunId) return;
    state.image = image;
    state.imageData = imageFromSource(image);
    drawImageCanvas();
    setStatus("Image loaded - drag a box around the object to match.");
  } catch (err) {
    if (runId !== detectionRunId) return;
    setStatus(`Image load failed: ${(err as Error).message}`);
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

function clearImageCanvas(): void {
  const ctx = imageCanvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
}

function drawImageCanvas(): void {
  if (!state.image) return;
  fitCanvas(imageCanvas, state.image);
  const ctx = imageCanvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(state.image, 0, 0, imageCanvas.width, imageCanvas.height);
  if (state.boxes.length > 0) drawDetections(state.boxes, state.scores);
  if (state.bbox) strokeBbox(imageCanvas, state.bbox, "#facc15");
}

function fitCanvas(canvas: HTMLCanvasElement, img: HTMLImageElement): void {
  const maxW = 800;
  const ratio = Math.min(1, maxW / img.naturalWidth);
  canvas.width = Math.round(img.naturalWidth * ratio);
  canvas.height = Math.round(img.naturalHeight * ratio);
  canvas.dataset.ratio = ratio.toString();
}

function strokeBbox(
  canvas: HTMLCanvasElement,
  bbox: Bbox,
  color: string,
): void {
  const ratio = parseFloat(canvas.dataset.ratio || "1");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const [x1, y1, x2, y2] = bbox;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1 * ratio, y1 * ratio, (x2 - x1) * ratio, (y2 - y1) * ratio);
  ctx.restore();
}

function wireBboxSelection(): void {
  let dragStart: { x: number; y: number } | null = null;
  let pointerId: number | null = null;

  imageCanvas.addEventListener("pointerdown", (e) => {
    if (!state.image) return;
    pointerId = e.pointerId;
    imageCanvas.setPointerCapture(e.pointerId);
    const { x, y } = toImageCoords(imageCanvas, e);
    dragStart = { x, y };
    detectionRunId++;
    state.bbox = null;
    state.boxes = [];
    state.scores = [];
    drawImageCanvas();
  });

  imageCanvas.addEventListener("pointermove", (e) => {
    if (!dragStart || pointerId !== e.pointerId || !state.image) return;
    updateDraggedBbox(e, dragStart);
    drawImageCanvas();
  });

  const finish = (e: PointerEvent) => {
    if (!dragStart || pointerId !== e.pointerId) return;
    updateDraggedBbox(e, dragStart);
    dragStart = null;
    pointerId = null;
    if (imageCanvas.hasPointerCapture(e.pointerId)) {
      imageCanvas.releasePointerCapture(e.pointerId);
    }
    if (!state.bbox || !isUsableBbox(state.bbox)) {
      state.bbox = null;
      drawImageCanvas();
      setStatus("Draw a larger box around the object to match.");
      return;
    }
    drawImageCanvas();
    void handleDetect();
  };
  imageCanvas.addEventListener("pointerup", finish);
  imageCanvas.addEventListener("pointercancel", finish);

  function updateDraggedBbox(e: PointerEvent, start: { x: number; y: number }): void {
    if (!state.image) return;
    const { x, y } = toImageCoords(imageCanvas, e);
    state.bbox = [
      Math.min(start.x, x),
      Math.min(start.y, y),
      Math.max(start.x, x),
      Math.max(start.y, y),
    ];
  }
}

function toImageCoords(canvas: HTMLCanvasElement, e: MouseEvent): {
  x: number;
  y: number;
} {
  const rect = canvas.getBoundingClientRect();
  const ratio = parseFloat(canvas.dataset.ratio || "1");
  const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
  const naturalW = canvas.width / ratio;
  const naturalH = canvas.height / ratio;
  return {
    x: clamp(canvasX / ratio, 0, naturalW),
    y: clamp(canvasY / ratio, 0, naturalH),
  };
}

async function handleDetect(): Promise<void> {
  if (!state.image || !state.imageData || !state.bbox) return;
  const runId = ++detectionRunId;
  state.boxes = [];
  state.scores = [];
  drawImageCanvas();
  setStatus("Preparing template and detecting...");
  try {
    await matcherReady;
    await detector.setTemplate(state.imageData, xyxyToCxcywh(state.bbox));
    if (runId !== detectionRunId) return;
    const t0 = performance.now();
    const { boxes, scores } = await detector.detect(state.imageData, {
      threshold: DETECTION_THRESHOLD,
      pyramid: DETECTION_PYRAMID,
      overlap: DETECTION_OVERLAP,
      maxSearchSide: DETECTION_MAX_SEARCH_SIDE,
    });
    if (runId !== detectionRunId) return;
    const dt = performance.now() - t0;
    state.boxes = boxes;
    state.scores = scores;
    drawImageCanvas();
    setStatus(
      `Found ${boxes.length} match(es) in ${dt.toFixed(0)} ms using a ${DETECTION_MAX_SEARCH_SIDE}px fast search cap.`,
    );
  } catch (err) {
    setStatus(`Detect failed: ${(err as Error).message}`);
  }
}

function drawDetections(boxes: Bbox[], scores: number[]): void {
  const ratio = parseFloat(imageCanvas.dataset.ratio || "1");
  const scaled: Bbox[] = boxes.map(([x1, y1, x2, y2]) => [
    x1 * ratio,
    y1 * ratio,
    x2 * ratio,
    y2 * ratio,
  ]);
  drawBoxes(imageCanvas, scaled, scores);
}

function xyxyToCxcywh([x1, y1, x2, y2]: Bbox): CxCyWh {
  return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1];
}

function isUsableBbox([x1, y1, x2, y2]: Bbox): boolean {
  return x2 - x1 >= 4 && y2 - y1 >= 4;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function imageFromSource(img: HTMLImageElement | HTMLCanvasElement): ImageData {
  let w: number;
  let h: number;
  if (img instanceof HTMLImageElement) {
    w = img.naturalWidth;
    h = img.naturalHeight;
  } else {
    w = img.width;
    h = img.height;
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas unsupported");
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, w, h);
}

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}
