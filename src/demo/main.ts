/**
 * Demo driver for UMatcher-in-jax-js. Entirely client-side.
 *
 * Two tabs: Detection (single image + drawn match box) and Tracking
 * (video playback with UTracker). Tracking is preloaded with the same sample
 * video and default ROI as the upstream UMatcher reference implementation.
 */

import {
  buildUMatcher,
  UDetector,
  UTracker,
  drawBoxes,
  type Bbox,
  type CxCyWh,
} from "@umatcher";

import { TRACKER_SAMPLES, type TrackerSample } from "./samples.js";

type DetectionState = {
  image: HTMLImageElement | null;
  bbox: Bbox | null; // in natural coordinates of image
  boxes: Bbox[];
  scores: number[];
};

const state: DetectionState = { image: null, bbox: null, boxes: [], scores: [] };
let detectionRunId = 0;
const DETECTION_THRESHOLD = 0.3;
const DETECTION_PYRAMID = [0.7, 1.0, 1.3];

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
const tracker = new UTracker(matcher);

// ---------------------- Tabs ----------------------
const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".tab"));
const views = Array.from(document.querySelectorAll<HTMLElement>(".view"));
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    tabs.forEach((t) => t.classList.toggle("active", t === tab));
    views.forEach((v) => v.classList.toggle("active", v.id === `${target}-view`));
  });
});

// ---------------------- Detection ----------------------
const backendLine = byId<HTMLParagraphElement>("backend-line");
const imageDrop = byId<HTMLDivElement>("imageDrop");
const imageFile = byId<HTMLInputElement>("imageFile");
const imageCanvas = byId<HTMLCanvasElement>("imageCanvas");
const statusEl = byId<HTMLParagraphElement>("status");

wireDropzone(imageDrop, imageFile, (img) => {
  detectionRunId++;
  state.image = img;
  state.bbox = null;
  state.boxes = [];
  state.scores = [];
  drawImageCanvas();
  setStatus("Image loaded - drag a box around the object to match.");
});

wireBboxSelection();

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

function wireDropzone(
  dz: HTMLDivElement,
  input: HTMLInputElement,
  onImage: (img: HTMLImageElement) => void,
): void {
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    const file = e.dataTransfer?.files[0];
    if (file) loadFile(file, onImage);
  });
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) loadFile(file, onImage);
  });
}

function loadFile(file: File, onImage: (img: HTMLImageElement) => void): void {
  const img = new Image();
  img.onload = () => onImage(img);
  img.onerror = () => setStatus(`Could not load ${file.name}`);
  img.src = URL.createObjectURL(file);
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
  if (!state.image || !state.bbox) return;
  const runId = ++detectionRunId;
  state.boxes = [];
  state.scores = [];
  drawImageCanvas();
  setStatus("Preparing template and detecting...");
  try {
    await matcherReady;
    const imageData = imageFromSource(state.image);
    await detector.setTemplate(imageData, xyxyToCxcywh(state.bbox));
    if (runId !== detectionRunId) return;
    const t0 = performance.now();
    const { boxes, scores } = await detector.detect(imageData, {
      threshold: DETECTION_THRESHOLD,
      pyramid: DETECTION_PYRAMID,
    });
    if (runId !== detectionRunId) return;
    const dt = performance.now() - t0;
    state.boxes = boxes;
    state.scores = scores;
    drawImageCanvas();
    setStatus(`Found ${boxes.length} match(es) in ${dt.toFixed(0)} ms.`);
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

function imageFromSource(
  img: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
): ImageData {
  let w: number;
  let h: number;
  if (img instanceof HTMLVideoElement) {
    w = img.videoWidth;
    h = img.videoHeight;
  } else if (img instanceof HTMLImageElement) {
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

// ---------------------- Tracking ----------------------
const videoDrop = byId<HTMLDivElement>("videoDrop");
const videoFile = byId<HTMLInputElement>("videoFile");
const trackerCanvas = byId<HTMLCanvasElement>("trackerCanvas");
const trackerVideo = byId<HTMLVideoElement>("trackerVideo");
const initTrackerBtn = byId<HTMLButtonElement>("initTrackerBtn");
const playTrackerBtn = byId<HTMLButtonElement>("playTrackerBtn");
const stopTrackerBtn = byId<HTMLButtonElement>("stopTrackerBtn");
const trackerFpsCap = byId<HTMLSelectElement>("trackerFpsCap");
const trackerStatus = byId<HTMLParagraphElement>("trackerStatus");
const trackerPresetGrid = byId<HTMLDivElement>("trackerPresets");

let trackerBbox: Bbox | null = null; // bbox drawn on the first frame in video coords
let trackerRunning = false;
let trackerInited = false;

wireDropzone(videoDrop, videoFile, () => {
  /* not used - we load via URL */
});
videoFile.addEventListener("change", () => {
  const file = videoFile.files?.[0];
  if (file) loadTrackerVideo(URL.createObjectURL(file));
});
videoDrop.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files[0];
  if (file) loadTrackerVideo(URL.createObjectURL(file));
});

initTrackerBtn.addEventListener("click", handleInitTracker);
playTrackerBtn.addEventListener("click", handlePlayTracker);
stopTrackerBtn.addEventListener("click", handleStopTracker);

// Render tracker presets.
for (const sample of TRACKER_SAMPLES) {
  trackerPresetGrid.appendChild(buildTrackerPresetCard(sample));
}

function buildTrackerPresetCard(sample: TrackerSample): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "preset-card";
  btn.innerHTML = `
    <div class="thumb-row">
      <video src="${sample.videoUrl}" muted preload="metadata"></video>
    </div>
    <h4>${sample.label}</h4>
    <p>${sample.description}</p>
  `;
  btn.addEventListener("click", async () => {
    await loadTrackerVideo(sample.videoUrl);
    // Convert cxcywh to xyxy for drawing; init() will re-convert.
    const [cx, cy, w, h] = sample.initRoi;
    trackerBbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    drawTrackerFrame();
    initTrackerBtn.disabled = false;
    setTrackerStatus("Preset loaded - hit Init tracker to run on the first frame.");
  });
  return btn;
}

async function loadTrackerVideo(url: string): Promise<void> {
  trackerVideo.src = url;
  trackerVideo.hidden = false;
  trackerBbox = null;
  trackerInited = false;
  playTrackerBtn.disabled = true;
  stopTrackerBtn.disabled = true;
  initTrackerBtn.disabled = true;
  await new Promise<void>((resolve, reject) => {
    trackerVideo.onloadedmetadata = () => resolve();
    trackerVideo.onerror = () => reject(new Error("video load failed"));
  });
  trackerVideo.currentTime = 0;
  await new Promise<void>((resolve) => {
    trackerVideo.onseeked = () => resolve();
  });
  fitVideoCanvas();
  drawTrackerFrame();
  setTrackerStatus("Video ready - draw a box on the first frame or pick a preset.");
}

function fitVideoCanvas(): void {
  const maxW = 900;
  const ratio = Math.min(1, maxW / trackerVideo.videoWidth);
  trackerCanvas.width = Math.round(trackerVideo.videoWidth * ratio);
  trackerCanvas.height = Math.round(trackerVideo.videoHeight * ratio);
  trackerCanvas.dataset.ratio = ratio.toString();
}

function drawTrackerFrame(): void {
  const ctx = trackerCanvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(trackerVideo, 0, 0, trackerCanvas.width, trackerCanvas.height);
  if (trackerBbox) {
    const ratio = parseFloat(trackerCanvas.dataset.ratio || "1");
    const [x1, y1, x2, y2] = trackerBbox;
    ctx.strokeStyle = trackerInited ? "#22c55e" : "#facc15";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * ratio, y1 * ratio, (x2 - x1) * ratio, (y2 - y1) * ratio);
    if (trackerInited && tracker.score > 0) {
      ctx.fillStyle = "#22c55e";
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText(`score ${tracker.score.toFixed(2)}`, x1 * ratio + 4, Math.max(12, y1 * ratio - 4));
    }
  }
}

// Let the user draw a bbox on the first frame of the video.
{
  let dragStart: { x: number; y: number } | null = null;
  trackerCanvas.addEventListener("mousedown", (e) => {
    if (!trackerVideo.src) return;
    const { x, y } = videoCoordsFromEvent(e);
    dragStart = { x, y };
  });
  trackerCanvas.addEventListener("mousemove", (e) => {
    if (!dragStart) return;
    const { x, y } = videoCoordsFromEvent(e);
    trackerBbox = [
      Math.min(dragStart.x, x),
      Math.min(dragStart.y, y),
      Math.max(dragStart.x, x),
      Math.max(dragStart.y, y),
    ];
    trackerInited = false;
    drawTrackerFrame();
  });
  const finish = () => {
    dragStart = null;
    initTrackerBtn.disabled = !trackerBbox;
  };
  trackerCanvas.addEventListener("mouseup", finish);
  trackerCanvas.addEventListener("mouseleave", finish);
}

function videoCoordsFromEvent(e: MouseEvent): { x: number; y: number } {
  const rect = trackerCanvas.getBoundingClientRect();
  const ratio = parseFloat(trackerCanvas.dataset.ratio || "1");
  return {
    x: (e.clientX - rect.left) / ratio,
    y: (e.clientY - rect.top) / ratio,
  };
}

async function handleInitTracker(): Promise<void> {
  if (!trackerBbox || !trackerVideo.src) return;
  setTrackerStatus("Initialising tracker on the first frame...");
  trackerVideo.currentTime = 0;
  await new Promise<void>((resolve) => {
    trackerVideo.onseeked = () => resolve();
  });
  try {
    const frame = imageFromSource(trackerVideo);
    const [x1, y1, x2, y2] = trackerBbox;
    const cxcywh: CxCyWh = [
      (x1 + x2) / 2,
      (y1 + y2) / 2,
      x2 - x1,
      y2 - y1,
    ];
    await tracker.init(frame, cxcywh);
    trackerInited = true;
    drawTrackerFrame();
    playTrackerBtn.disabled = false;
    setTrackerStatus("Tracker initialised. Hit Play to start tracking.");
  } catch (err) {
    setTrackerStatus(`Init failed: ${(err as Error).message}`);
  }
}

async function handlePlayTracker(): Promise<void> {
  if (!trackerInited) return;
  trackerRunning = true;
  playTrackerBtn.disabled = true;
  stopTrackerBtn.disabled = false;
  const capFps = parseFloat(trackerFpsCap.value) || 0;
  const frameMs = capFps > 0 ? 1000 / capFps : 0;
  let frameCount = 0;
  const t0 = performance.now();
  // Nudge the video forward a bit so currentTime advances each step.
  const step = 1 / (trackerVideo.duration > 0 ? 30 : 24);
  while (trackerRunning && trackerVideo.currentTime < trackerVideo.duration - step) {
    const tStart = performance.now();
    trackerVideo.currentTime = Math.min(
      trackerVideo.duration,
      trackerVideo.currentTime + step,
    );
    await new Promise<void>((resolve) => {
      const handler = () => {
        trackerVideo.removeEventListener("seeked", handler);
        resolve();
      };
      trackerVideo.addEventListener("seeked", handler);
    });
    const frame = imageFromSource(trackerVideo);
    const { pos, score } = await tracker.track(frame);
    const [cx, cy, w, h] = pos;
    trackerBbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    drawTrackerFrame();
    frameCount++;
    const elapsed = (performance.now() - t0) / 1000;
    setTrackerStatus(
      `Tracking ${frameCount} frames, ${(frameCount / elapsed).toFixed(1)} fps, score ${score.toFixed(2)}`,
    );
    if (frameMs > 0) {
      const wait = frameMs - (performance.now() - tStart);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
  }
  handleStopTracker();
}

function handleStopTracker(): void {
  trackerRunning = false;
  playTrackerBtn.disabled = !trackerInited;
  stopTrackerBtn.disabled = true;
}

function setTrackerStatus(msg: string): void {
  trackerStatus.textContent = msg;
}
