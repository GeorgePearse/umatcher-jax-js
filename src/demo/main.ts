/**
 * Demo driver for UMatcher-in-jax-js. Entirely client-side.
 *
 * Two tabs: Detection (static reference + search images) and Tracking
 * (video playback with UTracker). Both are preloaded with the exact same
 * sample images/videos the upstream UMatcher reference implementation
 * uses, with default ROIs that match its defaults.
 */

import {
  buildUMatcher,
  UDetector,
  UTracker,
  drawBoxes,
  type Bbox,
  type CxCyWh,
} from "@umatcher";

import {
  DETECTION_SAMPLES,
  TRACKER_SAMPLES,
  type DetectionSample,
  type TrackerSample,
} from "./samples.js";

type DetectionState = {
  refImage: HTMLImageElement | null;
  searchImage: HTMLImageElement | null;
  bbox: Bbox | null; // in natural coordinates of refImage
};

const state: DetectionState = { refImage: null, searchImage: null, bbox: null };

// Model URLs are relative to the dev server's public root.
const MODEL_BASE = "/models";
const matcher = buildUMatcher({
  templateBranchUrl: `${MODEL_BASE}/template_branch.onnx`,
  searchBranchUrl: `${MODEL_BASE}/search_branch.onnx`,
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
const refDrop = byId<HTMLDivElement>("refDrop");
const refFile = byId<HTMLInputElement>("refFile");
const refCanvas = byId<HTMLCanvasElement>("refCanvas");
const templateCanvas = byId<HTMLCanvasElement>("templateCanvas");
const setTemplateBtn = byId<HTMLButtonElement>("setTemplateBtn");
const searchDrop = byId<HTMLDivElement>("searchDrop");
const searchFile = byId<HTMLInputElement>("searchFile");
const searchCanvas = byId<HTMLCanvasElement>("searchCanvas");
const detectBtn = byId<HTMLButtonElement>("detectBtn");
const statusEl = byId<HTMLParagraphElement>("status");
const thresholdInput = byId<HTMLInputElement>("threshold");
const thresholdValue = byId<HTMLSpanElement>("thresholdValue");
const pyramidSelect = byId<HTMLSelectElement>("pyramid");
const detectionPresetGrid = byId<HTMLDivElement>("detectionPresets");

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

wireDropzone(refDrop, refFile, (img) => {
  state.refImage = img;
  state.bbox = null;
  drawRef();
  clearTemplate();
});

wireDropzone(searchDrop, searchFile, (img) => {
  state.searchImage = img;
  drawSearch();
});

wireBboxSelection();
setTemplateBtn.addEventListener("click", handleSetTemplate);
detectBtn.addEventListener("click", handleDetect);

// Render preset cards for detection.
for (const sample of DETECTION_SAMPLES) {
  detectionPresetGrid.appendChild(buildDetectionPresetCard(sample));
}

function buildDetectionPresetCard(sample: DetectionSample): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "preset-card";
  btn.innerHTML = `
    <div class="thumb-row">
      <img src="${sample.referenceUrl}" alt="reference" loading="lazy" />
      ${
        sample.referenceUrl !== sample.searchUrl
          ? `<img src="${sample.searchUrl}" alt="search" loading="lazy" />`
          : ""
      }
    </div>
    <h4>${sample.label}</h4>
    <p>${sample.description}</p>
  `;
  btn.addEventListener("click", () => loadDetectionPreset(sample));
  return btn;
}

async function loadDetectionPreset(sample: DetectionSample): Promise<void> {
  setStatus(`Loading preset: ${sample.label}...`);
  try {
    const [refImg, searchImg] = await Promise.all([
      loadImage(sample.referenceUrl),
      loadImage(sample.searchUrl),
    ]);
    state.refImage = refImg;
    state.searchImage = searchImg;
    state.bbox = sample.defaultBbox;
    if (sample.threshold !== undefined) {
      thresholdInput.value = String(sample.threshold);
      thresholdValue.textContent = sample.threshold.toFixed(2);
    }
    drawRef();
    drawSearch();
    clearTemplate();
    if (state.bbox) {
      setTemplateBtn.disabled = false;
      await handleSetTemplate();
      await handleDetect();
    } else {
      setStatus("Preset loaded - draw a box on the reference image, then Set template.");
    }
  } catch (err) {
    setStatus(`Preset load failed: ${(err as Error).message}`);
  }
}

// Kick off model loading up front so the user doesn't wait on the first click.
(async () => {
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

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

function drawRef(): void {
  if (!state.refImage) return;
  fitCanvas(refCanvas, state.refImage);
  const ctx = refCanvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(state.refImage, 0, 0, refCanvas.width, refCanvas.height);
  if (state.bbox) drawBboxOnCanvas(refCanvas, state.bbox, state.refImage);
  setTemplateBtn.disabled = !state.bbox;
}

function drawSearch(): void {
  if (!state.searchImage) return;
  fitCanvas(searchCanvas, state.searchImage);
  const ctx = searchCanvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(state.searchImage, 0, 0, searchCanvas.width, searchCanvas.height);
  detectBtn.disabled = !state.searchImage || !detector.templateEmbedding;
}

function fitCanvas(canvas: HTMLCanvasElement, img: HTMLImageElement): void {
  const maxW = 800;
  const ratio = Math.min(1, maxW / img.naturalWidth);
  canvas.width = Math.round(img.naturalWidth * ratio);
  canvas.height = Math.round(img.naturalHeight * ratio);
  canvas.dataset.ratio = ratio.toString();
}

function drawBboxOnCanvas(
  canvas: HTMLCanvasElement,
  bbox: Bbox,
  img: HTMLImageElement,
): void {
  const ratio = parseFloat(canvas.dataset.ratio || "1");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const [x1, y1, x2, y2] = bbox;
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1 * ratio, y1 * ratio, (x2 - x1) * ratio, (y2 - y1) * ratio);
}

function wireBboxSelection(): void {
  let dragStart: { x: number; y: number } | null = null;
  refCanvas.addEventListener("mousedown", (e) => {
    if (!state.refImage) return;
    const { x, y } = toImageCoords(refCanvas, e);
    dragStart = { x, y };
  });
  refCanvas.addEventListener("mousemove", (e) => {
    if (!dragStart || !state.refImage) return;
    const { x, y } = toImageCoords(refCanvas, e);
    const bbox: Bbox = [
      Math.min(dragStart.x, x),
      Math.min(dragStart.y, y),
      Math.max(dragStart.x, x),
      Math.max(dragStart.y, y),
    ];
    state.bbox = bbox;
    drawBboxOnCanvas(refCanvas, bbox, state.refImage);
  });
  const finish = () => {
    dragStart = null;
    setTemplateBtn.disabled = !state.bbox;
  };
  refCanvas.addEventListener("mouseup", finish);
  refCanvas.addEventListener("mouseleave", finish);
}

function toImageCoords(canvas: HTMLCanvasElement, e: MouseEvent): {
  x: number;
  y: number;
} {
  const rect = canvas.getBoundingClientRect();
  const ratio = parseFloat(canvas.dataset.ratio || "1");
  return {
    x: (e.clientX - rect.left) / ratio,
    y: (e.clientY - rect.top) / ratio,
  };
}

async function handleSetTemplate(): Promise<void> {
  if (!state.refImage || !state.bbox) return;
  setStatus("Running template branch...");
  try {
    const imageData = imageFromSource(state.refImage);
    await detector.setTemplate(imageData, state.bbox);
    if (detector.templateImage) {
      templateCanvas.width = detector.templateImage.width;
      templateCanvas.height = detector.templateImage.height;
      templateCanvas.getContext("2d")?.putImageData(detector.templateImage, 0, 0);
    }
    detectBtn.disabled = !state.searchImage;
    setStatus("Template set.");
  } catch (err) {
    setStatus(`Template failed: ${(err as Error).message}`);
  }
}

async function handleDetect(): Promise<void> {
  if (!state.searchImage) return;
  if (!detector.templateEmbedding) {
    setStatus("Set a template first.");
    return;
  }
  setStatus("Detecting...");
  detectBtn.disabled = true;
  try {
    const imageData = imageFromSource(state.searchImage);
    const t0 = performance.now();
    const { boxes, scores } = await detector.detect(imageData, {
      threshold: parseFloat(thresholdInput.value),
      pyramid: pyramidSelect.value.split(",").map((v) => parseFloat(v)),
    });
    const dt = performance.now() - t0;
    drawDetections(boxes, scores);
    setStatus(`Found ${boxes.length} match(es) in ${dt.toFixed(0)} ms.`);
  } catch (err) {
    setStatus(`Detect failed: ${(err as Error).message}`);
  } finally {
    detectBtn.disabled = false;
  }
}

function drawDetections(boxes: Bbox[], scores: number[]): void {
  if (!state.searchImage) return;
  drawSearch();
  const ratio = parseFloat(searchCanvas.dataset.ratio || "1");
  const scaled: Bbox[] = boxes.map(([x1, y1, x2, y2]) => [
    x1 * ratio,
    y1 * ratio,
    x2 * ratio,
    y2 * ratio,
  ]);
  drawBoxes(searchCanvas, scaled, scores);
}

function clearTemplate(): void {
  const ctx = templateCanvas.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, templateCanvas.width, templateCanvas.height);
  detectBtn.disabled = true;
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
