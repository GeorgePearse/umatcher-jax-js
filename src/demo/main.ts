/**
 * Demo driver for UMatcher-in-jax-js. Entirely client-side.
 *
 * Wires up:
 *   - Reference image dropzone + interactive bbox selection
 *   - "Set template" button -> runs the template branch once
 *   - Search image dropzone + "Detect" button -> pyramid sliding-window search
 */

import {
  buildUMatcher,
  UDetector,
  drawBoxes,
  type Bbox,
} from "@umatcher";

type State = {
  refImage: HTMLImageElement | null;
  searchImage: HTMLImageElement | null;
  bbox: Bbox | null; // in natural coordinates of refImage
};

const state: State = { refImage: null, searchImage: null, bbox: null };

// Model URLs are relative to the dev server's public root.
// Drop your ONNX exports here (see README for instructions).
const MODEL_BASE = "/models";
const matcher = buildUMatcher({
  templateBranchUrl: `${MODEL_BASE}/template_branch.onnx`,
  searchBranchUrl: `${MODEL_BASE}/search_branch.onnx`,
});

const detector = new UDetector(matcher);

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
  // Fit to a reasonable max width so we can still see the whole image on a
  // laptop. Internal bbox math always uses the natural image coordinates.
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

function imageFromSource(img: HTMLImageElement): ImageData {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas unsupported");
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height);
}

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}
