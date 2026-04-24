/**
 * UClassifier - small kNN classifier over UMatcher template embeddings.
 *
 * This mirrors the upstream classification example: build labeled template
 * embeddings, compare a target embedding by cosine similarity, and take the
 * majority label from the top-k neighbors.
 */

import { centerCrop, imageDataToTensor, resize } from "./image.js";
import { UMatcher } from "./matcher.js";
import type { CxCyWh } from "./types.js";

export interface ClassificationNeighbor {
  label: string;
  score: number;
}

export interface ClassificationResult {
  label: string | null;
  score: number;
  neighbors: ClassificationNeighbor[];
}

export class UClassifier {
  readonly matcher: UMatcher;
  private readonly embeddings = new Map<string, Float32Array[]>();

  constructor(matcher: UMatcher) {
    this.matcher = matcher;
  }

  clear(): void {
    this.embeddings.clear();
  }

  labels(): string[] {
    return Array.from(this.embeddings.keys());
  }

  addEmbedding(label: string, embedding: Float32Array): void {
    const expected = this.matcher.cfg.embeddingDim;
    if (embedding.length !== expected) {
      throw new Error(
        `Template embedding has wrong length: expected ${expected}, got ${embedding.length}`,
      );
    }
    const bucket = this.embeddings.get(label) ?? [];
    bucket.push(l2Normalize(embedding));
    this.embeddings.set(label, bucket);
  }

  async addTemplate(label: string, image: ImageData, bbox: CxCyWh): Promise<Float32Array> {
    const embedding = await this.embedTemplate(image, bbox);
    this.addEmbedding(label, embedding);
    return embedding;
  }

  async classifyTemplate(
    image: ImageData,
    bbox: CxCyWh,
    k = 3,
  ): Promise<ClassificationResult> {
    return this.classifyEmbedding(await this.embedTemplate(image, bbox), k);
  }

  classifyEmbedding(embedding: Float32Array, k = 3): ClassificationResult {
    const target = l2Normalize(embedding);
    const similarities: ClassificationNeighbor[] = [];

    for (const [label, embeddings] of this.embeddings) {
      for (const candidate of embeddings) {
        similarities.push({ label, score: cosineSimilarity(target, candidate) });
      }
    }

    similarities.sort((a, b) => b.score - a.score);
    const neighbors = similarities.slice(0, Math.max(1, k));
    if (neighbors.length === 0) {
      return { label: null, score: 0, neighbors: [] };
    }

    const counts = new Map<string, { count: number; best: number }>();
    for (const n of neighbors) {
      const entry = counts.get(n.label) ?? { count: 0, best: -Infinity };
      entry.count++;
      entry.best = Math.max(entry.best, n.score);
      counts.set(n.label, entry);
    }

    let bestLabel: string | null = null;
    let bestCount = -1;
    let bestScore = -Infinity;
    for (const [label, entry] of counts) {
      if (
        entry.count > bestCount ||
        (entry.count === bestCount && entry.best > bestScore)
      ) {
        bestLabel = label;
        bestCount = entry.count;
        bestScore = entry.best;
      }
    }

    return {
      label: bestLabel,
      score: bestScore,
      neighbors,
    };
  }

  private async embedTemplate(image: ImageData, bbox: CxCyWh): Promise<Float32Array> {
    const { templateSize, templateScale } = this.matcher.cfg;
    const cropped = centerCrop(image, bbox, templateScale);
    const resized = resize(cropped, templateSize, templateSize);
    return this.matcher.embedTemplate(imageDataToTensor(resized));
  }
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Embedding length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function l2Normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum) + 1e-12;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}
