/**
 * Built-in samples that mirror the upstream UMatcher reference demos.
 *
 * The ROIs and image paths come directly from the upstream defaults so the
 * output is numerically comparable to the reference implementation.
 */

import type { Bbox, CxCyWh } from "@umatcher";

export interface DetectionSample {
  id: string;
  label: string;
  description: string;
  /** Reference image used to set the template. */
  referenceUrl: string;
  /**
   * Default template bbox in [x1, y1, x2, y2] pixels of the reference image.
   * Null means "let the user draw it".
   */
  defaultBbox: Bbox | null;
  /** Use the full reference image as the template when no explicit bbox is set. */
  useFullReference?: boolean;
  /** Search image to run detection on. */
  searchUrl: string;
  /** Default detection threshold to use for this preset. */
  threshold?: number;
}

export interface TrackerSample {
  id: string;
  label: string;
  description: string;
  videoUrl: string;
  /** Initial tracking ROI in [cx, cy, w, h] pixels of the first frame. */
  initRoi: CxCyWh;
}

/** Convert the upstream [cx, cy, w, h] ROIs to the [x1, y1, x2, y2] we use. */
function cxcywhToXyxy(cx: number, cy: number, w: number, h: number): Bbox {
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

// Upstream default detection ROI [110, 233, 52, 99] (cxcywh) on test_1.png.
const TEST_1_ROI = cxcywhToXyxy(110, 233, 52, 99);

export const DETECTION_SAMPLES: DetectionSample[] = [
  {
    id: "test-1",
    label: "Test 1 (upstream default)",
    description:
      "Self-match on test_1.png using the upstream default ROI (cxcywh [110, 233, 52, 99]).",
    referenceUrl: "/samples/test_1.png",
    defaultBbox: TEST_1_ROI,
    searchUrl: "/samples/test_1.png",
    threshold: 0.3,
  },
  {
    id: "test-2",
    label: "Test 2",
    description:
      "Reference and search images are identical - draw a box around any object to find its duplicates.",
    referenceUrl: "/samples/test_2.png",
    defaultBbox: null,
    searchUrl: "/samples/test_2.png",
    threshold: 0.4,
  },
  {
    id: "test-3-one-shot",
    label: "Test 3 (one-shot)",
    description:
      "One-shot detection: template_3.png is the isolated template (use the whole image), test_3.png is the scene. Tests cross-image matching.",
    referenceUrl: "/samples/template_3.png",
    defaultBbox: null,
    useFullReference: true,
    searchUrl: "/samples/test_3.png",
    threshold: 0.35,
  },
  {
    id: "test-4",
    label: "Test 4",
    description:
      "Test 4 from the upstream repo. Draw a template and detect matches in the same scene.",
    referenceUrl: "/samples/test_4.png",
    defaultBbox: null,
    searchUrl: "/samples/test_4.png",
    threshold: 0.4,
  },
];

export const TRACKER_SAMPLES: TrackerSample[] = [
  {
    id: "girl-dance",
    label: "Girl dance (upstream default)",
    description:
      "Single-object tracking on girl_dance.mp4 with the upstream init ROI (cxcywh [547, 188, 43, 57]).",
    videoUrl: "/samples/girl_dance.mp4",
    initRoi: [547, 188, 43, 57],
  },
];
