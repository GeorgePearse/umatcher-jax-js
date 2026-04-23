/**
 * Built-in samples that mirror the upstream UMatcher Python demos
 * (`scripts/detection_example.py`, `scripts/tracking_example.py`).
 *
 * The ROIs and image paths come directly from the upstream defaults.
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

// Corresponds to scripts/detection_example.py default: ROI [110, 233, 52, 99] on test_1.png.
const TEST_1_ROI = cxcywhToXyxy(110, 233, 52, 99);

export const DETECTION_SAMPLES: DetectionSample[] = [
  {
    id: "test-1",
    label: "Test 1 (upstream default)",
    description:
      "Matches `scripts/detection_example.py --template_img test_1.png --search_img test_1.png`. Self-match with the preset ROI baked in.",
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
    // Use the full image as the template by default - user can tighten it if they want.
    defaultBbox: null,
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
      "Matches `scripts/tracking_example.py --input_path girl_dance.mp4 --init_roi 547 188 43 57`.",
    videoUrl: "/samples/girl_dance.mp4",
    initRoi: [547, 188, 43, 57],
  },
];
