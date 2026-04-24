/**
 * Built-in tracking samples that mirror the upstream UMatcher reference demos.
 *
 * The ROIs and image paths come directly from the upstream defaults so the
 * output is numerically comparable to the reference implementation.
 */

import type { CxCyWh } from "@umatcher";

export interface TrackerSample {
  id: string;
  label: string;
  description: string;
  videoUrl: string;
  /** Initial tracking ROI in [cx, cy, w, h] pixels of the first frame. */
  initRoi: CxCyWh;
}

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
