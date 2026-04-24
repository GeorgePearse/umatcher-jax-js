/** Built-in image samples shipped in `public/samples`. */
export interface ImageSample {
  id: string;
  label: string;
  url: string;
}

export const IMAGE_SAMPLES: ImageSample[] = [
  {
    id: "test-1",
    label: "test_1.png",
    url: "/samples/test_1.png",
  },
  {
    id: "test-2",
    label: "test_2.png",
    url: "/samples/test_2.png",
  },
  {
    id: "test-3",
    label: "test_3.png",
    url: "/samples/test_3.png",
  },
  {
    id: "test-4",
    label: "test_4.png",
    url: "/samples/test_4.png",
  },
  {
    id: "template-3",
    label: "template_3.png",
    url: "/samples/template_3.png",
  },
];
