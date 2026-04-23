# UMatcher ONNX weights

Drop `template_branch.onnx` and `search_branch.onnx` here before running the
demo. These are produced by the upstream `scripts/export_onnx.py` script from
[aemior/UMatcher](https://github.com/aemior/UMatcher). See
`../../scripts/export_umatcher_onnx.sh` for a one-liner that runs the export
for you, or the "Exporting the model" section of the top-level README.

Expected input / output names for compatibility with this port:

- `template_branch.onnx`
  - input: `template_img` (1, 3, 128, 128)
  - output: `template_embedding` (1, 128, 1, 1)
- `search_branch.onnx`
  - inputs: `search_img` (1, 3, 256, 256), `template_embedding` (1, 128, 1, 1)
  - outputs: `score_map` (1, 1, 16, 16), `scale_map` (1, 2, 16, 16), `offset_map` (1, 2, 16, 16)
