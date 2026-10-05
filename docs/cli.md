# CLI reference

```
wgsl-raytrace [OPTIONS] --config <CONFIG>
```

| Flag                  | Description                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `-c, --config <FILE>` | Scene TOML. Required.                                                                                                               |
| `-o, --output <FILE>` | Where to write the PNG. Default `render.png`.                                                                                       |
| `-s, --samples <N>`   | Override `camera.samples`.                                                                                                          |
| `-p, --preview`       | Draw the frame in the terminal as it converges (Kitty graphics protocol). Ctrl-C stops early and keeps what has been traced.        |
| `--denoise`           | Run the [à-trous filter](rendering.md#denoising) over the finished frame. `--output` gets the filtered frame.                       |
| `--debug <DIR>`       | Write `raw.png` (the unfiltered frame) and the `normal.png`, `albedo.png` and `depth.png` feature buffers to an existing directory. |
| `--variance <FILE>`   | Write a heatmap of per-pixel variance, normalised to the frame's peak.                                                              |
| `--outlier-k <K>`     | Override `outliers.k`. `0` is off.                                                                                                  |
| `--seed <N>`          | Draw an independent sample sequence, e.g. for a bias reference that shares no noise with the render. Default `0`.                   |

## Console output

| Line       | Meaning                                                                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene:`   | Triangle, material and BVH node counts, and tree depth.                                                                                                     |
| `render:`  | Wall time, covering the mesh load, BVH build, uploads and host stalls.                                                                                      |
| `denoise:` | Filter settings and wall time. Only shown with `--denoise`.                                                                                                 |
| `noise:`   | Mean per-pixel standard error, mean linear luminance, and peak variance. Use the standard error to compare runs of the same scene at the same sample count. |
| `gpu:`     | Per-dispatch GPU time from timestamp queries. Use this when judging a shader change. Missing on adapters that can't write timestamps.                       |
