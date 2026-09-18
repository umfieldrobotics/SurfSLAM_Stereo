# Demo

Run a released model on a stereo pair. Everything below runs inside the container (`docker/run.sh`); see [docker.md](docker.md).

```bash
python -m demo --scene monohansett_hull      # command line
python -m demo.web                           # browser, http://localhost:7860
```

Both front ends are thin wrappers around `demo/core.py:predict_pair`, so they produce identical results.

## Weights

The demo reads the same checkpoint registry training uses (see [data.md](data.md) for details). To list available models, run:

```bash
python -m demo --list-checkpoints
```

Example output:
```
  defom_stereo/ours                       ok        .../stereo_weights/ours_vitl/best.pth
  defom_stereo/full_pretrain              ok        .../stereo_weights/ablation/fa_ia/best.pth
  defom_stereo/warp_finetune_full         ok        .../stereo_weights/ablation/fa_wl_ia/best.pth
  ...
```

`defom_stereo/ours` is the model from the paper and the default. The ablation stages are named after the configuration that produced them ([ablations/naming.md](../config/train/ablations/naming.md)).

A path to any `.pth` works too (as long as it is compatible with the specified model):

```bash
python -m demo --checkpoint runs/my_experiment/best.pth --model defom_stereo
```

## Calibration & Rectification

The bundled examples are raw frames

`demo/data/<scene>/{left,right}.png` are unrectified frames shipwreck surveys. The calibration in `demo/data/stereo_calib.yaml` is used to rectify the frames before running inference.

## Your own images

An already-rectified pair needs nothing special:

```bash
python -m demo --left left.png --right right.png
```

Disparity is produced. To compute depth, you must specify a focal length and stereo baseline:

```bash
python -m demo --left left.png --right right.png --focal-px 1692.5 --baseline-m 0.12
```

or hand over a [Kalibr](https://github.com/ethz-asl/kalibr) stereo yaml, which also rectifies the pair for you:

```bash
python -m demo --left raw_left.png --right raw_right.png --calib my_calib.yaml
```

The yaml needs `cam0` and `cam1` with `intrinsics` (`[fx, fy, cx, cy]`), `distortion_coeffs` (radtan `[k1, k2, p1, p2]`), `resolution`, and `cam1`'s `T_cn_cnm1`. `demo/data/stereo_calib.yaml` is a working example.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--scene NAME` | — | A bundled example; implies rectification |
| `--left`, `--right` | — | Your own pair; assumed rectified unless `--calib` |
| `--checkpoint SPEC` | `defom_stereo/ours` | Registry entry or path to a `.pth` |
| `--model NAME` | `defom_stereo` | Architecture, when `--checkpoint` is a bare path |
| `--calib PATH` | scene calib for `--scene` | Kalibr yaml: rectify + metric depth |
| `--no-rectify` | off | Use a scene's images as-is |
| `--focal-px`, `--baseline-m` | — | Metric depth without a calibration file |
| `--max-depth M` | — | Clip the depth colormap |
| `--scale S` | `1.0` | Run the network at a fraction of full resolution |
| `--iters N` | model config (32) | Refinement iterations |
| `--device` | `cuda` | Torch device |
| `--out DIR` | `demo/outputs/<name>` | Where results are written |
| `--list-scenes`, `--list-checkpoints` | — | Print and exit |

## Outputs

| File | What it is |
| --- | --- |
| `panel.png` | Left image, disparity and depth side by side |
| `disparity.png` | Disparity, magma colormap, 2nd–98th percentile range |
| `depth.png` | Depth, turbo colormap |
| `left_rectified.png` | Exactly what the network saw |
| `disparity.npy` | Float32 `[H, W]`, **pixels** |
| `depth.npy` | Float32 `[H, W]`, **metres**; `0` means invalid |
| `arrays.npz` | Both arrays plus `focal_px` and `baseline_m` |

## The web UI

```bash
python -m demo.web --preload         # --preload avoids a slow first click
```

`--share` publishes a temporary public `gradio.live` URL, which needs outbound internet from inside the container.

