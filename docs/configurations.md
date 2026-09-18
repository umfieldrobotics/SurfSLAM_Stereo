# Configs

Everything lives under `config/`, split by what it drives:

| Path | Drives | Axes |
| --- | --- | --- |
| `config/train/` | `train.py`, via `train_scripts/launch.sh` | model × ablation |
| `config/eval/` | `evaluate.py`, via `eval_scripts/inference.sh` | model × suite |

Both trees are parsed by the same code with the same rules; only the root directory
and the name of the second axis differ. Training is described first, then what is
specific to evaluation.

## Machine-local config

Two files directly under `config/` describe *your machine* rather than an experiment, so
both are gitignored. Each has a tracked `.example.yaml` next to it — copy and fill in:

```bash
cp config/dataset_paths.example.yaml config/dataset_paths.yaml
cp config/wandb.example.yaml config/wandb.yaml
```

| File | What it sets |
| --- | --- |
| `config/dataset_paths.yaml` | where each dataset and the released weights live. Not an overlay: resolved on demand by `utils/dataset_paths.py`, see [data.md](data.md) |
| `config/wandb.yaml` | which wandb account runs are logged to |

`config/wandb.yaml` keeps a personal entity out of the tracked configs:

```yaml
# config/wandb.yaml
entity: surfslam_stereo   # team or username the runs land under
user: null                # your username; used as the entity when `entity` is null
project: null             # overrides logging.wandb_project
```

A key left null changes nothing. It is layer 2b below, so an experiment config or the
command line still wins (`logging.wandb_entity=someone_else`), and `$SURF_WANDB_CONFIG`
points at a file somewhere else. Credentials are not part of it — run `wandb login`
once, or set `$WANDB_API_KEY`. To log nothing at all, set `logging.use_wandb=false`.

## Training configs

We have a rather involved config parsing system to allow ablations. If you only care about training or reproducing our experiments, the commands immediately below are all you need. Otherwise, read on.

An experiment is a **model** crossed with an **ablation**. Some examples of the syntax you can use:

```bash
# equivalent to --model defom_stereo --ablation warp_finetune_full --gpus 0,1
./train_scripts/launch.sh defom_stereo/warp_finetune_full --gpus 0,1

# equivalent to igev_pp/full_pretrain --gpus 0
./train_scripts/launch.sh --model igev_pp --ablation full_pretrain --gpus 0

# what exists on each axis
./train_scripts/launch.sh --list
```

We use an overlay system to parse the configurations. Configs are parsed in the following order, with later layers taking precedence over earlier layers. The base layer is the python implementation, then several possible layers of yaml, then the command line. This allows for ablations to be nested neatly.

```
1. Python Base (utils/arguments.py)
2. YAML layers
  2a. defaults.yaml                  applied to every run, config or not
  2b. ../wandb.yaml                  this machine's wandb account, if it exists
  2c. models/<model>/<model>.yaml    model identity, schedule, checkpoint registry
  2d. ablations/<ablation>.yaml      datasets, augmentation, which stage to resume
  2e. models/<model>/<ablation>.yaml  per-pair tuning, only where needed. this is a specialization on top of loading models/<model.yaml> AND ablations/<ablation>.yaml
3. Command Line (omegaconf style; hierarchical.key=value)
```

The chain comes from the `includes:` key at the top of each file, resolved relative to that file (`utils/arguments.py:load_config_file`). It takes a single path or a list.

## Ablations

See [ablations/naming.md](ablations/naming.md) for what the names mean.

In general training takes two steps: training a model from an upstream checkpoint using simulated underwater data, then fine-tuning using a self-supervised warping loss. That is, every warping run is a fine-tune of the matching `*_pretrain` run, so `warp_finetune_full` resumes `full_pretrain`.

Our model results from fine-tuning DEFOM stereo using `full_pretrain` then `warp_finetune_full` configs.

## Checkpoints

Our trained checkpoints are available [here](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411). 

Each model declares every checkpoint it can start from:

```yaml
# models/defom_stereo/defom_stereo.yaml
checkpoints:
  ours: stereo_weights/ours_vitl/best.pth
  full_pretrain: stereo_weights/ablation/fa_ia/best.pth

  # This refers to the upstream checkpoint supplied by the base model (e.g. by the original DEFOM stereo).
  # This is not shipped with our release, but may be downloaded to reproduce our results.
  pretrained: null
```

Relative paths are resolved against the `model_weights` root from `config/dataset_paths.yaml` (`utils/dataset_paths.py:resolve_checkpoint`). Absolute paths override the name resolution system.

An ablation names the *stage* it resumes from. This will automatically pull in the correct checkpoint from the corresponding `models/<whichever_model>/<whichever_model>.yaml`

```yaml
# ablations/warp_finetune_full.yaml
io:
  restore_checkpoints: ${checkpoints.full_pretrain}
```

## Layout

Paths are relative to `config/train/`.

| Path | What lives there |
| --- | --- |
| `defaults.yaml` | Shared values and defaults for all runs |
| `ablations/*.yaml` | One file per ablation configuration |
| `models/<model>/<model>.yaml` | Defaults for an individual model |
| `models/<model>/<ablation>.yaml` | Tuning values that OVERRIDE the config that would be loaded from a given model + ablation specification |


## Running one

```bash
./train_scripts/launch.sh defom_stereo/warp_finetune_full --gpus 0,1
./train_scripts/defom_stereo/warp_finetune_full.sh 0,1          # convenience wrappers
```

All configs are handled with OmegaConf, so anything in the schema can be set
inline:

```bash
./train_scripts/launch.sh igev_pp/full_pretrain 0 optimization.learning_rate=1e-5
```

## Adding things

- **New ablation, all models**: add `ablations/<NAME>.yaml`
- **New model**: add `models/<name>/<name>.yaml` with a `checkpoints:` registry covering the stages the ablations you care about reference.
- **One pair needs different hyperparameters**: add `models/<model>/<ablation>.yaml` with `includes: [<model>.yaml, ../../ablations/<ablation>.yaml]` and only the keys that differ.
- Wrappers at `train_scripts/<model>/<ablation>.sh` are optional convenience


# Eval configs

An evaluation is a **model** crossed with a **suite** — same `includes:` chain,
merge order and OmegaConf overrides as training, with `config/eval/` as the root
and `suites/` as the second axis:

```bash
./eval_scripts/inference.sh defom_stereo/suds_test 0
./eval_scripts/inference.sh --model igev_pp --suite qualitative --gpus 0
./eval_scripts/inference.sh --list
```

The one thing that differs: **an eval model file inherits its training counterpart**
rather than restating anything.

```yaml
# config/eval/models/defom_stereo/defom_stereo.yaml
includes: ../../../train/models/defom_stereo/defom_stereo.yaml
evaluation:
  checkpoints: [ours, full_pretrain]
```

That include carries the architecture block and the `checkpoints:` registry, so a
checkpoint is always named by **stage**, never by path, and a stage added to the
training model file is visible to both. Stages resolve against the composed
registry rather than the `model:` name — `defom_stereo` and `defom_stereo_vits`
both declare `model: defom_stereo` but have different weights.

`config/eval/defaults.yaml` includes the training defaults, then turns off what is
training-only (wandb, the qualitative logging loader, synthetic water).

| Path | What lives there |
| --- | --- |
| `config/eval/defaults.yaml` | Applied to every evaluation run |
| `config/eval/suites/*.yaml` | One file per suite: which datasets, which split |
| `config/eval/models/<model>/<model>.yaml` | Which checkpoints to sweep, memory settings |
| `config/eval/models/<model>/<suite>.yaml` | Per-pair tuning, if a pair ever needs it |
| `config/eval/scoring/*.yaml` | Re-scoring precomputed predictions (see below) |
| `config/eval/reconstruction/*.yaml` | 3D reconstruction evaluation (see below) |

## Suites

| Suite | Data | Metrics |
| --- | --- | --- |
| `suds_test` | SUDS test split, 9 scenes / 1468 frames, full resolution | yes — the release ships ground truth for every frame |
| `qualitative` | SVIn2 and Lizard Island | no ground truth; predictions and pictures only |

## Models

| Model | What it is |
| --- | --- |
| `defom_stereo`, `defom_stereo_vits` | ours, ViT-L and ViT-S |
| `foundation_stereo`, `igev_pp` | upstream baselines; weights not in our release |
| `underwater_stereo` | upstream baseline (BGNet). Evaluation only |

`underwater_stereo` (BGNet) is evaluation-only: its input conventions live in
`evaluate.py:forward_eval` rather than `demo/core.py`, and its architecture caps
disparity at **192 px** — below the SUDS test range, so `evaluate.py` warns and
records `architectural_max_disparity` in `summary.json`. The clipping concentrates
in the close-range scenes, so read the per-scene table. RAFT-Stereo was removed
for the release and is not evaluable.

### How the scored pixel set is defined

Two knobs decide it, both recorded in `summary.json` so a table can be traced back:

| Key | Default | Effect |
| --- | --- | --- |
| `evaluation.use_foreground_mask` | `true` | ANDs the SUDS foreground mask into the valid mask. Only 607 of 1468 frames have one, so the aggregate mixes two definitions; off scores all finite non-zero GT (what the previous pipeline did). |
| `evaluation.check_gt_rectification` | `true` | Refuses to load a scene whose GT was generated against different intrinsics. Catches `wilson_candidate3`; no shape check can. |

`evaluation.max_gt_disparity` (default null) clips GT above a threshold before scoring.

### Naming checkpoints in a sweep

A registry stage is its own name. A path is named after its **run directory**
(every run writes `best.pth`); two checkpoints resolving to the same name would
share an output directory, so a collision is refused. Name one explicitly:

```bash
./eval_scripts/inference.sh defom_stereo/suds_test 0 \
    'evaluation.checkpoints=[haze_only=/runs/a/best.pth,no_warp=/runs/b/best.pth]'
```

Evaluation does **not** go through the training data pipeline, which resizes every
sample to 320×736; `evaluate.py` builds its datasets at native resolution with no
transform.

## Scoring precomputed predictions

`score_predictions.py` re-scores the `<scene>/data/*.pt` payloads an evaluation
run wrote, without loading a model — different masks, thresholds or regions cost
seconds instead of a re-inference:

```bash
./eval_scripts/score.sh tbnms scoring.predictions_dir=eval/defom_stereo_suds_test/ours
./eval_scripts/score.sh tbnms scoring.run=eval/defom_stereo_suds_test   # every stage
./eval_scripts/score.sh --list
```

It writes `metrics.jsonl` / `metrics.csv` (one row per frame × variant × region),
`summary.json`, and `latex_rows.txt` (paste rows per scene) to
`scoring.output_dir`, default `<predictions_dir>/scores`; resumable like
evaluate.py. Two **variants**, each scoring a few pixel **regions**:

| Variant | Region | Pixels |
| --- | --- | --- |
| `masked` | `on_geometry` | annotated GT ∧ finite prediction ∧ foreground mask |
| `masked` | `water_column` | finite GT ∧ finite prediction ∧ *not* foreground |
| `masked` | `combined` | union of the two |
| `unmasked` | `annotated` | annotated GT ∧ finite prediction |
| `unmasked` | `full` | finite prediction (∧ finite GT) |

Two **ground-truth sources** (`scoring.gt_source`); the foreground rule is tied
to the source because the two mask releases encode the value 1 with opposite
meanings:

| Source | Layout | Foreground rule |
| --- | --- | --- |
| `suds` (default) | the SUDS release, via `io.suds_stereo_dir` | mask > 0 ({0,1,255} masks) |
| `legacy` | `scoring.legacy_gt_dir`/`legacy_mask_dir`, the pre-release `TBNMS_EVALUATION_FINAL` tree the CVPR tables came from | mask > 127, resized to the GT shape |

`summary.json` reports every region both pixel-weighted (`overall`,
`per_sequence`) and as unweighted per-frame means (`*_frame_averaged`). The
paper's masked tables were frame-averaged, its unmasked ones pixel-weighted;
`latex_rows.txt` follows that.

Frames without a foreground mask are skipped for the masked variant and counted
in `summary.json`; a scene with no masks at all fails loudly, since a wrong mask
directory would otherwise produce numbers that look fine.
`scoring.require_masks=false` scores such frames anyway, with `on_geometry`
degrading to all annotated GT. `scoring.save_visualizations=true` writes a 2×3
GT/prediction/region-error panel per frame under `<scene>/viz_error/`.

## 3D reconstruction evaluation

`evaluate_reconstruction.py` measures reconstructions against ground-truth point
clouds. open3d and kaolin (plus `point_cloud_utils` for mesh mode, `kornia` for
frames mode) are imported on first use, so nothing else in the repo depends on them.

```bash
# one mesh: Chamfer L1/L2, accuracy/completeness (mean/median/p95), F-score, Hausdorff
./eval_scripts/reconstruction.sh mesh reconstruction.mesh=recon.obj \
    reconstruction.gt_cloud=fused.ply reconstruction.output_dir=results/

# per-frame depth maps (*.npy) as point clouds; sweeps <results_root>/<method>/<scene>/depths,
# GT at <gt_root>/<scene>/COLMAP/dense/fused.ply, resuming past finished scenes
./eval_scripts/reconstruction.sh frames reconstruction.results_root=... \
    reconstruction.gt_root=... reconstruction.intrinsics=intrinsics.txt
```

## Adding things

- **New suite**: add `config/eval/suites/<NAME>.yaml` naming its datasets. A dataset also
  needs a branch in `evaluate.py:build_eval_dataset`.
- **New model**: add `config/eval/models/<name>/<name>.yaml` that includes the training
  model file and names the stages to evaluate.
- **Lab-only baselines**: `config/eval/models/*_internal/` is git-excluded, the same way
  `config/train/models/*_internal/` is. A one-line file that includes the internal
  training config gets you the baselines whose weights are not part of the public
  release (see `config/train/models_internal/how_to_use.txt`).