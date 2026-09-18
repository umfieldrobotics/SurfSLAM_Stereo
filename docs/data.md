# Data Setup

Start by downloading data. The following datasets are supported. We trained on all datasets; to use a subset, the configurations will need to be adjusted to remove the datasets you do not have.

| Dataset Name | Link | Description | Notes |
|---|---|---| --- |
| SUDS | [https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411) | Real-World Shipwreck Dataset | Download STEREO.tar which contains SUDS_STEREO|
| UWSim | [https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411) | Simulated underwater scenes with no underwater effects | Download STEREO.tar which contains UWSim|
| TartanAir | [https://theairlab.org/tartanair-dataset/](https://theairlab.org/tartanair-dataset/) | Simulated in-air stereo across many environments | Third-party. Register as `tartanair`; `train_datasets: [tartanair]` |
| FlyingThings3D / SceneFlow | [https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | Synthetic stereo (FlyingThings3D + Driving + Monkaa) | Third-party. Use `frames_finalpass`. Register as `flyingthings` |
| SVIn2 | [https://huggingface.co/datasets/afrl-uw/stereo-vi-underwater-dataset](https://huggingface.co/datasets/afrl-uw/stereo-vi-underwater-dataset) | Real underwater stereo (caves, wrecks, pool); no disparity GT | Third-party. Register as `svin2` |
| Lizard Island | [https://deepblue.lib.umich.edu/data/concern/data_sets/rj430494s](https://deepblue.lib.umich.edu/data/concern/data_sets/rj430494s) | Real underwater reef stereo, COLMAP reconstruction; no disparity GT | Third-party. Calibration is baked into `data/lizard_island_dataset.py`. Register as `lizard_island` |



```bash
# one-time setup
cp config/dataset_paths.example.yaml config/dataset_paths.yaml
<editor> config/dataset_paths.yaml    # fill in the paths you have; leave the rest null
python utils/dataset_paths.py         # print what resolves
```

---

## 1. What's in the release

The public release tree looks like this:

```
DATA_RELEASE/
    SUDS_STEREO/          # real-world stereo (this is what the code used to call "TBNMS")
    UWSim/                # simulated stereo  (what the code calls "oceansim")
    SUDS_SLAM/            # SLAM sequences: measurements.hdf5, trajectories, reconstructions
    weights/              # pretrained stereo + SLAM checkpoints
```

| Folder | Needed for | Notes |
|---|---|---|
| `SUDS_STEREO` | stereo training (`train_datasets: [tbnms]`) and real-world evaluation | Raw distorted pairs + calibration + split + ground truth |
| `UWSim` | stereo training (`train_datasets: [oceansim]`) | Unaugmented renders; the water augmentation is applied at train time |
| `weights` | inference / fine-tuning from released checkpoints | `stereo_weights/{ours_vitl,ours_vits,ablation}`, `slam_weights/`. Registered as `model_weights` |
| `SUDS_SLAM` | SLAM experiments only | Not used by any loader in this repo |

## 2. Download and extract

The archives are Zstandard-compressed with a long-distance matching window. **You must pass
`--long=30` when decompressing** or extraction fails with a window-size error:

```bash
tar -I 'zstd -d --long=30' -xf SUDS_STEREO.tar.zst
tar -I 'zstd -d --long=30' -xf UWSim.tar.zst
```

Each archive extracts to a single top-level folder (`SUDS_STEREO/`, `UWSim/`).

## 3. Point the repo at your data

Copy the template and edit it. `config/dataset_paths.yaml` is gitignored — it is
machine-specific. (It used to live at the repo root; move an existing one into `config/`
or point `$SURF_DATASET_PATHS` at it.)

```bash
cp config/dataset_paths.example.yaml config/dataset_paths.yaml
```

```yaml
# config/dataset_paths.yaml
suds_stereo: /data/DATA_RELEASE/SUDS_STEREO
uwsim: /data/DATA_RELEASE/UWSim
model_weights: /data/DATA_RELEASE/weights
svin2: null            # leave null for datasets you don't use
lizard_island: /data/LizardIslandColmap
tartanair: null
flyingthings: null
```

A `null` (or omitted) key is only an error if a run actually needs that dataset. To see what
resolves and whether each path looks right:

```bash
python utils/dataset_paths.py
```

```
paths file: /home/you/SurfSLAM_stereo/config/dataset_paths.yaml
  flyingthings   unset     -
  lizard_island  ok        /data/LizardIslandColmap
  suds_stereo    ok        /data/DATA_RELEASE/SUDS_STEREO
  tartanair      unset     -
  uwsim          suspect   /data/DATA_RELEASE/UWSim
  ...
```

`suspect` means the directory exists but doesn't contain the file listed under *Verified by*
below — usually you pointed one level too high or too low. It's a warning, not an error; the
run continues.

### Keys

| Key | Points at | Verified by | Needed when `train_datasets` includes |
|---|---|---|---|
| `suds_stereo` | extracted `SUDS_STEREO/` | `data/calibration/stereo_calib.yaml` | `tbnms` |
| `uwsim` | extracted `UWSim/` | `metadata_files.json` | `oceansim` |
| `svin2` | SVIn2 root | `<sequence>/calib.yaml` | `svin2` |
| `lizard_island` | Lizard Island COLMAP root | `images/left/` | `lizard_island` |
| `tartanair` | TartanAir root | `*/*/*/image_left/` | `tartanair` |
| `flyingthings` | SceneFlow root | `*/frames_finalpass/` | `flyingthings` |
| `model_weights` | released checkpoints | — | (inference only) |

Released checkpoints are named relative to the `model_weights` root
(`stereo_weights/ours_vitl/best.pth`), so a config referring to one works on any machine that
has the release. See [configurations.md](configurations.md).

`suds_stereo` is also required by the qualitative logging loader, which runs on every training
job (see `logging.tbnms_samples` in `config/train/defaults.yaml`).

If a key is missing you get a `DatasetNotRegisteredError` naming the key and the line to add,
rather than a `FileNotFoundError` from inside a glob.

## 4. Third-party datasets

These are not part of the release; download them from their original sources and put the root
in `config/dataset_paths.yaml`. Expected layouts:

**SVIn2** — `<root>/<sequence>/{images/left,images/right,calib.yaml}`. Any top-level folder
containing a `calib.yaml` is treated as a sequence, so you can add or remove sequences freely.

**Lizard Island** — `<root>/images/{left,right}/NNNN.png` (zero-padded 4-digit names).
Calibration is baked into `data/lizard_island_dataset.py`.

**TartanAir** — `<root>/<env>/<Easy|Hard>/<P0xx>/{image_left,image_right,depth_left,depth_right}/`.

**FlyingThings3D / SceneFlow** — `<root>/{flyingthings,driving,monkaa}/` each with
`frames_finalpass/` and `disparity/` subtrees. See the docstring in
`data/flyingthings_dataset.py` for the full tree.

## 5. Overriding without editing the file

Two escape hatches, both useful for containers and cluster jobs:

- `SURF_<KEY>_DIR` environment variables, e.g. `SURF_SUDS_STEREO_DIR=/scratch/SUDS_STEREO`.
- `SURF_DATASET_PATHS=/shared/dataset_paths.yaml` to point at a shared paths file, instead
  of `config/dataset_paths.yaml`.

You can also override a single dataset from a config or the command line:

```bash
python train.py config_path=... io.suds_stereo_dir=/scratch/SUDS_STEREO
```

Precedence, highest first: `io.*_dir` config value → `SURF_<KEY>_DIR` →
`config/dataset_paths.yaml`.

### Docker

`docker/run.sh` reads `config/dataset_paths.yaml` itself and bind-mounts every non-null entry at
`/data/<key>` — `lizard_island: /path/to/LizardIslandColmap` becomes
`/data/lizard_island` inside the container. It prints the mount list on startup, and skips
keys that are null or point at something that doesn't exist on the host.

Inside the container, paths read from `config/dataset_paths.yaml` are re-pointed to `/data/<key>`
automatically, so the same (host) yaml file works on both sides. The remap only kicks in when
the process is in a container *and* `/data/<key>` exists, so a container started some other
way falls back to the literal yaml path. `SURF_<KEY>_DIR` and `io.*_dir` are never remapped —
they are assumed to already be container paths.

| Variable | Effect |
|---|---|
| `SURF_IN_DOCKER=1` | set by `docker/run.sh`; forces the in-container branch |
| `SURF_DOCKER_DATA_ROOT` | where the datasets are mounted (default `/data`) |

## 6. SUDS_STEREO layout reference

```
SUDS_STEREO/
    data/
        calibration/stereo_calib.yaml            # Kalibr: cam0/cam1, intrinsics, radtan, T_cn_cnm1
        splits/train.txt  val.txt  test.txt      # scene names, one per line
        splits/train/<scene>/raw_left/<timestamp_ns>.png
        splits/train/<scene>/raw_right/<timestamp_ns>.png
        splits/val/<scene>/...
        splits/test/<scene>/...
    ground_truth/                                # test scenes only
        disparity/<scene>/<timestamp_ns>.pt      # tensor under the "disparity" key
        masks/<scene>/masks/<timestamp_ns>_left_mask.png
        masks/<scene>/overlays/                  # visualizations, for spot-checking
        meshes/<scene>.ply
```

| Split | Scenes | Image pairs |
|---|---|---|
| `train` | 10 | 4,192 |
| `val` | 12 | 5,662 |
| `test` | 9 | 1,468 |

**The split files are authoritative.** `data/tbnms_dataset.py` reads `splits/*.txt` to decide
which scenes belong to train, val, and test. Any leftover `tbnms/<scene>` entries in a config's
`test_sequences` are ignored (with a warning) — that list now only governs the simulated
datasets (`tartanair/*`, `oceansim/*`).

The val split is large (5,662 frames) and has no ground truth, so training subsamples it to
`num_val_frames_per_dataset` frames — drawn round-robin across all 12 val scenes, deterministic
given `optimization.seed`. Raise `num_val_frames_per_dataset` to use more of it.

Images are raw and distorted; the loader rectifies them on the fly using
`data/calibration/stereo_calib.yaml`. Left and right filenames correspond 1:1.

`ground_truth/` is read only during evaluation (`load_ground_truth=True` in
`data/tbnms_dataset.py`). Note that masks are sparser than disparity maps (1,214 vs 1,468), so
they must be matched by scene and filename rather than by index.

## 7. Config notes

- `io.oceansim_dir` is deprecated in favour of `io.uwsim_dir`; the old name still works and
  prints a warning.
- `io.tartanair_dir` / `io.uwsim_dir` and the new `io.suds_stereo_dir`, `io.svin2_dir`,
  `io.lizard_island_dir`, `io.flyingthings_dir` all default to `null`, meaning "use the registry".
- `tbnms/*` entries were removed from `test_sequences` across `config/train/*.yaml`.

Naming: the class is still `TBNMSDataset` and the `train_datasets` key is still `tbnms`, so
existing configs and checkpoints keep working. `SUDSStereoDataset` is available as an alias.
