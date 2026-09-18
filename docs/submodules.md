# Submodules and patches

Every submodule in this project points at its **original upstream repository**, pinned
to an exact commit. We maintain no public forks. The changes we need are kept as patch
files in [`patches/`](../patches/), applied by:

```bash
./scripts/setup_submodules.sh          # idempotent; safe to re-run
./scripts/setup_submodules.sh --check  # report state, change nothing
./scripts/setup_submodules.sh --reset  # discard submodule changes, re-apply
```

`train_scripts/launch.sh` and `eval_scripts/inference.sh` run `--check` before launching
and refuse to start if anything is out of place, because a moved pin produces numbers
that look plausible and are not comparable to the reported ones.

What `--check` actually verifies, so you know what it is worth:

* every submodule is at the exact commit its patch declares (`# base:`) — this is the
  one that catches a moved pin;
* the patch is applied (it reverse-applies cleanly).

It does **not** detect arbitrary hand-edits inside an already-patched file. That is
deliberate: a rename patch leaves its targets as untracked files, so `git diff` cannot
see them anyway, and `ignore = dirty` exists precisely so you *can* hack on a submodule
while developing. The guard is there to catch the accidents — a forgotten setup run, or
a pin that moved underneath you — not to police intentional edits.

## What is pinned, and what we change

| Submodule | Upstream | Pinned commit | Our patch |
|---|---|---|---|
| `models/FoundationStereo` | [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | `6ea8225` | 1 file: build DINOv2 from vendored copy, not torch.hub |
| `models/DEFOM-Stereo` | [Insta360-Research-Team/DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) | `5b27591` | rename `core` → `defom_core` |
| `models/IGEV-plusplus` | [gangweiX/IGEV-plusplus](https://github.com/gangweiX/IGEV-plusplus) | `099dec9` | rename `core` → `igev_core`, `core_rt` → `igev_core_rt` |
| `models/Underwater_Stereo` | [Jinyi-Z/Underwater_Stereo](https://github.com/Jinyi-Z/Underwater_Stereo) | `9972b9c` | 5 files, ~11 lines |
| `utils/dtd` | [madhubabuv/dtd](https://github.com/madhubabuv/dtd) | `9d965a6` | 3 files, stereo warping |

Each patch opens with a comment header giving its base commit and the reason for every
change in it. Read those first — they are the real documentation. `git apply` ignores
the header, so it is free to be verbose.

DEFOM-Stereo and IGEV-plusplus are pinned to commits that are **identical to their
upstream `main` tips**. Our `models/Underwater_Stereo` pin `9972b9c` is a squashed
re-upload of Jinyi-Z `402edf6`; all 95 blob hashes match, so the two are interchangeable.

### Why the renames

All three stereo architectures ship a top-level package called `core`, and this project
puts all three on one `sys.path`. Only one can own that name. FoundationStereo keeps it,
which is what lets FoundationStereo need no rename patch; DEFOM-Stereo and IGEV-plusplus
rename theirs. Nothing else about either model is modified — the patches are directory
renames plus the import lines that follow, generated mechanically.

### Why FoundationStereo needs almost nothing

Everything this project needs it gets from upstream unmodified. It imports as
`FoundationStereo.core.*` through PEP 420 namespace packages, because `models/` is on
`sys.path` — no `__init__.py` required. `utils/train_utils.py` constructs it with an
OmegaConf, which supports both `cfg['max_disp']` and `cfg.max_disp`, so upstream's own
accessors work unchanged.

Its patch touches one file, `depth_anything/dpt.py`: the DINOv2 backbone is built from
FoundationStereo's own vendored `dinov2/` tree (`dinov2.hub.backbones`) instead of
`torch.hub.load('facebookresearch/dinov2', ...)`, and an unused
`vit_small`/`vit_base`/`vit_large` import is dropped (upstream master removed it too).

Upstream's hub call downloads facebookresearch/dinov2 **main** at whatever state it is
in that day. That broke in Dec 2025 — their `hubconf.py` grew
`from dinov2.hub.cell_dino.backbones import ...`, which dies with
`ModuleNotFoundError: No module named 'dinov2.hub.cell_dino'` whenever the vendored
`dinov2` is already bound in `sys.modules` — and it can drift again any time Meta
pushes to main. It stays invisible if you have a torch.hub cache predating the drift,
which is why long-lived containers keep working while a fresh clone or rebuilt image
fails outright.

Building from the vendored copy removes the network and the drift entirely, and it is
verified bit-exact against the hub route the published runs used: identical state-dict
keys and shapes, max abs diff 0.0 on `get_intermediate_layers` outputs with identical
weights. The hub call never supplied weights anyway (`pretrained_dino` defaults to
`False`); all weights come from our checkpoints. Worth knowing if you ever consider
dropping the patch to get back to pure upstream: it is load-bearing for
reproducibility, not tidiness.

#### One module is imported twice, harmlessly

`core/utils/utils.py` ends up in `sys.modules` under two names: `core.utils.utils`
(how FoundationStereo's own files reach it) and `FoundationStereo.core.utils.utils`
(how we import `InputPadder`). So there are two `InputPadder` classes.

This is fine. The module is a ~90-line leaf with nine public names, no module-level
state and no registry, and each copy is used entirely within its own scope — nothing
does `isinstance` across the boundary. It costs one extra parse.

Worth knowing because a *previous* version of this code had a much worse instance of
the same thing: a half-finished conversion to relative imports left `core/update.py`
importing `core.extractor` absolutely while its neighbours used `from .submodule`,
which loaded `extractor` and `submodule` twice — two copies of the timm- and
DepthAnything-backed classes. Reverting to upstream's consistent absolute imports
fixed that. If you ever want the last duplicate gone too, import `InputPadder` from
`core.utils.utils` instead of `FoundationStereo.core.utils.utils`; it is not worth
making the parent's imports inconsistent for.

### Why not NVlabs/master

master is ~21 commits ahead with things we want: a cuDNN fix for hierarchical inference,
a python 3.11 environment, explicit imports replacing `import *`, and flash-attn replaced
by `F.scaled_dot_product_attention` (which would drop a slow dependency from
`docker/container_base.Dockerfile`).

It also carries a regression. In `core/submodule.py`, `FlashMultiheadAttention.forward`
reshapes Q/K/V to `(B, L, H, D)` — the layout `flash_attn_func` expects — and upstream
now hands those tensors to `F.scaled_dot_product_attention`, which reads the last two
dimensions as `(sequence, head_dim)`. So it attends over the head axis with the sequence
axis as a batch dimension: different math from the flash-attn our checkpoints trained
with, and there is no transpose correcting it.

Moving to master is therefore its own piece of work — fix the transpose, then re-validate
every released checkpoint. See `patches/foundation-stereo.patch` for the full note.

## Regenerating a patch

If you deliberately move a pin, the patch must be regenerated against the new base or
`setup_submodules.sh` will refuse to run.

For the two mechanical renames:

```bash
./scripts/regenerate_rename_patches.sh    # rebuilds defom-stereo.patch, igev-plusplus.patch
```

For the hand-authored patches (`underwater-stereo`, `dtd`), rebase the existing patch:

```bash
cd models/Underwater_Stereo
git checkout --force --detach <new-base>
git apply --3way ../../patches/underwater-stereo.patch    # resolve any conflicts
git diff > /tmp/new.patch
# then splice /tmp/new.patch under the original '# ...' header, updating '# base:'
```

Keep the header. It is the only place the reasoning lives, and the `# base:` line is
what the setup script validates against.

## Where the old forks went

The project previously used `umfieldrobotics` forks. Those repositories still hold the
full experiment history and can stay private. In each submodule the fork remains
configured as a remote named `archive` (`internal`, for `utils/dtd`), so the history is
still reachable locally:

```bash
git -C models/FoundationStereo log archive/3dv_experiments
```

Tips at the time of the switch:

| Submodule | Fork | Branch | Tip |
|---|---|---|---|
| `models/FoundationStereo` | `umfieldrobotics/FoundationStereo` | `3dv_experiments` | `8f26af6` |
| `models/DEFOM-Stereo` | `umfieldrobotics/uw_defom_stereo` | `3dv_experiments` | `6565439` |
| `models/IGEV-plusplus` | `umfieldrobotics/IGEV-plusplus` | `main` | `d90910e` |
| `models/Underwater_Stereo` | `umfieldrobotics/Underwater_Stereo` | `main` | `19b8510` |
| `utils/dtd` | `umfieldrobotics/uw_dtd` | `master` | `de95506` |

Everything dropped in the switch was either dead code, an abandoned experiment
(the attenuation and background-segmentation branches), reformatting, or scratch
scripts. The uncommitted working-tree changes that existed at switchover were archived
to `.submodule-archive-20260813/` — delete that directory once you are satisfied.

## Nested submodules

`utils/dtd` declares three submodules of its own — `unimatch` and two RobotCar dataset
tools — that nothing here imports; we use only `dtd.losses.photometric_loss` and
`dtd.models.image_warping`. `setup_submodules.sh` deliberately does **not** pass
`--recursive`, so those are never cloned. Do not add it: it would pull ~34MB of
unrelated code and make setup fail whenever one of those hosts is unavailable.
