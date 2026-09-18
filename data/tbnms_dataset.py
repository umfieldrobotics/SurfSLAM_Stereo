"""Loader for the SUDS real-world stereo release (formerly referred to as TBNMS).

Expected on-disk layout, rooted at the ``suds_stereo`` entry in config/dataset_paths.yaml::

    SUDS_STEREO/
        data/
            calibration/stereo_calib.yaml       # Kalibr format, cam0/cam1
            splits/{train,val,test}.txt         # scene names, one per line
            splits/{train,val,test}/<scene>/raw_left/<timestamp_ns>.png
            splits/{train,val,test}/<scene>/raw_right/<timestamp_ns>.png
        ground_truth/                           # test scenes only
            disparity/<scene>/<timestamp_ns>.pt
            masks/<scene>/masks/<timestamp_ns>_left_mask.png
            meshes/<scene>.ply                  # not consumed by any loader

The split is defined by the release itself (``splits/*.txt``), not by the
``test_sequences`` list in ``config/train/*.yaml``. See ``docs/data.md``.

``ground_truth/`` is read only when ``load_ground_truth=True`` -- evaluation asks
for it, training does not, so the training path is unaffected by anything here.
"""

import os, sys
import glob
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

code_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(code_dir, os.pardir, "models")
defom_stereo_path = os.path.join(models_dir, "DEFOM-Stereo")
foundation_stereo_path = os.path.join(models_dir, "FoundationStereo")


sys.path.append(models_dir)
sys.path.append(defom_stereo_path)
sys.path.append(foundation_stereo_path)

import torch.nn.functional as F

from FoundationStereo.core.utils.utils import InputPadder
from utils.dataset_paths import get_dataset_root
from utils.progress import progress
from utils.rectification import load_kalibr_calibration

SPLITS = ("train", "val", "test")


class TBNMSDataset(Dataset):
    """SUDS real-world stereo pairs, rectified on the fly.

    Args:
        config: optional object with ``sequences`` and ``n_samples_per_sequence``.
            When given, those scenes are used verbatim (looked up across all
            splits) and ``split`` is ignored -- this is how the qualitative
            logging loader selects a handful of frames.
        split: one of ``train``, ``val``, ``test``, or ``all``.
        root: overrides the ``suds_stereo`` entry in config/dataset_paths.yaml.
        scene_filter: optional subset of scene names to keep within ``split``.
        max_frames: cap the number of frames, sampled round-robin across scenes
            so every scene stays represented. Deterministic given ``seed``.
        load_ground_truth: read ``ground_truth/`` and return real disparity plus a
            pixel-level ``disparity_valid`` mask. Only the test scenes have it.
            Left off for training, where SUDS frames feed the self-supervised
            warping loss and no GT term.
    """

    def __init__(self,
                 config=None,
                 transform=None,
                 device='cpu',
                 split: str = "train",
                 root=None,
                 scene_filter=None,
                 max_frames=None,
                 seed: int = 42,
                 test_sequences=None,
                 test_mode=False,
                 load_ground_truth: bool = False,
                 use_foreground_mask: bool = True,
                 check_ground_truth_rectification: bool = True):
        self.samples = []
        self.device = device
        self.root = get_dataset_root("suds_stereo", override=root)
        self.load_ground_truth = load_ground_truth
        self.use_foreground_mask = use_foreground_mask
        self.check_ground_truth_rectification = check_ground_truth_rectification
        self.gt_disparity_dir = self.root / "ground_truth" / "disparity"
        self.gt_mask_dir = self.root / "ground_truth" / "masks"
        #: Scenes whose GT intrinsics have been compared already, so the check costs
        #: one comparison per scene rather than one per frame.
        self._rectification_checked = set()

        self.splits_dir = self.root / "data" / "splits"
        if not self.splits_dir.is_dir():
            raise FileNotFoundError(
                f"{self.splits_dir} not found. Is 'suds_stereo' pointing at an extracted "
                f"SUDS_STEREO folder? Check `python utils/dataset_paths.py`."
            )

        # scene -> split, read from the release's own split files
        self.scene_splits = self._read_split_files()

        if config is not None:
            sequences = list(config.sequences)
            n_samples_per_sequence = config.n_samples_per_sequence
            split_label = "config"
        else:
            if split not in SPLITS and split != "all":
                raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
            if split == "all":
                sequences = sorted(self.scene_splits)
            else:
                sequences = [s for s, sp in sorted(self.scene_splits.items()) if sp == split]
            n_samples_per_sequence = None
            split_label = split

            if scene_filter:
                requested = set(scene_filter)
                missing = requested - set(sequences)
                if missing:
                    print(f"[SUDS/{split_label}] scene_filter entries not in this split, ignoring:",
                          *sorted(missing))
                sequences = [s for s in sequences if s in requested]

        if test_sequences:
            print("[SUDS] warning: `test_sequences` no longer defines the SUDS split -- "
                  "the release's splits/{train,val,test}.txt do. Ignoring:", *sorted(test_sequences))

        skipped_sequences = set()
        for sequence in progress(sequences,
                                 desc=f"[SUDS/{split_label}] scanning sequences",
                                 unit="seq"):
            sequence_path = self.sequence_path(sequence)
            if sequence_path is None:
                print(f"[SUDS/{split_label}] Unknown scene {sequence!r} (not in any split file). Skipping")
                skipped_sequences.add(sequence)
                continue

            left_frames = sorted(glob.glob(os.path.join(sequence_path, "raw_left", "*.png")))

            if len(left_frames) == 0:
                print(f"[SUDS/{split_label}] No frames found in sequence {sequence_path}. Skipping")
                skipped_sequences.add(sequence)
                continue

            if n_samples_per_sequence is None or n_samples_per_sequence < 0:
                selected = left_frames
            else:
                n = min(n_samples_per_sequence, len(left_frames))
                selected = np.random.choice(left_frames, n, replace=False)

            for path in selected:
                # Filenames are nanosecond capture timestamps; keep the stem as the
                # source of truth for file lookups and carry the int for logging.
                frame_stem = os.path.basename(path).split(".")[0]
                self.samples.append({
                    "sequence": sequence,
                    "frame_stem": frame_stem,
                    "frame_id": int(frame_stem),
                    "id": 0  # assigned below, after any subsampling
                })

        if max_frames is not None and 0 <= max_frames < len(self.samples):
            self.samples = self._subsample_across_scenes(self.samples, max_frames, seed)

        for i, sample in enumerate(self.samples):
            sample["id"] = i

        calibration_path = self.root / "data" / "calibration" / "stereo_calib.yaml"
        self.load_calib(calibration_path)

        self.transform = transform

        print(f"[SUDS/{split_label} mode] Initialized SUDS stereo dataset from {self.root}. "
              f"Scenes: {len(sequences) - len(skipped_sequences)}. Num samples: {len(self)}."
              + (" Skipped: " + " ".join(sorted(skipped_sequences)) if skipped_sequences else ""))

        if self.load_ground_truth:
            if not self.gt_disparity_dir.is_dir():
                raise FileNotFoundError(
                    f"load_ground_truth=True but {self.gt_disparity_dir} does not exist. "
                    f"Only the test scenes ship ground truth; check that 'suds_stereo' points "
                    f"at a complete SUDS_STEREO folder (`python utils/dataset_paths.py`).")
            n_gt = n_masked = 0
            for s in self.samples:
                disparity_path, mask_path = self.ground_truth_paths(
                    s["sequence"], s["frame_stem"])
                n_gt += disparity_path.exists()
                n_masked += mask_path.exists()
            self.n_frames_with_ground_truth = n_gt
            self.n_frames_with_foreground_mask = n_masked
            print(f"[SUDS/{split_label} mode] Ground truth disparity for {n_gt}/{len(self)} frames.")

            # Check one frame per scene now rather than discovering a mismatched scene
            # part way through a long run. Cheap: a handful of loads, once.
            if self.check_ground_truth_rectification:
                for sequence in dict.fromkeys(s["sequence"] for s in self.samples):
                    stems = [s["frame_stem"] for s in self.samples
                             if s["sequence"] == sequence]
                    if stems:
                        self.read_ground_truth(sequence, stems[0])
            if self.use_foreground_mask and 0 < n_masked < n_gt:
                # Worth stating plainly: the scored pixel set is not defined the same
                # way on every frame, and the aggregate pools both kinds.
                print(f"[SUDS/{split_label} mode] {n_masked}/{n_gt} of those also have a "
                      f"foreground mask, which is ANDed into the valid mask. The other "
                      f"{n_gt - n_masked} are scored on all finite non-zero ground truth. "
                      f"Set use_foreground_mask=False for one definition throughout.")
            elif not self.use_foreground_mask and n_masked:
                print(f"[SUDS/{split_label} mode] Ignoring {n_masked} foreground masks "
                      f"(use_foreground_mask=False); scoring all finite non-zero GT.")

    @staticmethod
    def _subsample_across_scenes(samples, max_frames, seed):
        """Take ``max_frames`` samples round-robin over scenes, so a scene with
        many frames can't crowd out the others. Deterministic given ``seed``."""
        rng = np.random.default_rng(seed)
        by_scene = {}
        for s in samples:
            by_scene.setdefault(s["sequence"], []).append(s)

        queues = []
        for scene in sorted(by_scene):
            frames = by_scene[scene]
            order = rng.permutation(len(frames))
            queues.append([frames[i] for i in order])

        kept = []
        for round_idx in range(max(len(q) for q in queues)):
            for q in queues:
                if round_idx < len(q):
                    kept.append(q[round_idx])
                    if len(kept) == max_frames:
                        return sorted(kept, key=lambda s: (s["sequence"], s["frame_stem"]))
        return sorted(kept, key=lambda s: (s["sequence"], s["frame_stem"]))

    def _read_split_files(self):
        """Map every scene named in ``splits/*.txt`` to its split."""
        scene_splits = {}
        for split in SPLITS:
            list_path = self.splits_dir / f"{split}.txt"
            if not list_path.exists():
                print(f"[SUDS] warning: missing split file {list_path}")
                continue
            with open(list_path, "r") as f:
                for line in f:
                    scene = line.strip()
                    if not scene or scene.startswith("#"):
                        continue
                    if scene in scene_splits:
                        print(f"[SUDS] warning: scene {scene!r} listed in both "
                              f"{scene_splits[scene]}.txt and {split}.txt")
                    scene_splits[scene] = split
        if not scene_splits:
            raise FileNotFoundError(f"No split files found under {self.splits_dir}")
        return scene_splits

    def sequence_path(self, sequence_name):
        """Absolute path to a scene folder, or None if the scene is unknown."""
        split = self.scene_splits.get(sequence_name)
        if split is None:
            return None
        return str(self.splits_dir / split / sequence_name)

    def format_path(self, sequence_name, frame_stem):
        """Left/right image paths for one frame."""
        base_dir = self.sequence_path(sequence_name)
        left_file = os.path.join(base_dir, "raw_left", f"{frame_stem}.png")
        right_file = os.path.join(base_dir, "raw_right", f"{frame_stem}.png")
        return left_file, right_file

    def ground_truth_paths(self, sequence_name, frame_stem):
        """Where this frame's GT disparity and foreground mask would live.

        Neither is guaranteed to exist: only the test scenes have disparity, and
        masks cover a subset of those frames (607 of 1468 in the release; the 1214
        files under masks/<scene>/ are 607 masks plus 607 overlays).
        """
        disparity = self.gt_disparity_dir / sequence_name / f"{frame_stem}.pt"
        mask = self.gt_mask_dir / sequence_name / "masks" / f"{frame_stem}_left_mask.png"
        return disparity, mask

    def _check_rectification(self, gt_intrinsics, sequence_name, disparity_path):
        """Fail if this scene's ground truth was built on a different rectification.

        Each GT payload carries the rectified ``K`` it was generated with. Disparity is
        ``fx * baseline / depth``, so a scene whose ``fx`` disagrees with the one
        ``load_kalibr_calibration`` computes from the shipped calibration carries a
        proportional bias in every disparity -- and a ``cy`` disagreement means its
        epipolar geometry is not ours at all. Neither is visible in the array shape,
        which is why this is checked rather than assumed.

        Only warns once per scene per process, then raises, so the message is not
        buried under one line per frame.
        """
        if sequence_name in self._rectification_checked:
            return
        self._rectification_checked.add(sequence_name)

        gt = torch.as_tensor(gt_intrinsics).float().reshape(3, 3)
        ours = self.left_intrinsics.float().reshape(3, 3)
        # Sub-1e-2 px differences are just float32 round-trips through the yaml.
        if torch.allclose(gt, ours, atol=1e-2):
            return

        fx_gt, fx_ours = gt[0, 0].item(), ours[0, 0].item()
        scale = fx_gt / fx_ours if fx_ours else float("nan")
        raise ValueError(
            f"ground truth for scene '{sequence_name}' was generated against a different "
            f"rectification than utils/rectification.py computes from "
            f"{self.root / 'data' / 'calibration' / 'stereo_calib.yaml'}.\n"
            f"  ground truth: fx={fx_gt:.4f} fy={gt[1, 1]:.4f} "
            f"cx={gt[0, 2]:.4f} cy={gt[1, 2]:.4f}   (from {disparity_path.name})\n"
            f"  this loader:  fx={fx_ours:.4f} fy={ours[1, 1]:.4f} "
            f"cx={ours[0, 2]:.4f} cy={ours[1, 2]:.4f}\n"
            f"Either the scene needs its own calibration file or its ground truth needs "
            f"regenerating. To measure the rest of the split meanwhile, exclude it:\n"
            f"  evaluation.scene_filter=[<the scenes you want>]\n"
            f"or, to accept the bias knowingly, "
            f"evaluation.check_gt_rectification=false")

    def read_ground_truth(self, sequence_name, frame_stem):
        """GT disparity and its validity mask for one frame, or ``(None, None)``.

        Adapted from the prior implementation on branch origin/inference/stereo_eval
        (visualize_inference.py:load_gt_disparity_for_frame), which read the
        pre-release ``DISPARITY_MAPS/<scene>/disparity_maps/`` layout.

        Returns float32 ``[H, W]`` disparity in pixels and a bool ``[H, W]`` mask.
        A pixel is valid when the disparity is finite and non-zero and, where a
        foreground mask exists for the frame, inside that mask.
        """
        disparity_path, mask_path = self.ground_truth_paths(sequence_name, frame_stem)
        if not disparity_path.exists():
            return None, None

        payload = torch.load(disparity_path.as_posix(), map_location="cpu", weights_only=False)
        # The release stores a dict; tolerate a bare tensor in case that changes.
        if isinstance(payload, dict):
            if "disparity" not in payload:
                raise KeyError(
                    f"{disparity_path} has no 'disparity' key (found: {sorted(payload)})")
            disparity = payload["disparity"]
            if self.check_ground_truth_rectification and "K" in payload:
                self._check_rectification(payload["K"], sequence_name, disparity_path)
        else:
            disparity = payload

        disparity = torch.as_tensor(disparity).squeeze().float()
        valid = torch.isfinite(disparity) & (disparity != 0)
        disparity = torch.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_foreground_mask and mask_path.exists():
            mask = imageio.imread(mask_path.as_posix())
            mask = torch.as_tensor(np.asarray(mask)).squeeze()
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != disparity.shape:
                raise ValueError(
                    f"{mask_path} is {tuple(mask.shape)} but its disparity is "
                    f"{tuple(disparity.shape)}")
            # Masks are {0, 1, 255}: 0 is outside the annotation, and both non-zero
            # classes are inside it. Verified against the shipped overlays -- class 0
            # is the only one the overlay leaves untinted, and 99% of GT pixels are
            # non-zero, so >0 is "annotated", not "background".
            valid &= mask > 0

        return disparity, valid

    def load_calib(self, path):
        """Build the rectification maps for the release's ZED rig.

        The geometry lives in utils/rectification.py so that anything else
        reading these raw frames -- the demo, in particular -- rectifies them
        exactly the way training and evaluation do.
        """
        rect = load_kalibr_calibration(path)
        self.map1_l, self.map2_l = rect.map1_l, rect.map2_l
        self.map1_r, self.map2_r = rect.map1_r, rect.map2_r
        self.baseline = rect.baseline_m
        self.left_intrinsics = torch.from_numpy(rect.left_intrinsics)
        self.right_intrinsics = torch.from_numpy(rect.right_intrinsics)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = sample["sequence"]
        frame_stem = sample["frame_stem"]
        left_path, right_path = self.format_path(sequence, frame_stem)

        imgL = cv2.imread(str(left_path), cv2.IMREAD_COLOR_RGB)
        imgR = cv2.imread(str(right_path), cv2.IMREAD_COLOR_RGB)
        try:
            rectL = cv2.remap(imgL, self.map1_l, self.map2_l, interpolation=cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)
        except:
            raise OSError("Failed to rectify images. Images are probably truncated.")

        left_img = torch.as_tensor(rectL).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)
        right_img = torch.as_tensor(rectR).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)

        padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
        native_shape = left_img.shape[-2:]
        left_img, right_img = padder.pad(left_img, right_img)

        fake_depth = torch.zeros_like(left_img.squeeze()[0])

        # Ground truth is stored at native resolution (1920x1080), while the images
        # above are padded up to a multiple of 32 (1920x1088). Pad the GT into the
        # same frame using this padder's own offsets so the two cannot drift, with
        # zeros rather than `replicate` -- the band carries no information, and the
        # validity mask marks it unusable.
        disparity = fake_depth
        disparity_valid = torch.zeros_like(fake_depth, dtype=torch.bool)
        depth_valid = False
        if self.load_ground_truth:
            gt_disparity, gt_valid = self.read_ground_truth(sequence, frame_stem)
            if gt_disparity is not None:
                if tuple(gt_disparity.shape) != tuple(native_shape):
                    raise ValueError(
                        f"GT disparity for {sequence}/{frame_stem} is "
                        f"{tuple(gt_disparity.shape)} but the rectified image is "
                        f"{tuple(native_shape)}. The GT was produced against a different "
                        f"rectification than utils/rectification.py computes.")
                pad = padder._pad  # [left, right, top, bottom], as passed to F.pad
                disparity = F.pad(gt_disparity[None, None], pad, value=0.0)[0, 0]
                disparity_valid = F.pad(
                    gt_valid[None, None].float(), pad, value=0.0)[0, 0] > 0.5
                depth_valid = True

        data = {
            "sequence": sequence,
            "frame_stem": frame_stem,
            "frame_id": torch.tensor(sample["frame_id"], dtype=torch.int64).to(self.device),
            "id": torch.tensor(sample["id"]).to(self.device),
            "left_rgb": left_img.squeeze(),
            "right_rgb": right_img.squeeze(),
            "left_uw": left_img.squeeze(),
            "right_uw": right_img.squeeze(),
            "intrinsics": self.left_intrinsics.to(self.device),
            "right_intrinsics": self.right_intrinsics.to(self.device),
            "baseline": torch.tensor(self.baseline).to(self.device),
            "depth_valid": torch.tensor(depth_valid).to(self.device),
            "has_augmentation": torch.tensor(False, dtype=torch.bool).to(self.device),
            "left_attenuation": torch.zeros_like(left_img.squeeze()),
            "right_attenuation": torch.zeros_like(right_img.squeeze()),
            "left_depth": fake_depth,
            "right_depth": fake_depth,
            "disparity": disparity.to(self.device),
        }

        # Per-pixel GT validity, present only when ground truth was asked for. Kept
        # out of the training dict on purpose: StereoTransform resizes `disparity`
        # but knows nothing about this key, and a full-resolution bool map the
        # training loop never reads is 2 MB per sample of wasted collate.
        if self.load_ground_truth:
            data["disparity_valid"] = disparity_valid.to(self.device)

        if self.transform:
            return self.transform(data)
        return data


#: Release-facing name. ``TBNMSDataset`` is kept for backwards compatibility with
#: existing configs and checkpoints.
SUDSStereoDataset = TBNMSDataset
