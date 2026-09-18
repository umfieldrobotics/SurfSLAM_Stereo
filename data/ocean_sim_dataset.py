import os
from pathlib import Path
import json
import numpy as np
from typing import Dict
from PIL import Image
from glob import glob
import sys

import torch
from torch.utils.data import Dataset
from typing import List, Optional

# Repo root, so first-party packages (`data.*`, `utils.*`) import by their real
# names rather than relying on whatever the caller happened to put on sys.path.
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from data.water_augmentations.uw_augmentations_transform import Intrinsics
from utils.dataset_paths import get_dataset_root
from utils.index_cache import cached_index
from utils.progress import progress

try:
    import orjson as _fastjson

    def _read_json(path: Path):
        return _fastjson.loads(path.read_bytes())

except ModuleNotFoundError:
    import json

    def _read_json(path: Path):
        with path.open("r") as fh:
            return json.load(fh)


def rgb_to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a CUDA RGB tensor of shape ([B?], 3, H, W) to grayscale.
    """
    in_dim = img.dim()
    if in_dim == 3:
        img = img[None]

    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input tensor must have shape ([B?], 3, H, W)")

    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    if in_dim == 3:
        return gray  # Return shape (1,H,W)
    return gray[:, None]  # Return shape (B, 1, H, W)


def _read_img(filename, dtype):
    # convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert("RGB"))
    return torch.from_numpy(img).to(dtype)


def _read_depth(file_path, dtype):
    depth = np.load(file_path)
    return torch.from_numpy(depth).to(dtype)


def _norm_join(base: Path, rel: str) -> str:
    """Path join → normalized str (cheaper than pathlib in tight loops)."""
    return os.path.normpath(os.path.join(base, rel))


class OceanSimDataset(Dataset):

    intrinsics = Intrinsics(
        fx=1099.498930273687,
        fy=1099.498930273687,
        cx=480.0,
        cy=272.0,
        width=960,
        height=544,
    )

    @staticmethod
    def _find_metadata_files(root: Path) -> List[str]:
        """Locate every scene's ``metadata.json`` under the UWSim root.

        The release ships a top-level ``metadata_files.json`` index. Its entries
        may have been written on another machine, so they are treated as
        root-relative first and only trusted if they actually resolve; otherwise
        we fall back to a recursive scan.
        """
        index_path = root / "metadata_files.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                entries = json.load(f)

            resolved, unresolved = [], 0
            for entry in entries:
                candidate = Path(entry)
                # Relative entries are root-relative; absolute ones are only trusted
                # if they still live under this root (they may be stale).
                option = candidate if candidate.is_absolute() else root / candidate
                option = option.resolve()
                if option.exists() and option.is_relative_to(root):
                    resolved.append(str(option))
                else:
                    unresolved += 1

            if resolved and unresolved == 0:
                return resolved
            print(f"[OceanSim] {index_path} lists {unresolved}/{len(entries)} paths that do not "
                  f"exist under {root}; falling back to a recursive scan.")

        return glob(f"{root}/**/metadata.json", recursive=True)

    #: Bump when _scan_index changes what it records, so on-disk caches invalidate.
    _INDEX_VERSION = 1

    @staticmethod
    def _scan_index(root: Path) -> list:
        """Complete split-independent frame index; run through cached_index.

        Depth-validity filtering and the intrinsics check happen here, once,
        so cached runs skip both. Entries carry no 'id'; ids are assigned per
        split after filtering.
        """
        K_expected = np.array(
            [
                [OceanSimDataset.intrinsics.fx, 0.0, OceanSimDataset.intrinsics.cx],
                [0.0, OceanSimDataset.intrinsics.fy, OceanSimDataset.intrinsics.cy],
                [0.0, 0.0, 1.0],
            ]
        )
        metadata_files = OceanSimDataset._find_metadata_files(root)

        entries = []
        for meta_file in progress(metadata_files,
                                  desc="[uwsim] reading scene metadata",
                                  unit="scene"):
            base = Path(meta_file).parent
            # top-level folder (e.g. “ocean”, “cave”, …)
            seq_name = base.relative_to(root).parts[0]

            frames = _read_json(Path(meta_file))["frames"]

            valid_path = base / "depth_validity.json"
            with open(valid_path, "r") as f:
                json_validity = json.load(f)

            for fr in frames:
                # every frame must match the class-level intrinsics, which is what
                # __getitem__ actually reports
                if not np.allclose(fr["K"], K_expected, rtol=0, atol=1e-6):
                    raise ValueError(
                        f"Frame {fr['frame_id']} in {meta_file} has intrinsics "
                        f"{fr['K']} that do not match the expected "
                        f"{K_expected.tolist()}"
                    )

                fr["left"] = fr["left"].replace("\\", "/")
                fr["right"] = fr["right"].replace("\\", "/")
                fr["depth"] = fr["depth"].replace("\\", "/")
                left_depth_path = fr["depth"]
                assert left_depth_path in json_validity, f"depth path {left_depth_path} missing on scene {seq_name}"

                if not json_validity[left_depth_path]:
                    continue

                entries.append(
                    {
                        "left": _norm_join(base, fr["left"]),
                        "right": _norm_join(base, fr["right"]),
                        "left_depth": _norm_join(base, fr["depth"]),
                        "right_depth": _norm_join(base, fr["depth"]).replace(
                            "left", "right"
                        ),
                        "frame_id": fr["frame_id"],
                        "param_id": fr["param_id"],
                        "baseline": float(fr["baseline"]),
                        "sequence": seq_name
                    }
                )
        return entries

    def __init__(
        self,
        root: str = None,
        test_sequences: Optional[List[str]] = ["ocean"],
        transform=None,
        device: str | torch.device = "cpu",
        test_mode=False,
        dtype=torch.float32
    ):
        self.dtype = dtype
        self.root = get_dataset_root("uwsim", override=root)
        self.device = torch.device(device)
        self.test_sequences = test_sequences if test_sequences is not None else []

        self.samples = []
        self.baselines_by_param = {}

        entries = cached_index("uwsim", self.root, self._INDEX_VERSION,
                               lambda: self._scan_index(self.root))

        skipped_sequences = set()
        for entry in entries:
            seq_name = entry["sequence"]
            if (test_mode and seq_name not in self.test_sequences) or (
                not test_mode and seq_name in self.test_sequences
            ):
                skipped_sequences.add(seq_name)
                continue  # Skip samples from the wrong split

            sample = dict(entry)
            sample["id"] = len(self.samples)
            # baseline consistency inside param_* sub-trees
            self.baselines_by_param.setdefault(sample["param_id"], sample["baseline"])
            self.samples.append(sample)

        self.transform = transform

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root}")

        mode_str = "Test" if test_mode else "Train"
        print(f"[{mode_str} mode] Initialized OceanSim dataset. Num samples: {len(self)}. " \
              "Skipped the following sequences:", *skipped_sequences)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample: Dict[str, torch.Tensor] = {}
        sample_path = self.samples[idx]

        sample["left_rgb"] = _read_img(sample_path["left"], dtype=self.dtype).permute(
            2, 0, 1
        )  # [H, W, 3]
        sample["right_rgb"] = _read_img(sample_path["right"], dtype=self.dtype).permute(2, 0, 1)
        left_depth = _read_depth(sample_path["left_depth"], dtype=self.dtype)  # [H, W]
        sample["left_depth"] = left_depth
        right_depth = _read_depth(sample_path["right_depth"], dtype=self.dtype)  # [H, W]
        sample["right_depth"] = right_depth

        # intrinsics and baseline
        baseline = float(sample_path["baseline"])
        fx = self.intrinsics.fx

        # compute disparity, isfinite mask accounts for nans and infs
        valid = (sample["left_depth"] > 0) & np.isfinite(sample["left_depth"])

        disp = torch.zeros_like(sample["left_depth"])
        disp[valid] = fx * baseline / sample["left_depth"][valid]
        sample["disparity"] = disp

        sample = {k: v.to(self.device) for k, v in sample.items()}

        sample["intrinsics"] = torch.tensor(
            [
                [self.intrinsics.fx, 0, self.intrinsics.cx],
                [0, self.intrinsics.fy, self.intrinsics.cy],
                [0, 0, 1],
            ],
            device=self.device,
        )
        # sample["height"] = torch.tensor(sample["left_rgb"].shape[1]).to(self.device)
        # sample["width"] = torch.tensor(sample["left_rgb"].shape[2]).to(self.device)
        sample["id"] = torch.tensor(sample_path["id"]).to(self.device)
        sample["frame_id"] = torch.tensor(sample_path["frame_id"]).to(self.device)
        sample["baseline"] = torch.tensor(baseline).to(self.device)
        sample["sequence"] = sample_path["sequence"]

        transformed = self.transform(sample) if self.transform else sample
        return transformed
