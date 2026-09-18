import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path

from data.water_augmentations.uw_augmentations_transform import Intrinsics
from utils.dataset_paths import get_dataset_root
from utils.index_cache import cached_index
from utils.progress import main_rank_print, progress


def read_img(filename):
    # convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert("RGB")).astype(np.float32)
    return torch.from_numpy(img)


def read_disp(file_path, focal_times_baseline):
    depth = np.load(file_path)
    disp = focal_times_baseline / depth  # Convert depth to disparity
    return torch.from_numpy(disp)


def read_depth(file_path):
    depth = np.load(file_path)
    return torch.from_numpy(depth.astype(np.float32))


#: Bump when _scan_index changes what it records, so on-disk caches invalidate.
_INDEX_VERSION = 1


def _scan_index(data_dir) -> list:
    """Complete split-independent frame index. Slow; run through cached_index."""
    main_rank_print(f"[tartanair] Globbing left images under {data_dir}...")
    left_files = [Path(p) for p in sorted(glob(os.path.join(data_dir, "*/*/*/image_left/*.png")))]

    entries = []
    missing = []
    for i, left_file in enumerate(progress(
            left_files, desc="[tartanair] verifying frames", unit="frame")):
        frame_id = left_file.stem.split("_")[0]
        traj_dir = left_file.parent.parent

        right_file = traj_dir / "image_right" / f"{frame_id}_right.png"
        left_depth_file = traj_dir / "depth_left" / f"{frame_id}_left_depth.npy"
        right_depth_file = traj_dir / "depth_right" / f"{frame_id}_right_depth.npy"

        absent = [p for p in (right_file, left_depth_file, right_depth_file) if not p.exists()]
        if absent:
            missing.extend(absent)
            continue

        entries.append({
            "sequence": left_file.parent.parent.parent.parent.stem,
            "left": str(left_file),
            "right": str(right_file),
            "left_depth": str(left_depth_file),
            "right_depth": str(right_depth_file),
            "id": i,
            "frame_id": int(frame_id),
        })

    if missing:
        raise RuntimeError(
            f"TartanAir dataset at {data_dir} is incomplete: {len(missing)} file(s) "
            f"referenced by a left image are absent, e.g. {missing[0]}"
        )
    return entries


class TartanAirDataset(Dataset):

    intrinsics = Intrinsics(
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480
    )

    baseline = 0.25
    focal_times_baseline = intrinsics.fx * baseline

    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: torch.device = 'cpu',
        transform=None,
        test_sequences: Optional[List[str]] = ['endofworld'],
        test_mode=False
    ):
        self.device = device
        data_dir = get_dataset_root("tartanair", override=data_dir)

        entries = cached_index("tartanair", data_dir, _INDEX_VERSION,
                               lambda: _scan_index(data_dir))
        self.sequences = {e["sequence"] for e in entries}

        skipped_sequences = set()
        self.samples = []
        for entry in entries:
            sequence = entry["sequence"]
            if (test_mode and sequence not in test_sequences) \
                or (not test_mode and sequence in test_sequences):
                    skipped_sequences.add(sequence)
                    continue  # Skip samples from the wrong split
            self.samples.append(dict(entry))

        self.transform = transform
        self.K_tensor = torch.tensor(
            [[self.intrinsics.fx, 0, self.intrinsics.cx],
             [0, self.intrinsics.fy, self.intrinsics.cy],
             [0,0,1]], device=self.device)
        self.width_tensor = torch.tensor(self.intrinsics.width, device=self.device)
        self.height_tensor = torch.tensor(self.intrinsics.height, device=self.device)
        
        mode_str = "Test" if test_mode else "Train"
        print(f"[{mode_str} mode] Initialized TartanAir dataset. Num samples: {len(self)}. " \
              "Skipped the following sequences:", *skipped_sequences)
        
    def __getitem__(self, index):
        sample: Dict[str, torch.Tensor] = {}
        sample_path = self.samples[index]

        sample["left_rgb"] = read_img(sample_path["left"]).permute(2,0,1)  # [H, W, 3]
        sample["right_rgb"] = read_img(sample_path["right"]).permute(2,0,1)
        sample["left_depth"] = read_depth(sample_path["left_depth"])  # [H, W]
        sample["right_depth"] = read_depth(sample_path["right_depth"])  # [H, W]

        if "left_depth" in sample_path and sample_path["left_depth"] is not None:
            sample["disparity"] = read_disp(sample_path["left_depth"], self.focal_times_baseline)  # [H, W]

        sample = {k:v.to(self.device) for k,v in sample.items()}
        
        sample["intrinsics"] = self.K_tensor
        # sample["width"] = self.width_tensor
        # sample["height"] = self.height_tensor
        sample["baseline"] = torch.tensor(self.baseline).to(self.device)
        sample["sequence"] = sample_path["sequence"]
        sample["id"] = torch.tensor(sample_path["id"], device=self.device)
        sample["frame_id"] = torch.tensor(sample_path["frame_id"], device=self.device)

        transformed = self.transform(sample) if self.transform is not None else sample
        

        return transformed
        
    def __len__(self):
        return len(self.samples)
