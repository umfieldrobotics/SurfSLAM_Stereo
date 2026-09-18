import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.dataset_paths import get_dataset_root
from utils.index_cache import cached_index
from utils.progress import progress


def _read_img(path, dtype):
    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).to(dtype)


def _read_disp_png(path, dtype):
    disp = np.array(Image.open(path)).astype(np.float32)
    return torch.from_numpy(disp).to(dtype)


def _read_disp_pfm(path, dtype):
    file = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return torch.from_numpy(data.copy()).to(dtype).contiguous()

def _read_disp_any(path, dtype):
    if path.endswith(".pfm"):
        return _read_disp_pfm(path, dtype)
    return _read_disp_png(path, dtype)


def _safe_exists(*paths):
    return all([Path(p).exists() for p in paths])


def _norm_join(a, b):
    return os.path.normpath(os.path.join(str(a), str(b)))


class FlyingThings3DDataset(Dataset):
    """
    Loads FlyingThings3D_subset with layout:
        root/driving/disparity/15mm_focallength/scene_backwards/fast/left/0001.pfm
        root/driving/disparity/15mm_focallength/scene_backwards/fast/right/0001.pfm
        root/driving/frames_finalpass/15mm_focallength/scene_backwards/fast/left/0001.png
        root/driving/frames_finalpass/15mm_focallength/scene_backwards/fast/right/0001.png
        root/driving/disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm
        root/driving/disparity/35mm_focallength/scene_backwards/fast/right/0001.pfm
        root/driving/frames_finalpass/35mm_focallength/scene_backwards/fast/left/0001.png
        root/driving/frames_finalpass/35mm_focallength/scene_backwards/fast/right/0001.png
        root/flyingthings/disparity/TEST/{A/B/C}/0000/left/0001.pfm
        root/flyingthings/disparity/TEST/{A/B/C}/0000/right/0001.pfm
        root/flyingthings/frames_finalpass/TEST/{A/B/C}/0000/left/0001.png
        root/flyingthings/frames_finalpass/TEST/{A/B/C}/0000/right/0001.png
        (same structure for TRAIN/)
        monkaa/disparity/{SEQUENCE}/left/0001.pfm
        monkaa/disparity/{SEQUENCE}/right/0001.pfm
        monkaa/frames_finalpass/{SEQUENCE}/left/0001.png
        monkaa/frames_finalpass/{SEQUENCE}/right/0001.png
        
        Intrinsics for 15mm focal length:
            fx=450.0	0.0	cx=479.5
            0.0	fy=450.0	cy=269.5
            0.0	0.0	1.0
        Intrinsics for 35mm focal length and every other scene:
            fx=1050.0	0.0	cx=479.5
            0.0	fy=1050.0	cy=269.5
            0.0	0.0	1.0
        Baseline: 1.0
    """
    def __init__(
        self,
        root=None,
        test_mode=False,
        transform=None,
        device="cpu",
        dtype=torch.float32,
        pad_to_32=True
    ):
        self.root = get_dataset_root("flyingthings", override=root)
        self.split = "val" if test_mode else "train"

        self.transform = transform
        self.dtype = dtype
        self.device = torch.device(device)
        self.pad_to_32 = pad_to_32
        self.samples = []

        entries = cached_index("flyingthings", self.root, self._INDEX_VERSION,
                               lambda: self._scan_index(self.root))
        # NOTE: only the flyingthings/ subtree has a TRAIN/TEST split, so driving/ and monkaa/
        # are kept in full for both train and val, i.e. those subsets overlap between the
        # two splits. This bug is left as-is to retain parity with the published results.
        wanted_ft_tag = "flyingthings/" + {"train": "TRAIN", "val": "TEST"}[self.split]
        for entry in entries:
            tag = entry["tag"]
            if tag.startswith("flyingthings/") and tag != wanted_ft_tag:
                continue
            sample = dict(entry)
            sample["id"] = len(self.samples)
            self.samples.append(sample)

        if not self.samples:
            raise RuntimeError("No FlyingThings3D samples found in {}".format(self.root))

        print("Initialized FlyingThings3D with {} samples from split '{}'".format(
            len(self.samples), self.split
        ))
        
        self.intrinsics = torch.tensor([[1050.0, 0.0, 479.5], [0, 1050, 269.5], [0,0,1]]).to(self.device)
        self.baseline = 1.0
        
        self.fb = self.intrinsics[0,0]*self.baseline


    # ----------------------------------------------------------------------
    #: Bump when the scan functions change what they record, so caches invalidate.
    _INDEX_VERSION = 1

    @staticmethod
    def _scan_index(root) -> list:
        """Complete index across every subset, both flyingthings/ splits included.

        Entries carry no 'id'; ids are assigned per split after filtering.
        Slow (one stat per referenced file); run through cached_index.
        """
        entries = []
        FlyingThings3DDataset._scan_driving(root, entries)
        for split_name in ("TRAIN", "TEST"):
            FlyingThings3DDataset._scan_flyingthings(root, split_name, entries)
        FlyingThings3DDataset._scan_monkaa(root, entries)
        return entries

    @staticmethod
    def _scan_driving(root, entries):
        """Scan driving dataset with 15mm and 35mm focal lengths"""
        driving_path = root / "driving"
        if not driving_path.exists():
            return
        
        for focal_length in ["15mm_focallength", "35mm_focallength"]:
            for direction in ["scene_backwards", "scene_forwards"]:
                for speed in ["fast", "slow"]:
                    img_base = driving_path / "frames_finalpass" / focal_length / direction / speed
                    disp_base = driving_path / "disparity" / focal_length / direction / speed
                    
                    if not img_base.exists() or not disp_base.exists():
                        continue
                    
                    left_imgs = sorted((img_base / "left").glob("*.png"))
                    
                    # Determine intrinsics based on focal length
                    if focal_length == "15mm_focallength":
                        fx = fy = 450.0
                    else:
                        fx = fy = 1050.0
                    
                    for left_img_path in progress(
                            left_imgs,
                            desc=f"[flyingthings] driving/{focal_length}/{direction}/{speed}",
                            unit="frame", leave=False):
                        name = left_img_path.stem

                        left_img = str(left_img_path)
                        right_img = str(img_base / "right" / f"{name}.png")
                        disp_left = str(disp_base / "left" / f"{name}.pfm")
                        disp_right = str(disp_base / "right" / f"{name}.pfm")

                        if not _safe_exists(left_img, right_img, disp_left, disp_right):
                            continue

                        entries.append({
                            "left": left_img,
                            "right": right_img,
                            "disp_left": disp_left,
                            "disp_right": disp_right,
                            "tag": f"driving/{focal_length}",
                            "fx": fx,
                            "fy": fy,
                            "cx": 479.5,
                            "cy": 269.5,
                        })
    
    @staticmethod
    def _scan_flyingthings(root, split_name, entries):
        """Scan flyingthings dataset (TRAIN or TEST)"""
        flyingthings_path = root / "flyingthings"
        if not flyingthings_path.exists():
            return
        
        img_base = flyingthings_path / "frames_finalpass" / split_name
        disp_base = flyingthings_path / "disparity" / split_name
        
        if not img_base.exists() or not disp_base.exists():
            return
        
        # Scan through A, B, C subdirectories
        for letter in ["A", "B", "C"]:
            letter_path = img_base / letter
            if not letter_path.exists():
                continue
            
            # Find all sequence directories
            for seq_dir in progress(
                    sorted(letter_path.iterdir()),
                    desc=f"[flyingthings] flyingthings/{split_name}/{letter}",
                    unit="seq", leave=False):
                if not seq_dir.is_dir():
                    continue

                left_imgs = sorted((seq_dir / "left").glob("*.png"))
                
                for left_img_path in left_imgs:
                    name = left_img_path.stem
                    
                    left_img = str(left_img_path)
                    right_img = str(seq_dir / "right" / f"{name}.png")
                    
                    # Corresponding disparity paths
                    disp_seq_path = disp_base / letter / seq_dir.name
                    disp_left = str(disp_seq_path / "left" / f"{name}.pfm")
                    disp_right = str(disp_seq_path / "right" / f"{name}.pfm")
                    
                    if not _safe_exists(left_img, right_img, disp_left, disp_right):
                        continue

                    entries.append({
                        "left": left_img,
                        "right": right_img,
                        "disp_left": disp_left,
                        "disp_right": disp_right,
                        "tag": f"flyingthings/{split_name}",
                        "fx": 1050.0,
                        "fy": 1050.0,
                        "cx": 479.5,
                        "cy": 269.5,
                    })
    
    @staticmethod
    def _scan_monkaa(root, entries):
        """Scan monkaa dataset"""
        monkaa_path = root / "monkaa"
        if not monkaa_path.exists():
            return
        
        img_base = monkaa_path / "frames_finalpass"
        disp_base = monkaa_path / "disparity"
        
        if not img_base.exists() or not disp_base.exists():
            return
        
        # Find all sequence directories
        for seq_dir in progress(
                sorted(img_base.iterdir()),
                desc="[flyingthings] monkaa",
                unit="seq", leave=False):
            if not seq_dir.is_dir():
                continue

            left_imgs = sorted((seq_dir / "left").glob("*.png"))
            
            for left_img_path in left_imgs:
                name = left_img_path.stem
                
                left_img = str(left_img_path)
                right_img = str(seq_dir / "right" / f"{name}.png")
                
                # Corresponding disparity paths
                disp_seq_path = disp_base / seq_dir.name
                disp_left = str(disp_seq_path / "left" / f"{name}.pfm")
                disp_right = str(disp_seq_path / "right" / f"{name}.pfm")
                
                if not _safe_exists(left_img, right_img, disp_left, disp_right):
                    continue

                entries.append({
                    "left": left_img,
                    "right": right_img,
                    "disp_left": disp_left,
                    "disp_right": disp_right,
                    "tag": f"monkaa/{seq_dir.name}",
                    "fx": 1050.0,
                    "fy": 1050.0,
                    "cx": 479.5,
                    "cy": 269.5,
                })


    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)


    # ----------------------------------------------------------------------
    def _pad_to_32(self, tensor):
        h, w = tensor.shape[-2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h == 0 and pad_w == 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))


    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        e = self.samples[idx]

        left = _read_img(e["left"], self.dtype).permute(2, 0, 1)
        right = _read_img(e["right"], self.dtype).permute(2, 0, 1)

        disp_left = _read_disp_any(e["disp_left"], self.dtype)
        disp_right = _read_disp_any(e["disp_right"], self.dtype)

        # Get sample-specific intrinsics
        intrinsics = torch.tensor([
            [e["fx"], 0.0, e["cx"]],
            [0.0, e["fy"], e["cy"]],
            [0.0, 0.0, 1.0]
        ]).to(self.device)
        fb = e["fx"] * self.baseline
        
        left_disp = disp_left.to(self.device)
        left_depth = fb / left_disp
        left_depth[left_disp == 0] = 0
        
        right_disp = disp_right.to(self.device)
        right_depth = fb / right_disp
        right_depth[right_disp == 0] = 0
                                   
        if self.pad_to_32:
            left = self._pad_to_32(left)
            right = self._pad_to_32(right)
            disp_left = self._pad_to_32(disp_left)
            disp_right = self._pad_to_32(disp_right)
            left_depth = self._pad_to_32(left_depth)
            right_depth = self._pad_to_32(right_depth)

        # pretend UW fields
        left_uw = left.clone()
        right_uw = right.clone()
        left_att = torch.ones_like(left)
        right_att = torch.ones_like(right)
        
        
        sample = {
            "sequence": e["tag"],
            "frame_id": torch.tensor(e["id"], device=self.device),
            "id": torch.tensor(e["id"], device=self.device),

            "left_rgb": left.to(self.device),
            "right_rgb": right.to(self.device),

            "left_uw": left_uw.to(self.device),
            "right_uw": right_uw.to(self.device),

            "left_attenuation": left_att.to(self.device),
            "right_attenuation": right_att.to(self.device),

            "left_depth": left_depth.to(self.device),
            "right_depth": right_depth.to(self.device),

            "disparity": disp_left.to(self.device),
            
            "intrinsics": intrinsics.to(self.device),
            "right_intrinsics": intrinsics.to(self.device),
            "depth_valid": torch.tensor(True).to(self.device),
            "has_augmentation": torch.tensor(False, dtype=torch.bool, device=self.device),
            "baseline": torch.tensor(1.0, device=self.device),
        }

        if self.transform:
            return self.transform(sample)
        return sample
