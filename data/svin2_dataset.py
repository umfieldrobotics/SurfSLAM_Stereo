import os, sys
import glob
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(code_dir, os.pardir, "models")
defom_stereo_path = os.path.join(models_dir, "DEFOM-Stereo")
foundation_stereo_path = os.path.join(models_dir, "FoundationStereo")


sys.path.append(models_dir)
sys.path.append(defom_stereo_path)
sys.path.append(foundation_stereo_path)


from FoundationStereo.core.utils.utils import InputPadder
from utils.dataset_paths import get_dataset_root


def _load_sequence_calibration(calib_path):
    with open(calib_path, "r") as f:
        calibration_data = yaml.safe_load(f)
    seq_calib = {}
    seq_calib["baseline_m"] = calibration_data['stereo_camera']['stereo']['baseline']

    for camera_key in ["left_camera", "right_camera"]:
        cam_data = calibration_data["stereo_camera"][camera_key]
        width = int(cam_data["width"])
        height = int(cam_data["height"])

        K = np.array(cam_data["K"], dtype=np.float32).reshape(3, 3)
        D = np.array(cam_data["D"], dtype=np.float32).reshape(-1)
        assert D.size >= 4
        D = D[:4].astype(np.float32)

        # Optional rectification and projection from file; fall back sensibly
        R = np.array(cam_data["R"], dtype=np.float32).reshape(3, 3)
        P = np.array(cam_data["P"], dtype=np.float32).reshape(3, 4)

        size = (width, height)

        # Precompute rectification maps
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, R, P[:, :3], size, cv2.CV_32FC1
        )

        seq_calib[camera_key] = {
            "width": width,
            "height": height,
            "K": K,
            "D": D,
            "R": R,
            "P": P,
            "size": size,
            "map1": map1,
            "map2": map2,
        }
    return seq_calib

def _load_sequence_calibration_kalibr(calib_path):
    with open(calib_path, "r") as f:
        calibration_data = yaml.safe_load(f)

    cam_mapping = {
        "cam0": "left_camera",
        "cam1": "right_camera"
    }


    seq_calib = {}
    # Baseline: take translation between cam0 and cam1 from T_cn_cnm1 if available
    if "cam1" in calibration_data and "T_cn_cnm1" in calibration_data["cam1"]:
        T = np.array(calibration_data["cam1"]["T_cn_cnm1"], dtype=np.float32)
        seq_calib["baseline_m"] = abs(T[0, 3])
    else:
        seq_calib["baseline_m"] = 0.0

    for cam_key in ["cam0", "cam1"]:
        if cam_key not in calibration_data:
            continue
        cam_data = calibration_data[cam_key]

        width, height = cam_data["resolution"]
        fx, fy, cx, cy = cam_data["intrinsics"]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        D = np.array(cam_data["distortion_coeffs"], dtype=np.float32)
        assert D.size >= 4
        D = D[:4]

        # Rectification matrix R: use identity (no rectification by default)
        R = np.eye(3, dtype=np.float32)

        # Projection matrix P: build from intrinsics
        P = np.zeros((3, 4), dtype=np.float32)
        P[:3, :3] = K

        size = (width, height)

        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, R, P[:, :3], size, cv2.CV_32FC1
        )

        seq_calib[cam_mapping[cam_key]] = {
            "width": width,
            "height": height,
            "K": K,
            "D": D,
            "R": R,
            "P": P,
            "size": size,
            "map1": map1,
            "map2": map2,
        }

    return seq_calib

class SVIN2Dataset(Dataset):
    def __init__(self, transform=None, device='cpu', test_sequences=[], test_mode=False, root=None):
        self.samples = []
        self.device = device
        self.root = get_dataset_root("svin2", override=root)

        # A sequence is any top-level folder carrying its own calibration.
        sequences = sorted(p.parent.name for p in self.root.glob("*/calib.yaml"))
        if not sequences:
            raise FileNotFoundError(
                f"No sequences found under {self.root} (expected <sequence>/calib.yaml). "
                f"Check `python utils/dataset_paths.py`."
            )

        self.transform = transform

        self.calib = {}

        skipped_sequences = set()

        for sequence in sequences:
            if test_mode and sequence not in test_sequences:
                skipped_sequences.add(sequence)
                continue
            elif (not test_mode) and sequence in test_sequences:
                skipped_sequences.add(sequence)
                continue

            sequence_path = self.sequence_path(sequence)

            left_frames = sorted(glob.glob(os.path.join(sequence_path, "images", "left/*")))

            for path in left_frames:
                frame_id = int(os.path.basename(path).split(".")[0])
                self.samples.append({
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "id": len(self.samples)
                })
            
            calib_path = f"{sequence_path}/calib.yaml"
            seq_calib = _load_sequence_calibration(calib_path)
   
            self.calib[sequence] = seq_calib

        mode_str = "Test" if test_mode else "Train"
        print(f"[{mode_str} mode] Initialized SVIN2 dataset from {self.root}. Num samples: {len(self)}. " \
              "Skipped the following sequences:", *skipped_sequences)

    def sequence_path(self, sequence_name):
        return str(self.root / sequence_name)

    def format_path(self, sequence_name, sample_id):
        base_dir = self.root / sequence_name / "images"
        return str(base_dir / "left" / f"{sample_id}.png"), str(base_dir / "right" / f"{sample_id}.png")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = sample["sequence"]
        frame_id = sample["frame_id"]
        left_path, right_path = self.format_path(sequence, frame_id)

        left_raw = imageio.imread(left_path)
        right_raw = imageio.imread(right_path)
        if left_raw is None or right_raw is None:
            raise FileNotFoundError(f"Missing images for sequence={sequence}, frame_id={frame_id}")

        calib_seq = self.calib[sequence]
        lmaps = calib_seq["left_camera"]
        rmaps = calib_seq["right_camera"]

        left_rect = cv2.remap(left_raw, lmaps["map1"], lmaps["map2"], interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, rmaps["map1"], rmaps["map2"], interpolation=cv2.INTER_LINEAR)

        left_img = torch.as_tensor(left_rect).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)
        right_img = torch.as_tensor(right_rect).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)

        padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
        left_img, right_img = padder.pad(left_img, right_img)

        fake_depth = torch.zeros_like(left_img.squeeze(0)[0])  # HxW

        data = {
            "sequence": sequence,
            "frame_id": torch.tensor(frame_id, device=self.device),
            "id": torch.tensor(sample["id"], device=self.device),
            "left_rgb": left_img.squeeze(0),   # 3xHxW
            "right_rgb": right_img.squeeze(0), # 3xHxW
            "left_uw": left_img.squeeze(0),
            "right_uw": right_img.squeeze(0),
            "intrinsics": torch.from_numpy(lmaps["K"]).float().to(self.device),
            "right_intrinsics": torch.from_numpy(rmaps["K"]).float().to(self.device),
            "baseline": torch.tensor(float(calib_seq["baseline_m"]), device=self.device),
            "depth_valid": torch.tensor(False, device=self.device),
            "has_augmentation": torch.tensor(False, dtype=torch.bool, device=self.device),
            "left_attenuation": torch.zeros_like(left_img.squeeze(0)),
            "right_attenuation": torch.zeros_like(right_img.squeeze(0)),
            "left_depth": fake_depth,
            "right_depth": fake_depth,
            "disparity": fake_depth,
        }

        if self.transform:
            return self.transform(data)
        return data
