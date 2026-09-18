
import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(code_dir, os.pardir, "models")
defom_stereo_path = os.path.join(models_dir, "DEFOM-Stereo")
foundation_stereo_path = os.path.join(models_dir, "FoundationStereo")

sys.path.append(models_dir)
sys.path.append(defom_stereo_path)
sys.path.append(foundation_stereo_path)

from FoundationStereo.core.utils.utils import InputPadder
from utils.dataset_paths import get_dataset_root
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class LizardIslandDataset(Dataset):
    def __init__(self, transform=None, device='cpu', test_mode=False, root=None):
        if test_mode:
            raise ValueError("Test mode not supported for Lizard Island dataset")
        
        self.samples = []
        self.device = device

        self.transform = transform

        self.K = np.array([[1.27377361e+03, 0.00000000e+00, 6.77500000e+02],
                      [0.00000000e+00, 1.27377361e+03, 5.01000000e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.extrinsics = np.array(
            [[9.99999952e-01, 1.46388731e-04, -2.74263925e-04, -9.99294684e-02],
             [-1.46388453e-04, 9.99999989e-01, 1.03256163e-06, -3.15647258e-05],
             [2.74264073e-04, -9.92412512e-07, 9.99999962e-01, -4.82871686e-04],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # The COLMAP reconstruction these frames come from is already undistorted.
        dist_coeffs = np.zeros((4, 1))
        
        self.calib = {}
        self.width = 1355
        self.height = 1002
        
        image_shape = (self.width, self.height)
        R = self.extrinsics[:3, :3]
        # Column vector: OpenCV 5's stereoRectify rejects a 1-D translation where
        # 4.x accepted it (it fails inside gemm rather than saying so).
        t = self.extrinsics[:3, 3].reshape(3, 1)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K, dist_coeffs, self.K, dist_coeffs, image_shape, R, t
        )
        self.baseline = np.linalg.norm(t)


        self.data_root = get_dataset_root("lizard_island", override=root)
        self.left_folder = str(self.data_root / "images" / "left")
        self.right_folder = str(self.data_root / "images" / "right")
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(self.K, dist_coeffs, R1, P1, image_shape, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(self.K, dist_coeffs, R2, P2, image_shape, cv2.CV_32FC1)

        
        frames = os.listdir(self.left_folder)

        for path in frames:
            stem = os.path.basename(path).split(".")[0]
            if len(stem) != 4 or not stem.isdigit():
                raise ValueError(f"Unexpected frame filename in {self.left_folder}: {path}")
            frame_id = int(stem)
            self.samples.append({
                "frame_id": frame_id,
                "id": len(self.samples)
            })
            
        mode_str = "Test" if test_mode else "Train"
        print(f"[{mode_str} mode] Initialized Lizard Island dataset from {self.data_root}. "
              "Num samples:", len(self))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_id = sample["frame_id"]
        
        left_path = os.path.join(self.left_folder, f"{frame_id:04d}.png")
        right_path = os.path.join(self.right_folder, f"{frame_id:04d}.png")

        left_raw = imageio.imread(left_path)
        right_raw = imageio.imread(right_path)
        
        left_rect = cv2.remap(left_raw, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        left_img = torch.as_tensor(left_rect).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)
        right_img = torch.as_tensor(right_rect).float()[None].permute(0, 3, 1, 2).contiguous().to(self.device)
        
        padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
        left_img, right_img = padder.pad(left_img, right_img)

        fake_depth = torch.zeros_like(left_img.squeeze(0)[0])  # HxW

        data = {
            "sequence": "lizard_island",
            "frame_id": torch.tensor(frame_id, device=self.device),
            "id": torch.tensor(sample["id"], device=self.device),
            "left_rgb": left_img.squeeze(0),   # 3xHxW
            "right_rgb": right_img.squeeze(0),  # 3xHxW
            "left_uw": left_img.squeeze(0),
            "right_uw": right_img.squeeze(0),
            "intrinsics": torch.from_numpy(self.K).float().to(self.device),
            "right_intrinsics": torch.from_numpy(self.K).float().to(self.device),
            "baseline": torch.tensor(self.baseline, device=self.device),
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
