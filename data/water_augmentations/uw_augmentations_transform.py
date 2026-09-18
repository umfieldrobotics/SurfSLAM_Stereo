from dataclasses import dataclass
import torch

from data.geometry.fast_geometry_utils import depth_to_normals
from data.water_augmentations.uw_augmentations import WaterAugmentations, WaterAugAblationSettings

@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

class UWAugmentationsTransform:
    def __init__(self,
                 water_aug_cfg_path: str,
                 aug_settings: WaterAugAblationSettings,
                 prob: float = 0.5,
                 device = torch.device("cpu")):
        
        self.device = device
        self.prob = prob
        self.aug_severity = 1.0
        self.water_aug = WaterAugmentations(water_aug_cfg_path,
                                            aug_settings,
                                            self.device)

    def set_aug_severity(self, severity: float):
        self.aug_severity = severity
        self.water_aug.set_aug_severity(severity)

    def __call__(self, item: dict):
        K = item["intrinsics"]
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        
        # compute normals and 3D points
        left_depth = item["left_depth"]
        left_normals, left_pts3d = depth_to_normals(left_depth, fx, fy, cx, cy)
        
        item["left_normals"] = left_normals
        item["left_pts"] = left_pts3d
        
        right_depth = item["right_depth"]
        right_normals, right_pts3d = depth_to_normals(right_depth, fx, fy, cx, cy)
        item["right_normals"] = right_normals
        item["right_pts"] = right_pts3d
        
        left_augmented, right_augmented, left_attn, right_attn \
            = self.water_aug(item)
        
        item.pop("left_normals")
        item.pop("right_normals")
        item.pop("left_pts")
        item.pop("right_pts")
        
        item["left_uw"] = left_augmented
        item["right_uw"] = right_augmented
        item["left_attenuation"] = left_attn
        item["right_attenuation"] = right_attn
        
        item["right_intrinsics"] = item["intrinsics"]
        item["depth_valid"] = torch.tensor(True, dtype=torch.bool, device=self.device)
        item["has_augmentation"] = torch.tensor(True, dtype=torch.bool, device=self.device)

        return item
        