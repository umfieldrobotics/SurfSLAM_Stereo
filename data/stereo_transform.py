import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class StereoTransform:
    def __init__(self, new_size=(320, 736),
                 is_train=False,
                 max_depth: float = None):

        self.new_h, self.new_w = new_size
        self.new_size = new_size
        self.is_train = is_train
        self.max_depth = max_depth

        self._keys = ["left_rgb", "right_rgb", "left_uw",
                      "right_uw", "left_depth", "right_depth", "disparity",
                      "left_attenuation", "right_attenuation"]

    def __call__(self, sample):
        # Add default flag if not already present (will be overridden by UWAugmentationsTransform if applied)
        if "has_augmentation" not in sample:
            sample["has_augmentation"] = torch.tensor(False, dtype=torch.bool)
        
        init = False
        for key in self._keys:
            if key not in sample:
                continue
            
            if not init:
                h0, w0 = sample[key].shape[-2:]
                sx = self.new_w / w0
                sy = self.new_h / h0
                init = True
        
            interpolation_mode = InterpolationMode.NEAREST if "depth" in key else InterpolationMode.BILINEAR
            current_val = sample[key]
            if "disparity" in key or "depth" in key:
                current_val = current_val.unsqueeze(0)
            new_val = TF.resize(current_val, (self.new_h, self.new_w), interpolation=interpolation_mode)
            if "disparity" in key or "depth" in key:
                new_val = new_val.squeeze(0)
            if "disparity" in key:
                new_val *= sx
                
            if "depth" in key and self.max_depth is not None:
                new_val = torch.clamp(new_val, min=0, max=self.max_depth)
            sample[key] = new_val
        
        for key in ["intrinsics", "right_intrinsics"]:
            if key in sample:
                K = sample[key].clone()
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy
                sample[key] = K

        sample["valid_mask"] = torch.ones_like(sample["left_depth"]).bool()

        return sample
