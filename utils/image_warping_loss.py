import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from dtd.losses.photometric_loss import PhotometricLoss
from typing import Dict, Any
import torch
import torch.nn.functional as F
from kornia.geometry.depth import warp_frame_depth


class WarpingLoss:
    def __init__(self,
                 image_width: int,
                 image_height: int,
                 lambda_occam: float = 0.0,
                 occam_margin: float = 0.05,):
        
        self._loss = PhotometricLoss()

        self.lambda_occam = lambda_occam
        self._occam_margin = occam_margin
        
            
    def __call__(self,
                disp_preds,
                data_batch: Dict[Any, torch.Tensor],
                return_images=False):

        left_images = data_batch["left_uw"]
        right_images = data_batch["right_uw"]
        K_left = data_batch["intrinsics"].to(disp_preds)
        K_right = data_batch["right_intrinsics"].to(disp_preds)
        b = abs(data_batch["baseline"]).to(disp_preds)

        fx_left = K_left[:, 0, 0]
        fx_right = K_right[:, 0, 0]
        batch_size = disp_preds.shape[0]

        # R->L and L->R extrinsics (rectified baseline along +x)
        T_right_to_left = torch.eye(4).to(K_left).repeat(batch_size, 1, 1)
        T_right_to_left[:, 0, 3] = -b
        T_left_to_right = torch.eye(4).to(K_left).repeat(batch_size, 1, 1)
        T_left_to_right[:, 0, 3] = b

        disp_preds = disp_preds.clamp(min=1e-6)

        # Left <- Right
        depth_left = (fx_right * b).view(batch_size, 1, 1, 1) / disp_preds
        warped_left = warp_frame_depth(
            image_src=right_images.to(depth_left.dtype) / 255.,
            depth_dst=depth_left,
            src_trans_dst=T_right_to_left,
            camera_matrix=K_left
        )

        # Right <- Left
        depth_right = (fx_left * b).view(batch_size, 1, 1, 1) / disp_preds
        warped_right = warp_frame_depth(
            image_src=left_images.to(depth_right.dtype) / 255.,
            depth_dst=depth_right,
            src_trans_dst=T_left_to_right,
            camera_matrix=K_right
        )

        # Far-away: copy originals where disparity is too small
        far_mask = (disp_preds < 1)
        far_mask_L = far_mask.expand(left_images.shape)
        far_mask_R = far_mask.expand(right_images.shape)
        warped_left[..., far_mask_L] = left_images[..., far_mask_L].to(depth_left.dtype) / 255.
        warped_right[..., far_mask_R] = right_images[..., far_mask_R].to(depth_right.dtype) / 255.

        # Mask empty warps
        empty_mask_left = (warped_left == 0).all(1, keepdim=True)
        empty_mask_right = (warped_right == 0).all(1, keepdim=True)

        err_left = self._loss(warped_left, left_images.to(depth_left.dtype) / 255.)
        err_right = self._loss(warped_right, right_images.to(depth_right.dtype) / 255.)

        err_left[empty_mask_left] = 0
        err_right[empty_mask_right] = 0
        
        err = 0.5 * (err_left + err_right)
        # if it was zero in either frame, the "average" is just the value from the other frame.
        err[empty_mask_left | empty_mask_right] *= 2

        # Symmetric Occam penalty
        occam_improvement = None
        if self.lambda_occam > 0:
            zero_warped_left = right_images.to(depth_left.dtype) / 255.
            zero_err_left = self._loss(zero_warped_left, left_images.to(depth_left.dtype) / 255.)

            zero_warped_right = left_images.to(depth_right.dtype) / 255.
            zero_err_right = self._loss(zero_warped_right, right_images.to(depth_right.dtype) / 255.)

            improvement_left = zero_err_left - err_left
            improvement_right = zero_err_right - err_right
            occam_improvement = 0.5 * (improvement_left + improvement_right)

            occam_penalty = F.relu(self._occam_margin - occam_improvement)
            err += self.lambda_occam * occam_penalty

        if return_images:
            out_dict = {
                "warped_left": warped_left,
                "warped_right": warped_right,
                "error_left": err_left,
                "error_right": err_right,
                "error": err
            }
            return err, out_dict

        return err
