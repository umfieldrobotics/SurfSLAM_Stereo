import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .arguments import Config
from utils.image_warping_loss import WarpingLoss

# From: https://arxiv.org/abs/2405.11158
def get_disparity_smooth_loss(disp, img):
    if img.max() > 10:
        img = img.to(disp) / 255.0

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
    grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class IgevPlusPlusLoss:
    def __init__(self, args: Config):
        self.max_disp0 = args.igev_pp.max_disp0
        self.max_disp1 = args.igev_pp.max_disp1
        self.max_disp = args.igev_pp.max_disp
        self.loss_gamma = args.igev_pp.loss_gamma

        width = args.rendering.output_width
        height = args.rendering.output_height
        self.warping_loss = WarpingLoss(
            image_width=width,
            image_height=height,
            lambda_occam=args.optimization.lambda_occam,
            occam_margin=args.optimization.occam_margin,
        )
        self.lambda_warping = args.optimization.lambda_warping

    def __call__(self, batch, agg_preds, iter_preds):
        n_predictions = len(iter_preds)
        assert n_predictions >= 1

        disp_gt = batch["disparity"].unsqueeze(1).to(iter_preds[0])
        depth_valid = batch["depth_valid"]

        valid = (disp_gt.abs() > 0) & (disp_gt.abs() < self.max_disp)
        valid[~depth_valid] = 1

        mag = torch.sum(disp_gt * disp_gt, dim=1).sqrt()
        mask0 = (valid & (mag[:,None] < self.max_disp0))
        mask1 = (valid & (mag[:,None] < self.max_disp1))
        mask  = (valid & (mag[:,None] < self.max_disp))

        l1_mask = valid.clone()
        l1_mask[~depth_valid] = 0
        warp_mask = valid.clone()
        warp_mask[depth_valid] = 0

        disp_loss = 0.0

        if l1_mask.any():
            disp_loss += 1.0 * F.smooth_l1_loss(
                agg_preds[0][mask0.bool()], disp_gt[mask0.bool()], reduction="mean"
            )
            disp_loss += 0.5 * F.smooth_l1_loss(
                agg_preds[1][mask1.bool()], disp_gt[mask1.bool()], reduction="mean"
            )
            disp_loss += 0.2 * F.smooth_l1_loss(
                agg_preds[2][mask.bool()], disp_gt[mask.bool()], reduction="mean"
            )

        adjusted_gamma = self.loss_gamma ** (15.0 / (n_predictions - 1))

        for i in range(n_predictions):
            pred = iter_preds[i]
            weight = adjusted_gamma ** (n_predictions - i - 1)

            if l1_mask.any():
                i_loss = (pred - disp_gt).abs()
                disp_loss += weight * i_loss[l1_mask.bool()].mean()

            if warp_mask.any():
                warp_item = self.lambda_warping * self.warping_loss(pred, batch)
                disp_loss += weight * warp_item[warp_mask].mean()

        epe = torch.sum((iter_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[mask.view(-1)]

        final_warp = self.lambda_warping * self.warping_loss(iter_preds[-1], batch)
        final_warp = final_warp[valid].mean()

        metrics = {
            "epe": epe.mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
            "warping_loss": final_warp.item(),
        }

        return disp_loss, metrics

class FoundationStereoLoss(nn.Module):
    def __init__(self,
                 args: Config,
                 gamma: float = 0.9,
                 max_disp: int = 416):
        
        self.gamma = gamma
        self.max_disp = max_disp
        
        width = args.rendering.output_width
        height = args.rendering.output_height
        self.warping_loss = WarpingLoss(image_width=width,
                                        image_height=height,
                                        lambda_occam=args.optimization.lambda_occam,
                                        occam_margin=args.optimization.occam_margin,)
        self.atten_thresh = args.rendering.background_attenuation_threshold
        
        self.freeze_disparity = args.optimization.freeze_disparity

        self.lambda_warping = args.optimization.lambda_warping
        self.lambda_smoothness = args.optimization.lambda_smoothness
        self.enable_bpX_filtering = args.optimization.enable_bpX_filtering
        self.bpX_threshold = args.optimization.bpX_threshold
        self.bpX_criteria = args.optimization.bpX_criteria
        self.bpX_min_samples = args.optimization.bpX_min_samples
        self.bpX_fallback_strategy = args.optimization.bpX_fallback_strategy
        
    def _compute_smoothness_per_sample(self, disp, img):
        """Compute smoothness loss per sample in the batch.
        
        Args:
            disp: [B, C, H, W] disparity predictions
            img: [B, C, H, W] RGB images
            
        Returns:
            [B] smoothness loss per sample
        """
        if img.max() > 10:
            img = img.to(disp) / 255.
            
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # Compute per-sample mean (reduce over spatial dims only)
        smooth_x = grad_disp_x.mean(dim=[1, 2, 3])  # [B]
        smooth_y = grad_disp_y.mean(dim=[1, 2, 3])  # [B]
        
        return smooth_x + smooth_y

        
    def __call__(self,
                 batch,
                 init_disp: torch.Tensor,
                 disp_preds: List[torch.Tensor],
                 ):

                                                            
        n_preds = len(disp_preds)
        assert n_preds >= 1, "disp_preds must contain at least one tensor"
        atten_gt=batch.get("left_attenuation", None).to(disp_preds[0])
        disp_gt = batch["disparity"].unsqueeze(1).to(disp_preds[0])

        depth_valid = batch["depth_valid"]
        valid = (disp_gt.abs() > 0) & (disp_gt.abs() < self.max_disp)
        valid[~depth_valid] = 1 # GT disparity can't be used for validity check 
        
        def upsample(pred):
            return F.interpolate(pred,
                                size=disp_gt.shape[-2:],
                                mode='bilinear',
                                align_corners=False)

        init_disp = upsample(init_disp)
        disp_preds = [upsample(d) for d in disp_preds]
        
        if init_disp.isnan().any() or init_disp.isinf().any():
            print(f"Warning: NaNs encountered in init_disp. Zeroing out.")
            init_disp = init_disp.nan_to_num(0, 0, 0)

        if atten_gt is not None:
            atten_thresh = self.atten_thresh
            fg_gt = (atten_gt > atten_thresh).any(dim=1, keepdim=True)

        else:
            fg_gt = torch.ones_like(valid)    

        disp_gt[~fg_gt] = 0.0
            
        
        loss = 0
        fallback_applied = False
        batch_size = disp_preds[-1].shape[0]
        
        if not self.freeze_disparity:
            l1_loss_mask = valid.clone()
            l1_loss_mask[~depth_valid] = 0
            
            warp_mask = valid.clone()
            warp_mask[depth_valid] = 0
            
            # Compute bpX filtering mask
            if self.enable_bpX_filtering:
                err = disp_preds[-1] - disp_gt
                epe_map = torch.linalg.vector_norm(err, ord=2, dim=1)
                # compute it per sample in the batch
                bpX = (epe_map > self.bpX_criteria).float().mean(dim=[1, 2])
                
                bpX_filter_mask = (bpX < self.bpX_threshold)
                # comput number of kept samples for maikng sure we have enough
                num_kept_samples = bpX_filter_mask.sum().item()
                
                # fallback strategy -- default to keeping the top k
                if num_kept_samples < self.bpX_min_samples:
                    if self.bpX_fallback_strategy == "disable":
                        bpX_filter_mask = torch.ones_like(bpX_filter_mask)
                        fallback_applied = True
                        
                    elif self.bpX_fallback_strategy == "relax":
                        _, indices = torch.sort(bpX)
                        bpX_filter_mask = torch.zeros_like(bpX_filter_mask)
                        bpX_filter_mask[indices[:self.bpX_min_samples]] = True
                        fallback_applied = True
                        
                    elif self.bpX_fallback_strategy == "skip":
                        bpX_filter_mask = torch.zeros_like(bpX_filter_mask)
                        fallback_applied = True
                
                bpX_spatial_mask = bpX_filter_mask.view(-1, 1, 1, 1).expand_as(valid)
                
                l1_loss_mask = l1_loss_mask & bpX_spatial_mask
                warp_mask = warp_mask & bpX_spatial_mask
            else:
                # No bpX filtering - use all samples
                pass
            
            if l1_loss_mask.any():
                loss += F.smooth_l1_loss(init_disp[l1_loss_mask], disp_gt[l1_loss_mask],
                                        reduction='mean')
                
            K = n_preds - 1
            for k, pred in enumerate(disp_preds):
                weight = self.gamma ** (K - k)
                    
                if pred.isnan().any() or pred.isinf().any():
                    print(f"Warning: NaNs encountered in prediction. Zeroing out.")
                    pred = pred.nan_to_num(0, 0, 0)
                                
                if l1_loss_mask.any():
                    loss += weight * torch.abs(pred - disp_gt)[l1_loss_mask].mean()
                
                # mask applies to the warping loss
                if k == K and warp_mask.any() and self.warping_loss is not None:
                    warp_loss_final = weight * self.lambda_warping * self.warping_loss(pred, batch)
                    loss += warp_loss_final[warp_mask].mean()

                # Regularizers - apply smoothness loss only to samples WITHOUT augmentation
                if self.lambda_smoothness > 0:
                    has_aug = batch.get("has_augmentation", torch.tensor(False, dtype=torch.bool, device=pred.device))
                    
                    if isinstance(has_aug, torch.Tensor):
                        if has_aug.dim() == 0:
                            if not has_aug.item():
                                smooth_loss = self.lambda_smoothness * weight * get_disparity_smooth_loss(pred, batch["left_rgb"])
                                loss += smooth_loss
                                smoothness_loss_for_metrics = smooth_loss.item()
                            else:
                                smoothness_loss_for_metrics = 0.0
                        else:
                            no_aug_mask = ~has_aug  # [B]
                            
                            if no_aug_mask.any():
                                no_aug_spatial = no_aug_mask.view(-1, 1, 1, 1).expand_as(pred)
                                
                                pred_masked = pred.clone()
                                img_masked = batch["left_rgb"].clone()
                                
                                # Zero out augmented samples (won't contribute to mean)
                                pred_masked[~no_aug_spatial] = 0
                                img_masked[~no_aug_spatial.expand_as(img_masked)] = 0
                                
                                # Compute smoothness loss (contributions from augmented samples are 0)
                                smooth_loss_map = self._compute_smoothness_per_sample(pred, batch["left_rgb"])  # [B]
                                
                                # Mask out augmented samples and compute mean over non-augmented samples
                                smooth_loss_per_sample = smooth_loss_map * no_aug_mask.float()
                                num_clean_samples = no_aug_mask.sum().clamp(min=1)  # Avoid division by zero
                                smooth_loss = self.lambda_smoothness * weight * smooth_loss_per_sample.sum() / num_clean_samples
                                
                                loss += smooth_loss
                                smoothness_loss_for_metrics = smooth_loss.item()
                            else:
                                smoothness_loss_for_metrics = 0.0
                    else:
                        # Plain boolean
                        if not has_aug:
                            smooth_loss = self.lambda_smoothness * weight * get_disparity_smooth_loss(pred, batch["left_rgb"])
                            loss += smooth_loss
                            smoothness_loss_for_metrics = smooth_loss.item()
                        else:
                            smoothness_loss_for_metrics = 0.0
                else:
                    smoothness_loss_for_metrics = 0.0

        err = disp_preds[-1] - disp_gt
        epe_map = torch.linalg.vector_norm(err, ord=2, dim=1)
        bpX_for_metrics = (epe_map > self.bpX_criteria).float().mean(dim=[1, 2])  # [B]
        num_filtered = (bpX_for_metrics >= self.bpX_threshold).sum().item()
        
        # Get original l1_loss_mask for metrics (without bpX filtering)
        l1_loss_mask_metrics = valid.clone()
        l1_loss_mask_metrics[~depth_valid] = 0
        
        epe = epe_map.view(-1)[l1_loss_mask_metrics.view(-1)].nan_to_num(0, 0, 0)
        
        # Compute final warping loss for metrics
        warp_loss_for_metrics = self.lambda_warping * self.warping_loss(disp_preds[-1], batch)
        final_warping_loss = warp_loss_for_metrics[valid].mean()
        
        metrics = {
            "epe": epe.mean().item(),
            "1px": (epe < 1.0).float().mean().item(),
            "2px": (epe < 2.0).float().mean().item(),
            "3px": (epe < 3.0).float().mean().item(),
            "5px": (epe < 5.0).float().mean().item(),
            "warping_loss": final_warping_loss.item(),
            "smoothness_loss": smoothness_loss_for_metrics,  # Always include in metrics
            "mean_bpX": bpX_for_metrics.mean().item(),
            "max_bpX": bpX_for_metrics.max().item(),
            "min_bpX": bpX_for_metrics.min().item(),
            "num_filtered_by_bpX": num_filtered,
            "bpX_filter_rate": num_filtered / batch_size,  # Percentage of batch filtered
            "num_samples_kept": batch_size - num_filtered,
            "fallback_applied": 1.0 if fallback_applied else 0.0,
        }
        
        return loss, metrics


