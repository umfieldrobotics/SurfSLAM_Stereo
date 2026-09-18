import logging
from typing import Dict, Optional
import torch
import torch.distributed as dist
from utils.arguments import Config
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
from tqdm import tqdm
from data.water_augmentations.utils import rgb_to_grayscale
from utils.image_warping_loss import WarpingLoss
autocast = torch.amp.autocast

# matplotlib logs "Clipping input data to the valid range for imshow..." on every
# imshow call (a log record, not a warning, so it never dedupes) whenever a float
# RGB image pokes outside [0..1]. The augmented/warped panels logged to wandb do
# that legitimately, and clipping is exactly the display behavior we want, so
# drop this one message instead of spamming it every validation pass.
logging.getLogger("matplotlib.image").addFilter(
    lambda record: "Clipping input data" not in record.getMessage())


class OSSafeIterator:
    def __init__(self, iterable):
        self._iterator = iter(iterable)
        self._size = len(iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return self._size

    def __next__(self):
        while True:
            try:
                return next(self._iterator)
            except OSError:
                continue


def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='gloo',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


_error_vmax_cache: dict[int, float] = {}


def log_disp_panel(
        step: int,
        img_l: torch.Tensor,
        img_r: torch.Tensor,
        disp_gt: Optional[torch.Tensor],
        disp_pred: torch.Tensor,
        tag: str = "disp_panel",
        bg_threshold: float = 0.95,
        atten_gt: Optional[torch.Tensor] = None,
        save_path: str = None,
        return_img: bool = False,
        warp_imgs: Dict[str, torch.Tensor] = None,
):
    """
    Visualize left image, predicted disparity, and any combination of
    ground-truth disparity, error, and a binary mask of ground-truth attenuation.
    """
    # --- prepare numpy arrays --------------------------------------------
    img = img_l[0].detach().float().cpu().numpy().transpose(1, 2, 0) / 255.0
    img_r = img_r[0].detach().float().cpu().numpy().transpose(1, 2, 0) / 255.0

    # pred is always present
    pred_np = disp_pred[0].detach().cpu().numpy().squeeze(0)

    # optional gt and error
    if disp_gt is not None:
        gt_np = disp_gt[0].detach().cpu().numpy().squeeze(0)
        err_np = np.abs(pred_np - gt_np)
    else:
        gt_np = None
        err_np = None

    # compute display ranges
    pos = pred_np[pred_np > 0]
    if gt_np is not None:
        pos = np.concatenate([pos, gt_np[gt_np > 0]])
    vmax_disp = np.percentile(pos, 95) if pos.size else 1.0

    if err_np is not None:
        key = hash(gt_np.tobytes())
        if key not in _error_vmax_cache:
            _error_vmax_cache[key] = np.percentile(err_np, 95) * 1.05
        vmax_err = _error_vmax_cache[key]
    else:
        vmax_err = None

    # optional ground-truth attenuation → binary mask
    if atten_gt is not None:
        atten_np = atten_gt[0].detach().cpu().numpy().squeeze(0)
        fg_gt_np = (atten_np > bg_threshold).astype(np.float32)

    # --- build panels list -----------------------------------------------
    panels = [
        {"data": img, "title": "Left image", "cmap": None, "vmin": None,
         "vmax": None, "colorbar": False},
        {"data": img_r, "title": "Right image", "cmap": None, "vmin": None,
         "vmax": None, "colorbar": False},
        {"data": pred_np, "title": "Disparity pred", "cmap": "magma", "vmin": 0,
         "vmax": vmax_disp, "colorbar": True},
        ]

    if gt_np is not None:
        panels.insert(
            1,
            {"data": gt_np, "title": "Disparity GT", "cmap": "magma", "vmin": 0,
             "vmax": vmax_disp, "colorbar": True})
        panels.append({"data": err_np,    "title": "Absolute Error",
                      "cmap": "magma", "vmin": 0,          "vmax": vmax_err, "colorbar": True})

    if atten_gt is not None and gt_np is not None:
        panels.append(
            {"data": fg_gt_np, "title": "GT mask", "cmap": "viridis", "vmin": 0,
             "vmax": 1, "colorbar": True})
        
    if warp_imgs is not None:
        warped_img_left = warp_imgs["warped_left"][0].detach().float().cpu().numpy().transpose(1, 2, 0)
        warped_img_right = warp_imgs["warped_right"][0].detach().float().cpu().numpy().transpose(1, 2, 0)

        err = warp_imgs["error"][0].detach().squeeze().float().cpu().numpy()
        
        panels.insert(2, {
            "data": warped_img_left, "title": "Warped: Right img to Left Frame", "cmap": None,
            "vmin": 0, "vmax": 1, "colorbar": False
        })
        panels.insert(3, {
            "data": warped_img_right, "title": "Warped: Left img to Right Frame", "cmap": None,
            "vmin": 0, "vmax": 1, "colorbar": False
        })
        panels.append({
            "data": err, "title": "Warp Error", "cmap": "magma",
            "vmin": 0, "vmax": err.max(), "colorbar": True
        })

    # --- create figure ---------------------------------------------------
    n_panels = len(panels)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))
    axes = axes.flatten()

    for i, p in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(
            p["data"],
            vmin=p["vmin"],
            vmax=p["vmax"],
            cmap=p["cmap"])
        ax.set_title(p["title"])
        ax.axis("off")
        if p["colorbar"]:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide any extra axes
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # --- output -----------------------------------------------------------
    if return_img:
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        arr = np.frombuffer(
            canvas.tostring_argb(),
            dtype=np.uint8).reshape(
            int(h),
            int(w),
            4)[..., 1:]
        plt.close(fig)
        return arr

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_validation_samples(
        log_loader,
        the_model,
        step: int,
        seed: int,
        device,
        dtype,
        rank: int,
        is_master: bool,
        args: Config,
        is_hardware: bool):
    """
    Log one batch per rank, gathering panels to rank 0.
    Drop-in for both single-GPU and DDP jobs.
    """
    if log_loader is None:
        return

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_panels = []

    # save RNG
    _py_state, _np_state = random.getstate(), np.random.get_state()
    _cpu_state, _gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state(device)

    # rank-specific seed
    viz_seed = seed + rank
    random.seed(viz_seed)
    np.random.seed(viz_seed)
    torch.manual_seed(viz_seed)
    torch.cuda.manual_seed_all(viz_seed)
    
    the_model = the_model.eval()

    warping_loss_obj = None

    with torch.no_grad():  # Critical: prevent gradient accumulation
        for b, batch in enumerate(
            tqdm(log_loader, desc=f"Logging (rank {rank})",
                 disable=not is_master, dynamic_ncols=True)
        ):
            
            if warping_loss_obj is None:
                warping_loss_obj = WarpingLoss(batch["left_uw"].shape[-1],
                                               batch["left_uw"].shape[-2])
            
            # Print memory status at start of iteration (helpful for debugging)
            img_l = batch["left_uw"].to(device, dtype=dtype)
            img_r = batch["right_uw"].to(device, dtype=dtype)

            if "disparity" in batch and batch["depth_valid"][0]:
                disp_gt = batch["disparity"].unsqueeze(1).to(device, dtype=dtype)
            else:
                disp_gt = None

            if args.model == "foundation_stereo":
                _, disp_preds = the_model(
                    img_l, img_r, iters=args.train_iters,
                    test_mode=False)
                if "left_attenuation" in batch:
                    atten_gt = rgb_to_grayscale(batch["left_attenuation"])
                else:
                    atten_gt = None
            elif args.model == "igev_plusplus":
                _, disp_preds = the_model(
                    img_l, img_r, iters=args.igev_pp.valid_iters,
                )
                atten_gt = None
            else:
                disp_preds = the_model(image1=img_l, image2=img_r,
                                    iters=args.train_iters)
                atten_gt = None
                
            _, warp_imgs = warping_loss_obj(disp_preds[-1], batch, return_images=True)

            # move to CPU so we can all_gather even on tiny GPUs
            img_np = log_disp_panel(
                step=step,
                img_l=img_l,
                img_r=img_r,
                disp_gt=disp_gt,
                disp_pred=disp_preds[-1],
                tag=f"sample_{b}",
                atten_gt=atten_gt,
                return_img=True,
                bg_threshold=args.rendering.background_attenuation_threshold,
                warp_imgs=warp_imgs
            )
            local_panels.append({"img": img_np, "tag": f"sample_{b}"})
            
            # Explicitly free GPU memory after each iteration
            del img_l, img_r, disp_gt, disp_preds, warp_imgs
            if atten_gt is not None:
                del atten_gt
            torch.cuda.empty_cache()
            
    # gather everything to rank 0
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_panels)
    else:
        gathered[0] = local_panels

    tag_prefix = "hardware_log" if is_hardware else "val_disp_panel"
    if is_master:
        for r, rank_panels in enumerate(gathered):
            for p in rank_panels:
                wandb.log(
                    {f"{tag_prefix}/r{r}_{p['tag']}": wandb.Image(p["img"])},
                    step=step
                )
    # restore RNG
    random.setstate(_py_state)
    np.random.set_state(_np_state)
    torch.set_rng_state(_cpu_state)
    torch.cuda.set_rng_state(_gpu_state, device)
