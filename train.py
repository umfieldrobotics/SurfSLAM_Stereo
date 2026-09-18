
import sys
import os
import shutil


code_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(code_dir, "models")
defom_stereo_path = os.path.join(models_dir, "DEFOM-Stereo")
foundation_stereo_path = os.path.join(models_dir, "FoundationStereo")
igevpp_path = os.path.join(models_dir, "IGEV-plusplus")

sys.path.append(models_dir)
sys.path.append(igevpp_path)
sys.path.append(defom_stereo_path)
sys.path.append(foundation_stereo_path)


from utils.losses import FoundationStereoLoss, IgevPlusPlusLoss
from utils.utils import cleanup_ddp, setup_ddp
from utils.arguments import *
import torch.nn as nn

import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import numpy as np
import random

import torch


import torch.distributed as dist
import torch.multiprocessing as mp

from utils.logger import Logger

from utils.train_utils import (create_dataset_manager,
                               evaluate,
                               load_model,
                               create_optimizer)

from torch.amp import GradScaler
autocast = torch.amp.autocast

class DDPTrainer:
    def __init__(self, args: Config, rank: int, world_size: int):
        self.args: Config = args
        self.rank: int = rank
        self.world_size: int = world_size
        
        # === Initialize DDP and wandb ===
        if self.world_size > 1:
            setup_ddp(self.rank, self.world_size)
            assert dist.is_initialized(), f"[rank {self.rank}]: DDP not initialized!"

        self.is_master = not dist.is_initialized() or dist.get_rank() == 0
        
        wandb_mode = "online" if (self.args.logging.use_wandb and self.is_master) else "disabled"
        wandb.init(
            project=self.args.logging.wandb_project,
            group=self.args.group,
            mode=wandb_mode,
            config=vars(self.args),
            name=self.args.logging.wandb_name,
            dir=self.args.logging.wandb_dir,
            entity=self.args.logging.wandb_entity
        )
            
        precision_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        self.dtype = precision_map[args.optimization.precision_dtype]
        
        dst = os.path.join(wandb.run.dir, "args.json")
        if "/tmp/" not in dst:
            OmegaConf.save(vars(self.args), dst)

            underwater_aug_path = self.args.augmentations.underwater_aug_config
            shutil.copyfile(underwater_aug_path,
                            os.path.join(wandb.run.dir, "underwater_config.json"))
    
        device = torch.device(f"cuda:{self.rank}")

        # === Load Model and Resume Checkpoints ===
        logging.info("Loading Model...")
        self.model: nn.Module = load_model(self.args, device, world_size, rank)

        # === Dataset and Dataloader ===
        logging.info(f"[rank {self.rank}]: Model Loaded! Loading Datasets... this may take a while.")
        self.dataset_manager = create_dataset_manager(self.args, device, self.world_size)

        
        logging.info(f"[rank {self.rank}]: Dataset Loaded!")
        
        self.using_stages = len(self.args.stages)
        
        if self.using_stages:
            stages = self.args.stages
            num_its = [stage.num_steps for stage in stages]
            self.stage_transitions = np.cumsum(num_its)
            self.current_stage = None
            self.stage_step() # initialize the stages and create the optimizer
            self.total_train_steps: int = sum(num_its)
        else:
            self.total_train_steps: int = args.optimization.num_steps
            self.setup_optimization() # just create the optimizer

        self.total_steps = 0
        self.epoch = 0
        self.aug_severity = 0.0
        
        self.foundation_stereo_loss = FoundationStereoLoss(
            self.args
        )
        
        self.igevpp_loss = IgevPlusPlusLoss(self.args)

    def _load_setting_stacks(self, stage_config: dict):
        def _flatten_dict(d, parent_keys=()):
            result = []
            for k, v in d.items():
                current_keys = parent_keys + (k,)
                if isinstance(v, dict):
                    result.extend(_flatten_dict(v, current_keys))
                else:
                    result.append((current_keys, v))
            return result

        return _flatten_dict(stage_config)

    def setup_optimization(self):
        # === Optimizer and Scheduler ===
        logging.info(f"[rank {self.rank}]: Creating Optimizer and Scheduler...")
        self.optimizer, self.scheduler = create_optimizer(self.args, self.model)
        logging.info(f"[rank {self.rank}]: Optimizer and Scheduler Fetched!")
        
        if not self.args.logging.use_wandb:
            self.logger = Logger(self.model, self.scheduler, self.args)
        

        self.scaler = GradScaler(enabled=True,
                    init_scale=2**8,
                    growth_factor=2.0,
                    backoff_factor=0.1,
                    growth_interval=100)
        
    def stage_step(self):
        if not self.using_stages:
            return
        
        needs_load = False
        
        if self.current_stage is None:
            needs_load = True
            self.current_stage = 0
        else:
            transition_point = self.stage_transitions[self.current_stage]
            if self.total_steps >= transition_point:
                curr_stage_name = self.args.stages[self.current_stage].stage_name
                self.current_stage += 1
                
                if self.current_stage < len(self.args.stages): # don't bother logging if we're about to exit anyway
                    ep_path = Path(self.args.io.output_dir) / f"stage_{curr_stage_name}.pth"
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "epoch": self.epoch,
                        "step": self.total_steps,
                    }
                
                    torch.save(checkpoint, ep_path)
                needs_load = self.current_stage < len(self.args.stages)
            
            
        if needs_load:
            print("Configuring stage to", self.args.stages[self.current_stage].stage_name)
            stage_config = self.args.stages[self.current_stage].stage_config
            kv_pairs = self._load_setting_stacks(stage_config)

            for key_stack, value in kv_pairs:
                obj = self.args
                for attr in key_stack[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, key_stack[-1], value)
            self.setup_optimization()

        return self.current_stage >= len(self.stage_transitions)

    def update_augmentation_severity(self):
        steps_per_epoch = self.args.optimization.train_steps_per_epoch
        total_epochs = self.args.optimization.num_steps // steps_per_epoch
        
        if self.args.augmentations.aug_schedule == "constant":
            self.aug_severity = 1.0
        elif self.args.augmentations.aug_schedule == "linear":
            self.aug_severity = min(1.0, self.epoch / total_epochs)
        elif self.args.augmentations.aug_schedule == "exponential":
            self.aug_severity = 1 - np.exp(-self.args.augmentations.aug_exponential_rate * self.epoch)
        else:
            raise ValueError(f"Unknown augmentation schedule: {self.args.augmentations.aug_schedule}")
        
        self.dataset_manager.set_aug_severity(self.aug_severity)
        
        if self.is_master:
            logging.info(f"Epoch {self.epoch}: Augmentation severity set to {self.aug_severity:.4f}")
    def run(self):
        
        device = torch.device(f"cuda:{self.rank}")

        self.model.train()

        batch_num = 0
        best_epe = float('inf')
        best_path = None
        keep_training = True

        if dist.is_initialized():
            dist.barrier()
            
        logging.info(f"[rank {self.rank}]: Start Training...")
        while keep_training:

            print(f"[rank {self.rank}]: Starting Epoch", self.epoch)
            
            self.update_augmentation_severity()

            epe_val, warping_val, smoothness_val = evaluate(self.model,
                                        self.epoch,
                                        self.total_steps,
                                        device,
                                        self.dataset_manager,
                                        self.world_size,
                                        self.rank,
                                        self.args)

            if self.is_master:
                print(f"*** Validation EPE at epoch {self.epoch} (step {self.total_steps}): {epe_val:.4f} ***")
                log_dict = {"val_epe": epe_val, "epoch": self.epoch}
                log_dict["val_warping"] = warping_val
                log_dict["val_smoothness"] = smoothness_val
                log_dict["aug_severity"] = self.aug_severity
                
                wandb.log(log_dict, step=self.total_steps)

                ep_path = Path(self.args.io.output_dir) / "latest.pth"
                checkpoint = {
                    "model": self.model.state_dict(),
                    "epoch": self.epoch,
                    "step": self.total_steps,
                    "val_epe": epe_val
                }
                torch.save(checkpoint, ep_path)

                if epe_val < best_epe:
                    best_epe = epe_val
                    best_path = Path(self.args.io.output_dir) / "best.pth"
                    best_checkpoint = {
                        "model": self.model.state_dict(),
                        "epoch": self.epoch,
                        "step": self.total_steps,
                        "val_epe": best_epe
                    }
                    torch.save(best_checkpoint, best_path)
                    wandb.log({
                        "best_model_epe": best_epe,
                        "best_model_epoch": self.epoch,
                        "best_model_step": self.total_steps
                    }, step=self.total_steps)
                    logging.info(f"✨  New best model (EPE={best_epe:.4f}) saved")

            self.epoch += 1
            
            if self.args.max_epochs is not None and self.epoch >= self.args.max_epochs:
                break

            train_iterator = self.dataset_manager.get_train_iterator(self.epoch, self.args.optimization.train_steps_per_epoch)

            for data in tqdm(train_iterator, desc=f"Training (rank {self.rank})", disable=not self.is_master,
                            dynamic_ncols=True):

                dtype=args.get_dtype()
                
                self.optimizer.zero_grad(set_to_none=True)
                img_1 = data["left_uw"].to(device, dtype)
                img_2 = data["right_uw"].to(device, dtype)

                with autocast(device_type='cuda', enabled=self.args.optimization.mixed_precision):
                    if self.args.model == 'foundation_stereo':
                        assert self.model.training
                        init_disp, disp_preds = self.model(
                            image1=img_1,
                            image2=img_2,
                            iters=self.args.train_iters
                        )
                        assert self.model.training

                        loss, metrics = self.foundation_stereo_loss(
                            data,
                            init_disp,
                            disp_preds,
                        )
                    elif self.args.model == "defom_stereo":
                        assert self.model.training
                        disp_preds = self.model(
                            image1=img_1,
                            image2=img_2,
                            iters=self.args.defom_stereo.train_iters,
                            scale_iters=self.args.defom_stereo.scale_iters
                        )
                        loss, metrics = self.foundation_stereo_loss(
                            data,
                            disp_preds[0],
                            disp_preds[1:]
                        )
                    elif self.args.model == "igev_plusplus":
                        assert self.model.training
                        agg_preds, iter_preds = self.model(
                            image1=img_1,
                            image2=img_2,
                            iters=self.args.igev_pp.train_iters,
                            test_mode=False
                        )
                        loss, metrics = self.igevpp_loss(
                            data,
                            agg_preds,
                            iter_preds
                        )  
                    else:
                        raise ValueError(f"Unsupported model: {self.args.model}")

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Some extra sanitizing for good measure.
                has_printed = False
                for p in self.model.parameters():
                    if p.grad is not None:
                        nan_mask = torch.isnan(p.grad)
                        inf_mask = torch.isinf(p.grad)
                        if nan_mask.any() or inf_mask.any():
                            if not has_printed:
                                print(f"[WARN] Gradient anomaly detected at iteration {self.total_steps}: "
                                    f"{nan_mask.sum().item()} NaNs, {inf_mask.sum().item()} Infs. Zeroing out.")
                            has_printed = True
                            p.grad[nan_mask | inf_mask] = 0.0
                            
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if not self.args.logging.use_wandb:
                    self.logger.writer.add_scalar("loss", loss.item(), batch_num)
                    self.logger.writer.add_scalar("learning_rate",
                                            self.optimizer.param_groups[0]["lr"], batch_num)
                    self.logger.push(metrics)

                log_dict = {
                    "total_loss": loss.item(),
                    "stereo_epe": metrics["epe"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.total_steps,
                }
                
                if "warping_loss" in metrics:
                    log_dict["warping_loss"] = metrics["warping_loss"]
                    
                if "smoothness_loss" in metrics:
                    log_dict["smoothness_loss"] = metrics["smoothness_loss"]
                
                # Log bpX filtering metrics if available
                if "mean_bpX" in metrics:
                    log_dict["mean_bpX"] = metrics["mean_bpX"]
                if "max_bpX" in metrics:
                    log_dict["max_bpX"] = metrics["max_bpX"]
                if "min_bpX" in metrics:
                    log_dict["min_bpX"] = metrics["min_bpX"]
                if "num_filtered_by_bpX" in metrics:
                    log_dict["num_filtered_by_bpX"] = metrics["num_filtered_by_bpX"]
                if "bpX_filter_rate" in metrics:
                    log_dict["bpX_filter_rate"] = metrics["bpX_filter_rate"]
                if "num_samples_kept" in metrics:
                    log_dict["num_samples_kept"] = metrics["num_samples_kept"]
                if "fallback_applied" in metrics:
                    log_dict["fallback_applied"] = metrics["fallback_applied"]
                
                if self.is_master:
                    wandb.log(log_dict, step=self.total_steps)
                    
                self.total_steps += 1
                batch_num += 1
                
                all_done = self.stage_step()
                if all_done or self.total_steps > self.total_train_steps:
                    keep_training = False
                    break

            if dist.is_initialized():
                dist.barrier()
                
        print(f"[rank {self.rank}]: Finished Training!")

        save_path = None

        if self.is_master:
            if not self.args.logging.use_wandb:
                self.logger.close()
            save_path = os.path.join(self.args.io.output_dir, f"{self.args.name}.pth")
            try:
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")
            wandb.log({
                "training_completed": True,
                "total_steps": self.total_steps,
                "final_epoch": self.epoch,
                "best_epe": best_epe
            })
        if dist.is_initialized():
            cleanup_ddp()

        return save_path


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    args: Config = get_args()
    
    args.name = args.name
    args.io.output_dir = os.path.join(args.io.output_dir_base, args.name)
    Path(args.io.output_dir).mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config=OmegaConf.create(args), f=f"{args.io.output_dir}/args.yaml")


    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    seed = args.optimization.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        trainer = DDPTrainer(args, rank, world_size)
        trainer.run()
    finally:
        wandb.finish()
