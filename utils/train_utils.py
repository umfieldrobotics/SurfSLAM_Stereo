from FoundationStereo.core.foundation_stereo import FoundationStereo
from data.tbnms_dataset import TBNMSDataset
from data.multi_dataset_manager import MultiDatasetManager
import torch
from tqdm import tqdm
from utils.arguments import Config
from utils.dataset_paths import resolve_checkpoint
from utils.losses import FoundationStereoLoss, IgevPlusPlusLoss
import torch.distributed as dist
from utils.utils import log_validation_samples
from pathlib import Path
from omegaconf import OmegaConf
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.init as init
from data.underwater_dataset_manager import UnderwaterDatasetManager, DatasetType
from data.multi_dataset_manager import MultiDatasetManager
import math

autocast = torch.amp.autocast
from defom_core.defom_stereo import DEFOMStereo

from igev_core.igev_stereo import IGEVStereo

def load_state_dict_strict_except(model, state_dict,
                                  exclude_prefixes=(),
                                  initialize_prefixes=(),
                                  unexpected_ignore_prefixes=()):
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False)
    rank = dist.get_rank() if dist.is_initialized() else 0

    for key in missing_keys:
        if not any(
                key.startswith(prefix)
                for prefix in (*initialize_prefixes, *exclude_prefixes)):
            msg = "Missing expected keys in state_dict:\n  " + "\n  ".join(
                missing_keys)
            raise RuntimeError(msg)

        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            print(f"No loaded value for {key}. Using the default.")
            continue

        log_str = f"No loaded value for key {key}. Initializing using "

        module_name, param_name = key.rsplit('.', 1)
        module = model
        for attr in module_name.split('.'):
            module = getattr(module, attr)
        tensor = getattr(module, param_name)

        if 'weight' in param_name:
            if tensor.ndim >= 2:
                init.kaiming_normal_(
                    tensor, mode='fan_out', nonlinearity='relu')
                log_str += "kaiming_normal_."
            else:
                init.normal_(tensor, mean=0, std=1)
                log_str += "normal_"
        elif 'bias' in param_name:
            init.zeros_(tensor)
            log_str += "Zeros"
        elif 'running_mean' in param_name:
            tensor.zero_()
            log_str += "Zeros"
        elif 'running_var' in param_name:
            tensor.fill_(1)
            log_str += "Ones"
        else:
            init.zeros_(tensor)
            log_str += "Zeros"

        if rank == 0:
            print(log_str)

    if unexpected_keys:
        actually_unexpected_keys = [k for k in unexpected_keys if not any(
            k.startswith(prefix) for prefix in unexpected_ignore_prefixes)]
        if len(actually_unexpected_keys) > 0:
            msg = "Unexpected keys in state_dict:\n  " + \
                "\n  ".join(actually_unexpected_keys)
            raise RuntimeError(msg)

@torch.no_grad()
def evaluate(the_model,
             epoch,
             step,
             device,
             dataset_manager: MultiDatasetManager,
             world_size,
             rank,
             args: Config):

    print(f"[rank {rank}]: Evaluating after {epoch} Epochs")
    the_model.eval()
    log_loader = dataset_manager.get_log_iterator()
    tbnms_loader = dataset_manager.get_tbnms_iterator()
    
    seed = args.optimization.seed
    precision_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = precision_map[args.optimization.precision_dtype]

    the_model.eval()
    epe_sum = torch.tensor(0.0, device=device)
    n_samples = torch.tensor(0.0, device=device)
    is_master = rank == 0
    warping_loss_sum = torch.tensor(0.0, device=device)
    smoothness_loss_sum = torch.tensor(0.0, device=device)

    fs_loss_obj = FoundationStereoLoss(args)
    igevpp_loss_obj = IgevPlusPlusLoss(args)
    
    with autocast(device_type='cuda', enabled=args.optimization.mixed_precision):

        val_iterator = dataset_manager.get_val_iterator()
        for batch in tqdm(val_iterator,
                            desc=f"Validation (rank {rank})",
                            disable=not is_master,
                            dynamic_ncols=True):

            # TODO: clean up dataloader to output these to left and right
            img_l = batch["left_uw"].to(device, dtype)
            img_r = batch["right_uw"].to(device, dtype)
                
            assert not img_l.isnan().any() and not img_r.isnan().any()
            if args.model == 'foundation_stereo':
                init_disp, disp_preds = the_model(image1=img_l,
                                                 image2=img_r,
                                                 iters=args.train_iters)

                _, metrics = fs_loss_obj(batch, 
                                         init_disp,
                                         disp_preds)
            elif args.model == "defom_stereo":
                disp_preds = the_model(
                    image1=img_l,
                    image2=img_r,
                    iters=args.defom_stereo.valid_iters,
                    scale_iters=args.defom_stereo.scale_iters
                )
                _, metrics = fs_loss_obj(batch, disp_preds[0], disp_preds[1:])
            elif args.model == "igev_plusplus":
                agg_disp, disp_preds = the_model(
                    image1=img_l,
                    image2=img_r,
                    iters=args.igev_pp.valid_iters,
                )
                _, metrics = igevpp_loss_obj(batch, agg_disp, disp_preds)
            else:
                raise ValueError(f"Unsupported model: {args.model}")
            bs = img_l.size(0)
            if math.isnan(metrics["epe"]) or math.isinf(metrics["epe"]):
                epe_sum += 0.0
            else:
                epe_sum += metrics["epe"] * bs
            if "warping_loss" in metrics:
                warping_loss_sum += metrics["warping_loss"] * bs
            
            if "smoothness_loss" in metrics:
                smoothness_loss_sum += metrics["smoothness_loss"] * bs
            
            n_samples += bs
                
        if log_loader is not None:
            log_validation_samples(log_loader, the_model, step, seed,
                                    device, dtype, rank, is_master, args,
                                    False)
            
        
        if tbnms_loader is not None:
            print(f"[rank {rank}]: Logging TBNMS samples")
            log_validation_samples(tbnms_loader, the_model, step, seed,
                                    device, dtype, rank, is_master, args,
                                    True)
            
    if world_size > 1:
        dist.all_reduce(epe_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
        
        
        dist.all_reduce(warping_loss_sum, op=dist.ReduceOp.SUM)
        warping_loss_sum /= n_samples
        
        dist.all_reduce(smoothness_loss_sum, op=dist.ReduceOp.SUM)
        smoothness_loss_sum /= n_samples
        

    
    the_model.train()
    print(f"[rank {rank}]: Done Evaluating")
    return (epe_sum / n_samples).item(), warping_loss_sum.item(), smoothness_loss_sum.item()

    
def load_model(args: Config,
               device,
               world_size,
               rank):
    # Every model here starts from a checkpoint. A null io.restore_checkpoints
    # means the ablation asked for a training stage this model never ran, so say
    # so instead of falling through with an unbuilt model.
    if args.io.restore_checkpoints is None:
        raise ValueError(
            f"io.restore_checkpoints is null, so there is nothing to load for "
            f"model '{args.model}'. The ablation names a stage this model does "
            f"not have a checkpoint for; declare it under `checkpoints:` in "
            f"config/train/models/<model>/<model>.yaml, or set io.restore_checkpoints "
            f"explicitly for this model/ablation pair.")

    # Registry paths are relative to the model_weights root.
    args.io.restore_checkpoints = resolve_checkpoint(
        args.io.restore_checkpoints, args.io.model_weights_dir)
    args.defom_stereo.depth_anything_checkpoint = resolve_checkpoint(
        args.defom_stereo.depth_anything_checkpoint, args.io.model_weights_dir)
    args.foundation_stereo.depth_anything_checkpoint = resolve_checkpoint(
        args.foundation_stereo.depth_anything_checkpoint, args.io.model_weights_dir)

    if args.model == 'foundation_stereo':
        args.foundation_stereo.mixed_precision = args.optimization.mixed_precision

        # FoundationStereo
        if args.io.restore_checkpoints is not None:
            assert args.io.restore_checkpoints.endswith(".pth")
            cfg_path = Path(args.io.restore_checkpoints).parent / "cfg.yaml"
            cfg = OmegaConf.load(cfg_path)

            if "vit_size" not in cfg:
                cfg["vit_size"] = "vitl"

            cfg = OmegaConf.create(cfg)

            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(levelname)s %(message)s")
            logging.info("Loading model from %s", args.io.restore_checkpoints)

            model = FoundationStereo(cfg)

            ckpt = torch.load(args.io.restore_checkpoints, map_location="cpu", weights_only=False)
            if "model" in ckpt:
                state_dict = ckpt.get("state_dict", ckpt)["model"]
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            else:
                state_dict = ckpt

            
            # Checkpoints from the attenuation experiments carry attenuation_branch.*
            # keys the architecture no longer has. Note the trailing comma: without
            # it this is a bare string, and iterating a string yields characters, so
            # every unexpected key starting with a/t/e/n/u/i/o/_/b/r/c/h/. would be
            # silently ignored -- which is most of them.
            load_state_dict_strict_except(model, state_dict,
                                          unexpected_ignore_prefixes=("attenuation_branch.",))

            if args.foundation_stereo.depth_anything_checkpoint is not None:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Loading Depth Anything checkpoint!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                da_cpkt = torch.load(args.foundation_stereo.depth_anything_checkpoint,
                                     map_location='cpu', weights_only=False)
                N = len("module.")
                da_cpkt = {k[N:]:v for k,v in da_cpkt["model"].items()}
                model.feature.dino.depth_anything.load_state_dict(da_cpkt)
                
            model.to(device=device)
    elif args.model == 'defom_stereo':

        if args.io.restore_checkpoints is not None:
            assert args.io.restore_checkpoints.endswith(".pth")
            logging.info(f"[rank {rank}]: Loading checkpoint...")
            
            ckpt = torch.load(args.io.restore_checkpoints, map_location="cpu")
            model = DEFOMStereo(args.defom_stereo)
            if 'model' in ckpt:
                ckpt = ckpt['model']
                to_load = {}
                for k,v in ckpt.items():
                    if k.startswith("module."):
                        to_load[k[len("module."):]] = v
                    else:
                        to_load[k] = v
                model.load_state_dict(to_load)
            else:
                model.load_state_dict(ckpt)
            logging.info(f"[rank {rank}]: Done loading checkpoint")
            
            if args.defom_stereo.depth_anything_checkpoint is not None:
                print("<Actually> Loading Depth Anything checkpoint!!")
                da_cpkt = torch.load(args.defom_stereo.depth_anything_checkpoint,
                                     map_location='cpu', weights_only=False)
                N = len("module.")
                da_cpkt = {k[N:]:v for k,v in da_cpkt["model"].items()}
                model.defomencoder.depth_anything.load_state_dict(da_cpkt, strict=False)
            
                
            model.to(device=device)
        else:
            raise ValueError("DeFOM-Stereo requires a checkpoint to be specified.")

    elif args.model == "igev_plusplus":
        if args.io.restore_checkpoints is not None:
            assert args.io.restore_checkpoints.endswith(".pth")
            logging.info(f"[rank {rank}]: Loading checkpoint...")
            ckpt = torch.load(args.io.restore_checkpoints, map_location="cpu")
            model = IGEVStereo(args.igev_pp)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            to_load = {}
            for k,v in ckpt.items():
                if k.startswith("module."):
                    to_load[k[len("module."):]] = v
                else:
                    to_load[k] = v
            model.load_state_dict(to_load)
            logging.info(f"[rank {rank}]: Done loading checkpoint")
            model = model.to(device=device)
            ""

    elif args.model == "underwater_stereo":
        # BGNet from the Underwater_Stereo baseline. Evaluation only -- there is no
        # training path for it here, and it takes no architecture config: the
        # network is fixed, including its 192 px disparity ceiling (see
        # demo/core.py:forward_disparity).
        #
        # Imported here rather than at module scope so training does not depend on a
        # baseline submodule being checked out.
        #
        # Imported by its full dotted path, via namespace packages, because
        # Nets/submodules2d.py does `from ..Utils.warp import disp_warp` -- it reaches
        # above Nets, so Nets has to be a subpackage of Underwater_Stereo. Putting
        # models/Underwater_Stereo on sys.path and importing `Nets` directly fails
        # with "attempted relative import beyond top-level package".
        try:
            from models.Underwater_Stereo.Nets.bgnet import BGNet
        except ImportError as err:
            raise ImportError(
                f"could not import the Underwater_Stereo baseline ({err}). Check out the "
                f"submodule (`git submodule update --init models/Underwater_Stereo`) and "
                f"run from the repo root, which must be on sys.path.") from err

        if args.io.restore_checkpoints is None:
            raise ValueError("Underwater_Stereo requires a checkpoint to be specified.")

        logging.info(f"[rank {rank}]: Loading checkpoint...")
        model = BGNet()
        ckpt = torch.load(args.io.restore_checkpoints, map_location="cpu",
                          weights_only=False)
        # The released .ckpt wraps a DDP state dict under "model"; the KITTI .pth
        # that ships in the submodule is a bare, unprefixed one.
        state_dict = ckpt.get("state_dict", ckpt)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        logging.info(f"[rank {rank}]: Done loading checkpoint")
        model = model.to(device=device)

    else:
        raise RuntimeError("Unsupported model you dofus")

    if world_size > 1:
        model = DDP(model,
                    device_ids=[rank],      # length-1 list
                    output_device=rank,
                    broadcast_buffers=False,      # usually faster
                    find_unused_parameters=True, # turn back on only if needed
                    static_graph=False)
    return model
        
def create_dataset_manager(args: Config,
                           device,
                           world_size):
    seed = args.optimization.seed
    
    datasets = []
    
    if "tartanair" in args.optimization.train_datasets:
        tartanair_dataset_manager = UnderwaterDatasetManager.create(
            args.io.tartanair_dir,
            DatasetType.TARTAN_AIR,
            device,
            args
        )
        datasets.append(tartanair_dataset_manager)

    if "oceansim" in args.optimization.train_datasets:
        oceansim_dataset_manager = UnderwaterDatasetManager.create(
            args.io.uwsim_dir,
            DatasetType.OCEAN_SIM,
            device,
            args
        )
        datasets.append(oceansim_dataset_manager)

    if "tbnms" in args.optimization.train_datasets:
        tbnms_train_manager = UnderwaterDatasetManager.create(
            args.io.suds_stereo_dir,
            DatasetType.TBNMS,
            device,
            args
        )
        datasets.append(tbnms_train_manager)
        
    if "svin2" in args.optimization.train_datasets:
        svin2_train_manager = UnderwaterDatasetManager.create(
            args.io.svin2_dir,
            DatasetType.SVIN2,
            device,
            args
        )
        datasets.append(svin2_train_manager)

    if "lizard_island" in args.optimization.train_datasets:
        lizard_island_train_manager = UnderwaterDatasetManager.create(
            args.io.lizard_island_dir,
            DatasetType.LIZARD_ISLAND,
            device,
            args
        )
        datasets.append(lizard_island_train_manager)

    if "flyingthings" in args.optimization.train_datasets:
        middlebury_train_manager = UnderwaterDatasetManager.create(
            args.io.flyingthings_dir,
            DatasetType.FLYING_THINGS,
            device,
            args
        )
        datasets.append(middlebury_train_manager)

    # Qualitative logging on the real-world SUDS scenes is opt-in: an empty
    # logging.tbnms_samples.sequences means a sim-only run, which shouldn't need
    # the SUDS release registered at all.
    tbnms_samples = args.logging.tbnms_samples
    if tbnms_samples is not None and getattr(tbnms_samples, "sequences", None):
        tbnms_test_dataset = TBNMSDataset(tbnms_samples,
                                          device=device,
                                          root=args.io.suds_stereo_dir)
    else:
        tbnms_test_dataset = None

    dataset_manager = MultiDatasetManager(datasets,
                                          seed=seed,
                                          tbnms_dataset=tbnms_test_dataset,)
    dataset_manager.create_loaders(
        world_size,
        args.optimization.batch_size_per_gpu,
        args.optimization.num_workers,
        args.logging.num_log_imgs
    )
    return dataset_manager
    
def create_optimizer(args: Config, model):
    # === Optimizer ===
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.optimization.learning_rate,
                                  weight_decay=args.optimization.wdecay, eps=1e-8)
    
    if args.optimization.lr_schedule_type == "foundation_stereo":
        def lr_lambda(current_step):
            total_steps = args.optimization.num_steps
            decay_at = 0.8
            decay_factor = 0.1
            return 1.0 if current_step < int(total_steps * decay_at) else decay_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.optimization.lr_schedule_type == "foundation_stereo_ramp":
        def lr_lambda(current_step):
            ramp_up_steps = 1000
            if current_step < ramp_up_steps:
                return float(current_step) / ramp_up_steps
            total_steps = args.optimization.num_steps
            decay_at = 0.8
            decay_factor = 0.1
            return 1.0 if current_step < int(total_steps * decay_at) else decay_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.optimization.lr_schedule_type == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        args.optimization.learning_rate,
                                                        args.optimization.num_steps + 100,
                                                        pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    else:
        raise ValueError("Uknown LR Schedule Type", args.optimization.lr_schedule_type)
    return optimizer, scheduler


