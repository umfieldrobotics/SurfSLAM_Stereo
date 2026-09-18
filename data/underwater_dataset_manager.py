from functools import partial
from typing import Union
import torch

from data.ocean_sim_dataset import OceanSimDataset
from data.tartanair_dataset import TartanAirDataset
from data.water_augmentations.uw_augmentations_transform import UWAugmentationsTransform
from data.stereo_transform import StereoTransform
from data.flyingthings_dataset import FlyingThings3DDataset

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from enum import Enum
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import random
import json
from data.lizard_island_dataset import LizardIslandDataset
from data.svin2_dataset import SVIN2Dataset
from data.tbnms_dataset import TBNMSDataset
from utils.arguments import Config
from utils.progress import main_rank_print
from utils.utils import OSSafeIterator
import copy

from data.water_augmentations.uw_augmentations import WaterAugAblationSettings, resolve_repo_path
    
class DatasetType(Enum):
    TARTAN_AIR = "TartanAir"
    OCEAN_SIM = "OceanSim"
    TBNMS = "TBNMS"
    LIZARD_ISLAND = "LizardIsland"
    SVIN2 = "SVIN2"
    FLYING_THINGS = "FlyingThings3D"


def worker_init_fn(base_seed, worker_id):
    rank = dist.get_rank() if dist.is_initialized() else 0
    np.random.seed(base_seed + rank * 1000 + worker_id)
    random.seed(base_seed + rank * 1000 + worker_id)
    torch.manual_seed(base_seed + rank * 1000 + worker_id)
    torch.cuda.manual_seed_all(base_seed + rank * 1000 + worker_id)


class EpochIterator:
    def __init__(self,
                 uw_manager: "UnderwaterDatasetManager",
                 raw_loader: DataLoader,
                 num_steps: int):

        self._uw_manager = uw_manager
        self._raw_loader = raw_loader

        self._raw_iterator = OSSafeIterator(raw_loader)
        self._count = 0
        self._num_steps = num_steps

    def __iter__(self):
        return self

    def __len__(self):
        return self._num_steps

    def __next__(self):
        if self._count >= self._num_steps:
            raise StopIteration

        self._count += 1
        try:
            return next(self._raw_iterator)
        except StopIteration:
            self._raw_iterator = OSSafeIterator(self._raw_loader)
            return next(self._raw_iterator)


class UnderwaterDatasetManager:
    def __init__(self,
                 base_dataset: Union[TartanAirDataset, OceanSimDataset],
                 underwater_aug_config,
                 water_aug_settings: WaterAugAblationSettings,
                 out_resolution=(320, 736),
                 prep_device: torch.device = torch.device("cuda"),
                 num_val_frames_per_dataset: int = 1000,
                 seed: int = 42,
                 test_dataset=None,
                 val_dataset=None,
                 uw_aug_prob: float = 0.5):
        
        underwater_aug_config = resolve_repo_path(underwater_aug_config)
        with open(underwater_aug_config, 'r') as f:
            aug_settings = json.load(f)
            
        max_depth = aug_settings.get("max_depth", None)
        train_geom = StereoTransform(
            out_resolution,
            is_train=True,
            max_depth=max_depth,
        )
        
        if underwater_aug_config and water_aug_settings:
            uw_transform = UWAugmentationsTransform(underwater_aug_config,
                            aug_settings=water_aug_settings,
                            prob=uw_aug_prob,
                            device=prep_device)
            self._uw_transform = uw_transform
            self._train_transform = transforms.Compose([uw_transform, train_geom])
        else:
            self._uw_transform = None
            self._train_transform = train_geom
        
        self._val_transform = self._train_transform

            
        self.device = prep_device
        self.seed = seed
        self.full_dataset = base_dataset
        if val_dataset is not None:
            # The dataset defines its own validation split (e.g. SUDS ships one),
            # so don't carve one out of train.
            self.train_ds = self.full_dataset
            self.val_ds = val_dataset
            self.train_len = len(self.train_ds)
            self.val_len = len(self.val_ds)

            self.train_ds.transform = self._train_transform
            self.val_ds.transform = self._val_transform
        else:
            self.val_len = num_val_frames_per_dataset
            self.train_len = len(self.full_dataset) - self.val_len

            g = torch.Generator().manual_seed(seed)
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                self.full_dataset, [self.train_len, self.val_len], generator=g
            )

            self.train_ds.dataset.transform = self._train_transform
            self.val_ds.dataset.transform = self._val_transform

        # optional external test dataset
        if test_dataset is not None:
            self.test_ds = copy.deepcopy(torch.utils.data.Subset(test_dataset, range(len(test_dataset))))
            self.test_ds.dataset.transform = self._val_transform
        else:
            self.test_ds = None

        # placeholders for loaders / samplers
        self.train_sampler = None
        self.val_sampler = None
        self.tbnms_sampler = None
        

    def create_loaders(self, world_size, batch_per_gpu, num_workers, num_log_imgs=32):
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.train_sampler = DistributedSampler(self.train_ds,
                                                num_replicas=world_size,
                                                rank=rank,
                                                shuffle=True)

        self.val_sampler = DistributedSampler(self.val_ds,
                                              num_replicas=world_size,
                                              rank=rank,
                                              shuffle=False)
        
        self.test_sampler = DistributedSampler(self.test_ds,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=False)
        


        init_fn = partial(worker_init_fn, self.seed)
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_per_gpu,
                                       sampler=self.train_sampler, num_workers=num_workers,
                                       pin_memory=False, worker_init_fn=init_fn,
                                       persistent_workers=num_workers > 0
                                       )
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_per_gpu,
                                     sampler=self.val_sampler, num_workers=num_workers,
                                     pin_memory=False, worker_init_fn=init_fn,
                                     persistent_workers=num_workers > 0
                                     )
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_per_gpu,
                                sampler=self.test_sampler, num_workers=num_workers,
                                pin_memory=False, worker_init_fn=init_fn,
                                )

        
        K_LOG = num_log_imgs
        rng = np.random.default_rng(self.seed)
        log_ids = rng.choice(len(self.test_ds), K_LOG, replace=False)
        self.test_log_loader = DataLoader(
            torch.utils.data.Subset(self.test_ds, log_ids),
            batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
        )

    def set_aug_severity(self, severity: float):
        if self._uw_transform is not None:
            self._uw_transform.set_aug_severity(severity)

    @staticmethod
    def create(data_dir, data_type: DatasetType, device, args: Config) -> "UnderwaterDatasetManager":
        main_rank_print(f"[data] Loading {data_type.value}...")

        water_aug_settings = WaterAugAblationSettings(
            args.rendering.enable_caustics,
            args.rendering.enable_water_column,
            args.rendering.enable_directional_light,
            args.rendering.enable_particles,
            args.rendering.enable_halo
        )
        
        no_aug=False
        val_ds = None  # set only by datasets that ship their own validation split
        if data_type == DatasetType.OCEAN_SIM:
            test_sequences = []
            for seq in args.test_sequences:
                dataset, sequence = seq.split("/")
                if dataset == "oceansim":
                    test_sequences.append(sequence)
                        
            train_ds = OceanSimDataset(data_dir,
                                       device=device,
                                       test_sequences=test_sequences,
                                       test_mode=False,
                                       transform=None)     
                   
            test_ds = OceanSimDataset(data_dir,
                                      device=device,
                                      test_sequences=test_sequences,
                                      test_mode=True,
                                      transform=None)

        elif data_type == DatasetType.TARTAN_AIR:
            test_sequences = []
            for seq in args.test_sequences:
                dataset, sequence = seq.split("/")
                if dataset == "tartanair":
                    test_sequences.append(sequence)
            
            train_ds = TartanAirDataset(data_dir, device, None, test_sequences=test_sequences, test_mode=False)            
            test_ds = TartanAirDataset(data_dir, device, None, test_sequences=test_sequences, test_mode=True)
        elif data_type == DatasetType.TBNMS:
            # The SUDS release ships its own train/val/test split files, which are
            # authoritative. Any leftover `tbnms/<scene>` entries in test_sequences
            # are ignored -- warn so stale configs are obvious.
            stale = [s.split("/")[1] for s in args.test_sequences if s.split("/")[0] == "tbnms"]
            if stale:
                print("[SUDS] ignoring `tbnms/*` entries in test_sequences; the release's "
                      "splits/{train,val,test}.txt define the split. See docs/data.md. Ignored:",
                      *sorted(stale))

            train_ds = TBNMSDataset(device=device, split="train", root=data_dir)
            # Cap val to the same budget every other dataset gets -- the release's
            # val split is ~5.7k frames and carries no ground truth, so evaluating
            # all of it would dominate validation time for nothing.
            val_ds = TBNMSDataset(device=device, split="val", root=data_dir,
                                  max_frames=args.num_val_frames_per_dataset,
                                  seed=args.optimization.seed)
            test_ds = TBNMSDataset(device=device, split="test", root=data_dir)
            no_aug=True
        elif data_type == DatasetType.SVIN2:
            test_sequences = []
            for seq in args.test_sequences:
                dataset, sequence = seq.split("/")
                if dataset == "svin2":
                    test_sequences.append(sequence)
            
            train_ds = SVIN2Dataset(device=device, test_sequences=test_sequences, test_mode=False, root=data_dir)
            if len(test_sequences) > 0:
                test_ds = SVIN2Dataset(device=device, test_sequences=test_sequences, test_mode=True, root=data_dir)
            else:
                test_ds = None
            no_aug = True
        elif data_type == DatasetType.LIZARD_ISLAND:
            train_ds = LizardIslandDataset(device=device, test_mode=False, root=data_dir)
            test_ds = None
            no_aug = True
        elif data_type == DatasetType.FLYING_THINGS:
            train_ds = FlyingThings3DDataset(root=data_dir, test_mode=False, device=device)
            test_ds = FlyingThings3DDataset(root=data_dir, test_mode=True, device=device)
            no_aug = True
        else:    
            raise RuntimeError("Unknown data_type:", data_type)
    
        return UnderwaterDatasetManager(
            train_ds,
            underwater_aug_config=args.augmentations.underwater_aug_config,
            water_aug_settings=None if no_aug else water_aug_settings,
            out_resolution=(args.rendering.output_height, args.rendering.output_width),
            prep_device=torch.device('cuda'),
            num_val_frames_per_dataset=args.num_val_frames_per_dataset,
            seed=args.optimization.seed,
            test_dataset=test_ds,
            val_dataset=val_ds,
            uw_aug_prob=args.augmentations.uw_aug_probability)


    @property
    def full_dataset(self) -> Dataset:
        return self._full_dataset

    @full_dataset.setter
    def full_dataset(self, value: Dataset):
        self._full_dataset = value

    @property
    def train_dataset(self) -> Dataset:
        return self.train_ds

    @train_dataset.setter
    def train_dataset(self, value: Dataset):
        self.train_ds = value

    @property
    def val_dataset(self) -> Dataset:
        return self.val_ds

    @val_dataset.setter
    def val_dataset(self, value: Dataset):
        self.val_ds = value

    def _get_epoch_iterator(self, base_loader, num_steps=None):
        if num_steps == None:
            return base_loader
        return EpochIterator(self, base_loader, num_steps)

    def get_train_iterator(self, epoch, num_steps_per_epoch=None):
        self.train_sampler.set_epoch(epoch)
        return self._get_epoch_iterator(self.train_loader, num_steps_per_epoch)

    def get_val_iterator(self):
        return self.val_loader

    def get_log_iterator(self):
        return self.test_log_loader
