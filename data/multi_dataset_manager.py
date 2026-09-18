from functools import partial
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, DistributedSampler
import torch.distributed as dist
import numpy as np
import random
from data.underwater_dataset_manager import UnderwaterDatasetManager
from data.tbnms_dataset import TBNMSDataset
from utils.utils import OSSafeIterator


class BalancedConcatDataset(Dataset):
    """
    Oversamples the smaller datasets so that each underlying dataset is seen
    equally often within an epoch.
    """

    def __init__(self, datasets: List[Dataset]) -> None:
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.max_len = max(self.lengths)
        self.total_len = self.max_len * len(self.datasets)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        ds_idx = idx % len(self.datasets)
        sample_idx = (idx // len(self.datasets)) % self.lengths[ds_idx]
        return self.datasets[ds_idx][sample_idx]


def _seed_worker(base_seed: int, worker_id: int) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = base_seed + rank * 1000 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiDatasetManager:
    """
    Wraps several UnderwaterDatasetManager objects and presents them
    as one balanced dataset.
    """

    def __init__(self,
                 managers: List[UnderwaterDatasetManager],
                 seed: int = 42,
                 tbnms_dataset: TBNMSDataset = None) -> None:
        self.managers = managers
        self.seed = seed

        # Compose train/val/test datasets.
        self.train_ds = BalancedConcatDataset(
            [m.train_dataset for m in managers])
        self.val_ds = ConcatDataset([m.val_dataset for m in managers])
        self.test_ds = ConcatDataset(
            [m.test_ds for m in managers if m.test_ds is not None])

        # These are filled by `create_loaders`
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.tbnms_dataset = tbnms_dataset
        self.tbnms_sampler = None
        self.tbnms_loader = None

    # --------------------------------------------------------------------- #
    # Dataloader creation                                                   #
    # --------------------------------------------------------------------- #
    def create_loaders(
        self,
        world_size: int,
        batch_per_gpu: int,
        num_workers: int,
        num_log_imgs: int = 32,
    ) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0

        self.train_sampler = DistributedSampler(
            self.train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        self.val_sampler = DistributedSampler(
            self.val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        self.test_sampler = DistributedSampler(
            self.test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )

        init_fn = partial(_seed_worker, self.seed)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_per_gpu,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=init_fn,
            persistent_workers=num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_per_gpu,
            sampler=self.val_sampler,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=init_fn,
            persistent_workers=num_workers > 0,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_per_gpu,
            sampler=self.test_sampler,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=init_fn,
        )

        if self.has_tbnms_dataset:
            self.tbnms_sampler = DistributedSampler(self.tbnms_dataset,
                                                       num_replicas=world_size,
                                                       shuffle=False,
                                                       rank=rank)
            self.tbnms_loader = DataLoader(
                self.tbnms_dataset, batch_size=1,
                sampler=self.tbnms_sampler, num_workers=num_workers,
                pin_memory=False, worker_init_fn=init_fn,)

        if num_log_imgs > 0:
            per_ds = num_log_imgs // len(self.managers)
            remainder = num_log_imgs % len(self.managers)

            subsets = []
            rng = np.random.default_rng(self.seed)

            for i, m in enumerate(self.managers):
                n_select = per_ds + (1 if i < remainder else 0)
                val_len = len(m.val_dataset)
                if val_len == 0:
                    continue
                idxs = rng.choice(val_len, min(n_select, val_len), replace=False)
                subsets.append(torch.utils.data.Subset(m.val_dataset, idxs))

            if subsets:
                combined_subset = ConcatDataset(subsets)
                self.val_log_loader = DataLoader(
                    combined_subset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
            else:
                self.val_log_loader = None
        else:
            self.val_log_loader = None
    @property
    def has_tbnms_dataset(self):
        return self.tbnms_dataset is not None

    def set_aug_severity(self, severity: float):
        """Update the augmentation severity for all dataset managers.
        
        Args:
            severity: Float between 0.0 and 1.0 controlling augmentation intensity
        """
        for manager in self.managers:
            manager.set_aug_severity(severity)

    def get_train_iterator(
            self, epoch: int, num_steps_per_epoch: int | None = None):
        self.train_sampler.set_epoch(epoch)
        return self._bounded_iter(self.train_loader, num_steps_per_epoch)

    def get_val_iterator(self):
        return OSSafeIterator(self.val_loader)

    def get_log_iterator(self):
        # May return None on non-master ranks; caller should handle that.
        return OSSafeIterator(self.val_log_loader)

    def get_tbnms_iterator(self):
        # None when qualitative SUDS logging is disabled; caller must handle that.
        if self.tbnms_loader is None:
            return None
        return OSSafeIterator(self.tbnms_loader)
    
    class _BoundedIter:
        def __init__(self, loader: DataLoader, steps: int | None):
            self.loader = loader
            self.steps = steps
            self.iter = OSSafeIterator(loader)
            self.count = 0

        def __iter__(self):
            return self

        def __len__(self):
            return self.steps

        def __next__(self):
            if self.steps is not None and self.count >= self.steps:
                raise StopIteration
            self.count += 1
            try:
                return next(self.iter)
            except StopIteration:
                self.iter = OSSafeIterator(self.loader)
                return next(self.iter)

    def _bounded_iter(self, loader: DataLoader, steps: int | None):
        return loader if steps is None else self._BoundedIter(loader, steps)
