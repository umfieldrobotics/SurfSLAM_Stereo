"""TensorBoard logging for training runs that aren't using wandb.

This is the RAFT-Stereo/FoundationStereo training logger. It used to live in
``models/FoundationStereo/Utils.py``, appended to the upstream file, which meant
train.py did ``from FoundationStereo.Utils import *`` purely to reach this one class
-- dragging open3d, pandas and transformations into the import path of every training
run along the way. It is our code, so it lives here now and the submodule stays
pristine. See docs/submodules.md.

Only constructed when ``logging.use_wandb`` is false; wandb runs log through their own
path in train.py.
"""

import logging

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Accumulates scalar metrics and flushes a running mean every SUM_FREQ steps."""

    SUM_FREQ = 100

    def __init__(self, model, scheduler, args):
        self.args = args
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        # args.io.output_dir, not args.output_dir: Config has no top-level
        # output_dir, so the original raised AttributeError the moment anyone ran
        # with use_wandb=false. It is set programmatically before training starts.
        self.writer = SummaryWriter(log_dir=self.args.io.output_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ
                        for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        logging.info("Training Metrics (%d): %s", self.total_steps,
                     training_str + metrics_str)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ,
                                   self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
