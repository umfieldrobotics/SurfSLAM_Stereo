from dataclasses import dataclass, field
from typing import Dict, List, Optional
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError
import sys
import os
import torch

@dataclass
class AugmentationConfig:
    aug_crop: bool = False
    aug_color: bool = False
    aug_flip: bool = False
    uw_aug_probability: float = 1.0
    underwater_aug_config: str = "../data/water_augmentations/cfg/water_aug_config.json"
    aug_schedule: str = "linear" # Options: constant, linear, exponential
    aug_exponential_rate: float = 0.5

@dataclass
class OptimizationConfig:
    batch_size_per_gpu: int = 4
    learning_rate: float = 1.0e-5
    num_steps: int = 10000
    max_epochs: Optional[int] = None
    wdecay: float = 0.00001
    mixed_precision: bool = True
    precision_dtype: str = "float16"
    train_steps_per_epoch: Optional[int] = None
    time_limit: int = 14400
    num_workers: int = 0
    seed: int = 42
    lambda_warping: float = 1.0
    lambda_smoothness: float = 0.1
    lambda_occam: float = 1.0
    occam_margin: float = 0.01

    freeze_disparity: bool = False
    freeze_foreground: bool = False
    lr_schedule_type: str = "one_cycle"
    
    enable_bpX_filtering: bool = True
    bpX_threshold: float = 0.5
    bpX_criteria: float = 2.0
    bpX_min_samples: int = 1 
    bpX_fallback_strategy: str = "relax"
    # Options: oceansim, tartanair, tbnms
    train_datasets: List[str] = field(default_factory=lambda: ["oceansim", "tartanair"])


@dataclass
class FoundationStereoConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128] * 3)
    # Refinement iterations at test time. 32 is what the released checkpoint's own
    # cfg.yaml asks for and what evaluation has always used; training uses the
    # smaller top-level `train_iters`.
    valid_iters: int = 32
    max_disp: int = 416
    corr_levels: int = 2
    corr_radius: int = 4
    low_memory: bool = False
    n_downsample: int = 2
    n_gru_layers: int = 3
    depth_anything_checkpoint: Optional[str] = None

    ## Placeholders that will be set by code. don't set manually.
    mixed_precision: bool = False


@dataclass
class IgevPPConfig:
    train_iters: int = 22
    valid_iters: int = 32

    corr_levels: int = 2
    corr_radius: int = 4
    n_downsample: int = 2
    n_gru_layers: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [128] * 3)
    
    max_disp: int = 768
    # loss param
    max_disp1: int = 384
    max_disp0: int = 192
    loss_gamma: float = 0.9
    
    s_disp_range: int = 48
    m_disp_range: int = 96
    l_disp_range: int = 192
    s_disp_interval: int = 1
    m_disp_interval: int = 2
    l_disp_interval: int = 4

    mixed_precision: bool = True
    precision_dtype: str = "float16"

@dataclass
class HardwareSamples:
    n_samples_per_sequence: int = 5
    sequences: List[str] = field(default_factory=lambda: ["baycity_candidate2"])

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_name: Optional[str] = None
    wandb_project: str = "SurfSLAM_Stereo"
    wandb_dir: Optional[str] = None
    num_log_imgs: int = 32

    # Whose wandb the runs land in. Normally comes from config/wandb.yaml (see
    # config/wandb.example.yaml) rather than a tracked config; null logs to the
    # default entity of whoever `wandb login` ran as.
    wandb_entity: Optional[str] = None
    tbnms_samples: HardwareSamples = field(default_factory=lambda: HardwareSamples)


@dataclass
class IOConfig:
    config_file: str = ""
    output_dir_base: str = "./runs/"

    # Dataset roots. Leave these unset (null) to resolve them from
    # config/dataset_paths.yaml -- see docs/data.md. Setting one here overrides that
    # file for that dataset only.
    suds_stereo_dir: Optional[str] = None    # SUDS real-world stereo (train_datasets: tbnms)
    uwsim_dir: Optional[str] = None          # UWSim simulated data   (train_datasets: oceansim)
    tartanair_dir: Optional[str] = None
    flyingthings_dir: Optional[str] = None
    svin2_dir: Optional[str] = None
    lizard_island_dir: Optional[str] = None

    # Deprecated alias for uwsim_dir, kept so existing configs keep working.
    oceansim_dir: Optional[str] = None

    # Root of the released checkpoints; unset resolves it from config/dataset_paths.yaml.
    model_weights_dir: Optional[str] = None

    restore_checkpoints: Optional[str] = None

    output_dir: Optional[str] = None # set programatticaly, usually don't set from cmd line or config


@dataclass
class RenderingConfig:
    enable_caustics: bool = True
    enable_water_column: bool = True
    enable_directional_light: bool = True
    enable_particles: bool = True
    enable_halo: bool = True
    background_attenuation_threshold: float = 0.05
    output_width: int = 736
    output_height: int = 320

@dataclass
class TrainingStage:
    stage_name: str = ""
    num_steps: int = 0
    stage_config: dict = field(default_factory=lambda: {})


@dataclass
class EvaluationConfig:
    """What `evaluate.py` runs, and what it writes.

    One eval run is a model crossed with a suite (see config/eval/). `checkpoints`
    names the stages of that model to sweep, so a whole ablation table comes out of
    a single invocation.
    """

    # Stages from the model's `checkpoints:` registry, or explicit paths to .pth
    # files. Resolved by demo.core.find_checkpoint, so a stage this model never ran
    # fails with a list of the ones it did.
    checkpoints: List[str] = field(default_factory=lambda: ["ours"])

    # Datasets to evaluate, by the same names optimization.train_datasets uses.
    # Only real-world data is wired up: the simulated sets need the augmentation
    # pipeline to produce left_uw/right_uw at all. See docs/configurations.md.
    datasets: List[str] = field(default_factory=lambda: ["tbnms"])
    split: str = "test"

    # Restrict to a subset of scenes / cap the frame count. Both are for smoke runs;
    # a reported number should come from the whole split.
    scene_filter: Optional[List[str]] = None
    max_frames: Optional[int] = None

    # Refinement iterations. Null uses the model's own configured test-time value
    # (defom_stereo.valid_iters, igev_pp.valid_iters, or train_iters).
    iters: Optional[int] = None

    # Where results land. Null means <io.output_dir_base>/<name>, matching how
    # train.py derives its run directory.
    output_dir: Optional[str] = None

    save_predictions: bool = True       # per-frame .pt next to the metrics
    save_visualizations: bool = True    # per-frame colormapped disparity png
    skip_existing: bool = True          # resume a killed run instead of redoing it

    # Metric parameters. bad_thresholds are in disparity pixels; max_gt_disparity
    # drops GT pixels above it before scoring (null keeps everything).
    bad_thresholds: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    max_gt_disparity: Optional[float] = None

    # AND the SUDS foreground masks into the valid mask. Only 607 of the 1468 test
    # frames have one, so with this on the scored pixel set is defined slightly
    # differently on those frames than on the rest (it drops ~1% of their GT, mostly
    # the mesh-reprojection fringe where errors concentrate). False scores all finite
    # non-zero ground truth, one definition throughout, which is what the previous
    # evaluation pipeline did.
    use_foreground_mask: bool = True

    # Verify each scene's ground truth was generated against the rectification this
    # code computes, by comparing the intrinsics stored in the GT payload. Catches
    # wilson_candidate3, whose GT carries fx 0.67% larger and cy 2.5 px off, which no
    # shape check can see. Set False to evaluate it anyway, knowing the bias.
    check_gt_rectification: bool = True

@dataclass
class ScoringConfig:
    """What `score_predictions.py` scores: precomputed prediction .pt files
    against SUDS ground truth, without loading a model. See config/eval/scoring/.
    """

    # A directory of <scene>/data/*.pt prediction payloads -- an evaluate.py stage
    # directory, or any tree with the same shape.
    predictions_dir: Optional[str] = None
    # Alternative: an evaluate.py run directory; every stage subdirectory that
    # contains <scene>/data is scored.
    run: Optional[str] = None

    # Scenes to score. Null discovers them from predictions_dir.
    scenes: Optional[List[str]] = None

    # masked: on_geometry / water_column / combined regions from the foreground
    # masks. unmasked: annotated-GT and everywhere-prediction-is-finite regions.
    variants: List[str] = field(default_factory=lambda: ["masked", "unmasked"])

    # Ground-truth source.
    #   suds:   the SUDS release layout via io.suds_stereo_dir. Foreground masks
    #           are {0,1,255} and annotated means > 0.
    #   legacy: <legacy_gt_dir>/<scene>/disparity_maps/*.pt with a depth_mask key;
    #           masks at <legacy_mask_dir>/<scene>/masks_fg/<stem>_left_mask.png,
    #           foreground means > 127, nearest-neighbour resized to the GT shape.
    # The mask rule is tied to the source, not configurable: the two releases
    # encode the value 1 with opposite meanings.
    gt_source: str = "suds"
    legacy_gt_dir: Optional[str] = None
    legacy_mask_dir: Optional[str] = None

    bad_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 3.0, 5.0, 10.0, 15.0])

    # Fail when a scored frame has no foreground mask (masked variant only).
    # False scores such frames on annotated GT alone, with an empty water column.
    require_masks: bool = True

    # Null writes to <predictions_dir>/scores (or <stage>/scores under `run`).
    output_dir: Optional[str] = None

    save_visualizations: bool = False   # 2x3 GT/prediction/error panels per frame
    max_error_for_vis: Optional[float] = None  # colormap ceiling in px; null = per-image

    max_frames: Optional[int] = None    # cap per scene, for smoke runs
    skip_existing: bool = True


@dataclass
class ReconstructionConfig:
    """What `evaluate_reconstruction.py` measures: a reconstruction against a
    ground-truth point cloud. See config/eval/reconstruction/."""

    mode: str = "mesh"                  # mesh | frames

    # mesh mode: one reconstructed mesh vs gt_cloud.
    mesh: Optional[str] = None
    n_samples: int = 100_000            # points sampled from the mesh
    fscore_radius: float = 0.02         # meters
    chamfer_cross_check: bool = True    # warn if pcu and kaolin disagree

    # frames mode: per-frame depth maps (*.npy) projected to point clouds.
    depth_dir: Optional[str] = None
    # Alternative: auto-discover <results_root>/<method>/<scene>/depths, with GT
    # at <gt_root>/<scene>/COLMAP/dense/fused.ply.
    results_root: Optional[str] = None
    gt_root: Optional[str] = None
    intrinsics: Optional[str] = None    # text file: fx fy cx cy
    max_points: int = 200_000           # subsample cap per frame
    max_frames: Optional[int] = None

    gt_cloud: Optional[str] = None      # explicit GT .ply
    output_dir: Optional[str] = None
    skip_existing: bool = True
    seed: int = 42
    verbose: bool = False


@dataclass
class DefomStereoConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128] * 3)
    scale_iters: int = 8
    idepth_scale: float = 0.5
    corr_levels: int = 2
    corr_radius: int = 4
    scale_list: List[float] = field(
        default_factory=lambda: [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    )
    n_downsample: int = 2
    context_norm: str = "batch"
    n_gru_layers: int = 3
    corr_implementation: str = "reg"
    scale_corr_radius: int = 2
    dinov2_encoder: str = "vitl"
    valid_iters: int = 32
    train_iters: int = 18
    mixed_precision: bool = False
    depth_anything_checkpoint: Optional[str] = None


@dataclass
class Config:
    name: str = "exp01"
    model: str = "foundation_stereo"
    group: str = "CVPR Experiments V2"
    vit_size: str = "vits"
    train_iters: int = 22
    max_epochs: Optional[int] = None
    num_val_frames_per_dataset: int = 1000
    devices: Optional[List[int]] = None
    test_sequences: Optional[List[str]] = None

    # Named checkpoints for the selected model, declared once in
    # config/train/models/<model>/<model>.yaml. Ablation configs refer to the stage they
    # warm-start from by name (io.restore_checkpoints: ${checkpoints.full_pretrain}),
    # so the same ablation works for every model. A stage this model never ran is
    # null, which load_model() reports rather than loading something else.
    checkpoints: Dict[str, Optional[str]] = field(default_factory=dict)

    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    foundation_stereo: FoundationStereoConfig = field(default_factory=FoundationStereoConfig)
    igev_pp: IgevPPConfig = field(default_factory=IgevPPConfig)
    defom_stereo: DefomStereoConfig = field(default_factory=DefomStereoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    io: IOConfig = field(default_factory=IOConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    stages: List[TrainingStage] = field(default_factory=lambda: [])
    
    
    def get_dtype(self):
        precision_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        dtype = precision_map[self.optimization.precision_dtype]
        return dtype


CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'config'))

#: Config roots. Training and evaluation use the same overlay machinery, differing
#: only in which tree they search and what the second axis is called: training
#: crosses a model with an `ablations/` entry, evaluation with a `suites/` one.
TRAIN_CONFIG_DIR = os.path.join(CONFIG_DIR, 'train')
EVAL_CONFIG_DIR = os.path.join(CONFIG_DIR, 'eval')

#: Directory holding the second axis, per config root.
AXIS_DIR = {TRAIN_CONFIG_DIR: 'ablations', EVAL_CONFIG_DIR: 'suites'}


def axis_dir(config_root: str) -> str:
    """Name of the second-axis directory under `config_root`."""
    return AXIS_DIR.get(config_root, 'ablations')


def resolve_config_path(config_path: str, relative_to: str = None,
                        config_root: str = TRAIN_CONFIG_DIR) -> str:
    """Turn a config reference into a path on disk.

    Accepted forms, tried in order:
        an existing path, as given (absolute or relative to the CWD)
        the same with a '.yaml' suffix
        a path under `relative_to` (the including config's directory)
        a path under `config_root`, e.g. 'defom_stereo/warp_finetune_full' or 'models/igev_pp/igev_pp'

    Raises FileNotFoundError rather than silently falling back to the defaults.
    """
    candidate_dirs = [None]
    if relative_to is not None:
        candidate_dirs.append(relative_to)
    candidate_dirs.append(config_root)

    tried = []
    for base_dir in candidate_dirs:
        for name in (config_path, config_path + '.yaml'):
            candidate = name if base_dir is None else os.path.join(base_dir, name)
            if os.path.isabs(name) and base_dir is not None:
                continue
            tried.append(candidate)
            if os.path.isfile(candidate):
                return os.path.realpath(candidate)

    raise FileNotFoundError(
        f"config '{config_path}' not found. Tried:\n  " + "\n  ".join(tried))


def load_config_file(config_path: str, _stack=(),
                     config_root: str = TRAIN_CONFIG_DIR) -> "OmegaConf":
    """Load one config, resolving its `includes:` chain first.

    `includes` takes a single path or a list of them, resolved relative to the
    including file (see resolve_config_path). Includes are merged left to right
    and the including file wins over everything it includes, so the effective
    order for config/train/models/defom_stereo/warp_finetune_full.yaml is

        models/defom_stereo/defom_stereo.yaml -> ablations/warp_finetune_full.yaml
        -> models/defom_stereo/warp_finetune_full.yaml

    Because includes resolve relative to the including file first, an eval config
    can reach across into the training tree with a relative path and inherit the
    architecture and checkpoint registry rather than restating them.
    """
    resolved = resolve_config_path(
        config_path, relative_to=os.path.dirname(_stack[-1]) if _stack else None,
        config_root=config_root)

    if resolved in _stack:
        chain = " -> ".join(list(_stack) + [resolved])
        raise ValueError(f"circular config include: {chain}")

    cfg = OmegaConf.load(resolved)
    includes = cfg.pop('includes', None)
    if includes is None:
        return cfg

    if isinstance(includes, str):
        includes = [includes]

    merged = OmegaConf.create({})
    for include in includes:
        merged = OmegaConf.merge(
            merged, load_config_file(include, _stack=tuple(_stack) + (resolved,),
                                     config_root=config_root))
    return OmegaConf.merge(merged, cfg)


def model_config_file(model, config_root: str = TRAIN_CONFIG_DIR):
    """Path to models/<model>/<model>.yaml, or None if there is no such model."""
    path = os.path.join(config_root, 'models', model, model + '.yaml')
    return path if os.path.isfile(path) else None


def list_models(config_root: str = TRAIN_CONFIG_DIR):
    models_dir = os.path.join(config_root, 'models')
    return sorted(d for d in os.listdir(models_dir)
                  if model_config_file(d, config_root))


def list_ablations(config_root: str = TRAIN_CONFIG_DIR):
    """The second axis: `ablations/` under the training root, `suites/` under eval."""
    axis = os.path.join(config_root, axis_dir(config_root))
    return sorted(f[:-5] for f in os.listdir(axis) if f.endswith('.yaml'))


def load_experiment_config(config_path: str,
                           config_root: str = TRAIN_CONFIG_DIR) -> "OmegaConf":
    """Load an experiment, which is a (model, ablation) pair.

    `config_path` is normally '<model>/<ablation>', e.g. 'defom_stereo/warp_finetune_full'.
    Two ways that resolves:

      <config_root>/models/<model>/<ablation>.yaml exists
          Use it. Those files exist only for pairs that need tuning on top of
          the two axes, and they include both axes themselves.
      no such file
          Compose <config_root>/models/<model>/<model>.yaml with
          <config_root>/ablations/<ablation>.yaml. The ablation wins on conflict.

    Any other reference (an explicit path, 'models/igev_pp', ...) is loaded
    directly, so one axis on its own is still usable.

    Under EVAL_CONFIG_DIR the second axis is 'suites/' rather than 'ablations/';
    everything else is identical.
    """
    axis = axis_dir(config_root)
    try:
        return load_config_file(config_path, config_root=config_root)
    except FileNotFoundError:
        model, _, ablation = config_path.partition('/')
        if not ablation:
            raise

        model_file = model_config_file(model, config_root)
        ablation_file = os.path.join(config_root, axis, ablation + '.yaml')
        if model_file and os.path.isfile(ablation_file):
            pair_file = os.path.join(config_root, 'models', model, ablation + '.yaml')
            if os.path.isfile(pair_file):
                return load_config_file(pair_file, config_root=config_root)
            return OmegaConf.merge(load_config_file(model_file, config_root=config_root),
                                   load_config_file(ablation_file, config_root=config_root))

        missing = []
        if not model_file:
            missing.append(f"  no model '{model}'; available: "
                           f"{', '.join(list_models(config_root))}")
        if not os.path.isfile(ablation_file):
            missing.append(f"  no {axis[:-1]} '{ablation}'; available: "
                           f"{', '.join(list_ablations(config_root))}")
        raise FileNotFoundError(
            f"cannot resolve experiment '{config_path}':\n" + "\n".join(missing))


#: Machine-local wandb account, alongside config/dataset_paths.yaml and gitignored for
#: the same reason: it says who is running, not what is being run.
WANDB_CONFIG_FILE = os.path.join(CONFIG_DIR, 'wandb.yaml')
WANDB_EXAMPLE_FILE = os.path.join(CONFIG_DIR, 'wandb.example.yaml')
ENV_WANDB_CONFIG = 'SURF_WANDB_CONFIG'

#: Key in config/wandb.yaml -> the `logging.` field it fills in. `user` is a fallback
#: for `entity`, since a personal wandb account is just an entity named after you, so
#: someone not on a team only has to say who they are. `entity` wins if both are set.
WANDB_CONFIG_KEYS = {
    'entity': 'wandb_entity',
    'user': 'wandb_entity',
    'project': 'wandb_project',
}


def wandb_config_file() -> str:
    """Path to config/wandb.yaml, honouring `$SURF_WANDB_CONFIG`."""
    override = os.environ.get(ENV_WANDB_CONFIG)
    return os.path.expanduser(override) if override else WANDB_CONFIG_FILE


def load_wandb_config() -> "OmegaConf":
    """The `logging:` overlay from config/wandb.yaml, empty if that file is absent.

    Its keys are the flat ones a user cares about (see WANDB_CONFIG_KEYS); a null
    value means "not set" and leaves whatever the configs say alone.
    """
    path = wandb_config_file()
    if not os.path.isfile(path):
        return OmegaConf.create({})

    raw = OmegaConf.load(path)
    if not OmegaConf.is_dict(raw):
        raise ValueError(f"{path} must contain a mapping of "
                         f"{' / '.join(WANDB_CONFIG_KEYS)} -> value "
                         f"(see {os.path.basename(WANDB_EXAMPLE_FILE)})")

    unknown = sorted(set(raw) - set(WANDB_CONFIG_KEYS))
    if unknown:
        print(f"warning: {path} has unrecognized keys: {', '.join(unknown)}. "
              f"Known keys: {', '.join(WANDB_CONFIG_KEYS)}", file=sys.stderr)

    logging_cfg = {}
    for key, field_name in WANDB_CONFIG_KEYS.items():
        value = raw.get(key)
        if value is not None and field_name not in logging_cfg:
            logging_cfg[field_name] = value
    return OmegaConf.create({'logging': logging_cfg})


def get_args(defaults_path: str = None, config_path: str = None,
             use_cli: bool = True,
             config_root: str = TRAIN_CONFIG_DIR) -> Config:
    """Build the run Config from defaults.yaml, an experiment config, and the CLI.

    Layers, each winning over the one before it: the dataclasses above,
    defaults.yaml, config/wandb.yaml (this machine's wandb account, if it exists),
    the experiment config and everything it includes, then the command line.

    `config_path` names an experiment config; a `config_path=` argument on the
    command line takes precedence over the caller's value. Either may use any
    form resolve_config_path() accepts, e.g. 'defom_stereo/warp_finetune_full'.

    `config_root` selects which config tree to resolve names in and, unless
    `defaults_path` says otherwise, which defaults.yaml applies. Pass
    EVAL_CONFIG_DIR for evaluation runs.

    `use_cli=False` skips OmegaConf.from_cli() for callers that parse their own
    command line (demo/cli.py) or build a Config programmatically; without it,
    argparse-style flags in sys.argv reach OmegaConf and raise.
    """
    cli_cfg = OmegaConf.from_cli() if use_cli else OmegaConf.create({})
    config_path = cli_cfg.pop('config_path', config_path)

    # 1. Load structured defaults
    cfg = OmegaConf.structured(Config)

    # 2. Apply defaults.yaml if provided
    if defaults_path is None:
        defaults_path = os.path.join(config_root, 'defaults.yaml')
    if os.path.exists(defaults_path):
        # load_config_file rather than OmegaConf.load, so a defaults file may itself
        # use `includes:` -- config/eval/defaults.yaml inherits the training defaults
        # and overrides only what is unsafe at eval time.
        defaults_yaml = load_config_file(defaults_path, config_root=config_root)
        cfg = OmegaConf.merge(cfg, defaults_yaml)

    # 3. Apply this machine's wandb account. Above defaults.yaml so it does not have to
    #    carry an entity, below the experiment config so a config or the command line
    #    can still send a run somewhere else.
    cfg = OmegaConf.merge(cfg, load_wandb_config())

    # 4. Apply the experiment (plus whatever it includes) if provided
    if config_path is not None:
        cfg = OmegaConf.merge(cfg, load_experiment_config(config_path, config_root))

    # 5. Apply CLI overrides
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # 6. Fold the deprecated io.oceansim_dir into io.uwsim_dir
    if cfg.io.oceansim_dir:
        if not cfg.io.uwsim_dir:
            print("warning: io.oceansim_dir is deprecated; use io.uwsim_dir "
                  "(or register the 'uwsim' dataset). See docs/data.md.")
            cfg.io.uwsim_dir = cfg.io.oceansim_dir
        else:
            print("warning: both io.uwsim_dir and the deprecated io.oceansim_dir are set; "
                  "using io.uwsim_dir.")

    # to_object() resolves ${checkpoints.<name>} references, so a stage this
    # model does not declare fails here rather than deep inside load_model().
    try:
        return OmegaConf.to_object(cfg)
    except InterpolationKeyError as err:
        raise KeyError(
            f"{err}\nThe selected model declares these checkpoints: "
            f"{sorted(cfg.checkpoints.keys())}. Either declare that stage in "
            f"config/train/models/<model>/<model>.yaml (null if this model never ran it), "
            f"or set io.restore_checkpoints for this model/ablation pair. Do not "
            f"point one stage at another run's weights.") from err
