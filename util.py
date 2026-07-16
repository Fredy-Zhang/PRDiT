"""Utility functions for DiT model training, including setup, logging, and evaluation helpers.

Sections
--------
1.  Configuration & Data Structures  — ``Config``, ``Args``, ``load_config``
2.  Environment & Distributed Setup  — ``setup_torch_config``, ``create_logger``, ``cleanup``, …
3.  Weights & Biases                 — ``wandb_enabled``, ``setup_wandb``
4.  Data Loading                     — ``setup_dataloader``, ``return_train_val_loaders``
5.  Model Utilities                  — ``requires_grad``, ``update_ema``, ``initialize_model_with_pretrained``
6.  Optimiser                        — ``initialize_optimizer``
7.  Checkpoint Management            — ``manage_checkpoints``, ``manage_inverse_checkpoints``
8.  Logging & Diagnostics            — ``log_params``, ``print_optimizer_params``, ``weights_detection``
9.  Evaluation & Visualisation       — ``save_evaluation_samples``
10. Miscellaneous                    — ``center_crop_arr``, ``get_project_root``, ``getting_basename``
"""

import logging
import os
import random
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import get_voxel_dataset


# =============================================================================
# 1.  Configuration & Data Structures
# =============================================================================


class Config:
    """Recursive namespace providing dot-notation access to a configuration dict.

    Nested dicts are converted to nested ``Config`` instances automatically.

    Parameters
    ----------
    config_dict : dict
        Flat or nested dictionary loaded from a YAML config file.

    Examples
    --------
    ::

        cfg = Config({"training": {"lr": 1e-4}})
        print(cfg.training.lr)  # 1e-4
    """

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert the config back to a plain dictionary."""
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self.__dict__.items()
        }


def convert_to_numeric(value: Any) -> Any:
    """Recursively convert string representations of numbers to ``int`` or ``float``.

    Parameters
    ----------
    value : any
        Arbitrary value (scalar, dict, or list) read from YAML.

    Returns
    -------
    any
        The input with string numbers replaced by their Python numeric types.
    """
    if isinstance(value, str):
        try:
            return int(value) if value.isdigit() else float(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: convert_to_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_to_numeric(item) for item in value]
    return value


def load_config(config_path: str) -> "Config":
    """Load a YAML configuration file and return a ``Config`` object.

    Numeric strings (e.g. ``"128"``, ``"1e-4"``) are coerced to their Python
    numeric types via :func:`convert_to_numeric`.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Populated ``Config`` instance (empty config if the file is blank).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(convert_to_numeric(config) if config else {})


class Args:
    """Flat argument namespace derived from a ``Config`` object.

    Provides a stable, attribute-based interface expected by helper functions
    such as :func:`setup_dataloader` and :func:`create_experiment_dirs`.

    Parameters
    ----------
    config : Config
        Populated experiment configuration.
    """

    def __init__(self, config: Config):
        wandb_config = getattr(config, "wandb", Config({}))
        self.image_size = config.data.image_size
        self.data_path = config.data.path
        self.results_dir = config.output.results_dir
        self.model = config.model.name
        self.num_classes = config.model.num_classes
        self.total_steps = config.training.total_steps
        self.global_batch_size = config.training.batch_size
        self.global_seed = config.training.seed
        self.num_workers = config.data.num_workers
        self.lr = config.training.learning_rate
        self.log_every = config.logging.log_every
        self.ckpt_every = config.logging.ckpt_every
        self.eval_every = config.logging.eval_every
        self.wandb_project = getattr(wandb_config, "project", "")
        self.wandb_entity = getattr(wandb_config, "entity", "")
        self.gradient_clip = config.training.gradient_clip


# =============================================================================
# 2.  Environment & Distributed Setup
# =============================================================================

def setup_torch_config() -> None:
    """Configure PyTorch / cuDNN settings for optimal training performance.

    Enables TF32 matmul and cuDNN paths, activates cuDNN auto-tuning, and sets
    the default tensor dtype to ``float32``.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.
    deterministic : bool, optional
        When ``True``, forces cuDNN into deterministic mode and disables
        benchmark auto-tuning (slower but reproducible, default ``True``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from monai.utils import set_determinism
    set_determinism(seed=seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_training_environment(args: Any) -> Tuple:
    """Initialise the NCCL process group and create per-rank experiment dirs.

    Intended as a convenience wrapper for scripts that do not use the
    :class:`~train.Trainer` class directly.

    Parameters
    ----------
    args : namespace
        Namespace with ``global_batch_size``, ``global_seed``, ``results_dir``,
        ``model``, and ``num_workers`` attributes.

    Returns
    -------
    tuple
        ``(rank, device, experiment_dir, checkpoint_dir, logger)``
    """
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    set_random_seed(seed)

    if rank == 0:
        experiment_dir, checkpoint_dir = create_experiment_dirs(args)
        logger = create_logger(experiment_dir)
        logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}")
    else:
        experiment_dir = checkpoint_dir = None
        logger = create_logger(None)

    return rank, device, experiment_dir, checkpoint_dir, logger


def create_experiment_dirs(args: Any) -> Tuple[str, str]:
    """Create and return the experiment output and checkpoint directories.

    The experiment directory is numbered sequentially inside ``args.results_dir``
    to avoid overwriting previous runs.

    Parameters
    ----------
    args : namespace
        Namespace with ``results_dir`` and ``model`` attributes.

    Returns
    -------
    tuple of str
        ``(experiment_dir, checkpoint_dir)``
    """
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir


def create_logger(logging_dir: Optional[str]) -> logging.Logger:
    """Create a logger that writes to stdout and (on rank 0) to a log file.

    On ranks other than 0, a ``NullHandler`` is attached so that log calls are
    silently discarded.  The function is safe to call before the process group
    is initialised.

    Parameters
    ----------
    logging_dir : str or None
        Directory in which ``log.txt`` is created (rank 0 only).
        Pass ``None`` on non-zero ranks.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    is_rank_zero = (
        dist.is_available() and dist.is_initialized() and dist.get_rank() == 0
    ) or not (dist.is_available() and dist.is_initialized())

    logger = logging.getLogger(__name__)
    if is_rank_zero and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
    else:
        # Silence all INFO-and-below on non-rank-0 processes, including any
        # library logger that calls logging.basicConfig() at import time.
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        if not root.handlers:
            root.addHandler(logging.NullHandler())
        logger.addHandler(logging.NullHandler())
    return logger


def cleanup() -> None:
    """Destroy the distributed process group if one is active."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# 3.  Weights & Biases
# =============================================================================


def wandb_enabled(config: Config) -> bool:
    """Return ``True`` when wandb logging is enabled in *config*.

    Safe to call even when the ``wandb`` section is absent from the config.

    Parameters
    ----------
    config : Config
        Experiment configuration object.

    Returns
    -------
    bool
        ``True`` when ``config.wandb.enable`` is truthy.
    """
    return bool(getattr(getattr(config, "wandb", Config({})), "enable", False))


def setup_wandb(config: Config, rank: int) -> None:
    """Initialise a wandb run on rank 0 when wandb logging is enabled.

    Has no effect on non-zero ranks or when ``config.wandb.enable`` is falsy.

    Parameters
    ----------
    config : Config
        Experiment configuration object.
    rank : int
        Process rank in the distributed group.
    """
    wandb_config = getattr(config, "wandb", Config({}))
    if rank != 0 or not bool(getattr(wandb_config, "enable", False)):
        return

    init_kwargs: Dict[str, Any] = {
        "project": getattr(wandb_config, "project", None),
        "entity": getattr(wandb_config, "entity", None),
        "name": f"{config.data.task}-{config.model.name}-{config.data.image_size}",
        "config": {
            "architecture": config.model.name,
            "image_size": config.data.image_size,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "total_steps": config.training.total_steps,
            "num_workers": config.data.num_workers,
            "seed": config.training.seed,
            "val_frac": config.data.val_frac,
            "test_frac": getattr(config.data, "test_frac", None),
        },
    }
    tags = getattr(wandb_config, "tags", "")
    if tags:
        init_kwargs["tags"] = tags if isinstance(tags, list) else [tags]
    wandb.init(**init_kwargs)


# =============================================================================
# 4.  Data Loading
# =============================================================================


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker with a deterministic but unique value.

    Should be passed as ``worker_init_fn`` to :class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    worker_id : int
        Worker index assigned by PyTorch (unused directly; the seed is derived
        from ``torch.initial_seed()``).
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_dataloader(
    args: Any,
    config: Config,
    rank: int = 0,
    data_type: str = "train",
    roi_size: Tuple[int, int, int] = (32, 32, 32),
    seed: int = 42,
    augment: bool = True,
) -> DataLoader:
    """Create and return a DataLoader for single-GPU or distributed training.

    Automatically switches between :class:`~torch.utils.data.distributed.DistributedSampler`
    (when the process group is active) and simple shuffle-based loading.

    Parameters
    ----------
    args : namespace
        Namespace with ``data_path``, ``task``, ``global_batch_size``, and
        ``num_workers`` attributes.
    config : Config
        Experiment configuration supplying train/val list paths.
    rank : int, optional
        Process rank; used for sampler construction and diagnostics (default ``0``).
    data_type : str, optional
        ``"train"`` or ``"val"``; controls shuffling and augmentation (default ``"train"``).
    roi_size : tuple of int, optional
        Spatial crop size ``(D, H, W)`` passed to the dataset (default ``(32, 32, 32)``).
    seed : int, optional
        Base random seed for the sampler and worker init (default ``42``).
    augment : bool, optional
        Whether to apply data augmentation (default ``True``).

    Returns
    -------
    DataLoader
        Configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = get_voxel_dataset(
        args.data_path,
        task=args.task,
        train_list=config.data.train_list,
        val_list=config.data.val_list,
        roi_size=roi_size,
        data_type=data_type,
        augment=augment,
        rank=rank,
    )

    is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    should_shuffle = data_type == "train"

    if rank == 0:
        print(f"Data type: {data_type}, Shuffle: {should_shuffle}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=should_shuffle,
            seed=seed,
        )
        batch_size = args.global_batch_size // dist.get_world_size()
        shuffle = False
        if rank == 0:
            print(f"Distributed training: batch_size={batch_size}, world_size={dist.get_world_size()}")
    else:
        sampler = None
        batch_size = args.global_batch_size
        shuffle = should_shuffle
        if rank == 0:
            print(f"Single GPU training: batch_size={batch_size}")

    dataloader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    return DataLoader(**dataloader_kwargs)


def return_train_val_loaders(
    args: Any,
    rank: int,
    config: Config,
    seed: int,
    debug: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create and return train and validation data loaders.

    Mutates *args* in-place to inject ``task`` and ``data_path`` from *config*
    before forwarding to :func:`setup_dataloader`.

    Parameters
    ----------
    args : Args
        :class:`Args` namespace (mutated in-place with ``task`` / ``data_path``).
    rank : int
        Process rank forwarded to :func:`setup_dataloader`.
    config : Config
        Experiment configuration object.
    seed : int
        Random seed for the training loader.  The validation loader always uses
        ``config.training.seed`` for reproducibility.
    debug : bool, optional
        When ``True``, print a seed diagnostic on rank 0 (default ``False``).

    Returns
    -------
    tuple of DataLoader
        ``(train_loader, val_loader)``
    """
    args.task = config.data.task
    args.data_path = config.data.path

    if rank == 0 and debug:
        print(f"Setting up data loaders with seed: {seed}")

    train_loader = setup_dataloader(
        args=args,
        config=config,
        rank=rank,
        data_type="train",
        roi_size=(config.data.image_size,) * 3,
        augment=config.data.augment,
        seed=seed,
    )
    val_loader = setup_dataloader(
        args=args,
        config=config,
        rank=rank,
        data_type="val",
        roi_size=(config.data.image_size,) * 3,
        augment=False,
        seed=config.training.seed,
    )
    return train_loader, val_loader


# =============================================================================
# 5.  Model Utilities
# =============================================================================


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set ``requires_grad`` on all parameters of *model*.

    Parameters
    ----------
    model : torch.nn.Module
        Any ``nn.Module``.
    flag : bool, optional
        Target value for ``requires_grad`` (default ``True``).
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """In-place exponential moving average update of *ema_model* from *model*.

    Formula: ``ema_param = decay * ema_param + (1 - decay) * param``

    Parameters
    ----------
    ema_model : torch.nn.Module
        Shadow model whose parameters are updated.
    model : torch.nn.Module
        Source model providing the latest weights.
    decay : float, optional
        EMA decay factor; higher values slow the update.  ``0`` copies *model*
        exactly (default ``0.9999``).
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)


def initialize_model_with_pretrained(
    model: torch.nn.Module,
    config: Config,
    device: str,
    gain: float = 0.3,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.nn.Module, bool, set]:
    """Initialise model weights and optionally load a pretrained checkpoint.

    Handles three cases:

    - ``config.model.pretrained_path`` is set: loads matching weights, freezes
      coarse-path parameters for ``depth > 0`` models.
    - ``config.model.pretrained_path`` is absent: returns the model unchanged.

    New parameters (shape mismatch or missing from checkpoint) are identified
    and returned so the optimiser can assign them the full learning rate.

    Parameters
    ----------
    model : torch.nn.Module
        The (optionally DDP-wrapped) model to initialise.
    config : Config
        Experiment configuration object.
    device : str
        Device string for ``torch.load`` (e.g. ``"cuda:0"``).
    gain : float, optional
        Weight-init gain forwarded to ``model.initialize_weights`` (default ``0.3``).
    rank : int, optional
        Process rank; print statements execute only on rank 0 (default ``0``).
    logger : logging.Logger or None, optional
        Logger instance; falls back to ``print`` when ``None`` (default ``None``).

    Returns
    -------
    tuple
        ``(model, pretrained_flag, new_param_names)`` — *pretrained_flag* is
        ``True`` when weights were loaded; *new_param_names* is the set of
        parameter names not present in the checkpoint.
    """
    model_to_init = model.module if hasattr(model, "module") else model
    new_param_names: set = set()
    pretrained_flag = False

    is_dit_model = hasattr(model_to_init, "depth")
    is_depth_gt_0 = is_dit_model and model_to_init.depth > 0

    if not getattr(config.model, "pretrained_path", None):
        return model, pretrained_flag, new_param_names

    pretrained_flag = True
    pretrained_dict = torch.load(config.model.pretrained_path, map_location=device)

    if "model" in pretrained_dict:
        pretrained_dict = pretrained_dict["model"]
    elif "ema" in pretrained_dict:
        pretrained_dict = pretrained_dict["ema"]
        if rank == 0:
            print("Using EMA weights from pretrained checkpoint")

    # Strip DDP prefix if present.
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    model_state_dict = model_to_init.state_dict()

    for name in model_state_dict:
        if (
            name.startswith("t_embedder.fine_head.")
            or name not in pretrained_dict
            or model_state_dict[name].shape != pretrained_dict[name].shape
        ):
            new_param_names.add(name)

    if rank == 0:
        print("\n=== Parameter Analysis ===")
        print(f"Total parameters in model: {len(model_state_dict)}")
        print(f"New parameters identified: {len(new_param_names)}")

    if hasattr(model_to_init, "initialize_weights"):
        model_to_init.initialize_weights(gain=gain)

    filtered_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_state_dict and k not in new_param_names and v.shape == model_state_dict[k].shape
    }
    model_to_init.load_state_dict(filtered_dict, strict=False)

    if is_depth_gt_0:
        if hasattr(model_to_init, "coarse"):
            requires_grad(model_to_init.coarse, False)
            if rank == 0:
                print("Frozen MLP denoiser weights")
        if hasattr(model_to_init, "t_embedder"):
            requires_grad(model_to_init.t_embedder.mlp, False)
            requires_grad(model_to_init.t_embedder.coarse_head, False)
            if hasattr(model_to_init.t_embedder, "fine_head"):
                requires_grad(model_to_init.t_embedder.fine_head, True)
            if rank == 0:
                print("Frozen shared MLP and coarse head in timestep embedder")
                print("Fine head in timestep embedder set as trainable")
        for name, param in model_to_init.named_parameters():
            if not (
                name.startswith("coarse.")
                or name.startswith("t_embedder.mlp.")
                or name.startswith("t_embedder.coarse_head.")
            ):
                param.requires_grad = True
    else:
        for _, param in model_to_init.named_parameters():
            param.requires_grad = True

    if rank == 0:
        print("\n=== Weight Loading Summary ===")
        print(f"Successfully loaded: {len(filtered_dict)} / {len(model_state_dict)} parameters")

    return model, pretrained_flag, new_param_names


# =============================================================================
# 6.  Optimiser
# =============================================================================


def initialize_optimizer(
    model: torch.nn.Module,
    config: Config,
    pretrained_flag: bool,
    new_param_names,
    rank: int = 0,
    debug: bool = False,
) -> torch.optim.AdamW:
    """Build an AdamW optimiser with optional differential learning rates.

    When *pretrained_flag* is ``False`` (training from scratch), all trainable
    parameters share ``config.training.learning_rate``.

    When fine-tuning from pretrained weights, newly added parameters receive the
    full LR while pretrained parameters use ``config.training.fine_tune_lr``
    (defaults to 10 % of the full LR).

    Parameters
    ----------
    model : torch.nn.Module
        The (optionally DDP-wrapped) model.
    config : Config
        Experiment configuration object.
    pretrained_flag : bool
        ``True`` when the model was partially initialised from a checkpoint.
    new_param_names : set or list
        Parameter names *not* loaded from the checkpoint (used when fine-tuning).
    rank : int, optional
        Process rank; diagnostics print only on rank 0 (default ``0``).
    debug : bool, optional
        When ``True``, print a summary of param counts and LRs (default ``False``).

    Returns
    -------
    torch.optim.AdamW
        Configured optimiser instance.
    """
    model_module = model.module if hasattr(model, "module") else model
    param_groups: List[Dict[str, Any]] = []

    if not pretrained_flag:
        param_groups.append({
            "params": [p for p in model_module.parameters() if p.requires_grad],
            "lr": config.training.learning_rate,
            "weight_decay": 0,
        })
    else:
        new_params, pretrained_params = [], []
        for name, param in model_module.named_parameters():
            if not param.requires_grad:
                continue
            (new_params if name in new_param_names else pretrained_params).append(param)

        if new_params:
            param_groups.append({
                "params": new_params,
                "lr": config.training.learning_rate,
                "weight_decay": 0,
            })
        if pretrained_params:
            fine_tune_lr = getattr(
                config.training, "fine_tune_lr", config.training.learning_rate * 0.1
            )
            param_groups.append({
                "params": pretrained_params,
                "lr": fine_tune_lr,
                "weight_decay": 0,
            })

    optimizer = torch.optim.AdamW(param_groups)

    if rank == 0 and debug:
        trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_module.parameters())
        print("\nOptimizer Configuration:")
        print(f"- Training mode: {'FROM SCRATCH' if not pretrained_flag else 'FINE-TUNING'}")
        print(f"- Model depth: {getattr(model_module, 'depth', 'Not available')}")
        print(f"- Learning rate: {config.training.learning_rate}")
        print(f"- Total trainable parameters: {trainable:,}")
        print(f"- Total parameters: {total:,}")

    return optimizer


# =============================================================================
# 7.  Checkpoint Management
# =============================================================================

MAX_CHECKPOINTS = 3


def manage_checkpoints(checkpoint_dir: str, rank: int) -> None:
    """Delete the oldest checkpoints, keeping at most ``MAX_CHECKPOINTS`` files.

    Only executes on rank 0.  Uses pure Python (no shell) to avoid injection
    risks from directory paths containing special characters.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing ``*.pt`` checkpoint files.
    rank : int
        Process rank; the function is a no-op on non-zero ranks.
    """
    if rank != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = sorted(glob(os.path.join(checkpoint_dir, "*.pt")))
    for old_ckpt in checkpoints[:-MAX_CHECKPOINTS]:
        os.remove(old_ckpt)


def manage_inverse_checkpoints(checkpoint_dir: str, rank: int) -> None:
    """Delete the newest checkpoints, keeping only the oldest ``MAX_CHECKPOINTS`` files.

    Mirror of :func:`manage_checkpoints` for workflows where the earliest
    checkpoints are the ones to preserve.  Uses pure Python (no shell).

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing ``*.pt`` checkpoint files.
    rank : int
        Process rank; the function is a no-op on non-zero ranks.
    """
    if rank != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = sorted(glob(os.path.join(checkpoint_dir, "*.pt")))
    for new_ckpt in checkpoints[MAX_CHECKPOINTS:]:
        os.remove(new_ckpt)


# =============================================================================
# 8.  Logging & Diagnostics
# =============================================================================


def log_params(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Log parameter statistics for transformer blocks or all model parameters.

    For small tensors (< 100 elements) the full values are logged; for larger
    tensors only min / max / mean statistics are logged.

    Parameters
    ----------
    model : torch.nn.Module
        The (DDP-wrapped) model whose parameters to inspect.
    logger : logging.Logger
        Logger instance to write to.
    """
    if hasattr(model.module, "blocks"):
        logger.info("Logging parameters from transformer blocks:")
        named_params = model.module.blocks.named_parameters()
    else:
        logger.info("Model has no transformer blocks (depth=0). Logging all parameters:")
        named_params = model.module.named_parameters()

    for name, param in named_params:
        if param.numel() < 100:
            logger.info(f"{name} | Shape: {param.shape} | Values: {param.tolist()}")
        else:
            logger.info(
                f"{name} | Shape: {param.shape} | "
                f"Min: {param.min().item():.4f} | "
                f"Max: {param.max().item():.4f} | "
                f"Mean: {param.mean().item():.4f}"
            )


def print_optimizer_params(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    learning_rate: float,
    fine_tune_lr: float,
    logger: logging.Logger,
) -> None:
    """Log a table of parameter names, grad flags, LRs, and param groups.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The active optimiser.
    model : torch.nn.Module
        The (DDP-wrapped) model whose parameters to inspect.
    learning_rate : float
        Full LR (used to identify the "New" param group).
    fine_tune_lr : float
        Fine-tune LR (used to identify the "Fine" param group).
    logger : logging.Logger
        Logger instance to write to.
    """
    param_lrs: Dict[int, float] = {
        id(p): pg.get("lr")
        for pg in optimizer.param_groups
        for p in pg["params"]
    }

    logger.info(f"{'Name':<60}  {'Req_grad':<10}  {'LR':<8}  {'Group'}")
    logger.info("-" * 90)
    for name, p in model.named_parameters():
        lr = param_lrs.get(id(p))
        grp = "New" if lr == learning_rate else ("Fine" if lr == fine_tune_lr else "")
        logger.info(f"{name:<60}  {str(p.requires_grad):<10}  {str(lr):<8}  {grp}")


def weights_detection(
    model: torch.nn.Module,
    device: str,
    pretrained_path: str,
) -> torch.nn.Module:
    """Load pretrained weights with detailed shape-mismatch diagnostics.

    Prints matched, shape-mismatched, and missing parameter names to stdout,
    then loads the compatible subset with ``strict=False``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load weights into.
    device : str
        Device string passed to ``torch.load``.
    pretrained_path : str
        Path to the pretrained checkpoint file.

    Returns
    -------
    torch.nn.Module
        The model with compatible weights loaded in-place.

    Raises
    ------
    Exception
        Re-raises any error from ``torch.load`` or ``load_state_dict``.
    """
    try:
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()

        print(f"\nLoading pretrained weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
            print("Found 'model' key in pretrained dict")

        missing_params = [k for k in model_state_dict if k not in pretrained_dict]

        filtered_dict: Dict[str, torch.Tensor] = {}
        matched_params: List[str] = []
        mismatched_params: List[str] = []

        for k, v in pretrained_dict.items():
            k = k.replace("module.", "")
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_dict[k] = v
                    matched_params.append(f"{k} (shape: {v.shape})")
                else:
                    mismatched_params.append(
                        f"{k} (pretrained: {v.shape}, model: {model_state_dict[k].shape})"
                    )

        print(f"\n=== Weight Loading Summary ===")
        print(f"\nSuccessfully matched parameters ({len(matched_params)}):")
        for param in matched_params:
            print(f"  {param}")
        if mismatched_params:
            print(f"\nShape mismatched parameters ({len(mismatched_params)}):")
            for param in mismatched_params:
                print(f"  {param}")
        if missing_params:
            print(f"\nMissing parameters ({len(missing_params)}):")
            for param in missing_params:
                print(f"  {param}")

        target = model.module if is_ddp else model
        target.load_state_dict(filtered_dict, strict=False)
        print(f"\nTotal parameters loaded: {len(filtered_dict)}/{len(model_state_dict)}")
        return model

    except Exception as e:
        print(f"Error loading weights: {e}")
        raise


# =============================================================================
# 9.  Evaluation & Visualisation
# =============================================================================


def save_evaluation_samples(
    samples: torch.Tensor,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    nii_number: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save a batch of 3-D volumes as orthogonal-view PNGs and NIfTI files.

    For each sample, three 2-D slices (axial, coronal, sagittal) are rendered
    side-by-side and saved under ``experiment_dir/visualizations/``.  A
    configurable number of full 3-D volumes are additionally saved as
    ``*.nii.gz`` directly in ``experiment_dir``.

    Parameters
    ----------
    samples : torch.Tensor
        Batch tensor of shape ``(B, C, D, H, W)`` on any device.
    experiment_dir : str
        Root output directory (created if absent).
    image_size : int
        Spatial size used to compute the middle slice index.
    epoch : int
        Current epoch, embedded in saved file names.
    nii_number : int or None, optional
        Maximum number of NIfTI volumes to write; ``None`` saves all (default ``None``).
    logger : logging.Logger or None, optional
        Logger for progress messages (default ``None``).
    """
    import matplotlib.pyplot as plt

    os.makedirs(experiment_dir, exist_ok=True)
    vis_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    samples = samples.detach()
    mid_slice = image_size // 2

    def _save_orthogonal_views(volume: torch.Tensor, idx: int) -> None:
        """Render and save axial / coronal / sagittal slices for one volume."""
        slices_np = [
            volume[:, :, mid_slice].cpu().float().numpy(),
            volume[:, mid_slice, :].cpu().float().numpy(),
            volume[mid_slice, :, :].cpu().float().numpy(),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax, s in zip(axes, slices_np):
            ax.imshow(s, cmap="gray")
            ax.axis("off")
        plt.savefig(
            os.path.join(vis_dir, f"sample_{idx}_views_epoch_{epoch}.png"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close(fig)

    def _save_nifti_volumes(vols: torch.Tensor) -> None:
        """Write *vols* as NIfTI-1 images; silently skips if nibabel is absent."""
        try:
            import nibabel as nib
            for i, vol in enumerate(vols):
                nib.save(
                    nib.Nifti1Image(vol.squeeze(0).cpu().float().numpy(), np.eye(4)),
                    os.path.join(experiment_dir, f"sample_{i}_{epoch}.nii.gz"),
                )
            if logger:
                logger.info(f"Saved {len(vols)} NIfTI volumes at epoch {epoch}")
        except ImportError:
            if logger:
                logger.warning("nibabel not installed, skipping NIfTI saving")

    batch_size = samples.shape[0]
    for idx in range(batch_size):
        _save_orthogonal_views(samples[idx, 0], idx)

    if logger:
        logger.info(f"Saved orthogonal views for {batch_size} samples at epoch {epoch}")

    n = min(nii_number, batch_size) if nii_number is not None else batch_size
    _save_nifti_volumes(samples[:n])


# =============================================================================
# 10. W&B Image Logging
# =============================================================================


def log_slices_to_wandb(tag: str, volumes: torch.Tensor, epoch: int) -> None:
    """Upload orthogonal middle slices of a volume batch to wandb.

    For each volume in the batch the axial, coronal, and sagittal centre slices
    are normalised to [0, 255] and logged as ``wandb.Image`` objects under
    ``tag``.  No-ops silently when ``wandb.run`` is not active.

    Parameters
    ----------
    tag : str
        Panel name in the wandb UI (e.g. ``"eval/reconstruction"``).
    volumes : torch.Tensor
        Float tensor of shape ``(B, C, D, H, W)`` on any device.
    epoch : int
        Current epoch, logged alongside the images.
    """
    try:
        import wandb as _wandb
    except ImportError:
        return

    if _wandb.run is None:
        return

    volumes = volumes.detach().cpu().float()
    B, _C, D, H, W = volumes.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        return (arr * 255).clip(0, 255).astype(np.uint8)

    images = []
    for b in range(B):
        vol = volumes[b, 0].numpy()  # [D, H, W]
        images += [
            _wandb.Image(_to_uint8(vol[mid_d, :, :]), caption=f"sample{b} axial"),
            _wandb.Image(_to_uint8(vol[:, mid_h, :]), caption=f"sample{b} coronal"),
            _wandb.Image(_to_uint8(vol[:, :, mid_w]), caption=f"sample{b} sagittal"),
        ]

    _wandb.log({tag: images, "epoch": epoch})


# =============================================================================
# 11. Miscellaneous
# =============================================================================


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Centre-crop a PIL image to ``image_size × image_size``.

    Down-samples by 2× repeatedly until the shorter side is less than
    ``2 * image_size``, then rescales and crops.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Source PIL image.
    image_size : int
        Target square side length in pixels.

    Returns
    -------
    PIL.Image.Image
        Cropped image of size ``(image_size, image_size)``.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_project_root() -> str:
    """Return the name of the directory containing this file.

    Returns
    -------
    str
        Parent directory name of the resolved script path.
    """
    return Path(__file__).resolve().parent.name


def getting_basename() -> str:
    """Return the name of the directory containing this script.

    Also prints the directory name to stdout.

    Returns
    -------
    str
        Directory name string.
    """
    directory_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    print("The script is located in directory:", directory_name)
    return directory_name
