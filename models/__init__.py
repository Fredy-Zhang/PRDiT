"""PRDiT model registry.

This module defines the supported PRDiT model variants and provides helpers for:
- constructing a PRDiT instance from model-scale parameters
- registering named model variants in a global registry
- loading a model from the experiment configuration

Naming convention:
- `PRDiT-{size}/{patch_size}/{depth}`
- `size` is one of `XS`, `S`, `B`, `L`, `XL`
- `patch_size` controls patch extraction size
- `depth` controls the number of transformer blocks
"""

from __future__ import annotations

from models.models import PRDiT


MODEL_HIDDEN_SIZES = {
    "XS": 192,
    "S": 384,
    "B": 768,
    "L": 1024,
    "XL": 1152,
}

STANDARD_PATCH_SIZES = (10, 12, 14, 16)
STANDARD_STRIDE = 8
STANDARD_DEPTHS = (0, 1, 2, 3, 4, 8, 10, 12)
STANDARD_MLP_RATIO = 4.0
STANDARD_SIZES_NUM_HEADS = (
    ("XS", 6),
    ("S", 6),
    ("B", 12),
    ("L", 16),
    ("XL", 16),
)

SPECIAL_MODEL_VARIANTS = (
    {
        "name": "PRDiT-XS/4/0",
        "size": "XS",
        "patch_size": 4,
        "depth": 0,
        "stride": 4,
        "padding": 0,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    {
        "name": "PRDiT-XS/4/1",
        "size": "XS",
        "patch_size": 4,
        "depth": 1,
        "stride": 2,
        "padding": 0,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
)


PRDiT_models = {}


def create_prdit_model(
    size: str = "S",
    patch_size: int = 12,
    depth: int = 0,
    stride: int = 8,
    padding: int = 2,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    **kwargs,
):
    """Create a :class:`~models.models.PRDiT` from a compact set of architecture parameters.

    Parameters
    ----------
    size : str, optional
        Model scale key — one of ``"XS"``, ``"S"``, ``"B"``, ``"L"``, ``"XL"``
        (default ``"S"``).
    patch_size : int, optional
        Patch extraction edge length (default ``12``).
    depth : int, optional
        Number of transformer blocks; ``0`` builds a stage-1 local denoiser
        (default ``0``).
    stride : int, optional
        Patch extraction stride (default ``8``).
    padding : int, optional
        Padding before patch extraction (default ``2``).
    num_heads : int, optional
        Number of attention heads (default ``6``).
    mlp_ratio : float, optional
        MLP expansion ratio (default ``4.0``).
    **kwargs
        Extra keyword arguments forwarded to :class:`~models.models.PRDiT`.

    Returns
    -------
    PRDiT
        Constructed model instance.
    """
    if size not in MODEL_HIDDEN_SIZES:
        raise ValueError(f"Invalid model size: {size}. Must be one of {list(MODEL_HIDDEN_SIZES.keys())}")

    return PRDiT(
        depth=depth,
        hidden_size=MODEL_HIDDEN_SIZES[size],
        patch_size=patch_size,
        stride=stride,
        padding=padding,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        **kwargs,
    )


def register_prdit_model(name: str, **model_args):
    """Register a named PRDiT constructor in the global ``PRDiT_models`` registry.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"PRDiT-B/12/4"``.
    **model_args
        Architecture arguments forwarded to :func:`create_prdit_model`.

    Returns
    -------
    callable
        The registered constructor function.
    """

    def model_fn(**kwargs):
        return create_prdit_model(**model_args, **kwargs)

    PRDiT_models[name] = model_fn
    return model_fn


def load_model(config):
    """Instantiate the configured PRDiT model from the ``PRDiT_models`` registry.

    Parameters
    ----------
    config : Config
        Experiment configuration; ``config.model.name`` selects the variant.

    Returns
    -------
    PRDiT
        Constructed and configured model instance.

    Raises
    ------
    ValueError
        If ``config.model.name`` is not registered.
    """
    model_name = config.model.name
    if model_name not in PRDiT_models:
        raise ValueError(f"Model name {model_name} is not recognized.")

    return PRDiT_models[model_name](
        input_size=config.data.image_size,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        learn_sigma=(config.model.out_channels == 2),
        flash_attn=config.model.flash_attn,
    )


def register_all_prdit_models(
    patch_sizes: tuple[int, ...] = STANDARD_PATCH_SIZES,
    stride: int = STANDARD_STRIDE,
    sizes_num_heads: tuple[tuple[str, int], ...] = STANDARD_SIZES_NUM_HEADS,
    depths: tuple[int, ...] = STANDARD_DEPTHS,
    mlp_ratio: float = STANDARD_MLP_RATIO,
) -> None:
    """Register the standard grid of PRDiT model variants."""
    for patch_size in patch_sizes:
        padding = (patch_size - stride) // 2
        for size, num_heads in sizes_num_heads:
            for depth in depths:
                register_prdit_model(
                    name=f"PRDiT-{size}/{patch_size}/{depth}",
                    size=size,
                    patch_size=patch_size,
                    depth=depth,
                    stride=stride,
                    padding=padding,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )


def register_special_prdit_models() -> None:
    """Register hand-picked non-standard variants."""
    for variant in SPECIAL_MODEL_VARIANTS:
        register_prdit_model(variant["name"], **{k: v for k, v in variant.items() if k != "name"})


register_all_prdit_models()
register_special_prdit_models()
