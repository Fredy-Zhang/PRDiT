"""TensorBoard visualization helpers for PRDiT-style models."""

from __future__ import annotations

import torch
from torch.utils.tensorboard import SummaryWriter


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first model parameter."""
    return next(model.parameters()).device


def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and flatten a tensor before logging."""
    return tensor.detach().reshape(-1).cpu()


class ModelVisualizer:
    """Utility for logging intermediate model activations to TensorBoard."""

    def __init__(self, model: torch.nn.Module, log_dir: str = "runs/dit_model"):
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def visualize_layer_outputs(self, x: torch.Tensor, t: torch.Tensor, step: int = 0) -> None:
        """Log embeddings, block outputs, and attention weights when available."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "x_embedder") and hasattr(self.model, "pos_embed"):
                x_embedded = self.model.x_embedder(x) + self.model.pos_embed
                self.writer.add_histogram("embeddings/patch_embed", _flatten_tensor(x_embedded), step)
            else:
                x_embedded = x

            if not hasattr(self.model, "t_embedder"):
                raise AttributeError("ModelVisualizer expects the model to expose `t_embedder`.")
            t_embedded = self.model.t_embedder(t)
            self.writer.add_histogram("embeddings/time_embed", _flatten_tensor(t_embedded), step)

            x_current = x_embedded
            for i, block in enumerate(getattr(self.model, "blocks", [])):
                x_current = block(x_current, t_embedded)
                self.writer.add_histogram(
                    f"transformer/block_{i}/output",
                    _flatten_tensor(x_current),
                    step,
                )

                attn = getattr(block, "attn", None)
                attn_weights = getattr(attn, "attention_weights", None)
                if attn_weights is not None:
                    self.writer.add_histogram(
                        f"attention/block_{i}",
                        _flatten_tensor(attn_weights),
                        step,
                    )

    def visualize_parameter_distributions(self, step: int = 0) -> None:
        """Log trainable parameter distributions to TensorBoard."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"parameters/{name}", _flatten_tensor(param), step)

    def close(self) -> None:
        """Close the underlying TensorBoard writer."""
        self.writer.close()


def add_model_graph(
    model: torch.nn.Module,
    writer: SummaryWriter,
    input_shape: tuple[int, int, int, int, int] = (2, 1, 256, 256, 256),
) -> None:
    """Add the model graph to TensorBoard with a dummy input."""
    device = _get_model_device(model)
    dummy_input = (
        torch.randn(input_shape, device=device),
        torch.randint(0, 1000, (input_shape[0],), device=device),
    )
    writer.add_graph(model, dummy_input)


def visualize_attention_patterns(
    model: torch.nn.Module,
    writer: SummaryWriter,
    input_shape: tuple[int, int, int, int, int] = (2, 1, 256, 256, 256),
    step: int = 0,
) -> None:
    """Log attention maps for blocks that expose `attention_weights`."""
    device = _get_model_device(model)
    x = torch.randn(input_shape, device=device)
    t = torch.randint(0, 1000, (input_shape[0],), device=device)

    model.eval()
    with torch.no_grad():
        model(x, t)

        for i, block in enumerate(getattr(model, "blocks", [])):
            attn = getattr(block, "attn", None)
            attn_weights = getattr(attn, "attention_weights", None)
            if attn_weights is None:
                continue

            attn_map = attn_weights[0].detach()
            if attn_map.ndim == 2:
                writer.add_image(f"attention_map/block_{i}", attn_map.unsqueeze(0), step, dataformats="CHW")
