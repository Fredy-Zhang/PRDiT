"""Reusable building blocks for PRDiT-style architectures.

This module groups lower-level neural network components used by the main model
definition, including attention layers, patch extractors, simple MLP blocks,
and a few normalization / conditioning helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from models.utils import to_3tuple
from models.utils import drop_path

class Attention(nn.Module):
    """Multi-head self-attention with optional flash-attention execution.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    num_heads : int, optional
        Number of attention heads (default ``8``).
    qkv_bias : bool, optional
        Use bias in QKV projections (default ``False``).
    qk_norm : bool, optional
        Normalize query and key heads (default ``False``).
    attn_drop : float, optional
        Dropout probability on attention weights (default ``0.``).
    proj_drop : float, optional
        Dropout probability on the output projection (default ``0.``).
    norm_layer : nn.Module, optional
        Normalization applied to query and key heads (default ``nn.LayerNorm``).
    use_flash_attention : bool, optional
        Use ``F.scaled_dot_product_attention`` (default ``False``).
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Standard scale without modification
        self.use_flash_attention = use_flash_attention
        
        # Initialize layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a token sequence of shape `[B, N, C]`."""
        B, N, C = x.shape
        
        # Reshape qkv to split heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        if self.use_flash_attention:
            # Flash attention expects (B, H, N, D) format for q, k, v
            qkv = qkv.permute(0, 3, 1, 2, 4)  # (B, H, N, 3, D)
            q = qkv[..., 0, :]  # (B, H, N, D)
            k = qkv[..., 1, :]  # (B, H, N, D)
            v = qkv[..., 2, :]  # (B, H, N, D)
            
            # Apply normalization
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            # Cast to appropriate precision for flash attention
            with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                # Use scaled_dot_product_attention with explicit scale parameter
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                ).transpose(1, 2)
                
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            
        else:
            # Traditional attention path
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q) * self.scale
            k = self.k_norm(k)
            
            attn = q @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = (attn @ v).transpose(1, 2)
            
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class EmbDecoder(nn.Module):
    """
    Decode patch embeddings back into patch-space predictions.

    This lightweight head is used where no extra conditioning path is needed.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process and project features without conditioning."""
        return self.linear(self.norm_final(x))

class RMSNorm(nn.Module):
    """
    Root-mean-square normalization over the last feature dimension.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor with optional affine parameters."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y

class PatchExtractor3D(nn.Module):
    """
    Extract 3D patches with a strided convolution and flatten them into tokens.
    """
    def __init__(
            self,
            vol_size: int = 128,
            patch_size: int = 16,
            stride: int = 4, 
            padding: int = 2,
            in_chans: int = 1,
            strict_vol_size: bool = True,
            dynamic_vol_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.vol_size = to_3tuple(vol_size) if vol_size is not None else None
        
        # Calculate grid size and number of patches
        if self.vol_size is not None:
            self.grid_size = tuple(
                [((s + 2 * padding - p) // stride) + 1 
                 for s, p in zip(self.vol_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.grid_size = None
            self.num_patches = None
        
        self.strict_vol_size = strict_vol_size
        self.dynamic_vol_pad = dynamic_vol_pad

        # Patch extraction layer
        self.proj = nn.Conv3d(
            in_chans, in_chans,  # Keep same number of channels initially
            kernel_size=self.patch_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an input volume `[B, C, D, H, W]` into patch tokens `[B, N, C]`."""
        x = self.proj(x)  # [B, C, D', H', W']
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchEmbedder3D(nn.Module):
    """
    Project raw patch tokens into a learned embedding space.
    """
    def __init__(
            self,
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            activation: Callable = nn.ReLU(),
    ):
        super().__init__()
        # Linear projection layer
        self.embed = nn.Linear(in_chans, embed_dim, bias=True)
        self.activation = activation
        self.mlps = MLPWithSkip(embed_dim, activation=activation)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map patch tokens `[B, N, C]` to embedded tokens `[B, N, embed_dim]`."""
        x = self.embed(x)
        x = self.activation(x)
        x = self.mlps(x)
        return self.norm(x)

class MLPWithSkip(nn.Module):
    """Two-layer MLP with a residual skip connection."""
    def __init__(self, dim, activation: Callable = nn.ReLU):
        super(MLPWithSkip, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = activation
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        """Apply the residual MLP block to the input tensor."""
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = out + identity
        out = self.act(out)
        return out

class CrossAttention(nn.Module):
    """Bidirectional cross-attention between latent tokens and patch tokens.

    Parameters
    ----------
    dim_lat : int
        Latent token dimension.
    dim_pat : int
        Patch token dimension.
    dim_attn : int
        Internal attention dimension (must be divisible by ``num_heads``).
    num_heads : int, optional
        Number of attention heads (default ``8``).
    rv_bias : bool, optional
        Use bias in attention projections (default ``False``).
    attn_drop : float, optional
        Dropout probability on attention weights (default ``0.``).
    proj_drop : float, optional
        Dropout probability on output projections (default ``0.``).
    """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.rv_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)             # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_pat)             # 'out-projection' for patches/tokens
        self.proj_drop_pat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        """Update latent and patch tokens through bidirectional cross-attention."""
        B_lat, N_lat, _ = x_latents.shape
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.rv_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_lat.unbind(0)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale  # query from latent, key from patches
        attn_T = attn.transpose(-2, -1)  # bidirectional attention, associated with the values from the query q

        attn = attn.softmax(dim=-1)  # softmax along patch token dimension
        attn_T = attn_T.softmax(dim=-1)  # softmax along latent token dimension

        attn = self.attn_drop(attn)
        attn_T = self.attn_dropT(attn_T)

        # Retrieve information from patches into latents.
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        # Push information from latents back into patch tokens.
        x_patches = (attn_T @ v_lat).transpose(1, 2).reshape(B_pat, N_pat, self.dim_attn)
        x_patches = self.proj_pat(x_patches)
        x_patches = self.proj_drop_pat(x_patches)

        return x_latents, x_patches

class CrossAttentionLatents(nn.Module):
    """Cross-attention variant that updates only the latent tokens."""
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.r_latents = nn.Linear(dim_lat, dim_attn, bias=rv_bias)             # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)        # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)                            # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        """Refine latents using patch tokens as the attention source."""
        B_lat, N_lat, _ = x_latents.shape
        B_pat, N_pat, _ = x_patches.shape
        r_lat = self.r_latents(x_latents).reshape(B_lat, N_lat, 1, self.num_heads,
                                                  self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)

        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)   # softmax along patch token dimension
        attn = self.attn_drop(attn)

        # Retrieve information from patches into latents.
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        return x_latents

class CrossAttentionPatches(nn.Module):
    """Cross-attention variant that updates only the patch tokens."""
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.r_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)             # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn, bias=rv_bias)        # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_lat)                            # 'out-projection' for latents
        self.proj_drop_pat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        """Refine patch tokens using latent tokens as the attention source."""
        B_lat, N_lat, _ = x_latents.shape
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.r_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                  self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_lat.unbind(0)
        
        r_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 1, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale # R(M, N)

        attn = attn.softmax(dim=-1)   # softmax along patch token dimension
        attn = self.attn_drop(attn)

        # Retrieve information from latents into patches.
        x_patches = (attn.transpose(-2, -1) @ v_lat).transpose(1, 2).reshape(-1, N_pat, self.dim_attn)
        x_patches = self.proj_pat(x_patches)
        x_patches = self.proj_drop_pat(x_patches)

        return x_patches

    
class SelfAttention(nn.Module):
    """Self-attention layer with a configurable internal attention dimension."""
    def __init__(
            self,
            dim: int,
            dim_attn: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim_attn = dim_attn

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a token sequence."""
        B, N, _ = x.shape
        # Reshape qkv to split heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_attn)
        return self.proj_drop(self.proj(x))

class LabelEmbedder(nn.Module):
    """Class-label embedder with classifier-free guidance token dropping."""
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Replace selected class labels with the unconditional CFG token."""
        drop_ids = force_drop_ids == 1 if force_drop_ids is not None else \
                  torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed labels, optionally applying classifier-free guidance dropout."""
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

class LayerScale(nn.Module):
    """Learnable residual scaling layer for stabilizing deep networks."""
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        """Scale the input by a learned per-channel gain."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class DropPath(nn.Module):
    """Stochastic depth layer for residual-path regularization."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Apply stochastic depth to the input tensor."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
