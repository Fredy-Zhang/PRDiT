"""
PRDiT model definitions.

This module implements PRDiT, a two-stage architecture for 3D medical volume
generation and denoising.

The model is organized into two branches:
- Coarse branch: an MLP-based patch denoiser that captures the base structure
- Fine branch: a transformer-based refiner that models higher-frequency residual detail

Training modes:
- `depth = 0`: coarse-only model used for stage-1 training
- `depth > 0`: hybrid model used for stage-2 refinement, where the coarse branch
  can be frozen and the fine branch is trained on top

Main components:
- `ExtractPatches3D`: extracts patch tokens from 3D volumes
- `PatchEmbed3D`: projects patch tokens into the hidden space
- `TimestepEmbedder`: produces separate timestep conditioning for coarse and fine paths
- `CoarseDenoiser`: MLP-based coarse denoising branch
- `FineRefiner`: transformer-based refinement branch
- `PRDiTBlock`: conditioned transformer block used in the fine path
- `PRDiT`: top-level model tying all components together
"""

# Standard library imports
import math
import logging
from typing import Callable, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SwiGLU
from timm.models.vision_transformer import Mlp

# Local imports
from models.utils import _ntuple, modulate, unpatchify_3d
from models.classes import Attention, RMSNorm
from util import requires_grad
from models.utils import get_normalized_3d_pos_enc

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Helper functions
to_3tuple = _ntuple(3)


class ExtractPatches3D(nn.Module):
    """
    Extract 3D patches from a volume and flatten them into a token sequence.

    The module uses chained `unfold` operations so it can support overlapping
    patches as well as non-overlapping ones. The output is arranged in the
    transformer-friendly shape `[B, N, patch_dim]`.
    
    Args:
        patch_size: Size of patches to extract (scalar or 3-tuple)
        stride: Stride for patch extraction (scalar or 3-tuple)
        padding: Padding to apply before extraction
    """
    
    def __init__(self, patch_size: Union[int, Tuple[int, int, int]], 
                 stride: Union[int, Tuple[int, int, int]], 
                 padding: int = 0):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.stride = to_3tuple(stride)
        self.padding = padding
    
    def compute_num_patches(self, input_size: Union[int, Tuple[int, int, int]]) -> Tuple[int, Tuple[int, int, int]]:
        """Return the number of extracted patches and the 3D patch grid shape."""
        input_size = to_3tuple(input_size)
        if self.padding > 0:
            input_size = tuple(s + 2 * self.padding for s in input_size)
        
        grid_size = tuple(
            ((s - p) // st) + 1 
            for s, p, st in zip(input_size, self.patch_size, self.stride)
        )
        return grid_size[0] * grid_size[1] * grid_size[2], grid_size
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract and flatten 3D patches from an input volume.
        
        Args:
            volume: Input volume tensor [B, C, D, H, W]
            
        Returns:
            Patches tensor [B, num_patches, C * patch_volume]
        """
        B, C, D, H, W = volume.size()
        
        # Apply padding if specified
        if self.padding > 0:
            volume = F.pad(volume, (self.padding,) * 6, mode='reflect')

        # Unfold the volume into patches
        patches = (volume
                  .unfold(2, self.patch_size[0], self.stride[0])
                  .unfold(3, self.patch_size[1], self.stride[1])
                  .unfold(4, self.patch_size[2], self.stride[2]))
        
        # Calculate dimensions and reshape
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        num_patches = patches.numel() // (B * C * patch_volume)
        
        # Reshape to sequence format: [B, num_patches, C * patch_volume]
        patches = (patches.contiguous()
                  .view(B, C, num_patches, patch_volume)
                  .permute(0, 2, 1, 3)
                  .reshape(B, num_patches, -1))
        return patches

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return f'patch_size={self.patch_size}, stride={self.stride}, padding={self.padding}'

class PatchEmbed3D(nn.Module):
    """
    Embed flattened 3D patches with an MLP plus a linear skip projection.

    The main path learns a nonlinear projection into the hidden space, while the
    skip path preserves a direct linear route from raw patch values to the final
    embedding. This makes the patch projection more expressive without losing a
    simple residual path.
    
    Args:
        patch_size: Size of cubic patch edge
        in_chans: Number of input channels (e.g., 1 for CT scans)
        embed_dim: Output embedding dimension
        norm_layer: Normalization layer (None for no normalization)
        mlp_ratio: Hidden layer expansion ratio
        activation: Activation function instance
        dropout: Dropout probability
    """
    
    def __init__(self,
                 patch_size: int = 16,
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = nn.LayerNorm,
                 mlp_ratio: float = 4.0,
                 activation: Callable = nn.GELU(approximate="tanh"),
                 dropout: float = 0.0):
        super().__init__()
        
        # Calculate layer dimensions
        input_dim = in_chans * (patch_size ** 3)
        hidden_dim = int(embed_dim * mlp_ratio)
        
        logger.debug(f"PatchEmbed3D: input_dim={input_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}")
        
        # Main transformation path (two-layer MLP)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)
        
        # Skip connection (direct projection)
        self.skip = nn.Linear(input_dim, embed_dim, bias=False)
        
        # Optional normalization and dropout
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MLP transformation and skip connection.
        
        Args:
            x: Flattened patches [B, N, C * P^3]
            
        Returns:
            Embedded patches [B, N, embed_dim]
        """
        # Main transformation path
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)

        # Skip connection
        s = self.skip(x)

        # Combine, normalize, and apply dropout
        out = self.norm(h + s)
        return self.drop(out)

class TimestepEmbedder(nn.Module):
    """
    Timestep embedder with separate coarse and fine conditioning heads.

    Scalar diffusion timesteps are first converted into sinusoidal embeddings,
    then passed through a shared MLP. The resulting shared representation is
    projected into two different conditioning spaces:

    - a coarse embedding for the MLP denoiser
    - a fine embedding for the transformer refiner

    This split is important for staged training because the coarse branch can be
    frozen while the fine branch keeps learning.
    
    Args:
        hidden_size: Hidden dimension for shared MLP
        coarse_hidden_size: Output dimension for coarse head
        fine_hidden_size: Output dimension for fine head  
        frequency_embedding_size: Dimension of sinusoidal embeddings
        is_depth_zero: If True, fine_head becomes Identity (depth=0 models)
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 coarse_hidden_size: int, 
                 fine_hidden_size: int, 
                 frequency_embedding_size: int = 256,
                 is_depth_zero: bool = True):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        
        # Shared MLP for timestep processing
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU()
        )
        
        # Separate heads for coarse and fine paths
        self.coarse_head = nn.Linear(hidden_size, coarse_hidden_size, bias=True)
        self.fine_head = (nn.Identity() if is_depth_zero 
                         else nn.Linear(hidden_size, fine_hidden_size, bias=True))

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: Timestep tensor [B]
            dim: Embedding dimension
            max_period: Maximum period for frequency computation
            
        Returns:
            Sinusoidal embeddings [B, dim]
        """
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process timesteps through shared MLP and dual heads.
        
        Args:
            t: Timestep tensor [B]
            
        Returns:
            Tuple of (coarse_embedding, fine_embedding) each [B, respective_hidden_size]
        """
        # Generate sinusoidal embeddings and process through shared MLP
        timestep_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        shared_features = self.mlp(timestep_emb)
        
        # Project through separate heads
        coarse_emb = self.coarse_head(shared_features)
        fine_emb = self.fine_head(shared_features)
        
        return coarse_emb, fine_emb

class PRDiTBlock(nn.Module):
    """
    Transformer refinement block with AdaLN-Zero conditioning.

    Each block applies attention and MLP updates under timestep-conditioned
    modulation. The conditioning network predicts the shift, scale, and gating
    values used to control both sublayers.
    
    Args:
        hidden_size: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension expansion ratio
        flash_attn: Whether to use flash attention
        **block_kwargs: Additional arguments for attention layer
    """
    
    def __init__(self,
                 hidden_size: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 flash_attn: bool = False, 
                 **block_kwargs):
        super().__init__()
        
        # Layer normalization (no learnable parameters - controlled by conditioning)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Core transformer components
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True, 
            use_flash_attention=flash_attn, 
            **block_kwargs
        )
        
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0
        )
        
        # AdaLN conditioning network (6 params: shift/scale/gate for attn and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    Output projection layer for the PRDiT refinement path.

    The layer maps hidden patch features back to output patch predictions. For
    dual-output setups (`out_channels == 2`), it keeps separate projections for
    image and noise predictions and can optionally unpatchify them back into
    dense 3D volumes.
    
    Args:
        hidden_size: Feature dimension
        patch_size: Size of output patches
        out_channels: Number of output channels
        input_size: Size of input volume (for unpatchify)
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, input_size: int = None):
        super().__init__()
        # TODO: use LayerNorm for noise head
        self.norm_noise = nn.LayerNorm(hidden_size, 
                                       elementwise_affine=False, 
                                       eps=1e-6)
        # TODO: use RMSNorm for image head
        self.norm_image = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        
        # Store configuration
        self.patch_size = patch_size
        self.input_size = input_size
        self.out_channels = out_channels
        
        # Dual-head projection
        if out_channels == 2:
            self.linear_noise = nn.Linear(hidden_size, patch_size**3*1, bias=True)
            self.linear_image = nn.Linear(hidden_size, patch_size**3*1, bias=True)
        else:
            self.linear = nn.Linear(hidden_size, 
                                patch_size**3 * out_channels, 
                                bias=True)
            
        # AdaLN conditioning for both heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Project hidden patch features into output patch or volume predictions.
        
        Args:
            x: Input features [B, N, hidden_size]
            c: Conditioning embedding [B, hidden_size]
            
        Returns:
            Output predictions in patch format or dense volume format, depending
            on `input_size` and `out_channels`.
        """
        if self.out_channels == 2:
            # Generate conditioning parameters for both heads
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            
            # Process through separate heads with different normalization
            h_noise = self.linear_noise(modulate(self.norm_noise(x), shift, scale))
            h_image = self.linear_image(modulate(self.norm_image(x), shift, scale))
            
            # If input_size is provided, unpatchify to volume format
            if self.input_size is not None:
                noise_vol = unpatchify_3d(h_noise, out_channels=1, 
                                        patch_size=self.patch_size, 
                                        input_size=self.input_size)
                image_vol = unpatchify_3d(h_image, out_channels=1, 
                                        patch_size=self.patch_size, 
                                        input_size=self.input_size)
                return torch.cat([noise_vol, image_vol], dim=1)
            else:
                # Return patch format
                return torch.cat([h_noise, h_image], dim=-1)
        else:
            # Single head for other configurations
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            return self.linear(modulate(self.norm_image(x), shift, scale))

class MlpDenoiser(nn.Module):
    """
    Patchwise MLP denoiser used in the coarse PRDiT path.

    This module performs lightweight denoising directly in patch space using two
    SwiGLU blocks and timestep-conditioned modulation. It is the main workhorse
    for stage-1 coarse denoising.
    
    Args:
        input_size: Target cubic volume size used for unpatchifying outputs
        hidden_size: Feature dimension (input/output)
        patch_size: Size of output patches
        out_channels: Number of output channels
        mlp_ratio: Hidden layer expansion ratio
        swiglu_mlp: Retained for compatibility with older constructor calls
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 patch_size: int, 
                 out_channels: int, 
                 mlp_ratio: float = 1.0,
                 swiglu_mlp: bool = False):
        super().__init__()
        
        # Normalization layers (parameters controlled by conditioning)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # TODO: use RMSNorm for image head, use LayerNorm for noise head
        self.norm_image = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm_noise = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        
        # Two-layer MLP with SwiGLU activation
        hidden_features = int(hidden_size * mlp_ratio)
        self.patch_size = patch_size
        self.input_size = input_size
        
        self.mlp1 = SwiGLU(
            in_features=hidden_size,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
            drop=0,
        )
        
        self.mlp2 = SwiGLU(
            in_features=hidden_size,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
            drop=0,
        )
        
        # Final projection to patch space
        if out_channels == 2:
            self.linear_nos = nn.Linear(hidden_size, patch_size**3*1, bias=True)
            self.linear_img = nn.Linear(hidden_size, patch_size**3*1, bias=True)
        else:
            self.linear_final = nn.Linear(hidden_size, patch_size**3 * out_channels, 
                                          bias=True)
        
        # AdaLN conditioning (6 params: 2 layers × 3 params each)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Denoise patch tokens with timestep-conditioned MLP blocks.
        
        Args:
            x: Input patches [B, N, hidden_size]
            c: Conditioning embedding [B, hidden_size]
            
        Returns:
            Denoised output volume in channel-first 3D format.
        """
        # Generate conditioning parameters
        shift1, scale1, shift2, scale2, shift, scale = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Two-layer MLP with conditioning
        h = self.mlp1(modulate(self.norm1(x), shift1, scale1))
        h = self.mlp2(modulate(self.norm2(h), shift2, scale2))
        
        # TODO: use RMSNorm for image head, use LayerNorm for noise head
        h = h + x
        #return self.linear_final(modulate(self.norm3(h + x), shift3, scale3))
        h_img = self.linear_img(modulate(self.norm_image(h),   shift, scale))
        h_noise = self.linear_nos(modulate(self.norm_noise(h), shift, scale))
        # return torch.cat([h_noise, h_img], dim=-1)
        img = unpatchify_3d(h_img, out_channels=1, 
                            patch_size=self.patch_size, 
                            input_size=self.input_size)
        noise = unpatchify_3d(h_noise, out_channels=1, 
                              patch_size=self.patch_size, 
                              input_size=self.input_size)
        return torch.cat([noise, img], dim=1)

class CoarseDenoiser(nn.Module):
    """
    Coarse denoising branch of PRDiT.

    This branch extracts 3D patches from the input volume and denoises them with
    the lightweight MLP denoiser. It supplies the base reconstruction used both
    in standalone coarse models and in the staged hybrid setting.
    
    Args:
        in_channels: Number of input channels
        extract_patch_size: Size of patches to extract from input
        patch_size: Size of output patches
        out_channels: Number of output channels
        input_size: Size of input volume (cubic)
        stride: Stride for patch extraction
        padding: Padding for patch extraction
        mlp_ratio: MLP hidden layer expansion ratio
        swiglu_mlp: Whether to use SwiGLU activation
    """
    
    def __init__(self,
                 in_channels: int,
                 extract_patch_size: int,
                 patch_size: int,
                 out_channels: int,
                 input_size: int,
                 stride: int = 4,
                 padding: int = 2,
                 mlp_ratio: float = 1.0,
                 swiglu_mlp: bool = True):
        super().__init__()
        
        # Patch extraction
        self.patch_extractor = ExtractPatches3D(
            patch_size=extract_patch_size,
            stride=stride,
            padding=padding,
        )
        
        # Calculate patch grid dimensions
        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(input_size)
        
        # MLP denoiser with adaptive conditioning
        input_dim = in_channels * extract_patch_size**3
        self.mlp_denoise = MlpDenoiser(
            input_size=input_size,
            hidden_size=input_dim,
            patch_size=patch_size,
            out_channels=out_channels,
            swiglu_mlp=swiglu_mlp,
            mlp_ratio=mlp_ratio
        )
    
    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor, 
                return_patches: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the coarse PRDiT branch on an input volume.
        
        Args:
            x: Input volume [B, C, D, H, W]
            c: Timestep conditioning [B, hidden_size]
            return_patches: If True, return both input patches and output
            
        Returns:
            The coarse output volume, or `(input_patches, coarse_output)` when
            `return_patches=True`.
        """
        # Extract patches from input volume
        patches = self.patch_extractor(x)  # [B, N, C * extract_patch_size^3]
        
        # Process through MLP denoiser with conditioning
        denoised = self.mlp_denoise(patches, c)
        
        return (patches, denoised) if return_patches else denoised

class FineRefiner(nn.Module):
    """
    Transformer-based refinement branch of PRDiT.

    This branch takes patch tokens, adds positional information, and refines
    the coarse representation with attention-based blocks. It is intended to
    model the higher-frequency residual detail that the coarse MLP path misses.
    
    Args:
        in_channels: Number of input channels
        extract_patch_size: Size of input patches
        hidden_size: Transformer hidden dimension
        patch_size: Size of output patches
        out_channels: Number of output channels  
        depth: Number of transformer layers
        num_heads: Number of attention heads
        num_patches: Total number of patches (for compatibility)
        input_size: Size of input volume (cubic)
        stride: Stride for patch processing
        padding: Padding for patch processing
        mlp_ratio: MLP expansion ratio in transformer blocks
        flash_attn: Whether to use flash attention
    """
    
    def __init__(self,
                 in_channels: int,
                 extract_patch_size: int,
                 hidden_size: int,
                 patch_size: int,
                 out_channels: int,
                 depth: int,
                 num_heads: int,
                 num_patches: int,  # For compatibility
                 input_size: int,
                 stride: int = 4,
                 padding: int = 2,
                 mlp_ratio: float = 4.0,
                 flash_attn: bool = False):
        super().__init__()
        
        logger.debug(f"FineRefiner: depth={depth}, hidden_size={hidden_size}, num_heads={num_heads}")
        
        # Patch embedding for input projection
        self.patch_embedder = PatchEmbed3D(
            patch_size=extract_patch_size,
            in_chans=in_channels, 
            embed_dim=hidden_size, 
            norm_layer=nn.LayerNorm,
            activation=nn.GELU(approximate="tanh")
        )
        
        # Fixed positional embeddings
        grid_size = input_size // patch_size
        pos_embed = get_normalized_3d_pos_enc(grid_size=grid_size, embed_dim=hidden_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0), persistent=False)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            PRDiTBlock(hidden_size, 
                       num_heads, 
                       mlp_ratio=mlp_ratio, 
                       flash_attn=flash_attn)
            for _ in range(depth)
        ])
        
        # Final output projection
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, input_size)
        
        # Store configuration
        self.input_size = input_size
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Refine patch tokens with positional encoding and conditioned transformer blocks.
        
        Args:
            x: Input patch sequence [B, N, C * extract_patch_size^3]
            c: Timestep conditioning [B, hidden_size]
            
        Returns:
            Refined output in the format produced by `FinalLayer`.
        """
        # Project patches to transformer dimension
        h = self.patch_embedder(x)  # [B, N, hidden_size]
        
        # Add positional encoding
        h = h + self.pos_embed
        
        # Process through transformer blocks with conditioning
        for block in self.blocks:
            h = block(h, c)
            
        # Final projection to output space
        return self.final_layer(h, c)

# =============================================================================
# Main PRDiT Model Architecture
# =============================================================================
class PRDiT(nn.Module):
    """
    Main PRDiT architecture for 3D diffusion-based volume modeling.

    PRDiT combines two complementary branches:

    - a coarse MLP denoiser that works directly on extracted 3D patches
    - a fine transformer refiner that learns residual corrections on top of the
      coarse prediction

    Training is staged by design:

    - Stage 1 (`depth=0`): train only the coarse branch
    - Stage 2 (`depth>0`): freeze the coarse branch and train the fine refiner

    This structure makes optimization easier on high-resolution 3D data while
    still allowing the model to recover fine detail later in training.
    
    Args:
        input_size: Size of cubic input volume
        patch_size: Size of cubic patches for processing
        stride: Stride for patch extraction
        padding: Padding for patch extraction
        in_channels: Number of input channels (e.g., 1 for CT)
        hidden_size: Hidden feature dimension
        depth: Number of transformer layers (0=MLP-only, >0=hybrid)
        num_heads: Number of attention heads in transformer
        mlp_ratio: MLP hidden dimension expansion ratio
        class_dropout_prob: Dropout probability for class conditioning
        num_classes: Number of conditioning classes
        learn_sigma: Whether to predict noise variance
        flash_attn: Whether to use optimized flash attention
    """
    def __init__(self,
                 input_size: int = 32,
                 patch_size: int = 2,
                 stride: int = 4,
                 padding: int = 2,
                 in_channels: int = 1,
                 hidden_size: int = 1152,
                 depth: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 class_dropout_prob: float = 0.1,
                 num_classes: int = 1,
                 learn_sigma: bool = False,
                 flash_attn: bool = False):
        super().__init__()
        
        # =====================================================================
        # Model Configuration
        # =====================================================================
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = stride  # Note: using stride as effective patch size
        self.depth = depth  
        self.hidden_size = hidden_size
        
        # Log model configuration (will be logged when rank is passed from trainer)
        self._config_to_log = {
            'input_size': input_size, 'patch_size': patch_size, 'stride': stride,
            'in_channels': in_channels, 'hidden_size': hidden_size, 'depth': depth,
            'num_heads': num_heads, 'learn_sigma': learn_sigma
        }
        
        # =====================================================================
        # Timestep Embedding (Shared with Dual Heads)
        # =====================================================================
        self.t_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            coarse_hidden_size=int(in_channels * patch_size**3),
            fine_hidden_size=hidden_size,
            frequency_embedding_size=256,
            is_depth_zero=(depth == 0)
        )
        
        # =====================================================================
        # Dual Denoising Paths
        # =====================================================================
        
        # Coarse Path: MLP-based denoising (always present)
        self.coarse = CoarseDenoiser(
            in_channels=in_channels,
            extract_patch_size=patch_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            input_size=input_size,
            stride=stride,
            padding=padding,
            mlp_ratio=1.0,
            swiglu_mlp=True
        )
        
        # Fine Path: Transformer-based refinement (only when depth > 0)
        self.fine = None
        if depth > 0:
            self.fine = FineRefiner(
                in_channels=in_channels,
                extract_patch_size=patch_size,
                hidden_size=hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                depth=depth,
                num_heads=num_heads,
                num_patches=self.coarse.num_patches,
                input_size=input_size,
                stride=stride,
                padding=padding,
                mlp_ratio=mlp_ratio,
                flash_attn=flash_attn
            )
        
        # =====================================================================
        # Weight Initialization and Training Setup
        # =====================================================================
        self.initialize_weights()
        
        # Stage 2 setup: freeze coarse path when training fine path
        if depth > 0:
            self.freeze_coarse_path()
            logger.info(f"Stage 2 setup: Coarse path frozen, training {depth} transformer layers")

    def log_config(self, rank: int = 0) -> None:
        """Log the stored model configuration from the primary process."""
        if rank == 0 and hasattr(self, '_config_to_log'):
            logger.info("PRDiT Model Configuration:")
            for key, value in self._config_to_log.items():
                logger.info(f"  {key}: {value}")
    
    def _log_config(self, config: dict, rank: int = 0) -> None:
        """Log an explicit configuration dictionary from the primary process."""
        if rank == 0:
            logger.info("PRDiT Model Configuration:")
            for key, value in config.items():
                logger.info(f"  {key}: {value}")

    def freeze_coarse_path(self) -> None:
        """
        Freeze the coarse branch and its timestep-conditioning path.

        This is used in stage-2 training so the transformer refiner can learn on
        top of a fixed coarse baseline.
        """
        requires_grad(self.coarse, False)
        requires_grad(self.t_embedder.coarse_head, False)
        requires_grad(self.t_embedder.mlp, False)
        
        # Count frozen vs trainable parameters
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Frozen {frozen_params:,} parameters, {trainable_params:,} trainable")

    def initialize_weights(self, gain: float = 1.0) -> None:
        """
        Initialize PRDiT weights with branch-specific schemes.

        The initialization uses Xavier-style defaults for most layers, then
        applies more constrained initialization to conditioning and output
        projections so training starts from a stable near-identity state.
        
        Args:
            gain: Scaling factor for Xavier initialization
        """
        logger.info("Initializing model weights...")
        
        def _init_linear_layers(module):
            """Initialize linear and conv layers with Xavier uniform."""
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_init_linear_layers)
        
        # Timestep embedder: careful initialization for stable training
        self._init_timestep_embedder()
        
        # Coarse path: zero-init output layers for stable start
        self._init_coarse_path()
        
        # Fine path: zero-init conditioning and outputs if present
        if self.depth > 0 and self.fine is not None:
            self._init_fine_path(gain)
            
        logger.info("Weight initialization complete")
    
    def _init_timestep_embedder(self) -> None:
        """Initialize the shared timestep MLP and its coarse/fine output heads."""
        # Shared MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None:
            nn.init.zeros_(self.t_embedder.mlp[0].bias)
            
        # Coarse and fine heads
        nn.init.normal_(self.t_embedder.coarse_head.weight, std=0.02)
        if self.t_embedder.coarse_head.bias is not None:
            nn.init.zeros_(self.t_embedder.coarse_head.bias)
            
        if not isinstance(self.t_embedder.fine_head, nn.Identity):
            nn.init.normal_(self.t_embedder.fine_head.weight, std=0.02)
            if self.t_embedder.fine_head.bias is not None:
                nn.init.zeros_(self.t_embedder.fine_head.bias)
    
    def _init_coarse_path(self) -> None:
        """Initialize the coarse branch so it starts from a stable zero-output regime."""
        # Zero-init conditioning modulation (starts with identity transformation)
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-1].bias, 0)
        
        # Zero-init final output layer (model starts by predicting zero)
        # Handle different output layer configurations
        if hasattr(self.coarse.mlp_denoise, 'linear_final'):
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.bias, 0)
        elif hasattr(self.coarse.mlp_denoise, 'linear_img'):
            # For dual-output models (out_channels=2)
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.bias, 0)
            if hasattr(self.coarse.mlp_denoise, 'linear_nos'):
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.weight, 0)
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.bias, 0)
    
    def _init_fine_path(self, gain: float) -> None:
        """Initialize the transformer refinement branch and its output head."""
        # Patch embedder: main path and skip connection
        if hasattr(self.fine, 'patch_embedder'):
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc1.weight, gain=gain)
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc2.weight, gain=gain)
            
            # Skip connection with smaller initialization
            nn.init.xavier_uniform_(self.fine.patch_embedder.skip.weight, gain=0.1)
            
            # Zero biases
            for layer in [self.fine.patch_embedder.fc1, self.fine.patch_embedder.fc2]:
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Transformer blocks: zero-init conditioning for stable start
        for block in self.fine.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Final layer: zero-init for identity start
        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].bias, 0)
        
        # Handle different output layer configurations
        if hasattr(self.fine.final_layer, 'linear'):
            nn.init.constant_(self.fine.final_layer.linear.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear.bias, 0)
        elif hasattr(self.fine.final_layer, 'linear_image'):
            # For dual-output models (out_channels=2)
            nn.init.constant_(self.fine.final_layer.linear_image.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear_image.bias, 0)
            if hasattr(self.fine.final_layer, 'linear_noise'):
                nn.init.constant_(self.fine.final_layer.linear_noise.weight, 0)
                nn.init.constant_(self.fine.final_layer.linear_noise.bias, 0)

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a dense 3D volume from patch-space predictions.
        
        Args:
            x: Tensor of shape [B, N, patch_size^3 * C] containing patch embeddings
            
        Returns:
            Reconstructed tensor of shape [B, C, D, H, W].
        """
        c = self.out_channels
        p = self.patch_size
        grid_size = self.input_size // p
        
        # Reshape to 3D grid
        x = x.reshape(-1, grid_size, grid_size, grid_size, p, p, p, c)
        
        # Permute to get channels first and combine spatial dimensions
        # [B, D', H', W', p, p, p, C] -> [B, C, D', p, H', p, W', p]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        
        # Merge grid and patch dimensions: [B, C, D'*p, H'*p, W'*p]
        return x.reshape(-1, c, grid_size * p, grid_size * p, grid_size * p)

    def forward(
        self, 
        input: torch.Tensor, 
        t: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the full PRDiT forward pass.

        When `depth == 0`, only the coarse branch is used. When `depth > 0`,
        the model first computes the coarse prediction, then adds transformer
        refinement on top.
        
        Args:
            input (torch.Tensor): Input volume [B, C, D, H, W]
            t (torch.Tensor): Timestep tensor [B]
            y (torch.Tensor, optional): Class conditioning
            return_intermediate (bool): Whether to return intermediate features
            
        Returns:
            The final denoised output, or `(coarse_output, fine_output)` when
            `return_intermediate=True`.
        """
        # Get timestep embeddings for both paths
        c_coarse, c_fine = self.t_embedder(t)  # [B, hidden_size] for both
        
        # Process through Coarse path (always runs)
        with torch.no_grad() if self.depth > 0 else torch.enable_grad():
            # Extract patches and process through MLP
            if self.depth > 0 or return_intermediate:
                patches, coarse_out = self.coarse(input, c_coarse, return_patches=True)
            else:
                coarse_out = self.coarse(input, c_coarse)
        
        # Process through Fine path if depth > 0
        if self.depth > 0 and self.fine is not None:
            # Get transformer refinements
            fine_out = self.fine(patches, c_fine)
            
            # Return intermediate outputs if requested
            if return_intermediate:
                return coarse_out, fine_out
            
            # Combine coarse and fine outputs
            # Both outputs are already in volume format [B, 2*C, D, H, W]
            x = coarse_out + fine_out
        else:
            # For MLP-only model (stage 1), use only coarse output
            x = coarse_out
        return x

    def load_coarse_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a stage-1 checkpoint and use it to initialize the coarse branch.
        
        Args:
            checkpoint_path: Path to a checkpoint containing coarse-path weights
        """
        # Load stage 1 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load with strict=False to only load matching parameters
        self.load_state_dict(checkpoint['model'], strict=False)
        
        # Freeze the coarse path
        self.freeze_coarse_path()
        logger.info(f"Loaded stage 1 checkpoint from {checkpoint_path} and froze coarse path")
