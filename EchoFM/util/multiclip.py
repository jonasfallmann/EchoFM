"""
Multiclip encoder wrapper for EchoFM MAE models.

Processes multiple video clips through the frozen encoder and concatenates
the resulting token sequences for downstream probe evaluation.
Modeled after V-JEPA's ClipAggregation (ref_multiclip.py).

Usage:
    encoder = load_encoder(...)
    multiclip_encoder = MulticlipEncoder(encoder)
    # x = [clip_1_tensor, clip_2_tensor, ...] where each is [B, C, T, H, W]
    tokens = multiclip_encoder(x, clip_indices)
"""

import torch
import torch.nn as nn
import numpy as np


def get_1d_sincos_pos_embed(embed_dim: int, length: int, temperature: float = 10000.0) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embeddings (same as V-JEPA).

    Args:
        embed_dim: Embedding dimension (must be even).
        length: Number of positions.
        temperature: Scaling factor for frequency.

    Returns:
        np.ndarray of shape [length, embed_dim].
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (temperature ** omega)

    pos = torch.arange(length, dtype=torch.float32)
    out = torch.einsum("l,d->ld", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    return torch.cat([emb_sin, emb_cos], dim=1).numpy()


class MulticlipEncoder(nn.Module):
    """
    Wraps an EchoFM MaskedAutoencoderViT to process multiple clips per video.

    Takes a list of clip tensors (each [B, C, T_frames, H, W]), runs each
    through the frozen encoder independently with mask_ratio=0, and concatenates
    the resulting token sequences along the temporal dimension.

    Args:
        encoder: Frozen MaskedAutoencoderViT instance.
        use_temporal_pos_embed: If True, add learned 1D sinusoidal temporal
            position embeddings based on clip frame indices.
        max_frames: Max number of video frames for pos embed buffer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        use_temporal_pos_embed: bool = False,
        max_frames: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.tubelet_size = encoder.patch_embed.t_patch_size
        self.embed_dim = encoder.embed_dim
        self.num_heads = getattr(encoder, 'num_heads', None)
        self.use_temporal_pos_embed = use_temporal_pos_embed

        # 1D temporal positional embedding (learned, but frozen with sincos init)
        self.temporal_pos_embed = None
        if use_temporal_pos_embed:
            max_T = max_frames // self.tubelet_size
            self.temporal_pos_embed = nn.Parameter(
                torch.zeros(1, max_T, self.embed_dim), requires_grad=False
            )
            sincos = get_1d_sincos_pos_embed(self.embed_dim, max_T)
            self.temporal_pos_embed.copy_(
                torch.from_numpy(sincos).float().unsqueeze(0)
            )

    def forward(self, x, clip_indices=None):
        """
        Args:
            x: List of clip tensors, each of shape [B, C, T, H, W].
               Length of list = num_clips.
            clip_indices: Optional list of frame-index tensors, each [B, T].
               Length of list = num_clips. Used only when
               use_temporal_pos_embed=True.

        Returns:
            tokens: Tensor of shape [B, num_clips * N_tokens_per_clip, D].
        """
        num_clips = len(x)
        B, C, T, H, W = x[0].shape

        # ---- Concat all clips along batch dimension ----
        x_cat = torch.cat(x, dim=0)  # [B * num_clips, C, T, H, W]

        # ---- Run through frozen encoder (no masking) ----
        latent, mask, ids_restore = self.encoder.forward_encoder(
            x_cat, mask_ratio=0.0
        )
        # latent: [B * num_clips, N_tokens, D]

        N_tok = latent.shape[1]   # tokens per clip
        D = latent.shape[-1]

        # ---- Reshape to [B, num_clips * N_tokens, D] ----
        latent = latent.view(num_clips, B, N_tok, D)
        latent = latent.permute(1, 0, 2, 3).reshape(B, num_clips * N_tok, D)

        # ---- Optionally add temporal position embeddings ----
        if self.use_temporal_pos_embed and clip_indices is not None:
            # Determine number of temporal tokens per clip
            T_tok = self.encoder.patch_embed.input_size[0]  # temporal tokens
            S = N_tok // T_tok  # spatial tokens per temporal position

            all_pos = []
            for i in range(num_clips):
                indices = clip_indices[i]  # [B, T] frame indices
                # Map frame indices to tubelet indices
                tubelet_idx = indices[:, :: self.tubelet_size]  # [B, T_tok]

                # Gather position embeddings (loop for clarity)
                pos_list = []
                for b in range(B):
                    pos_list.append(
                        self.temporal_pos_embed[0, tubelet_idx[b], :]
                    )
                pos = torch.stack(pos_list, dim=0)  # [B, T_tok, D]

                # Expand to spatial tokens
                pos = pos.unsqueeze(2).expand(-1, -1, S, -1)  # [B, T_tok, S, D]
                pos = pos.reshape(B, T_tok * S, D)
                all_pos.append(pos)

            total_pos = torch.cat(all_pos, dim=1)  # [B, num_clips * N_tok, D]
            latent = latent + total_pos

        return latent
