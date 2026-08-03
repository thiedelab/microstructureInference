#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSet baseline model for Bragg-disk-table orientation regression.

This file is intended to be a drop-in alternative to transformerModel.py for
reviewer-requested ablation/comparison experiments.

Key design choice:
    - Keep the same per-disk embedding construction as transformerModel.py.
    - Replace Transformer self-attention blocks with independent per-disk MLP
      blocks, followed by masked set pooling and a set-level MLP head.
    - Keep the same output interface so the existing point-group geodesic loss
      and trainer can be reused.

Expected input:
    x:        LongTensor with shape [B, S, 3]
              columns are [radial_bin, angle_bin, intensity_bin]
    pad_mask: BoolTensor with shape [B, 1, 1, S], where True means padding.

Expected output:
    If config.multiTask is False:
        Tensor with shape [B, num_feature], normally num_feature = 9.
    If config.multiTask is True:
        (rotation_prediction, auxiliary_prediction)

For your current point-group rotation trainer, use config.multiTask = 0 and
config.num_feature = 9.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    d_embed: int
    # d_ff is the hidden dimension of the feed-forward layers.
    d_ff: int
    # h is kept only for compatibility with transformerModel.ModelConfig.
    # DeepSet does not use attention heads.
    angle_bin_centers: torch.Tensor
    intensity_bin_centers: torch.Tensor
    num_bins_radialDistance: int
    device: torch.device
    num_feature: int
    h: int
    N_encoder: int
    max_seq_len: int
    dropout: float
    multiTask: int


def make_model(config: ModelConfig) -> nn.Module:
    """Construct and Xavier-initialize the DeepSet baseline."""
    model = DeepSetModel(config, config.num_feature).to(config.device)

    # Match the initialization convention used in transformerModel.py.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class absolutePositionEmbedding(nn.Module):
    def __init__(self, num_bins: int, embed_dim: int, base: float = 5000.0):
        super().__init__()
        assert embed_dim % 2 == 0, "Sinusoidal embedding expects even d_embed."
        self.embed_dim = embed_dim
        self.base = base

        positions = torch.arange(num_bins).reshape(-1, 1)
        pos_embedding = self.generate_sinusoidal1D(positions)
        self.register_buffer("pos_embedding", pos_embedding)

    def generate_sinusoidal1D(self, sequence: torch.Tensor) -> torch.Tensor:
        denominator = torch.pow(
            torch.tensor(self.base, dtype=torch.float32),
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) / self.embed_dim,
        )
        pos_embedding = torch.zeros(1, sequence.shape[0], self.embed_dim)
        phase = sequence.float() / denominator
        pos_embedding[:, :, ::2] = torch.sin(phase)
        pos_embedding[:, :, 1::2] = torch.cos(phase)
        return pos_embedding.reshape(pos_embedding.shape[1], pos_embedding.shape[2])


class absolutePositionEmbedding_I(nn.Module):
    def __init__(self, num_bins: int, embed_dim: int, base: float = 10000.0):
        super().__init__()
        assert embed_dim % 2 == 0, "Sinusoidal embedding expects even d_embed."
        self.embed_dim = embed_dim
        self.base = base

        positions = torch.arange(num_bins).reshape(-1, 1)
        pos_embedding = self.generate_sinusoidal1D(positions)
        self.register_buffer("pos_embedding", pos_embedding)

    def generate_sinusoidal1D(self, sequence: torch.Tensor) -> torch.Tensor:
        denominator = torch.pow(
            torch.tensor(self.base, dtype=torch.float32),
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) / self.embed_dim,
        )
        pos_embedding = torch.zeros(1, sequence.shape[0], self.embed_dim)
        phase = sequence.float() / denominator
        pos_embedding[:, :, ::2] = torch.sin(phase)
        pos_embedding[:, :, 1::2] = torch.cos(phase)
        return pos_embedding.reshape(pos_embedding.shape[1], pos_embedding.shape[2])


class directioPositionEmbedding_A(nn.Module):
    def __init__(
        self,
        angle_bin_centers: torch.Tensor,
        embed_dim: int,
        device: torch.device,
        num_trainableVec: int = 31,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_trainableVec = num_trainableVec

        cosine_library, sine_library = self.generate_directional_library(angle_bin_centers)
        self.register_buffer("torch_cosine_angle_library", cosine_library.to(device=device))
        self.register_buffer("torch_sine_angle_library", sine_library.to(device=device))

        n_harmonics = int((num_trainableVec - 1) / 2)
        self.pos_embedding_learnable_bias = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
        self.pos_embedding_learnable_cosi = nn.Parameter(torch.zeros(n_harmonics, embed_dim), requires_grad=True)
        self.pos_embedding_learnable_sine = nn.Parameter(torch.zeros(n_harmonics, embed_dim), requires_grad=True)
        self.normalization_factor = float(n_harmonics)

    def generate_directional_library(self, angle_bin_centers: torch.Tensor):
        angle_bin_centers = angle_bin_centers.detach().float().cpu()
        cosine_terms = []
        sine_terms = []
        for k in range(1, int((self.num_trainableVec - 1) / 2) + 1):
            cosine_terms.append(torch.cos(k * angle_bin_centers))
            sine_terms.append(torch.sin(k * angle_bin_centers))
        cosine_terms = torch.vstack(cosine_terms)
        sine_terms = torch.vstack(sine_terms)
        return cosine_terms.permute(1, 0), sine_terms.permute(1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*S]
        cos_sliced = self.torch_cosine_angle_library[x]
        sin_sliced = self.torch_sine_angle_library[x]
        return (
            torch.matmul(cos_sliced, self.pos_embedding_learnable_cosi)
            + torch.matmul(sin_sliced, self.pos_embedding_learnable_sine)
        ) / self.normalization_factor + self.pos_embedding_learnable_bias


class EmbedLayer(nn.Module):
    """Same disk embedding logic as the current transformer model."""

    def __init__(
        self,
        embed_dim: int,
        angle_bin_centers: torch.Tensor,
        intensity_bin_centers: torch.Tensor,
        num_bins_radialDistance: int,
        device: torch.device,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.R_embedding = absolutePositionEmbedding(num_bins_radialDistance, embed_dim, base=5000.0)
        self.A_embedding = directioPositionEmbedding_A(angle_bin_centers, embed_dim, device)

        # Match current transformerModel.py: intensity uses absolute sinusoidal bin embedding,
        # not the older directional intensity embedding.
        self.I_embedding = absolutePositionEmbedding_I(intensity_bin_centers.shape[0], embed_dim, base=10000.0)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, 3]
        B = x.shape[0]
        S = x.shape[1]
        x_flat = x.reshape(B * S, x.shape[2])

        r_emb = self.R_embedding.pos_embedding[x_flat[:, 0]]
        a_emb = self.A_embedding(x_flat[:, 1])
        i_emb = self.I_embedding.pos_embedding[x_flat[:, 2]]

        out = r_emb + a_emb + i_emb
        out = out.reshape(B, S, r_emb.shape[1])
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class ResidualConnection(nn.Module):
    """Pre-norm residual MLP block: x + dropout(sublayer(layernorm(x)))."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.drop(sublayer(self.norm(x)))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_embed),
        )
        self.residual = ResidualConnection(d_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x, self.net)


class DeepSetEncoder(nn.Module):
    """
    DeepSet phi network.

    Unlike the Transformer encoder, these blocks act independently on each disk.
    Therefore, before pooling, there is no disk-disk communication. The only set
    operation is masked mean pooling in DeepSetModel.forward().
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = EmbedLayer(
            config.d_embed,
            config.angle_bin_centers,
            config.intensity_bin_centers,
            config.num_bins_radialDistance,
            config.device,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [FeedForwardBlock(config.d_embed, config.d_ff, config.dropout) for _ in range(config.N_encoder)]
        )
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del mask  # padding is handled during masked pooling, not inside per-disk MLP blocks.
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class DeepSetModel(nn.Module):
    def __init__(self, config: ModelConfig, num_feature: int):
        super().__init__()
        self.encoder = DeepSetEncoder(config)
        self.multiTask = bool(config.multiTask)

        # Set-level rho network after pooling. This is the MLP that acts on the
        # whole pooled diffraction-pattern representation.
        self.rho = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.LayerNorm(config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed),
            nn.LayerNorm(config.d_embed),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.MLP_head_0 = nn.Sequential(
            nn.Linear(config.d_embed, int(config.d_embed / 2)),
            nn.LayerNorm(int(config.d_embed / 2)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 2), int(config.d_embed / 4)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 4), int(config.d_embed / 8)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 8), num_feature),
        )

        if self.multiTask:
            self.MLP_head_1 = nn.Sequential(
                nn.Linear(config.d_embed, int(config.d_embed / 2)),
                nn.LayerNorm(int(config.d_embed / 2)),
                nn.GELU(),
                nn.Linear(int(config.d_embed / 2), int(config.d_embed / 4)),
                nn.GELU(),
                nn.Linear(int(config.d_embed / 4), int(config.d_embed / 8)),
                nn.GELU(),
                nn.Linear(int(config.d_embed / 8), 1),
            )

    def masked_mean_pool(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pad_mask is None:
            return torch.mean(x, dim=1)

        # Current trainer supplies pad_mask as [B, 1, 1, S], True for padding.
        valid_mask = torch.logical_not(pad_mask.reshape(pad_mask.shape[0], pad_mask.shape[-1], 1)).float()
        x = x * valid_mask
        valid_count = torch.sum(valid_mask, dim=1).clamp_min(1.0)
        return torch.sum(x, dim=1) / valid_count

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        x = self.encoder(x, pad_mask)
        x_final = self.masked_mean_pool(x, pad_mask)
        x_final = self.rho(x_final)

        if self.multiTask:
            return self.MLP_head_0(x_final), self.MLP_head_1(x_final)
        return self.MLP_head_0(x_final)


# Backward-compatible alias, in case older scripts expect a class called Transformer.
Transformer = DeepSetModel
