"""Small byte-level Prefix-LM model for BLT-GEC pipeline smoke training.

This model is intentionally lightweight and repo-local. It exercises the same
byte Prefix-LM data path that the official BLT wrapper will use, while avoiding
the heavy bytelatent/xformers/HF-gated dependency chain during early pipeline
validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from blt_gec.data_adapter import PAD_ID, VOCAB_SIZE


@dataclass
class BytePrefixTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_length: int = 1024
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ffn_dim: int = 1024


class BytePrefixTransformerLM(nn.Module):
    """Causal byte Transformer used as the first trainable BLT-GEC scaffold."""

    def __init__(self, config: BytePrefixTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            raise ValueError(f"seq_len={seq_len} exceeds max_length={self.config.max_length}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()

        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

