"""Custom transformer sequence encoder for user history modeling."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_functional
from torch import nn


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a lower-triangular causal attention mask of shape [L, L]."""
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


def build_attention_bias(history_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build additive attention bias for padding + causal constraints.

    Output shape is [B, 1, L, L] and can be passed to scaled_dot_product_attention.
    """
    batch_size, seq_len = history_mask.shape
    causal = build_causal_mask(seq_len, history_mask.device).unsqueeze(0)
    query_valid = history_mask.unsqueeze(-1)
    key_valid = history_mask.unsqueeze(1)
    allowed = query_valid & key_valid & causal

    bias = torch.zeros((batch_size, seq_len, seq_len), dtype=dtype, device=history_mask.device)
    min_value = torch.finfo(dtype).min
    bias = bias.masked_fill(~allowed, min_value)
    return bias.unsqueeze(1)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (sequence * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            msg = "embedding_dim must be divisible by num_heads"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
        )

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = values.shape
        return values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, _heads, seq_len, _head_dim = values.shape
        return values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(self, hidden: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(hidden)
        q = self._split_heads(self.q_proj(attn_input))
        k = self._split_heads(self.k_proj(attn_input))
        v = self._split_heads(self.v_proj(attn_input))

        attn_bias = build_attention_bias(history_mask, q.dtype)
        attn_out = torch_functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn_out = self._merge_heads(attn_out)
        hidden = hidden + self.dropout(self.out_proj(attn_out))
        hidden = hidden * history_mask.unsqueeze(-1)

        ffn_input = self.norm2(hidden)
        hidden = hidden + self.dropout(self.ffn(ffn_input))
        hidden = hidden * history_mask.unsqueeze(-1)
        return hidden


class TransformerUserEncoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        history_length: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        sequence_pooling: str,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.sequence_pooling = sequence_pooling
        self.position_embeddings = nn.Embedding(history_length, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def _pool_sequence(self, hidden: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if self.sequence_pooling == "mean":
            return _masked_mean(hidden, history_mask)

        # last valid token pooling
        valid_counts = history_mask.long().sum(dim=1)
        last_index = (valid_counts - 1).clamp_min(0)
        gather_index = last_index.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        pooled = hidden.gather(1, gather_index).squeeze(1)

        has_history = valid_counts > 0
        pooled = torch.where(has_history.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return pooled

    def forward(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = history_embeddings.shape
        if seq_len > self.history_length:
            msg = "history length exceeds configured positional embedding range"
            raise ValueError(msg)

        positions = torch.arange(seq_len, device=history_embeddings.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        hidden = history_embeddings + self.position_embeddings(positions)
        hidden = self.input_dropout(hidden)
        hidden = hidden * history_mask.unsqueeze(-1)

        for block in self.blocks:
            hidden = block(hidden, history_mask)

        return self._pool_sequence(hidden, history_mask)
