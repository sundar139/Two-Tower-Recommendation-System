"""Sequence augmentations for CL-style user history contrastive views."""

from __future__ import annotations

import torch


def _pack_tokens(
    tokens: torch.Tensor,
    *,
    history_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.zeros((history_length,), dtype=dtype, device=device)
    packed_mask = torch.zeros((history_length,), dtype=torch.bool, device=device)
    if tokens.numel() == 0:
        return packed, packed_mask
    clipped = tokens[-history_length:]
    packed[-clipped.numel() :] = clipped
    packed_mask[-clipped.numel() :] = True
    return packed, packed_mask


def _mask_tokens(tokens: torch.Tensor, *, mask_prob: float) -> torch.Tensor:
    if tokens.numel() <= 1 or mask_prob <= 0.0:
        return tokens
    keep = torch.rand(tokens.numel(), device=tokens.device) >= mask_prob
    if not torch.any(keep):
        keep[torch.randint(tokens.numel(), (1,), device=tokens.device)] = True
    return tokens[keep]


def _drop_tokens(tokens: torch.Tensor, *, dropout_prob: float) -> torch.Tensor:
    if tokens.numel() <= 1 or dropout_prob <= 0.0:
        return tokens
    keep = torch.rand(tokens.numel(), device=tokens.device) >= dropout_prob
    if not torch.any(keep):
        keep[torch.randint(tokens.numel(), (1,), device=tokens.device)] = True
    return tokens[keep]


def _crop_tokens(tokens: torch.Tensor, *, min_ratio: float) -> torch.Tensor:
    if tokens.numel() <= 1:
        return tokens
    safe_ratio = min(max(min_ratio, 0.0), 1.0)
    min_keep = max(1, int(torch.ceil(torch.tensor(tokens.numel() * safe_ratio)).item()))
    if min_keep >= tokens.numel():
        return tokens
    span = int(
        torch.randint(min_keep, tokens.numel() + 1, (1,), device=tokens.device).item()
    )
    max_start = tokens.numel() - span
    start = 0 if max_start <= 0 else int(
        torch.randint(0, max_start + 1, (1,), device=tokens.device).item()
    )
    return tokens[start : start + span]


def _reorder_tokens(
    tokens: torch.Tensor,
    *,
    reorder_prob: float,
    reorder_window: int,
) -> torch.Tensor:
    if tokens.numel() <= 2 or reorder_prob <= 0.0 or reorder_window <= 1:
        return tokens
    if float(torch.rand(1, device=tokens.device).item()) > reorder_prob:
        return tokens

    window = min(int(reorder_window), int(tokens.numel()))
    if window <= 1:
        return tokens

    max_start = tokens.numel() - window
    start = 0 if max_start <= 0 else int(
        torch.randint(0, max_start + 1, (1,), device=tokens.device).item()
    )
    reordered = tokens.clone()
    perm = torch.randperm(window, device=tokens.device)
    reordered[start : start + window] = reordered[start : start + window][perm]
    return reordered


def apply_sequence_augmentations(
    history_item_idx: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    target_item_idx: torch.Tensor | None,
    mask_prob: float,
    dropout_prob: float,
    crop_min_ratio: float,
    reorder_prob: float,
    reorder_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an augmented history while preserving [B, L] shape and mask semantics."""
    batch_size, history_length = history_item_idx.shape
    augmented_history = torch.zeros_like(history_item_idx)
    augmented_mask = torch.zeros_like(history_mask)

    for row_idx in range(batch_size):
        valid_tokens = history_item_idx[row_idx][history_mask[row_idx]]
        if valid_tokens.numel() == 0:
            continue

        original_tokens = valid_tokens
        valid_tokens = _mask_tokens(valid_tokens, mask_prob=mask_prob)
        valid_tokens = _drop_tokens(valid_tokens, dropout_prob=dropout_prob)
        valid_tokens = _crop_tokens(valid_tokens, min_ratio=crop_min_ratio)
        valid_tokens = _reorder_tokens(
            valid_tokens,
            reorder_prob=reorder_prob,
            reorder_window=reorder_window,
        )

        if target_item_idx is not None:
            valid_tokens = valid_tokens[valid_tokens != target_item_idx[row_idx]]

        # Avoid all-empty augmentations unless filtering to prevent leakage makes it unavoidable.
        if valid_tokens.numel() == 0 and original_tokens.numel() > 0:
            if target_item_idx is not None:
                non_target_tokens = original_tokens[original_tokens != target_item_idx[row_idx]]
                if non_target_tokens.numel() > 0:
                    valid_tokens = non_target_tokens[-1:]
            else:
                valid_tokens = original_tokens[-1:]

        packed, packed_mask = _pack_tokens(
            valid_tokens,
            history_length=history_length,
            dtype=history_item_idx.dtype,
            device=history_item_idx.device,
        )
        augmented_history[row_idx] = packed
        augmented_mask[row_idx] = packed_mask

    return augmented_history, augmented_mask
