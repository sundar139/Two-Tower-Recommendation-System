from __future__ import annotations

import torch

from movie_recsys.modeling.augmentations import apply_sequence_augmentations


def test_augmentations_preserve_shape_and_padding_semantics() -> None:
    torch.manual_seed(123)
    history = torch.tensor(
        [
            [0, 0, 2, 3, 4],
            [0, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    mask = history > 0
    target = torch.tensor([4, 8], dtype=torch.long)

    aug_history, aug_mask = apply_sequence_augmentations(
        history,
        mask,
        target_item_idx=target,
        mask_prob=0.2,
        dropout_prob=0.2,
        crop_min_ratio=0.6,
        reorder_prob=0.5,
        reorder_window=3,
    )

    assert aug_history.shape == history.shape
    assert aug_mask.shape == mask.shape
    assert aug_mask.dtype == torch.bool
    assert torch.equal(aug_history[~aug_mask], torch.zeros_like(aug_history[~aug_mask]))


def test_target_item_excluded_when_present() -> None:
    torch.manual_seed(7)
    history = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    mask = history > 0
    target = torch.tensor([4], dtype=torch.long)

    aug_history, aug_mask = apply_sequence_augmentations(
        history,
        mask,
        target_item_idx=target,
        mask_prob=0.0,
        dropout_prob=0.0,
        crop_min_ratio=1.0,
        reorder_prob=0.0,
        reorder_window=2,
    )

    assert int(target.item()) not in aug_history[aug_mask].tolist()


def test_target_only_history_does_not_reintroduce_target() -> None:
    history = torch.tensor([[0, 0, 0, 0, 4]], dtype=torch.long)
    mask = history > 0
    target = torch.tensor([4], dtype=torch.long)

    aug_history, aug_mask = apply_sequence_augmentations(
        history,
        mask,
        target_item_idx=target,
        mask_prob=0.0,
        dropout_prob=0.0,
        crop_min_ratio=1.0,
        reorder_prob=0.0,
        reorder_window=2,
    )

    assert not bool(aug_mask.any())
    assert torch.equal(aug_history, torch.zeros_like(aug_history))


def test_non_empty_history_does_not_collapse_to_empty() -> None:
    torch.manual_seed(42)
    history = torch.tensor(
        [
            [0, 0, 2, 3, 4],
            [0, 0, 0, 0, 5],
        ],
        dtype=torch.long,
    )
    mask = history > 0

    aug_history, aug_mask = apply_sequence_augmentations(
        history,
        mask,
        target_item_idx=None,
        mask_prob=1.0,
        dropout_prob=1.0,
        crop_min_ratio=0.2,
        reorder_prob=1.0,
        reorder_window=3,
    )

    assert bool(aug_mask[0].any())
    assert bool(aug_mask[1].any())
    assert int(aug_history[0][aug_mask[0]].numel()) >= 1
    assert int(aug_history[1][aug_mask[1]].numel()) >= 1


def test_empty_history_remains_empty() -> None:
    history = torch.zeros((2, 5), dtype=torch.long)
    mask = torch.zeros((2, 5), dtype=torch.bool)

    aug_history, aug_mask = apply_sequence_augmentations(
        history,
        mask,
        target_item_idx=None,
        mask_prob=0.5,
        dropout_prob=0.5,
        crop_min_ratio=0.7,
        reorder_prob=0.5,
        reorder_window=3,
    )

    assert not bool(aug_mask.any())
    assert torch.equal(aug_history, torch.zeros_like(aug_history))
