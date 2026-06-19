"""Tests for WeightedSFTTrainer.compute_loss shape handling."""

import pytest
import torch

from src.training.trainers.sft_trainer import WeightedSFTTrainer


class DummyModel(torch.nn.Module):
    """Minimal model that returns fixed logits for loss tests."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, **kwargs):
        class Out:
            pass

        out = Out()
        out.logits = self.logits
        return out


class TestWeightedSFTTrainer:
    def test_compute_loss_2d_labels_and_weights(self):
        """Trainer should flatten label/weight masks before indexing losses."""
        batch_size, seq_len, vocab = 2, 5, 8
        logits = torch.randn(batch_size, seq_len, vocab)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        # Only last two tokens of each sequence are assistant tokens.
        labels[:, -2:] = labels[:, -2:].clone()
        labels[:, -2] = 3
        labels[:, -1] = 5
        loss_weight = torch.ones_like(labels, dtype=torch.float)

        trainer = WeightedSFTTrainer(model=DummyModel(logits), args=None)
        loss = trainer.compute_loss(
            trainer.model,
            {"input_ids": torch.zeros_like(labels), "labels": labels, "loss_weight": loss_weight},
        )
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_compute_loss_no_weights(self):
        """Trainer should work when loss_weight is absent."""
        batch_size, seq_len, vocab = 1, 4, 6
        logits = torch.randn(batch_size, seq_len, vocab)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[:, -1] = 2

        trainer = WeightedSFTTrainer(model=DummyModel(logits), args=None)
        loss = trainer.compute_loss(
            trainer.model,
            {"input_ids": torch.zeros_like(labels), "labels": labels},
        )
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_compute_loss_all_masked(self):
        """Trainer should return zero loss when all labels are masked."""
        batch_size, seq_len, vocab = 1, 4, 6
        logits = torch.randn(batch_size, seq_len, vocab)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        trainer = WeightedSFTTrainer(model=DummyModel(logits), args=None)
        loss = trainer.compute_loss(
            trainer.model,
            {"input_ids": torch.zeros_like(labels), "labels": labels},
        )
        assert loss.item() == pytest.approx(0.0)
