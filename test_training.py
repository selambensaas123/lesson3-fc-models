"""Check training and evaluation on a small deterministic classification task."""
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from homework.utils.experiment_utils import epoch


class TrainingTests(unittest.TestCase):
    def test_training_updates_parameters_and_evaluation_does_not(self):
        torch.manual_seed(42)
        x = torch.tensor([[-1.], [1.], [-2.], [2.]])
        y = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        model = nn.Linear(1, 2)
        before = epoch(model, loader)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        for _ in range(8):
            epoch(model, loader, optimizer)
        snapshots = [p.detach().clone() for p in model.parameters()]
        after = epoch(model, loader)
        self.assertLess(after['loss'], before['loss'])
        self.assertEqual(after['accuracy'], 1.)
        for parameter, snapshot in zip(model.parameters(), snapshots):
            torch.testing.assert_close(parameter, snapshot)
