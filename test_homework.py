"""Unit tests for the configurable fully connected classifiers."""
import unittest

import torch
from torch import nn

from homework.utils.model_utils import build


class ModelTests(unittest.TestCase):
    def test_depth_and_output(self):
        for depth in (1, 2, 3, 5, 7):
            model = build(16, [8] * (depth - 1))
            self.assertEqual(sum(isinstance(x, nn.Linear) for x in model.modules()), depth)
            self.assertEqual(model(torch.ones(4, 1, 4, 4)).shape, (4, 10))

    def test_regularization_configuration(self):
        model = build(16, [8, 4], dropout=.3, momentum=.01)
        self.assertEqual([x.p for x in model.modules() if isinstance(x, nn.Dropout)], [.3, .3])
        self.assertEqual([x.momentum for x in model.modules() if isinstance(x, nn.BatchNorm1d)],
                         [.01, .01])


if __name__ == '__main__':
    unittest.main()
