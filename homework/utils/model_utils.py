"""Build classifiers using the lesson's configurable model."""
from torch import nn
from fully_connected_basics.models import FullyConnectedModel


def build(features, widths, dropout=0., momentum=None):
    """Create hidden layers and an automatic ten-class output."""
    layers = []
    for width in widths:
        layers.append(dict(type='linear', size=width))
        if momentum is not None:
            layers.append(dict(type='batch_norm'))
        layers.append(dict(type='relu'))
        if dropout:
            layers.append(dict(type='dropout', rate=dropout))
    model = FullyConnectedModel(input_size=features, layers=layers)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm1d):
            layer.momentum = momentum
    return model
