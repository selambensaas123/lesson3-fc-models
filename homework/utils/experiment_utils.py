"""Shared reproducible training routines."""
import json
import logging
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .model_utils import build

ROOT = Path(__file__).resolve().parents[1]


def data(name):
    """Partition the official training split and retain the official test."""
    source = {'mnist': datasets.MNIST, 'cifar': datasets.CIFAR10}[name]
    train, test = [source('./data', train=flag, download=True) for flag in (True, False)]
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(42))

    def tensors(dataset, indices):
        images = torch.as_tensor(dataset.data)[indices]
        if images.ndim == 3:
            images = images.unsqueeze(1)
        else:
            images = images.permute(0, 3, 1, 2)
        labels = torch.as_tensor(dataset.targets)[indices]
        return TensorDataset(images.contiguous().float().div_(255), labels)

    return (tensors(train, permutation[:12000]), tensors(train, permutation[12000:14000]),
            tensors(test, torch.arange(len(test))))


def epoch(model, loader, optimizer=None):
    """Return sample-weighted cross-entropy and accuracy."""
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            count += y.numel()
    return dict(loss=loss_sum / count, accuracy=correct / count)


def run(splits, widths, *, epochs=8, dropout=0., momentum=None, l2=0., adaptive=False):
    """Train one model, measuring the same fixed splits for every configuration."""
    torch.manual_seed(42)
    model = build(splits[0][0][0].numel(), widths, dropout, momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=l2)
    train, val, test = [DataLoader(split, batch_size=128, shuffle=i == 0,
                                   generator=torch.Generator().manual_seed(42))
                        for i, split in enumerate(splits)]
    history, seconds = [], 0.
    for number in range(epochs):
        if adaptive:
            for layer in model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = .1 + .4 * number / max(1, epochs - 1)
        start = perf_counter()
        train_scores = epoch(model, train, optimizer)
        seconds += perf_counter() - start
        row = dict(train=train_scores, val=epoch(model, val), test=epoch(model, test))
        logging.info('widths=%s epoch=%s %s', widths, number + 1, row)
        history.append(row)
    weights = [layer.weight.detach().flatten().tolist() for layer in model.modules()
               if isinstance(layer, nn.Linear)]
    return dict(widths=widths, params=sum(p.numel() for p in model.parameters()),
                seconds=seconds, history=history,
                layer_weight_std=[float(torch.tensor(w).std()) for w in weights]), weights


def save(group, rows):
    """Write machine-readable results for reporting."""
    folder = ROOT / 'results' / group
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


def setup():
    """Configure CPU threads and experiment logging."""
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO)
