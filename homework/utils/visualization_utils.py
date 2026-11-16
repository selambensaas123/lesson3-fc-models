"""Render accuracy curves and comparison figures."""
from .experiment_utils import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(fig, name):
    """Write and close a figure."""
    folder = ROOT / 'plots'
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f'{name}.png', dpi=140)
    plt.close(fig)


def curves(rows, name):
    """Show training, validation and test accuracy separately."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, split in zip(axes, ('train', 'val', 'test')):
        for label, row in rows.items():
            axis.plot(range(1, 1 + len(row['history'])),
                      [step[split]['accuracy'] for step in row['history']], label=label)
        axis.set(title=split, xlabel='Epoch', ylabel='Accuracy')
    axes[-1].legend(fontsize=7)
    export(fig, name)
