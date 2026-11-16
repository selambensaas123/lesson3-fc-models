"""Compare fixed and scheduled regularization with matched hidden layers."""
from .utils.experiment_utils import data, run, save, setup
from .utils.visualization_utils import curves, export, plt


def main():
    setup()
    splits = data('mnist')
    configs = {
        'baseline': {}, 'dropout .1': dict(dropout=.1), 'dropout .3': dict(dropout=.3),
        'dropout .5': dict(dropout=.5), 'BN .1': dict(momentum=.1),
        'BN+dropout': dict(momentum=.1, dropout=.3), 'L2': dict(l2=.001),
        'adaptive': dict(dropout=.1, adaptive=True), 'BN .01': dict(momentum=.01),
        'BN .5': dict(momentum=.5),
        'combined': dict(dropout=.1, adaptive=True, momentum=.1, l2=.001),
    }
    rows = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name, config in configs.items():
        rows[name], weights = run(splits, [256, 128, 64], **config)
        for number, (axis, weight) in enumerate(zip(axes.flat, weights), 1):
            axis.hist(weight, bins=60, histtype='step', density=True, label=name)
            axis.set_title(f'Linear layer {number}')
    axes.flat[-1].legend(fontsize=7)
    export(fig, 'weights')
    curves(rows, 'regularization')
    save('regularization_experiments', rows)


if __name__ == '__main__':
    main()
