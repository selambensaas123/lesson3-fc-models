"""Compare depth and hidden-layer regularization on MNIST and CIFAR-10."""
from .utils.experiment_utils import data, run, save, setup
from .utils.visualization_utils import curves


def main(names=('mnist', 'cifar')):
    setup()
    for name in names:
        splits = data(name)
        rows = {}
        for depth in (1, 2, 3, 5, 7):
            rows[str(depth)], _ = run(splits, [128] * (depth - 1))
            if depth > 1:
                rows[f'{depth} BN+DO'], _ = run(splits, [128] * (depth - 1),
                                               dropout=.3, momentum=.1)
        save('depth_experiments/' + name, rows)
        curves(rows, 'depth_' + name)


if __name__ == '__main__':
    main()
