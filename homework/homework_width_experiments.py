"""Compare three hidden-layer widths and search width schedules."""
from .utils.experiment_utils import data, run, save, setup
from .utils.visualization_utils import curves, export, plt


def main():
    setup()
    splits = data('mnist')
    widths = [[64, 32, 16], [256, 128, 64], [1024, 512, 256], [2048, 1024, 512]]
    rows = {str(w): run(splits, w)[0] for w in widths}
    curves(rows, 'width')
    schemes = ('decreasing', 'increasing', 'constant')
    grid = []
    for w in widths:
        scores = []
        for name, shape in zip(schemes, (w, w[::-1], [w[1]] * 3)):
            key = str(shape)
            if key not in rows:
                rows[key], _ = run(splits, shape)
            scores.append(rows[key]['history'][-1]['val']['accuracy'])
        grid.append(scores)
    fig, ax = plt.subplots()
    image = ax.imshow(grid)
    ax.set_xticks(range(3), schemes)
    ax.set_yticks(range(4), [str(w[1]) for w in widths])
    fig.colorbar(image, label='Final validation accuracy')
    export(fig, 'width_grid')
    save('width_experiments', rows)


if __name__ == '__main__':
    main()
