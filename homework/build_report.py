"""Build the Russian report from completed experiment JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    """Load one experiment group."""
    return json.loads((ROOT / 'results' / path / 'metrics.json').read_text(encoding='utf-8'))


def table(rows):
    """Report the final epoch and the minimum validation loss epoch."""
    lines = ['| Модель | Параметры | Обучение, с | Train acc | Val acc | Test acc | Мин. val loss, эпоха |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        history = row['history']
        best = min(range(len(history)), key=lambda index: history[index]['val']['loss'])
        train, val, test = [history[-1][split]['accuracy'] for split in ('train', 'val', 'test')]
        lines.append(f"| {name} | {row['params']} | {row['seconds']:.2f} | "
                     f"{train:.4f} | {val:.4f} | {test:.4f} | {best + 1} |")
    return '\n'.join(lines)


def main():
    """Generate the report from experiment groups available on disk."""
    sections = [
        '# Урок 3. Полносвязные сети',
        '## Задание и цель работы\n\nСравнить глубину, ширину и регуляризацию классификаторов.',
        '## Использованные методы\n\nAdam (0,001), CrossEntropyLoss, батч 128, 8 эпох на конфигурацию.',
        '## Реализация\n\n[Протокол и команды](homework/README.md). Архитектура использует конфигурируемую модель преподавателя. Train accuracy измеряется на обучающих батчах в ходе прохода.',
        '## Эксперименты, результаты, графики и таблицы',
    ]
    for key, label, graph in [
        ('depth_experiments/mnist', 'Глубина на MNIST', 'depth_mnist'),
        ('depth_experiments/cifar', 'Глубина на CIFAR-10', 'depth_cifar'),
        ('width_experiments', 'Ширина и поиск по сетке', 'width_grid'),
        ('regularization_experiments', 'Регуляризация', 'regularization'),
    ]:
        try:
            rows = read(key)
        except FileNotFoundError:
            sections.append(f'### {label}\n\nИзмерения пока не завершены.')
            continue
        winner = max(rows, key=lambda name: rows[name]['history'][-1]['val']['accuracy'])
        best_row = rows[winner]
        best_history = best_row['history']
        onset = next((i + 1 for i, (a, b) in enumerate(zip(best_history, best_history[1:]), 1)
                      if b['train']['loss'] < a['train']['loss']
                      and b['val']['loss'] > a['val']['loss']), None)
        signal = (f'Первое расхождение train и val loss у выбранной модели — эпоха {onset}.'
                  if onset else 'Рост val loss одновременно со снижением train loss не наблюдался.')
        sections.append(f'### {label}\n\n{table(rows)}\n\n'
                        f'Лучший вариант по валидационной точности на последней эпохе — {winner}. '
                        f'{signal} [График](homework/plots/{graph}.png).')
    regularization = ROOT / 'results' / 'regularization_experiments' / 'metrics.json'
    if regularization.exists():
        rows = read('regularization_experiments')
        section = ['### Стабильность и распределения весов',
                   '[Распределения весов по линейным слоям](homework/plots/weights.png).',
                   '| Вариант | Размах val loss, эпохи 5–8 | Стандартное отклонение весов по слоям |',
                   '|---|---:|---|']
        for name, row in rows.items():
            tail = [item['val']['loss'] for item in row['history'][-4:]]
            spreads = ', '.join(f'{std:.4f}' for std in row['layer_weight_std'])
            section.append(f'| {name} | {max(tail) - min(tail):.4f} | {spreads} |')
        sections.append('\n\n'.join(section[:2]) + '\n\n' + '\n'.join(section[2:]))
    sections.append('## Выводы\n\nПри сравнении точности учитываются число параметров и время обучения. '
                    'Минимум валидационной потери с последующим ростом при снижении обучающей потери '
                    'служит признаком переобучения.')
    (ROOT.parent / 'REPORT.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
