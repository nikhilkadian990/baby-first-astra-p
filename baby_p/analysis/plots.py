"""Headless plots with seed-level uncertainty, never timestep-level error bars."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from baby_p.analysis.metrics import mean_interval, read_jsonl


def line_plot(path, rows, x, y, label, title, ylabel, chance=False):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if y in row and row[y] is not None:
            grouped[(label(row), row[x])][row['seed']].append(float(row[y]))
    series = defaultdict(list)
    for (name, value), seeds in grouped.items():
        interval = mean_interval([np.mean(v) for v in seeds.values()], repetitions=1000)
        series[name].append((value, interval))
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, points in sorted(series.items()):
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        means = [p[1]['mean'] for p in points]
        artist, = ax.plot(xs, means, marker='o', markersize=3, label=name)
        if all(p[1]['low'] is not None for p in points):
            ax.fill_between(xs, [p[1]['low'] for p in points], [p[1]['high'] for p in points],
                            color=artist.get_color(), alpha=0.15)
    if chance:
        ax.axhline(0.5, color='black', linestyle=':', label='chance')
    if series:
        ax.legend(fontsize=7, loc='best')
    else:
        ax.text(0.5, 0.5, 'No matching completed measurements', ha='center', transform=ax.transAxes)
    ax.set(xlabel=x.replace('_', ' '), ylabel=ylabel, title=title)
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.01, 'Bands: pointwise seed-bootstrap 95% intervals; omitted for one seed. Exploratory.',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(root):
    root = Path(root)
    with (root / 'analysis.json').open() as handle:
        data = json.load(handle)
    summaries, curves, resources = data['summaries'], data['curves'], data['resources']
    directory = root / 'plots'
    directory.mkdir(exist_ok=True)
    primary = [r for r in curves if r['adaptation_family'] == 'swapped_roles' and r['horizon'] == 1]
    late_age = max((r['age'] for r in curves), default=0)
    line_plot(directory / '01_development_vs_adaptation_cost.png', primary, 'age', 'restricted_cost',
              lambda r: r['variant'], 'Developmental experience vs restricted adaptation cost',
              'Interactions to criterion (censored values capped at budget)')
    line_plot(directory / '01b_censoring.png', primary, 'age', 'censored', lambda r: r['variant'],
              'Fraction of seeds not attaining criterion', 'Right-censored fraction')
    line_plot(directory / '02_checkpoint_vs_transfer.png',
              [r for r in curves if r['variant'] == 'p' and r['horizon'] == 1], 'age', 'initial_accuracy',
              lambda r: r['adaptation_family'], 'Checkpoint experience vs zero-shot held-out transfer',
              'Contact discrimination accuracy', chance=True)
    lifetime = []
    for run in sorted(root.glob('*/seed_*')):
        events = {}
        for log in sorted(run.glob('lifetime*.jsonl')):
            for event in read_jsonl(log):
                events[event['interactions']] = event
        for event in events.values():
            if not event['losses']:
                continue
            # Fixed interaction bins reduce visual noise; these are not replicates.
            bin_id = (event['interactions'] // 100) * 100
            for loss in event['losses']:
                lifetime.append(dict(seed=int(run.name.split('_')[-1]), variant=run.parent.name,
                                     interactions=bin_id, prediction_loss=loss['prediction_loss']))
    line_plot(directory / '03_lifetime_prediction_loss.png', lifetime, 'interactions', 'prediction_loss',
              lambda r: r['variant'], 'Online latent loss (moving target; not cross-model accuracy)', 'Latent MSE')
    contacts = [r for r in summaries if r['task'] == 'contact_discrimination' and r['evaluation_kind'] == 'transfer'
                and r['adaptation_family'] == 'swapped_roles' and r['horizon'] == 1]
    line_plot(directory / '04_developmental_adaptation_curves.png',
              [r for r in contacts if r['variant'] == 'p'], 'budget', 'accuracy', lambda r: r['checkpoint'],
              'Fresh and developmental P: complete adaptation curves', 'Accuracy', chance=True)
    comparisons = [
        ('05_p_vs_reactive.png', ('p', 'reactive'), 'P vs approximately parameter-matched reactive predictor'),
        ('06_replay.png', ('p', 'no_replay'), 'Bounded replay vs repeated current evidence'),
        ('07_history.png', ('p', 'no_history'), 'Persistent recurrent history vs reset history'),
        ('08_horizons.png', ('p', 'short_horizon', 'long_horizon'), 'Mixed, short-only and long-only supervision'),
    ]
    for filename, variants, title in comparisons:
        selected = [r for r in contacts if (r['variant'] in variants and r['age'] == late_age)
                    or (r['variant'] == 'p' and r['checkpoint'] == 'fresh')]
        line_plot(directory / filename, selected, 'budget', 'accuracy',
                  lambda r: r['variant'] + '/' + r['checkpoint'], title, 'Accuracy', chance=True)
    retention = [r for r in summaries if r['task'] == 'contact_discrimination' and
                 r['evaluation_kind'] == 'retention' and r['horizon'] == 1 and r['budget'] == 0]
    line_plot(directory / '09_retention.png', retention, 'age', 'accuracy', lambda r: r['variant'],
              'Earlier-family retention across the declared development shift', 'ID contact accuracy', chance=True)
    line_plot(directory / '10_late_plasticity.png', primary, 'age', 'adaptation_gain', lambda r: r['variant'],
              'New-experience gain (interpret alongside ceiling effects)', 'Final minus initial accuracy')
    for horizon in sorted({r['horizon'] for r in summaries}):
        subset = [r for r in summaries if r['variant'] == 'p' and r['task'] == 'natural_prediction' and
                  r['evaluation_kind'] == 'transfer' and r['horizon'] == horizon and r['age'] == late_age]
        line_plot(directory / f'11_natural_prediction_h{horizon}.png', subset, 'budget', 'persistence_gain',
                  lambda r: r['adaptation_family'], f'Natural prediction vs persistence, horizon {horizon}',
                  'Persistence MSE minus prediction MSE (higher is better)')
    alias = [r for r in summaries if r['task'] == 'contact_discrimination' and r['evaluation_kind'] == 'transfer'
             and r['adaptation_family'] == 'history_alias' and r['age'] == late_age and r['horizon'] == 1]
    line_plot(directory / '12_history_alias.png', alias, 'budget', 'accuracy', lambda r: r['variant'],
              'Matched final images: identifying history is necessary', 'Accuracy', chance=True)
    for field in ('forward_macs', 'evidence_bytes', 'updates', 'trainable_parameters'):
        line_plot(directory / f'13_resources_{field}.png', [r for r in resources if r['age'] == late_age],
                  'budget', field, lambda r: r['variant'], f'Adaptation resource audit: {field}', field)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='baby_p/results')
    args = parser.parse_args()
    make_plots(args.results)


if __name__ == '__main__':
    main()
