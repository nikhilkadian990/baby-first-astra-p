"""Aggregate episodes within seeds; never treat timesteps as independent replicates."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from baby_p.training.common import write_json


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for values, group in groups.items():
        entry = dict(zip(keys, values))
        for field in fields:
            samples = [r[field] for r in group if field in r]
            if samples:
                entry[field] = float(np.mean(samples))
        entry['observations_aggregated'] = len(group)
        output.append(entry)
    return output


def mean_interval(values, repetitions=2000, seed=1729):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return dict(mean=None, low=None, high=None, seeds=0)
    if len(values) == 1:
        return dict(mean=float(values[0]), low=None, high=None, seeds=1)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (repetitions, len(values)), replace=True).mean(1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), seeds=len(values))


def adaptation_curves(summaries, criterion):
    selected = [r for r in summaries if r['task'] == 'contact_discrimination' and r['evaluation_kind'] == 'transfer']
    keys = ['variant', 'seed', 'checkpoint', 'age', 'adaptation_family', 'horizon']
    groups = defaultdict(list)
    for row in selected:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for values, rows in groups.items():
        rows.sort(key=lambda r: r['budget'])
        x = np.asarray([r['budget'] for r in rows])
        y = np.asarray([r['accuracy'] for r in rows])
        # Criterion must remain met at all subsequently measured budgets.
        achieved = [i for i in range(len(y)) if np.all(y[i:] >= criterion)]
        cost = int(x[achieved[0]]) if achieved else None
        span = int(x[-1] - x[0])
        entry = dict(zip(keys, values))
        entry.update(auc=float(np.trapezoid(y, x) / span) if span else float(y[0]),
                     initial_accuracy=float(y[0]), final_accuracy=float(y[-1]),
                     adaptation_gain=float(y[-1] - y[0]), criterion=criterion,
                     criterion_cost=cost, censored=cost is None,
                     restricted_cost=int(x[-1]) if cost is None else cost,
                     max_budget=int(x[-1]))
        output.append(entry)
    return output


def paired_effect(curves, left, right, repetitions, field='auc', family='swapped_roles', horizon=1):
    def extract(selector):
        group = defaultdict(list)
        for row in curves:
            if (row['variant'], row['checkpoint']) == selector and row['adaptation_family'] == family and row['horizon'] == horizon:
                group[row['seed']].append(row[field])
        return {s: float(np.mean(v)) for s, v in group.items()}
    a, b = extract(left), extract(right)
    common = sorted(set(a) & set(b))
    result = mean_interval([a[s] - b[s] for s in common], repetitions)
    result.update(left='/'.join(left), right='/'.join(right), metric=field, family=family, horizon=horizon)
    return result


def describe(effect):
    if effect['mean'] is None:
        return 'not available (required variant/checkpoint absent)'
    estimate = f"{effect['mean']:+.4f} (n={effect['seeds']} independent seeds)"
    if effect['low'] is not None:
        estimate += f"; paired seed-bootstrap 95% interval [{effect['low']:+.4f}, {effect['high']:+.4f}]"
    else:
        estimate += '; uncertainty unavailable for a single seed'
    return estimate


def analyze(root, config):
    root = Path(root)
    rows, resources = [], []
    for path in sorted(root.glob('*/seed_*/evaluation_*/complete.json')):
        rows.extend(read_jsonl(path.parent / 'probes.jsonl'))
        resources.extend(read_jsonl(path.parent / 'resources.jsonl'))
    if not rows:
        raise ValueError('No complete evaluations found')
    keys = ['variant', 'seed', 'checkpoint', 'age', 'adaptation_family', 'evaluation_kind',
            'family', 'task', 'horizon', 'budget']
    fields = ['accuracy', 'prediction_loss', 'candidate_distance', 'normalized_loss',
              'latent_std', 'persistence_loss', 'persistence_gain', 'target_std']
    summaries = aggregate(rows, keys, fields)
    curves = adaptation_curves(summaries, config['evaluation']['criterion'])
    write_csv(root / 'tables' / 'evaluation.csv', summaries)
    write_csv(root / 'tables' / 'adaptation.csv', curves)
    write_csv(root / 'tables' / 'resources.csv', resources)
    developmental_resources = []
    for path in sorted(root.glob('*/seed_*/resources.json')):
        with path.open() as handle:
            resource = json.load(handle)
        resource.update(variant=path.parent.parent.name, seed=int(path.parent.name.split('_')[-1]))
        developmental_resources.append(resource)
    write_csv(root / 'tables' / 'development_resources.csv', developmental_resources)
    write_json(root / 'analysis.json', dict(summaries=summaries, curves=curves, resources=resources,
                                          developmental_resources=developmental_resources))
    ages = {r['age']: r['checkpoint'] for r in curves if r['variant'] == 'p' and r['age'] > 0}
    early = ages[min(ages)] if ages else 'early'
    late = ages[max(ages)] if ages else 'late'
    repetitions = config['evaluation']['bootstrap_samples']
    contrasts = {
        'late_minus_fresh': paired_effect(curves, ('p', late), ('p', 'fresh'), repetitions),
        'late_minus_early': paired_effect(curves, ('p', late), ('p', early), repetitions),
        'history': paired_effect(curves, ('p', late), ('no_history', late), repetitions),
        'replay': paired_effect(curves, ('p', late), ('no_replay', late), repetitions),
        'horizon': paired_effect(curves, ('long_horizon', late), ('short_horizon', late), repetitions),
        'reactive': paired_effect(curves, ('p', late), ('reactive', late), repetitions),
        'alias_history': paired_effect(curves, ('p', late), ('no_history', late), repetitions, family='history_alias'),
    }
    write_csv(root / 'tables' / 'contrasts.csv', [dict(name=k, **v) for k, v in contrasts.items()])
    # Seed averages first, including when multiple adaptation branches repeat the same retention test.
    def summary_value(task, kind, field, checkpoint=late, budget=0, family=None):
        values = defaultdict(list)
        for r in summaries:
            if (r['variant'] == 'p' and r['checkpoint'] == checkpoint and r['task'] == task and
                    r['evaluation_kind'] == kind and r['budget'] == budget and r['horizon'] == 1 and
                    (family is None or r['family'] == family) and field in r):
                values[r['seed']].append(r[field])
        return mean_interval([np.mean(v) for v in values.values()], repetitions)
    id_gain = summary_value('natural_prediction', 'retention', 'persistence_gain')
    id_accuracy = summary_value('contact_discrimination', 'retention', 'accuracy')
    recombination = summary_value('contact_discrimination', 'transfer', 'accuracy',
                                  budget=max(config['evaluation']['budgets']), family='swapped_roles')
    collapse = [r for r in rows if r.get('candidate_distance', 1) < 1e-6]
    censored = sum(r['censored'] for r in curves)
    report = [
        '# BABY P exploratory experiment report',
        '',
        'Generated from local output files. Execution is not validation. These are behavioral prediction tests, not evidence of conceptual understanding.',
        'Primary endpoint fixed in advance: swapped-role contact-discrimination adaptation AUC at horizon 1; chance = 0.5. Other endpoints are exploratory.',
        'Intervals resample independent lifetime seeds, not episodes or timesteps. Three seeds are a pilot, not a confirmatory study; no multiplicity correction is applied.',
        '',
        '1. **Does P learn in distribution?** Late zero-adaptation contact accuracy: ' + describe(id_accuracy) +
        '. Natural-trajectory persistence-minus-predictor loss: ' + describe(id_gain) +
        '. Positive persistence gain is necessary supporting evidence, not sufficient: inspect target spread and earlier checkpoints.',
        '2. **Does earlier experience improve later adaptation?** Primary AUC, late minus fresh: ' + describe(contrasts['late_minus_fresh']) +
        '. Late minus early: ' + describe(contrasts['late_minus_early']) +
        '. Inspect the complete curves and censored criterion costs; AUC alone does not establish decreasing interactions-to-criterion.',
        '3. **Does transfer survive appearance/role recombination?** Final swapped-role accuracy: ' + describe(recombination) +
        '. Review every nuisance family separately in evaluation.csv; a pooled gain can hide failure on recombinations.',
        '4. **Does persistent history matter?** P minus no-history AUC: ' + describe(contrasts['history']) +
        '. On exactly aliased final frames: ' + describe(contrasts['alias_history']) +
        '. A reactive predictor must score chance on balanced alias pairs; otherwise investigate leakage.',
        '5. **Does bounded original-evidence rehearsal matter?** P minus no-replay AUC: ' + describe(contrasts['replay']) +
        '. Retention curves and late adaptation gains must both be inspected: stability without plasticity is not success.',
        '6. **Does longer-horizon prediction help?** Long-only minus short-only AUC at the common horizon-1 endpoint: ' + describe(contrasts['horizon']) +
        '. Per-horizon tables also show trained and untrained horizons; horizon-8 supervision cannot be assumed to improve horizon 1.',
        '7. **Does P outperform the reactive baseline under matched resources?** P minus reactive AUC: ' + describe(contrasts['reactive']) +
        '. Parameter counts are approximate matches, not proof of compute matching. See resources.csv for forward MAC proxies, updates, wall time and storage.',
        '8. **Could memorization, compute or storage explain it?** Not ruled out. Splits use disjoint environment seeds, swapped roles, novel colors and history aliases; optimizer and hidden state reset for adaptation. ' +
        f"Historical evaluation replay policy: {config['evaluation']['replay_policy']}. Development uses more computation by design. MACs omit backward/optimizer operations; historical replay is an explicit storage treatment.",
        f'9. **What failed?** {censored}/{len(curves)} criterion curves are right-censored; {len(collapse)}/{len(rows)} rows have candidate separation below 1e-6. ' +
        'These are diagnostics, not causal attribution. Inspect nonfinite errors, latent spread, persistence gains, retention after the declared shift, and late adaptation gains before attributing a null to the architecture.',
        '10. **What next?** If ID learning fails, test optimization and representation collapse first. If alias transfer fails despite ID learning, test longer informative histories. If retention fails, vary bounded replay capacity at fixed update budgets. ' +
        'If the primary effect survives these controls across more independent seeds, preregister a harder mechanism-preserving transfer family. Do not add a cognitive subsystem solely to rescue a null.',
        '',
        'See tables/adaptation.csv for measured first-sustained criterion crossing and censor flags; restricted_cost caps censored values and must not be called a measured acquisition time.',
        'Contact histories are evaluator-designed identification episodes, not demonstrations used to train the learner. The task is future-image discrimination, not autonomous goal achievement.',
    ]
    (root / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    return summaries, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='baby_p/results')
    args = parser.parse_args()
    with (Path(args.results) / 'suite.json').open() as handle:
        config = json.load(handle)['config']
    analyze(args.results, config)


if __name__ == '__main__':
    main()
