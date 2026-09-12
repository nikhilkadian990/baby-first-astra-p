import argparse
import json
from pathlib import Path
import time
from baby_p.experiments.ablations import VARIANTS, variant_config
from baby_p.training.common import load_config, write_json, load_checkpoint
from baby_p.training.train import train
from baby_p.training.evaluate import evaluate_checkpoint


def run_suite(config, variants=VARIANTS, resume=False):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    suite_manifest = root / 'suite.json'
    specification = dict(config=config, variants=list(variants), seeds=config['seeds'])
    if suite_manifest.exists():
        with suite_manifest.open() as handle:
            previous = json.load(handle)
        if not resume or previous != specification:
            raise FileExistsError('Suite exists; --resume requires exactly the original configuration')
    else:
        write_json(suite_manifest, specification)
    for variant in variants:
        variant_cfg = variant_config(config, variant)
        for seed in config['seeds']:
            run = root / variant / f'seed_{seed}'
            directory = Path(config['checkpoint_dir']) / variant / f'seed_{seed}'
            if (run / 'checkpoints.json').exists() and resume:
                with (run / 'checkpoints.json').open() as handle:
                    checkpoints = json.load(handle)
            else:
                resume_path = None
                if resume and (run / 'manifest.json').exists():
                    candidates = list(directory.glob('*.pt'))
                    if not candidates:
                        raise RuntimeError(f'No recoverable checkpoint in {directory}; use new output paths')
                    resume_path = max(candidates, key=lambda p: load_checkpoint(p)['interactions'])
                checkpoints = train(variant_cfg, seed, variant, resume_path)
            for path in checkpoints.values():
                payload = load_checkpoint(path)
                if payload['agent']['config'] != variant_cfg:
                    raise ValueError(f'Checkpoint configuration mismatch: {path}')
                target = run / f"evaluation_{payload['label']}"
                if (target / 'complete.json').exists() and resume:
                    continue
                if target.exists() and resume:
                    # Preserve partial outputs for diagnosis, but never analyze them as complete.
                    target.rename(run / f"incomplete_{payload['label']}_{time.time_ns()}")
                evaluate_checkpoint(path, target)
    from baby_p.analysis.metrics import analyze
    from baby_p.analysis.plots import make_plots
    analyze(root, config)
    make_plots(root)
    write_json(root / 'suite_complete.json', dict(executed=True, validated=False,
                                                note='Review local tests, diagnostics and controls before interpreting results'))
    print(f'Suite complete: {root / "report.md"}; execution is not validation.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run all developmental baselines, tests of transfer, and plots')
    parser.add_argument('--config', default='baby_p/config/base.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    config = load_config(args.config, args.smoke)
    if args.output:
        config['output'] = args.output
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    run_suite(config, args.variants, args.resume)


if __name__ == '__main__':
    main()
