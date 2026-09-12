# BABY-First — P research prototype

**Implementation for local testing; not executed or validated by the authoring assistant. No developmental result is claimed.**

The experiment asks whether continuous online predictive learning reduces later adaptation cost on held-out appearance/role/context recombinations. This is a falsifiable research instrument, not an intelligence demo.

## Run locally

Use Python **3.11** in a fresh environment. Run these commands from the repository root after checking out the completed implementation branch (or pulling it after merge):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m compileall -q baby_p tests train.py evaluate.py run_experiments.py
python -m unittest discover -s tests -v
python run_experiments.py --config baby_p/config/base.yaml --smoke
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of `source`. CPU is the default; no GPU, pretrained weights, datasets or network service are required after installation. Dependency pins are reproducibility choices, not a claim that installation was tested on every platform.

Run the complete multi-seed experiment, including every baseline, adaptation evaluation, table, plot and interpretation report:

```bash
python run_experiments.py --config baby_p/config/base.yaml
```

The smoke run writes separate `baby_p/results/smoke` and `baby_p/checkpoints/smoke` directories. It uses one seed and tiny budgets: **do not interpret it scientifically**. The full default suite uses six lifetime variants, three independent seeds, multiple checkpoints, ten evaluation families and all configured budgets. It may be slow on a CPU; no runtime estimate has been measured.

Resume a suite without overwriting complete measurements:

```bash
python run_experiments.py --config baby_p/config/base.yaml --resume
```

For an independent rerun, select **both** new output locations:

```bash
python run_experiments.py --config baby_p/config/base.yaml --output baby_p/results/replicate2 --checkpoint-dir baby_p/checkpoints/replicate2
```

## Individual commands

These commands are alternatives to running the suite, not additional required steps after it. Existing outputs are protected against accidental overwrite.

```bash
python train.py --config baby_p/config/base.yaml --seed 11
python evaluate.py --checkpoint baby_p/checkpoints/p/seed_11/late.pt --output baby_p/results/manual_late
python train.py --resume baby_p/checkpoints/p/seed_11/middle.pt
```

Evaluate the fresh/early/middle checkpoints by replacing `late.pt` with `fresh.pt`, `early.pt` or `middle.pt` and choosing a distinct output directory. A standalone evaluation writes raw measurements; suite analysis collects the suite's standard directory structure.

Regenerate tables/report and plots after a suite:

```bash
python -m baby_p.analysis.metrics --results baby_p/results
python -m baby_p.analysis.plots --results baby_p/results
```

Test only one ablation during development:

```bash
python run_experiments.py --config baby_p/config/base.yaml --smoke --variants no_history --output baby_p/results/no_history_smoke --checkpoint-dir baby_p/checkpoints/no_history_smoke
```

The full protocol, limitations and requirement-to-file map are in [baby_p/README.md](baby_p/README.md). Initial information is audited in [endowment_report.md](baby_p/endowment_report.md).
