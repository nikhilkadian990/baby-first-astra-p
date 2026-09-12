# P — Online Predictive-State Experimenter

## Status and hypothesis

**Written for local execution; tests and experiments have not been run in the authoring environment. Not validated.** Running the suite does not establish implementation correctness or validate the research hypothesis. Review local tests and diagnostics before interpreting results.

Hypothesis: earlier online predictive learning produces reusable organization such that genuinely new appearance/role/context combinations require less later experience. The desired pattern is reduced adaptation cost with age, preserved retention and held-out behavioral transfer that simpler controls cannot explain. Null results are useful. Predictive loss improvement alone is not conceptual understanding or abstraction.

## Architecture

```text
procedural world -> RGB uint8 pixels -> learned CNN -> z_t
                                                  |
previous primitive action + persistent h_(t-1) -> GRU -> h_t
                                                        |
precommitted random primitive action sequence -> predictor GRU
                                                        |
                                      predictions of future encoded observations
                                                        |
new pixels -> stop-gradient EMA target CNN -> latent prediction loss
                                      + variance/covariance regularization
                                                        |
             bounded current fragment + FIFO original-evidence replay
                                                        |
                                      budgeted AdamW online update
```

`h_t` persists within a world episode; learned parameters and bounded replay persist across the entire developmental lifetime. Episode resets are explicit independent-world boundaries. No object slots, semantic bottleneck, symbolic relations or pretrained components exist.

The action-conditioned predictor rolls forward using only the supplied primitive actions, never future observations. Multiple latent forecasts use shared predictor parameters. Future action sequences are known because V1 exploration is independent of observations; this is conditional forecasting, not planning. The replaceable `CoveragePolicy` shuffles bags containing each of five actions once. Policy randomness is shared across variants/checkpoints. No goals, rewards or sophisticated curiosity are used.

## Environment and information boundary

A deterministic grid world is rendered to a small image (default `3 x 27 x 27`). Primitive actions are no-op, north, south, west and east. A white controllable entity attempts cell moves. Objects can block a push or be pushed into a free cell. Some entities move with hidden velocity every third tick, bounce on collisions, and a successful push stops their autonomous motion. Physics is fixed; instances and seeds vary. Object identity does not affect rendering or physics.

Roles and appearances are separate generator variables. Development deliberately presents only a correlated subset: one role with red/orange, the other with blue. Evaluation includes reversed familiar-color/role pairings, unseen green/purple/cyan colors, new shapes, positions, concentrated/rotated layouts, background changes, changed identities, trajectories and joint nuisance changes. `history_alias` makes role independent of familiar appearance. These are recombinations of established push/block mechanisms, not arbitrary unseen laws.

Only pixels and primitive actions enter neural computation. Generator dictionaries, role bits, identities, seeds and counterfactual rules remain simulator/evaluator data. Training fragment metadata records provenance; the only metadata field consumed by the model is the previous primitive action. The [endowment audit](endowment_report.md) distinguishes agent, simulator and evaluator information.

## Online learning and bounded resources

Defaults are a learned two-convolution encoder with a 32-dimensional latent, a 64-dimensional GRU state, and horizons 1/4/8. Targets come from an EMA encoder initialized as a copy of the randomly initialized online encoder. Variance-floor and off-diagonal covariance penalties discourage representation collapse; **EMA alone does not prevent collapse**. Target spread, candidate distances and prediction-versus-persistence diagnostics are logged.

Learning uses AdamW, gradient clipping, fixed update intervals and a fixed number of updates per event. The loss averages supervised horizons. Latent stabilization and weight decay have explicit configurable weights. Backpropagation reconstructs history from bounded original-evidence windows: burn-in states detach before the learning portion, rather than backpropagating across a whole lifetime. This approximates long-range history and is a limitation.

The fixed FIFO memory stores copied original `uint8` observation fragments, actions, environment seed, episode, start timestep and previous action. Capacity is 128 fragments by default; oldest evidence is evicted. The current sliding window is separate and bounded. No acting-time replay retrieval exists. Default batches contain one current fragment plus up to three uniformly sampled FIFO fragments without replacement. Missing replay entries are filled by repeating current evidence so no-replay and early-training batches keep the same dimensions. Repeated evidence is not counted as a distinct replay sample.

All variants use the same rollout length and fragment length, including short-only supervision. Default learning starts no earlier than interaction 16: 15 transitions are required to fill the fragment, then the four-step update schedule applies. Thus budgets 1/2/5/10 honestly contain no gradient updates; they must not be presented as successful rapid learning. Increase budgets rather than supplying free future observations or pretraining on evaluation data.

## Continuous development, checkpoints and resumption

A lifetime is not separately retrained per task. Defaults save fresh initialization plus early (400), middle (4,000) and late (12,000) snapshots. A declared change at interaction 8,000 modifies trajectory/background statistics without changing laws or introducing held-out colors/role pairings. Pre/post-change snapshots are included where they do not coincide with an existing checkpoint.

Checkpoints include model and EMA target parameters, optimizer, replay, live hidden state, private and global RNG states, simulator state, action-policy state, current evidence, counters and configuration. Resumption uses the recorded configuration. The fresh control is a new evaluation instance of the identical architecture loaded with that lifetime's saved random initialization; pairing initialization reduces nuisance variance.

Load only your own checkpoints: complete resume files use Python/PyTorch serialization containing simulator and NumPy objects. Keep code, Python and dependency versions fixed for exact continuation. CPU deterministic behavior is the reference target; hardware/platform-independent bitwise identity is not promised.

## Primary experiment and evaluation budget

For every checkpoint and every evaluation family, create an isolated adaptation clone. **Reset AdamW state, live recurrence, current evidence and adaptation counters for all ages, including fresh.** By default clear historical replay too, so developmental transfer is carried by weights, not retrieved old episodes. `evaluation.replay_policy: retain` is an explicitly different storage-confounded treatment. New adaptation evidence can populate replay normally.

Each age and variant receives exactly the same seeded random-action adaptation trajectory for a family. Budgets are cumulative interactions on that trajectory; no gradients are taken between measured budgets except prescribed updates. Families start from independent copies of the checkpoint. Probe scenes and adaptation episodes use separate deterministic seed namespaces. Evaluation uses no gradients, does not mutate replay or live history, and reports its compute separately from adaptation. Repeated probes do not feed measurements back into learning or select hyperparameters.

The primary predeclared endpoint is **swapped-role contact-discrimination adaptation AUC at horizon 1**. The central scientific question also requires complete curves and interactions-to-criterion: an AUC gain caused only by better zero-shot performance is not proof of faster acquisition.

### Behavioral consequence discrimination

The evaluator constructs balanced pushable/blocking contact cases. A short action history (attempt contact, retreat, approach) reveals the functional consequence using only pixels/actions. The model predicts what the next contact and following no-ops will produce. It ranks the actual future image against an evaluator-only alternative produced by toggling the primary object's role at query time. Candidate images never enter learning updates or the policy. Ties score 0.5; chance is 0.5. This measures prediction of interaction outcomes, not navigation or goal achievement.

The `history_alias` pair ends with **identical current images and previous actions but different earlier histories and opposite correct futures**. A historyless model must score chance on the balanced pair. This is a concrete leakage/shortcut check. Novel-color cases similarly remove familiar role cues. Scripted identifying histories are evaluator-designed test episodes, never training demonstrations or privileged model inputs. Their interaction length is identical across ages; reported adaptation budgets count learning-stream interactions, not read-only assessment interactions.

The primary object is stationary before contact in these controlled probes; trajectory perturbations affect distractors. A second natural-trajectory test uses uncurated random-action worlds with moving entities and compares latent prediction with a same-encoder persistence baseline at every horizon. Do not generalize contact-probe success to arbitrary motion mechanisms.

## Baselines and ablations

| Variant | Change relative to P |
| --- | --- |
| `p` | Recurrent history, bounded replay, configured mixed horizons |
| `no_replay` | Zero historical fragment capacity; same-sized repeated-current batch |
| `no_history` | Same GRU parameters but hidden state reset at every observed frame |
| `reactive` | Current pixels only, parameter-matched MLP instead of recurrent observation state; previous-action input zeroed |
| `short_horizon` | Supervise horizon 1 only; fixed shared architecture and evidence delay |
| `long_horizon` | Supervise largest evaluation horizon only; fixed shared architecture and evidence delay |
| fresh controls | Random-initialization checkpoint evaluated for each architecture/variant |

P already supplies the recurrent-history and replay-present conditions; these are not duplicated runs. The predictor remains action-conditioned in reactive and historyless controls so the comparison does not remove the forecasting task. Reactive action selection remains the same randomized exploration policy as P; V1 has no learned control policy.

Parameter matching is approximate for the reactive MLP and exact for recurrent versus reset-GRU state. Matching interaction counts, tensor sizes and update schedules does **not** prove identical compute. Forward multiply-accumulate proxies, parameter counts, target encoder storage, optimizer bytes, evidence bytes, replay samples, updates and wall time are recorded. Forward MACs omit nonlinearities, pooling, backward and optimizer work; they are not measured FLOPs. Examine resource tables before claiming an architectural advantage. More developmental experience inherently uses more prior compute.

## Metrics, statistics, plots and automatic interpretation

The suite writes:

- `suite.json`: complete run configuration, variants and independent seeds.
- Per lifetime: source hash/revision and dependency/platform manifest, interaction-level learning/provenance logs, resources and checkpoint index.
- Per evaluation: raw per-probe/per-horizon outcomes, adaptation trajectory, resource logs and completion marker. Incomplete evaluation directories are excluded from analysis.
- `tables/evaluation.csv`: seed-level budget/horizon/family summaries, contact accuracy, latent losses, candidate spread, normalized loss, natural-trajectory persistence gain.
- `tables/adaptation.csv`: full-curve AUC, initial/final accuracy, late-learning gain, first **sustained** criterion crossing, censor flag and restricted cost.
- `tables/contrasts.csv`: paired seed-level primary contrasts; `tables/resources.csv`: adaptation resource accounting.
- `plots/`: developmental experience vs cost, checkpoint vs transfer, lifetime loss, fresh/early/middle/late curves, P/reactive, replay, history and horizon comparisons; additional censoring, retention, plasticity, natural-prediction, alias and resource diagnostics.
- `report.md`: automatic answers to the ten requested interpretation questions, with missing data and uncertainty stated rather than fabricated conclusions.

The first sustained crossing is the first measured budget at/above criterion that stays there at all later measured budgets. Unreached criteria are **right-censored**, not assigned a fake acquisition time. `restricted_cost` caps them at the largest budget for plotting; consult the censoring plot/table. A crossing at the final budget has no later stability evidence.

Episodes and timesteps are aggregated **within independent lifetime seeds** before uncertainty estimation. Main contrasts use paired seed bootstraps; plots have pointwise seed-bootstrap bands. A single-seed smoke run has no uncertainty interval. Three default seeds are a pilot, not a confirmatory sample size. Secondary endpoints are exploratory without multiplicity correction. No automatic positive result becomes a claim of abstraction.

## Local execution

Run from the repository root, not from this package directory. Exact setup, unit/integration tests, full-suite, smoke, resume and single-checkpoint commands are in the [root README](../README.md). All experiment hyperparameters are in [config/base.yaml](config/base.yaml). Initial architecture sizes and environmental laws are code-defined inductive biases, not swept hyperparameters.

## Requirements audit (implementation, not execution)

| Original sections | Implementation location/status |
| --- | --- |
| 1, 3–7: P loop, pixels, recurrence, action/horizon prediction, online updates | `models/`, `agents/predictive_agent.py`, `training/common.py` |
| 2, 12–14, 17: procedural rules, checkpoints, transfer and nuisance tests | `env/`, `experiments/transfer.py`, `training/train.py`, `training/evaluate.py` |
| 8–10: bounded evidence, continual protection, replaceable exploration | `memory/replay_buffer.py`, `CoveragePolicy`, YAML update/regularization controls |
| 11: initial information audit | `endowment_report.md` |
| 15, 18: required baselines/ablations | `agents/reactive_baseline.py`, `experiments/ablations.py`, `experiments/horizons.py`, fresh snapshots |
| 16, 19, 21, 24: metrics, plots, reproducibility, ten-question report | `analysis/`, manifests, raw JSONL, checkpoints, `experiments/suite.py` |
| 20, 22, 23: structure, documentation, commands | Package directories, root entry points, both READMEs, requirements |
| 25: excluded cognitive subsystems | None introduced |
| Tests | `tests/test_instrument.py`: deterministic physics, pixel boundary, splits, aliases, bounded replay, state controls, matched trajectories, continuation, isolation, metrics and tiny end-to-end pipeline |
| Actual execution/empirical comparisons | **Not performed in the authoring environment; required locally** |

## Limits and falsification

The world is deliberately tiny and roles are binary. Success might reflect a narrow contact heuristic rather than broad reusable structure. Counterfactual-image ranking depends on each model's own latent geometry: collapse or nuisance encoding can change difficulty, so inspect candidate distances and natural prediction instead of comparing raw latent MSE across encoders. The foreground agent and grid border are fixed visual priors. Episodes reset history, and bounded burn-in may not preserve long consequences. Random exploration may rarely supply useful contacts. Rehearsal duplicates overlapping evidence, not independent samples. Update budgets and predictive horizons can limit learning even when the hypothesis is viable.

A null should first be separated into optimization/collapse, inadequate history, retention, relevance of prediction, horizon mismatch or poor transfer. Only after those controls work does this instrument strongly test the developmental hypothesis. Conversely, a positive pilot still needs more seeds, broader tests and tighter resource controls. No language, symbolic reasoning, object slots, episodic acting-time retrieval, goals, planning, curiosity, meta-learning or architectural growth are part of V1.
