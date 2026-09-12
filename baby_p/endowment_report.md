# Initial endowment audit

**Implementation audit only; not an empirical validation.** No learned knowledge exists before lifetime interaction. The model uses seeded random PyTorch initialization, not pretrained weights. A fresh evaluation control restores the exact original initialization for a paired comparison.

## Information available to the learner

- RGB pixel tensor dimensions, channel order, `uint8` range and fixed division by 255.
- Five primitive action indices; the model receives previous actions and the chosen future primitive action sequence. Action names/physical meanings are not encoded as semantic features.
- A fixed CNN, latent dimensionality, recurrent-state dimensionality, action-conditioned prediction architecture, configured horizons, initial zero hidden state and episode-reset convention.
- Random initialized weights/biases, standard LayerNorm initialization, an identical stop-gradient copy for the EMA target, empty AdamW state, empty replay and empty current evidence.
- Learning biases: latent MSE, variance/covariance stabilization, EMA decay, gradient clipping, weight decay, fixed update budget and FIFO evidence capacity/sampling policy.
- A seeded shuffled-action coverage policy independent of observations and model predictions. No semantic reward or externally specified desired abstraction.

These are substantial architectural/optimization/sensorimotor priors. Bounded and randomly initialized does not mean prior-free.

## Simulator-only information

Grid coordinates, object lists and identities, binary push/block roles, colors and shapes, velocities, collision rules, tick phase, episode seeds and distribution-family names exist inside the simulator. The agent receives rendered pixels and actions only. The white controlled entity and border are fixed rendering conventions. Appearance and role are separate simulator variables; deliberately correlated acquisition splits and independently recombined evaluations are researcher choices, not labels delivered to the learner.

## Evaluator-only information

The evaluator knows which mechanisms and nuisance factors to recombine. It creates identifying contact histories, queries both role outcomes, and knows which candidate is the true future. It also knows chance level, evaluation horizon, role balance and the configured success criterion. These data are used only for measurement. No counterfactual candidates, role labels, correctness signals or benchmark scores drive gradient updates, action selection or hyperparameter search.

Controlled test histories are researcher-designed interactions, not training demonstrations. This limits interpretation: the task assesses prediction after an informative history, not autonomous discovery of informative actions. Probe inference temporarily encodes candidate images to score them, but it is isolated from the acting/learning path, carries no state to the next probe, and provides no extra adaptation experience.

## Evidence/provenance versus model input

Replay stores original pixel/action fragments and seed/episode/timestep provenance. The model reads only pixels, actions and the previous-action field; seeds/identities/roles are not numerical features. Fixed-capacity replay and a fixed-size recent window are the only original-evidence storage available to learning. External checkpoint and diagnostic files are researcher artifacts; the policy does not access them. Checkpoint resumption restores the same bounded memory rather than aggregating archived lifetimes.

## Explicitly absent

Pretrained models/features; language models; external datasets; task demonstrations for training; object categories/IDs/role labels as model inputs; handwritten relations; symbolic state; abstraction rewards; privileged simulator state; test feedback into learning; unlimited memory; acting-time episodic retrieval; goals; language; social learning; planning; learned curiosity; meta-learning; growth or unit replacement.

## Reproducibility

The configuration records seeds, image/action dimensions, architecture, horizons, replay schedule, optimizer/lr, batch/update budgets and adaptation budgets. Each lifetime records source hash, revision and software/platform versions. Environment/policy/probe seeds are independently derived using a stable SHA256 namespace. Checkpoints preserve RNG and optimizer/replay/live-stream state. Multiple independent lifetime seeds, not individual timesteps, are the experimental replicates.
