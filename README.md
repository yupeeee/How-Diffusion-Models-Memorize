# How Diffusion Models Memorize

Diffusion memorization experiments from preserved generation, SSCD and proximity
artifacts. The theory layer performs tensor reductions without loading models,
and regenerates figures from saved scalar tables.

## Run

From the repository root, in your Python environment:

```bash
python -m pip install -r requirements.txt
./run_all.sh --download
```

Defaults remain **20 seeds, 50 steps, guidance 7.5, GMM selection**. The supported
matrix is SDv1/DDIM, SDv1/DDPM, SDv2/DDIM and RealVis/DDIM. Experiment seeds are
`0..N-1`; frozen selection uses reference seeds `N..2N-1`. Every experiment seed
of each selected prompt is retained, including unsuccessful reproductions.

```bash
# Resume upstream stages, reduce compatible caches once, then plot saved tables
./run_all.sh

# Analyze one supported pair
./run_all.sh --model sdv1 --scheduler ddim

# Regenerate all designated theory and existing proximity figures from saved data
./run_all.sh --plot
./run_all.sh --model sdv1 --scheduler ddim --plot

# Also render saved optional injection and applicable terminal-terms diagnostics
./run_all.sh --plot --include-diagnostics

# Rebuild derived theory only; bypass every upstream executable
./run_all.sh --recompute-experiments

# Limit theory reduction to two GPUs, using generation's device convention
CUDA_VISIBLE_DEVICES=0,1 ./run_all.sh --recompute-experiments --device auto

# Run one theory configuration on a specific GPU
./theory_validation.sh --model sdv1 --scheduler ddim --device cuda:0

# Plot a copied scalar bundle without model weights or raw tensor payloads
./theory_validation.sh --bundle /path/to/analysis_hash --plot

./run_all.sh --help
```

`--recompute-experiments` requires existing generation, SSCD, frozen selection and
proximity artifacts; missing prerequisites are errors. It never forces upstream
cache regeneration. Root `--overwrite` retains its explicit upstream meaning:
regenerate generation/SSCD caches and rebuild downstream analyses.
`--plot` rejects `--download`, `--overwrite` and `--recompute-experiments`.
The entire requested plot matrix is validated before figures are written.

Use `--center reference-initial` (default), `zero`, or `cached-baseline`. The default
center is the mean initial unconditional clean estimate over unique reference
seeds, fixed across prompts and timesteps. It is held out from experiment seeds,
but shares the protected selection seed bank; it is not the known data mean.
`zero` is a sensitivity diagnostic. `cached-baseline` requires
`--cached-baseline PATH` to a compatible existing independently inferred baseline;
no center mode triggers new inference. Deprecated `--use-mu` retains only that
independent-baseline meaning, and `--no-mu` means `zero`.

Change run identity with `--N`, `--T` and `--g`. Finite negative or nonstandard
guidance remains valid for algebraic analysis; manuscript assumptions are recorded
separately. Theory uses generation's device convention: `--device auto` (the
default) uses every CUDA GPU visible to PyTorch, with CPU fallback when CUDA is
unavailable. `--device cuda` or `cuda:N` selects one GPU; `--device cpu` forces CPU.
Use `CUDA_VISIBLE_DEVICES` to select a GPU subset. Whole prompt records are divided
among devices, with at most one process per GPU and no more processes than selected
records. Each worker uses the same fixed center and candidate support, saves atomic
record shards, and the parent validates and assembles the scalar bundle. Completed
record shards are reused on resume, even if the number of GPUs changes. To resume
one configuration without forcing a rebuild, use `./theory_validation.sh --model
sdv1 --scheduler ddim --device auto`. Reference-cache consistency checks also use
parallel CPU readers; trajectory transfers and scalar copies are batched. The
bundle manifest records each worker's device, assigned prompts, process ID,
completion status and duration. A single `[Theory] Records` tqdm bar combines
progress across all devices, with reduced, resumed and failed prompt counts.
`--plot` and `--validate-only` do not initialize
GPUs or reduction workers. Every wrapper honors `PYTHON`; for example,
`PYTHON=/path/to/python ./run_all.sh --plot`.

## Evidence and outputs

The registry maps seven manuscript results and emits **four default figures**:

| Figure | Measurement and scope |
| --- | --- |
| `initial_recovery` | Paired conditional/unconditional target RMSE at the saved initialization; finite-noise recovery, not the forward-loss premise or a zero-SNR limit |
| `unconditional_center` | Initial-only ECDF of unique evaluation-seed distances from the fixed reference center; model-output dispersion, not data-mean convergence |
| `posterior_feedback` | Actual Equation-15 candidate-reference variation and Proposition-5 margin versus matched log-probability gain; estimated condition coverage under the declared candidate law |
| `target_synchronization` | Prompt-balanced conditional/unconditional median errors and interquartile bands in fixed terminal `SSCD > 0.75` / `<= 0.75` groups; paired joint diagnostics remain in tables |

`--include-diagnostics` adds initial `target_injection` and, only when the exact
terminal clean-update condition applies, `terminal_terms`. Matched displacement
remains vector/replay numerical QA with no figure. The old operational endpoint
scatter is retired. No fallback figure disguises an unavailable quantity.

The candidate bank is frozen from all valid complete cached targets in the
configuration's declared experiment run before selection, with equal mass on
exact distinct latent atoms and retained record/image aliases. It is not the
unknown training law. Equation 15 is integrated analytically on that candidate
mixture; no learned branch is evaluated on counterfactual states. Adaptive
quadrature records tolerances, budgets, uncertainty and saturation. Condition
coverage is **numerically estimated**, with unresolved rows preserved. Negative
gains and zero coverage remain visible. The actual training posterior, training
mean, forward loss, and single-target training assumption remain unidentified.

Each run publishes `outputs/<run>/theory_v2/<analysis_hash>/`. The directory name
is retained for discovery compatibility; the current scalar schema is **3** and
formula version is **cache-theory-3.0-proposition5**. Old feedback tables lacking
V are rejected with an analysis-only recomputation command. A version change
invalidates derived analysis, never protected trajectories or SSCD.

Bundles contain the registry, source identities/hashes, schedule, center/support
metadata, atomic per-record Parquet shards, initial/endpoint tables, six saved
figure tables, per-step/per-prompt/phase summaries, and
`applicability_report.json`. Figures have PNG/PDF and caption/formula sidecars;
the managed figure manifest records numerical exclusions, coverage and scales.
Only obsolete pipeline-owned figure files are archived; unrelated files remain.
Forced recomputation stages its result until success, preserving the previous
complete scientific result on failure.

Normal analysis publishes scalars and invokes the same renderer as `--plot`.
Plot-only never reads raw tensors, recomputes posteriors/quadrature, refits the
center/support, or invokes inference. Numerical and style identities are separate.
Keep the bundle's JSON, CSV and Parquet files for standalone plotting; auxiliary
center/support tensors are not needed. See the [implementation audit](implementation_audit.md),
[statement registry](docs/theory/statement_registry.json), and
[formula/schema dictionary](docs/theory/quantity_dictionary.md).

An independently supplied raw latent L2 tolerance can be recorded with
`--target-error-tolerance VALUE` during analysis, and the same option selects
that saved analysis during root plotting. It enables observed-grid entry/censor
summaries and the applicable terminal sufficient region using `VALUE/sqrt(d)`.
The pipeline never chooses a tolerance from SSCD or measured outcomes.

Linear branch reductions use bounded seed/time blocks. Candidate posterior
calculations cache target geometry and a support Gram matrix per worker; segment
logits are affine, and node/query chunks bound working memory. No reconstructed
clean-estimate trajectory or query-by-node-by-candidate-by-latent array is saved.

Protected trajectories, predictions and scores remain under `logs/`; recovered
data and frozen selections remain under `data/webster/`. Dataset preparation,
prompt eligibility, generation, SSCD and proximity scientific contracts are
preserved.

The old theorem, independent baseline, forward-corruption and decoded-gallery
wrappers remain explicitly invokable **legacy utilities**. They are absent from
the default theory pipeline. Options for fresh Gaussian sweeps or independent
loss draws (`--evaluation-source`, `--num-loss-seeds`, `--loss-seed`, and
`--num-baseline-seeds`) fail with a migration message in the new pipeline.

## Synthetic validation

```bash
python -m pytest tests/test_theory_*.py
```

These tests require no network or model downloads. Plot tests regenerate all
registered figures from copied scalar bundles while tensor/model imports are
blocked, compare repeated output data/axes/images, and check numerical file hashes
and timestamps. Root dispatch tests cover the supported matrix, all-configuration
plot preflight, incompatible flags and the isolation of theory recomputation.

Implementation verification and the scoped real-cache audit are recorded in
[the change report](docs/theory/change_report.md).

## Candidate-figure discovery

The theory discovery suite adds 34 registered designs and fixed snapshot/group
variants while preserving the four main-figure outputs. It saves direct matched
posterior gains, endpoint direction tests, local guidance dose curves, retrieval,
synchronization, and separately resumable refined integral diagnostics.

```bash
./run_all.sh --recompute-experiments --figure-suite candidates
./run_all.sh --plot --figure-suite candidates
./run_all.sh --plot                         # reloads the saved suite
./run_all.sh --plot --figure-suite main      # explicit four-figure suite
```

Model/scheduler filters and multi-GPU device selection are unchanged. Offline
galleries and every candidate's numerical/applicability status are saved under
`outputs/<run>/theory_candidates/<hash>/`; the cross-configuration gallery is
`outputs/theory_candidates/index.html`. Plot-only uses saved scalars. See the
[discovery guide](docs/theory/candidate_discovery.md) and
[formula mapping](docs/theory/candidate_formula_mapping.md).
