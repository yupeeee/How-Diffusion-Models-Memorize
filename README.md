# How Diffusion Models Memorize

Diffusion memorization experiments using preserved generation, SSCD, and proximity caches. The default pipeline finishes with the fixed paper figures: **four main figures, eight core appendix figures, and two terminal appendix figures when applicable**.

## Run

```bash
python -m pip install -r requirements.txt
./run_all.sh --download                       # Prepare data and run/resume the pipeline
./run_all.sh                                  # Resume all supported configurations
./run_all.sh --model sdv1 --scheduler ddim     # One configuration
./run_all.sh --recompute-experiments           # Analysis only; reuse protected caches
./run_all.sh --plot                           # Saved proximity + main and appendix figures
./run_all.sh --plot --diagnostics             # Also export saved optional diagnostics
```

Defaults remain guidance 7.5, 50 updates, 20 evaluation seeds, and GMM selection. Supported pairs are SDv1/DDIM, SDv1/DDPM, SDv2/DDIM, and RealVis/DDIM. Evaluation uses seeds `0..N-1`; frozen reference selection uses `N..2N-1`. Every evaluation seed of each retained prompt remains included. `--model`, `--scheduler`, `--g`, `--T`, and `--N` select a configuration. All wrappers honor `PYTHON`.

Normal runs preserve download, generation, SSCD, and proximity resume behavior. `--recompute-experiments` bypasses those stages and updates derived analysis only. Theory analysis never loads learned models. `--plot` reads saved compact theory CSVs and small manifests, plus the existing saved proximity inputs; it does not run a reducer, fit a center, evaluate candidate posteriors, integrate paths, deserialize raw tensors, or hash raw trajectories. It preflights the entire requested matrix and reports the exact analysis command if inputs are missing or obsolete.

`--overwrite` retains its explicit upstream regeneration meaning; it is not needed for analysis updates. Plot mode rejects `--download`, `--overwrite`, and `--recompute-experiments`. The default paper suite ignores historical saved `main`/`candidates` selections. `--figure-suite paper` is an optional alias; the old suite options are retired from the runner.

## Devices and scientific configuration

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_all.sh --recompute-experiments --device auto
./theory_validation.sh --model sdv1 --scheduler ddim --device cuda:0
PYTHON=/path/to/python ./run_all.sh --plot
```

Theory follows generation's device convention: `auto` uses all visible CUDA devices, with CPU fallback; `cpu`, `cuda`, or `cuda:N` selects one device. Whole records are distributed to workers sharing the fixed center and candidate law. The parent owns the single progress bar across devices; completed record shards are reusable. Sensitive arithmetic uses float64. Plot mode starts no device workers. Endpoint-only paper measurements do not wait for path quadrature; integrated diagnostics remain separately resumable.

The default `--center reference-initial` uses the fixed mean of initial unconditional estimates from unique reference seeds. It is an independently identified model-output center, not the unknown training-data mean. `--center zero` is a sensitivity configuration. `--center cached-baseline --cached-baseline PATH` requires an existing compatible independent baseline and performs no inference. Deprecated `--use-mu` and `--no-mu` retain those existing meanings. Changing the active scientific configuration requires explicit analysis recomputation, which archives the previous derived paper bundle.

An independently supplied raw latent L2 tolerance can be recorded with `--target-error-tolerance VALUE`. The corresponding coordinate error is L2 divided by `sqrt(d)`; an already computed RMSE must not be divided again. The SSCD threshold 0.75 is never a latent tolerance.

## Paper outputs

Canonical outputs follow the proximity run/seed-role convention:

```text
outputs/sdv1_ddim_g7.5_T50_N20/theory/experiment_S0_N20/
  run_config.json             # Scientific identity and source provenance
  summary.json, audit.json    # Measurements, counts, numerical/applicability checks
  figure_manifest.json        # Actual output hashes and status of every requested figure
  figure_captions.md          # Formulas, weighting, snapshots, and band definitions
  initial.csv, terminal.csv, failed.csv
  logical_tables.json        # Authoritative existing trajectory/feedback shard references
  plot_data/*.csv             # Compact immutable inputs; sufficient for rendering
  audit_data/*.csv            # Terminal, reference-tail, coverage, and offending-row audits
  main/*.png, main/*.pdf
  appendix/*.png, appendix/*.pdf
  diagnostics/               # Figure exports only when requested
```

Main figures:

- `initial_recovery`
- `initial_unconditional_concentration`
- `posterior_feedback_positive_fraction`
- `target_synchronization`

Core appendix figures:

- `initial_recovery_within_prompt`
- `initial_target_retrieval_rank`
- `posterior_feedback_initial_comparison`
- `posterior_feedback_normalized_gain`
- `posterior_feedback_increasing_profile_fraction`
- `branch_gap`
- `joint_target_recovery_early`
- `joint_target_recovery_late`

Conditional appendix figures are `terminal_error_terms` and `terminal_bound_tightness`. They require a structurally clean terminal update and successful numeric reconstruction; absence has a precise manifest reason. For T=50, joint-recovery snapshots remain k=10 and k=48. A very short run may alias coincident snapshots rather than export duplicates.

Each figure has one data axes and a same-stem PNG/single-page PDF pair. Figures use the shared proximity publisher, STIX typography, fixed SSCD coloring, and the reviewed gold/purple outcome colors. Conditional curves are solid and unconditional curves dashed. Descriptive IQRs and numerical-sign ambiguity bands have distinct saved definitions. Raw signed values and tails remain in the analysis.

Large source tables remain at their validated `theory_candidates/<hash>/` locations; new underlying shared reductions use `theory_measurements/<hash>/`. The active runner creates no `theory_v2` outputs. Scientific source data and old audit records are preserved. Only hash-verified, renderer-owned obsolete figures may be archived. Role locks, staged publication, and paired export rollback prevent half-published bundles.

## Reuse and audit

A portable paper bundle can be re-rendered without its backing trajectory shards:

```bash
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

An explicit migration from reviewed candidate measurements is available when their original logs were archived separately:

```bash
./theory_validation.sh --model sdv1 --scheduler ddim --recompute-experiments \
  --source-analysis outputs/sdv1_ddim_g7.5_T50_N20/theory_candidates/ANALYSIS_HASH \
  --source-logs _logs --device cpu
```

This validates and records the selected source, preserving candidate numerical shards. It never silently substitutes archived measurements for a newly generated run. `--source-logs` extends missing terminal accounting from matching saved records; mismatching provenance fails. Normal `run_all.sh` uses the current scientific caches.

The candidate law, same-seed SSCD, prompt-balanced populations, and original Proposition 5 margin remain fixed. Endpoint gains and endpoint profile tests use the same destination noise and candidate support at both matched endpoints. The original sufficient norm condition and newer directional diagnostics retain distinct names. Unresolved numerical signs remain in fraction denominators. Final-current reference availability is independent of destination-posterior eligibility.

```bash
python -m pytest tests/test_theory_*.py
```

See [the paper migration and audit report](docs/theory/paper_pipeline.md), [the fixed registry](utils/experiments/theory/paper_registry.py), and [the manuscript formula mapping](docs/theory/candidate_formula_mapping.md). Historical discovery documentation and scientific tests remain as provenance; the exploratory gallery is no longer a default CLI mode. Legacy independent-inference wrappers are outside the paper pipeline.
