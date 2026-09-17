> Historical mathematical context: the active contract is now `projected-gap-error-1`, documented in [four_stage_experiments.md](four_stage_experiments.md). It uses the signed projected gap error, and retains the positive part after signed directional integration. Norm-error/norm-integral formulas below describe preceding designs and are not active evaluated quantities. Historical numerical receipts are not relabeled.

# Direct seven-statement implementation report

This is a source-change report, not a validation receipt. No tests, project
imports/execution, inference, generation, downloads, experiments, posterior
integration, plotting, benchmarks, GPU jobs or data-dependent validation were
performed for this change. Existing data, logs, figures, checkpoints and
historical validation reports were not modified. Nothing was committed or pushed.

## Files and responsibilities

| File(s) | Change |
|---|---|
| `run_all.sh` | Validates/forwards direct science flags; preserves upstream stages; documents inference during analysis and saved-setting inheritance during plotting. |
| `scripts/theory_validation.py` | Explicit loss/reference options, probe batch size, independent marginal-stage switch, inherited plot configuration, direct analysis routing. |
| `utils/models/probe_loading.py` | Opt-in tokenizer/text encoder/UNet loading at the preserved checkpoint revision and precision, without VAE decoding; generation loader unchanged. |
| `utils/experiments/theory/direct_probes.py` | Shared spawned learned-probe tasks, deterministic independent streams, fixed native-grid Gaussian bank, exact-prompt embedding caches, bounded batches, hashed task resume, parent progress. |
| `utils/experiments/theory/direct_probe_math.py` | Float64 forward losses, independent clean-error identity, Pinsker bound, Gaussian recovery, total/optimal/direct-excess marginal loss and cross terms. |
| `utils/experiments/theory/reference_law.py`, `support.py` | One declared weighted/deduplicated finite law, exact mean/radius, cached-target and manifest adapters, posterior means, prompt-ambiguity provenance. |
| `utils/experiments/theory/direct_measurements.py` | One shared record pass for Corollary 3 through Theorem 7; vector distances, all current-state references, independent endpoint observations and compact integration payloads. |
| `utils/experiments/theory/scheduler_adapter.py` | Additional direct matched-update API distinguishes independent deterministic replay, independently supplied innovation, constructed DDPM comparison and non-affine inapplicability; existing methods preserved. |
| `utils/experiments/theory/direct_integration.py` | Original full-norm variation, integrated gain and signed lower bound from saved sufficient statistics; quadrature estimates and failures stay explicit. |
| `utils/experiments/theory/direct_reduce.py` | Shared probe/core/integration dependencies; disjoint worker records, on-disk law descriptors, scalar-only IPC receipts, independent completion markers, exact seed/step joins and failure retention. |
| `utils/experiments/theory/contracts.py`, `paper_contracts.py` | Explicit scientific configuration, new bundle/metric versions, reproducible commands, saved configuration inheritance and compact-input validation; publication safeguards retained. |
| `utils/experiments/theory/direct_figures.py` | Analysis-only compact tables, uncertainty/status counts, descriptive summaries and compatible former behavioral diagnostics. |
| `utils/experiments/theory/paper_registry.py`, `paper_plotting.py` | Seven direct main entries and nine supporting/optional appendix entries; bold notation, named guides, no false Gaussian equality, explicit constructed/unresolved/inapplicable markers and scalar-only PNG/PDF rendering. |
| `utils/experiments/theory/paper_reduce.py` | New direct dependency route and guarded canonical paper publication. Legacy candidate source flags cannot masquerade as independent measured losses. |
| `utils/experiments/theory/statements.py`, `docs/theory/legacy_statement_registry.json` | Preserve the literal legacy registry for old scalar readers. |
| `docs/theory/statement_registry.json` | Direct seven-row manuscript mapping and assumption/input-law metadata. |
| `tests/test_theory_direct_{probes,math,figures,pipeline}.py` | Authored CPU synthetic tests and mocks for probe arithmetic/streams, vector identities, quadrature payloads, scalar joins, resume, axes/statuses and CLI inheritance. |
| Existing paper/runner/CLI/generation tests | Update obsolete four-main/no-theory-inference assumptions while preserving compact-only and publication guard coverage. |
| `README.md`, `docs/theory/{paper_pipeline,quantity_dictionary}.md` | Current usage and contracts; earlier reports are explicitly marked historical and preserved. |

## Seven primary comparisons

All paths below are relative to the canonical paper bundle. Scalar distances
ending `_rmse` already equal raw L2 divided by sqrt(d); no second division is
performed by the renderer.

| Statement | Measured x | Measured y | PNG/PDF stem |
|---|---|---|---|
| Theorem 1 | `sqrt(forward_loss_summary.loss_mean_squared_l2/(latent_dimension*snr))` | `gaussian_conditional.conditional_error_rmse` | `main/theorem1_loss_recovery` |
| Lemma 2 | `gaussian_reference.snr` | `reference_mean_error_rmse`, `learned_mean_error_rmse`, `unconditional_reference_error_rmse` | `main/lemma2_unconditional_baseline` |
| Corollary 3 | `initial.direct_cor3_injection_bound_rmse` | `initial.direct_cor3_injection_discrepancy_rmse` | `main/corollary3_initial_cfg_amplification` |
| Lemma 4 | `matched_updates.direct_lemma4_predicted_rmse` | `matched_updates.direct_lemma4_displacement_rmse` | `main/lemma4_matched_displacement` |
| Proposition 5 | `matched_updates.direct_prop5_margin_rmse`, using original V | `matched_updates.direct_prop5_log_probability_gain` | `main/proposition5_posterior_feedback` |
| Lemma 6 | `trajectory.direct_lemma6_rhs_rmse` | `trajectory.direct_lemma6_gap_rmse` | `main/lemma6_target_specific_synchronization` |
| Theorem 7 | `terminal.direct_theorem7_bound_rmse` | `terminal.direct_theorem7_endpoint_error_rmse`, from actual final tensor | `main/theorem7_final_reproduction` |

## Execution and cache contract

The author-invoked analysis first resumes learned tasks, exits their workers,
then measures any missing vector cores and integrates their saved payloads.
The default Proposition 5 figure requires original V even without diagnostics.
An integrator change/missing integration can reuse core vectors and learned
probes. A missing/invalid core requires that record's raw tensors again.
Additional loss draw counts create new validated probe task identities;
deterministic prefixes exist, but prefix-shard reuse is not implemented.

The paper role stays at
`outputs/<base>/theory/experiment_S0_N<N>/`. Backing learned tasks are in
`outputs/<base>/theory_measurements/direct_probes/`; vector cores are in
`theory_measurements/direct_analysis/<core_hash>/`, with independent
`integrations/<integration_hash>/` results beneath them. `logical_tables.json`
registers scalar sources. No new top-level suite or `theory_v2` is created.

Paper bundle schema 2 / metric schema `direct-statements-1` rejects old behavior
plots as direct figures. Old scalar meanings, candidate data and historical
reports remain intact. The runtime publisher archives an old owned publication
only after its replacement is ready. Portable plotting uses compact CSVs and
small manifests and does not need backing trajectories, reference manifests,
models or CUDA. Omitted science settings inherit the saved configuration;
explicit conflicts require recomputation.

All posterior, mean and radius quantities use one declared law. Default cached
atoms do not establish the full training law. The single-target conditional
reference remains an assumed idealization. Missing mandatory loss blocks
publication; undefined mathematical quantities have explicit statuses. Tiny
positive terminal noise does not satisfy the exact clean-update assumption.
Constructed DDPM counterfactuals do not count as independent Lemma 4 checks.
Quadrature estimates and finite-sample plug-in probability bounds are not
rigorous statistical or interval certificates.

## Commands for the author (not executed)

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py

CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh \
  --model sdv1 --scheduler ddim --device auto \
  --recompute-experiments --num-loss-seeds 64 --loss-seed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh \
  --model sdv1 --scheduler ddim --device auto \
  --recompute-experiments --loss-timesteps saved \
  --num-loss-seeds 128 --num-unconditional-loss-seeds 256 --loss-seed 0

./run_all.sh --model sdv1 --scheduler ddim --plot

CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --device auto --recompute-experiments
```

These commands need the author's compatible environment and existing protected
caches. The all-native loss option intentionally adds substantially more learned
work. `PYTHON=/path/to/python` is preserved by every wrapper.
