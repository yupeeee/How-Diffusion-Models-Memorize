# Final paper figure curation

Presentation registry **`four-stage-paper-curation-13`** fixes twenty default figures for every supported model/sampler configuration: four main and sixteen appendix plots. The two zero-vector companions are retired from rendering, while their saved scalars and historical receipts remain intact. The per-prompt trajectories retain slots A10–A14; A15 retains the Corollary 3 guidance-fit/loss scatter and A16 appends the saved cross-step reference-variation curves without changing existing order or measurements. The selected μ still averages 10000 unconditional posterior clean references at the smallest SNR of the prespecified analytical extension on independent Gaussian inputs by default. Its estimation SNR and uncertainty remain separate from the exact finite-bank mean; explicit native-initial estimation and exact-bank selection remain available. Evaluation populations, support weights, seed streams, posterior/radius definitions and correction modes are unchanged. No tests, plots, migration, deletion or scientific computation were executed during this coding task.

## Fixed active order

| Main | Stable stem | Purpose | Supporting figures |
|---|---|---|---|
| 1 | `initial_loss_recovery` | Genuine forward-loss scale versus Gaussian target recovery, with paired unconditional controls | A1, A2 |
| 2 | `branch_gap_posterior_response` | First-update signed posterior response along the fixed guidance direction | A3, A4 |
| 3 | `branch_gap_synchronization` | Actual vector gap and same-sample joint target error over chronological predictions | A5 first, then A6, A7 |
| 4 | `terminal_bound_coverage` | Actual, corrected observable-bound and corrected reference-bound tolerance coverage | A8, A9 |

| Appendix | Stable stem | Purpose |
|---|---|---|
| A1 | `initial_unconditional_mean_concentration` | Unique Gaussian-seed and weighted distinct-atom distances about the same separately selected theory mean |
| A2 | `unconditional_reference_convergence` | Analytical reference convergence and separately supported native-noise learned quantities/errors |
| A3 | `posterior_feedback_over_time` | Full-range strict feedback, original-condition coverage and unresolved mass |
| A4 | `posterior_feedback_condition_margin` | Original sufficient margin versus signed matched gain |
| A5 | `branch_target_errors` | Separate paired conditional/unconditional target-error timing |
| A6 | `synchronization_bound` | Complete reference bound versus actual gap and joint target error |
| A7 | `branch_gap_peak_step` | Individual earliest-global-maximum timing, ties, incomplete curves and overlapping outcome groups |
| A8 | `terminal_observable_bound` | Actual endpoint error versus the corrected observable bound |
| A9 | `final_reproduction_bound` | Actual endpoint error versus the corrected reference bound |
| A10 | `branch_gap_per_prompt` | Mean per-seed branch-gap norm for each prompt–target pair, colored by its mean terminal SSCD |
| A11 | `conditional_reference_error_per_prompt` | Mean per-seed conditional clean-reference error on each actual trajectory |
| A12 | `unconditional_reference_error_per_prompt` | Mean per-seed learned unconditional error against its posterior clean reference |
| A13 | `target_probability_per_prompt` | Mean per-seed target posterior probability on actual saved trajectory states |
| A14 | `reference_branch_gap_per_prompt` | Mean per-seed analytical conditional-minus-unconditional reference-gap norm on actual saved states |
| A15 | `corollary3_guidance_scale_vs_loss` | Signed joint guidance coefficient versus sqrt of genuine forward loss divided by d*SNR_T |
| A16 | `reference_variation_per_prompt` | Mean per-seed cross-step reference variation on positive-noise transitions, with numerical-status audits |

Coverage moves from appendix to main. The reference-bound scatter moves from main to A9; it retains its own formula and identity. The tighter observable-bound scatter stays distinct as A8. Branch-resolved errors remain a prominent mandatory A5 rather than a fifth main figure. Peak timing remains even when outcome groups overlap.

A fully eligible configuration exports twenty PNGs and twenty single-page PDFs: eight image files in `main/`, thirty-two in `appendix/` (40 images, plus one captions file). Each file contains one independent plot. Missing required inputs, numerical errors and mathematical inapplicability have distinct statuses; no plot is fabricated to reach the file count. Explicitly supported optional diagnostics are outside these twenty entries. The learned counterfactual figure requires `--diagnostics`, counterfactual enablement in the saved bundle and available saved data; it renders only under `diagnostics/`, never as an additional appendix.

A10 uses every saved pre-update prediction index, with one arithmetic mean of normalized per-seed norms per pair and step. It does not take the norm of an averaged vector. Each curve uses one fixed color from mean terminal SSCD over the same complete seed cohort, with the shared [0,1] color scale. Pair identities and missing/nonfinite cohort reasons are retained in audit metadata. No terminal-output branch prediction or interpolation is fabricated.

A11–A13 add the conditional reference error, unconditional reference error, and target posterior probability for each prompt–target pair over chronological pre-update predictions `T-t`. Each curve uses the same full fixed seed cohort at every step and a fixed color from those seeds' mean terminal SSCD. The error curves average saved per-seed normalized norms, with no second division by `sqrt(d)`. Unconditional error compares the learned clean estimate with the posterior clean reference, not with the target. Probability is `mean(exp(direct_target_log_probability))`, computed per seed before averaging, on the actual saved generated trajectory state; `exp(mean(log_probability))` would be a different quantity. Missing or invalid trajectories exclude the entire pair from that figure and are audited; no curve averages a changing subset of seeds. These are compact reductions of existing scalar measurements, with no new posterior or network evaluation.

A14 `reference_branch_gap_per_prompt` shows $\|\bar{\mathbf{x}}_t(c)-\bar{\mathbf{x}}_t(\varnothing)\|/\sqrt{d}$ against $T-t$. The analytical reference difference is conditional minus unconditional at the same actual saved trajectory state and noise level. Under the existing single-target conditional assumption, $\bar{\mathbf{x}}_t(c)=\mathbf{x}^{\star}$, so its norm equals the saved opposite-sign distance $\|\bar{\mathbf{x}}_t(\varnothing)-\mathbf{x}^{\star}\|$. It therefore averages `direct_reference_target_error_rmse` once across the same complete fixed seed cohort, with a fixed color from those seeds' mean terminal SSCD. The learned branch gap, differences of error norms, and the radius-tail upper bound are different quantities and are not used. This adds no posterior or network evaluation; exclusions remain in `audit_data/reference_branch_gap_per_prompt_cohort.csv`.

The A11–A14 formula recipe is `prompt-trajectory-scalars-1`; each has a saved `audit_data/<stem>_cohort.csv` receipt. Existing support references A1–A9 and the relative order of all remaining figures are unchanged; slots following the retired zero companions close the two gaps. Missing compact inputs require explicit analysis-only reduction before plotting; the renderer never performs these reductions.

A15 `corollary3_guidance_scale_vs_loss` plots `sqrt(L_T(c)/(d*SNR_T))` against the signed least-squares guidance coefficient for Corollary 3, Equation 54. The compact `x` remains the genuine forward-target loss mean `L_T(c)/(d*SNR_T)`; only its display takes one square root, after averaging the loss over draws. It is not the mean of per-draw roots. For each prompt–target pair, the fit minimizes `sum_seed ||(xhat_T(c;g)-mu)-a*(x_star-mu)||^2` over one scalar `a`, using the full fixed saved initial seed cohort at the actual first-prediction states. Gaussian-bank match status is recorded separately and is not assumed for this fit. Its shared target direction makes the joint coefficient equal to the arithmetic mean of the signed per-seed coefficients. Color is mean terminal SSCD over exactly those seeds; the labeled horizontal guide shows the actual configured `g`. The fit uses the selected reference mean, retains negative coefficients, and excludes incomplete cohorts or unidentifiable directions with saved reasons. It is a projection of the full guided clean estimate around μ, not a projection of `g*Delta_T`, and it does not estimate or change the configured generation guidance.

A16 `reference_variation_per_prompt` shows $V_t/\sqrt{d}$ against $T-t$, using the saved Equation-15 cross-step reference-variation norm-integral estimates `direct_prop5_variation_rmse` from `matched_updates`. The integrand is defined relative to the current unconditional posterior reference $\bar{\mathbf{x}}_t(\varnothing)$, not the selected global μ. Each prompt–target curve averages the already normalized estimates over the same complete fixed seed cohort at every positive-noise transition, with a fixed color from those seeds’ mean terminal SSCD. Its domain is $t=2,\ldots,T$, or $T-t=0,\ldots,T-2$; no final-prediction or output value is inserted, and a one-step schedule is not applicable. Finite nonnegative original estimates are retained when condition signs or the quadrature budget remain unresolved; refined interval midpoints are not substituted. A missing, invalid or structurally inapplicable required estimate excludes the entire curve with an explicit reason. `audit_data/reference_variation_per_prompt_cohort.csv` records cohort exclusions and numerical-status counts, while `audit_data/feedback_endpoints.csv` retains the source status, error estimates and bounds. These are numerical estimates, not certified exact integrals. The new compact reduction performs no integration, posterior evaluation or model inference.

Its compact recipe is `corollary3-guidance-fit-1`, with full mean provenance, residual/direction diagnostics and `audit_data/corollary3_guidance_scale_vs_loss_cohort.csv`. No legacy standalone zero-centered coefficient is substituted.

## Measurements retained when renders are retired

Exactly four render families are retired from both the default registry and diagnostic/gallery dispatch:

- `branch_gap_motion_high_sscd`
- `branch_gap_motion_lower_sscd`
- `initial_unconditional_mean_concentration_zero`
- `unconditional_reference_convergence_zero`

The motion families retain their per-sample conditional/unconditional motion contributions, quadratic term, squared-gap change, target-error changes, identity residuals, common-row summaries, tests and historical provenance remain. No replacement motion figure is introduced. Retirement does not remove numerical dependencies, the peak distribution, reference-error curves or original-condition measurements.

The main coverage plot divides the existing eligible population into same-seed terminal SSCD > 0.75 and SSCD <= 0.75, then uses one common population and prompt-balanced weights within each group for `E_end<=tau`, `C_obs<=tau`, and `C_ref<=tau`, with `C_obs=B_obs+delta_sched` and `C_ref=B_ref+delta_sched`. The horizontal tolerance is `tau/sqrt(d)` once; the vertical axis is the manuscript-style `Pr(· ≤ tau)`, denoting the saved weighted fraction satisfying the tolerance. The six curves use a common tolerance axis; gold/purple identify group and solid/dashed/dash-dot identify actual/observable/reference. Each group normalizes separately with equal represented prompt mass. Missing/nonfinite scores and empty groups are audited explicitly; no sample silently joins the lower group. The full reference tail and exact zero mass remain visible. The original pooled curves and metadata remain saved as audit tables. Original clean-update prerequisites and finite-update corrections retain their distinct scope, including the saved probability contract where applicable.

Saved deterministic/pathwise row checks `E_end<=C_obs<=C_ref` remain independent of the expected coverage order `F_ref<=F_obs<=F_actual`. Violations are reported, never repaired by sorting or clamping curves. SSCD does not define a latent tolerance. Negative posterior responses, zero original-condition coverage, unresolved signs, wrong-target agreement and loose reference bounds remain in their supporting tables and figures.

The [manuscript notation vocabulary](figure_notation.md) uses plot recipe `four-stage-manuscript-notation-11`. A16 adds the existing scalar V_t definition to that vocabulary; all previous labels, data definitions and migration ownership rules remain unchanged.

## Saved-data presentation migration

For the first update adding A16, run `bash run_all.sh --recompute-experiments` to save its compact reduction and current analysis-source receipt, reusing compatible existing probes and variation integrals. Afterward, `bash run_all.sh --plot` is sufficient. Plot mode never backfills missing A16 measurements or summaries.

Canonical output stays under:

```text
outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/
```

The example SDv1 run remains `outputs/sdv1_ddim_g7.5_T50_N20/theory/experiment_S0_N20/`. A category change does not rename `plot_data/<stable_stem>.csv`, duplicate backing shards or change its numerical hash. Compatible four-stage compact bundles containing all required compact tables with current analysis-source receipts can update presentation manifests and render the new destinations with `--plot`, using the same renderer as normal execution. Omitted scientific options inherit saved values, including nondefault loss draw counts. Copied bundles retain `--bundle PATH --plot` support.

Plot-only migration starts no model/device workers and performs no inference, noise sampling, raw-tensor reading/hashing, posterior calculation, integration, support fitting, scalar summary recomputation or scientific reclassification. Genuinely missing or incompatible required scalars produce the missing field/reason and an exact analysis-only recomputation command. Presentation changes alone do not require recomputation. Older bundles missing A10 or the grouped terminal-coverage table need explicit analysis-only reduction from saved scalars with `--recompute-experiments`; compatible expensive measurements are reused, and plot mode never backfills this table.

The peak median and quartiles are an additive descriptive summary produced during explicit analysis from saved weighted peak bins (`audit_data/trajectory_peak_summary.csv`, recipe `saved-weighted-peak-bins-1`, with matching figure metadata). They use the inverse weighted CDF conditional on resolved individual peaks; the unchanged histogram, initialization mass and unresolved mass retain the original full denominator. Compatible older bundles remain plottable without this summary. Their captions disclose its absence and provide an exact analysis-only command to add it; plot mode never computes it. Adding the summary is not required to migrate figure placement and does not remeasure trajectories.

Main and supporting captions keep the verified printed references to `revised.pdf`, the distinction between the declared empirical law and an identified training marginal, and the existing endpoint/correction qualifications. Matching LaTeX was not available in the previously inspected manuscript sources; no new source labels or theorem changes are inferred from this curation.

## Exact retirement paths and transaction rules

After replacement PNG/PDF pairs have been staged, validated and installed successfully, the authorized category moves retire:

| Old pair | Active replacement pair |
|---|---|
| `appendix/terminal_bound_coverage.{png,pdf}` | `main/terminal_bound_coverage.{png,pdf}` |
| `main/final_reproduction_bound.{png,pdf}` | `appendix/final_reproduction_bound.{png,pdf}` |

The retired pairs are `appendix/branch_gap_motion_high_sscd.{png,pdf}`, `appendix/branch_gap_motion_lower_sscd.{png,pdf}`, `appendix/initial_unconditional_mean_concentration_zero.{png,pdf}`, and `appendix/unconditional_reference_convergence_zero.{png,pdf}`. The same exact stems in `main/` or `diagnostics/` are candidates only if a previous manifest explicitly proves renderer ownership. These paths are enumerated; no `branch_gap_*` glob or blanket image-directory purge is allowed.

Every retirement candidate must be a regular file inside the requested active bundle, at an authorized path, with a hash matching the previous ownership manifest. Traversal, symbolic links, absent ownership evidence, manual modifications and unrelated user files are preserved and reported as conflicts. Historical archives outside the active bundle are outside this migration.

The existing role lock and staged publication protect image pairs and publication metadata together. All requested replacements render before retirement. A failed PNG, PDF or publication step restores the previous valid pairs and metadata; a partially replaced pair is never successful completion. Temporary rollback copies are removed after commit. Protected caches, generation images, target latents, forward losses, scalar shards, plot-data CSVs, empirical-law manifests, SSCD/proximity data and historical audits are never retirement targets.

`figure_retirement.json` records each old path, stable identity, previous hash, destination when moved, decision/reason and completion/conflict status. Repeated `--plot` is idempotent: one active pair per stable stem, no resurrected motion or zero-companion renders, suffix variants, duplicate placements or newly accumulated retirement archives.

## Author commands — not executed here

Render this presentation-only square-root change from a bundle that already has the guidance-fit table and current measurement-source receipts:

```bash
bash run_all.sh --model sdv1 --scheduler ddim --plot
```

Omit `--model` and `--scheduler` only to refresh the full four-configuration matrix. Preflight checks every requested configuration and lists all failures before any renderer runs. One outdated configuration blocks the requested matrix; use the printed analysis-only command for each stale bundle, or explicitly select a compatible configuration. The error names the bundle and differing measurement-source files. The new Corollary 3 table must already exist; an axis or legend change alone does not invalidate it.

The saved `x=L_T(c)/(d*SNR_T)`, fit coefficients, scientific recipe `corollary3-guidance-fit-1`, and measurement hashes stay unchanged. The renderer displays `sqrt(saved x)` and records that transform in the figure audit.

Only older bundles missing the fit fields or requiring a measurement-source update need:

```bash
bash run_all.sh --recompute-experiments
```

This analysis reuses compatible measurement shards and adds no generation rollout. Existing source-hash guards remain active; plot mode never reconstructs missing fits.

A copied bundle must also contain the updated compact inputs and matching source receipt:

```bash
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

Analysis-only recomputation resumes compatible expensive work and bypasses upstream generation, SSCD and selection rebuilding. Upstream `--overwrite` keeps its separate meaning. The actual runtime figure/retirement manifests are written when the author runs these commands; this document does not claim a completed export or migration.

The retired zero companions retain saved scalar measurements and historical receipts. New runs reject `--no-mu` and `--center zero`; historical saved configurations remain readable. The selected theory mean still defaults to `reference-min-snr`. The convergence offset guide remains, with its meaning and value in the caption/audit rather than a “Reference limit” legend entry.
