# How Diffusion Models Memorize

Diffusion memorization experiments with preserved generation, SSCD and proximity caches. The default theory stage measures **four main figures** for the mechanism experiments and sixteen mandatory appendix figures. Analysis may run missing learned denoiser probes; plotting uses saved compact scalars only.

## Run

```bash
python -m pip install -r requirements.txt
./run_all.sh --download                      # Prepare data and resume the normal pipeline
./run_all.sh                                 # All four supported configurations
./run_all.sh --model sdv1 --scheduler ddim    # One configuration
./run_all.sh --recompute-experiments          # Theory only, including missing learned probes
./run_all.sh --refine-numerics                # Saved evidence arithmetic only; no learned probes
./run_all.sh --plot                          # Saved proximity and paper figures
./run_all.sh --plot --diagnostics            # Also export available behavioral diagnostics
```

Defaults remain guidance 7.5, 50 updates, 20 evaluation seeds and GMM selection. Supported pairs are SDv1/DDIM, SDv1/DDPM, SDv2/DDIM and RealVis/DDIM. Evaluation uses seeds `0..N-1`; reference selection uses `N..2N-1`. Every evaluation seed of each retained prompt remains included. All wrappers honor `PYTHON`.

`--recompute-experiments` bypasses download, generation, SSCD and proximity. It resumes valid independent probe and analytical shards, and computes missing work. `--refine-numerics` resumes numerical signs and original-condition intervals from existing evidence, bypassing learned probes as well as upstream stages. Missing inputs report an exact recomputation command. Normal derived analysis includes the numerical policy automatically. `--overwrite` retains its explicit upstream regeneration meaning. Plot mode rejects these rebuilding options and never invokes models, raw tensor readers, posterior integration or scalar reducers. It validates all requested bundles before rendering and reports every incompatible configuration before stopping without changing figures. Bare `--plot` selects all four configurations: one stale bundle blocks the matrix even if the others are current. Use `./run_all.sh --model sdv1 --scheduler ddim --plot` to refresh just that saved bundle, or `--model sdv1 --plot` for both SDv1 schedulers. A measurement-source mismatch names the affected bundle and changed files. Missing measurements still require the printed analysis-only command; plotting cannot create a missing guidance-fit table.

## Direct measurement options

| Option | Analysis default | Meaning |
|---|---:|---|
| `--mean-source` | `reference-min-snr` | Analytical reference average at the minimum reference-grid SNR; `reference-initial` retains the native first-step estimator, and `cached-targets` uses the exact declared bank mean |
| `--num-mean-samples` | 10000 | Independent standard Gaussian inputs for the shared reference-mean estimate |
| `--mean-seed` | 0 | Dedicated reference-mean RNG root, separate from evaluation and loss draws |
| `--num-loss-seeds` | 64 | Independent forward-target noise draws per pair and measured timestep |
| `--loss-seed` | 0 | Root for domain-separated deterministic probe streams |
| `--loss-timesteps` | `initial` | `initial` or all `saved` native prediction timesteps |
| `--num-unconditional-loss-seeds` | 256 | Optional forward-marginal draws; an explicit count enables the stage |
| `--unconditional-loss` / `--no-unconditional-loss` | enabled for `saved` | Override the optional marginal-loss stage |
| `--counterfactual-unconditional` | off | Separate learned empty-prompt comparison at matched next inputs; no rollout |
| `--counterfactual-steps` | `0` | Fixed comma-separated chronological update indices for the enabled supplement |
| `--probe-batch-size` | 8 | Execution batch size, with bounded CUDA OOM retry |
| `--reference-law` | `cached-targets` | Uniform distinct compatible cached targets before selection, or `manifest` |
| `--reference-manifest` | absent | Required local atoms/weights/provenance manifest for `manifest` mode |
| `--reference-snr-decades` | 6 | Depth of the fixed 97-point analytical-only grid below initial SNR; no network calls there |
| `--terminal-noise-run-alpha` | 0.05 | Predeclared simultaneous terminal Gaussian noise failure budget within one scientific run |
| `--numerical-max-products` | 2000000 | CUDA interval-operation budget per row; old `--numerical-max-decimal-products` spelling remains an alias; zero disables interval refinement |
| `--numerical-decimal-precision` | legacy metadata only | Computation rejects this override: the GPU enclosure backend uses binary64, not arbitrary Decimal digits |
| `--numerical-max-variation-nodes` | 65 | Per-row variation enclosure node budget |
| `--numerical-variation-absolute-width` | 1e-6 | Requested enclosure width in raw latent L2 units |

Gaussian probes use a single fixed bank of the `N` evaluation seeds on every saved native timestep. These are distinct from generated trajectory states and independent forward-target/marginal loss draws. Theorem 1 shows pair-level conditional RMS and a matched unconditional control, with independent forward-loss scale and saved Monte Carlo bootstrap intervals. Pair means use mean terminal SSCD colors; raw seed tails remain in audit tables.

Posterior references, target probabilities and radii use one declared law. By default, the comparison/injection vector `mu` is the average of 10,000 analytical unconditional posterior clean references evaluated at the smallest SNR in the existing analytical-only reference grid on independent standard Gaussian inputs. With `--reference-snr-decades 6`, this SNR is the initial native SNR multiplied by `10^-6`. Only the mean estimator uses this selected level; native prediction timesteps and the reference sweep stay unchanged. The vector is estimated once per bank/schedule/selected-SNR/seed identity, then reused at every measured timestep. Its computation uses every visible CUDA device with one coordinator progress bar; no denoiser or VAE inference is needed. `--mean-source cached-targets` explicitly selects the exact weighted bank mean instead. The cached-target law is qualified as `D_K`; it is not claimed to identify the checkpoint's full training distribution. Exact duplicate target atoms receive equal distinct-atom mass by default. The single-target conditional reference is an assumed idealization; repeated exact prompts with different targets are audited and retained. Legacy `--center` options never redefine this law's exact mean.

For this finite reference law, each analytical posterior mean converges to the exact weighted atom mean as SNR approaches zero. Using the smallest grid SNR reduces finite-SNR bias, while a finite SNR and sample count can still leave a nonzero offset. The estimator saves its Monte Carlo standard error, even/odd split agreement, selected analytical SNR and native initial noise level separately. The exact atom mean is retained for posterior identities and analytical limits, including the measured bank-to-selected-mean offset in `unconditional_reference_convergence`. Its uncertainty measures finite-sample error only. Neither vector identifies the full training-data mean from this finite bank.

The immutable full-vector cache is under `theory_measurements/reference_mean/`; `theory_mean.json` in the paper bundle records the selected estimate. Completed logical shards are reusable across restarts and larger draw budgets. To run just this estimator, without theory reduction or figure rendering:

```bash
bash theory_validation.sh --model sdv1 --scheduler ddim --estimate-mean-only --mean-source reference-min-snr --num-mean-samples 10000
```

Normal analysis prepares and uses the estimate automatically. Existing first-step estimates have a different cache identity; migrate their dependent theory measurements and figures with `bash run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --mean-source reference-min-snr`. Plot-only mode retains the mean recorded in each saved bundle. Increase `--num-mean-samples` to increase the independent mean-estimation budget. GPU computation has no CPU fallback.

## Devices and reuse

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --device auto --recompute-experiments --num-loss-seeds 64 --loss-seed 0
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --device auto --recompute-experiments --loss-timesteps saved --num-loss-seeds 128 --num-unconditional-loss-seeds 256 --loss-seed 0
./run_all.sh --model sdv1 --scheduler ddim --plot
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --device auto --recompute-experiments
PYTHON=/path/to/python ./run_all.sh --plot
```

`auto` uses all visible CUDA devices. Theory computation requires NVIDIA CUDA and rejects CPU/MPS execution; there is no CPU numerical fallback. Saved plotting and validation remain available without CUDA. Workers follow the shared device/sharding conventions and send progress to one parent-owned bar per stage. Fast vector cores and four-stage response/motion shards finish before expensive integration. Inference and analytical pools never compete for full model replicas. The direct stage reads each unfinished protected record once for its vector comparisons and preserves independently resumable Proposition 5 integrals. Missing evidence supplements read the preserved record separately for initial/final vectors, without repeating those histories or integrals.

Numerical policy `fixed-cache-numerics-cuda-3` batches the outward-rounded condition screen on the GPU, then refines unresolved rows with CUDA interval arithmetic. Raw input reconstruction, posterior/gain evaluation, variation enclosures, quadrature reductions, and learned-probe error reductions stay on the selected worker GPU. The exp/log enclosures use bounded series, not an assumed one-ULP bound on vendor transcendental kernels. Binary64 enclosures retain unresolved signs whenever precision or work limits prevent resolution; more work does not add decimal digits.

Host work remains necessary for file I/O, worker scheduling, small control decisions, scalar table aggregation and plotting. Original seeded CPU random streams are retained as input generation and transferred before numerical evaluation, preserving existing experimental inputs. This is not a promise of zero CPU utilization. See [GPU execution and budget controls](docs/theory/gpu_theory_computation.md).

Numerical receipts are checked before loading raw witnesses, and completed batches remain resumable. The GPU implementation has new source/policy identities; incompatible derived measurements are rebuilt while old caches and protected generation/SSCD/proximity observations remain preserved. `--refine-numerics` can rebuild analytical response shards when its GPU operation budget changes, but cannot backfill learned probes or missing core observations. The first migration from older source receipts may require `--recompute-experiments`.

Probe tasks and analytical recipes have independent cache identities. An integrator change or missing integral does not repeat learned probes. Vector cores and integration payloads have separate completion markers. Missing or changed quadrature reads saved sufficient statistics; within the direct stage, only incomplete vector cores require raw-record reloads. Successful records remain reusable. Changing loss draw count gives new probe task identities; the deterministic RNG prefix is preserved, but partial prefix reuse is not implemented. Rendering styles do not invalidate numerical task caches.

Derived recomputation preserves omitted saved loss-draw settings. Plotting inherits omitted scientific settings from the saved bundle, including loss draw counts, timestep mode and reference law. Explicit conflicting settings fail with a recompute command. Repository model/scheduler/g/T/N identities remain checked. A portable bundle needs only its compact scalar tables and manifests:

```bash
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

## Paper outputs

Canonical publication remains `outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/`:

```text
run_config.json, summary.json, audit.json, theory_mean.json
registry.json, figure_manifest.json, figure_captions.md, figure_retirement.json
initial.csv, terminal.csv, failed.csv, logical_tables.json
plot_data/*.csv, audit_data/*.csv
main/*.png, main/*.pdf
appendix/*.png, appendix/*.pdf
diagnostics/                 # Available optional behavioral figures
```

| Experiment | Main figure stem |
|---|---|
| Initial conditional recovery and unconditional baseline | `initial_loss_recovery` |
| Retained branch-gap contribution and matched evidence, fixed first update | `branch_gap_posterior_response` |
| Branch-gap dynamics and joint target accuracy | `branch_gap_synchronization` |
| Actual, observable-bound and reference-bound coverage over latent tolerances | `terminal_bound_coverage` |

The sixteen mandatory appendix figures have this fixed order:

| Appendix | Stable stem |
|---|---|
| A1 | `initial_unconditional_mean_concentration` |
| A2 | `unconditional_reference_convergence` |
| A3 | `posterior_feedback_over_time` |
| A4 | `posterior_feedback_condition_margin` |
| A5 | `branch_target_errors` |
| A6 | `synchronization_bound` |
| A7 | `branch_gap_peak_step` |
| A8 | `terminal_observable_bound` |
| A9 | `final_reproduction_bound` |
| A10 | `branch_gap_per_prompt` |
| A11 | `conditional_reference_error_per_prompt` |
| A12 | `unconditional_reference_error_per_prompt` |
| A13 | `target_probability_per_prompt` |
| A14 | `reference_branch_gap_per_prompt` |
| A15 | `corollary3_guidance_scale_vs_loss` |
| A16 | `reference_variation_per_prompt` |

A fully eligible default configuration exports twenty PNG/single-page-PDF
pairs (40 image files): eight in `main/` and thirty-two in `appendix/`, plus
`figure_captions.md` (41 owned publication files). Missing inputs
and mathematical inapplicability retain distinct statuses rather than fabricated
plots. Corollary 3 has the additional fitted-coefficient view A15; its vector audits and the Lemma 4 audits remain. The two
`branch_gap_motion_high_sscd` and `branch_gap_motion_lower_sscd` render families
are retired, including from `--diagnostics`; their scalar contributions, identity
checks and provenance remain. Other explicitly supported saved diagnostics remain
optional.

The two zero-vector companion figures are retired from the default suite and diagnostics; their saved scalar measurements remain intact. New runs reject `--no-mu` and `--center zero`; historical saved configurations remain readable. The selected theory mean still defaults to `reference-min-snr`. The convergence figure keeps its offset guide and caption/audit value without a “Reference limit” legend entry.

The new A16 compact table requires one analysis update. Run the following to reduce the saved variation measurements and update the recorded analysis-source receipts:

```bash
bash run_all.sh --recompute-experiments
```

This update reuses compatible saved probes and variation integrals; A16 adds no new integration or model inference. Existing hash guards remain in force. The earlier A15 square-root display alone did not change its saved loss ratio or `corollary3-guidance-fit-1`. After the required compact tables and source receipts are current, `bash run_all.sh --plot` renders them without scientific computation. The command skips upstream generation, SSCD and selection rebuilding. Owned obsolete figure files are retired only after successful publication and ownership/hash checks.

The baseline report is saved as `initial_baseline_summary.csv` and `.json`.
The posterior response retains the fixed grid `j/40` plus `1/g`, signed H,
negative responses and fixed prompt-balanced SSCD groups. Synchronization uses
chronological prediction indices (0 is initialization), paired D and Q, and no
fabricated prediction at the final output. Temporal shape is measured, not assumed.

`appendix/branch_gap_per_prompt` shows one curve per retained prompt–target pair: the mean of the saved per-seed branch-gap norms divided by `sqrt(d)`, versus chronological prediction index `T-t`. Color is mean terminal SSCD over the same complete seed population, on the fixed [0,1] scale. Missing or nonfinite seed trajectories are reported explicitly; they are not silently averaged over fewer seeds. This adds a compact scalar reduction without new model inference. Older bundles missing this table require analysis-only reduction via `--recompute-experiments` before `--plot`; compatible measurements are reused.

A11–A13 add the conditional reference error, unconditional reference error, and target posterior probability for each prompt–target pair over chronological pre-update predictions `T-t`. Each curve uses the same full fixed seed cohort at every step and a fixed color from those seeds' mean terminal SSCD. The error curves average saved per-seed normalized norms, with no second division by `sqrt(d)`. Unconditional error compares the learned clean estimate with the posterior clean reference, not with the target. Probability is `mean(exp(direct_target_log_probability))`, computed per seed before averaging, on the actual saved generated trajectory state; `exp(mean(log_probability))` would be a different quantity. Missing or invalid trajectories exclude the entire pair from that figure and are audited; no curve averages a changing subset of seeds. These are compact reductions of existing scalar measurements, with no new posterior or network evaluation.

A14 `reference_branch_gap_per_prompt` shows $\|\bar{\mathbf{x}}_t(c)-\bar{\mathbf{x}}_t(\varnothing)\|/\sqrt{d}$ against $T-t$. The analytical reference difference is conditional minus unconditional at the same actual saved trajectory state and noise level. Under the existing single-target conditional assumption, $\bar{\mathbf{x}}_t(c)=\mathbf{x}^{\star}$, so its norm equals the saved opposite-sign distance $\|\bar{\mathbf{x}}_t(\varnothing)-\mathbf{x}^{\star}\|$. It therefore averages `direct_reference_target_error_rmse` once across the same complete fixed seed cohort, with a fixed color from those seeds' mean terminal SSCD. The learned branch gap, differences of error norms, and the radius-tail upper bound are different quantities and are not used. This adds no posterior or network evaluation; exclusions remain in `audit_data/reference_branch_gap_per_prompt_cohort.csv`.

Older bundles missing A11–A14 require `bash run_all.sh --recompute-experiments` to save their compact tables before `--plot`; compatible existing measurements are reused. Plot-only never reconstructs these summaries.

A15 `corollary3_guidance_scale_vs_loss` plots `sqrt(L_T(c)/(d*SNR_T))` against the signed least-squares guidance coefficient for Corollary 3, Equation 54. The compact `x` remains the genuine forward-target loss mean `L_T(c)/(d*SNR_T)`; only its display takes one square root, after averaging the loss over draws. It is not the mean of per-draw roots. For each prompt–target pair, the fit minimizes `sum_seed ||(xhat_T(c;g)-mu)-a*(x_star-mu)||^2` over one scalar `a`, using the full fixed saved initial seed cohort at the actual first-prediction states. Gaussian-bank match status is recorded separately and is not assumed for this fit. Its shared target direction makes the joint coefficient equal to the arithmetic mean of the signed per-seed coefficients. Color is mean terminal SSCD over exactly those seeds; the labeled horizontal guide shows the actual configured `g`. The fit uses the selected reference mean, retains negative coefficients, and excludes incomplete cohorts or unidentifiable directions with saved reasons. It is a projection of the full guided clean estimate around μ, not a projection of `g*Delta_T`, and it does not estimate or change the configured generation guidance.

A16 `reference_variation_per_prompt` shows $V_t/\sqrt{d}$ against $T-t$, using the saved Equation-15 cross-step reference-variation norm-integral estimates `direct_prop5_variation_rmse` from `matched_updates`. The integrand is defined relative to the current unconditional posterior reference $\bar{\mathbf{x}}_t(\varnothing)$, not the selected global μ. Each prompt–target curve averages the already normalized estimates over the same complete fixed seed cohort at every positive-noise transition, with a fixed color from those seeds’ mean terminal SSCD. Its domain is $t=2,\ldots,T$, or $T-t=0,\ldots,T-2$; no final-prediction or output value is inserted, and a one-step schedule is not applicable. Finite nonnegative original estimates are retained when condition signs or the quadrature budget remain unresolved; refined interval midpoints are not substituted. A missing, invalid or structurally inapplicable required estimate excludes the entire curve with an explicit reason. `audit_data/reference_variation_per_prompt_cohort.csv` records cohort exclusions and numerical-status counts, while `audit_data/feedback_endpoints.csv` retains the source status, error estimates and bounds. These are numerical estimates, not certified exact integrals. The new compact reduction performs no integration, posterior evaluation or model inference.

The compact fit table and `audit_data/corollary3_guidance_scale_vs_loss_cohort.csv` are written during analysis; `--plot` never joins losses, fits vectors, or silently substitutes the old standalone Corollary 3 output.

The terminal primary compares same-seed terminal SSCD > 0.75 and SSCD <= 0.75.
Gold/purple identify the groups; solid/dashed/dash-dot lines identify actual error,
the corrected observable bound, and the corrected reference bound. Each group uses
one common eligible sample population and prompt-balanced weights for its three
curves; both groups share the full tolerance axis. Group probabilities normalize
separately, with equal mass per represented prompt divided among its eligible seeds.
Exact-threshold scores belong to the lower group. Missing scores, empty groups,
zero mass and infinite-bound mass retain explicit audits. The original pooled
curves remain in `audit_data/terminal_pooled_cdf.csv`. Older bundles containing
only pooled coverage require `--recompute-experiments` to save the group summaries
from compatible cached measurements before `--plot`; plotting never computes them. A8 and A9 retain the separate observable/reference scatter
comparisons, including loose bounds. Non-clean samplers require the independently
qualified finite-step correction; supported Gaussian corrections retain their
saved probability contract. Row-level inequalities remain numerical audits;
curve ordering alone does not establish them.

The optional learned supplement is a separate, explicit inference task:

```bash
./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto --counterfactual-unconditional
```

It uses matched next inputs at a fixed snapshot, not a new rollout or decode.
`--plot --diagnostics` renders this supplement under `diagnostics/` only when
counterfactual analysis was enabled in the saved bundle and its data are available.
It never adds an appendix figure or backfills inference; the paper suite stays
four main figures and sixteen appendices.

Existing learned tasks and direct analytical/integration caches retain their
identities. New bounded analytical tasks use
`theory_measurements/reference_analytical/<hash>/`; additive geometry/terminal
scalars use `theory_measurements/evidence_supplements/<hash>/`. Missing supplements
read only preserved vectors and existing scalars; they do not repeat compatible
model probes or original variation integration. No active stage creates `theory_v2`.
Numerical sufficient inputs and policy retries are separately resumable under
`theory_measurements/numerical_refinement/`; old classifications remain preserved.
Paper bundle schema 5 and compact metric schema `four-stage-evidence-1` remain
unchanged for this curation. Presentation registry `four-stage-paper-curation-13`
reuses compatible saved `plot_data/<stable_stem>.csv` and scientific hashes.
Placement and legend changes alone do not require measurement recomputation.

Theory uses one parent-owned tqdm bar for live record progress, without periodic “still running” messages. Aggregation, plot-input, and publication stages report their start and completion once, including elapsed time on completion. Saving scalar tables and publication metadata uses one progress bar that advances per completed file and shows the current filename. Figure preparation has its own progress bar; PNG/PDF export advances once per saved file and shows the current filename.

Role locking, staged publication and paired PNG/PDF export remain in place.
`--plot` migrates compatible saved bundles using compact tables and metadata only.
It stages replacement pairs before retiring enumerated old locations whose hashes
match previous renderer ownership. Unowned, modified, symbolic-link or unrelated
files are preserved and reported as conflicts. A failed export/publication rolls
back both images and publication metadata. Successful migration writes the small
`figure_retirement.json` ledger; repeated plotting creates no new retirement
archives or duplicate pairs. Scientific tables, cache receipts and historical
archives are unchanged. See [the curation and migration rules](docs/theory/figure_curation.md).
Plot mode does not access backing numerical shards. Axes and legends use the mathematical notation in `revised.pdf`, with bold vectors, single-line division and named reference lines. Colorbars read `SSCD`; pair-mean aggregation stays in the initial figure caption. See [the figure notation mapping](docs/theory/figure_notation.md).

## Author validation

This fixed evidence redesign was prepared under an **edit-code-only** instruction. No new measurements, tests, plots, GPU runs or data-dependent checks were executed. Historical reports document earlier implementations and do not validate this change.

Commands for the author, after review:

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
```

See [the active four-stage plan and manuscript mapping](docs/theory/four_stage_experiments.md), [the numerical refinement plan](docs/theory/numerical_refinement_plan.md), [the fixed evidence plan](docs/theory/final_evidence_plan.md), [required manuscript alignment](docs/theory/submission_alignment.md), [the earlier implementation report](docs/theory/direct_implementation.md), [the direct pipeline guide](docs/theory/paper_pipeline.md), [the quantity dictionary](docs/theory/quantity_dictionary.md), and [the figure registry](utils/experiments/theory/paper_registry.py). The manuscript mapping is recorded in [statement_registry.json](docs/theory/statement_registry.json).
