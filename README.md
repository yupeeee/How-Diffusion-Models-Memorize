# How Diffusion Models Memorize

The active contract `projected-gap-error-1` uses the signed error $\mathcal{E}_t=\mathbf{u}_t^\top(\boldsymbol{\Delta}_t-\bar{\boldsymbol{\Delta}}_t)$, where $\mathbf{u}_t=\boldsymbol{\Delta}_t/\|\boldsymbol{\Delta}_t\|$ is computed from the actual learned gap. It may be negative and is never replaced by a norm, absolute value or clipped scalar. The margin is $\|\boldsymbol{\Delta}_t\|-\mathcal{E}_t-\mathcal{V}_t$. The variation remains $\mathcal{V}_t=[\int_0^1 \mathbf{u}_t^\top(\bar{\mathbf{x}}_{t-1}(\mathbf{x}^{\mathrm{cf}}_{t-1}+sg\kappa_t\boldsymbol{\Delta}_t,\varnothing)-\bar{\mathbf{x}}_t(\mathbf{x}_t,\varnothing))\,ds]_+$ for $t=2,\ldots,T$, with positive part after the signed integral. Exactly zero gap uses the explicit computational extension $\mathbf{u}_t=\mathbf{0}$, $\mathcal{E}_t=0$ and $\mathcal{V}_t=0$, without claiming a unit direction. Older bundles without this scientific contract require `bash run_all.sh --recompute-experiments`; historical gap-error norms cannot be relabeled by `--plot`. The six-figure placement and opacity update alone uses `--plot` on compatible current bundles. Compatible independent model probes and protected generation/SSCD/proximity data remain reusable.

Diffusion memorization experiments with preserved generation, SSCD and proximity caches. The theory stage retains the full measurement suite and publishes **six selected PDFs** under the project-root `figures/<experiment>/theory/experiment_S0_N<N>/` directory. Analysis may run missing learned denoiser probes; plotting uses saved compact scalars only.

## Run

```bash
python -m pip install -r requirements.txt
./run_all.sh --download                      # Prepare data and resume the normal pipeline
./run_all.sh                                 # All four supported configurations
./run_all.sh --model sdv1 --scheduler ddim    # One configuration
./run_all.sh --recompute-experiments          # Theory only, including missing learned probes
./run_all.sh --refine-numerics                # Opt-in interval certification; no learned probes
./run_all.sh --plot                          # Saved proximity and paper figures
./run_all.sh --plot --diagnostics            # Compatibility flag; the same six theory figures
```

Defaults remain guidance 7.5, 50 updates, 20 evaluation seeds and GMM selection. Supported pairs are SDv1/DDIM, SDv1/DDPM, SDv2/DDIM and RealVis/DDIM. Evaluation uses seeds `0..N-1`; reference selection uses `N..2N-1`. Every evaluation seed of each retained prompt remains included. All wrappers honor `PYTHON`.

On successful completion, `run_all.sh` prints the total wall-clock time and the combined final size of files created or updated during that invocation, including `--plot` runs. It compares file size and modification time under project-local `data/`, `logs/`, `outputs/`, `figures/`, and `checkpoints/`; unchanged reused caches, deleted files, symlinks, and temporary staging/lock files are excluded. The reported size is logical file bytes, not net disk growth or the total size of all existing outputs. File accounting reads metadata only. If accounting fails, the size is reported as unavailable without changing the pipeline result.

`--recompute-experiments` bypasses download, generation, SSCD and proximity. It resumes valid independent probe and analytical shards, and computes missing work. `--refine-numerics` opts into interval certification from existing evidence, bypassing learned probes as well as upstream stages. It reuses a saved positive operation budget or selects 2,000,000 operations per row when none is saved. Missing inputs report an exact recomputation command. Normal and `--recompute-experiments` analysis use numerical point estimates with interval work disabled, even if the previous bundle enabled it. An explicit `--numerical-max-products` overrides this mode choice; positive values enable intervals and zero disables them. Plotting and validation preserve the saved mode. `--overwrite` retains its explicit upstream regeneration meaning. Plot mode rejects these rebuilding options and never invokes models, raw tensor readers, posterior integration or scalar reducers. It validates all requested bundles before rendering and reports every incompatible configuration before stopping without changing figures. Bare `--plot` selects all four configurations: one stale bundle blocks the matrix even if the others are current. Use `./run_all.sh --model sdv1 --scheduler ddim --plot` to refresh just that saved bundle, or `--model sdv1 --plot` for both SDv1 schedulers. A measurement-source mismatch names the affected bundle and changed files. Missing measurements still require the printed analysis-only command; plotting cannot create a missing guidance-fit table.

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
| `--reference-law` | `cached-targets` | Equal mass per compatible complete cached source record before selection, with duplicate atom masses combined; or explicit `manifest` weights |
| `--reference-manifest` | absent | Required local atoms/weights/provenance manifest for `manifest` mode |
| `--reference-snr-decades` | 6 | Depth of the fixed 97-point analytical-only grid below initial SNR; no network calls there |
| `--terminal-noise-run-alpha` | 0.05 | Predeclared simultaneous terminal Gaussian noise failure budget within one scientific run |
| `--numerical-max-products` | 0 | Optional CUDA interval-operation budget per row; positive values enable certification work; old `--numerical-max-decimal-products` spelling remains an alias |
| `--numerical-decimal-precision` | legacy metadata only | Computation rejects this override: the GPU enclosure backend uses binary64, not arbitrary Decimal digits |
| `--numerical-max-variation-nodes` | 65 | Per-row variation enclosure node budget |
| `--numerical-variation-absolute-width` | 1e-6 | Requested enclosure width in raw latent L2 units |

Gaussian probes use a single fixed bank of the `N` evaluation seeds on every saved native timestep. These are distinct from generated trajectory states and independent forward-target/marginal loss draws. Theorem 1 shows pair-level conditional RMS and a matched unconditional control, with independent forward-loss scale and saved Monte Carlo bootstrap intervals. Pair means use mean terminal SSCD colors; raw seed tails remain in audit tables.

Posterior references, target probabilities and radii use one declared law. By default, the comparison/injection vector `mu` is the average of 10,000 analytical unconditional posterior clean references evaluated at the smallest SNR in the existing analytical-only reference grid on independent standard Gaussian inputs. With `--reference-snr-decades 6`, this SNR is the initial native SNR multiplied by `10^-6`. Only the mean estimator uses this selected level; native prediction timesteps and the reference sweep stay unchanged. The vector is estimated once per bank/schedule/selected-SNR/seed identity, then reused at every measured timestep. Its computation uses every visible CUDA device with one coordinator progress bar; no denoiser or VAE inference is needed. `--mean-source cached-targets` explicitly selects the exact weighted bank mean instead. The cached-target law is qualified as `D_K`; it is not claimed to identify the checkpoint's full training distribution. Each compatible complete cached source record receives equal prior mass before retained-prompt selection. Exact duplicate latent atoms are merged computationally and receive the sum of their record masses: an atom appearing n times among N source records has prior n/N. A source record is counted once by source ID, independently of its number of generation seeds; repeated references to the same source ID do not create extra mass, while distinct source IDs with identical latents do; these frequencies describe the available cache, not unknown full-training frequencies. The single-target conditional reference is an assumed idealization; repeated exact prompts with different targets are audited and retained. Legacy `--center` options never redefine this law's exact mean.

This is the empirical distribution of the available source records. It matches the manuscript's general atom-prior formulation: Equation 58 defines each atom's probability, and Equation 60 multiplies that prior by its Gaussian likelihood. Uniform distinct-atom weighting was a different prior. Explicit manifest weights remain authoritative and are summed when their atoms coincide. The conditional single-target reference remains a separately stated assumption.

Reference-law version `direct-reference-law-2` records `source_record_count` and per-atom `source_record_multiplicities`. Its source-record prior changes the law identity. Posterior means, target probabilities, reference errors, directional variations, and dependent margins/bounds must be recomputed under it; the selected reference-based μ estimator is naturally invalidated by that identity and also reruns. The exact weighted bank mean can change, while the support-only maximum radius does not change when support atoms are unchanged. Run `bash run_all.sh --recompute-experiments` once. Protected generation, SSCD and proximity data remain intact; compatible independent learned measurements remain reusable. Plot-only mode cannot convert historical equal-distinct-atom results into the new prior.

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

Numerical policy `fixed-cache-numerics-cuda-8` displays ordinary float64 estimates by default. Condition signs use the saved margin `M=D-E-mathcal{V}`, assessed against the saved signed-integral embedded quadrature error, projection roundoff and float64 subtraction allowance. Clear positive or negative assessments contribute to figure fractions. Near-zero, invalid or missing assessments remain unresolved or unavailable in their audit classifications. The condition-margin scatter displays every finite saved pair as an ordinary dot regardless of that classification. These assessments are numerical estimates, not interval certificates, and do not enable certified implication claims. Turning off interval work does not remove the directional measurements or their quadrature uncertainty.

Optional interval certification (`--refine-numerics`, or an explicit positive `--numerical-max-products`) encloses the current posterior for each saved seed batch once on the GPU. If `D_upper-E_lower<0`, with `E=mathcal{E}_t` and `V>=0`, it certifies a negative condition before constructing endpoint intervals or evaluating their rigorous gains. Independently flagged gains and requested source-radius audits still receive refinement. Rows that continue reuse the same current-posterior/error enclosure. Exp/log enclosures use bounded series; binary64 precision or work limits can leave signs unresolved. Raw vector arithmetic, posterior/gain evaluation, interval quadrature and learned-probe error reductions stay on the selected worker GPU.

Host work remains necessary for file I/O, worker scheduling, small control decisions, scalar table aggregation and plotting. Original seeded CPU random streams are retained as input generation and transferred before numerical evaluation, preserving existing experimental inputs. This is not a promise of zero CPU utilization. See [GPU execution and budget controls](docs/theory/gpu_theory_computation.md).

Numerical receipts are checked before loading raw witnesses, and completed batches remain resumable. In the optional interval mode, screen work is charged against the same per-row operation budget; an exhausted screen preserves an unresolved condition without restarting the posterior. Zero skips interval work and retains the assessed point-estimate signs. This policy change leaves sufficient-input recipes and compatible measured probes/integrals reusable. The GPU implementation has new source/policy identities; incompatible derived measurements are rebuilt while old caches and protected generation/SSCD/proximity observations remain preserved. `--refine-numerics` can rebuild analytical response shards when its GPU operation budget changes, but cannot backfill learned probes or missing core observations. The first migration from older source receipts may require `--recompute-experiments`.

Probe tasks and analytical recipes have independent cache identities. An integrator change or missing integral does not repeat learned probes. Vector cores and integration payloads have separate completion markers. Missing or changed quadrature reads saved sufficient statistics; within the direct stage, only incomplete vector cores require raw-record reloads. Successful records remain reusable. Changing loss draw count gives new probe task identities; the deterministic RNG prefix is preserved, but partial prefix reuse is not implemented. Rendering styles do not invalidate numerical task caches.

Derived recomputation preserves omitted saved loss-draw settings. Plotting inherits omitted scientific settings from the saved bundle, including loss draw counts, timestep mode and reference law. Explicit conflicting settings fail with a recompute command. Repository model/scheduler/g/T/N identities remain checked. A portable bundle needs only its compact scalar tables and manifests:

```bash
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

## Paper outputs

Numerical tables and publication metadata remain in `outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/`:

```text
run_config.json, summary.json, audit.json, theory_mean.json
registry.json, figure_manifest.json, figure_captions.md, figure_retirement.json
initial.csv, terminal.csv, failed.csv, logical_tables.json
plot_data/*.csv, audit_data/*.csv
```

The active paper publisher exports exactly six single-page PDFs under project-root `figures/<experiment>/theory/experiment_S0_N<N>/`. There are no appendix, per-timestep or diagnostic figure exports; `--diagnostics` does not expand this selection. `measurement_registry()` retains the complete 20-measurement inventory and optional scalar audits, while `paper_registry()` selects only these six render entries. Scientific formulas, scalar tables, sample populations and cache identities are unchanged by this curation.

| Order | PDF basename in the theory publication directory | Saved measurement stem |
|---|---|---|
| M1 | `initial_loss_recovery` | `initial_loss_recovery` |
| M2 | `unconditional_reference_convergence` | `unconditional_reference_convergence` |
| M3 | `guidance_scale_vs_loss` | `corollary3_guidance_scale_vs_loss` |
| M4 | `posterior_feedback_condition_margin` | `posterior_feedback_condition_margin` |
| M5 | `synchronization_bound` | `synchronization_bound` |
| M6 | `terminal_bound_coverage` | `terminal_bound_coverage` |

`posterior_feedback_condition_margin` shows every finite saved margin/gain pair from all saved timesteps as an ordinary dot, with terminal SSCD colors where available. Its selected dense-layer opacity is 0.1; the selected sparse-scatter opacity is 0.8. Numerical signs do not filter points or create Observed/Unresolved markers or legends. Only the pooled PDF is exported; there are no per-timestep figures. Sign, uncertainty, endpoint-transfer and implication receipts remain unchanged in the audit tables.

`guidance_scale_vs_loss` changes only the published filename: its compact table and scientific ID remain `corollary3_guidance_scale_vs_loss`. It displays `sqrt(saved x)` for `saved x=L_T(c)/(d*SNR_T)` against the signed joint least-squares guidance coefficient. The square root follows averaging the forward loss over draws; the fitted coefficient and mean-terminal-SSCD color retain the same complete seed cohort. The horizontal reference remains the configured guidance scale.

`synchronization_bound` shows only the learned gap norm (solid) and the signed projected-error reference bound (dashed), against chronological `T-t`. The vertical label is `\cdot/\sqrt{d}`. `unconditional_reference_convergence` retains its selected-μ reference comparison and offset guide without a “Reference limit” legend entry.

`terminal_bound_coverage` retains the same-seed terminal-SSCD groups, three common-population curves per group, prompt-balanced weights, complete tolerance range, independent sampler correction and probability qualifications. Missing scores, empty groups, zero mass and infinite-bound mass remain audited. The baseline, posterior response, branch trajectories, shape/motion, error curves, condition fractions, terminal scatter comparisons and optional counterfactual measurements remain saved but are not exported as additional figures.

The `--counterfactual-unconditional` option remains an explicit analytical/learned measurement task. It does not add a publication figure, even with `--diagnostics`. No new rollout or decode is introduced by this output curation. Selected μ still defaults to `reference-min-snr`; new runs reject `--no-mu` and `--center zero`.

For an otherwise compatible bundle with current scientific receipts, this presentation-only update uses:

```bash
bash run_all.sh --model sdv1 --scheduler ddim --plot
```

Omit `--model` and `--scheduler` to request the full configuration matrix; every requested bundle must pass preflight. Copied bundles use `./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot`. A bundle missing required selected-figure scalars or carrying an outdated scientific contract still requires the reported `--recompute-experiments` command. Plotting does not reconstruct measurements, fit coefficients, evaluate posteriors, or run models.

The `projected-gap-error-1` scientific contract, empirical source-record prior, numerical-policy identities and compact metric schema `four-stage-evidence-projected-gap-error-1` are unchanged by the six-figure selection. Presentation registry `four-stage-paper-curation-20` reuses the existing `plot_data/<scientific-stem>.csv` files. The earlier signed-error or prior-law changes still require analysis when their receipts are absent; the opacity and output-path changes alone do not.

Theory uses one parent-owned tqdm bar for live record progress, without periodic “still running” messages. Aggregation, plot-input, and publication stages report their start and completion once, including elapsed time on completion. Saving scalar tables and publication metadata uses one progress bar that advances per completed file and shows the current filename. Figure preparation has its own progress bar; PDF export advances once per saved file and shows the current filename.

The publication layout is separate from caches:

```text
figures/<experiment>/
  proximity/experiment_S0_N<N>/*.pdf
  proximity/reference_S<N>_N<N>/*.pdf
  proximity/<seed-role>/examples/retained/*.pdf
  proximity/<seed-role>/examples/discarded/*.pdf
  theory/experiment_S0_N<N>/*.pdf
```

Here `<experiment>` is the canonical run name, for example `sdv1_ddim_g7.5_T50_N20`.
Both normal proximity runs and `--plot` export the existing retained/discarded
examples, including generated montages and their paired training images. Highest,
median and lowest refer to the existing prompt-mean terminal L2 ranking; every
seed tile remains present in the generated montage. PDFs use 150 DPI and the
existing display downsampling. The example manifest stays at
`outputs/<experiment>/proximity/<seed-role>/examples/manifest.json` and records the
external PDF directory, source identities and hashes. Plotting validates cached
montages, normalized training PNGs and completion metadata before publication;
it never regenerates images or loads latent tensors. Missing cached images must
be restored before replotting. Historical example PNGs remain in place.

Role and selection-strategy namespaces remain distinct. The figure tree contains
PDF exports only; captions, manifests and scalar inputs remain under `outputs/`.
Both normal execution and `--plot` use this layout. Existing cache images are
preserved; changing output paths does not recompute experiments or migrate old
images. Theory role locks and coordinated staging protect PDFs and their metadata
against partial publication. Unowned or modified theory outputs are not overwritten.

Plot mode does not access backing numerical shards. Axes and legends use the mathematical notation in `revised.pdf`, with bold vectors, single-line division and named reference lines. Colorbars read `SSCD`; pair-mean aggregation stays in the initial figure caption. See [the figure notation mapping](docs/theory/figure_notation.md).

## Figure appearance

The six exports use plotting-only style `six-theory-visual-6`, centralized in `utils/experiments/theory/paper_style.py` and scoped to the selected `figures/` entries. SSCD uses `magma(0.20+0.50*s)` on the unchanged [0,1] scale; opaque colorbars are independent of marker transparency. Higher-/lower-SSCD groups use the sampled coral/indigo endpoints. Analytical reference, learned estimate and reference error use slate blue, teal and charcoal. The proximity extension below has its own scoped categorical style; unselected theory renderer styles remain unchanged.

Initial-loss and fitted-guidance points use the same size and 0.8 opacity. The pooled feedback scatter uses smaller markers at uniform 0.1 opacity and a reproducible SSCD-independent display order; values and statuses remain paired. All raster exports for these six theory figures use 150 DPI. The dense scatter and opaque SSCD color strips are rasterized in PDFs; text, axes, guides and colorbar ticks/outlines remain vector artwork. Rasterizing each color strip avoids viewer-dependent seams between adjacent vector cells. Shared loss-axis display ranges include both selected scatters without trimming points or confidence intervals. A dashed one-entry guidance legend sits just below the guide near the right edge, and grouped legends separate colors from line meanings. All saved arrays, weights, band endpoints, scientific scales and qualifications remain unchanged.

For an otherwise compatible bundle, run `./run_all.sh --model sdv1 --scheduler ddim --plot`; no analysis update is required for this appearance change. See [the appearance contract and manuscript note](docs/theory/figure_appearance.md). Final-size, grayscale and color-vision-deficiency visual checks remain for the author. The original source-only refresh did not execute tests or rendering; the subsequent PDF-only replot is explicitly author-requested. Use `./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot` to write the six PDFs from a compatible saved bundle to this project’s figure tree; `--pdf-only` remains a compatibility alias. Existing cache images are preserved.

## Proximity figure appearance

The existing all-prompt and retained-prompt proximity scatters use presentation style `proximity-publication-2`. Original categories retain the labels $\mathtt{MV}$, $\mathtt{RV}$, $\mathtt{TV}$ and $\mathtt{N}$, with fixed coral, indigo, teal and slate-blue colors. Color describes the original category; SSCD remains the y-coordinate. Paired views share axes and a 3.1-inch square plotting box, with size-10 markers at uniform 0.35 opacity. A single-column legend sits inside the upper-right corner. The original rounded statistics box sits inside the lower-left corner, showing #Prompts, median rho and rho < 0 with the existing evaluable denominator. Exports use `bbox_inches="tight"` and `pad_inches=0.05`.

Use `bash compute_proximity.sh --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 --seed-start 0 --plot` for saved evaluation data; use `--seed-start 20` for the matching reference block. SDv2 and RealVis use their existing model options, and SDv1/DDPM is supported. Exports are PDF-only; the proximity point collection uses the shared 150-DPI setting, with labels, legend and statistics box kept as vector artwork. The six theory figures retain style 6 and 150-DPI raster layers. The reference GMM-fit diagnostic now shares the proximity typography, square plot area, grid and markers (`proximity-gmm-publication-1`). Its saved low/high SSCD components use indigo/coral with an untitled upper-right legend labeled “Low SSCD” and “High SSCD” in regular font; fitted means and covariance ellipses are unchanged. The broader `bash run_all.sh --plot` also renders this diagnostic. See [the proximity appearance contract and exact author commands](docs/proximity_figure_appearance.md). This proximity coding task ran no tests or rendering; final-size, grayscale and color-vision-deficiency reviews remain pending.

## Author validation

This fixed evidence redesign was prepared under an **edit-code-only** instruction. No new measurements, tests, plots, GPU runs or data-dependent checks were executed. Historical reports document earlier implementations and do not validate this change.

Commands for the author, after review:

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
```

See [the active four-stage plan and manuscript mapping](docs/theory/four_stage_experiments.md), [the numerical refinement plan](docs/theory/numerical_refinement_plan.md), [the fixed evidence plan](docs/theory/final_evidence_plan.md), [required manuscript alignment](docs/theory/submission_alignment.md), [the earlier implementation report](docs/theory/direct_implementation.md), [the direct pipeline guide](docs/theory/paper_pipeline.md), [the quantity dictionary](docs/theory/quantity_dictionary.md), and [the figure registry](utils/experiments/theory/paper_registry.py). The manuscript mapping is recorded in [statement_registry.json](docs/theory/statement_registry.json).
