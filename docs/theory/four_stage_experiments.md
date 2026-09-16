# Four-stage mechanism experiments

This is the active experiment plan. Presentation registry `four-stage-paper-curation-13` fixes four main and sixteen mandatory appendix figures; [figure_curation.md](figure_curation.md) records their order and migration rules. It supersedes the seven-primary registry and candidate-gallery plans without replacing their compatible scientific measurements. The coding task edits source, tests, schemas, and documentation only. Tests, measurements, inference, integration, plotting, and GPU execution are left to the author; this note reports no new numerical outcome.

Axis/legend notation is documented in [figure_notation.md](figure_notation.md). The presentation recipe `four-stage-manuscript-notation-11` uses the manuscript symbols without changing scalar units or scientific identities.

## Manuscript authority and mapping

The authority is `revised.pdf`, SHA-256 `fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`, Sections 2–3, Appendix C, and Appendix D.1–D.7. The PDF is unchanged from the previously inspected manuscript. Matching LaTeX was not found. Printed statement/equation numbers below are the cross-references; historical author-supplied LaTeX label strings are not verified source labels.

| Experiment and main stem | Manuscript connection | Comparison |
|---|---|---|
| `initial_loss_recovery` | Theorem 1, Lemma 2; Appendix C, D.1–D.2 | Independent pair-specific forward-loss scale versus conditional Gaussian recovery RMS, with a same-input unconditional target-error control; separate unique-seed mean-baseline report |
| `branch_gap_posterior_response` | Lemma 4 and Proposition 5; Equation 15 and Appendix D.5, Equations 66, 68–69 | Initial fixed-direction matched response as retained contribution s varies, with mandatory original-condition and endpoint measurements at every eligible step |
| `branch_gap_synchronization` | Lemma 6, Appendix D.6 | Per-sample gap D and joint target error Q over chronological prediction index k; full bound in the appendix |
| `terminal_bound_coverage` | Equation 5, Theorem 7, Appendix D.7 | Common-population coverage of actual saved final error, corrected observable bound and corrected reference bound over latent tolerances; A8/A9 retain their distinct scatter comparisons |

Corollary 3 retains its full-vector approximation/discrepancy audit and adds the A15 signed guidance-fit diagnostic against genuine forward loss. Lemma 4 remains an independently reconstructed vector displacement check. Neither has a separate main figure. Temporal shape descriptors and consecutive-branch motion accounting are analysis extensions, not new manuscript theorems.

## Distinct initial input laws and one reference law

The forward loss averages coordinate-summed squared epsilon error on independent `alpha_T*x_star + sigma_T*epsilon` inputs without CFG. The horizontal coordinate is `sqrt(L_hat/(d*SNR_T))`. Conditional and unconditional vertical quantities are paired RMS target errors on the common genuine Gaussian evaluation bank. They are different input laws: the forward clean-error identity audits the loss implementation but is not a Gaussian equality. Consequently the main initial plot has neither an equality diagonal nor a fitted upper envelope. The saved paired Gaussian bootstrap and independent forward bootstrap describe evaluation uncertainty conditional on the selected mean; the independent mean estimator's Monte Carlo standard error is reported separately. These intervals do not turn reused seeds or draws into independent prompts. Compatible saved draw counts and deterministic streams are reused.

The law is `D_K = sum pi_j delta_u_j`, with positive masses, exact-atom deduplication, and immutable target aliases. Default masses are equal per distinct compatible source atom before selection. Explicit saved compatible weights remain authoritative; downloaded duplication does not infer training frequency. All references use this same law:

- The exact bank mean is the full latent vector `sum pi_j*u_j`; it controls the posterior's zero-SNR limit.
- Displayed `mu` is selected separately. By default it is the full-vector average of 10000 unconditional posterior clean references evaluated at the smallest SNR of the prespecified analytical extension on independent Gaussian inputs. Default `reference-min-snr` uses analytical grid index zero, at `SNR_T * 10**(-reference_snr_decades)`, without a native timestep. Explicit `reference-initial` retains the first native noise level, while `cached-targets` selects the exact bank mean.
- `S_K` is the weighted RMS atom deviation about the exact bank mean, not radial deviation about selected `mu`; zero bank spread leaves relative baseline error undefined. `comparison_scale_rmse` separately records radial RMS about selected `mu`. The selected-mean estimator uses only the analytical posterior, with no denoiser calls.
- `p_K(t,z)` is the target's finite-mixture posterior mass, using raw squared Euclidean distances in Gaussian exponents.
- `bar_x_K(t,z) = sum w_j(t,z)*u_j` is the posterior clean reference.
- `e_u_K = ||m_u-bar_x_K||` is a reference error, distinct from `||m_u-x_star||`.
- `R_K(x_star) = max_j ||u_j-x_star||` is the maximum over the entire support, cached once per target atom.

This empirical evaluation law is not identified with the checkpoint's full training distribution. A per-pair singleton conditional idealization does not establish unique training-target multiplicity. The selected finite-SNR reference estimate never replaces the bank mean used by the posterior. Its independent estimator receipt records sample count, seed, native initial SNR, separate estimation SNR, analytical grid position, extension depth, finite-law identity, vector identity and Monte Carlo standard error. Finite-selected-SNR bias, Monte Carlo uncertainty and the measured finite-bank offset are separate limitations; none is proof of the exact training-data mean.

When cached initialization cannot provide a genuine Gaussian unconditional target-error control, analysis backfills only the missing unique seeds. One fresh empty-prompt prediction per seed is shared across retained targets; compatible controls and actual generation/SSCD identities remain unchanged. This is separate from the explicitly optional counterfactual network supplement.

The baseline report uses one canonical observation per unique Gaussian seed, audits repeated prompt observations for disagreements, and never averages disagreements away or selects by SSCD. `initial_baseline_summary.csv` and `.json` save B, spread, their ratio, uncertainty, seed count, and reference qualifications. Seed distances, atom distances, and law geometry remain in the scalar tables.

## Fixed-direction response and the original condition

At chronological k=0, the dose grid is `sorted(unique({j/40: j=0,...,40} union {1/g}))`. The matched path is `z(s)=x_cf_next+s*g*kappa*Delta` and the ordinate is the log-probability gain H relative to s=0. Endpoint intercepts/slopes are reused across the grid without candidate-by-latent duplication or denoiser calls. The 0, 1/g, and 1 markers denote matched unconditional, conditional-only, and CFG endpoints. An endpoint is called actual only when its independently recorded reconstruction agrees; recovered stochastic innovations and reconstructed segments remain qualified.

This comparison retains more of one fixed vector contribution. It does not claim an arbitrary larger gap norm improves posterior evidence. The derivative contains the signed alignment with `x_star-bar_x_K`, so declining and negative curves remain. The primary snapshot is always k=0; inapplicability never selects a later favorable step. Gold/purple cohorts are same-seed paired-target SSCD > 0.75 and SSCD <= 0.75. Curves use fixed complete-grid cohorts, equal prompt mass within each group, weighted medians/IQRs, and saved exclusion reasons. The signed axis uses a fixed symlog threshold of 1e-3 natural-log units.

Every structurally eligible nonterminal update retains endpoint probabilities/log quantities, stable H and G, reference errors, the original Equation-15 V, and `M=||Delta||-e_c-e_u_K-V`. H and G share a mathematical sign in a nondegenerate law but not a magnitude. Missing V remains missing when a justified negative precheck resolves the condition. Stable numerical assessment, enclosed arithmetic/quadrature, source sensitivity, and reference identification stay separate. Unresolved observations remain in structural denominators and possible-fraction bounds. Mandatory variation analysis cannot be declared complete from sign screening alone.

## Joint accuracy, trajectory shape, and branch motion

Prediction index k increases from 0 (initialization) through `K_steps-1`; no branch exists at the final output. Source/destination native labels and SNR are saved separately. Normalized progress is k/(K_steps-1), with zero for the single-prediction case.

`D=||m_c-m_u||/sqrt(d)` and `Q=max(||m_c-x_star||,||m_u-x_star||)/sqrt(d)` are computed per sample before aggregation. Small D with large Q means agreement away from the target. The full support bound `(e_c+e_u_K+R_K*(1-p_K))/sqrt(d)` and both branch errors remain appendix measurements through the final current prediction, even if its destination posterior is undefined. Lemma 6 is a pointwise bound; it does not assert decreasing, unimodal, or SSCD-determined temporal shape.

The additional A10 `branch_gap_per_prompt` plots one curve per retained prompt–target pair: the arithmetic mean of each seed's normalized gap at every saved prediction index. The fixed curve color is mean terminal SSCD across the same complete seed cohort. There is no SSCD split, vector averaging before the norm, interpolation, or extra final-output prediction. Incomplete or nonfinite cohorts retain explicit audit reasons.

A11–A13 add the conditional reference error, unconditional reference error, and target posterior probability for each prompt–target pair over chronological pre-update predictions `T-t`. Each curve uses the same full fixed seed cohort at every step and a fixed color from those seeds' mean terminal SSCD. The error curves average saved per-seed normalized norms, with no second division by `sqrt(d)`. Unconditional error compares the learned clean estimate with the posterior clean reference, not with the target. Probability is `mean(exp(direct_target_log_probability))`, computed per seed before averaging, on the actual saved generated trajectory state; `exp(mean(log_probability))` would be a different quantity. Missing or invalid trajectories exclude the entire pair from that figure and are audited; no curve averages a changing subset of seeds. These are compact reductions of existing scalar measurements, with no new posterior or network evaluation.

A14 `reference_branch_gap_per_prompt` shows $\|\bar{\mathbf{x}}_t(c)-\bar{\mathbf{x}}_t(\varnothing)\|/\sqrt{d}$ against $T-t$. The analytical reference difference is conditional minus unconditional at the same actual saved trajectory state and noise level. Under the existing single-target conditional assumption, $\bar{\mathbf{x}}_t(c)=\mathbf{x}^{\star}$, so its norm equals the saved opposite-sign distance $\|\bar{\mathbf{x}}_t(\varnothing)-\mathbf{x}^{\star}\|$. It therefore averages `direct_reference_target_error_rmse` once across the same complete fixed seed cohort, with a fixed color from those seeds' mean terminal SSCD. The learned branch gap, differences of error norms, and the radius-tail upper bound are different quantities and are not used. This adds no posterior or network evaluation; exclusions remain in `audit_data/reference_branch_gap_per_prompt_cohort.csv`.

Raw per-sample curves supply initial/final/fixed-phase values, rise, decline, upward/downward variation, peak and near-tie indices, flat/plateau status, and completeness. Missing interior values cannot silently produce a complete shape statistic. Prompt-level and mixed-outcome paired summaries keep overlapping prompt populations explicit.

Explicit analysis also reduces the saved weighted peak bins into `audit_data/trajectory_peak_summary.csv` and figure metadata. Its median and quartiles use the inverse weighted CDF conditional on resolved individual peaks. The histogram, initialization mass and unresolved mass retain the original full denominator. This descriptive reduction does not change the histogram or measure new trajectories. Compatible older bundles still support `--plot`; their captions disclose the absent summary and give an exact analysis-only command. Plotting never computes these statistics, and placement-only migration does not require them.

For consecutive predictions, `v_c=m_c,next-m_c` and `v_u=m_u,next-m_u` give

`D_next^2-D^2 = 2<Delta,v_c>/d - 2<Delta,v_u>/d + ||v_c-v_u||^2/d`.

The quadratic term is named `gap_motion_quadratic`, not manuscript V. Any saved group motion summaries use weighted means over one shared row set to preserve additivity. The two reviewed motion render families are retired from all publication/diagnostic routes; per-sample motion terms, numerical identity tests, target-error changes and historical metadata remain. These terms include changing prediction noise labels and describe motion, not a causal attribution to a single intervention. Target-error changes, noise gap, and the independent identity `D=(sigma/alpha)*noise_gap` accompany them.

A15 `corollary3_guidance_scale_vs_loss` plots `sqrt(L_T(c)/(d*SNR_T))` against the signed least-squares guidance coefficient for Corollary 3, Equation 54. The compact `x` remains the genuine forward-target loss mean `L_T(c)/(d*SNR_T)`; only its display takes one square root, after averaging the loss over draws. It is not the mean of per-draw roots. For each prompt–target pair, the fit minimizes `sum_seed ||(xhat_T(c;g)-mu)-a*(x_star-mu)||^2` over one scalar `a`, using the full fixed saved initial seed cohort at the actual first-prediction states. Gaussian-bank match status is recorded separately and is not assumed for this fit. Its shared target direction makes the joint coefficient equal to the arithmetic mean of the signed per-seed coefficients. Color is mean terminal SSCD over exactly those seeds; the labeled horizontal guide shows the actual configured `g`. The fit uses the selected reference mean, retains negative coefficients, and excludes incomplete cohorts or unidentifiable directions with saved reasons. It is a projection of the full guided clean estimate around μ, not a projection of `g*Delta_T`, and it does not estimate or change the configured generation guidance.

A16 `reference_variation_per_prompt` shows $V_t/\sqrt{d}$ against $T-t$, using the saved Equation-15 cross-step reference-variation norm-integral estimates `direct_prop5_variation_rmse` from `matched_updates`. The integrand is defined relative to the current unconditional posterior reference $\bar{\mathbf{x}}_t(\varnothing)$, not the selected global μ. Each prompt–target curve averages the already normalized estimates over the same complete fixed seed cohort at every positive-noise transition, with a fixed color from those seeds’ mean terminal SSCD. Its domain is $t=2,\ldots,T$, or $T-t=0,\ldots,T-2$; no final-prediction or output value is inserted, and a one-step schedule is not applicable. Finite nonnegative original estimates are retained when condition signs or the quadrature budget remain unresolved; refined interval midpoints are not substituted. A missing, invalid or structurally inapplicable required estimate excludes the entire curve with an explicit reason. `audit_data/reference_variation_per_prompt_cohort.csv` records cohort exclusions and numerical-status counts, while `audit_data/feedback_endpoints.csv` retains the source status, error estimates and bounds. These are numerical estimates, not certified exact integrals. The new compact reduction performs no integration, posterior evaluation or model inference.

The fit and its cohort audit are saved under recipe `corollary3-guidance-fit-1`. Analysis uses cached initial vectors and the existing genuine forward-loss measurement; no additional generation rollout is introduced. An older bundle missing these fields requires `bash run_all.sh --recompute-experiments` to save them under current source hashes. Bundles already containing compatible fit fields use `bash run_all.sh --plot` for the square-root display change; the stored loss ratio and scientific formula recipe remain unchanged.

## Terminal prerequisite and qualified extension

The main terminal figure is `terminal_bound_coverage`. For each same-seed terminal-SSCD group (>0.75 and <=0.75), using one common eligible population and identical weights within that group, it plots the fractions `E_end<=tau`, `B_obs+delta_sched<=tau`, and `B_ref+delta_sched<=tau` against `tau/sqrt(d)`. Its vertical label is `Pr(· ≤ tau)`, representing the same saved weighted fraction satisfying tolerance. The actual saved output supplies `E_end`, never the last guided clean estimate. The reference bound is `B_ref=g*e_c+(g-1)*(e_u_K+R_K*(1-p_K))`; the distinct observable bound is `B_obs=e_c+(g-1)*||Delta||`. Both paired scatter comparisons remain mandatory as A8 `terminal_observable_bound` and A9 `final_reproduction_bound`. All normalization occurs exactly once.

Both groups share the complete tolerance range. Gold/purple encode SSCD group; solid/dashed/dash-dot encode actual/observable/reference. Group weights are recomputed within each group as one unit per represented prompt divided over its eligible member seeds, then normalized by that group's total mass. The same prompt can appear in both groups through different seeds. Missing/nonfinite SSCD is explicitly excluded from grouping and saved in `audit_data/terminal_grouped_sscd_exclusions.csv`. Empty groups are documented without invented curves. The original pooled ECDF and its metadata remain in `audit_data/terminal_pooled_cdf.csv` and `terminal_pooled_cdf_metadata.csv`. Existing construction and row-level bound checks remain authoritative. An older pooled compact table requires analysis-only `--recompute-experiments` to save grouped summaries; compatible expensive measurements are reused.

The original theorem requires the structural clean update in Equation 5 and numerical consistency with the saved endpoint. Deterministic finite affine updates use the independently predicted defect `||a+(kappa-1)*m_g||`. Valid independently observed innovations may support a pathwise correction; the preserved supported Gaussian mode uses its predeclared simultaneous run probability budget. Recovered output noise cannot become an independent correction certificate. Nonzero stochastic variance is not silently set to zero. The correction and its probability scope are an extension requiring manuscript alignment.

Each supporting scatter retains its bound-comparison diagonal, qualified by the saved correction mode. Coverage retains the complete tolerance range, its loose reference tail and exact zero mass without epsilon substitution. All finite eligible rows, loose bounds and violations remain. Deterministic/pathwise-valid row checks retain `E_end<=C_obs<=C_ref`; their expected common-weight curve ordering is `F_ref<=F_obs<=F_actual`. Material unexplained violations are reported, not repaired by sorting curves. The identity

`B_ref-B_obs=(g-1)*(e_c+e_u_K+R_K*(1-p_K)-||Delta||)`

accounts for reference-bound looseness; the common sampler correction cancels. SSCD supplies independent image similarity and never chooses a latent tolerance.

## Active appendix and optional measurements

The sixteen mandatory appendix stems, in order, are:

1. `initial_unconditional_mean_concentration`
2. `unconditional_reference_convergence`
3. `posterior_feedback_over_time`
4. `posterior_feedback_condition_margin`
5. `branch_target_errors`
6. `synchronization_bound`
7. `branch_gap_peak_step`
8. `terminal_observable_bound`
9. `final_reproduction_bound`
10. `branch_gap_per_prompt`
11. `conditional_reference_error_per_prompt`
12. `unconditional_reference_error_per_prompt`
13. `target_probability_per_prompt`
14. `reference_branch_gap_per_prompt`
15. `corollary3_guidance_scale_vs_loss`
16. `reference_variation_per_prompt`

Main-caption support maps are initial recovery → A1/A2; retained response → A3/A4; synchronization → A5 first, then A6/A7; terminal coverage → A8/A9. Branch-resolved target errors remain A5, not a fifth main figure. The peak distribution remains even when groups overlap. Empty groups and mathematical inapplicability have explicit manifest reasons; incomplete required analysis is not a completed bundle. Existing supported mathematical audit figures remain optional under `--diagnostics`, which cannot recreate the retired motion or zero-companion families.

Placement-only changes preserve measurements, populations, numerical tolerances, correction modes and scientific hashes. The requested A10 and A11–A14 add compact reductions from already saved trajectory scalars; an older bundle missing one of these inputs needs the analysis-only `--recompute-experiments` command, which reuses compatible expensive measurements. Four-stage bundles containing the required compact tables and matching reduction-source receipts migrate under `--plot` without inference, posterior evaluation, raw-tensor reads or scalar recomputation. Stable plot-data stems remain unchanged. Renderer-owned old coverage/reference-scatter locations and the enumerated motion and zero-companion pairs are retired only after replacement publication succeeds and previous ownership hashes match. Conflicting or unknown files are preserved and reported; paired exports and metadata roll back together on failure. The retirement ledger is small and idempotent, and pre-existing historical archives remain untouched.

`--counterfactual-unconditional` enables a separate learned empty-prompt supplement, with fixed chronological `--counterfactual-steps 0` by default. It evaluates matched and actual next inputs in the same inference context, logs parity with saved predictions, and reports target-error improvement. No new rollout or image decode is performed. This supplement is absent by default. Its figure renders under `diagnostics/` only with `--diagnostics`, saved counterfactual enablement and available saved data; it never adds an appendix figure or triggers inference in plot mode. There are no newly executed supplemental measurements from this coding task. Availability is established only by the author's resulting receipts. An explicitly requested failed supplemental task is retained as incomplete and blocks that requested publication; running the default suite does not require or infer it.

## Execution, cache migration, and commands

Fast vector cores and four-stage primary shards are prepared before mandatory integration. Existing probe, endpoint, reference, and integration hashes remain authoritative. The new scalar/plot bundle uses schema 5 and metric recipe `four-stage-evidence-1`; additive new shards do not mutate old measurements. Base scientific configuration and the original numerical receipt remain separately readable for resume; optional counterfactual settings have their own saved configuration and identity. All visible GPUs use spawned workers, bounded device tensors, one aggregate parent progress bar per stage, and no overlapping full-model and analytical pools. The CUDA interval backend replaces the selective CPU Decimal fallback. GPU binary64 enclosures preserve unresolved signs when intervals overlap zero; a zero GPU operation budget disables interval refinement without inventing resolved signs. Numerical tensor work requires CUDA, while host I/O, scalar aggregation, exact seeded input streams and plotting remain. See [GPU execution](gpu_theory_computation.md).

Plot mode reads compact scalar inputs and metadata only, including copied standalone bundles. It never opens raw trajectories, hashes raw tensors, recomputes a posterior, integrates, or backfills model probes. Omitted options inherit saved measurement settings. Publication stages PNG/PDF pairs and manifests with rollback; only verified renderer-owned obsolete files are retired during author execution, never during this coding task.

Run these commands as the author (not executed during implementation):

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto
./run_all.sh --model sdv1 --scheduler ddim --plot
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto --counterfactual-unconditional
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

Inspect measured exclusions, unresolved mass, original-condition coverage, trajectory completeness, endpoint reconstruction, and terminal probability/applicability statuses before writing empirical claims. Success does not require favorable results.

## Retired zero-vector companions

The zero-vector companions are excluded from both default rendering and diagnostics. Their saved scalar measurements remain intact. New runs reject `--no-mu` and `--center zero`; historical saved configurations remain readable. The selected theory mean still defaults to `reference-min-snr`. The convergence offset guide remains without its former legend entry; the caption and audit retain its value. Bundles predating removal of the zero-comparison reduction need one `bash run_all.sh --recompute-experiments` source-receipt update before `--plot`; compatible measurement shards remain reusable. The later square-root display change requires no additional analysis. See [publication commands](figure_curation.md#author-commands--not-executed-here).

The exact finite-bank mean, selected reference estimate, their offset and existing zero-distance scalars keep their original definitions. Retiring these figures changes presentation only; it neither sets μ to zero nor recenters the posterior.
