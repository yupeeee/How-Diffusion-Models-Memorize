# Four-stage mechanism experiments

This is the active experiment plan. It supersedes the seven-primary registry and candidate-gallery plans without replacing their compatible scientific measurements. The coding task edits source, tests, schemas, and documentation only. Tests, measurements, inference, integration, plotting, and GPU execution are left to the author; this note reports no new numerical outcome.

## Manuscript authority and mapping

The authority is `revised.pdf`, SHA-256 `fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`, Sections 2–3, Appendix C, and Appendix D.1–D.7. The PDF is unchanged from the previously inspected manuscript. Matching LaTeX was not found. Printed statement/equation numbers below are the cross-references; historical author-supplied LaTeX label strings are not verified source labels.

| Experiment and main stem | Manuscript connection | Comparison |
|---|---|---|
| `initial_loss_recovery` | Theorem 1, Lemma 2; Appendix C, D.1–D.2 | Independent pair-specific forward-loss scale versus conditional Gaussian recovery RMS, with a same-input unconditional target-error control; separate unique-seed mean-baseline report |
| `branch_gap_posterior_response` | Lemma 4 and Proposition 5; Equation 15 and Appendix D.5, Equations 66, 68–69 | Initial fixed-direction matched response as retained contribution s varies, with mandatory original-condition and endpoint measurements at every eligible step |
| `branch_gap_synchronization` | Lemma 6, Appendix D.6 | Per-sample gap D and joint target error Q over chronological prediction index k; full bound in the appendix |
| `final_reproduction_bound` | Equation 5, Theorem 7, Appendix D.7 | Actual saved final latent error versus the empirical-reference reproduction bound, with a separately qualified finite-update correction when needed |

Corollary 3 remains a full-vector approximation/discrepancy audit. Lemma 4 remains an independently reconstructed vector displacement check. Neither has a separate main figure. Temporal shape descriptors and consecutive-branch motion accounting are analysis extensions, not new manuscript theorems.

## Distinct initial input laws and one reference law

The forward loss averages coordinate-summed squared epsilon error on independent `alpha_T*x_star + sigma_T*epsilon` inputs without CFG. The horizontal coordinate is `sqrt(L_hat/(d*SNR_T))`. Conditional and unconditional vertical quantities are paired RMS target errors on the common genuine Gaussian evaluation bank. They are different input laws: the forward clean-error identity audits the loss implementation but is not a Gaussian equality. Consequently the main initial plot has neither an equality diagonal nor a fitted upper envelope. The saved paired Gaussian bootstrap and independent forward bootstrap describe Monte Carlo estimator uncertainty; they do not turn reused seeds or draws into independent prompts. Compatible saved draw counts and deterministic streams are reused.

The law is `D_K = sum pi_j delta_u_j`, with positive masses, exact-atom deduplication, and immutable target aliases. Default masses are equal per distinct compatible source atom before selection. Explicit saved compatible weights remain authoritative; downloaded duplication does not infer training frequency. All references use this same law:

- `mu_K = sum pi_j*u_j` is a full latent vector.
- `S_K = sqrt(sum pi_j*||u_j-mu_K||^2/d)` is the weighted spread; a zero spread leaves relative baseline error undefined.
- `p_K(t,z)` is the target's finite-mixture posterior mass, using raw squared Euclidean distances in Gaussian exponents.
- `bar_x_K(t,z) = sum w_j(t,z)*u_j` is the posterior clean reference.
- `e_u_K = ||m_u-bar_x_K||` is a reference error, distinct from `||m_u-x_star||`.
- `R_K(x_star) = max_j ||u_j-x_star||` is the maximum over the entire support, cached once per target atom.

This empirical evaluation law is not identified with the checkpoint's full training distribution. A per-pair singleton conditional idealization does not establish unique training-target multiplicity. Legacy model-output centers do not redefine `mu_K`.

When cached initialization cannot provide a genuine Gaussian unconditional target-error control, analysis backfills only the missing unique seeds. One fresh empty-prompt prediction per seed is shared across retained targets; compatible controls and actual generation/SSCD identities remain unchanged. This is separate from the explicitly optional counterfactual network supplement.

The baseline report uses one canonical observation per unique Gaussian seed, audits repeated prompt observations for disagreements, and never averages disagreements away or selects by SSCD. `initial_baseline_summary.csv` and `.json` save B, spread, their ratio, uncertainty, seed count, and reference qualifications. Seed distances, atom distances, and law geometry remain in the scalar tables.

## Fixed-direction response and the original condition

At chronological k=0, the dose grid is `sorted(unique({j/40: j=0,...,40} union {1/g}))`. The matched path is `z(s)=x_cf_next+s*g*kappa*Delta` and the ordinate is the log-probability gain H relative to s=0. Endpoint intercepts/slopes are reused across the grid without candidate-by-latent duplication or denoiser calls. The 0, 1/g, and 1 markers denote matched unconditional, conditional-only, and CFG endpoints. An endpoint is called actual only when its independently recorded reconstruction agrees; recovered stochastic innovations and reconstructed segments remain qualified.

This comparison retains more of one fixed vector contribution. It does not claim an arbitrary larger gap norm improves posterior evidence. The derivative contains the signed alignment with `x_star-bar_x_K`, so declining and negative curves remain. The primary snapshot is always k=0; inapplicability never selects a later favorable step. Gold/purple cohorts are same-seed paired-target SSCD > 0.75 and SSCD <= 0.75. Curves use fixed complete-grid cohorts, equal prompt mass within each group, weighted medians/IQRs, and saved exclusion reasons. The signed axis uses a fixed symlog threshold of 1e-3 natural-log units.

Every structurally eligible nonterminal update retains endpoint probabilities/log quantities, stable H and G, reference errors, the original Equation-15 V, and `M=||Delta||-e_c-e_u_K-V`. H and G share a mathematical sign in a nondegenerate law but not a magnitude. Missing V remains missing when a justified negative precheck resolves the condition. Stable numerical assessment, enclosed arithmetic/quadrature, source sensitivity, and reference identification stay separate. Unresolved observations remain in structural denominators and possible-fraction bounds. Mandatory variation analysis cannot be declared complete from sign screening alone.

## Joint accuracy, trajectory shape, and branch motion

Prediction index k increases from 0 (initialization) through `K_steps-1`; no branch exists at the final output. Source/destination native labels and SNR are saved separately. Normalized progress is k/(K_steps-1), with zero for the single-prediction case.

`D=||m_c-m_u||/sqrt(d)` and `Q=max(||m_c-x_star||,||m_u-x_star||)/sqrt(d)` are computed per sample before aggregation. Small D with large Q means agreement away from the target. The full support bound `(e_c+e_u_K+R_K*(1-p_K))/sqrt(d)` and both branch errors remain appendix measurements through the final current prediction, even if its destination posterior is undefined. Lemma 6 is a pointwise bound; it does not assert decreasing, unimodal, or SSCD-determined temporal shape.

Raw per-sample curves supply initial/final/fixed-phase values, rise, decline, upward/downward variation, peak and near-tie indices, flat/plateau status, and completeness. Missing interior values cannot silently produce a complete shape statistic. Prompt-level and mixed-outcome paired summaries keep overlapping prompt populations explicit.

For consecutive predictions, `v_c=m_c,next-m_c` and `v_u=m_u,next-m_u` give

`D_next^2-D^2 = 2<Delta,v_c>/d - 2<Delta,v_u>/d + ||v_c-v_u||^2/d`.

The quadratic term is named `gap_motion_quadratic`, not manuscript V. Group plots use weighted means over one shared row set to preserve additivity. These terms include changing prediction noise labels and describe motion, not a causal attribution to a single intervention. Target-error changes, noise gap, and the independent identity `D=(sigma/alpha)*noise_gap` accompany them.

## Terminal prerequisite and qualified extension

The main terminal comparison uses the actual saved output error on the y axis, never the last guided clean estimate. The x axis is `(B_ref+delta_sched)/sqrt(d)`, with `B_ref=g*e_c+(g-1)*(e_u_K+R_K*(1-p_K))`. The observable companion uses `B_obs=e_c+(g-1)*||Delta||`. All normalization occurs exactly once.

The original theorem requires the structural clean update in Equation 5 and numerical consistency with the saved endpoint. Deterministic finite affine updates use the independently predicted defect `||a+(kappa-1)*m_g||`. Valid independently observed innovations may support a pathwise correction; the preserved supported Gaussian mode uses its predeclared simultaneous run probability budget. Recovered output noise cannot become an independent correction certificate. Nonzero stochastic variance is not silently set to zero. The correction and its probability scope are an extension requiring manuscript alignment.

The diagonal is the bound comparison, qualified by its saved mode. Exact zeros get explicit edge representation/counts, not epsilon substitution. All finite eligible rows, loose bounds, and violations remain. The observable scatter and three-curve common-population tolerance coverage are mandatory support. The identity

`B_ref-B_obs=(g-1)*(e_c+e_u_K+R_K*(1-p_K)-||Delta||)`

accounts for reference-bound looseness; the common sampler correction cancels. SSCD supplies independent image similarity and never chooses a latent tolerance.

## Active appendix and optional measurements

The eleven fixed appendix stems are `initial_unconditional_mean_concentration`, `unconditional_reference_convergence`, `posterior_feedback_over_time`, `posterior_feedback_condition_margin`, `branch_target_errors`, `synchronization_bound`, `branch_gap_peak_step`, `branch_gap_motion_high_sscd`, `branch_gap_motion_lower_sscd`, `terminal_observable_bound`, and `terminal_bound_coverage`. Empty groups and mathematically inapplicable figures have explicit manifest reasons. Incomplete required analysis is not a completed bundle. Existing mathematical audit figures remain optional under `--diagnostics`.

`--counterfactual-unconditional` enables a separate learned empty-prompt supplement, with fixed chronological `--counterfactual-steps 0` by default. It evaluates matched and actual next inputs in the same inference context, logs parity with saved predictions, and reports target-error improvement. No new rollout or image decode is performed. This supplement is absent by default; when explicitly computed its saved table and optional appendix can be rendered without inference. There are no newly executed supplemental measurements from this coding task. Availability is established only by the author's resulting receipts. An explicitly requested failed supplemental task is retained as incomplete and blocks that requested publication; running the default suite does not require or infer it.

## Execution, cache migration, and commands

Fast vector cores and four-stage primary shards are prepared before mandatory integration. Existing probe, endpoint, reference, and integration hashes remain authoritative. The new scalar/plot bundle uses schema 5 and metric recipe `four-stage-evidence-1`; additive new shards do not mutate old measurements. Base scientific configuration and the original numerical receipt remain separately readable for resume; optional counterfactual settings have their own saved configuration and identity. All visible GPUs use spawned workers, bounded device tensors, one aggregate parent progress bar per stage, and no overlapping full-model and analytical pools. Selective high-precision CPU fallback retains its existing numerical policy; zero Decimal budget disables that fallback without inventing resolved signs.

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
