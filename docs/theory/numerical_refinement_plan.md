> Current computation defaults to numerical estimates, with optional CUDA-only interval certification under `fixed-cache-numerics-cuda-8` and `projected-gap-error-1` quantities; see [GPU execution](gpu_theory_computation.md) and the [active experiment plan](four_stage_experiments.md). Seven-primary figure placements below are historical. Fixed-input, source-sensitivity and reference-law distinctions remain separate. Matching LaTeX labels remain unverified.

# Precision and presentation refinement of the fixed evidence suite

This document describes source changes, not generated results. No tests, project imports, experiments, downloads, inference, posterior calculations, quadrature, plotting, benchmarks, or GPU jobs were executed for this revision. Existing scientific caches, logs, figures, and historical reports remain untouched. The author will execute the updated code.

## Authority and fixed outputs

The mathematical authority remains the mounted `revised.pdf` identified in [submission_alignment.md](submission_alignment.md) and the author's precise seven-statement brief. A matching LaTeX source is not available in this checkout. The refined condition uses e_t^parallel(Delta)=u^T(Delta-bar{Delta}) and uses the requested directional positive-part variation on manuscript t>=2, revising the historical Equation-15 norm integral. The refined terminal bound retains the clean-update prerequisite or the separately qualified correction extension.

The seven primary stems are unchanged. The mandatory appendix registry adds only `lemma2_native_gaussian_sweep` and `proposition5_numerical_resolution`: twelve appendix entries total, with terminal components conditional on a supported terminal contract. One stem produces one data axes, one PNG, and one single-page PDF. The established proximity publisher, canonical output paths, safe staged publication, rollback, and ownership-based archival remain in use.

## Three uncertainty layers

1. **Fixed saved-input arithmetic.** The function is defined by the stored states, already-canonical epsilon predictions, target/support atoms, positive prior masses, and saved scalar scheduler coefficients. Float64 posterior-gain and condition-sign computations are the default numerical assessments. Optional interval mode uses the batched CUDA screen to enclose the current posterior and vector branch-gap error from original saved inputs. Directed CUDA binary64 intervals may enclose this function from those raw inputs. An enclosure of rounded sufficient logits alone is explicitly labelled a reduced-input enclosure, not a certificate of constructing those logits from original vectors.
2. **Source/reconstruction sensitivity.** Legacy dtype-multiplier diagnostics remain saved as sensitivity assessments. They are not mathematical uncertainty radii. Native coefficient derivation, checkpoint approximation, conversion history, and an unspecified perturbation of stored vectors are not silently covered by a fixed-input certificate. Actual saved-versus-affine endpoint residuals are recorded separately.
3. **Reference identification.** All calculations use the same declared `D_K`. Its discrepancy from the complete checkpoint training distribution remains unidentified; extra precision cannot reduce it. The default now weights cached source records equally before selection and combines exact-duplicate masses, giving atom prior n_j/N. Uniform distinct-atom results belong to an older law identity. Recompute posterior-dependent analysis and the law-keyed μ estimator for this prior change; an interval retry or relabeling does not convert old values.

The schema records `input_contract_status`, `endpoint_construction_method`, `arithmetic_status`, `arithmetic_error_method`, `quadrature_status`, `quadrature_error_scope`, `source_sensitivity_status`, `source_sensitivity_model`, `reference_law_id`, `fixed_cache_gain_sign`, `source_robust_gain_sign`, `condition_sign_status`, `condition_arithmetic_status`, `condition_estimate_sign_status`, `numerical_condition_estimate_uncertainty_l2`, `numerical_condition_estimate_status`, and `condition_value_status`. Per-row precision, operation/node budgets, stopping reason, legacy classification, enclosure widths, and publication blockers remain in the numerical row audit. A migration receipt identifies the change from the legacy source-sensitivity interpretation to the fixed-input interpretation. Old measurements/classifications are retained in their original columns and files.

## Auxiliary numerical derivations

These are numerical tools for evaluating the original objects, not new experimentally verified manuscript results.

For a nonempty target complement, with target-relative log densities `b_j`, endpoint slopes `a_j=beta*<h,u_j-x_star>`, and `nu=softmax_non_target(b)`, evaluate

```
G = -log(sum_j nu_j*exp(a_j)).
```

Center the log-sum-exp; do not obtain the new net gain only by subtracting two enormous absolute log odds. For bounded small slopes, evaluate `-log1p(sum_j nu_j*expm1(a_j))`, accumulating positive and negative terms separately and flagging cancellation for refinement. Large slopes never enter an unbounded expm1 branch. The old endpoint difference stays as a cross-check. Log-probability gain `H` retains its own magnitude, including saturation/underflow; `G` may resolve a sign but cannot replace that magnitude. Zero displacement and a single-atom law have explicit semantics; singleton log odds are undefined.

Let `D=||Delta||`, `bar{Delta}=x_star-bar_x_current` under the single-target conditional reference, and `E=e_t^parallel(Delta)=u^T(Delta-bar{Delta})`.

```
u = Delta/D for D>0
W = integral_0^1 <u,bar_x_next(z0+s*h)-bar_x_current> ds
mathcal{V} = max(0,W)
M = D-E-mathcal{V}.
For D=0: u=0, E=0 and mathcal{V}=0 by computational extension; no unit direction is claimed.
```

By default, `--numerical-max-products 0` skips interval screening, endpoint enclosures and interval variation refinement. The saved margin is assessed against the sum of its saved signed-integral embedded error, projection roundoff and a float64 subtraction allowance `128*eps64*(D+abs(E)+abs(mathcal{V}))`. A margin strictly outside that estimated uncertainty receives a positive or negative numerical sign. Near-zero, missing, invalid or failed measurements remain unresolved; a structural singleton identity may establish exact zero. The receipt is `condition_arithmetic_status=float64_numerical_assessment`, not a certificate, and cannot establish `implication_eligible`. Source-dtype sensitivity is not included as an invented mathematical error bound. The new directional measurement has a distinct recipe; genuine QA failures remain visible. Its signed quadrature error and projected-geometry roundoff propagate through the one-Lipschitz positive-part map as numerical uncertainty, not a certificate.

In optional interval mode, since V is nonnegative, a certified upper bound `D_upper-E_lower<0` can resolve the refined margin as negative without assigning a value to missing V. Policy `fixed-cache-numerics-cuda-8` computes this enclosure once for the complete saved seed batch before any endpoint interval construction. A certified negative row skips endpoint/gain/quadrature enclosure only when its stable gain is resolved and no independent source-radius refinement is requested. Rows that continue pass the current posterior, learned gap norm and branch-gap-error intervals directly to endpoint refinement; they are not recomputed. E is the signed projection of the gap-reference difference on the actual learned direction; it must be enclosed from those vectors without a norm or absolute value. If an interval norm includes zero without proving an exactly zero gap, direction and error remain unresolved. Individual conditional error alone cannot screen this margin: the former `D<ec` shortcut is removed, because shared conditional/unconditional error can cancel in E. A rounded point estimate does not certify the sign; the default numerical assessment remains separately labeled. GPU outward arithmetic retains overlapping or nonfinite enclosures as unresolved; it does not invent a value for V or M.

A condition-sign enclosure does not independently certify the saved gain, establish source-perturbation robustness or identify the checkpoint training law. Gain, condition, arithmetic, quadrature and source receipts remain separate. Directional variation estimates remain available independently of sign resolution; historical norm diagnostics do not replace them.

For a validated support-diameter upper bound `L_D`,

```
||d bar_x_next(z(s))/ds|| <= beta*||h||*L_D^2/4.
```

The derivative is `beta*Cov(U|z)*h`; every unit-direction variance is at most `L_D^2/4`. Thus the signed projection `f(s)=<u,bar_x_next(z(s))-bar_x_current>` for fixed unit u is Lipschitz with that same bound. A width-w midpoint cell contributes truncation at most `L_f*w^2/4`. A useful equivalent general bound for stored arbitrary logits/slopes is `L_f <= L_D*(max_j a_j-min_j a_j)/4`, by the covariance range bound. Directed arithmetic includes the node posterior/mean/projection evaluations and the bound construction. Adaptive splitting chooses the largest enclosure contribution; it is not based on SSCD or an attractive measured sign. For two references in the same convex hull, `-L_D<=W<=L_D` and `0<=mathcal{V}<=L_D`. Sum the signed cell intervals before mapping the final integral enclosure through max(0,·); clipping each node or cell first is a different integral. A legacy rounded-weight payload that lacks an exact unit-mass contract cannot use that cap without a mass-defect correction.

Propagate the refined margin as

```
mathcal{V}_lower = max(0,W_lower)
mathcal{V}_upper = max(0,W_upper)
M_lower = D_lower-E_upper-mathcal{V}_upper
M_upper = D_upper-E_lower-mathcal{V}_lower.
```

In interval mode, strictly positive lower and strictly negative upper bounds certify the corresponding sign within the recorded scope. A boundary or exhausted interval budget stays unresolved in that mode. Embedded quadrature disagreement remains an estimate, not a certificate. Repeated agreement at higher precision also remains a convergence diagnostic, not an enclosure.

For fixed coefficients and law, both `||grad log p||` and `||grad logit p||` are bounded by `beta*R_K`. A justified endpoint radius therefore contributes at most `beta*R_K*radius`; two perturbed endpoints contribute the sum of their radii. The code may transfer an affine implication to a recorded endpoint using an enclosed, actually computed saved-versus-affine residual. It does not invent a radius from dtype epsilon. Stochastic matched baselines recovered from the saved endpoint are labelled constructed shared-innovation baselines, never independent replay.

The refined margin, affine gain, and saved gain retain explicit endpoint contracts. Positive-margin implication audits require the same affine path or a justified endpoint transfer. H/G sign consistency, endpoint concavity, independent integrated-gain comparisons when their endpoint contract matches, and retained refined-margin ordering remain separately auditable. A resolved contradiction is a publication blocker with offending identities; missing or inconclusive arithmetic is a different status.

## Cache and execution design

`numerical_reduce.run_precision_analysis` wraps the established evidence stage. Compatible published evidence is followed by its exact manifest receipt. On normal analysis, missing evidence goes through the existing resumable direct/evidence stages. On `--refine-numerics`, a missing or incompatible required source produces an exact recomputation command and **never** falls through to learned probes or upstream stages.

New data live only below `theory_measurements/numerical_refinement/`:

- `inputs/<hash>/`: independently versioned finite sufficient tensors, batch identifiers and receipts, with pointers to immutable original inputs; no duplicated raw histories;
- `policies/<hash>/records/<hash>/`: additive per-row results, independently resumable batch completions, original classifications, budgets, and failure receipts;
- `collections/<hash>/`: one additive matched-update scalar table, original authoritative file references, exact one-to-one join instructions, and migration provenance.

Changing the numerical budget changes policy identities. It does not invalidate forward-loss draws, native Gaussian probes, generation, SSCD, proximity, geometry supplements, or already compatible directional-variation observations. The mathematical switch from norm to signed projection does require a new integral recipe; old norm observations cannot be reused as the new quantity. Input construction and numerical retry algorithms have separate identities. The directional payload is version 4 and stores the actual learned-gap vector; CUDA numerical policy 7 consumes signed-projection receipts. Projection roundoff is direct_prop5_projection_roundoff_l2; the old Gram allowance remains only a norm-diagnostic field. Style-only changes require only plot mode.

Workers use the existing device resolver, round-robin record sharding, spawned processes, bounded query/candidate work, thread limits, and one parent-owned progress bar. When interval certification is requested, the batched current-posterior enclosure runs on the selected worker GPU. A still-unresolved condition, independently flagged gain or justified source-radius request enters endpoint/refinement work using the same posterior intervals. Gain triggers are nonfinite stable values, unresolved stable log-odds sign, a resolved H/G sign disagreement or rounded-zero ambiguity against the original vectors. H underflow with resolved G and disagreement with old absolute-log-odds subtraction remain diagnostics; neither alone forces extra gain enclosure. There is no CPU fallback.

GPU refinement remains within its owning analytical worker; it creates no learned-model replicas or central process that serializes all GPU workers. Budget preflight occurs before required raw-vector device transfers and interval construction; a prepared CUDA batch is reused across its seed rows. Per-row stopping reasons distinguish an unattempted unaffordable enclosure, exhausted work, and a completed enclosure. Unresolved mass stays visible. Point-estimate and interval modes evaluate the same mathematical condition and support, with distinct saved arithmetic scopes.

Resume checks operate before tensor loading. Complete numerical records require no raw-record reload. Within an incomplete record, valid completed batch receipts can be reused without recreating witnesses or loading their payload tensors. A raw record is loaded lazily only when a genuinely pending input or refinement batch needs its original vectors; subsequent pending batches share that loaded record. Input identities remain separate from policy identities. The default point-assessment policy changes the numerical policy hash (`fixed-cache-numerics-cuda-8`), while compatible finite sufficient inputs and all upstream measurements remain reusable. Only small progress/receipt messages cross worker boundaries, and workers do not create their own bars.

Normal analysis and `--recompute-experiments` default to zero interval operations and do not inherit a previously saved positive budget. `--refine-numerics` explicitly opts in: it retains a saved positive budget, or uses 2,000,000 charged interval operations per row otherwise. An explicit `--numerical-max-products` (legacy alias `--numerical-max-decimal-products`) always wins, including zero. `--plot` and `--validate-only` retain the saved policy. Optional interval mode uses CUDA binary64, at most 65 variation nodes by default, and requested variation enclosure width `1e-6` in raw latent L2 units. Its per-row budget counts screen, endpoint and quadrature work together. Budgets below unavoidable screen work skip it; a partially exhausted screen stops explicitly rather than repeating the posterior. Shared setup is conservatively charged within each seed row budget. No precision reduction or larger budget is needed to display the default point estimates.

The existing direct stage owns the requested signed directional integral measurements. Technical failure of required source observations is not reclassified as a favorable sign or silently published as a complete evidence suite. Within any available numerical row, missing V does not remove an independently resolved negative-condition sign.

## Figure and scalar contracts

- **Theorem 1:** neutral shared target-error RMS axis; the same faint vertical pairing rule for every conditional/unconditional pair; unchanged loss draws, seed-coherent bootstrap, MC intervals, colors and Gaussian-transfer appendix.
- **Lemma 2:** the main analytical curve spans the specified lower-SNR grid through initialization, with only three separate native initial markers there. The complete native Gaussian sweep remains mandatory and linked in the caption. No extrapolated network tail or main `s_K` line.
- **Corollary 3:** equal aspect and uncut adverse points/ideal marker; directly labelled fixed 0.25/0.5/1 error contours. Saved sample/pair quantiles, contour coverage, `A=(ec+bu)/||v||`, and the cached vector-error inner product audit the full geometry without replacing it.
- **Lemma 4:** neutral points with actual counterfactual provenance markers; full vector residuals retained. Equal norms alone do not establish the vector identity.
- **Proposition 5:** one structural population and common prompt-balanced weights per timestep/group, including unknown signs. Each event reports `[positive_mass, positive_mass+unknown_mass]`, separately from unweighted counts. Positive mass includes clear numerical estimates by default, or optionally certified signs with their recorded scope. The A4 margin/gain scatter shows every finite saved pair as an ordinary dot, irrespective of its numerical sign classification, with no Observed/Unresolved marker legend. Only the pooled margin plot is published, at opacity 0.01; the shared scatter opacity is 0.5. Per-step views are not exported. Nonfinite pairs remain disclosed without fabricated values. Sign and interval receipts remain in audits and continue to determine the retained condition-fraction table classifications; that table is not exported as an extra figure. This range is neither a confidence interval nor certified condition coverage. Light feedback bands and thin condition boundaries replace hatching. Shared zero condition baselines disclose unresolved counts. Retained resolution tables preserve all three uncertainty layers and complete cause/status receipts; endpoint-adjacent manuscript `t=1` rows remain a separate diagnostic.
- **Lemma 6:** paired per-row `Q=max(conditional_target_error,unconditional_target_error)`, D and S before aggregation. The main synchronization figure retains Q and its IQR; appendix `synchronization_bound` displays only solid D and dashed S medians, without a band. Row-level `D<=S` uses `S=E+R*(1-p)`; Q is separately descriptive and Q<=S is not asserted. Fixed `.1,.2,.3,.5,1` target-error tolerance coverage remains saved. These arithmetic audits use declared numerical-assessment tolerances, not formal interval certificates.
- **Theorem 7:** unchanged three full-range weighted tolerance ECDFs and clean-case count; non-clean runs explicitly use the finite-terminal-update extension. Zero mass stays in caption data, and nonzero mass remains visible. Exact paired looseness is `B_ref-B_obs=(g-1)*(S_raw-D_raw)=(g-1)*(slack_radius+slack_projection_alignment)`, where `slack_radius=R_K*(1-p_K)-||bar_x_K-x_star||` and `slack_projection_alignment=E+||bar_x_K-x_star||-D_raw=||bar{Delta}||-u dot bar{Delta}`. The same independently computed scheduler correction cancels.

Current paper bundle schema is 5; compact metric schema is `four-stage-evidence-projected-gap-error-1`. Portable plot-only reads compact scalar CSVs and small manifests. It never opens trajectories, computes posteriors/integrals, refits centers, reruns draws, or hashes large tensor inputs.

## Author execution and stopping criteria

Tests are supplied but unexecuted. They cover numerical examples, sign/value separation, estimated versus enclosed signs, fixed denominators and endpoint contracts, rendering and publisher isolation, CLI forwarding, independent cache identities, and input preservation. The normal analysis command uses point estimates; interval certification is a separate optional request:

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto
# Optional interval certification of existing evidence:
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --refine-numerics --device auto
./run_all.sh --model sdv1 --scheduler ddim --plot
```

If no compatible fixed evidence collection exists, use the reported full derived-analysis command instead:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto
```

Inspect all audit offenders, numerical cause counts, unknown mass, negative gains, amplification errors, revised-condition coverage, and terminal qualifications. Success does not require positive condition coverage, small geometry error, a tight reference bound, or an affirmative empirical conclusion. The manuscript still requires author alignment on finite noise, the empirical law, source sensitivity, and the finite-terminal extension.

When requested, the optional screen reports outward lower/upper intervals for D and E and the upper bound on D-E. An inconclusive negative screen does not prove a nonnegative full margin because V remains. The operation preflight only selects affordable work; it is not a sign certificate. Saved audit fields separate total GPU work, screen work, endpoint execution and reuse of the current posterior. The retired ec-only comparison supplies no condition sign.