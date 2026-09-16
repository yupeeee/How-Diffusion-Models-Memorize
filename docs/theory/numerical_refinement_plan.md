> The active figure organization is now [four_stage_experiments.md](four_stage_experiments.md). Seven-primary filenames and roles below describe the preceding design; compatible mathematical measurements and numerical contracts remain reusable. Matching LaTeX is unavailable; historical label strings are author-supplied, not verified source labels.

# Precision and presentation refinement of the fixed evidence suite

This document describes source changes, not generated results. No tests, project imports, experiments, downloads, inference, posterior calculations, quadrature, plotting, benchmarks, or GPU jobs were executed for this revision. Existing scientific caches, logs, figures, and historical reports remain untouched. The author will execute the updated code.

## Authority and fixed outputs

The mathematical authority remains the mounted `revised.pdf` identified in [submission_alignment.md](submission_alignment.md) and the author's precise seven-statement brief. A matching LaTeX source is not available in this checkout. The original Proposition 5 keeps its full reference errors and current-reference variation integral on manuscript `t>=2`; the original Theorem 7 keeps its clean-update prerequisite.

The seven primary stems are unchanged. The mandatory appendix registry adds only `lemma2_native_gaussian_sweep` and `proposition5_numerical_resolution`: twelve appendix entries total, with terminal components conditional on a supported terminal contract. One stem produces one data axes, one PNG, and one single-page PDF. The established proximity publisher, canonical output paths, safe staged publication, rollback, and ownership-based archival remain in use.

## Three uncertainty layers

1. **Fixed saved-input arithmetic.** The function is defined by the stored states, already-canonical epsilon predictions, target/support atoms, positive prior masses, and saved scalar scheduler coefficients. Float64 posterior-gain computations remain numerical assessments; the separately implemented outward-arithmetic condition screen certifies only its declared squared comparison. Selective directed Decimal intervals may enclose this function from those raw inputs. An enclosure of rounded sufficient logits alone is explicitly labelled a reduced-input enclosure, not a certificate of constructing those logits from original vectors.
2. **Source/reconstruction sensitivity.** Legacy dtype-multiplier diagnostics remain saved as sensitivity assessments. They are not mathematical uncertainty radii. Native coefficient derivation, checkpoint approximation, conversion history, and an unspecified perturbation of stored vectors are not silently covered by a fixed-input certificate. Actual saved-versus-affine endpoint residuals are recorded separately.
3. **Reference identification.** All calculations use the same declared `D_K`. Its discrepancy from the complete checkpoint training distribution remains unidentified; extra precision cannot reduce it.

The schema records `input_contract_status`, `endpoint_construction_method`, `arithmetic_status`, `arithmetic_error_method`, `quadrature_status`, `quadrature_error_scope`, `source_sensitivity_status`, `source_sensitivity_model`, `reference_law_id`, `fixed_cache_gain_sign`, `source_robust_gain_sign`, `condition_sign_status`, and `condition_value_status`. Per-row precision, operation/node budgets, stopping reason, legacy classification, enclosure widths, and publication blockers remain in the numerical row audit. A migration receipt identifies the change from the legacy source-sensitivity interpretation to the fixed-input interpretation. Old measurements/classifications are retained in their original columns and files.

## Auxiliary numerical derivations

These are numerical tools for evaluating the original objects, not new experimentally verified manuscript results.

For a nonempty target complement, with target-relative log densities `b_j`, endpoint slopes `a_j=beta*<h,u_j-x_star>`, and `nu=softmax_non_target(b)`, evaluate

```
G = -log(sum_j nu_j*exp(a_j)).
```

Center the log-sum-exp; do not obtain the new net gain only by subtracting two enormous absolute log odds. For bounded small slopes, evaluate `-log1p(sum_j nu_j*expm1(a_j))`, accumulating positive and negative terms separately and flagging cancellation for refinement. Large slopes never enter an unbounded expm1 branch. The old endpoint difference stays as a cross-check. Log-probability gain `H` retains its own magnitude, including saturation/underflow; `G` may resolve a sign but cannot replace that magnitude. Zero displacement and a single-atom law have explicit semantics; singleton log odds are undefined.

Let `D=||Delta||`, `ec=||m_c-x_star||`, `eu=||m_u-bar_x_current||` and

```
V = integral_0^1 ||bar_x_next(z0+s*h)-bar_x_current|| ds
M = D-ec-eu-V.
```

Since `V>=0`, `M <= D_upper-ec_lower-eu_lower`. A strictly negative upper bound proves the original condition false even if V is missing. A still cheaper sufficient precheck uses `eu>=0`. Neither precheck sets V to zero or manufactures a point value for M. Requested legacy V measurements remain separately available.

The `fixed-cache-numerics-2` policy performs the cheaper `D<ec` test as one batched CPU/CUDA float64 interval screen before considering Decimal work. With positive saved alpha, the common division by alpha cancels and squaring preserves this comparison:

```
sigma^2 * ||epsilon_u-epsilon_c||^2
    < ||state-sigma*epsilon_c-alpha*target||^2.
```

The screen propagates outward `nextafter` bounds through its arithmetic and reductions; a left upper bound strictly below the right lower bound proves `D<ec`, hence `M<0` because `eu,V>=0`. It uses the original saved canonical-epsilon/state/target values and coefficients, and requires no square roots, division, posterior evaluation, or Decimal conversion. A rounded point estimate alone does not certify the comparison. Nonfinite or overlapping bounds leave the condition unresolved for the declared fallback policy. The saved method, interval gap and status define this narrow certificate's scope.

This condition-only certificate does not promote an independently computed float64 gain assessment into a gain certificate. It also does not manufacture a value of V or M, establish source-perturbation robustness, or identify the checkpoint's training distribution. The complete condition/gain/arithmetic/source status fields remain separate, including when Decimal fallback is disabled.

For a validated support-diameter upper bound `L_D`,

```
||d bar_x_next(z(s))/ds|| <= beta*||h||*L_D^2/4.
```

The derivative is `beta*Cov(U|z)*h`; every unit-direction variance is at most `L_D^2/4`. Thus `f(s)=||bar_x_next(z(s))-bar_x_current||` is Lipschitz with that same bound. A width-w midpoint cell contributes truncation at most `L_f*w^2/4`. A useful equivalent general bound for stored arbitrary logits/slopes is `L_f <= L_D*(max_j a_j-min_j a_j)/4`, by the covariance range bound. Directed arithmetic includes the node posterior/mean/norm evaluations and the bound construction. Adaptive splitting chooses the largest enclosure contribution; it is not based on SSCD or an attractive measured sign. For two references in the same convex hull, `0<=V<=L_D`. A legacy rounded-weight payload that lacks an exact unit-mass contract cannot use that cap without a mass-defect correction.

Propagate the original margin as

```
M_lower = D_lower-ec_upper-eu_upper-V_upper
M_upper = D_upper-ec_lower-eu_lower-V_lower.
```

Strictly positive lower and strictly negative upper bounds resolve the corresponding sign. A boundary or exhausted budget stays unresolved. Embedded quadrature disagreement remains an estimate, not a certificate. Repeated agreement at higher precision also remains a convergence diagnostic, not an enclosure.

For fixed coefficients and law, both `||grad log p||` and `||grad logit p||` are bounded by `beta*R_K`. A justified endpoint radius therefore contributes at most `beta*R_K*radius`; two perturbed endpoints contribute the sum of their radii. The code may transfer an affine implication to a recorded endpoint using an enclosed, actually computed saved-versus-affine residual. It does not invent a radius from dtype epsilon. Stochastic matched baselines recovered from the saved endpoint are labelled constructed shared-innovation baselines, never independent replay.

The original margin, affine gain, and saved gain retain explicit endpoint contracts. Positive-margin implication audits require the same affine path or a justified endpoint transfer. H/G sign consistency, endpoint concavity, independent integrated-gain comparisons when their endpoint contract matches, and retained refined-margin ordering remain separately auditable. A resolved contradiction is a publication blocker with offending identities; missing or inconclusive arithmetic is a different status.

## Cache and execution design

`numerical_reduce.run_precision_analysis` wraps the established evidence stage. Compatible published evidence is followed by its exact manifest receipt. On normal analysis, missing evidence goes through the existing resumable direct/evidence stages. On `--refine-numerics`, a missing or incompatible required source produces an exact recomputation command and **never** falls through to learned probes or upstream stages.

New data live only below `theory_measurements/numerical_refinement/`:

- `inputs/<hash>/`: independently versioned finite sufficient tensors, batch identifiers and receipts, with pointers to immutable original inputs; no duplicated raw histories;
- `policies/<hash>/records/<hash>/`: additive per-row results, independently resumable batch completions, original classifications, budgets, and failure receipts;
- `collections/<hash>/`: one additive matched-update scalar table, original authoritative file references, exact one-to-one join instructions, and migration provenance.

Changing the numerical budget changes policy identities. It does not invalidate forward-loss draws, native Gaussian probes, generation, SSCD, proximity, geometry supplements, or original variation observations. Input construction and numerical retry algorithms have separate identities. Style-only changes require only plot mode.

Workers use the existing device resolver, round-robin record sharding, spawned processes, bounded query/candidate work, thread limits, and one parent-owned progress bar. The batched outward float64 condition screen runs before selective CPU fallback. Only a still-unresolved condition or a gain that needs refinement enters the fixed-budget Decimal path. Gain refinement triggers are nonfinite stable values, unresolved stable log-odds sign, a resolved H/G sign disagreement, or a rounded-zero ambiguity against the original vectors. Raw H underflow with a resolved G sign and disagreement with the old subtraction of absolute log odds remain recorded diagnostics; neither alone forces an expensive fallback. H retains its own measured magnitude.

Fallback remains within its owning analytical worker; it creates no learned-model replicas or central process that serializes all GPU workers. Budget preflight occurs before expensive raw-vector conversion to Python/Decimal. Per-row stopping reasons distinguish an unattempted unaffordable enclosure, exhausted work, and a completed enclosure. Unresolved mass stays visible; the optimization does not relax the mathematical condition, alter the support, or change default budgets.

Resume checks operate before tensor loading. Complete numerical records require no raw-record reload. Within an incomplete record, valid completed batch receipts can be reused without recreating witnesses or loading their payload tensors. A raw record is loaded lazily only when a genuinely pending input or refinement batch needs its original vectors; subsequent pending batches share that loaded record. Input identities remain separate from policy identities. The new screening/retry implementation changes the numerical policy hash (`fixed-cache-numerics-2`), while compatible finite sufficient inputs and all upstream measurements remain reusable. Only small progress/receipt messages cross worker boundaries, and workers do not create their own bars.

Default policy: 64 Decimal digits, a retry at 128 digits, at most 2,000,000 charged Decimal operations across both attempts per row, at most 65 variation nodes, and target variation enclosure width `1e-6` in raw latent L2 units. The historic option name `--numerical-max-decimal-products` counts charged interval operations, including transcendental operations, rather than only scalar products. A zero budget disables Decimal fallback; the batched condition screen still runs, and gain assessment versus condition-certificate statuses remain distinct. These conservative budgets can leave substantial unknown mass; execution must establish how much.

The existing direct stage still owns requested original integral measurements. Technical failure of required source observations is not reclassified as a favorable sign or silently published as a complete evidence suite. Within any available numerical row, missing V does not remove an independently resolved negative-condition sign.

## Figure and scalar contracts

- **Theorem 1:** neutral shared target-error RMS axis; the same faint vertical pairing rule for every conditional/unconditional pair; unchanged loss draws, seed-coherent bootstrap, MC intervals, colors and Gaussian-transfer appendix.
- **Lemma 2:** the main analytical curve spans the specified lower-SNR grid through initialization, with only three separate native initial markers there. The complete native Gaussian sweep remains mandatory and linked in the caption. No extrapolated network tail or main `s_K` line.
- **Corollary 3:** equal aspect and uncut adverse points/ideal marker; directly labelled fixed 0.25/0.5/1 error contours. Saved sample/pair quantiles, contour coverage, `A=(ec+bu)/||v||`, and the cached vector-error inner product audit the full geometry without replacing it.
- **Lemma 4:** neutral points with actual counterfactual provenance markers; full vector residuals retained. Equal norms alone do not establish the vector identity.
- **Proposition 5:** one structural population and common prompt-balanced weights per timestep/group, including unknown signs. Each event reports `[positive_mass, positive_mass+unknown_mass]`, separately from unweighted counts. Light feedback bands and thin condition boundaries replace hatching. Shared zero condition baselines disclose unresolved counts. The resolution appendix preserves all three uncertainty layers and complete cause/status tables; endpoint-adjacent manuscript `t=1` rows remain a separate diagnostic.
- **Lemma 6:** paired per-row `Q=max(conditional_target_error,unconditional_target_error)`, D and S before aggregation, with Q-only IQR. Row-level `D<=S`, `Q<=S` and fixed `.1,.2,.3,.5,1` RMSE tolerance coverage remain saved. These arithmetic audits use declared numerical-assessment tolerances, not formal interval certificates.
- **Theorem 7:** unchanged three full-range weighted tolerance ECDFs and clean-case count; non-clean runs explicitly use the finite-terminal-update extension. Zero mass stays in caption data, and nonzero mass remains visible. Exact paired looseness is `B_ref-B_obs=(g-1)*(S_raw-D_raw)=(g-1)*(slack_radius+slack_triangle)`, where `slack_radius=R_K*(1-p_K)-||bar_x_K-x_star||` and `slack_triangle=ec+eu+||bar_x_K-x_star||-D_raw`. The same independently computed scheduler correction cancels.

Paper bundle schema is 4; compact metric schema is `precise-evidence-2`. Portable plot-only reads compact scalar CSVs and small manifests. It never opens trajectories, computes posteriors/integrals, refits centers, reruns draws, or hashes large tensor inputs.

## Author execution and stopping criteria

Tests are supplied but unexecuted. They cover numerical examples, sign/value separation, fixed denominators and endpoint contracts, rendering and publisher isolation, CLI forwarding, independent cache identities, and input preservation. Run them before the scientific retry:

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --refine-numerics --device auto
./run_all.sh --model sdv1 --scheduler ddim --plot
```

If no compatible fixed evidence collection exists, use the reported full derived-analysis command instead:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto
```

Inspect all audit offenders, numerical cause counts, unknown mass, negative gains, amplification errors, original-condition coverage, and terminal qualifications. Success does not require positive condition coverage, small geometry error, a tight reference bound, or an affirmative empirical conclusion. The manuscript still requires author alignment on finite noise, the empirical law, source sensitivity, and the finite-terminal extension.

The batched screen also keeps a lower squared-difference bound. A nonnegative lower bound establishes only `D>=ec`, so the cheap negative test cannot help; it does not establish a nonnegative original margin. In that case the operation-budget preflight includes the unavoidable raw-posterior projection. Its conservative charge floor is `[22*(K-1)+10*K]*d`, covering two target-relative logit constructions and both slope arrays, while omitting all other work. This prevents entering a CPU calculation that cannot finish under the declared policy.
