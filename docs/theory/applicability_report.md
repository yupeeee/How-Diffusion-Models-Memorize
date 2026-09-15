# Theory applicability and reference-availability report

## Status and source of truth

**All four primary schema-3 analyses completed on three GPUs**, with zero failed
records. They retain all 837 frozen prompts, 16,740 evaluation samples, and
837,000 stored-prediction rows. The complete scalar audit, combined guarded
rendering, visual review, bounded protected-cache comparison, and managed-image
archival are complete.
The protected-cache comparison records changes observed during the user's concurrent
matrix rerun and does not claim a full content hash of every upstream artifact.
Earlier CPU analyses are supplemental and are not mixed into the primary counts.

The authoritative manuscript is `revised.pdf`, SHA-256
`fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`.
Measurements use schema 3 and formula version
`cache-theory-3.0-proposition5`; see
[quantity_dictionary.md](quantity_dictionary.md) for exact formulas and units.
The completed results retain zero estimated sufficient-condition coverage,
negative gains, and unresolved numerical rows. Old schema-2 competitor-margin
scalars are not evidence for the corrected Equation-15 condition.

An initial three-GPU attempt exposed a Boolean scatter-reduction dtype defect.
The corrected integer accumulator was exercised by all four completed GPU runs.
The assistant's host still has no visible CUDA device; actual GPU execution is
established by the user's published manifests, not by the locally skipped CUDA
unit test.

## Preserved configurations and expected observations

The same frozen selection, evaluation seeds, source trajectories, native
schedulers, target latents, and same-seed SSCD outcomes are retained. The expected
counts below come from preserved complete configuration manifests and the newly
launched validation logs; they are not new condition-coverage results.

| Configuration | Retained prompts | Evaluation seeds per prompt | Initial/endpoint samples | Stored-prediction rows |
|---|---:|---:|---:|---:|
| SDv1 / DDIM, g=7.5, T=50 | 248 | 20 | 4,960 | 248,000 |
| SDv1 / DDPM, g=7.5, T=50 | 260 | 20 | 5,200 | 260,000 |
| SDv2 / DDIM, g=7.5, T=50 | 86 | 20 | 1,720 | 86,000 |
| RealVis / DDIM, g=7.5, T=50 | 243 | 20 | 4,860 | 243,000 |
| Total | 837 | — | 16,740 | 837,000 |

Evaluation seeds are 0–19; reference seeds are 20–39. Every evaluation seed of
every retained prompt remains included. Outcome groups do not prune the analysis.
The initial unconditional ECDF has **20 unique seed inputs per configuration**;
repeated evaluations of these inputs under different prompt records do not
multiply that sample size.

## Reference availability

| Required object | Availability and interpretation |
|---|---|
| Saved current/destination states and canonical-epsilon branches | Available for preserved complete records. T predictions correspond to T+1 states; the endpoint has no additional branch prediction. |
| Same-seed terminal target SSCD | Available from protected saved scoring results, joined by complete sample identity. |
| Initial model-output center `mu_ref` | Available from the existing reference seed block, subject to repeated-input consistency checks. One canonical record is selected deterministically without outcome inspection. The center is fixed across the analysis. |
| Candidate distribution K | Available from all compatible valid complete target records in one declared experiment source run, before retained-prompt filtering. Exact duplicate latent atoms receive one default equal mass; aliases and source hashes remain recorded. Completed schema-3 banks have 442 distinct atoms for SDv1/DDIM, SDv1/DDPM, and RealVis/DDIM, and 401 for SDv2/DDIM. Every bundle records its bank and source hashes. |
| Candidate mean `mu_K`, posterior, reference error, and target-specific radius | Analytically available under that declared finite law when target membership/noise conditions permit. These are not identified full-training quantities. |
| Actual training law, mean `mu`, posterior, and full support radius | Unavailable from ordinary trajectory caches. No candidate or model-center result silently substitutes for these. |
| Pair-specific forward-loss premise | Unavailable: saved reverse-trajectory residuals are not independently evaluated forward-corruption losses. |
| Single-target training-law assumption | Unverified. A downloaded paired caption/image does not establish `Pr(X=x_star|C=c)=1`. |
| Learned branches on new counterfactual segment nodes | Unavailable without new inference and never evaluated by this analysis. Segment integration uses only analytic candidate posterior means. |

`mu_ref`, `mu_K`, and the unidentified training mean are deliberately distinct.
Repeated initial states must agree exactly, while unconditional epsilon outputs
must pass the declared source-precision check. Material disagreement is a failure,
not a reason to average inconsistent observations or fit the center to evaluation
samples. Reference seeds remain disjoint from evaluation but also informed the
existing frozen selection.

The Appendix-D.1 Eq.31 Gaussian-transfer upper bound is available only when the
recorded initialization contract is compatible: sampler contract version 2,
standard initial scale, and the VP coefficient relation. It is an analytic bound
on intended Gaussian distributions, not measured total variation and not a
substitute for the unavailable forward loss.

## Exact terminal scheduler applicability

The adapter requires all Eq.5 structural conditions before testing an observed
residual: the final update must be affine in the raw clean estimate, the
current-state scalar coefficient must be zero, kappa must equal one, destination
alpha/sigma must be 1/0, and the additive noise coefficient must be exactly zero.
Here `A_state` is the scheduler's internal `A`, distinct from the Eq.16 conditional
error magnitude A used by the optional terminal-terms diagnostic.

The following values are the recorded float values from the preserved schedules.
All final saved prediction indices are k=49, at scheduler training timestep 1.

| Configuration | Destination timestep | Destination alpha | Destination sigma | A_state | kappa | Additive noise standard deviation |
|---|---:|---:|---:|---:|---:|---:|
| SDv1 / DDIM | -19 | 0.999574898724882 | 0.029155134010013496 | 0.7062900746235099 | 0.2938868318904333 | 0 |
| SDv1 / DDPM | -1 | 1 | 0 | 0 | 1 | 9.999999841327611e-11 |
| SDv2 / DDIM | -19 | 0.999574898724882 | 0.029155134010013496 | 0.7062900746235099 | 0.2938868318904333 | 0 |
| RealVis / DDIM | -19 | 0.999574898724882 | 0.029155134010013496 | 0.7062900746235099 | 0.2938868318904333 | 0 |

**All four configurations fail the exact structural Eq.5 gate.** The three DDIM
configurations have nonclean destination coefficients. DDPM has clean destination
mean coefficients but retains the native positive variance floor at the last
saved training timestep. A particular endpoint residual can round to zero without
making that stochastic coefficient structurally zero.

No scheduler setting, clipping option, timestep, zero-terminal-SNR rescaling, or
trajectory is changed to make the condition pass. Therefore these configurations
cannot produce an applicable `terminal_terms` figure. Their actual endpoint errors,
last guided-clean errors, Eq.16 A/B magnitudes and vector cross terms, candidate
Theorem-7 expression, and `rho=x0-m_g,1` discrepancy are still logged separately.
They do not establish the manuscript's terminal reproduction guarantee.

The A+B triangle bound, candidate-radius Theorem-7 expression, and operational
scheduler bound are three distinct quantities. An endpoint-derived rho does not
turn any of them into an ex-ante certificate. Ratios with zero observed error are
undefined and explicitly counted rather than silently set to zero.

## What each designated figure can support

| Default figure | What is measured | What it cannot establish |
|---|---|---|
| `initial_recovery` | Initial paired unconditional/conditional target RMSE, colored by that sample's terminal target SSCD. The diagonal means equal target errors. | Forward-loss premises, convergence as initial SNR tends to zero, or accurate reproduction without an independently supplied latent tolerance. |
| `unconditional_center` | Initial-only empirical CDF of held-out unconditional dispersion around the declared fixed model-output center, using unique seeds and no SSCD coloring. | Agreement with the training-data mean or convergence measured from later generated states. |
| `posterior_feedback` | `M_t^K/sqrt(d)` against destination matched log-probability gain `H_t^K`, using the Equation-15 integral under a fixed candidate law. Negative gains, unmet conditions, and unresolved rows remain visible. | Identification of the training posterior or a formal numerical certificate. It is a candidate-distribution diagnostic with estimated quadrature/sign classification. |
| `target_synchronization` | Conditional/unconditional target-error summaries in two fixed same-seed SSCD cohorts, with prompt-balanced medians/interquartile ranges and separate paired joint-error tables. | The Bayes-reference premises of Lemma 6, universal catch-up ordering, or small paired joint error inferred only from marginal medians. |

The optional `target_injection` diagnostic retains both the target-direction
coefficient and perpendicular component, with the `(g,0)` limiting reference.
Positive alignment alone is not agreement with the full limiting vector. Optional
terminal terms require the Eq.5 gate described above. Matched-displacement images
are never produced; the vector identity remains algebraic QA.

The candidate feedback integrator uses adaptive Gauss–Kronrod 15/Gauss 7,
absolute/relative V tolerances 1e-5, identity absolute tolerance 1e-7, and a default
4095-node budget per query. Source/Gram sensitivities and refinement diagnostics
are recorded. Coverage is numerically estimated; exhausted budgets and near-zero
margins remain unresolved. Stable endpoint log odds preserve strict ordering even
when displayed probability gains underflow to zero. No artificial positive gain
is inserted.

## Primary completed multi-GPU results

These tables use only the four explicitly identified GPU bundles below, reading
`manifest.json`, `applicability_report.json`, and the saved per-step feedback
summaries. CPU and GPU floating-point outputs need not be identical. Every
bundle preserves its own recorded source and device provenance.

| Configuration | Complete GPU bundle | Prompts / samples / prediction rows |
|---|---|---:|
| SDv1 / DDIM | [`81085d17…06bd83`](../../outputs/sdv1_ddim_g7.5_T50_N20/theory_v2/81085d1787c9e3f4f09b4f879ff3150cb94b3468ef01ae8b8b48e3104106bd83/manifest.json) | 248 / 4,960 / 248,000 |
| SDv1 / DDPM | [`2fe4344c…45e467`](../../outputs/sdv1_ddpm_g7.5_T50_N20/theory_v2/2fe4344cda4ba068436d6361d2c8413c265e9c9ba6efb791a719044d7f45e467/manifest.json) | 260 / 5,200 / 260,000 |
| SDv2 / DDIM | [`59fd6647…3df7f7`](../../outputs/sdv2_ddim_g7.5_T50_N20/theory_v2/59fd66475728e525fa8f189e3ac1ca8a865309214114aa16a046e87ea63df7f7/manifest.json) | 86 / 1,720 / 86,000 |
| RealVis / DDIM | [`4749db8a…6411c9`](../../outputs/realvis_ddim_g7.5_T50_N20/theory_v2/4749db8a1e496cc4419220452e93c90a45aa540d5a7e464a0991843b216411c9/manifest.json) | 243 / 4,860 / 243,000 |

| Configuration | Eligible feedback | Estimated met | Estimated not met | Numerically unresolved | Positive gain | Negative gain | Negative gain, SSCD>0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SDv1 / DDIM | 243,040 | 0 | 230,711 | 12,329 | 123,486 | 119,554 | 30,796 |
| SDv1 / DDPM | 254,800 | 0 | 243,289 | 11,511 | 138,094 | 116,706 | 7,434 |
| SDv2 / DDIM | 84,280 | 0 | 83,719 | 561 | 30,848 | 53,432 | 1,414 |
| RealVis / DDIM | 238,140 | 0 | 228,718 | 9,422 | 120,707 | 117,433 | 30,479 |
| Total | 820,260 | 0 | 786,437 | 33,823 | 413,135 | 407,125 | 70,123 |

Estimated condition coverage is **0 / 820,260**. All 33,823 unresolved conditions
remain in that denominator. There are 16,740 terminal transitions outside this
nonterminal comparison; these are inapplicable measurements, not failed records.
Both signs remain represented even in the high-SSCD group: its 70,123 negative-gain
transitions are retained. No eligible gain ordering is classified zero or
unresolved by the saved sign rule. This does not mean every numerical H is
nonzero: stable log-odds ordering can resolve its sign after H underflows to zero.

| Configuration | Endpoint saturation count | H-underflow count with sign retained | Integral QA flags already unresolved | Candidate atoms | Eq.5-applicable endpoints |
|---|---:|---:|---:|---:|---:|
| SDv1 / DDIM | 84,505 | 64,763 | 8 | 442 | 0 / 4,960 |
| SDv1 / DDPM | 87,696 | 67,133 | 17 | 442 | 0 / 5,200 |
| SDv2 / DDIM | 7,045 | 3,192 | 12 | 401 | 0 / 1,720 |
| RealVis / DDIM | 87,380 | 64,347 | 19 | 442 | 0 / 4,860 |

The 56 producer integral QA flags are retained as numerically unresolved. They
are explicitly distinguished from failed record computations. The detailed
[numerical QA investigation](numerical_qa_investigation.md) and JSON sidecars
record full scalar checks, sample identities, independent integration results,
and source hashes. The completed audit checked **71 contracts per configuration**
with zero hard measurement/structural contract violations, zero proof-slack
violations beyond saved allowances, zero exhausted integration budgets, and zero
computational record failures. No tolerance or producer classification was
replaced to obtain a favorable result.

The full investigation distinguishes small endpoint/backend roundoff from
embedded-quadrature underestimates. It also confirms a narrow missed-tail case
in SDv1/DDPM: tighter tolerance alone misses an approximately 3.354e-4 tail when
reusing the original partition. A separate wider-partition diagnostic recovers
the endpoint integral, while the saved condition remains unresolved. This is a
measured limitation of the numerical estimate, not a formal integration
certificate or a reason to alter unfavorable data.

Every completed center uses 20 reference inputs and 20 distinct initial
evaluation inputs. The candidate support has 442 exact atoms for SDv1 and RealVis,
and 401 for SDv2. All 16,740 terminal updates fail the structural Eq.5 gate, so
none qualifies for `terminal_terms`. This conclusion follows from the preserved
scheduler settings, not from whether a particular observed residual is small.

### Supplemental CPU audits and numerical interpretation

Earlier complete CPU analyses are retained separately where available, including
SDv2 bundle `80966a1f…48e50e` and RealVis bundle `8289be88…9c9f12`. Their counts
are **not** used in the primary GPU tables above. For example, the supplemental
SDv2 CPU analysis has 559 unresolved conditions and ten integral QA flags, while
the primary SDv2 GPU analysis has 561 and twelve. These differences are preserved
rather than conflating backend-specific floating-point results.

The independent investigation of the supplemental SDv2 CPU bundle traced nine
tiny endpoint/integral residuals (8.99e-13 to 2.41016e-11) to independently
computed endpoint logits versus the equivalent affine segment endpoint.
Independent integration agreed with the affine endpoint H within 7.11e-15.
The remaining CPU case, record 776379145 / seed 19 / step 48, has saved residual
3.31685e-4, narrowly above its 3.30643e-4 allowance. Tighter integration agreed
with endpoint H within 1.28e-8 but reported a roundoff-limit warning. Its unresolved
status remains unchanged.

The primary SDv1/DDIM GPU investigation identifies five tiny endpoint/backend
roundoff-scale discrepancies and three step-48 embedded-quadrature
underestimates. Tighter independent integration agrees with their endpoint H
within 4.51e-9, without convergence warnings for those three cases. All eight
margins are negative and their proof slacks positive. These checks diagnose the
flags; they are not formal integration certificates. The independent scalar
audit of this bundle found no hard contract violations.

The prior style-8 SDv2 CPU rerender remains supplemental evidence. The final
verification below instead uses all four primary GPU bundles.

### Final saved-data renderer and visual verification

The combined guarded renderer passed for all four primary GPU bundles. All
**3,456 numerical files, including the four bundle manifests, retained identical
SHA-256 hashes and modification times**. Metadata for all 20 auxiliary `.pt`
files was unchanged; the guarded renderer did not read their tensor contents.
Every one of the **16 PNGs was byte-identical** across the repeat render, and all
**32 PNG/PDF artifact hashes** validated.

Each configuration contains exactly the four designated default figures, using
`theory-measurement-contract-8`. All four configuration sets passed visual
inspection. Model loaders and raw tensor readers were guarded during the saved-data
render; no posterior calculation, quadrature, center fitting, or support rebuilding
was needed.

### Protected-cache comparison and its limits

The before/after comparison contains the same **25,542 protected files**, with no
added or removed files. It compares metadata for the full inventory and SHA-256
hashes for an **8,008-file subset**. All generation and SSCD metadata and checked
hashes were unchanged. Every selection CSV and proximity CSV retained its
SHA-256 hash.

During the user's concurrent full-matrix rerun, eight selection/proximity files
per configuration changed metadata: **32 files in total**. Of these, 28 were
byte-identical and four `proximity/experiment_S0_N20/summary.json` files changed
content. The comparison cannot establish which process wrote each file. The
earlier snapshot saved hashes rather than JSON payloads, so the exact changed
summary fields cannot be reconstructed. The comparison also does not claim to
have hashed all approximately 325 GiB of upstream cache content.
See the durable
[protected-cache integrity report](../../outputs/theory_v3_audit/protected_cache_integrity.json)
for the inventory, comparison scope, and concurrency caveat.

Managed-image archival moved **42 obsolete owned image artifacts** and verified
their hashes. Repeating the archival check found no remaining planned moves.
An old SDv1 analysis directory was already absent before archival; it is recorded
as absent, not as successfully archived. The combined
[rendering and integrity record](validation/render_and_integrity.json) pins the
final bundle identities, artifact hashes, and archival result separately from
the upstream-cache comparison.

## Runtime compatibility correction and execution evidence

The first GPU attempt failed because CUDA `scatter_reduce_(reduce="amax")`
does not support a Boolean accumulator. The invalid-node accumulator now uses
int64 0/1 values, preserving logical OR across chunks. The separate roundoff
accumulator remains float64. No variation, margin, gain, source-error formula,
or numerical threshold was changed by this device compatibility correction.

The targeted regression suite passed 23 tests, with one actual CUDA test skipped
on the CPU-only audit host. Old/new CPU feedback outputs were bit-identical
across all 40 recorded fields for eight synthetic cases and six real SDv2
transitions. The pre-fix CPU source is retained at
[`outputs/theory_v3_audit/cpu_source_before_cuda_fix`](../../outputs/theory_v3_audit/cpu_source_before_cuda_fix)
for completed CPU analyses that had already imported it. The primary GPU bundles
use the corrected implementation and record their own source hashes.

Each primary configuration used `cuda:0`, `cuda:1`, and `cuda:2`, with zero failed
records on every worker:

| Configuration | Records per worker, devices 0 / 1 / 2 | Recorded worker durations, seconds |
|---|---|---|
| SDv1 / DDIM | 83 / 83 / 82 | 144.17 / 145.55 / 143.82 |
| SDv1 / DDPM | 87 / 87 / 86 | 167.02 / 213.97 / 165.87 |
| SDv2 / DDIM | 29 / 29 / 28 | 52.57 / 54.51 / 55.70 |
| RealVis / DDIM | 81 / 81 / 81 | 138.85 / 141.96 / 140.16 |

These are **per-worker reduction durations**, not total end-to-end wall time or
a measured speedup against CPU. Coordinator validation, aggregation, and
rendering are separate stages. The manifests establish actual corrected
multi-GPU execution; the numerical results remain estimated candidate-law
diagnostics rather than formal theorem verification.

A bounded source review found no remaining Boolean scatter/gather/index-reduction
calls in the feedback, support, metrics, or scheduler adapter modules. The two
scatter reductions use matching float64 or int64 values and long indices on the
support device. Other Boolean values are masks, logical reductions, counts,
and assignments. This source audit complements the completed real GPU execution.

## Reproduction commands and remaining scope limits

Analysis-only recomputation and saved-data rendering use the existing matrix
filters:

```bash
./run_all.sh --recompute-experiments
CUDA_VISIBLE_DEVICES=0,1,2 ./run_all.sh --recompute-experiments --device auto
./run_all.sh --plot
./run_all.sh --plot --include-diagnostics
```

A filtered CPU invocation is:

```bash
PYTHON=/nas/home/juyeop/miniconda3/envs/py313/bin/python3.13 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
./run_all.sh --recompute-experiments --model sdv1 --scheduler ddim --device cpu
```

These analysis/plot modes preserve generation, target preprocessing, SSCD,
frozen selection, proximity measurements, and native scheduler settings.
Plot-only uses saved scalar/provenance files and never reads raw `.pt` tensors,
recomputes posteriors/quadrature, refits centers, rebuilds support, or loads
models. Old/missing-V schemas are rejected with an analysis-only recomputation
command. Plain `./run_all.sh` retains upstream resume behavior.

The assistant's tool environment has no visible CUDA devices (PyTorch
2.11.0+cu130); the separately executed user GPU runs supply the real device
execution evidence above. No run identifies the actual training distribution,
its Bayes-reference errors, or the forward-loss premise. Zero condition coverage,
negative gains, unresolved numerical estimates, and terminal incompatibility are
reported directly rather than interpreted as verification of those unavailable
premises.
