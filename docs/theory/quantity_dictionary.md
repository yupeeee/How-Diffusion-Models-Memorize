# Four-stage quantity and schema dictionary

The active plan is [four_stage_experiments.md](four_stage_experiments.md), bundle schema **5**, metric schema **`four-stage-evidence-1`**. Backing direct/probe/integration measurements keep their existing independent identities. All vector norms are raw L2; display RMSE is L2/sqrt(d) exactly once, and Gaussian exponents always use raw squared L2.

| Logical table | Population and quantities |
|---|---|
| `reference_law` | Immutable support/alias/mass manifest, full mean vector receipt, weighted spread S_K, and per-target maximum whole-support radius R_K |
| `initial_samples` | Pair/seed genuine Gaussian target errors, mean/reference errors, exact initialization and SSCD provenance; generated-state diagnostics are explicitly distinguished |
| `initial_pairs` | A_c=sqrt(L_hat/(d*SNR_T)), paired conditional/control RMS Y_c/U_c, independent forward and paired Gaussian bootstrap, pair mean SSCD |
| `initial_baseline_summary` | One canonical observation per unique Gaussian seed; B_T_K, S_K, ratio (undefined for zero spread), seed count and uncertainty; repeated observations have disagreement audits |
| `feedback_endpoints` | Alias plus explicit additive joins to preserved all-step `matched_updates`: coefficients, endpoint contract, H/G, original V and M, numerical/source/reconstruction status |
| `feedback_response` | Initial fixed-grid per-sample log-probability response and distinct log-odds gain; s=0,1/g,1 included; full-curve missingness/status retained |
| `trajectory_metrics` | Chronological k, native label, normalized progress, raw and normalized conditional/unconditional target errors, D, paired Q, full reference bound, noise gap and schedule scale |
| `branch_motion` | Consecutive prediction C=2<Delta,v_c>/d, U=-2<Delta,v_u>/d, `gap_motion_quadratic`=||v_c-v_u||²/d, squared-gap change/residual, target-error changes |
| `trajectory_shapes` | Raw-curve peak/tie/plateau/completeness descriptors, fixed-phase gaps, rise/decline and upward/downward variation; related prompt and mixed-outcome summaries |
| `terminal_metrics` | Alias plus preserved additive joins to `terminal`: actual saved final error, B_obs/B_ref, independent correction/mode, applicability, slack, ratios, and reference-bound looseness |
| `counterfactual_unconditional` | Optional learned empty-prompt target-error change at paired matched next inputs, same-context saved-prediction parity; no default inference requirement |
| `plot_data` | Immutable compact renderer inputs, shared-population summaries and transform settings; no raw tensor dependency for plotting |

The source keys contain run, record/pair, target identity, seed, chronological step and native source/destination labels where applicable. Outcome groups use paired-target same-seed SSCD > 0.75 versus <= 0.75. Prompt mass is equal within each represented group and divided among eligible member seeds. Initial main points are pair-level without an outcome split; their colors are pair mean SSCD. Motion components use means on identical rows so additivity is retained. Other group curves use descriptive weighted medians/IQRs. Unresolved signs remain in structural denominators.

Missing variation remains missing despite a resolved condition sign. Numerical arithmetic/quadrature, source sensitivity, and empirical-reference identification have separate status fields. Bootstrap intervals are empirical Monte Carlo uncertainty, IQRs are descriptive spread, and probabilistic terminal bounds carry their own saved simultaneous-run scope. None is relabeled a theorem certificate.

The historical dictionary below is retained to interpret compatible backing columns; it does not select the active figure registry.

# Backing direct measurements and previous compact schema

The preceding direct suite used paper bundle schema **2** and metric schema
**`direct-statements-1`**. Its backing measurement formula version remains
**`direct-seven-statements-1`** so compatible numerical shards stay reusable.
Older numerical columns keep their original meanings; the historical dictionary
below remains for interpreting those artifacts. It does not describe the active
default renderer. No measurements or tests were executed for this code change.

| Table | Input law and unit of observation | Direct quantities |
|---|---|---|
| `forward_loss_draws` | Independent forward corruption of one target, pair/step/draw | Raw squared epsilon loss, clean target error, forward identity residual, precision/RNG identity |
| `forward_loss_summary` | Pair/step mean over independent draws | Measured loss mean and MC SE, clean squared-error mean, SNR, Pinsker upper bound |
| `gaussian_conditional` | Fixed Gaussian seed for each pair/native measured step | Conditional target L2/squared L2/MSE/RMSE; SSCD only for an exactly matching saved initialization |
| `gaussian_reference` | Unique Gaussian run/seed/native step | Reference–law mean distance, learned–law mean distance, learned–reference error, decomposition cross term and proof bound |
| `forward_unconditional_loss` | Optional independent mixture atom/noise draw | Total, optimal and direct nonnegative excess losses; forward/Gaussian laws stay separate |
| `initial` | Every retained generated pair/seed at k=0 | Both full-vector Corollary 3 discrepancies and distinct g/(g−1) triangle RHS terms |
| `trajectory` | Every retained generated pair/seed/current step | Conditional error, unconditional reference error, target log probability/complement, whole-law radius, samplewise Lemma 6 RHS/gap |
| `matched_updates` | Every retained generated pair/seed/update | Independent or constructed Lemma 4 displacement, vector residual, original Proposition 5 variation/margin, independent endpoint and integrated gains, signed lower bound |
| `terminal` | Every retained generated pair/seed at last prediction and actual final output | Exact Theorem 7 RHS, raw endpoint distance, clean-update structure/numeric gates and decomposition residual/cross term |

Every squared-L2 value sums over latent coordinates. MSE is squared L2/d;
coordinate RMSE is L2/sqrt(d), not RMSE divided again. Exponential posteriors use
raw squared L2 before any display normalization. Source precision is recorded;
conversion from native predictions to epsilon occurs once, and already-canonical
cached epsilon is never converted again.

`target_id` identifies the paired target image hash; law atom IDs and payload
hashes are separate. Pair identity includes run/original_index/record_id/target_id.
Generated/Gaussian observations use a seed, forward observations a draw identity.
Stored k=0 is manuscript T; k=T−1 is the last prediction (manuscript 1). Saved
k+1 is the actual update output. Destination coefficients come from the saved
native scheduler contract, never from subtracting one from a training timestep.

The posterior, exact mean and maximum target-relative radius share one declared
law. The default gives equal mass to distinct cached target atoms before
selection; supplied manifest masses are validated and exact duplicates have
aggregated mass. The full training marginal and single-target training condition
are not inferred from checkpoint parameters, model-output centers or SSCD.

The original Proposition 5 variation is the integral of the full posterior-mean
norm difference along the matched displacement. Embedded quadrature error and
source sensitivity are estimates, not rigorous enclosures; estimated positive,
negative, unresolved, strict/nonstrict and zero-shift statuses remain distinct.
The signed lower bound is retained even when negative. Theorem 7 structural
inapplicability remains distinct from unavailable numbers or reconstruction
failure; no alternate clean projection replaces the saved endpoint.

## Historical scalar schemas (preserved reference)

# Theory measurement and scalar-schema dictionary

This dictionary describes the corrective implementation of the current working
pipeline. Its source of truth is `revised.pdf`, SHA-256
`fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`, Sections 2–3
and Appendix D. The pre-edit formula audit is `implementation_audit.md` at the
repository root.

The bundle schema is **3**, with formula version
**`cache-theory-3.0-proposition5`**. Branch helpers also record
`measurement_formula_version=manuscript-measurements-3.0`. The historical output
parent directory remains named `theory_v2`; the manifest, rather than that
folder name, identifies the schema. Old feedback tables without Equation-15
variation cannot be plotted as Proposition 5 by changing their labels.

## Figure inventory and manuscript mapping

Default output has four figure designs per requested scientific configuration.
Each available design has a PNG, PDF, and JSON caption/provenance sidecar. An
unavailable design produces an explicit report rather than a different metric
under the same filename.

| Design | Manuscript association | Saved x / y values | Scope |
|---|---|---|---|
| `initial_recovery` | Theorem 1, `thm:initial_conditional_recovery`; Eqs. 2, 7–10; Appendix D.1 | `unconditional_target_error_rmse` / `conditional_target_error_rmse`, at k=0 | Paired finite-noise initial target errors, one retained prompt–evaluation-seed per point. |
| `unconditional_center` | Lemma 2, `lem:initial_unconditional_baseline`; Eqs. 8–11; Appendix D.2 | Initial `unconditional_center_rmse` / empirical cumulative fraction | Held-out initial dispersion around a fixed declared model-output center; one observation per unique evaluation seed. |
| `posterior_feedback` | Proposition 5, `prop:matched_one_step_target_posterior_feedback`; Eqs. 14–15; Appendix D.5 | `candidate_condition_margin_rmse` / `candidate_log_probability_gain` | Matched posterior feedback under the declared candidate distribution. |
| `target_synchronization` | Lemma 6, `lem:target_specific_synchronization`; Appendix D.6 | Saved `snr` / branch-resolved target RMSE | Four prompt-balanced descriptive curves for two branches and two fixed SSCD cohorts. |
| `target_injection`, optional | Corollary 3, `cor:initial_cfg_amplification`; Appendix D.3 | `a_parallel` / `off_target`, at k=0 | Finite-noise target-direction coefficient and relative perpendicular component. |
| `terminal_terms`, optional and applicability-gated | Eq. 16 and Theorem 7, `thm:final_reproduction`; Appendix D.7 | `terminal_A_rmse` / `terminal_B_rmse` | The two Eq.16 magnitudes for observations satisfying the structural and numerical Eq.5 gate. |

Lemma 4 (`lem:matched_one_step_cfg_displacement`, Eqs. 4, 12–13; Appendix D.4)
remains numerical QA. Neither default nor diagnostic rendering produces
`matched_displacement.png/pdf`. The former operational-bound scatter
`terminal_reproduction.png/pdf` is also removed. Disabled diagnostic figures do
not disable their scalar measurements.

Axis labels name quantities in short phrases. Full formulas and limitations
belong in sidecars. The latest corrective specification supersedes the earlier
long-formula labels, seven-figure inventory, all-step baseline curve, and
suppression of essential compact counts. In particular, the initial labels are
“Initial unconditional target error (RMSE)” and “Initial conditional target error
(RMSE)”. Initial SNR and counts are shown compactly. “Equal target errors” labels
the diagonal: equal distances do not establish equal vectors.

## Observation identity, indexing, precision, and units

For update index k, the current state is `latents[:, k]`, the destination state
is `latents[:, k+1]`, and both saved branch predictions are from index k at
`timesteps[k]`. There are T predictions and T+1 states. No branch prediction is
invented at the final latent. The scheduler's training timestep, saved update
index, and actual destination noise level are separate fields; neither
`raw_timestep - 1` nor an assumed zero-noise endpoint supplies the destination.

Complete sample identity uses `run_id`, `original_index`, `record_id`,
`target_id`, and actual `seed`. Here `target_id`/`target_latent_sha256` identifies
the paired target tensor; `target_image_sha256` and `support_target_id` retain
image and candidate-bank membership identities. `candidate_target_atom` and
`target_group` identify the exact candidate atom. `terminal_sscd` and the
compatibility column `sscd` contain the same sample's saved terminal target SSCD.
No row-order or prompt-mean outcome join is used.

Evaluation seeds are `0..N-1`; reference seeds are `N..2N-1`. Frozen prompt
selection retains every evaluation seed, regardless of outcome or whether a
sufficient condition holds. Original dataset categories remain provenance.

Stored predictions are already canonical epsilon, including caches from native
v-prediction models. For b in {c,u}, with u the null branch:

```text
m_b,t = (x_t - sigma_t * epsilon_b,t) / alpha_t
Delta_t = m_c,t - m_u,t
m_g,t = m_u,t + g * Delta_t
E_b,t = ||m_b,t - x_star|| / sqrt(d)
G_t = ||Delta_t|| / sqrt(d)
E_joint,t = max(E_c,t, E_u,t)
```

The implementation promotes before arithmetic to float64. It reconstructs the
branch difference using `(sigma/alpha)*(epsilon_u-epsilon_c)` and separately
checks agreement with `m_c-m_u`. Promotion cannot recover inference/storage
rounding. `source_dtypes`, `source_dtype_epsilon`, `branch_status`, and
`numerical_budget_status` record that limitation.

All manuscript norms are raw Euclidean norms. For d latent coordinates,
`RMSE(v)=sqrt(sum(v_i^2)/d)=||v||/sqrt(d)`. A saved `_rmse` column already has this
normalization: the renderer does not divide it again. `_squared_l2` is a raw sum
of squares, and `_l2` is its square root. Gaussian likelihood exponents use raw
squared L2, never RMSE. SNR, projection ratios, log probabilities, and log odds
are dimensionless and are not divided by sqrt(d).

## Evidence and unavailable premises

| Evidence class | Meaning |
|---|---|
| `finite_noise_observation` | Direct behavior of the saved learned estimates or endpoint. |
| `model_center_diagnostic` | Geometry relative to an explicitly estimated model-output center. |
| `candidate_distribution_diagnostic` | Analytic reference/posterior calculation under a declared finite law K. |
| `identified_training_reference` | Reserved for independently identified actual training-law inputs; ordinary caches do not provide these. |
| `algebraic_qa` | Reconstruction, identities, analytic bounds, and implementation checks subject to numerical precision. |
| `unavailable` / `not_applicable` | Missing information or a mathematical condition that does not apply, with a reason. |

The compatibility field `effective_target_residual_evidence=exact_cache_algebra`
is explicitly marked as a legacy alias. Its canonical counterpart is
`effective_target_identity_evidence=algebraic_qa`.

The `paper_*` population fields remain unavailable: pair-specific forward loss,
training mean, full-training posterior and unconditional Bayes reference,
full-support radius, and the corresponding Proposition-5/Lemma-6/Theorem-7
training-law bounds. Under the single-target idealization, conditional target
error equals conditional reference error. A paired caption/image cache does not
verify that idealization. Candidate calculations implement a declared diagnostic
law rather than supplying the missing training-law premises.

## Branch, paired, and initial measurements

| Canonical fields | Definition / interpretation |
|---|---|
| `conditional_target_error_{l2,squared_l2,rmse}` | Norm of `m_c-x_star`. |
| `unconditional_target_error_{l2,squared_l2,rmse}` | Norm of `m_u-x_star`; **not** the manuscript's unconditional reference error. |
| `guided_target_error_{l2,squared_l2,rmse}` | Norm of `m_g-x_star`. |
| `branch_gap_{l2,squared_l2,rmse}` | Norm of Delta. |
| `joint_target_error_{l2,squared_l2,rmse}` | Maximum of the two same-sample branch target errors. |
| `paired_target_error_improvement_{l2,rmse}` | Unconditional minus conditional target error; negative values are retained. |
| `relative_target_error_improvement` | Paired RMSE improvement divided by unconditional target RMSE. Zero denominator gives an undefined value and an explicit denominator flag. |
| `state_*`, `state_error_*`, `target_*` | Norms of the saved current state, its target residual, and the target. |
| `branch_difference_identity_rmse` | `||(m_c-m_u)-Delta||/sqrt(d)`. |
| `effective_target_noise_residual_mse` | `||epsilon_c-(x_t-alpha*x_star)/sigma||^2/d`; reverse-cache algebra, not forward population loss. |
| `effective_target_identity_residual` | Conditional target MSE minus effective target-noise residual MSE divided by SNR. |
| `clean_estimate_rounding_sensitivity_rmse` | `32*source_eps*(state_rmse+sigma*max(branch_epsilon_rmse))/alpha`; a screening sensitivity, not a certified inference-error bound. |

`e_u_*`, `e_c_*`, `e_g_*`, `joint_rmse`, and
`conditional_error_reduction_rmse` remain internal compatibility aliases.
Designated axes use the explicit target-error names. Equal wrong branches can
have zero gap while both target errors and the joint error are large.

Initial recovery uses k=0 only. The summary retains absolute errors, paired
improvement, fraction improved, finite denominators, and descriptive quantiles,
including per-prompt summaries. Relative improvement is not accurate reproduction
without a predeclared target tolerance. Actual `initial_snr`, `initial_noise_scale`,
and initialization provenance are recorded; later reverse states are not treated
as independent Gaussian probes.

### Appendix D.1, Equation 31

For `Q=N(alpha_T*x_star,sigma_T^2 I)` and `P=N(0,I)`, the logged bound is

```text
TV(P,Q) <= 0.5*sqrt(alpha_T^2*||x_star||^2
                    + d*(sigma_T^2 - 1 - log(sigma_T^2)))
```

`initial_transfer_bound_unclipped` stores this expression;
`initial_transfer_bound_clipped` stores its minimum with one.
`initial_forward_to_initial_gaussian_kl` stores `KL(Q||P)` and
`initial_transfer_bound_status` records applicability. The reducer requires the
preserved `sampler_contract_version==2` initialization contract, standard initial
scale (absolute tolerance 1e-12), valid positive noise, and a VP coefficient check
(absolute tolerance 2e-6). Version 2 records seeded CPU float32 standard Gaussian
initialization multiplied by the saved scheduler scale. The quantity concerns
intended continuous Gaussian laws, not the exact law of quantized tensors, and
is an upper bound rather than measured total variation.

The separately retained `initial_gaussian_kl` uses the opposite direction,
`KL(N(0,s0^2 I)||N(alpha*x_star,sigma^2 I))`. Its direction is explicitly named.
Neither KL nor the transfer bound supplies the unavailable pair-specific forward
loss. No fitted zero-SNR endpoint or convergence claim is added.

## Fixed reference center and initial ECDF

The default `mu_ref` is the average initial unconditional clean estimate from
one canonical complete reference record, one observation per actual reference
seed. The record is chosen by minimum `(source_row_number, original_index)`
independently of frozen selection and outcomes. The center is fixed across all
prompts, targets, seeds, and timesteps. Evaluation seeds never fit this center.

Repeated initial states must agree exactly. Repeated canonical-epsilon outputs
must have maximum absolute discrepancy no larger than
`8*source_dtype_eps*max(1,maxabs(canonical_epsilon))`. Material disagreement raises
an error; it is not averaged away. This is a declared source-precision screening
policy. Reference and evaluation discrepancies and thresholds are recorded.

`center_metadata.json` records `center_kind`, vector hash, source run/schedule
hashes, seed IDs, unique observation count, norm, initial prediction dtype,
canonical record/rule, repeated-source checks, and descriptive dispersion
quantiles. `mc_standard_error_rmse` is a coordinate-averaged standard-error
summary, not a confidence radius. Reference seeds are disjoint from evaluation,
but also inform the existing frozen prompt selection.

Three quantities remain distinct:

- `mu_ref`: estimated initial model-output center.
- `mu_K`: weighted mean of the declared candidate atoms, saved independently.
- `mu=E_D[X]`: manuscript training-distribution mean, unavailable here.

Explicit `zero` and compatible already-saved `cached-baseline` center choices are
allowed and recorded. No legacy baseline inference is invoked. Center-distance
fields use the selected center, with `model_center_diagnostic` evidence.

The baseline plot contains only initial `unconditional_center_rmse`, deduplicated
by run and evaluation seed after repeated-input consistency checks. With N=20
there are 20 independent seed inputs, not 20 times the prompt count. The ECDF is
`F_n(a)=count(distance<=a)/n`. No SSCD colorbar appears because a unique
prompt-specific endpoint does not belong to this prompt-independent observation.
Per-step center distances remain saved diagnostics. Tight held-out model-output
dispersion does not establish data-mean agreement.

## Fixed finite candidate law K

`build_support` freezes all valid complete target records from the declared
**experiment source run** before retained-prompt filtering. Model/revision,
VAE/revision, latent shape, preprocessing, and target-latent definition must
match. Target files lacking complete record provenance are excluded with an
audit reason; malformed or stale complete targets are failures. Other compatible
run directories do not silently enlarge the bank.

Exactly identical latent values are aggregated into one atom after float64
promotion; signed zeros are canonicalized. Image and record aliases remain
mapped to the atom. Conflicting latents for one image identity fail validation.
Perceptually similar but nonidentical latents remain distinct. Default mass is
uniform over exact distinct atoms, not proportional to duplicate file records.
The support helper can accept explicit declared positive weights, but the default
reducer does not estimate training frequencies.

`support_metadata.json`, `source_records.parquet`, `support.pt`, and
`candidate_mean.pt` retain source and tensor hashes, source-run policy, exact atom
weights, aliases, atom IDs, membership, candidate mean, and missing-source reasons.
The bank and weights are fixed across prompts, seeds, steps, and both endpoints.
No nearest-competitor pruning, final-sample targets, latent rescaling, or tuned
likelihood temperature is used.

```text
w_j,t(z) proportional to pi_j*exp(-||z-alpha_t*u_j||^2/(2*sigma_t^2))
bar_x_t^K(z,u) = sum_j w_j,t(z)*u_j
p_t^K(z) = posterior mass of the fixed exact target atom
R_K = max_j ||u_j-x_star||
```

`target_log_probability`, `target_log_complement`, and `target_log_odds` describe
the current state under this law. `candidate_unconditional_reference_error_l2`
is `||m_u-bar_x_t^K(x_t,u)||`; `candidate_radius_l2` is R_K. Existing
`network_to_surrogate_rmse` and `support_radius_rmse` aliases have the same
candidate qualification. Missing target membership is unavailable. A single-atom
law has posterior one and radius zero, so feedback is inapplicable; its reference
and terminal terms can still be computed.

## Matched update and numerical QA

For an applicable scheduler update,

```text
x_next = a_t + kappa_t*m_g,t
x_cf   = a_t + kappa_t*m_u,t
h_t    = g*kappa_t*Delta_t
```

The adapter derives coefficients from the saved scheduler contract. Its internal
`A` is the current-state scalar coefficient and `B`/`kappa` is the raw-clean
coefficient; neither is the Eq.16 terminal A/B defined below. For deterministic,
unclipped DDIM, `A=sigma_next/sigma_t` and
`kappa=alpha_next-alpha_t*sigma_next/sigma_t`. DDPM uses its own posterior-mean
coefficients and recorded state-independent variance convention. Native prediction
conversion, clipping, thresholding, destination settings, and scheduler version
are checked. `kappa>0` is required for the manuscript identity.

With the affine contract established, the implementation reconstructs
`x_cf=x_next-h_t`. The saved state/history and any recovered additive stochastic
innovation are shared. This is not a separately evolved unconditional trajectory.
Non-affine drift can retain a separately named matched-drift diagnostic, but its
manuscript identity and Proposition-5 comparison remain inapplicable.

`matched_construction_vector_residual_rmse` and its compatibility alias
`matched_displacement_identity_rmse` measure
`||(x_next-x_cf)-g*kappa*Delta||/sqrt(d)`. Agreement of norms is insufficient.
Subtraction construction is explicitly marked as **not independent
verification**. `independent_replay_rmse` compares a deterministic saved endpoint
with the analytic drift without fitting coefficients to the endpoint. Stochastic
replay is unavailable when an independent saved noise realization is absent;
recovering that realization from the endpoint does not turn it into an independent
replay test. Unit tests separately compare native scheduler updates with analytic
matched reconstruction under shared supplied randomness.

## Proposition 5 under K: variation, condition, and outcome

Every eligible nonterminal transition uses the current reference at the current
noise level and both matched endpoints at the actual destination noise level:

```text
e_c,t   = ||m_c,t-x_star||
e_u,t^K = ||m_u,t-bar_x_t^K(x_t,u)||
V_t^K   = integral_0^1 ||bar_x_next^K(x_cf+s*h_t,u)
                           -bar_x_t^K(x_t,u)|| ds
M_t^K   = ||Delta_t||-e_c,t-e_u,t^K-V_t^K
H_t^K   = log p_next^K(x_next)-log p_next^K(x_cf)
```

V integrates the **norm of each reference difference**, not the norm of an
integrated vector, one endpoint discrepancy, or a competitor margin. Segment
nodes evaluate only the analytic candidate posterior; no learned branch is
called at new inputs. Positive current/destination noise, valid target membership, at least two
atoms, affine positive kappa, and positive finite guidance are required for the
analytic calculation. The manuscript assumes g>1;
`candidate_manuscript_guidance_status` separately marks 0<g<=1 as an analytic
diagnostic outside that manuscript domain. Initial/branch measurements remain
available when feedback is not.

| Feedback field | Quantity / unit |
|---|---|
| `candidate_conditional_reference_error_l2` | e_c under the explicitly stated single-target idealization; raw latent L2. |
| `candidate_unconditional_reference_error_l2` | e_u^K; raw latent L2. |
| `candidate_variation_l2` | V estimate; raw latent L2. |
| `candidate_variation_error_l2` | Embedded quadrature error estimate for V; raw latent L2. |
| `candidate_condition_margin_l2`, `_rmse` | M and M/sqrt(d); signed. |
| `candidate_condition_numerical_uncertainty_l2` | Sum of variation error estimate, Gram allowance, and source sensitivity. |
| `candidate_{matched,guided}_{log_probability,log_complement,log_odds}` | Destination endpoint quantities from the same candidate law. |
| `candidate_log_probability_gain` | H; plotted outcome. |
| `candidate_log_odds_gain` | Diagnostic endpoint log-odds difference; not the plotted substitute for H. |
| `candidate_gain_status`, `_saturated`, `_underflow` | Sign interpretation and rounding/saturation flags. |
| `candidate_quadrature_evaluations`, `_refinements`, `_budget_exhausted` | Per-query integration work and budget outcome. |
| `candidate_integral_status`, `candidate_integral_qa_status`, `feedback_status` | Convergence estimate, identity QA, and comparison status. |

### Numerical policy and sign interpretation

The default policy is adaptive embedded **Gauss–Kronrod 15 / Gauss 7**, absolute
V tolerance **1e-5 raw latent L2**, relative tolerance **1e-5**, and integrated
identity absolute tolerance **1e-7 log-probability units**. The evaluation budget
is **4095 nodes per query**; the initial uniform partition has two intervals.
Dominant affine-logit crossings and neighboring slope-width points seed additional
intervals to resolve sharp interior transitions. All atoms still contribute to
all softmax evaluations. The policy and effective tolerances are saved in the
analysis identity/manifest and scalar fields.

After removing common likelihood terms, each segment logit is affine in s.
Intercepts and slopes are precomputed; posterior nodes are evaluated in bounded
chunks (default 256). A centered support Gram matrix is cached per support/device
for posterior-mean distances. The roundoff screening allowance on a squared norm
is `128*eps64*K*maxabs(Gram)*||weight_difference||_1^2`. Negative squared norms are
clamped only within that allowance; larger negatives become unresolved numerical
failures. Direct-vector and Gram reductions are compared by tests.

The source term uses the recorded clean-estimate sensitivity converted to raw L2.
These allowances are not certified enclosures of neural inference or floating
point BLAS error. Embedded quadrature agreement is likewise an estimated error
check. The optional analytic Lipschitz enclosure in the task is not used to claim
conservative or formal sign certification.

`candidate_condition_status` is `estimated_met` only when M exceeds the summed
uncertainty and integration/identity QA passes, `estimated_not_met` when M is
negative beyond uncertainty, and `numerically_unresolved` otherwise. Exhausted
budgets and near-zero margins are not turned into zero variation or favorable
conditions. Missing or mathematically inapplicable comparisons have separate
eligibility/status fields.

Float64 target-relative logits and stable log-sum-exp/log-add-exp preserve endpoint
log probabilities and log complements. Gain sign is determined by log-odds ordering
with a fixed magnitude-scaled float64 allowance. Positive/negative ordering can
remain resolved when H rounds to zero because probabilities saturate near one.
The stored H remains zero: no artificial positive number is substituted.
`candidate_gain_saturated` flags either endpoint probability rounding to one;
`candidate_gain_underflow` flags H=0 with resolved odds ordering. `zero` is reserved
for identical endpoints; insufficiently resolved nonidentical endpoints remain
`numerically_unresolved`.

Independent endpoint H is compared with the Appendix-D.5 identity

```text
H_t^K = alpha_next*g*kappa_t/sigma_next^2
          * integral_0^1 Delta_t dot [x_star-bar_x_next^K(x_cf+s*h_t,u)] ds
```

`candidate_integrated_log_probability_gain`,
`candidate_integral_identity_residual`, and its error estimate record this QA.
`candidate_proof_lower_bound` is
`alpha_next*g*kappa/sigma_next^2 * ||Delta|| * M`; the corresponding slack is H
minus this value. Reconstruction residuals are also checked. A numerical
inconsistency is unresolved QA, not an empirical refutation of the exact algebra
on the same candidate law.

The feedback figure uses all finite eligible points, including negative gains and
unmet conditions. Its x-axis is linear M/sqrt(d); y is H on symmetric-log scale,
base 10 with linear-region parameter **1e-3**. There is no equality diagonal
because the axes have different units. Zero lines are labeled. Numerically
unresolved points use hollow neutral markers; counts retain unplottable rows.
Coverage uses all eligible observations, including unresolved ones, in its
denominator. Zero coverage is distinct from unavailable comparison or failure.

## Synchronization, summaries, and optional tolerance

The two fixed descriptive outcome groups are `SSCD > 0.75` and `SSCD <= 0.75`.
They do not change eligibility; the latter is not called nonmemorized or normal.
Conditional curves are solid, unconditional curves dashed. The groups use fixed
viridis colors (0.8 and 0.2) with categorical legends, not a continuous SSCD
colorbar. Empty groups remain empty. Saved SNR follows denoising update order on
a logarithmic axis, with the actual initial SNR marked. No t=0 branch,
zero-SNR extrapolation, smoothing, or monotonic constraint is imposed.

For each step, branch, and outcome group, each represented complete prompt
identity receives equal total weight, divided across its finite member seed
observations. Quantiles use the inverse weighted empirical CDF: the first sorted
value whose cumulative normalized weight reaches the requested probability.
The four medians and interquartile bands are descriptive, not confidence
intervals. Prompt counts use full prompt identities and seed counts use full
sample identities, not repeated timestep rows.

| Saved table under `summaries/` | Content |
|---|---|
| `synchronization.parquet` | Four branch/cohort median and interquartile curves, SNR, prompt/sample/observation counts. |
| `paired_synchronization.parquet` | Per-step paired target errors, joint maximum, branch gap, and paired difference, with prompt-balanced quantiles and finite counts. |
| `phases.parquet` | Fixed early/intermediate/late phase summaries for both cohorts, including empty phases/groups. |
| `initial_by_prompt.parquet` | Both initial absolute errors, paired/relative improvement quantiles, improvement fraction and denominators. |
| `feedback_by_step.parquet` | Source/destination SNR, total/eligible counts, condition coverage/components, sign counts overall and when met, endpoint/gain quantiles, saturation/underflow, unresolved counts, and negative-gain high-SSCD count. |
| `feedback_by_prompt.parquet` | Corresponding feedback summaries by complete prompt identity. |
| `tolerance_entries.parquet`, only when configured | Same-sample joint-error first and sustained observed-grid entries, non-entry and censoring. |

The phase rule is fixed `floor(3*k/T)`: early k/T in [0,1/3), intermediate in
[1/3,2/3), late in [2/3,1). Phase summaries give each represented prompt equal
total weight across its finite phase observations. No phase boundary is chosen
from the observed curves. Marginally small branch medians do not establish small
joint error on the same samples; the paired table answers that separate question.
Late clean estimates may approach the current state by construction, so early
and intermediate summaries are retained.

`target_error_tolerance`, when explicitly supplied, is a nonnegative **raw latent
L2** value tau. The RMSE comparison is tau/sqrt(d). It is never inferred from
SSCD 0.75, quantiles, or favorable observations. Entry tables record first entry,
first suffix remaining inside on the observed grid, `not_reached`, missing
observations, last saved prediction, and right censoring. These are observed-grid
summaries, not continuous-time guarantees.

## Optional injection diagnostics

For the selected fixed center, define `v=x_star-center`, `J=g*Delta_T`.

```text
a_parallel = dot(J,v)/||v||^2
off_target = r_perp = ||J-a_parallel*v||/||v||
injection_relative_mismatch = sqrt((a_parallel-g)^2+r_perp^2)/g
```

The mismatch requires positive finite g. The perpendicular vector is reduced
directly, avoiding subtraction of nearly equal squared norms. Projection is
undefined when `||v|| <= 8*source_eps*max(1,||target||,||center||)`; the direction
tolerance and status are saved. Zero injection also leaves cosine undefined.
`injection_perpendicular_rmse` and `injection_total_residual_rmse` retain absolute
perpendicular and limiting-vector residual magnitudes.

The optional figure uses short coefficient/perpendicular labels and marks `(g,0)`
and zero target projection. Positive alignment alone is not agreement with the
limiting vector. The center source and finite-noise limitation are in its caption.

## Terminal structure, cancellation, and distinct bounds

`terminal_l2`/`terminal_rmse` and `terminal_endpoint_error_*` always measure the
actual final saved latent. The strict Eq.5 gate requires the final update, affine
raw-clean dependence, **A=0**, **kappa=1**, destination **alpha=1**, destination
**sigma=0**, and **exactly zero additive noise standard deviation**. It then
checks the endpoint-minus-guided-clean residual against a separately reported
source-precision screening tolerance. A small residual cannot override a failed
structural condition.

`terminal_clean_structural`, `terminal_clean_applicable`,
`terminal_clean_status`, `terminal_clean_structural_reason`, and numerical
residual/tolerance fields distinguish these checks. In the four preserved default
configurations inspected for this correction, no terminal update satisfies exact
Eq.5: SDv1/SDv2/RealVis DDIM have nonzero destination sigma and coefficients near
A=0.706290, kappa=0.293887; SDv1 DDPM has A=0, kappa=1 but a nonzero final variance
floor, noise standard deviation approximately 1e-10. Consequently these settings
have no applicable `terminal_terms` figure. Settings are not changed to pass the
gate. This is a scheduler-contract finding, not a claim that newly recomputed
measurements empirically validate a theorem.

For the last stored prediction, Eq.16 terms are always retained as diagnostics:

```text
first_vector = m_c,1-x_star
second_vector = (g-1)*Delta_1
m_g,1-x_star = first_vector+second_vector
A = ||first_vector||/sqrt(d)
B = (g-1)*||Delta_1||/sqrt(d)
```

`terminal_A_*`, `terminal_B_*`, `terminal_cross_inner_product`,
`terminal_cross_term_squared_l2`, `terminal_cross_term_per_dimension`,
`terminal_terms_cosine`, `terminal_clean_error_*`, and
`terminal_eq16_*` record magnitudes, cancellation, actual guided-clean error,
and vector/squared-norm identities. `terminal_two_term_bound_*` records A+B in
corresponding units for g>1; its slack and ratio are separate. Zero-error ratios
are undefined and have explicit Boolean flags. A=B can be large while opposite
vectors cancel to a small actual error.

For K, the separate candidate Theorem-7 expression is

```text
B_thm^K = g*e_c,1 + (g-1)*e_u,1^K + (g-1)*R_K*(1-p_1^K)
```

`candidate_terminal_conditional_term_*`,
`candidate_terminal_unconditional_term_*`, and
`candidate_terminal_concentration_term_*` store each summand;
`candidate_terminal_bound_*` stores their sum. Raw L2 and RMSE are both saved.
The complement uses `exp(target_log_complement)`, avoiding `1-rounded_p`.
Reference radius, non-target mass, clean-estimate slack/ratio, endpoint
slack/ratio, undefined-zero-error flags, candidate evidence, and endpoint
applicability are distinct fields. A missing reference does not become zero.
Only `candidate_terminal_bound_applies_to_endpoint` permits an endpoint
interpretation; otherwise the terms concern the last guided clean estimate.
This is a candidate-radius expression, not the unknown full-training bound.

`scheduler_rho_*` measures `rho=x0-m_g,1`. It is an observed scheduler discrepancy,
not an ex-ante error allowance. Adding it to an endpoint-derived bound does not
create a prospective reproduction certificate.

The legacy `operational_bound_rmse` remains separately logged, with components
for scheduler current-state error, unconditional/conditional target errors,
`|A+B-1|*target_rmse`, and prospective known-variance noise. It is neither A+B nor
the candidate or actual-training Theorem-7 expression, and has no designated
figure. Operational noise uses its recorded per-update delta/coverage scope;
observed innovations do not supply that prospective term.

The optional terminal-terms plot admits only `terminal_clean_applicable` rows.
A sufficient region `A+B <= tau/sqrt(d)` appears only for an explicitly supplied
raw-L2 tolerance. Otherwise there is no tolerance line. The figure concerns
Eq.16 magnitudes and cancellation, not verification of training-reference
premises.

## Storage, rendering, and failure contracts

`manifest.json` records active code revision, reducer source hashes,
formula/schema identity, configuration, source metadata inventories,
generation/reference/schedule/selection/SSCD identities, source dtypes,
numerical and integration policies, expected counts, and worker/device execution.
Target and source-record hashes are retained in the scalar/provenance tables.
The scalar bundle includes `initial_metrics.parquet`, `endpoint_metrics.parquet`,
per-record `trajectory_metrics/`, six `plotdata/` tables, summaries, center/support
metadata, the manuscript registry, `summary.json`, `failed.csv`, and
`applicability_report.json`. Optional tensor artifacts support analysis resume;
they are not renderer inputs.

All branch, feedback, and terminal measurements share one record-streaming
reducer. Query/time/node/candidate chunks bound device memory. Generation's
visible-device resolution, round-robin process convention, and one aggregate
progress bar are reused; source validation uses CPU readers where appropriate.
No reconstructed clean-estimate trajectory or query×step×candidate×latent array
is written. Per-record scalar shards are resumable, and completion is atomic.
Computational failures do not silently publish a complete full-mode analysis.
Unavailable premises and zero condition coverage are distinct from failed
computation. A failed recomputation does not replace an existing complete result.

Normal analysis publishes scalars before invoking the same renderer used by
`--plot`. Plot-only reads validated saved scalar/provenance artifacts; it may
aggregate saved scalar rows but does not read raw `.pt` tensors, refit centers,
rebuild support, run quadrature, or load/evaluate any model. Schema/version or
missing-V failures give an analysis-only recomputation command. A style change
has a separate figure hash and does not invalidate numerical measurements.

Figure manifests and sidecars record formulas, captions, axis/scales, reference
labels, counts, exclusions, scientific hashes, and style identity. Managed
obsolete pipeline-owned images are archived; raw logs and unrelated files are
preserved. Individual SSCD-colored figures use fixed [0,1] viridis and label the
colorbar **SSCD**. The ECDF and grouped synchronization figures have no misleading
continuous SSCD colorbar. Typography is STIXGeneral/STIX math, nominal 4×4 inches,
150-DPI PNG plus PDF, 15-point axis labels, 12-point ticks, compact legends, light
grids, tight output, and rasterized dense artists. Reference lines identify their
meaning; finite negative data and numerical uncertainty remain visible.

## Precision refinement (unexecuted revision)

See [numerical_refinement_plan.md](numerical_refinement_plan.md) for exact auxiliary bounds and scopes. `numerical_log_probability_gain` retains H; `numerical_log_odds_gain` is stable G and must never replace H's magnitude. `numerical_affine_*` refers to the reconstructed affine endpoint, while the primary gain refers to the recorded endpoint. `condition_sign_status` may be negative while `condition_value_status` remains unavailable. Directed bounds and legacy embedded quadrature estimates have separate fields. `fixed_cache_gain_sign`, `source_robust_gain_sign`, and `reference_model_identification` describe distinct uncertainty layers. Budget exhaustion is unknown mass, not a zero gain or variation.

The Corollary-3 premise is `(ec+bu)/||x_star-mu_K||`; its full error decomposition uses the saved vector inner product. Lemma-6 Q is the paired maximum before aggregation. The terminal looseness is `(g-1)*(S_raw-D_raw)`, decomposed into the radius-tail slack and the reference-triangle slack. All of these are saved scalar audits; none changes the empirical law or replaces the manuscript statement.
