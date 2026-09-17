# Candidate discovery formula mapping

The active contract is `projected-gap-error-1`: mathcal{E}_t=u^T(Delta-bar{Delta}) is signed and mathcal{V} is the positive part after the unchanged signed unit-gap projection integral. Exactly zero gap extends u=0, mathcal{E}_t=0 and mathcal{V}=0 by convention without claiming a unit direction. The PDF formula table below is historical attribution; the active candidate margins use the revised formulas documented below. Run `bash run_all.sh --recompute-experiments`; historical gap-error norms cannot be relabeled as signed errors.

## Audited source and implementation

| Item | Audit result |
|---|---|
| Authoritative file | Repository `revised.pdf` |
| Actual SHA-256 | `fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f` |
| Requested SHA-256 | Identical; no PDF revision difference detected |
| Inspected equations | Sections 2–3, Appendix D; especially PDF pp. 5–6 and 18–19 for Equations 15, 63, 64, 68, 69, 70, 72 |
| Matching LaTeX source | Not present in the repository, workspace source search, or available attachment tree; no `.tex` or matching source archive was found |
| LaTeX label provenance | Labels below come from the supplied candidate brief and agree with the corresponding numbered PDF results. Their literal definitions cannot be independently verified against an unavailable `.tex` file. |
| Baseline implementation | Active `utils/experiments/theory/{metrics,scheduler_adapter,support,feedback,reduce,summaries}.py`; existing schema-3 scalar products remain intact |
| Additional pure measurements | `candidate_metrics.py`, formula version `candidate-measurements-1.0`; inputs are tensors supplied by the shared reducer, never raw-file paths |
| Additional summaries | `candidate_summaries.py`; saved DataFrames only, with no model loading or posterior evaluation |
| Endpoint and integration calculations | Separate candidate feedback helpers, called by the candidate reducer; endpoint products do not depend on the integrated-margin stage |

| Family | PDF result | Supplied source label |
|---|---|---|
| IR | Theorem 1 | `thm:initial_conditional_recovery` |
| UB | Lemma 2 | `lem:initial_unconditional_baseline` |
| IA | Corollary 3 | `cor:initial_cfg_amplification` |
| MD | Lemma 4 | `lem:matched_one_step_cfg_displacement` |
| PF | Proposition 5 | `prop:matched_one_step_target_posterior_feedback` |
| TS | Lemma 6 | `lem:target_specific_synchronization` |
| TR | Theorem 7 | `thm:final_reproduction` |

## Shared measurement contracts

The stored update index is `k`: canonical epsilon predictions `[:,k]` apply to
`latents[:,k]`, and the update produces `latents[:,k+1]`. The last cached state
has no additional prediction. Current and destination noise coefficients come
from the saved scheduler contract, not arithmetic on timestep labels.

All sensitive arithmetic promotes cached values before operations:

- `m_u=(x-sigma*epsilon_u)/alpha`, `m_c=(x-sigma*epsilon_c)/alpha`.
- `Delta=m_c-m_u`, `m_g=m_u+g*Delta`, `D=||Delta||`.
- `E_b=||m_b-x_star||/sqrt(d)` and `E_joint=max(E_c,E_u)` on the same sample row.
- Raw L2 equals RMSE times `sqrt(d)`. Dividing an already computed RMSE by
  `sqrt(d)` again is incorrect. Dot products have raw squared-L2 units; dividing
  a dot product by `d` gives its per-coordinate value.
- `mu_ref` is the fixed initial learned unconditional center from distinct
  reference seeds `N..2N-1`; initial evaluation observations use seeds `0..N-1`.
  `mu_K=sum_j pi_j*u_j` is the separately identified candidate-law mean.
- `unconditional_target_error_*` measures distance to `x_star`;
  `candidate_current_reference_error_u_l2` measures distance to the current
  candidate posterior mean. These are separate quantities.
- Sample joins use `run_id, original_index, record_id, target_id, seed`.
  Individual rows retain their same-seed terminal SSCD. No prompt-mean score is
  substituted for the individual terminal outcome.

The fixed candidate law has positive declared masses `pi_j` and exact distinct
atoms. By default each compatible complete cached source record contributes equal
mass before selection; an atom occurring n times among N source records has prior
n/N. Exact latent duplicates have their masses summed before evaluation; aliases
remain traceable. Explicit manifest weights retain their declared values after
duplicate aggregation. Similar images are not merged. The candidate bank and masses are
unchanged across endpoints, doses, controls, and integrals. Candidate quantities
carry `candidate_` columns and superscript `K` in formula sidecars. The declared
candidate law is not identified as the pretrained model's training law.

The matched adapter verifies `x_next=a+kappa*m_g`, `kappa>0`, with a shared
remaining contribution, history, and realized noise. Under this verified affine
contract, `h=g*kappa*Delta` and `x_cf=x_next-h`. Construction residuals and
independent scheduler replay residuals retain different names and provenance.
The DDPM and DDIM coefficients are independently taken from their saved native
scheduler contracts.

## Verified posterior formulas and new analysis derivations

For the fixed finite Gaussian law,

`w_j(z;t) proportional to pi_j*exp(-||z-alpha_t*u_j||^2/(2*sigma_t^2))`,
`bar_x_t^K(z)=sum_j w_j(z;t)*u_j`, and `p_t^K(z)` is the exact target atom's mass.
Gaussian exponents use raw squared L2, without RMSE normalization or an extra
latent-dimension temperature.

| PDF equation | Supplied label | Verified expression and measurement use |
|---|---|---|
| 15 | `eq:cross_step_reference_variation` | `V=integral_0^1 ||bar_x_next(z(s))-bar_x_current(x)|| ds`, with `z(s)=x_cf+s*h` |
| 63 | `eq:proof_target_posterior_gradient` | `gradient_z log p_t(z)=(alpha_t/sigma_t^2)*(x_star-bar_x_t(z))` |
| 64 | `eq:proof_branch_gap_reference_decomposition` | `Delta-(x_star-bar_x_current)=r_c-r_u`, where `r_c=m_c-x_star`, `r_u=m_u-bar_x_current` |
| 68 | `eq:proof_target_posterior_segment_derivative` | `d log p_next(z(s))/ds=(alpha_next*g*kappa/sigma_next^2)*Delta dot (x_star-bar_x_next(z(s)))` |
| 69 | `eq:proof_integrated_target_posterior_change` | `H=log p_next(x_next)-log p_next(x_cf)=(alpha_next*g*kappa/sigma_next^2)*integral Delta dot (x_star-bar_x_next(z(s))) ds` |
| 70 | `eq:proof_alignment_decomposition` | The integrand is `D^2+Delta dot (r_u-r_c)-Delta dot d(s)`, where `d(s)=bar_x_next(z(s))-bar_x_current` |
| 72 | `eq:proof_integrated_alignment_lower_bound` | The integrated alignment is at least `D*(D-||r_c||-||r_u||-V)` |

For `D>0`, put `u=Delta/D`, `S=u dot (r_u-r_c)`,
`V_parallel=integral |u dot d(s)| ds`, and `W=integral u dot d(s) ds`.
The following are explicitly new analysis refinements derived from the verified
identities. Under `projected-gap-error-1`, revised `M0` and `M1` use mathcal{V}=[W]_+, with the positive part after the signed integral. The old norm integral V_norm and absolute-projection integral V_parallel remain separate diagnostics:

| Name | Formula | Saved prefix |
|---|---|---|
| Refined gap-error margin `M0` | `D-mathcal{E}_t-[W]_+`, where `mathcal{E}_t=u^T(r_c-r_u)=-S` | `candidate_margin_original` |
| Signed projected-error alias `M1` | `D-mathcal{E}_t-[W]_+` | `candidate_margin_combined` |
| Signed-error margin `M2` | `D+S-V_norm` | `candidate_margin_signed_error` |
| Projected-variation margin `M3` | `D+S-V_parallel` | `candidate_margin_projected_variation` |
| Exact directional average `A` | `D+S-W` | `candidate_exact_directional_average` |

Since mathcal{E}_t=-S, the valid chain is `M2<=M3<=M0=M1<=A`: the norm and absolute-projection diagnostic integrals exceed [W]_+. The opposite M1<=M2 ordering is not asserted. Remaining diagnostic gaps are `V_norm-V_parallel` and `V_parallel-W`; `A-M0=[W]_+-W>=0`. Exactly zero gap extends u=0, E_parallel=0 and mathcal{V}=0 and never claims a unit direction. Equality of M0 and M1 is definitional, not an independent measured agreement. Equation 69 becomes
`H=(alpha_next*g*kappa*D/sigma_next^2)*A`. Thus `A` characterizes the measured
gain's sign and is not an independent predictive certificate. Numerical
condition signs retain their integration uncertainty and status. Common-mode residuals `r_c=r_u` give zero gap error even when individual branch errors are large; the shared target error Q is therefore not bounded by the refined S.

The endpoint test is another new finite-law derivation. With
`beta=alpha_next/sigma_next^2`, fixed non-target atoms satisfy

`b_j=log(pi_j/pi_star)+beta*x_cf dot (u_j-x_star)`
`    -alpha_next^2*(||u_j||^2-||x_star||^2)/(2*sigma_next^2)`,

`q_j=beta*h dot (x_star-u_j)` and
`Lambda(s)=-LSE_{j!=star}(b_j-s*q_j)`.

The implementation can evaluate the algebraically equivalent target-centered
form of `b_j` to avoid subtraction of large query norms. With
`nu_j(s)=softmax_{j!=star}(b_j-s*q_j)`,

- `Lambda'(s)=sum nu_j(s)*q_j`;
- `Lambda''(s)=-Var_nu(s)(q_j)<=0`;
- `C1=Lambda'(1) <= G_odds=Lambda(1)-Lambda(0) <= C0=Lambda'(0)`.

This is concavity of the **target log odds along the fixed segment**, not
log-concavity of the mixture density. `C1>0` certifies increasing evidence
throughout; `C0<0` certifies decreasing evidence throughout. `C0>0>C1` marks an
interior peak whose net endpoint gain still must be measured. Flat requires a
structurally constant or numerically resolved zero-variation slope. Ambiguous
signs remain unresolved.

Stable endpoints save `log_p=-softplus(-Lambda)` and
`log_one_minus_p=-softplus(Lambda)` independently. Non-target weights come from
a direct non-target log-sum-exp, never `1-rounded_target_probability`.
`H` and `G_odds` have the same resolved sign. Underflow of the represented
magnitude of `H` does not become arithmetic zero. For a displacement whose
observed branch gap is resolved in float64 arithmetic, normalized gains divide
by `beta*||h||*sqrt(d)`; this positive normalization does not change their signs.
Zero displacement gives exact zero matched gain and undefined direction-normalized
quantities. A branch gap at the float64 subtraction-resolution scale remains
arithmetically unresolved rather than receiving an artificial direction.

Arithmetic resolution and source-prediction confidence are separate contracts.
The float64 subtraction screen is based on the observed clean-estimate norms,
`128*epsilon_float64*(||m_c||+||m_u||)`. The measured displacement norm must
also exceed `128*epsilon_float64*(||x_next||+||x_cf||)`, its endpoint-subtraction
arithmetic screen. The supplied storage/inference sensitivity is retained as a separate source-confidence diagnostic; it is not added to this
arithmetic screen to erase a resolved observed direction. Thus normalized gains,
`S`, and current-reference alignment remain measured when their arithmetic
requirements hold, even if the conservative source sensitivity exceeds `D`.
The current-reference alignment additionally screens its target-to-reference
vector against its own float64 subtraction scale. None of these measurements
asserts that the unobserved exact learned predictions share the measured
vector's direction.

The `candidate-endpoints-2` contract saves these concepts separately:

| Field | Meaning / status rule |
|---|---|
| `candidate_direction_tolerance_l2` | Float64 branch-subtraction arithmetic threshold only |
| `candidate_displacement_arithmetic_tolerance_l2` | Arithmetic sensitivity of the observed endpoint subtraction, separately from the branch-gap threshold |
| `candidate_direction_normalization_status` | `resolved`, `zero_displacement`, `arithmetic_unresolved`, or `not_applicable` |
| `candidate_direction_input_sensitivity_ratio` | Supplied source sensitivity divided by observed `D`; undefined when `D=0` |
| `candidate_direction_input_precision_status` | `source_sensitivity_below_observed_gap`, `source_sensitivity_at_or_above_observed_gap`, `undefined_zero_branch_gap`, or `not_applicable` |
| `candidate_source_sensitivity_l2` | The full supplied source-prediction sensitivity, retained unchanged |
| `candidate_source_precision_status` | Provenance/qualification of that supplied sensitivity; not an arithmetic gain-sign status |

The full supplied source sensitivity still contributes to the integrated-margin
uncertainty and source-error diagnostics. Separating observation from source
confidence does not shrink a margin budget, turn an unresolved condition into a
certificate, or imply a particular gain sign. This paragraph concerns PF
branch-gap direction normalization; the separately documented IA target-to-center
degeneracy contract remains distinct.

Dose values use exactly `linspace(0,1,21)` union `{1/g}`. They evaluate the same
finite law at `x_cf+lambda*h` with branches, state, noise, and history fixed.
`lambda=0`, `1/g`, and `1` mean matched unconditional, matched conditional-only,
and actual CFG displacement, respectively. These are local reference
interventions; source-trajectory terminal SSCD is only a descriptor.

## Registry-to-estimand mapping

Each row is one base design; snapshot and fixed-tolerance variants do not add
new base designs. PNG/PDF are two formats of the same figure.

| ID | Estimand / saved quantities | Source / evidence |
|---|---|---|
| IR01 | Initial paired `E_u` against `E_c`; same-seed SSCD | Theorem 1 context; `learned_behavior` |
| IR02 | Initial `E_c` against terminal SSCD; fixed 20 equal-width all-range error bins | Descriptive recovery association; `learned_behavior` |
| IR03 | Within-prompt-centered `(E_u-E_c)` and SSCD; weighted descriptive slope | Seed association; `learned_behavior` |
| IR04 | `initial_conditional_target_rank`, `initial_unconditional_target_rank`; weighted ECDF | Euclidean candidate retrieval; `candidate_reference` |
| UB01 | Unique run/seed initial `||m_u-mu_ref||/sqrt(d)` ECDF | Lemma 2 context; `model_center` |
| UB02 | UB01 plus all distinct bank atoms' `||u_j-mu_ref||/sqrt(d)` | Separate observed populations; `model_center` |
| UB03 | `initial_candidate_reference_movement_rmse` against `initial_unconditional_candidate_reference_error_rmse`, unique initial run/seed | Reference movement versus learned discrepancy; `candidate_reference` |
| UB04 | Fixed-seed analytical `||bar_x_r^K(x_initial)-mu_K||/sqrt(d)` | Lemma 2 reference component; `reference_only_illustration` |
| IA01 | `a_parallel=<g*Delta,v>/||v||^2`; `off_target=||g*Delta-proj_v(g*Delta)||/||v||` | Corollary 3 direction, `v=x_star-mu_ref`; `model_center` |
| IA02 | `initial_target_coordinate_c` against `initial_target_coordinate_g`; retain `u` coordinate and all perpendicular norms | Corollary 3 geometry; `model_center` |
| IA03 | `injection_relative_mismatch=||g*Delta-g*v||/(g*||v||)` against SSCD | Complete-vector comparison; `model_center` |
| MD01 | `independent_replay_rmse` with `update_rounding_sensitivity_rmse`; replay status and construction residual separately logged | Lemma 4; `algebraic_qa` |
| MD02 | Actual `||g*kappa*Delta||/sqrt(d)` over saved SNR | Inserted displacement magnitude; `learned_behavior` |
| PF01 | `candidate_log_probability_gain` over current SNR | Direct Equation-69 matched effect; `candidate_reference` |
| PF02 | `candidate_normalized_log_odds_gain` over current SNR | New positive normalization; `candidate_reference` |
| PF03 | Weighted positive/negative/arithmetic-zero/unresolved/not-applicable sign counts and fractions | Direct matched effect, independent of `M0` eligibility; `candidate_reference` |
| PF04 | `candidate_matched_log_odds` against `candidate_guided_log_odds` at fixed snapshots | Same destination noise, same law; `candidate_reference` |
| PF05 | `candidate_current_alignment_cosine` | Equation-64 current alignment, not full-step sign; `candidate_reference` |
| PF06 | `candidate_dose_log_probability_gain` on the complete fixed lambda grid | Local analytical reference intervention; `candidate_reference` |
| PF07 | Refined `M0=M1` and separately derived `M1/M2/M3` positive coverage, compared with observed positive gain | Requested signed projected-gap-error refinement plus separately labeled directional refinements; `derived_directional_condition` |
| PF08 | `candidate_profile_class` and endpoint derivative `C0/C1` coverage | New concavity/endpoint test; `derived_directional_condition` |
| PF09 | `candidate_next_reference_target_contraction_rmse` | Difference of destination candidate-mean target errors, measured independently of gain; `candidate_reference` |
| PF10 | Early mean normalized gain versus subsequent actual learned `E_u` improvement; raw/within-prompt figure views and a saved all-time lag-one counterpart | Temporal descriptive association; `candidate_reference` |
| PF11 | Positive matched-gain rate by fixed progress bin and counterfactual odds regime | Conditional descriptive coverage; `candidate_reference` |
| PF12 | Actual target `H` minus median fixed-control atom `H`; retain each control identity and gain | Fixed mismatched-target specificity; `candidate_reference` |
| TS01 | Separate conditional/unconditional target-error group curves | Lemma 6 context; `learned_behavior` |
| TS02 | Actual `branch_gap_rmse` | Vector synchronization; `learned_behavior` |
| TS03 | Weighted CDF of paired `joint_target_error_rmse` at fixed snapshots | Simultaneous target recovery; `learned_behavior` |
| TS04 | First observed conditional and unconditional crossing of each of five fixed RMSE thresholds; recrossings/censoring | Paired observed-grid timing; `learned_behavior` |
| TS05 | Gap, `E_c`, candidate unconditional reference error, and `R_K*(1-p_current^K)/sqrt(d)` | Lemma 6 candidate-law terms; `candidate_reference` |
| TR01 | Last prediction `A=||m_c-x_star||/sqrt(d)` and `B=(g-1)*D/sqrt(d)` | Equation 16 under clean terminal gate; otherwise explicitly `last_prediction_geometry` |
| TR02 | Both distinct bound/observed-endpoint-error ratios | Theorem 7 plus Equation-16 triangle bound; `candidate_reference` |
| TR03 | Observed endpoint and both bound tolerance curves across full range | Clean-update reproduction/certification; `candidate_reference` |
| TR04 | Cosine of conditional residual and gap against endpoint error divided by `A+B` | Equation-16 cancellation geometry; `learned_behavior` under clean gate |

IR04 uses rank `1+number of distinct atoms strictly closer in Euclidean
latent distance`. The exact float64 distance tie count includes the paired
atom. Near numerical ties are recomputed using direct distance sums; near-tie
counts are saved separately. Candidate priors do not enter retrieval ranks.
A missing target cannot be silently replaced by a nearest atom.

IA coordinates are `a_b=<m_b-mu_ref,v>/||v||^2`. Perpendicular norms are measured
from `m_b-mu_ref-a_b*v`, divided by `sqrt(d)`. A target-center norm within the
source-precision screen `8*epsilon_source*max(1,||x_star||,||mu_ref||)` has no
assigned direction; valid target-error measurements remain available.

UB03 uses one observation per unique evaluation input, not repeated prompt
rows. UB04 uses exactly `r_i=initial_snr*10^(-i/8)`, `i=0..48`,
`alpha=sqrt(r/(1+r))`, and `sigma=sqrt(1/(1+r))`. Every SNR reuses the exact same
cached Gaussian input tensors. No learned prediction is extrapolated to a new
SNR.

TR uses two different bounds: `B_obs=A+B` and
`B_thm^K=[e_c+(g-1)*(mathcal{E}_t+R_K*(1-p_current^K))]/sqrt(d)`.
Actual endpoint certification requires the structural Equation-5 clean update
and its numerical check. A final DDPM variance floor remains nonzero even if
its measured residual rounds to zero. A candidate-bound ratio at zero endpoint
error is explicitly undefined; no denominator floor is added. Non-clean
last-prediction terms remain geometry and do not certify the actual endpoint.

## Fixed populations, aggregation, and status rules

- Complete prompt identity is `run_id, original_index, record_id, target_id`.
  Each represented prompt has equal total mass within each reported subset,
  divided equally across its included seeds. If a seed contributes repeated
  observations to a cell, those observations divide that seed's mass equally.
  Weighted quantiles use the inverse weighted empirical CDF.
- Outcome groups are all samples, `SSCD>0.75`, and `SSCD<=0.75`; both latter
  groups are always retained. Overlapping prompt counts are explicit. No
  group filters scientific inclusion. SSCD bins are `[0,.1,...,1]`, with
  separate finite lower/upper tails and missing-score counts. No score is
  numerically clamped to the display color range.
- PF04/PF06 snapshot positions are unique in-range members of
  `{0,1,2,4,9,floor((L-1)/2),L-1}` within eligible noisy-destination updates.
  TS03 positions are `{0,ceil(.2*L),floor((L-1)/2),L-1}`, clipped to range.
  Stored indices and actual SNRs are saved; gains do not choose snapshots.
- PF10 uses `k20=ceil(.2*L)` and `k40=ceil(.4*L)`, clipped to available
  prediction indices. Exposure averages all finite signed normalized gains at
  `0<=k<k20`; response is `E_u[k20]-E_u[k40]`, with finite/missing exposure
  counts. A degenerate or missing response window is unavailable. All-time
  lagged analysis uses every adjacent actual prediction pair at fixed lag one.
  Within-prompt lagged centering is performed separately at each update.
- PF11 has ten normalized-progress bins and counterfactual log-odds edges
  `[-inf,-10,-5,-2,0,2,5,10,inf]`. Empty, low-count, and unassigned-regime cells
  are explicit. The interval from positive mass to positive-plus-unresolved
  mass is an unresolved-sign envelope, not a confidence interval.
- The common TS03 tolerance grid is 201 evenly spaced values from zero through
  the full finite joint-error maximum. No extremes are trimmed. TS04 uses every
  threshold in `{0.1,0.2,0.3,0.5,1.0}`. First passage means strictly below the
  diagnostic threshold; simultaneous TS03 coverage uses `<=`. Never-reached
  times remain censored, not ordinary finite steps. Recrossings and consecutive
  three-prediction persistence are separately recorded.
- Fixed thirds of the full prediction range provide descriptive phase
  summaries. Target splits use SHA-256 of
  `candidate-target-stability-v1|canonical target identity`; no gain, SSCD, or
  error enters the hash. Exact target aliases should carry their canonical
  candidate atom ID. Splits are descriptive checks on explored data, not
  untouched confirmatory evaluation.
- Controls use a fixed salted hash of `(bank_hash,target_atom,control_atom)` to
  select up to 16 distinct non-target atoms, reused at every seed/step.
  Individual gains remain saved. Control atom choices never depend on measured
  gain signs.
- Numerical gain signs, profile statuses, and integrated-margin signs remain
  distinct. Positive-coverage denominators retain eligible unresolved rows.
  Not-applicable counts and population fractions are separately saved. No
  summary requires positive gain, nonempty sufficient-condition coverage, or
  favorable group ordering.
- Terminal certification curves retain the same applicable observed-endpoint population for both bounds. Missing bound values remain in the denominator and contribute to an explicit unresolved envelope; they do not inflate resolved coverage by being dropped.
- Reported intervals are descriptive IQRs or unresolved-sign envelopes. No
  independent-timestep bootstrap or inferential confidence interval is used.
  Whole trajectories and shared-target dependence remain visible in scalar
  identities and fixed split summaries.
