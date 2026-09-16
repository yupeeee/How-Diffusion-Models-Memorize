> The active figure organization is now [four_stage_experiments.md](four_stage_experiments.md). Seven-primary filenames and roles below describe the preceding design; compatible mathematical measurements and numerical contracts remain reusable. Matching LaTeX is unavailable; historical label strings are author-supplied, not verified source labels.

# Submission alignment for the fixed seven-statement evidence suite

This note describes the implemented measurement plan and a proposed finite-terminal-update extension. It does not report new experimental results or establish that the suite is publication-ready. The author must execute and inspect the measurements, resolve the manuscript decisions below, and decide how to incorporate the extension. This change does not edit the manuscript.

The mounted [revised.pdf](../../revised.pdf) was inspected; its verified SHA-256 is `fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`. Its visible statements, equations, and terminal-premise text are the mathematical source. No matching LaTeX source was found. The label strings below come from the author's implementation brief and are mapped to the visible results; their spelling has not been independently checked against matching LaTeX. Older manuscripts, candidate-design names, and historical reviewer documents are not substituted as authorities.

| Existing result and supplied label | Fixed primary stem | Empirical meaning and mandatory supporting evidence |
| --- | --- | --- |
| Theorem 1, `thm:initial_conditional_recovery` | `theorem1_loss_recovery` | Pair-specific normalized genuine forward loss versus Gaussian conditional recovery RMS, with a matched unconditional target-error control. Appendices retain initial branch comparison and Gaussian distribution transfer. |
| Lemma 2, `lem:initial_unconditional_baseline` | `lemma2_unconditional_baseline` | Analytical reference convergence below and through initialization, with only native initial network/reference markers in the primary; the complete native Gaussian sweep is mandatory in the appendix. Initial concentration uses the same empirical-law mean. |
| Corollary 3, `cor:initial_cfg_amplification` | `corollary3_initial_cfg_amplification` | Complete normalized injection geometry relative to the limiting vector. The full guided-error conclusion remains an appendix comparison. |
| Lemma 4, `lem:matched_one_step_cfg_displacement` | `lemma4_matched_displacement` | Predicted versus independently observed matched displacement, with full-vector residuals and explicit independent/constructed provenance. This is an implementation identity check. |
| Proposition 5, `prop:matched_one_step_target_posterior_feedback` | `proposition5_posterior_feedback` | Strict feedback prevalence and strict original-condition coverage on identical weighted populations. Both condition-versus-gain and first-update posterior comparisons remain mandatory appendices. |
| Lemma 6, `lem:target_specific_synchronization` | `lemma6_target_specific_synchronization` | Paired joint target error, actual vector branch gap, and the full original bound. Appendices retain branch target errors, bound versus gap, and posterior concentration. |
| Theorem 7, `thm:final_reproduction`, with the explicitly proposed extension below when needed | `theorem7_final_reproduction` | Actual latent reproduction and two sufficient-bound coverages over all tolerance change points. Terminal components are appended when supported; original uncorrected applicability is retained. |

## Common empirical reference and input laws

The empirical law is `D_K=sum_j pi_j delta_(u_j)`, with its existing frozen atoms, exact-atom deduplication, masses, preprocessing, and source provenance. The default is equal mass per distinct compatible complete cached target before prompt selection. An explicitly declared alternative law keeps its validated supplied masses. Neither construction identifies a checkpoint's complete training corpus or its frequencies.

Every law-based comparison uses `mu_K=sum_j pi_j u_j`, the same Gaussian posterior mean `bar_x_K(t,z)=sum_j w_j(t,z)u_j`, target mass `p_K(t,z)`, and fixed full-law radius `R_K=max_j ||u_j-x_star||`. In particular, `e_u_K=||m_u-bar_x_K||` is a reference error, not unconditional target error. The mean of model outputs is not substituted for `mu_K`; any retained model-center diagnostic has the distinct notation `mu_ref`.

The conditional reference `e_c=||m_c-x_star||` uses the stated single-target idealization. It is not inferred from SSCD. Exact-prompt ambiguity and absent target atoms remain explicit scope information; records are not filtered to establish this assumption.

Forward-corrupted targets, forward-corrupted draws from `D_K`, Gaussian probes, generated states, analytical reference-only queries, and matched counterfactual states retain different input-source identities. Cached branches remain canonical epsilon; estimates use the unscaled state and saved native coefficients without another epsilon conversion. Consecutive stored updates use their actual destination levels, not an assumed adjacent training timestep.

## Finite-SNR interpretation of the first three results

Theorem 1 states convergence in probability under its high-noise loss assumption. The primary coordinates `A_c=sqrt(Lhat_T(c)/(d*SNR_T))`, `Y_c=sqrt(mean_n e_T(Z_n,c)^2/d)`, and the analogous unconditional target-error control are finite-sample empirical RMS summaries. These are not a new moment-convergence result. The horizontal and vertical measurements use different independent input streams, so no equality diagonal is asserted between them.

The appendix keeps the forward-error identity and distribution-transfer argument, using the supplied labels `eq:proof_initial_forward_error_identity` and `eq:proof_initial_recovery_bound`. At normalized tolerance `rho`, the transfer inequality is

`P(e_T(Z,c)/sqrt(d)>rho) <= L_T(c)/(d*SNR_T*rho^2)+delta_T`.

The total-variation term is retained, including its unclipped Pinsker value, as are vacuous probability bounds. Replacing the population loss by Monte Carlo loss produces a plug-in comparison. Monte Carlo bootstrap intervals are not certified theorem bounds. Shared Gaussian seed IDs and shared targets require coherent population resampling; pair-specific intervals alone do not establish population significance.

For Lemma 2, the 97-point reference-only grid extends from `SNR_T*1e-6` to `SNR_T` by default, using a fixed Gaussian bank and the same `D_K`. This tests the analytical reference component. Learned curves retain the entire available native range and stop where native measurements stop. This does not observe a trained network at arbitrarily small SNR.

For Corollary 3, let `v=x_star-mu_K` and `J=g*(m_c-m_u)`. The geometry records `q=<J,v>/(g*||v||^2)`, the directly formed perpendicular vector `J_perp=J-g*q*v`, `r_perp=||J_perp||/(g*||v||)`, and `E_rel=||J-g*v||/(g*||v||)`. Thus `E_rel^2=(q-1)^2+r_perp^2`. This normalization is an explicitly derived visualization of the full discrepancy, not a new numbered theorem or a fitted coefficient. The guidance cancellation is algebra at fixed `g`, not a guidance sweep. Exact-zero and float64-arithmetic-unresolved directions retain their absolute errors and exclusion reasons.

## Original Proposition-5 condition and target-specific behavior

The primary sufficient test remains `M=||Delta||-e_c-e_u_K-V_K`. Its `V_K` is the original integral associated with `eq:cross_step_reference_variation`; sharper exploratory margins cannot replace it. The gain `H=log p_K(next,x_next)-log p_K(next,x_cf_next)` uses the same destination law at both endpoints. The signed integrated identity associated with `eq:proof_integrated_target_posterior_change` remains independently audited.

Strict condition coverage uses `M>0`; observed strict feedback uses a numerically resolved positive gain. Stable endpoint log odds can resolve a gain sign when its log-probability magnitude underflows. Zero displacement is not a strict case. Nonnegative comparisons are logged separately. Both fractions use the same eligible seed population, prompt-balanced weights, and timestep; unresolved signs remain in the denominator and contribute only to separately labeled ambiguity bounds. Quadrature and source-sensitivity estimates do not become rigorous interval certificates. Positive feedback does not imply that the sufficient condition held, and zero condition coverage must remain a numerical zero.

Lemma 6 retains `D=||m_c-m_u||/sqrt(d)` and `S=(e_c+e_u_K+R_K*(1-p_K))/sqrt(d)`. The paired quantity `Q=max(||m_c-x_star||,||m_u-x_star||)/sqrt(d)` is computed before aggregation. Besides `D<=S`, the definitions and posterior concentration give `Q<=S`: conditional target error is bounded by `e_c`, and unconditional target error by `e_u_K+R_K*(1-p_K)`. This is recorded as a direct consequence rather than the lemma's verbatim statement. Small branch gap alone can describe agreement on the wrong target. Current-reference quantities remain available at the last prediction even when the destination posterior is undefined.

## Proposed finite-terminal-update extension

The PDF states the clean terminal condition in Equation 5, supplied label `eq:terminal_clean_update`, as an additional condition rather than a consequence of the general update in Equation 4, `eq:reverse_process`. It also contains the phrase “including the deterministic and stochastic samplers considered here.” That assertion must be reconciled with the saved per-sampler applicability audit; it must not be defended by weakening the implementation gate.

The unchanged original structural gate requires a final affine update, `A=0`, `kappa=1`, destination `alpha=1` and `sigma=0`, and additive noise exactly zero. Numerical consistency is checked separately. A tiny nonzero variance still fails the exact structural premise. The following extension does not change those original flags.

For the final current prediction, define `m_g=m_c+(g-1)*Delta` and write the saved affine update as `x_out=a+kappa*m_g`. Then

`r_sched=a+(kappa-1)*m_g`, and `x_out=m_g+r_sched`.

The CFG identity in Equation 3 and terminal target decomposition in Equation 16, supplied label `eq:terminal_target_decomposition`, give

`||m_g-x_star|| <= B_obs := e_c+(g-1)*||Delta||`.

Applying Lemma 6 gives

`B_obs <= B_ref := g*e_c+(g-1)*(e_u_K+R_K*(1-p_K))`.

For an independently valid bound `delta_sched>=||r_sched||`, the triangle inequality therefore yields

`||x_out-x_star|| <= B_obs+delta_sched <= B_ref+delta_sched`.

For a clean update `delta_sched=0`, this includes the existing Theorem-7 comparison. For a non-clean update it is a proposed finite-terminal-update extension and must be identified as such in the manuscript and figure metadata. No existing manuscript result label is assigned to this extension.

For deterministic affine updates, the predictor computes `r_sched_pre=A*z+(kappa-1)*m_g` from pre-update quantities. Its API does not accept the observed endpoint. A separate QA function compares the independently predicted update with `x_out`; `x_out-m_g` is retained only as observed-output QA. Source-dtype sensitivity envelopes are numerical screens, not proved machine-arithmetic error bounds.

An independently saved or validly replayed additive innovation permits a pathwise defect calculation. An innovation reconstructed from the endpoint is rejected as a predictor-side input. If only a supported, conditionally isotropic Gaussian variance law is available, set the predeclared run failure budget to `eta=0.05`, assign `eta_i=eta/m`, and let `q_i=sqrt(chi2_quantile(d,1-eta_i))`. The implementation evaluates the equivalent inverse-survival quantile to avoid rounding `1-eta_i` to one. It uses

`delta_sched=||A*z+(kappa-1)*m_g||+rho*q_i`.

Here `m` is the predeclared number of retained final updates in that scientific run, conservatively reserving a share even when a row cannot later be used. Conditional Gaussian tail bounds and the union bound give simultaneous coverage at least `1-eta` across those updates. Independence between trajectories is not required. The exact run identity, `m`, `eta`, variance type, version contract, `rho`, and quantile are recorded. This is a probability-qualified comparison under the declared sampler law, not a deterministic certificate or a confidence band for an empirical CDF. A realized noise-bound exceedance remains a possible probability event, separate from reconstruction or integration failure.

The three terminal curves use one common population and weighting: actual endpoint error, `B_obs+delta_sched`, and `B_ref+delta_sched`. They retain all tolerance change points and explicitly report exact-zero mass. Their expected ordering is `F_ref<=F_obs<=F_actual` under the stated deterministic or simultaneous-probability scope. Latent tolerance is not selected from SSCD. Metadata distinguishes the original clean theorem, deterministic/pathwise extension, Gaussian-noise-bound extension, and unsupported contracts; non-clean supported cases carry `manuscript_extension_required=true`.

## Decisions for the authors

- Supply matching LaTeX and confirm the supplied statement/equation label strings before editing cross-references.
- Decide where and how to state the finite-terminal-update extension, its short proof, and its Gaussian qualification; no automatic manuscript edit has been made.
- Reconcile the PDF's assertion about clean endpoints for the experimental samplers with the unchanged saved gate and numerical QA.
- Describe `D_K`, the single-target conditional assumption, finite-SNR measurements, and separate input laws explicitly; do not identify the empirical law with the complete training corpus.
- Approve the fixed figure plan and uncertainty terminology. It was chosen after reviewing SDv1-DDIM and is not held-out confirmation on those same observations.
- Execute the author-facing commands in the final evidence plan, inspect failures and unfavorable outcomes, and fill empirical statements only from the resulting validated artifacts. The code-edit task has not produced new plots or empirical verification.

## Precision revision and material empirical qualifications

The [numerical refinement plan](numerical_refinement_plan.md) keeps all seven statements and both existing theorem prerequisites. Its bounds are auxiliary numerical derivations. Fixed-cache signs are functions of the declared saved tensors, scalar coefficients and `D_K`. Directed arithmetic can certify the operations explicitly enclosed; a float64 assessment, agreement across precisions, or a dtype-multiplier envelope cannot. Source perturbation sensitivity is reported separately, and reference-model identification remains unresolved regardless of precision. New fixed-cache classifications preserve the legacy source-sensitivity comparisons and migration receipts.

The initial displays remain finite-noise measurements of distinct forward and Gaussian input laws. Only the analytical empirical reference is evaluated below checkpoint initialization. Native initial errors and the complete native-noise sweep remain visible; there is no measured network limit. Corollary-3 approximation errors and premise/error decomposition remain empirical descriptions, without selected favorable seeds or refitted reference centers.

The Proposition-5 caption must report actual original-condition coverage and unresolved numerical mass after author execution. A resolved negative condition may coexist with an unknown variation magnitude. Zero condition coverage does not invalidate observed positive feedback, but it prevents attributing that feedback to observed applicability of the sufficient condition. A saved-endpoint gain and an affine-path variation cannot share an implication claim without a consistent endpoint contract or justified residual transfer. The last prediction update is outside manuscript `t>=2` and remains separately identified.

The terminal plot remains an explicitly labelled finite-terminal-update extension whenever the original clean-update gate fails. Preserve the original clean-case applicability count in the visible caption/status. The exact looseness decomposition attributes the difference between original bounds without replacing either bound. The manuscript statement/proof and experimental sampler description still need author alignment before submission; this revision does not perform that manuscript edit or generate new evidence.
