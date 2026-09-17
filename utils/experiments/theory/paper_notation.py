"""Presentation vocabulary checked against revised.pdf, Appendix A and D.5.

These strings never define measurements or change saved scalar units. In figure
captions, the empirical evaluation law and finite-step correction retain their
qualifications; adopting the manuscript's symbols does not identify that law
with the model's complete training marginal.
"""

MANUSCRIPT_SHA256 = "fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f"
NOTATION_VERSION = "four-stage-manuscript-notation-16"

NORM_AXIS = r"$\|\cdot\|/\sqrt{d}$"
INITIAL_BRANCH_LABELS = (r"$b=c$", r"$b=\varnothing$")
INITIAL_DISTRIBUTION_LABELS = {
    "initial_unconditional": r"$\hat{\mathbf{x}}_T(\varnothing)$",
    "candidate_atoms": r"$\mathbf{X}\sim\mathcal{D}$",
}
CHRONOLOGICAL_LABELS = {
    "gap": r"$\|\boldsymbol{\Delta}_t\|$",
    "joint_error": r"$\max_{b\in\{c,\varnothing\}}\|\hat{\mathbf{x}}_t(b)-\mathbf{x}^{\star}\|$",
    "bound": r"$e_t^{\parallel}(\boldsymbol{\Delta})+R(1-p_t)$",
    "conditional": INITIAL_BRANCH_LABELS[0],
    "unconditional": INITIAL_BRANCH_LABELS[1],
}
REFERENCE_LABELS = {
    "reference": r"$\|\bar{\mathbf{x}}_t(\mathbf{z},\varnothing)-\boldsymbol{\mu}\|$",
    "learned": r"$\|\hat{\mathbf{x}}_t(\mathbf{z},\varnothing)-\boldsymbol{\mu}\|$",
    "reference_error": r"$e_t(\mathbf{z},\varnothing)$",
}
# Historical zero-companion labels; these figures are no longer active.
ZERO_REFERENCE_LABELS = {
    "reference": r"$\|\bar{\mathbf{x}}_t(\mathbf{z},\varnothing)-\mathbf{0}\|$",
    "learned": r"$\|\hat{\mathbf{x}}_t(\mathbf{z},\varnothing)-\mathbf{0}\|$",
    "reference_error": REFERENCE_LABELS["reference_error"],
}
DOSE_REFERENCE_TICK_LABEL = r"$1/g$"
FEEDBACK_GAIN_LABEL = r"$p_{t-1}(\mathbf{x}_{t-1})>p_{t-1}(\mathbf{x}_{t-1}^{\mathrm{cf}})$"
FEEDBACK_CONDITION_LABEL = r"$\|\boldsymbol{\Delta}_t\|>e_t^{\parallel}(\boldsymbol{\Delta})+\mathcal{V}_t$"
MARGIN = r"\|\boldsymbol{\Delta}_t\|-e_t^{\parallel}(\boldsymbol{\Delta})-\mathcal{V}_t"
LOG_GAIN = r"\log[p_{t-1}(\mathbf{x}_{t-1})/p_{t-1}(\mathbf{x}_{t-1}^{\mathrm{cf}})]"

# These are concise legend/axis identifiers for the manuscript bound terms.
# Plotted bound values still include the saved independent terminal correction.
# NOTATION_DETAILS explicitly defines that caption-only abbreviation and scope.
TERMINAL_EXPRESSIONS = {
    "actual": r"\|\mathbf{x}_0-\mathbf{x}^{\star}\|",
    "observable": r"e_1(c)+(g-1)\|\boldsymbol{\Delta}_1\|",
    "reference": r"e_1(c)+(g-1)[e_1^{\parallel}(\boldsymbol{\Delta})+R(1-p_1)]",
}
TERMINAL_LABELS = {name: "$" + value + "$" for name, value in TERMINAL_EXPRESSIONS.items()}

PAPER_AXES = {
    "reference_variation_per_prompt": {"x": r"$T-t$", "y": r"$\mathcal{V}_t/\sqrt{d}$"},
    "corollary3_guidance_scale_vs_loss": {"x": r"$\sqrt{L_T(c)/(d\,\mathrm{SNR}_T)}$", "y": r"$\widehat{g}$"},
    "initial_loss_recovery": {
        "x": r"$\sqrt{L_T(c)/(d\,\mathrm{SNR}_T)}$",
        "y": r"$\sqrt{\mathbb{E}_{\mathbf{x}_T}[\|\hat{\mathbf{x}}_T(b)-\mathbf{x}^{\star}\|^2]/d}$",
    },
    "branch_gap_posterior_response": {
        "x": r"$s$",
        "y": r"$\log[p_{T-1}(\mathbf{z}_T(s))/p_{T-1}(\mathbf{z}_T(0))]$",
    },
    "branch_gap_synchronization": {"x": r"$T-t$", "y": NORM_AXIS},
    "branch_gap_per_prompt": {"x": r"$T-t$", "y": r"$\|\boldsymbol{\Delta}_t\|/\sqrt{d}$"},
    "conditional_reference_error_per_prompt": {"x": r"$T-t$", "y": r"$e_t(c)/\sqrt{d}$"},
    "unconditional_reference_error_per_prompt": {"x": r"$T-t$", "y": r"$e_t(\varnothing)/\sqrt{d}$"},
    "target_probability_per_prompt": {"x": r"$T-t$", "y": r"$p_t$"},
    "reference_branch_gap_per_prompt": {"x": r"$T-t$", "y": r"$\|\bar{\boldsymbol{\Delta}}_t\|/\sqrt{d}$"},
    "terminal_bound_coverage": {"x": r"$\tau/\sqrt{d}$", "y": r"$\Pr(\,\cdot\leq\tau)$"},
    "initial_unconditional_mean_concentration": {
        "x": r"$\|\cdot-\boldsymbol{\mu}\|/\sqrt{d}$", "y": "CDF",
    },
    "unconditional_reference_convergence": {"x": r"$\mathrm{SNR}_t$", "y": NORM_AXIS},
    "posterior_feedback_over_time": {"x": r"$\mathrm{SNR}_t$", "y": r"$\Pr(\cdot)$"},
    "posterior_feedback_condition_margin": {
        "x": "$[" + MARGIN + r"]/\sqrt{d}$", "y": "$" + LOG_GAIN + "$",
    },
    "branch_target_errors": {
        "x": r"$T-t$", "y": r"$\|\hat{\mathbf{x}}_t(b)-\mathbf{x}^{\star}\|/\sqrt{d}$",
    },
    "synchronization_bound": {"x": r"$T-t$", "y": r"$\cdot/\sqrt{d}$"},
    "branch_gap_peak_step": {
        "x": r"$\min\,\mathrm{arg\,max}_{T-t}\|\boldsymbol{\Delta}_t\|$", "y": r"$\Pr(\cdot)$",
    },
    **{stem: {"x": "$[" + TERMINAL_EXPRESSIONS[name] + r"]/\sqrt{d}$",
              "y": "$" + TERMINAL_EXPRESSIONS["actual"] + r"/\sqrt{d}$"}
       for stem, name in (("terminal_observable_bound", "observable"), ("final_reproduction_bound", "reference"))},
}

NOTATION_SCOPE = (
    "Labels follow revised.pdf Appendix A and Equations 2, 7-9, 12-15, 66-69. "
    "The requested projected-gap-error-1 contract uses the signed projected error e_t^parallel(Delta)=u_t^T(Delta_t-bar{Delta}_t), where u_t=Delta_t/||Delta_t|| and bar{Delta}_t=bar{x}_t(c)-bar{x}_t(empty). It is evaluated from the actual learned and reference gap vectors, may be negative, and is never replaced by a norm, absolute value or positive part. The unchanged revised mathcal{V}_t takes its positive part only after integrating the signed unit-gap projection of next-step posterior reference minus the current reference. At exactly zero Delta, u=0, e^parallel=0 and mathcal{V}=0 form an explicit computational extension; no unit direction is claimed. The identity ||Delta||=e^parallel+u^T bar{Delta}, followed by Cauchy and posterior concentration, gives D<=S and the qualified terminal bound. The condition retains its matched-endpoint qualifications. These are requested derived refinements, not unchanged PDF expressions. Q remains descriptive and is not asserted below S. "
    "Vectors are bold. The displayed mu is the selected theory centre: by default the "
    "saved average of 10000 unconditional posterior clean references at the smallest SNR of the prespecified analytical extension on independent Gaussian inputs. "
    "Its finite-selected-SNR reference bias and Monte Carlo uncertainty do not establish an exact training-data mean; this estimator performs no denoiser inference. "
    "The empirical posterior, reference error, radius and exact finite-bank mean retain "
    "the same saved atom law independently of selected mu. The receipt identifies any "
    "explicit native-initial estimator or exact finite-bank-mean fallback; omitting K does not identify the law with the training distribution. "
    "All distances and signed projected errors are divided by sqrt(d) once, as indicated by the axes; a projected error is not a nonnegative RMSE norm. Raw expressions "
    "in legends identify the quantities before this shared display normalization. "
    "Terminal bound labels abbreviate their corrected quantities; the additive "
    "correction and its scope are stated explicitly in the figure-specific caption. "
    "Pr and CDF on fraction axes denote the saved weighted sample fractions, with their "
    "original populations and unresolved mass, not a new population-probability certificate. "
    "Default condition signs are numerical estimates assessed using saved quadrature and float64 "
    "roundoff uncertainty; optional interval certificates have separately recorded scope."
)
NOTATION_DETAILS = {
    "reference_variation_per_prompt": "mathcal{V}_t is the requested revised directional positive variation: [integral_0^1 <Delta_t/||Delta_t||,bar{x}_{t-1}(x_cf+s*g*kappa_t*Delta_t,empty)-bar{x}_t(x_t,empty)> ds]_+. The positive part is taken after the signed integral, not inside the integrand; neither a norm nor an absolute value replaces the projection. This revises the historical Equation 15 norm integral. At exactly zero Delta, u=0, e^parallel=0 and mathcal{V}=0 are the computational extension and no unit direction is formed. The saved direct_prop5_variation_rmse already normalizes each per-seed value by sqrt(d); seed averaging comes afterward, never before the positive part. Curves use one complete fixed seed cohort and its mean terminal SSCD as a fixed color. The domain is t=2,...,T, so T-t ends at T-2; no t=1 or output value is fabricated. Finite nonnegative estimates remain included despite unresolved condition signs; refined interval midpoints are not substituted. Missing, nonfinite, negative, wrong-contract or structurally inapplicable required estimates exclude the whole curve with an audit reason. Numerical statuses remain explicit, and estimates are not certified exact integrals.",
    "corollary3_guidance_scale_vs_loss": "L_T(c) is the saved genuine forward-target squared-L2 loss averaged over its independent draws. The saved x is L_T(c)/(d*SNR_T), normalized once. The display takes one square root of saved x, outside the mean over forward draws; it is not the mean of per-draw roots. The saved scalar and fitted coefficient are unchanged. The ordinate is the signed joint no-intercept least-squares coefficient from Corollary 3 Equation 54, fitting hat{x}_T(c;g)-mu to x_star-mu over the complete saved initial seed cohort. Fits use actual saved first-prediction states; Gaussian-bank matches are audited separately rather than assumed. It equals the arithmetic mean of seed coefficients because their target direction is identical. Mu is the selected reference-based mean with its saved receipt, not zero. Pair color is the same complete seeds' mean terminal SSCD. The horizontal guide is the actual configured g and is labeled with its value. No g*Delta substitution, clipping, or division of the coefficient by sqrt(d) is applied.",
    "reference_branch_gap_per_prompt": "bar{Delta}_t is conditional minus unconditional analytical clean reference at the same actual saved trajectory state and noise level. Under the existing single-target conditional idealization, bar{x}_t(c)=x_star, so its norm equals that of the opposite-sign saved vector bar{x}_t(empty)-x_star. The saved direct_reference_target_error_rmse already divides this norm by sqrt(d) once. Each curve averages individual seed norms over the same complete fixed seed cohort at all chronological pre-update predictions T-t, and its fixed color is the same seeds' mean terminal SSCD. This is not the learned branch gap, a difference of learned-reference errors or a radius-tail upper bound.",
    "conditional_reference_error_per_prompt": "e_t(c) is the conditional clean-reference error under the fixed single-target conditional reference assumption. Each curve averages saved per-seed e_t(c)/sqrt(d) norms for one prompt-target pair; norm and dimensional normalization precede averaging. The same complete seed cohort is used at every chronological pre-update prediction T-t and for its fixed mean terminal-SSCD color.",
    "unconditional_reference_error_per_prompt": "e_t(empty) is the norm of the learned unconditional clean estimate minus its declared-law posterior clean reference at the actual saved trajectory state; it is not unconditional target distance. Each curve averages the saved per-seed normalized errors once over the same complete seed cohort, also used for its fixed mean terminal-SSCD color. T-t is the chronological pre-update prediction index.",
    "target_probability_per_prompt": "p_t is the target posterior probability evaluated on the actual saved generated trajectory state under the declared finite law. Each curve is the arithmetic mean of exp(saved per-seed direct_target_log_probability), not exp of the mean log probability, over the same complete fixed seed cohort at every chronological pre-update prediction T-t. Its fixed color is mean terminal SSCD over those same seeds. Probability remains dimensionless on [0,1], with no dimensional normalization or conditional-only counterfactual substitution.",
    "initial_loss_recovery": "L_T(c) and the Gaussian expectation are estimated by the saved independent forward draws and paired Gaussian seeds. The square root is outside the mean of squared target errors for both b=c and b=empty; this is not the mean of individual norm errors. Colors are pair mean terminal SSCD; CI means the saved Monte Carlo bootstrap interval, not uncertainty across independent prompts.",
    "branch_gap_posterior_response": "The bold z_T(s) is the segment defined in Appendix D.5 Equation 66, evaluated at the fixed first prediction: z_T(s)=x_cf,T-1+s*g*kappa_T*Delta_T. s=0, 1/g, 1 denote the matched unconditional, conditional-only and guided endpoints. Only s=1/g has a vertical reference line: it yields x_cf,T-1+kappa_T*Delta_T=a_T+kappa_T*xhat_T(c). The complete interval and the endpoint measurements at s=0 and s=1 remain plotted. The independently saved endpoint status determines whether s=1 coincides with the actual update; the endpoint coordinate alone makes no such claim.",
    "branch_gap_synchronization": "The increasing horizontal coordinate T-t is the saved chronological prediction index k; t=T-k, with k=0 at initialization. Native scheduler noise labels remain distinct. The maximum target distance is formed per sample before aggregation; it is not max of two group medians.",
    "branch_gap_per_prompt": "Each curve is the arithmetic mean of per-seed ||Delta_t||/sqrt(d) for one prompt-target pair, using the same complete evaluation seed cohort at every prediction. Its fixed color is mean terminal SSCD over those same seeds. T-t is the saved chronological prediction index k; no distance sorting is applied. The norm is computed before seed averaging, and the saved dimensional normalization is applied exactly once.",
    "branch_target_errors": "b=c and b=empty select the conditional and unconditional target distances. The latter is not e_t(empty), which measures learned-to-reference error. The increasing coordinate T-t equals saved prediction index k, with t=T-k.",
    "synchronization_bound": "The refined bound is S=(e_t^parallel(Delta)+R(1-p_t))/sqrt(d), with e_t^parallel(Delta)=u_t^T(Delta_t-bar{Delta}_t), u_t=Delta_t/||Delta_t||, and bar{Delta}_t=bar{x}_t(c)-bar{x}_t(empty). The error is signed. D<=S follows from ||Delta||=e^parallel+u^T bar{Delta}, Cauchy and posterior concentration; it does not require nonnegative e^parallel. Solid curves show D=||Delta_t||/sqrt(d); dashed curves show S, the right-hand side of Lemma 6 under the requested gap-error substitution. Only these two quantities are displayed for each SSCD group, using the same saved population and weights. The axis dot stands for either legend quantity, divided by sqrt(d) once. The separately saved maximum target distance is not displayed. T-t is chronological prediction index k.",
    "unconditional_reference_convergence": "The explicit bold z denotes the same fixed Gaussian evaluation bank across native noise labels, z~N(0,I), independently of the mean-estimation draws. The selected mu is identified by its saved receipt. The horizontal guide is the distance from the exact finite-bank weighted atom mean to selected mu divided by sqrt(d); it has no legend entry, and its numeric value and meaning remain in caption/audit metadata. A nonzero offset is expected when those centres differ. The analytical range contains no model extrapolation and does not imply exact convergence to the chosen finite-SNR estimate; the saved bank-to-selected offset is not forced to zero.",
    "initial_unconditional_mean_concentration": "Both populations use the same selected mu. By default it is a vector Monte Carlo average of 10000 unconditional posterior clean references at the smallest SNR of the saved analytical extension on independent Gaussian inputs; the Gaussian evaluation bank is separate. Its receipt records sample count, seed, finite-law identity, native initial SNR, separate estimation SNR, analytical grid index, extension depth, vector identity and Monte Carlo standard error; no model evaluation is performed. Finite-selected-SNR bias and the measured finite-bank offset remain distinct from sampling uncertainty. Explicit reference-initial and exact-bank modes retain their recorded meanings.",
    "branch_gap_peak_step": "The displayed min argmax is the earliest resolved global maximum over chronological T-t, using the saved exact/precision-aware tie rule. The vertical quantity is weighted bin mass on the original population, retaining unassigned mass. No peak location or scalar input is recomputed for the label.",
    "posterior_feedback_over_time": "Each mathematical legend identifies a strict event. Its plotted fraction includes signs resolved by the saved numerical assessment, including default numerical estimates, and any optional interval certificates. Unresolved bands and condition upper boundaries retain possible unresolved mass; neither is a confidence interval or certified condition coverage. Absence of a certificate alone adds no unresolved mass. Shared zero means zero assessed strict-positive condition fraction in both outcome groups, not equality of every sample's margin to zero.",
    "posterior_feedback_condition_margin": "The vertical zero guide marks the refined projected-gap-error sufficient-condition boundary; the horizontal zero guide marks no change in target posterior probability between the matched endpoints. These guides have no legend entries. Every finite saved margin/gain pair is shown as an ordinary dot, including numerically unresolved estimates, without sign-based filtering, Observed/Unresolved marker classes or a status legend. The combined figure contains all saved timesteps. Additional views group the same rows by chronological step_index k, with manuscript t=T-k recorded when T is available; no native scheduler label or terminal-output prediction is invented. Steps without finite pairs retain explicit unavailable publication status. Numerical sign, uncertainty, endpoint-transfer and implication-eligibility fields remain in the audit and do not determine this plot's marker shape or finite-row inclusion.",
    "terminal_bound_coverage": "The actual-error curve uses ||x_0-x_star||. The observable and refined reference-bound legends abbreviate their base expressions: their plotted fractions are Pr(base expression + saved independent terminal correction <= tau), on the same saved population and weights within each terminal-SSCD group. Gold/purple denote SSCD>0.75 and SSCD<=0.75; solid/dashed/dash-dot denote actual/observable/reference. Group probabilities are normalized separately with equal represented prompt mass, and share a common tolerance axis. The correction remains included numerically even though it is omitted from the artwork. The horizontal display divides tau by sqrt(d) once. The refined reference bound uses e_1^parallel(Delta), and the correction keeps its recorded deterministic/pathwise/probabilistic scope. D<=S yields the observable-to-reference ordering; row-level checks remain distinct from weighted curve ordering.",
    "terminal_observable_bound": "The horizontal label abbreviates the observable base expression. The plotted x coordinate is [e_1(c)+(g-1)||Delta_1|| + saved independent terminal correction]/sqrt(d), not the uncorrected expression alone. The additive correction is omitted only from the artwork; its saved scope, zero cases and independent construction remain unchanged. The diagonal marks equality of the corrected plotted coordinates, not a deterministic guarantee when the saved mode is probabilistic.",
    "final_reproduction_bound": "The horizontal label abbreviates the reference base expression. The plotted x coordinate is [e_1(c)+(g-1)(e_1^parallel(Delta)+R(1-p_1)) + saved independent terminal correction]/sqrt(d), not the uncorrected expression alone. The additive correction is omitted only from the artwork; its saved scope, zero cases and independent construction remain unchanged. All posterior/reference terms use the declared empirical evaluation law. The diagonal marks equality of corrected plotted quantities. The bound follows from the refined gap inequality under the same recorded terminal and correction qualifications; it is a derived refinement, not the unchanged PDF expression verbatim.",
}
