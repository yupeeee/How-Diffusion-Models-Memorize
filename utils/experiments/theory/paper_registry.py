"""Fixed paper figure definitions, independent of measured effect directions."""

from __future__ import annotations

import copy

REGISTRY_VERSION = "paper-notation-3"
FORMULA_VERSION = "revised-paper-contract-1"
# Stable CSV population keys; mathematical typography belongs only to display labels.
GROUPS = ("SSCD > 0.75", "SSCD <= 0.75")
GROUP_LABELS = {GROUPS[0]: r"SSCD $> 0.75$", GROUPS[1]: r"SSCD $\leq 0.75$"}
GROUP_COLORS = {GROUPS[0]: "#e5ba26", GROUPS[1]: "#563580"}
NORMALIZATION = (
    "Latent Euclidean (L2) distance divided by sqrt(d), equal to coordinate RMSE."
)
PROMPT_WEIGHTING = "Each represented prompt has equal mass within the outcome group, divided over its structurally eligible member seeds. IQR bands are descriptive, not confidence intervals."


AXIS_NORMALIZATION = {
    "initial_recovery_within_prompt": "x: within-prompt centered difference of latent L2 target errors / sqrt(d). y: within-prompt centered SSCD, dimensionless.",
    "initial_injection_geometry": "Dimensionless target-direction coefficient and perpendicular norm / target-direction norm.",
    "initial_target_coordinates": "Dimensionless target-direction coordinates <m_b-mu,v>/||v||^2.",
    "initial_injection_mismatch": "Both axes are dimensionless: terminal SSCD and relative full-vector injection mismatch.",
    "terminal_cancellation": "Both axes are dimensionless: error-guidance cosine and observed terminal error / observed triangle bound.",
}


# Presentation conventions only; saved scientific fields retain their identities.
AXIS_NOTATION = (
    "Notation follows revised.pdf Eqs. 2, 8-14 and Appendix A: hat{x}_t(b) is "
    "the learned clean estimate, b is c or the null input, Delta_t is their vector "
    "difference, and omitting an input means evaluation at x_t. T denotes the "
    "initial prediction, not stored row index zero. e_t(c) denotes the measured "
    "conditional target distance via Eq. 10's single-target reference. Dividing "
    "a displayed raw norm by sqrt(d) describes the existing coordinate RMSE; "
    "saved numbers are not divided again. Vector symbols x, z and Delta are bold "
    "upright, mu is bold italic, and hats/bars cover the bold x; scalar symbols "
    "and indices remain ordinary weight, matching Appendix A's printed fonts."
)
CANDIDATE_NOTATION = (
    "Figure notation uses p_t, bar{x}_t and e_t without a support marker: every "
    "displayed posterior/reference uses the same declared candidate law. Its "
    "support and masses remain recorded in the saved measurement provenance. "
    "In the candidate-reference diagnostics, mu denotes this law's mean; in "
    "the initial concentration and injection figures it denotes the fixed "
    "independent center specified by their saved measurement contracts."
)
AXIS_NOTES = {
    "initial_unconditional_concentration": "The norm placeholder is the initial unconditional estimate or a distinct candidate atom, as identified by the two curves; both use the same fixed mu.",
    "initial_recovery_within_prompt": "Centered means subtracting the mean over the same eligible seeds within each prompt, for each axis separately, before outcome grouping.",
    "posterior_feedback_positive_fraction": CANDIDATE_NOTATION + " The event compares two inputs at the same destination time. The line counts only uncertainty-resolved strict increases; the band adds unresolved mass.",
    "posterior_feedback_initial_comparison": CANDIDATE_NOTATION + " logit(p)=log[p/(1-p)]; these are first-update endpoints at T-1, not log probabilities.",
    "posterior_feedback_normalized_gain": CANDIDATE_NOTATION + " Normalized log-odds gain has no named manuscript symbol, so its descriptive axis label is retained; its exact normalization is recorded above.",
    "posterior_feedback_increasing_profile_fraction": CANDIDATE_NOTATION + " z_t(s)=x_{t-1}^{cf}+s g kappa_t Delta_t, s in [0,1], is Eq. 66. Increasing refers to candidate target log odds throughout this segment, certified by the resolved strictly positive guided-endpoint derivative; the band adds unresolved classifications.",
    "joint_target_recovery_early": "The maximum is over b in {c, null}; t is the saved early prediction snapshot below. This is the paired maximum before aggregation.",
    "joint_target_recovery_late": "The maximum is over b in {c, null}; t is the saved late prediction snapshot below. This is the paired maximum before aggregation.",
    "terminal_bound_tightness": CANDIDATE_NOTATION + " Bound means the unnormalized L2 bound identified by each legend entry; dividing both bound and error by sqrt(d) leaves the displayed ratio unchanged.",
    "initial_candidate_reference_discrepancy": CANDIDATE_NOTATION,
    "initial_reference_snr_sweep": CANDIDATE_NOTATION + " Here t parametrizes the analytical noise schedule at fixed x_T; the input is held fixed, unlike a generated trajectory. The vertical line marks the generation initialization SNR_T.",
    "initial_target_coordinates": "Both axes are scalar projections onto x_star-mu; they do not establish full-vector agreement.",
    "initial_injection_mismatch": "The positive guidance factor g cancels from numerator and denominator of the saved relative injection mismatch.",
    "posterior_feedback_initial_dose": CANDIDATE_NOTATION + " z_T(s) is the matched segment of Eq. 66. The saved lambda column is exactly the manuscript segment coordinate s; 0, 1/g and 1 identify matched unconditional, matched conditional and actual guidance endpoints.",
    "posterior_feedback_coverage_audit": CANDIDATE_NOTATION + " Derived margins have no manuscript symbols; their descriptive legend labels remain distinct from the original Proposition 5 condition.",
    "synchronization_bound_components": CANDIDATE_NOTATION,
    "last_prediction_error_terms": "Here t is the last stored prediction; this label does not assert the clean final-update prerequisite or identify its input as a clean endpoint.",
}


def _entry(legacy, stem, section, kind, question, x, y, formula, columns, **extra):
    return {
        "legacy_design_id": legacy,
        "stem": stem,
        "category": section,
        "kind": kind,
        "semantic_question": question,
        "source_table": f"plot_data/{stem}.csv",
        "required_columns": list(columns),
        "formula_version": FORMULA_VERSION,
        "formula": formula,
        "normalization": AXIS_NORMALIZATION.get(stem, NORMALIZATION),
        "group_rule": (
            "All eligible saved sample pairs, without an outcome split; continuous paired-target terminal SSCD colors when shown."
            if kind == "scatter"
            else "Same-seed paired-target terminal SSCD > 0.75 versus <= 0.75; groups can share prompts."
        ),
        "weighting": PROMPT_WEIGHTING,
        "axes": {"x": x, "y": y},
        "axis_notation": AXIS_NOTATION + " " + AXIS_NOTES.get(stem, ""),
        "band_definition": "Saved descriptive interquartile range, when present.",
        "snapshot_specification": None,
        "required_applicability": "validated saved numerical inputs",
        "outputs": {
            extension: f"{section}/{stem}.{extension}" for extension in ("png", "pdf")
        },
        "caption_key": stem,
        **extra,
    }


CORE = [
    _entry(
        "IR01",
        "initial_recovery",
        "main",
        "scatter",
        "Is the initial conditional estimate closer to the paired target?",
        '$\\|\\hat{\\mathbf{x}}_T(\\varnothing)-\\mathbf{x}^\\star\\|/\\sqrt{d}$',
        '$\\|\\hat{\\mathbf{x}}_T(c)-\\mathbf{x}^\\star\\|/\\sqrt{d}$',
        "E_b,0=||m_b,0-x_star||_2/sqrt(d); x=E_u,0; y=E_c,0; m_b,k=(x_k-sigma_k epsilon_b,k)/alpha_k.",
        ["x", "y", "terminal_sscd"],
        equal=True,
        sscd=True,
        reference="Equal target errors",
    ),
    _entry(
        "UB02",
        "initial_unconditional_concentration",
        "main",
        "ecdf",
        "How concentrated are unique initial unconditional estimates around the fixed reference center compared with distinct candidate atoms?",
        '$\\|\\,\\cdot-\\boldsymbol{\\mu}\\|/\\sqrt{d}$',
        'CDF',
        "Initial: ||m_u,0-mu||_2/sqrt(d); candidate atoms: ||u_j-mu||_2/sqrt(d), about exactly the same fixed independent mu.",
        ["value", "cdf", "distribution"],
        group_rule="Separate unique evaluation seeds and distinct candidate atoms.",
        weighting="Separate equal-mass ECDFs over unique evaluation run/seed pairs and distinct candidate atoms.",
    ),
    _entry(
        "PF03",
        "posterior_feedback_positive_fraction",
        "main",
        "fraction",
        "How often does the matched candidate target probability strictly increase?",
        '$\\mathrm{SNR}_t$',
        'Fraction with $p_{t-1}>p_{t-1}(\\mathbf{x}_{t-1}^{\\mathrm{cf}})$',
        "f_plus=sum(w_i 1[resolved positive G_i])/sum(structurally eligible w_i); upper=f_plus+sum(unresolved w_i)/sum(eligible w_i). G=Lambda(1)-Lambda(0) at one destination noise level and one fixed candidate law.",
        ["group", "step_index", "snr", "fraction", "upper_fraction"],
        band_definition="Possible additional unresolved numerical-sign mass on the same structurally eligible denominator; not a statistical confidence interval.",
        normalization="Dimensionless weighted fraction.",
    ),
    _entry(
        "TS01",
        "target_synchronization",
        "main",
        "curve",
        "How do the paired conditional and unconditional target errors evolve?",
        '$\\mathrm{SNR}_t$',
        '$\\|\\hat{\\mathbf{x}}_t(b)-\\mathbf{x}^\\star\\|/\\sqrt{d}$',
        "Saved weighted median and quartiles of E_b,k=||m_b,k-x_star||_2/sqrt(d), for b=conditional/unconditional and both fixed outcome groups.",
        ["group", "branch", "step_index", "snr", "median", "q25", "q75"],
        branches=True,
    ),
    _entry(
        "IR03",
        "initial_recovery_within_prompt",
        "appendix",
        "scatter",
        "Within a prompt, is initial paired improvement associated with terminal SSCD?",
        'Centered $[\\|\\hat{\\mathbf{x}}_T(\\varnothing)-\\mathbf{x}^\\star\\|-e_T(c)]/\\sqrt{d}$',
        'Centered SSCD',
        "x=(E_u,0-E_c,0)-mean_eligible_seed(E_u,0-E_c,0); y=SSCD-mean_same_eligible_seed(SSCD), centered before outcome grouping. The displayed weighted descriptive trend is saved during analysis.",
        ["x", "y"],
        zero_guides=True,
        saved_trend=True,
        group_rule="Within-prompt centering uses the saved common eligible-seed population before any outcome grouping; all saved centered pairs are shown.",
    ),
    _entry(
        "IR04",
        "initial_target_retrieval_rank",
        "appendix",
        "ecdf",
        "Does conditioning identify the paired candidate atom in each outcome group?",
        "Initial target retrieval rank",
        'Weighted CDF',
        "rank_b=1+sum_j 1[||m_b,0-u_j||_2<||m_b,0-x_star||_2]; distinct candidate atoms, strict closer-than rank, exact ties retained.",
        ["value", "cdf", "group", "branch"],
        branches=True,
        normalization="Dimensionless Euclidean retrieval rank.",
    ),
    _entry(
        "PF04",
        "posterior_feedback_initial_comparison",
        "appendix",
        "scatter",
        "At the first update, how does guidance change matched candidate target log odds?",
        '$\\mathrm{logit}\\,p_{T-1}(\\mathbf{x}_{T-1}^{\\mathrm{cf}})$',
        '$\\mathrm{logit}\\,p_{T-1}(\\mathbf{x}_{T-1})$',
        "x=Lambda(0)=logit p_next^K(z0); y=Lambda(1)=logit p_next^K(z0+h); both endpoints use the same candidate bank, masses, paired target and destination noise.",
        ["x", "y", "terminal_sscd"],
        equal=True,
        sscd=True,
        reference="Equal candidate evidence",
        snapshot_specification={"step_index": 0},
        normalization="Dimensionless natural-log odds.",
    ),
    _entry(
        "PF02",
        "posterior_feedback_normalized_gain",
        "appendix",
        "curve",
        "How strong is the signed matched log-odds gain per local displacement?",
        "$\\mathrm{SNR}_t$",
        "Normalized candidate log-odds gain",
        "G_tilde=[Lambda(1)-Lambda(0)]/[beta ||h||_2 sqrt(d)], beta=alpha_next/sigma_next^2, h=g kappa Delta. Zero or unresolved direction denominators remain undefined.",
        ["group", "step_index", "snr", "median", "q25", "q75"],
        signed=True,
        normalization="Matched log-odds gain / (destination beta * displacement L2 norm * sqrt(d)).",
    ),
    _entry(
        "PF08",
        "posterior_feedback_increasing_profile_fraction",
        "appendix",
        "fraction",
        "How often do candidate target log odds increase throughout the matched segment?",
        '$\\mathrm{SNR}_t$',
        'Fraction with increasing $p_{t-1}(\\mathbf{z}_t(s))$',
        "Resolved increasing uses guided-endpoint C(1)>its saved arithmetic uncertainty under validated concavity; line=sum(increasing weights)/sum(eligible weights), upper adds unresolved classification weights on the same denominator.",
        ["group", "step_index", "snr", "fraction", "upper_fraction"],
        band_definition="Possible additional unresolved classification mass; not a statistical confidence interval.",
        normalization="Dimensionless weighted fraction.",
    ),
    _entry(
        "TS02",
        "branch_gap",
        "appendix",
        "curve",
        "How far apart are the actual paired clean-estimate vectors?",
        '$\\mathrm{SNR}_t$',
        '$\\|\\mathbf{\\Delta}_t\\|/\\sqrt{d}$',
        "D_k=||m_c,k-m_u,k||_2/sqrt(d), not |E_c,k-E_u,k|; saved weighted median and IQR within each outcome group.",
        ["group", "step_index", "snr", "median", "q25", "q75"],
    ),
    _entry(
        "TS03",
        "joint_target_recovery_early",
        "appendix",
        "ecdf",
        "How close are both branch estimates to the target at the fixed early snapshot?",
        '$\\mathrm{max}_b\\|\\hat{\\mathbf{x}}_t(b)-\\mathbf{x}^\\star\\|/\\sqrt{d}$',
        'Weighted CDF',
        "Q_k=max(E_c,k,E_u,k), computed per paired sample before aggregation; exact weighted empirical CDF. Early and late plots share full-support limits across both groups.",
        ["value", "cdf", "group"],
        snapshot_specification={
            "policy": "recorded normalized snapshot policy; fallback floor(0.2*T) clipped to prediction range",
            "T50_step_index": 10,
        },
        shared_limits=True,
    ),
    _entry(
        "TS03",
        "joint_target_recovery_late",
        "appendix",
        "ecdf",
        "How close are both branch estimates to the target at the fixed late snapshot?",
        '$\\mathrm{max}_b\\|\\hat{\\mathbf{x}}_t(b)-\\mathbf{x}^\\star\\|/\\sqrt{d}$',
        'Weighted CDF',
        "Q_k=max(E_c,k,E_u,k), computed per paired sample before aggregation; exact weighted empirical CDF. Early and late plots share full-support limits across both groups.",
        ["value", "cdf", "group"],
        snapshot_specification={
            "policy": "recorded normalized snapshot policy; fallback max(0,T-2)",
            "T50_step_index": 48,
        },
        shared_limits=True,
    ),
]
TERMINAL = [
    _entry(
        "TR01",
        "terminal_error_terms",
        "appendix",
        "scatter",
        "What conditional-error and residual-guidance terms enter a verified clean final update?",
        '$e_1(c)/\\sqrt{d}$',
        '$(g-1)\\|\\mathbf{\\Delta}_1\\|/\\sqrt{d}$',
        "A=||m_c,last-x_star||_2/sqrt(d); B=(g-1)||Delta_last||_2/sqrt(d). The theorem interpretation requires independently verified structural a=0,kappa=1 and numerical reconstruction.",
        ["x", "y", "terminal_sscd"],
        conditional_terminal=True,
        sscd=True,
        required_applicability="clean_update_verified",
        independent_tolerance=True,
    ),
    _entry(
        "TR02",
        "terminal_bound_tightness",
        "appendix",
        "ecdf",
        "How tight are the distinct observed triangle and candidate-reference bounds when the final update is clean?",
        'Bound / $\\|\\mathbf{x}_0-\\mathbf{x}^\\star\\|$',
        'Weighted CDF',
        "Observed bound=A+B. Candidate bound=[g e_c+(g-1)(e_u^K+R_K(1-p^K))]/sqrt(d). Each is divided by observed terminal RMSE only when nonzero; zero-error statuses are counted without smoothing.",
        ["value", "cdf", "distribution"],
        conditional_terminal=True,
        required_applicability="clean_update_verified",
        normalization="Dimensionless bound / observed RMSE ratio.",
        reference_value=1.0,
        reference="Equal bound and error",
    ),
]


def _diagnostic(legacy, stem, kind, question, x, y, formula, columns, **extra):
    return _entry(
        legacy, stem, "diagnostics", kind, question, x, y, formula, columns, **extra
    )


DIAGNOSTICS = [
    _diagnostic(
        "UB03",
        "initial_candidate_reference_discrepancy",
        "scatter",
        "How do initial candidate-mean movement and learned-reference discrepancy compare?",
        '$\\|\\bar{\\mathbf{x}}_T(\\varnothing)-\\boldsymbol{\\mu}\\|/\\sqrt{d}$',
        '$e_T(\\varnothing)/\\sqrt{d}$',
        "x=||bar_x_0^K-mu_K||_2/sqrt(d); y=||m_u,0-bar_x_0^K||_2/sqrt(d).",
        ["x", "y"],
        group_rule="One initial observation per unique evaluation run/seed; no outcome split.",
        weighting="Unaggregated unique initial evaluation seeds; no prompt replication or outcome reweighting.",
    ),
    _diagnostic(
        "UB04",
        "initial_reference_snr_sweep",
        "curve",
        "How does the fixed analytical candidate reference approach its own mean?",
        '$\\mathrm{SNR}_t$',
        '$\\|\\bar{\\mathbf{x}}_t(\\mathbf{x}_T,\\varnothing)-\\boldsymbol{\\mu}\\|/\\sqrt{d}$',
        "Fixed cached initial-noise seeds and the saved prespecified analytical SNR sweep; ||bar_x_r^K(x_initial)-mu_K||_2/sqrt(d). This is not a neural-network sweep.",
        ["analytical_snr", "median", "q25", "q75"],
        analytical_snr=True,
        group_rule="The same unique initial-noise evaluation seeds at every fixed analytical SNR; no outcome split.",
        weighting="Equal weight over unique evaluation seeds at each analytical SNR.",
    ),
    _diagnostic(
        "IA01",
        "initial_injection_geometry",
        "scatter",
        "How aligned is the full initial injection with the target direction?",
        '$\\langle g\\mathbf{\\Delta}_T,\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\rangle/\\|\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\|^2$',
        'Relative perpendicular component',
        "v=x_star-mu; J=g Delta_0; x=<J,v>/||v||^2; y=||J-proj_v J||/||v||, with full-vector mismatch diagnostics retained.",
        ["x", "y", "terminal_sscd"],
        sscd=True,
    ),
    _diagnostic(
        "IA02",
        "initial_target_coordinates",
        "scatter",
        "How do conditional and guided target-direction coordinates compare?",
        '$\\langle\\hat{\\mathbf{x}}_T(c)-\\boldsymbol{\\mu},\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\rangle/\\|\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\|^2$',
        '$\\langle\\hat{\\mathbf{x}}_T(c;g)-\\boldsymbol{\\mu},\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\rangle/\\|\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\|^2$',
        "Saved target-direction coordinates; coordinate agreement alone does not establish full-vector agreement, whose mismatch diagnostics remain in the saved inputs and audit.",
        ["x", "y", "terminal_sscd"],
        sscd=True,
    ),
    _diagnostic(
        "IA03",
        "initial_injection_mismatch",
        "scatter",
        "How does full-vector initial injection mismatch vary with terminal SSCD?",
        'SSCD',
        '$\\|\\mathbf{\\Delta}_T-(\\mathbf{x}^\\star-\\boldsymbol{\\mu})\\|/\\|\\mathbf{x}^\\star-\\boldsymbol{\\mu}\\|$',
        "||g Delta_0-g(x_star-mu)||/||g(x_star-mu)||; status retains degenerate target directions.",
        ["x", "y"],
    ),
    _diagnostic(
        "PF06",
        "posterior_feedback_initial_dose",
        "dose",
        "What is the matched candidate gain along the first update segment?",
        '$s$',
        '$\\log[p_{T-1}(\\mathbf{z}_T(s))/p_{T-1}(\\mathbf{z}_T(0))]$',
        "Saved H(lambda)=log p_next^K(z0+lambda h)-log p_next^K(z0), at fixed update 0; reference lambda=0,1/g,1; no counterfactual terminal SSCD.",
        ["lambda", "group", "median", "q25", "q75"],
        signed=True,
        snapshot_specification={"step_index": 0},
    ),
    _diagnostic(
        "PF07",
        "posterior_feedback_coverage_audit",
        "coverage",
        "How do original and derived positive-margin coverage compare on the same eligible population?",
        "$\\mathrm{SNR}_t$",
        "Positive coverage",
        "Original M0 and derived M1/M2/M3 retain separate resolved/estimated status and strict-positive comparisons to observed gain on one structurally eligible weighted population.",
        [
            "snr",
            "margin",
            "resolved_positive_fraction",
            "estimated_strict_positive_fraction",
            "stable_gain_positive_fraction",
        ],
        group_rule="Saved all-eligible-population diagnostic; outcome-specific counts remain in its compact table.",
        normalization="Dimensionless weighted fraction.",
    ),
    _diagnostic(
        "TS05",
        "synchronization_bound_components",
        "curve",
        "How do vector branch gap and individual synchronization-bound terms evolve?",
        "$\\mathrm{SNR}_t$",
        "Branch gap and terms / $\\sqrt{d}$",
        "Saved D/sqrt(d), e_c/sqrt(d), e_u^K/sqrt(d), R_K(1-p^K)/sqrt(d). Valid last-current references remain available even with excluded destination posterior.",
        ["snr", "metric", "median", "q25", "q75"],
        group_rule="Saved all-population diagnostic curves; outcome-specific summaries remain in the compact CSV.",
    ),
    _diagnostic(
        "MD01",
        "matched_update_residual",
        "curve",
        "How large is the independent scheduler-vector replay residual?",
        "Stored update index",
        "Replay residual / $\\sqrt{d}$",
        "Saved independent deterministic replay residual and source-precision update envelope. Recovered DDPM innovation is not independent replay.",
        ["step_index", "metric", "median"],
        update_index=True,
        group_rule="Saved all-population diagnostic curves; outcome-specific summaries remain in the compact CSV.",
    ),
    _diagnostic(
        "TR01",
        "last_prediction_error_terms",
        "scatter",
        "What error geometry exists at the last stored prediction without asserting a clean terminal theorem?",
        '$e_t(c)/\\sqrt{d}$',
        '$(g-1)\\|\\mathbf{\\Delta}_t\\|/\\sqrt{d}$',
        "A=||m_c,last-x_star||/sqrt(d), B=(g-1)||Delta_last||/sqrt(d); prediction geometry only, without a terminal certificate.",
        ["x", "y", "terminal_sscd"],
        sscd=True,
    ),
    _diagnostic(
        "TR-cancellation",
        "terminal_cancellation",
        "scatter",
        "How does vector cancellation relate to observed clean-terminal error?",
        '$\\cos\\angle(\\hat{\\mathbf{x}}_1(c)-\\mathbf{x}^\\star,\\mathbf{\\Delta}_1)$',
        '$\\|\\mathbf{x}_0-\\mathbf{x}^\\star\\|/[e_1(c)+(g-1)\\|\\mathbf{\\Delta}_1\\|]$',
        "x=<m_c-x_star,Delta>/(||m_c-x_star|| ||Delta||); y=||x_out-x_star||/[||m_c-x_star||+(g-1)||Delta||]. Clean-update prerequisite verified independently; zero denominators remain undefined.",
        ["x", "y"],
    ),
]


def paper_registry(diagnostics=False):
    """Return fixed entry dictionaries; unavailable entries retain metadata."""
    return copy.deepcopy(CORE + TERMINAL + (DIAGNOSTICS if diagnostics else []))
