"""Fixed paper figure definitions, independent of measured effect directions."""

from __future__ import annotations

import copy

REGISTRY_VERSION = "four-stage-evidence-1"
FORMULA_VERSION = "revised-fixed-evidence-2"
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


LEGACY_CORE = [
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
LEGACY_TERMINAL = [
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


LEGACY_DIAGNOSTICS = [
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


DIRECT_REFERENCE = (
    "One declared finite law D_K supplies mu_K, bar{x}^K, e_u^K, p^K and R_K. "
    "Cached-target mode uses every compatible complete preselection target and "
    "equal mass per distinct atom; manifest mode uses its explicit atoms and weights. "
    "K does not identify the checkpoint training distribution. Legacy model-output "
    "centers never set mu_K. All quantities in a comparison use the same law."
)
DIRECT_ASSUMPTIONS = [
    "Equation 6 is an assumed single-target conditional idealization, not inferred from SSCD.",
    "Reference atoms, masses, preprocessing and target membership are recorded.",
    "Native schedule and canonical prediction conversion retain their saved identities.",
]
RESULT_LABELS = {
    1: ("Theorem 1", "thm:initial_conditional_recovery"),
    2: ("Lemma 2", "lem:initial_unconditional_baseline"),
    3: ("Corollary 3", "cor:initial_cfg_amplification"),
    4: ("Lemma 4", "lem:matched_one_step_cfg_displacement"),
    5: ("Proposition 5", "prop:matched_one_step_target_posterior_feedback"),
    6: ("Lemma 6", "lem:target_specific_synchronization"),
    7: ("Theorem 7", "thm:final_reproduction"),
}


def _direct(number, stem, category, question, x, y, x_definition, y_definition,
            source, interpretation, proof, *, kind="scatter", **extra):
    label, manuscript_label = RESULT_LABELS[number]
    entry = _entry(
        label, stem, category, kind, question, x, y,
        "x=" + x_definition + "; y=" + y_definition + ". " + interpretation,
        ["x", "y", "marker_class"] if kind == "scatter" else
        ["metric", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum"],
        direct_statement=True, result_number=number, result_label=label,
        manuscript_label=manuscript_label,
        x_definition=x_definition, y_definition=y_definition,
        reference_law=DIRECT_REFERENCE, input_source=source,
        applicability_assumptions=list(DIRECT_ASSUMPTIONS),
        interpretation=interpretation, supporting_equation=proof,
        averaging_measure="Unaggregated retained scientific observations; shared seeds and repeated timesteps are dependent.",
        averaging_unit="Complete pair identity and Gaussian/generation seed with native step identity where present.",
        group_rule="All retained observations; SSCD only colors an exactly matched saved trajectory observation.",
        weighting="No outcome filtering, calibration fit, or separately aggregated bound components.",
        axis_notation=AXIS_NOTATION + " " + DIRECT_REFERENCE,
        band_definition="None unless explicitly specified by the saved measurement contract.",
        allow_unavailable=True,
    )
    entry.update(extra)
    if entry.get("required_loss"):
        entry["allow_unavailable"] = False
    return entry


CORE = [
    _direct(
        1, "theorem1_loss_recovery", "main",
        "How does independently measured forward loss compare with Gaussian initial recovery?",
        r"$\sqrt{\widehat{L}_T(c)/(d\,\mathrm{SNR}_T)}$", r"$e_T(\mathbf{Z},c)/\sqrt{d}$",
        "sqrt(loss_mean_squared_l2/(d*SNR_T))", "conditional_error_l2 on Gaussian Z / sqrt(d)",
        ["forward_target", "gaussian_probe"],
        "Finite Gaussian transfer comparison supporting an asymptotic statement; no pointwise equality or upper-bound line.",
        "Eqs. 7, 28, 33; Appendix D.1", sscd=True, include_zero=True,
        dependencies=["forward_loss_summary", "gaussian_conditional"], required_loss=True,
        averaging_measure="One pair-level forward-draw mean, repeated across that pair's Gaussian seeds.",
        averaging_unit="Loss MC sample size is draw_count, not the repeated horizontal-coordinate count.",
        snapshot_specification={"step_index": 0},
    ),
    _direct(
        2, "lemma2_unconditional_baseline", "main",
        "How do reference concentration, learned concentration and reference error behave on fixed Gaussian inputs?",
        r"$\mathrm{SNR}_t$", r"Distance / $\sqrt{d}$",
        "actual saved native SNR", "reference_mean_error, learned_mean_error and unconditional_reference_error, each L2/sqrt(d)",
        ["gaussian_probe"],
        "Separate reference conclusion, learned conclusion and additional error condition; noninitial inputs are Gaussian probes, not reverse states.",
        "Lemma 2; Appendix D.2", kind="curve", direct_metrics=True,
        dependencies=["gaussian_reference"],
        averaging_measure="Equal mass over unique run/seed Gaussian probes at each native timestep, with the same seeds throughout.",
        averaging_unit="Unique run/seed; no prompt replication or SSCD grouping.",
        band_definition="Descriptive seed-level interquartile range, not a confidence interval.",
    ),
    _direct(
        3, "corollary3_initial_cfg_amplification", "main",
        "How close is the full initial CFG injection to the first vector limit?",
        r"$g[e_T(c)+b_T]/\sqrt{d}$", r"$\|g\mathbf{\Delta}_T-g(\mathbf{x}^{\star}-\boldsymbol{\mu}_K)\|/\sqrt{d}$",
        "direct_cor3_injection_bound_rmse = g*(e_T(c)+b_T)/sqrt(d)",
        "direct_cor3_injection_discrepancy_rmse = ||g*Delta_T-g*(x_star-mu_K)||/sqrt(d)",
        ["generated_state"],
        "Finite triangle bound derived from CFG for the corollary's first vector limit; compliance alone does not show a small discrepancy.",
        "Eq. 3; Appendix D.3", equal=True, include_zero=True,
        reference="Triangle upper bound", sscd=True, dependencies=["initial"],
        auxiliary_definitions="b_T=||hat{x}_T(empty)-mu_K||; J_T=g Delta_T.",
    ),
    _direct(
        4, "lemma4_matched_displacement", "main",
        "Does the actual matched update displacement equal its CFG prediction?",
        r"$g\kappa_t\|\mathbf{\Delta}_t\|/\sqrt{d}$",
        r"$\|\mathbf{x}_{t-1}-\mathbf{x}_{t-1}^{\mathrm{cf}}\|/\sqrt{d}$",
        "direct_lemma4_predicted_rmse", "direct_lemma4_displacement_rmse",
        ["generated_state"],
        "Exact affine-update identity check; independently replayed and constructed rows retain separate marker classes.",
        "Eqs. 4, 12, 55-57", equal=True, include_zero=True,
        reference="Update identity", sscd=True, dependencies=["matched_updates"],
    ),
    _direct(
        5, "proposition5_posterior_feedback", "main",
        "Does the original sufficient condition imply matched target-posterior gain?",
        r"$[\|\mathbf{\Delta}_t\|-e_t(c)-e_t^K(\varnothing)-\mathcal{V}_t^K]/\sqrt{d}$",
        r"$\log[p_{t-1}^K(\mathbf{x}_{t-1})/p_{t-1}^K(\mathbf{x}_{t-1}^{\mathrm{cf}})]$",
        "direct_prop5_margin_rmse using the original full norm-variation integral",
        "direct_prop5_log_probability_gain from independently evaluated matched endpoints",
        ["generated_state"],
        "Original condition-versus-conclusion implication; quadrature signs are estimates, not certified integral enclosures.",
        "Eq. 15; Eqs. 66, 69-73", zero_guides=True, signed=True, sscd=True,
        dependencies=["matched_updates", "original_variation_integration"],
        normalization="x: latent L2/sqrt(d); y: dimensionless natural-log probability gain; different units.",
    ),
    _direct(
        6, "lemma6_target_specific_synchronization", "main",
        "How tight is the upper bound on the target-specific vector branch gap?",
        r"$[e_t(c)+e_t^K(\varnothing)+R_K(1-p_t^K)]/\sqrt{d}$",
        r"$\|\mathbf{\Delta}_t\|/\sqrt{d}$",
        "direct_lemma6_rhs_rmse formed per observation", "direct_lemma6_gap_rmse of the paired vector difference",
        ["generated_state"],
        "Upper bound at every positive-current-noise prediction including the last; absolute values and tightness remain visible.",
        "Lemma 6; Eqs. 77-80", equal=True, include_zero=True,
        reference="Lemma 6 upper bound", sscd=True, dependencies=["trajectory"],
    ),
    _direct(
        7, "theorem7_final_reproduction", "main",
        "Does the last-prediction bound control the actual endpoint when the clean-update prerequisite holds?",
        r"$[g e_1(c)+(g-1)(e_1^K(\varnothing)+R_K(1-p_1^K))]/\sqrt{d}$", r"$\|\mathbf{x}_0-\mathbf{x}^{\star}\|/\sqrt{d}$",
        "direct_theorem7_bound_rmse = [g*e_1(c)+(g-1)*(e_1^K(empty)+R_K*(1-p_1^K))]/sqrt(d)",
        "direct_theorem7_endpoint_error_rmse from actual saved latents[:,T]",
        ["generated_state"],
        "Theorem upper bound only for independently verified clean updates. Well-defined inapplicable pairs remain separately marked; no alternate bound or clean projection substitutes for the endpoint.",
        "Eqs. 5, 16, 81-84", equal=True, include_zero=True,
        reference="Theorem 7 bound (applicable)", sscd=True,
        terminal_applicability=True, dependencies=["terminal", "last_current_reference"],
        auxiliary_definitions="B_7^K=g e_1(c)+(g-1)[e_1^K(empty)+R_K(1-p_1^K)], shorthand for the exact theorem RHS.",
    ),
]


APPENDIX = [
    _direct(1, "theorem1_squared_loss_recovery", "appendix", "What is the Gaussian comparison on the squared scale?",
            r"$\widehat{L}_T(c)/(d\,\mathrm{SNR}_T)$", r"$e_T(\mathbf{Z},c)^2/d$",
            "loss_mean_squared_l2/(d*SNR_T)", "conditional_error_squared_l2/d", ["forward_target", "gaussian_probe"],
            "Squared-scale finite Gaussian comparison without an equality line.", "Eqs. 7, 33",
            sscd=True, include_zero=True, required_loss=True, normalization="Both axes are squared latent L2/d."),
    _direct(1, "theorem1_forward_identity", "appendix", "Do independently evaluated forward clean errors satisfy the loss identity?",
            r"$\widehat{L}_T(c)/(d\,\mathrm{SNR}_T)$", r"$\widehat{\mathbb{E}}_{\mathrm{fwd}}[e_T^2]/d$",
            "loss_mean_squared_l2/(d*SNR_T)", "forward_clean_error_mean_squared_l2/d", ["forward_target"],
            "Forward-input identity and implementation check, separate from Gaussian transfer.", "Eq. 28",
            equal=True, include_zero=True, reference="Forward-input identity", required_loss=True,
            normalization="Squared latent L2/d; one mean over independent forward draws per pair."),
    _direct(1, "theorem1_gaussian_transfer", "appendix", "How does Gaussian failure frequency compare with the plug-in probability bound?",
            "Plug-in upper bound", r"$\widehat{\Pr}_{\mathbf{Z}}(e_T>\rho\sqrt{d})$",
            "min(1,loss_mean_squared_l2/(SNR_T*d*rho^2)+pinsker_tv_bound)",
            "Gaussian failure frequency at every fixed rho in {0.1,0.2,0.3,0.5,1.0}", ["forward_target", "gaussian_probe"],
            "Plug-in empirical check, not a rigorous statistical certificate from finite unbounded-loss estimates. Raw unclipped bounds remain saved.", "Eqs. 31-33",
            equal=True, include_zero=True, reference="Equal frequency and plug-in bound", required_loss=True,
            normalization="Dimensionless probabilities; raw tolerance r=rho*sqrt(d).", frequency_intervals=True),
    _direct(2, "lemma2_reference_to_learned_bound", "appendix", "Does the finite reference decomposition control learned concentration?",
            r"$[e_t^K(\mathbf{Z},\varnothing)+\|\bar{\mathbf{x}}_t^K(\mathbf{Z},\varnothing)-\boldsymbol{\mu}_K\|]/\sqrt{d}$",
            r"$\|\hat{\mathbf{x}}_t(\mathbf{Z},\varnothing)-\boldsymbol{\mu}_K\|/\sqrt{d}$",
            "baseline_bound_rmse", "learned_mean_error_rmse", ["gaussian_probe"],
            "Finite triangle bound on unique Gaussian probes; no duplicate prompt rows.", "Appendix D.2",
            equal=True, include_zero=True, reference="Triangle upper bound"),
    _direct(2, "lemma2_excess_loss_gaussian_error", "appendix", "How does direct marginal excess loss compare with Gaussian reference error?",
            r"$\widehat{L}^{\mathrm{ex}}_T/(d\,\mathrm{SNR}_T)$", r"$e_T^K(\mathbf{Z},\varnothing)^2/d$",
            "mean(excess_loss_squared_l2)/(d*SNR_T)", "unconditional_reference_error_l2^2/d on Gaussian probes",
            ["forward_marginal", "gaussian_probe"],
            "Different input laws; no Gaussian equality line. Direct nonnegative excess loss is not total-minus-optimal Monte Carlo loss.", "Appendix C",
            optional_measurement=True, include_zero=True, normalization="Squared latent L2/d; prompt-independent marginal draws and Gaussian probes."),
    _direct(3, "corollary3_guided_estimate_error", "appendix", "How close is the guided estimate to the second vector limit?",
            r"$[g e_T(c)+(g-1)b_T]/\sqrt{d}$",
            r"$\|\hat{\mathbf{x}}_T(c;g)-[\boldsymbol{\mu}_K+g(\mathbf{x}^{\star}-\boldsymbol{\mu}_K)]\|/\sqrt{d}$",
            "direct_cor3_guided_bound_rmse", "direct_cor3_guided_discrepancy_rmse", ["generated_state"],
            "Finite triangle comparison for the second vector limit, retaining distinct g and g-1 coefficients.", "Eq. 3; Appendix D.3",
            equal=True, include_zero=True, reference="Triangle upper bound", sscd=True),
    _direct(4, "lemma4_vector_residual", "appendix", "What vector residual remains relative to recorded precision?",
            r"Source precision / $\sqrt{d}$", r"$\|\mathbf{x}_{t-1}-\mathbf{x}_{t-1}^{\mathrm{cf}}-g\kappa_t\mathbf{\Delta}_t\|/\sqrt{d}$",
            "direct_lemma4_source_tolerance_rmse", "direct_lemma4_vector_residual_rmse", ["generated_state"],
            "Numerical vector-identity residual; constructed rows are not independent sampling verification.", "Eqs. 55-57",
            include_zero=True, auxiliary_definitions="r_t^vec=(x_next-x_cf)-g*kappa_t*Delta_t."),
    _direct(6, "lemma6_posterior_concentration", "appendix", "Does the whole-support radius control reference-to-target distance?",
            r"$R_K(1-p_t^K)/\sqrt{d}$", r"$\|\bar{\mathbf{x}}_t^K(\varnothing)-\mathbf{x}^{\star}\|/\sqrt{d}$",
            "direct_radius_tail_rmse", "direct_reference_target_error_rmse", ["generated_state"],
            "Upper bound with the full-law maximum radius and stable complement at every current prediction.", "Eq. 78",
            equal=True, include_zero=True, reference="Posterior concentration bound", sscd=True),
    _direct(7, "theorem7_certification", "appendix", "How often does a predeclared sufficient condition imply actual proximity?",
            "Certified fraction", "Actual proximity fraction",
            "fraction of applicable rows with raw B7<=tau", "fraction of the same applicable rows with raw endpoint_error<=tau",
            ["generated_state"],
            "Sufficient-event check restricted to applicable rows and independently declared raw latent tolerances; SSCD never sets tau.", "Theorem 7; Definition 1",
            equal=True, include_zero=True, reference="Equal event frequencies", optional_measurement=True,
            normalization="Dimensionless fractions; both events use the same predeclared raw latent L2 tolerance."),
]


def direct_audit_registry(diagnostics=False):
    """Seven direct statements plus supporting checks; legacy designs require opt-in."""
    entries = copy.deepcopy(CORE + APPENDIX)
    if diagnostics:
        for entry in copy.deepcopy(LEGACY_CORE + LEGACY_TERMINAL + LEGACY_DIAGNOSTICS):
            entry["category"] = "diagnostics"
            entry["outputs"] = {ext: f"diagnostics/{entry['stem']}.{ext}" for ext in ("png", "pdf")}
            entry["direct_statement"] = False
            entry["interpretation"] = "Retained behavioral diagnostic, not a direct statement comparison."
            entries.append(entry)
    return entries


DIRECT_AUDIT_ENTRIES = {entry["stem"]: entry for entry in CORE + APPENDIX}


def _evidence(original, stem, category, kind, x, y, formula, columns, **metadata):
    """Keep manuscript attribution separate from a derived evidence design."""
    entry = copy.deepcopy(DIRECT_AUDIT_ENTRIES[original])
    for key in ("equal", "zero_guides", "terminal_applicability", "direct_metrics", "sscd", "signed", "frequency_intervals"):
        entry.pop(key, None)
    entry.update(stem=stem, category=category, kind=kind,
                 source_table=f"plot_data/{stem}.csv", caption_key=stem,
                 outputs={ext: f"{category}/{stem}.{ext}" for ext in ("png", "pdf")},
                 axes={"x": x, "y": y}, formula=formula, formula_version=FORMULA_VERSION,
                 required_columns=columns,
                 x_definition=x, y_definition=y, interpretation=formula,
                 evidence_classification="Fixed empirical consequence with a separately retained direct audit")
    entry.update(metadata)
    return entry


def _appendix_copy(original, stem):
    entry = copy.deepcopy(DIRECT_AUDIT_ENTRIES[original])
    entry.update(stem=stem, category="appendix", source_table=f"plot_data/{stem}.csv", caption_key=stem,
                 outputs={ext: f"appendix/{stem}.{ext}" for ext in ("png", "pdf")})
    return entry


EVIDENCE_MAIN = [
    _evidence("theorem1_loss_recovery", "theorem1_loss_recovery", "main", "pair_loss",
              r"$\sqrt{\widehat{L}_T(c)/(d\,\mathrm{SNR}_T)}$",
              r"$\sqrt{\widehat{\mathbb{E}}_n[e_T(\mathbf{Z}_n,c)^2]/d}$",
              "Pair-level independent forward-loss scale A_c, conditional Gaussian RMS Y_c, and matched unconditional target-error RMS U_c. Empirical RMS does not assert moment convergence.",
              ["x", "y", "control_y", "mean_terminal_sscd", "x_low", "x_high", "y_low", "y_high"],
              include_zero=True, colorbar_label="Mean terminal SSCD",
              group_rule="One conditional point and matched unconditional control per retained pair; no outcome grouping.",
              averaging_unit="Pair-level independent forward draws and matched Gaussian seeds; shared banks do not create independent prompts.",
              band_definition="Monte Carlo bootstrap intervals: pair-specific, seed 0, 1000 replicates, central 95%; not theorem confidence bounds.",
              interpretation="Finite-SNR empirical RMS supporting a convergence-in-probability statement; no equality line or fitted monotonic relation.",
              x_definition="sqrt(loss_mean_squared_l2/(d*SNR_T))",
              y_definition="sqrt(mean_n conditional_error_squared_l2/d), with matched unconditional target-error RMS control"),
    _evidence("lemma2_unconditional_baseline", "lemma2_unconditional_baseline", "main", "reference_convergence",
              r"$\mathrm{SNR}$", r"Distance / $\sqrt{d}$",
              "Reference-to-mu_K distance on fixed analytical and native Gaussian probes; learned-to-mu_K and learned-to-reference distances only at native labels.",
              ["metric", "source_range", "segment_id", "snr", "median", "q25", "q75", "minimum", "maximum"],
              include_zero=True, group_rule="Unique Gaussian seeds, no prompt replication or SSCD grouping.",
              band_definition="Descriptive seed interquartile range, not a confidence interval.",
              interpretation="Analytical reference convergence is separate from supported native checkpoint behavior; no observed network asymptotic limit is claimed.",
              input_source=["analytical_reference_only", "gaussian_probe"]),
    _evidence("corollary3_initial_cfg_amplification", "corollary3_initial_cfg_amplification", "main", "injection_geometry",
              r"$\langle\mathbf{J},\mathbf{v}\rangle/(g\|\mathbf{v}\|^2)$",
              r"$\|\mathbf{J}_{\perp}\|/(g\|\mathbf{v}\|)$",
              "v=x_star-mu_K, J=g Delta; q=<J,v>/(g||v||^2), r_perp=||J-gqv||/(g||v||). Distance from (1,0) equals the complete relative injection error.",
              ["x", "y", "terminal_sscd", "marker_class"],
              sscd=True, include_zero=True, geometry_levels=[.25, .5, 1.0],
              interpretation="Exact vector orthogonal decomposition, not a fitted coefficient or guidance-sweep experiment.",
              axis_notation=AXIS_NOTATION + " v=x_star-mu_K; J=g Delta; J_perp is computed as a vector. These are explicitly derived empirical coordinates.",
              normalization="Both coordinates dimensionless; ideal location (1,0)."),
    copy.deepcopy(DIRECT_AUDIT_ENTRIES["lemma4_matched_displacement"]),
    _evidence("proposition5_posterior_feedback", "proposition5_posterior_feedback", "main", "feedback_fractions",
              r"$\mathrm{SNR}_t$", "Fraction of eligible samples",
              "Resolved H>0 and resolved original M>0 for each fixed outcome group, on identical prompt-balanced eligible denominators including unresolved signs.",
              ["group", "metric", "step_index", "snr", "fraction", "upper_fraction", "denominator_weight"],
              normalization="Dimensionless weighted fractions; original norm-variation condition only.",
              band_definition="Numerical ambiguity mass from resolved to resolved-plus-unresolved fraction; not statistical uncertainty.",
              group_rule=PROMPT_WEIGHTING,
              interpretation="Empirical feedback prevalence and sufficient-condition coverage differ; no converse implication.",
              x_definition="Actual saved source SNR for each eligible transition",
              y_definition="Strict gain and strict original margin resolved fractions on one common weighted population"),
    _evidence("lemma6_target_specific_synchronization", "lemma6_target_specific_synchronization", "main", "synchronization_curves",
              r"$\mathrm{SNR}_t$", r"Target error or gap / $\sqrt{d}$",
              "Q=max(e_c,unconditional_target_error)/sqrt(d), D=||Delta||/sqrt(d), S=(e_c+e_u_K+R_K(1-p_K))/sqrt(d), before common-weight medians. D<=S is Lemma 6; Q<=S is a direct consequence.",
              ["group", "metric", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum"],
              include_zero=True, group_rule=PROMPT_WEIGHTING,
              band_definition="Descriptive IQR for joint target error Q in default rendering; all quantity quantiles saved.",
              interpretation="Wrong-latent agreement can have small D and positive Q; the bound is never assembled from separate medians.",
              x_definition="All actual current prediction SNRs including the last",
              y_definition="Common-population weighted medians of paired Q, vector gap D and complete bound S"),
    _evidence("theorem7_final_reproduction", "theorem7_final_reproduction", "main", "terminal_cdf",
              r"$\tau/\sqrt{d}$", "Fraction of samples",
              "Exact common-population weighted ECDFs of actual endpoint error, B_obs+delta_sched and B_ref+delta_sched. Correction uses independent pre-update inputs or a declared simultaneous Gaussian noise bound.",
              ["distribution", "value", "cdf"],
              normalization="Raw latent L2 tolerance divided once by sqrt(d); weighted cumulative fraction.",
              group_rule="No SSCD split; equal prompt mass over the same eligible member seeds for every curve.",
              band_definition="No statistical confidence band. Saved Gaussian noise probability scope is simultaneous within each run.",
              interpretation="Original clean Theorem 7 or explicitly identified finite-terminal-update extension according to saved scope; manuscript alignment remains required.",
              x_definition="All exact positive normalized-tolerance change points, with zero mass retained separately",
              y_definition="Actual reproduction and sufficient-bound coverage over the identical eligible population"),
]

EVIDENCE_APPENDIX = [
    _evidence("theorem1_loss_recovery", "theorem1_initial_branch_comparison", "appendix", "scatter",
              r"$\|\hat{\mathbf{x}}_T(\mathbf{Z},\varnothing)-\mathbf{x}^{\star}\|/\sqrt{d}$", r"$e_T(\mathbf{Z},c)/\sqrt{d}$",
              "Same-input conditional and unconditional target errors for every matched Gaussian pair/seed; SSCD only from an exactly matched initialization.",
              ["x", "y", "marker_class"], sscd=True, equal=True, include_zero=True, reference="Equal branch target errors"),
    _appendix_copy("theorem1_gaussian_transfer", "theorem1_gaussian_transfer"),
    _evidence("lemma2_unconditional_baseline", "lemma2_initial_concentration", "appendix", "ecdf",
              r"$\|\mathbf{x}-\boldsymbol{\mu}_K\|/\sqrt{d}$", "CDF",
              "Exact ECDFs of unique initial unconditional Gaussian estimates and distinct declared-law atoms centered on the same mu_K; atoms use declared masses.",
              ["distribution", "value", "cdf"], group_rule="Unique Gaussian seeds and distinct weighted atoms; no outcome cohorts.", normalization=NORMALIZATION),
    _appendix_copy("corollary3_guided_estimate_error", "corollary3_full_guided_error"),
    _appendix_copy("proposition5_posterior_feedback", "proposition5_condition_vs_gain"),
    _evidence("proposition5_posterior_feedback", "proposition5_first_update_comparison", "appendix", "scatter",
              "Matched target log odds", "Guided target log odds",
              "Direct same-destination endpoint target log odds on every structurally eligible first update; no logits inferred from clipped probabilities.",
              ["x", "y", "marker_class"], sscd=True, equal=True, reference="Equal target log odds"),
    _evidence("lemma6_target_specific_synchronization", "lemma6_branch_target_errors", "appendix", "curve",
              r"$\mathrm{SNR}_t$", r"Target error / $\sqrt{d}$",
              "Same-population conditional and unconditional target-error summaries at every current prediction.",
              ["group", "branch", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum"], branches=True, include_zero=True, group_rule=PROMPT_WEIGHTING),
    _appendix_copy("lemma6_target_specific_synchronization", "lemma6_bound_vs_gap"),
    _appendix_copy("lemma6_posterior_concentration", "lemma6_posterior_concentration"),
    _evidence("theorem7_final_reproduction", "theorem7_terminal_components", "appendix", "terminal_components",
              r"Term / $\sqrt{d}$", "CDF",
              "Common-population exact ECDFs of pre-update conditional target error, residual guidance (g-1)||Delta|| and independent scheduler-defect bound.",
              ["component", "value", "cdf"], group_rule="Same eligible terminal population as primary corrected curves; no outcome split.", optional_measurement=True),
]

AUDIT_ALIASES = {
    "theorem1_loss_recovery": "theorem1_gaussian_seed_errors",
    "corollary3_initial_cfg_amplification": "corollary3_injection_bound_audit",
    "theorem7_final_reproduction": "theorem7_uncorrected_bound_audit",
}


def evidence_registry(diagnostics=False):
    """Fixed seven evidence primaries and twelve required appendix contracts."""
    entries = copy.deepcopy(EVIDENCE_MAIN + EVIDENCE_APPENDIX)
    if diagnostics:
        existing = {entry["stem"] for entry in entries}
        for entry in direct_audit_registry(diagnostics=True):
            stem = AUDIT_ALIASES.get(entry["stem"], entry["stem"])
            if stem in existing:
                continue
            entry.update(stem=stem, category="diagnostics", source_table=f"plot_data/{stem}.csv", caption_key=stem,
                         outputs={ext: f"diagnostics/{stem}.{ext}" for ext in ("png", "pdf")})
            entries.append(entry)
            existing.add(stem)
    return entries


# Fixed precision/presentation revision; no branch depends on measured outcomes.
for _item in EVIDENCE_MAIN:
    if _item["stem"] == "theorem1_loss_recovery":
        _item["axes"]["y"] = "Initial target error (RMS)"
        _item["pair_connectors"] = True
    elif _item["stem"] == "lemma2_unconditional_baseline":
        _item.update(
            formula="Analytical ||bar_x_K(r,Z)-mu_K||/sqrt(d) on the fixed lower-SNR grid, with reference, learned and reference-error seed summaries marked only at the actual initial native SNR_T. The complete native sweep is a mandatory appendix.",
            interpretation="Finite initial network measurements are markers, not an observed convergent network tail. The initial high-noise display window is question-driven, not claimed preregistration.",
            x_definition="Fixed analytical SNR grid through SNR_T; network markers only at SNR_T",
            y_definition="Reference convergence curve and initial native learned/reference-error summaries in common latent RMSE units",
            native_sweep_figure="appendix/lemma2_native_gaussian_sweep.png",
            show_reference_scale=False,
        )
    elif _item["stem"] == "lemma4_matched_displacement":
        _item["sscd"] = False
        _item["required_columns"] = [*_item["required_columns"], "direct_lemma4_verification_source"]
        _item["group_rule"] = "Neutral points; actual independent versus constructed matched-update provenance determines marker style, never terminal SSCD."
    elif _item["stem"] == "proposition5_posterior_feedback":
        _item["required_columns"] = [*_item["required_columns"], "unresolved_count"]
        _item.update(
            formula="Strict fixed-cache gain and strict original-condition coverage over identical structural populations and prompt-balanced weights; missing V can retain a resolved-negative precheck sign without a fabricated margin value.",
            band_definition="Feedback numerical partial-identification band and thin original-condition bounds; unknown signs remain in the denominator. Separate arithmetic/quadrature, source sensitivity and reference identification.",
            numerical_resolution_figure="appendix/proposition5_numerical_resolution.png",
            interpretation="Fixed-cache numerical signs are distinct from source-robust signs; empirical gain prevalence does not imply original-condition coverage or identify the checkpoint's full training law.",
        )
    elif _item["stem"] == "lemma6_target_specific_synchronization":
        _item["axes"]["y"] = "Target error, gap, or bound (RMSE)"
    elif _item["stem"] == "theorem7_final_reproduction":
        _item["axes"]["y"] = "Coverage of latent-error tolerance"

EVIDENCE_APPENDIX.insert(3, _evidence(
    "lemma2_unconditional_baseline", "lemma2_native_gaussian_sweep", "appendix", "reference_native_sweep",
    r"$\mathrm{SNR}_t$", r"Distance / $\sqrt{d}$",
    "Complete saved native Gaussian sweep of reference-to-mu_K, learned-to-mu_K and learned-to-reference errors, including increasing errors and every available native label; no analytical extrapolation of network measurements.",
    ["metric", "source_range", "segment_id", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum"],
    include_zero=True, group_rule="Unique Gaussian seeds at each native label; no prompt replication or SSCD grouping.",
    band_definition="Descriptive unique-seed IQR, not a confidence interval.",
))
EVIDENCE_APPENDIX.insert(7, _evidence(
    "proposition5_posterior_feedback", "proposition5_numerical_resolution", "appendix", "numerical_resolution",
    r"$\mathrm{SNR}_t$", "Numerically resolved fraction",
    "All-timestep fixed-cache gain, original-condition and source-robust gain resolution on the same structural population, with complete weighted and unweighted cause/status counts in the linked audit table.",
    ["group", "metric", "step_index", "snr", "resolved_fraction", "unknown_fraction", "denominator_weight", "unresolved_count"],
    normalization="Dimensionless prompt-balanced fractions; per-row causes and arithmetic/source/reference scopes are separate.",
    band_definition="Resolution rates and unknown mass are numerical classification summaries, not statistical confidence intervals.",
    group_rule=PROMPT_WEIGHTING,
))
for _item in EVIDENCE_APPENDIX:
    if _item["stem"] == "theorem7_terminal_components":
        _item["formula"] += " The linked exact looseness audit retains B_ref-B_obs=(g-1)(S_raw-D_raw)=(g-1)(slack_radius+slack_triangle); the shared scheduler correction cancels."
del _item


# Active suite: four mechanism stages. The previous seven-statement registry
# remains an explicit frozen diagnostic source, never the default selector.
_EVIDENCE_BY_STEM = {entry["stem"]: entry for entry in evidence_registry(True)}


def _four_stage(original, stem, category, kind, question, x, y, formula, columns, **metadata):
    entry = copy.deepcopy(_EVIDENCE_BY_STEM[original])
    entry.update(stem=stem, category=category, kind=kind, semantic_question=question,
                 source_table=f"plot_data/{stem}.csv", caption_key=stem,
                 outputs={extension: f"{category}/{stem}.{extension}" for extension in ("png", "pdf")},
                 axes={"x": x, "y": y}, formula=formula, required_columns=columns,
                 formula_version="four-stage-figures-1", x_definition=x, y_definition=y,
                 interpretation=formula, evidence_classification="Fixed four-stage mechanism comparison with retained manuscript audits")
    entry["axis_notation"] = "All latent vectors use bold symbols. K denotes the fixed empirical evaluation law, not its identification with the training marginal. Chronological prediction k, native timestep and theoretical t remain distinct. Normalized latent norms divide raw L2 by sqrt(d) exactly once; log-probability and log-odds ordinates remain distinct."
    entry.pop("native_sweep_figure", None)
    if "numerical_resolution_figure" in entry:
        entry["numerical_resolution_figure"] = "diagnostics/proposition5_numerical_resolution.png"
    entry["manuscript_label_status"] = "author_supplied_unverified_without_matching_latex"
    entry["source_label_status"] = "printed_result_and_equation_numbers_checked_against_revised_pdf; latex_labels_unverified"
    entry.update(metadata)
    return entry


FOUR_STAGE_MAIN = [
    _four_stage("theorem1_loss_recovery", "initial_loss_recovery", "main", "pair_loss",
        "How does genuine forward loss compare with initial Gaussian conditional recovery and its matched unconditional control?",
        r"$\sqrt{\widehat{L}_T(c)/(d\,\mathrm{SNR}_T)}$", "Initial target error (RMS)",
        "A_c=sqrt(Lhat_T(c)/(d SNR_T)); Y_c=sqrt(mean_n ||m_c,T(Z_n)-x_star||^2/d); U_c=sqrt(mean_n ||m_u,T(Z_n)-x_star||^2/d). Forward corruption and Gaussian initialization have different input laws. One colored conditional RMS and a hollow matched unconditional target-error control per pair; no equality claim. Unique-seed B_T^K about mu_K and S_K are reported separately.",
        ["x", "y", "control_y", "mean_terminal_sscd", "x_low", "x_high", "y_low", "y_high"],
        experiment=1, manuscript_results=["Theorem 1", "Lemma 2"], pair_connectors=True,
        group_rule="One pair-level conditional and unconditional RMS comparison using the complete Gaussian seed bank; no outcome split. Color is the mean terminal SSCD of all retained actual evaluation seeds for that pair, not an input-match claim for fresh probes.",
        initial_baseline_summary="initial_baseline_summary.csv", colorbar_label="Mean terminal SSCD", averaging_unit="Retained prompt-target pair; paired Gaussian seeds and independent forward-loss draws are separate Monte Carlo populations"),
    _four_stage("proposition5_posterior_feedback", "branch_gap_posterior_response", "main", "four_response",
        "How does retaining the actual initial branch-gap contribution change matched target evidence along one fixed local segment?",
        r"Retained gap contribution $s$", "Target log-probability gain",
        "At fixed k=0, z(s)=x_cf,next+s g kappa Delta_0 and H^K(s)=log p_K(next,z(s))-log p_K(next,z(0)). Fixed grid {j/40} union {1/g}; s=0 is matched unconditional, 1/g matched conditional-only, and 1 matched CFG with the saved reconstruction qualification. Direction, current estimates, law and destination coefficients stay fixed. This is neither a guidance rollout nor a claim that arbitrary larger gap norm helps.",
        ["group", "s", "median", "q25", "q75", "minimum", "maximum", "denominator_weight", "eligible_count"],
        experiment=2, manuscript_results=["Corollary 3", "Lemma 4", "Proposition 5"],
        input_source=["matched_affine_segment"], normalization="Natural-log probability gain, not log odds; s is dimensionless",
        band_definition="Descriptive prompt-balanced IQR on one complete-curve cohort per outcome group; unresolved finite estimates remain included",
        weighting=PROMPT_WEIGHTING, group_rule=PROMPT_WEIGHTING, allow_unavailable=False),
    _four_stage("lemma6_target_specific_synchronization", "branch_gap_synchronization", "main", "four_chronological",
        "Do the two branches agree with one another and with the paired target as denoising progresses?",
        r"Prediction index $k$", "Target error or branch gap (RMSE)",
        "D_k=||Delta_k||/sqrt(d); Q_k=max(||m_c,k-x_star||,||m_u,k-x_star||)/sqrt(d), formed on each sample before aggregation. Dashed D and solid Q use common rows/weights at chronological k=0,...,K_steps-1. k=0 is initialization; the final saved output has no additional branch prediction. Small D with large Q is wrong-target agreement. Lemma 6 does not assert temporal monotonicity or unimodality.",
        ["group", "metric", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum", "denominator_weight"],
        experiment=3, manuscript_results=["Lemma 6"], update_index=True, include_zero=True,
        band_definition="Light descriptive IQR for paired Q; D quantiles remain saved", allow_unavailable=False),
    _four_stage("theorem7_final_reproduction", "final_reproduction_bound", "main", "four_terminal_scatter",
        "How does actual saved endpoint target error compare with the full empirical-reference reproduction bound?",
        "Reference bound (RMSE)", "Final target error (RMSE)",
        "x=[B_ref+delta_sched]/sqrt(d), y=||x_final_saved-x_star||/sqrt(d); B_ref=g e_c+(g-1)(e_u^K+R_K(1-p_K)). The predictor-side correction uses pre-update quantities, never a fitted endpoint difference. The original clean theorem requires its structural clean update; other supported modes are explicitly qualified finite-terminal extensions. Equality is a deterministic/pathwise or probability-qualified upper-bound comparison, as recorded.",
        ["x", "y", "terminal_sscd", "applicable", "terminal_scope"],
        experiment=4, manuscript_results=["Theorem 7", "Explicit finite-terminal-update extension when required"],
        normalization="Every raw L2 bound/error divided by sqrt(d) exactly once", weighting="Unaggregated eligible same-seed endpoint/bound pairs; no weighted aggregate on the scatter", sscd=True, allow_unavailable=True),
]

FOUR_STAGE_APPENDIX = [
    _four_stage("lemma2_initial_concentration", "initial_unconditional_mean_concentration", "appendix", "ecdf",
        "How concentrated are unique initial unconditional estimates and declared-law atoms around the same empirical mean?",
        r"Distance to $\boldsymbol{\mu}_K$ (RMSE)", "CDF",
        "Exact ECDFs of initial learned unconditional Gaussian estimates and distinct atoms about the same mu_K. Evaluation seeds have equal mass; atoms have the declared masses. No prompt replication or terminal-outcome filtering.", ["distribution", "value", "cdf"]),
    _four_stage("lemma2_unconditional_baseline", "unconditional_reference_convergence", "appendix", "four_reference",
        "How do analytical reference concentration and genuine native Gaussian-probe errors compare?",
        r"$\mathrm{SNR}$", "Distance or reference error (RMSE)",
        "Reference-to-mu_K distance on the preserved lower-SNR analytical grid; all available native Gaussian reference, learned-to-mu_K, and learned-to-reference summaries remain visible. Network curves exist only at their actual native labels. The analytical extension is not a learned convergent tail.",
        ["metric", "source_range", "step_index", "segment_id", "snr", "median", "q25", "q75", "minimum", "maximum"]),
    _four_stage("proposition5_posterior_feedback", "posterior_feedback_over_time", "appendix", "feedback_fractions",
        "How do strict observed feedback and the original sufficient condition vary across every applicable transition?",
        r"$\mathrm{SNR}_t$", "Fraction of eligible samples",
        "Original M_K=D-e_c-e_u^K-V_K and saved-endpoint H_K retain their endpoint contracts. Resolved positive mass and positive-plus-unknown mass use the same structural rows and prompt weights. A negative precheck can resolve the sign without fabricating V. Zero original-condition coverage remains explicit; empirical feedback is not its converse.",
        ["group", "metric", "step_index", "snr", "fraction", "upper_fraction", "denominator_weight", "unresolved_count"], allow_unavailable=False),
    _four_stage("proposition5_condition_vs_gain", "posterior_feedback_condition_margin", "appendix", "scatter",
        "Where do the original margin and observed endpoint gain lie, including negative and unresolved measurements?",
        r"$M_K/\sqrt{d}$", r"Log-probability gain $H_K$",
        "The full original reference-error margin is compared with observed log-probability gain, never substituted by a directional condition or log-odds ordinate. Same-path implication requires the saved endpoint-consistency/transfer gate. Missing point values remain counted separately from resolved negative signs.",
        ["x", "y", "marker_class"], signed=True, zero_guides=True, equal=False, sscd=True, allow_unavailable=True),
    _four_stage("lemma6_branch_target_errors", "branch_target_errors", "appendix", "four_chronological",
        "How do conditional and unconditional paired-target errors evolve?", r"Prediction index $k$", "Target error (RMSE)",
        "Conditional and unconditional target errors on the common per-step branch population; unconditional target error is not e_u^K. Native labels/SNR remain saved with chronological k.",
        ["group", "metric", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum", "denominator_weight"], update_index=True, include_zero=True),
    _four_stage("lemma6_target_specific_synchronization", "synchronization_bound", "appendix", "four_chronological",
        "How loose is the complete synchronization bound relative to gap and joint target error?", r"Prediction index $k$", "Error, gap, or bound (RMSE)",
        "S_k^K=[e_c,k+e_u,k^K+R_K(1-p_K)]/sqrt(d), D_k and paired Q_k are formed before common-population aggregation. D<=S is the lemma comparison; Q<=S is a derived consequence. Full bounds and both groups remain visible without per-curve rescaling.",
        ["group", "metric", "step_index", "snr", "median", "q25", "q75", "minimum", "maximum", "denominator_weight"], update_index=True, include_zero=True),
    _four_stage("lemma6_target_specific_synchronization", "branch_gap_peak_step", "appendix", "four_peak",
        "Where do individual gap trajectories reach their first resolved global maximum?", r"First peak index $k$", "Weighted sample fraction",
        "Prompt-balanced distribution of first global-maximum indices from complete individual raw trajectories. Flat, incomplete and unresolved cases retain explicit mass and counts rather than an invented peak. All exact/near ties, continuous shape descriptors and prompt-level mixed-outcome summaries remain in audit tables; the peak of a group median is a different quantity.",
        ["group", "step_index", "fraction", "denominator_weight", "sample_count"], update_index=True),
    *[_four_stage("lemma6_target_specific_synchronization", stem, "appendix", "four_motion",
        "Which consecutive branch motions account for the measured change in squared gap?", r"Prediction index $k$", "Change in squared gap / coordinate",
        "change_k=D_(k+1)^2-D_k^2=C_k+U_k+gap_motion_quadratic, with C_k=2<Delta_k,v_c,k>/d and U_k=-2<Delta_k,v_u,k>/d. Weighted means use identical rows and weights for every additive term and the observed change. This is descriptive accounting including the changing noise label, not a causal intervention or a new named theorem.",
        ["group", "metric", "step_index", "mean", "minimum", "maximum", "denominator_weight"], update_index=True,
        outcome_group=group, band_definition="No inferential interval; common-population weighted means preserve the additive identity")
      for stem, group in (("branch_gap_motion_high_sscd", GROUPS[0]), ("branch_gap_motion_lower_sscd", GROUPS[1]))],
    _four_stage("theorem7_final_reproduction", "terminal_observable_bound", "appendix", "four_terminal_scatter",
        "How does the observable bound compare with actual saved endpoint error?", "Observable bound (RMSE)", "Final target error (RMSE)",
        "x=[e_c+(g-1)||Delta_last||+delta_sched]/sqrt(d), y=E_end/sqrt(d), on the same eligible samples as the full-reference comparison. The bound is distinct from B_ref; correction mode, clean applicability, zero cases and probability scope remain explicit.",
        ["x", "y", "terminal_sscd", "applicable", "terminal_scope"], sscd=True, weighting="Unaggregated eligible same-seed endpoint/bound pairs; no weighted aggregate on the scatter"),
    _four_stage("theorem7_final_reproduction", "terminal_bound_coverage", "appendix", "terminal_cdf",
        "What fraction satisfies each absolute latent-error tolerance?", r"$\tau/\sqrt{d}$", "Coverage of latent-error tolerance",
        "Exact weighted ECDFs of actual endpoint error, corrected observable bound and corrected reference bound on identical eligible rows and prompt weights. Zero/infinite mass, stochastic probability scope and original clean applicability remain saved; no tolerance is inferred from SSCD.",
        ["distribution", "value", "cdf"]),
]

_COUNTERFACTUAL = _four_stage("lemma6_branch_target_errors", "counterfactual_unconditional_response", "appendix", "four_counterfactual",
    "At a fixed matched update, how does the actual empty-prompt learned predictor respond at the two endpoints?",
    "Counterfactual target error (RMSE)", "Actual-next target error (RMSE)",
    "Optional actual-network probe: I_net=(||m_u(next,x_cf)-x_star||-||m_u(next,x_next)-x_star||)/sqrt(d). Both endpoints are evaluated under the recorded compatible inference contract; this is not the empirical posterior reference or a new rollout. Fixed snapshots and exclusions are saved. Equality denotes zero measured I_net.",
    ["x", "y", "terminal_sscd", "step_index", "counterfactual_I_net"], sscd=True, optional_measurement=True,
    input_source=["learned_network_counterfactual_probe"], weighting="Unaggregated common-context learned endpoint pairs", allow_unavailable=True)


def paper_registry(diagnostics=False, *, counterfactual=False):
    """Exactly four main and eleven mandatory appendix entries by default."""
    entries = copy.deepcopy(FOUR_STAGE_MAIN + FOUR_STAGE_APPENDIX)
    if counterfactual:
        entries.append(copy.deepcopy(_COUNTERFACTUAL))
    if diagnostics:
        existing = {entry["stem"] for entry in entries}
        for entry in evidence_registry(True):
            if entry["stem"] in existing:
                continue
            entry.update(category="diagnostics", outputs={ext: f"diagnostics/{entry['stem']}.{ext}" for ext in ("png", "pdf")})
            entries.append(entry)
            existing.add(entry["stem"])
        if not counterfactual:
            entry = copy.deepcopy(_COUNTERFACTUAL)
            entry.update(category="diagnostics", outputs={ext: f"diagnostics/{entry['stem']}.{ext}" for ext in ("png", "pdf")})
            entries.append(entry)
    return entries
