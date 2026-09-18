"""Fixed paper figure definitions, independent of measured effect directions."""

from __future__ import annotations

import copy

from .paper_notation import NOTATION_DETAILS, NOTATION_SCOPE, NOTATION_VERSION, PAPER_AXES

REGISTRY_VERSION = "four-stage-paper-curation-20"
PREVIOUS_REGISTRY_VERSION = "four-stage-evidence-1"
COMPATIBLE_PRESENTATION_REGISTRY_VERSIONS = frozenset({
    REGISTRY_VERSION, "four-stage-paper-curation-19", "four-stage-paper-curation-18", "four-stage-paper-curation-17", "four-stage-paper-curation-16", "four-stage-paper-curation-15", "four-stage-paper-curation-14", "four-stage-paper-curation-13", "four-stage-paper-curation-12", "four-stage-paper-curation-11", "four-stage-paper-curation-10", "four-stage-paper-curation-9", "four-stage-paper-curation-8", "four-stage-paper-curation-7", "four-stage-paper-curation-6", "four-stage-paper-curation-5", "four-stage-paper-curation-4", "four-stage-paper-curation-3", "four-stage-paper-curation-2", PREVIOUS_REGISTRY_VERSION,
})
PLOT_RECIPE_VERSION = NOTATION_VERSION

BRANCH_GAP_ERROR_CONTRACT = "projected-gap-error-1"
_BRANCH_GAP_ERROR_SCOPE = (
    "Requested signed projected-gap-error refinement: mathcal{E}_t=u_t^T(Delta_t-bar{Delta}_t), "
    "u_t=Delta_t/||Delta_t|| and bar{Delta}_t=bar{x}_t(c)-bar{x}_t(empty). "
    "The actual vector projection may be negative; it is not a norm, absolute value, or clipped scalar. "
    "The exact identity ||Delta||=mathcal{E}_t+u^T bar{Delta}, Cauchy and posterior concentration "
    "give D<=S and the qualified terminal bound. The directional condition retains matched-endpoint "
    "and applicability prerequisites. These are derived refinements requested by the author, not unchanged PDF expressions. "
    "mathcal{V}=[integral signed unit-gap projection of next reference minus current reference]_+ "
    "retains its positive part after integration. Exactly zero Delta uses the computational extension "
    "u=0, mathcal{E}_t=0 and mathcal{V}=0 without claiming a unit direction. "
    "Q<=S is not asserted; common-mode errors can make Q large."
)
_BRANCH_GAP_ERROR_FORMULAS = {
    "posterior_feedback_over_time": "Resolved gain H>0 and refined margin M=||Delta||-mathcal{E}_t-mathcal{V}>0 on the same structural population and prompt weights; unresolved mass remains explicit. The sufficient implication retains the matched-endpoint and applicability qualifications.",
    "posterior_feedback_condition_margin": "x=[||Delta||-mathcal{E}_t-mathcal{V}]/sqrt(d), y=saved matched-endpoint log-probability gain. mathcal{E}_t=u_t^T(Delta-bar{Delta}) is formed from vectors and retains its sign. All finite saved x/y pairs are ordinary dots in the combined and per-timestep views, without sign-based filtering or marker classes. Numerical signs and refined sufficient-condition/endpoint-transfer checks remain separate audit data.",
    "synchronization_bound": "D=||Delta||/sqrt(d) and S=[mathcal{E}_t+R(1-p)]/sqrt(d) are formed per sample before shared-population aggregation. The refined Lemma 6 bound D<=S follows from the exact gap-projection identity, Cauchy and posterior concentration.",
    "final_reproduction_bound": "x=[B_ref+delta_sched]/sqrt(d), y=E_end/sqrt(d), with B_ref=e_c+(g-1)*(mathcal{E}_t+R(1-p)). The refined gap bound implies B_obs<=B_ref for g>1; endpoint and independent correction qualifications remain unchanged.",
    "terminal_bound_coverage": "Common-population prompt-balanced ECDFs of E_end/sqrt(d), [B_obs+delta_sched]/sqrt(d), and [B_ref+delta_sched]/sqrt(d), separately by same-seed terminal SSCD. B_obs=e_c+(g-1)||Delta||; B_ref=e_c+(g-1)*(mathcal{E}_t+R(1-p)). The refined ordering uses the same applicability and independent-correction scope; curve ordering does not replace row-level checks.",
}

_BRANCH_GAP_ERROR_ALIASES = {
    "proposition5_posterior_feedback": "posterior_feedback_over_time",
    "proposition5_condition_vs_gain": "posterior_feedback_condition_margin",
    "proposition5_numerical_resolution": "posterior_feedback_over_time",
    "posterior_feedback_coverage_audit": "posterior_feedback_over_time",
    "terminal_bound_tightness": "final_reproduction_bound",
    "lemma6_target_specific_synchronization": "synchronization_bound",
    "lemma6_bound_vs_gap": "synchronization_bound",
    "theorem7_final_reproduction": "terminal_bound_coverage",
    "theorem7_uncorrected_bound_audit": "final_reproduction_bound",
    "theorem7_terminal_components": "final_reproduction_bound",
    "theorem7_certification": "terminal_bound_coverage",
}


def _apply_branch_gap_error_contract(entry):
    """Version current mathematical semantics; frozen descriptor lists stay historical."""
    source = _BRANCH_GAP_ERROR_ALIASES.get(entry["stem"], entry["stem"])
    if source not in _BRANCH_GAP_ERROR_FORMULAS:
        return entry
    if source == "posterior_feedback_over_time" and entry["kind"] == "scatter":
        source = "posterior_feedback_condition_margin"
    elif source == "terminal_bound_coverage" and entry["kind"] == "scatter":
        source = "final_reproduction_bound"
    formula = _BRANCH_GAP_ERROR_FORMULAS[source]
    if entry["stem"] == "posterior_feedback_coverage_audit":
        formula = "Revised M0=M1=D-mathcal{E}_t-[W]_+, M2=D+S_projection-V_norm, M3=D+S_projection-V_abs, A=D+S_projection-W. With E_parallel=-S_projection, M2<=M3<=M0=M1<=A; the projection remains signed. W is the signed projected integral. Resolved/estimated strict-positive coverage uses one shared structural population; endpoint/application qualifications remain explicit."
    elif entry["stem"] == "terminal_bound_tightness":
        formula = "B_obs=[e_c+(g-1)||Delta||]/sqrt(d); B_ref=[e_c+(g-1)(mathcal{E}_t+R(1-p))]/sqrt(d). Ratios divide each by the observed endpoint RMSE only when nonzero; zero-error statuses remain explicit."
    entry.update(formula=formula, interpretation=formula + " " + _BRANCH_GAP_ERROR_SCOPE,
                 formula_version=BRANCH_GAP_ERROR_CONTRACT, measurement_contract=BRANCH_GAP_ERROR_CONTRACT,
                 direct_statement=False, evidence_classification="Derived signed projected-gap-error refinement",
                 mathematical_scope=_BRANCH_GAP_ERROR_SCOPE,
                 manuscript_label_status="derived_refinement_from_gap_projection_identity_and_cauchy",
                 supporting_equation="Derived refinement using the exact gap-projection identity and Cauchy; same endpoint, reference-law and correction prerequisites",
                 presentation_note=_BRANCH_GAP_ERROR_SCOPE,
                 semantic_question="How does the refined branch-gap-error comparison relate to the saved measured outcome?")
    if entry["stem"] in {"theorem7_uncorrected_bound_audit", "theorem7_final_reproduction"} and entry["kind"] == "scatter":
        entry["formula"] = "x=[e_1(c)+(g-1)(mathcal{E}_1+R(1-p_1))]/sqrt(d), y=actual saved endpoint error/sqrt(d), under the original verified clean-update prerequisite; no finite-step correction is silently inserted."
        entry["interpretation"] = entry["formula"] + " " + _BRANCH_GAP_ERROR_SCOPE
    if source in {"posterior_feedback_over_time", "posterior_feedback_condition_margin"}:
        entry["normalization"] = ("Saved margin [||Delta||-mathcal{E}_t-mathcal{V}]/sqrt(d) and matched log-probability gain, or weighted fractions of their assessed signs. mathcal{V} takes its positive part after signed unit-gap projection integration; dimensional normalization is applied once.")
    entry["reference"] = "Equality of compared quantities"
    if entry["stem"] in _BRANCH_GAP_ERROR_FORMULAS or source == "posterior_feedback_condition_margin":
        entry["axes"] = copy.deepcopy(PAPER_AXES.get(source, entry["axes"]))
    elif source == "synchronization_bound" and entry["kind"] == "scatter":
        entry["axes"] = {"x": r"$[\mathcal{E}_t+R(1-p_t)]/\sqrt{d}$",
                         "y": r"$\|\boldsymbol{\Delta}_t\|/\sqrt{d}$"}
    elif source == "final_reproduction_bound" and entry["kind"] == "scatter":
        entry["axes"] = copy.deepcopy(PAPER_AXES["final_reproduction_bound"])
    elif entry["stem"] == "theorem7_certification":
        entry["axes"] = {"x": "Sufficient bound fraction", "y": "Actual proximity fraction"}
    entry["x_definition"], entry["y_definition"] = entry["axes"]["x"], entry["axes"]["y"]
    entry["notation_details"] = NOTATION_DETAILS.get(source, _BRANCH_GAP_ERROR_SCOPE)
    entry["axis_notation"] = NOTATION_SCOPE
    return entry

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
        weighting="Equal mass over unique evaluation run/seed pairs; candidate atoms use declared empirical masses, aggregating duplicate source records.",
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
    "One declared finite law supplies the posterior, learned-reference error, probabilities, "
    "radius and exact weighted atom mean. The selected theory centre mu is separate: "
    "by default it is the saved vector average of 10000 unconditional posterior clean "
    "references at the smallest SNR of the prespecified analytical extension on independent Gaussian inputs, "
    "with finite-selected-SNR and Monte Carlo qualifications and no learned-network evaluation. "
    "Explicit native-initial estimation and exact finite-bank-mean options are identified in the measurement receipt. "
    "Selecting mu never changes the atom masses or posterior and does not identify "
    "the finite law or selected reference estimate with the checkpoint's complete training distribution."
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


def direct_audit_registry(diagnostics=False, *, historical=False):
    """Seven direct statements plus supporting checks; legacy designs require opt-in."""
    entries = copy.deepcopy(CORE + APPENDIX)
    if diagnostics:
        for entry in copy.deepcopy(LEGACY_CORE + LEGACY_TERMINAL + LEGACY_DIAGNOSTICS):
            entry["category"] = "diagnostics"
            entry["outputs"] = {ext: f"diagnostics/{entry['stem']}.{ext}" for ext in ("png", "pdf")}
            entry["direct_statement"] = False
            entry["interpretation"] = "Retained behavioral diagnostic, not a direct statement comparison."
            entries.append(entry)
    return entries if historical else [_apply_branch_gap_error_contract(entry) for entry in entries]


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


def evidence_registry(diagnostics=False, *, historical=False):
    """Fixed seven evidence primaries and twelve required appendix contracts."""
    entries = copy.deepcopy(EVIDENCE_MAIN + EVIDENCE_APPENDIX)
    if diagnostics:
        existing = {entry["stem"] for entry in entries}
        for entry in direct_audit_registry(diagnostics=True, historical=historical):
            stem = AUDIT_ALIASES.get(entry["stem"], entry["stem"])
            if stem in existing:
                continue
            entry.update(stem=stem, category="diagnostics", source_table=f"plot_data/{stem}.csv", caption_key=stem,
                         outputs={ext: f"diagnostics/{stem}.{ext}" for ext in ("png", "pdf")})
            entries.append(entry)
            existing.add(stem)
    return entries if historical else [_apply_branch_gap_error_contract(entry) for entry in entries]


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
        _item["formula"] += " The linked exact looseness audit retains B_ref-B_obs=(g-1)(S_raw-D_raw)=(g-1)(slack_radius+slack_projection_alignment); the shared scheduler correction cancels."
del _item


# Active suite: four mechanism stages. The previous seven-statement registry
# remains an explicit frozen diagnostic source, never the default selector.
_EVIDENCE_BY_STEM = {entry["stem"]: entry for entry in evidence_registry(True, historical=True)}


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


# Frozen preceding scientific descriptors for receipt/readback fixtures. Current
# public selectors apply _apply_branch_gap_error_contract to affected entries;
# these historical e_c+e_u expressions are not current computation/display specs.
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
        "How concentrated are unique initial unconditional estimates and declared-law atoms around the selected theory mean?",
        r"Distance to $\boldsymbol{\mu}_K$ (RMSE)", "CDF",
        "Exact ECDFs of initial learned unconditional Gaussian estimates and distinct atoms about the same selected mu. The default mu is the independent 10000-draw unconditional posterior reference estimate at the analytical extension minimum SNR; its receipt distinguishes finite-selected-SNR bias, Monte Carlo uncertainty and the measured finite-bank offset. Evaluation seeds have equal mass; atoms retain declared masses. No prompt replication or terminal-outcome filtering.", ["distribution", "value", "cdf"]),
    _four_stage("lemma2_unconditional_baseline", "unconditional_reference_convergence", "appendix", "four_reference",
        "How do analytical reference concentration and genuine native Gaussian-probe errors compare?",
        r"$\mathrm{SNR}$", "Distance or reference error (RMSE)",
        "Reference-to-selected-mu distance on the preserved analytical grid, with all native learned-to-mu and unchanged learned-to-reference errors. The selected mean is separately estimated from unconditional posterior clean references at the analytical extension minimum SNR on independent Gaussian inputs by default. The horizontal guide is the normalized distance between the exact finite-bank mean and selected mu, which need not vanish; the caption and audit retain its meaning and value without a legend entry. Network curves exist only at measured native labels.",
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


# The historical pooled ECDF remains an audit; the active main view compares
# fixed SSCD groups using the same three corrected quantities within each group.
_GROUPED_TERMINAL_COVERAGE = _four_stage(
    "theorem7_final_reproduction", "terminal_bound_coverage", "main", "four_terminal_grouped_cdf",
    "How do actual-error and bound tolerance coverage differ between SSCD > 0.75 and SSCD <= 0.75?",
    r"$\tau/\sqrt{d}$", r"$\Pr(\,\cdot\leq\tau)$",
    "Within each same-seed terminal-SSCD group (>0.75 or <=0.75), exact weighted ECDFs of E_end/sqrt(d), [B_obs+delta_sched]/sqrt(d), and [B_ref+delta_sched]/sqrt(d) share the same eligible samples and prompt-balanced weights. All groups share one tolerance axis. Zero/infinite mass, original clean applicability, correction scope and row-level ordering audits remain explicit. SSCD selects the outcome group, never a latent tolerance.",
    ["group", "distribution", "value", "cdf", "denominator_weight", "eligible_count", "prompt_count"],
    formula_version="terminal-coverage-by-sscd-1", experiment=4, manuscript_results=["Theorem 7"],
    group_rule="Same-seed terminal SSCD > 0.75 versus SSCD <= 0.75; exact threshold belongs to the lower group. Missing/nonfinite scores are audited separately. Prompt identities may occur in both groups through different seeds.",
    weighting="Within each group, one mass unit per represented prompt-target pair divided over its eligible member seeds; identical rows and weights for actual, observable and reference curves",
    averaging_measure="Group-conditional prompt-balanced cumulative fraction at each common latent tolerance",
    normalization="Saved per-sample latent L2/sqrt(d) quantities and tolerance, each normalized exactly once",
    band_definition="None; exact saved group ECDFs. Gaussian noise-bound probability scope remains separate.",
    required_applicability="The original common terminal-correction eligibility plus a finite same-seed terminal SSCD; empty groups are explicit and never fabricated",
    group_common_population_table="audit_data/terminal_grouped_common_population.csv",
    group_sscd_exclusion_table="audit_data/terminal_grouped_sscd_exclusions.csv",
    pooled_cdf_audit_table="audit_data/terminal_pooled_cdf.csv",
    pooled_metadata_audit_table="audit_data/terminal_pooled_cdf_metadata.csv", allow_unavailable=False,
)


# This additional saved-scalar view does not alter the historical registry or
# any existing measurement identity, appendix slot, or support reference.
_PROMPT_BRANCH_GAP = _four_stage(
    "lemma6_target_specific_synchronization", "branch_gap_per_prompt", "appendix", "four_prompt_chronological",
    "How does each retained prompt-target pair's mean branch gap evolve across the same complete seed population?",
    r"$T-t$", r"$\|\boldsymbol{\Delta}_t\|/\sqrt{d}$",
    "One curve per retained prompt-target pair: the arithmetic mean across its complete fixed evaluation seed cohort of ||Delta_t||/sqrt(d) at each chronological prediction T-t. The norm is formed per seed before averaging; this is not the norm of a mean vector. Curve color is mean terminal SSCD across exactly the same seeds, fixed across the curve. All complete pair curves are shown without outcome grouping, per-curve rescaling, or a temporal-shape claim.",
    ["run_id", "original_index", "record_id", "target_id", "step_index", "mean_gap_rmse", "mean_terminal_sscd", "seed_count", "seed_ids_json", "latent_dimension", "cohort_complete"],
    formula_version="branch-gap-per-prompt-1", experiment=3, manuscript_results=["Lemma 6"],
    value_column="mean_gap_rmse", value_domain="nonnegative", source_scalar_column="direct_lemma6_gap_rmse",
    input_source=["saved_trajectory_metrics", "saved_initial_terminal_outcomes"],
    reference_law="The measured branch difference uses the paired learned estimates directly; no reference posterior or new model evaluation is required.",
    normalization="Mean of saved per-seed L2/sqrt(d) values; no additional dimensional normalization",
    averaging_measure="Arithmetic mean of individual seed branch-gap norms on the same complete seed cohort at every prediction",
    averaging_unit="Retained prompt-target pair, preserving run/original-index/record/target identity",
    group_rule="All complete retained prompt-target pairs; no SSCD outcome split or curve selection",
    weighting="Equal mass per seed within each prompt-target pair; the same seeds determine mean terminal SSCD",
    band_definition="None; one mean curve per prompt-target pair",
    required_applicability="Saved complete fixed seed cohort and all chronological prediction indices, with matched terminal SSCD",
    cohort_audit_table="audit_data/branch_gap_per_prompt_cohort.csv", colorbar_label="SSCD",
    sscd=True, update_index=True, include_zero=True, allow_unavailable=False,
)


def _prompt_trajectory_figure(stem, value_column, value_domain, axis, source_column, definition):
    """Additional saved-scalar views; existing paper slots and measurements stay fixed."""
    return _four_stage(
        "lemma6_target_specific_synchronization", stem, "appendix", "four_prompt_chronological",
        "How does this saved manuscript quantity evolve for each prompt-target pair over its complete fixed seed cohort?",
        r"$T-t$", axis,
        definition + " At each saved pre-update prediction, one curve per prompt-target pair averages the individual seed values over the same complete fixed evaluation seed cohort. Its fixed color is mean terminal SSCD across those same seeds. No posterior or network evaluation is performed for this scalar reduction, and no final-output prediction is invented.",
        ["run_id", "original_index", "record_id", "target_id", "step_index", value_column,
         "mean_terminal_sscd", "seed_count", "seed_ids_json", "latent_dimension", "cohort_complete"],
        formula_version="prompt-trajectory-scalars-1", experiment=3, manuscript_results=["Lemma 6"],
        value_column=value_column, value_domain=value_domain, source_scalar_column=source_column,
        input_source=["saved_trajectory_metrics", "saved_initial_terminal_outcomes"],
        reference_law="Saved clean-reference errors, analytical reference branch gaps and target posterior probabilities use the declared finite atom law; the conditional reference retains the stated single-target conditional idealization. The law is not identified with the complete training marginal.",
        normalization=("Dimensionless probability; exponentiate each saved seed log probability before averaging"
                       if value_domain == "probability" else
                       "Mean of saved per-seed L2/sqrt(d) latent norms; no additional dimensional normalization"),
        averaging_measure="Arithmetic mean of individual seed values on the same complete seed cohort at every prediction",
        averaging_unit="Retained prompt-target pair, preserving run/original-index/record/target identity",
        group_rule="All complete retained prompt-target pairs; no SSCD outcome split or curve selection",
        weighting="Equal mass per seed within each prompt-target pair; the same seeds determine mean terminal SSCD",
        band_definition="None; one mean curve per prompt-target pair",
        required_applicability="Saved complete fixed seed cohort and all chronological prediction indices, with matched terminal SSCD and valid values for the displayed quantity",
        cohort_audit_table=f"audit_data/{stem}_cohort.csv", colorbar_label="SSCD",
        sscd=True, update_index=True, include_zero=True, allow_unavailable=False,
    )


_PROMPT_TRAJECTORY_FIGURES = [
    _prompt_trajectory_figure(
        "conditional_reference_error_per_prompt", "mean_conditional_error_rmse", "nonnegative",
        r"$e_t(c)/\sqrt{d}$", "direct_conditional_error_rmse",
        "Mean conditional clean-reference error e_t(c)/sqrt(d), taken from the saved per-seed normalized norms under the fixed single-target conditional reference assumption; the norm is formed before averaging."),
    _prompt_trajectory_figure(
        "unconditional_reference_error_per_prompt", "mean_unconditional_reference_error_rmse", "nonnegative",
        r"$e_t(\varnothing)/\sqrt{d}$", "direct_unconditional_reference_error_rmse",
        "Mean learned unconditional clean-estimate error relative to its declared-law posterior reference, e_t(empty)/sqrt(d), taken from the saved per-seed normalized norms. This is not unconditional distance to the target; the norm is formed before averaging."),
    _prompt_trajectory_figure(
        "target_probability_per_prompt", "mean_target_probability", "probability",
        r"$p_t$", "direct_target_log_probability",
        "Mean target posterior probability on the actual saved generated trajectory state, formed as mean(exp(saved direct_target_log_probability)) across seeds. Exponentiation precedes averaging; exp(mean(log probability)) would be a different quantity. It is not a conditional-only counterfactual probability."),
    _prompt_trajectory_figure(
        "reference_branch_gap_per_prompt", "mean_reference_gap_rmse", "nonnegative",
        r"$\|\bar{\boldsymbol{\Delta}}_t\|/\sqrt{d}$", "direct_reference_target_error_rmse",
        "The analytical reference branch difference is conditional minus unconditional, bar{x}_t(c)-bar{x}_t(empty), at the same actual saved state and noise level. Under the stated single-target conditional law, bar{x}_t(c)=x_star, so its normalized norm equals saved direct_reference_target_error_rmse, whose vector has the opposite sign, bar{x}_t(empty)-x_star. Norm equality permits reuse without vector reconstruction or another normalization. Individual seed norms are averaged; the learned branch gap, differences of error norms, and the radius-tail bound are not substituted."),
]


_PROMPT_REFERENCE_VARIATION = _four_stage(
    "proposition5_posterior_feedback", "reference_variation_per_prompt", "appendix", "four_prompt_chronological",
    "How does the saved cross-step reference variation evolve for each complete prompt-target seed cohort?",
    r"$T-t$", r"$\mathcal{V}_t/\sqrt{d}$",
    "Requested directional revision of the Proposition 5 variation: mathcal{V}_t is the positive part after integrating <Delta_t/||Delta_t||,bar{x}_{t-1}(x_cf+s*g*kappa_t*Delta_t,empty)-bar{x}_t(x_t,empty)> over s in [0,1]. Neither a norm, an absolute value nor nodewise clipping is used. One curve per prompt-target pair averages the saved per-seed mathcal{V}_t/sqrt(d) values after each signed integral and positive part. Exactly zero Delta uses the explicit value-zero convention without a unit direction. Only t=2,...,T is represented, at chronological T-t=0,...,T-2, with no final-prediction or output padding. The same complete seed cohort defines all points and their fixed mean terminal-SSCD color. Finite nonnegative estimates remain included when sign assessment is unresolved; wrong-contract or missing values remain unavailable. Refined interval midpoints do not replace point estimates. This scalar reduction performs no integration, posterior evaluation or inference and does not certify exact integrals.",
    ["run_id", "original_index", "record_id", "target_id", "step_index", "mean_reference_variation_rmse",
     "mean_terminal_sscd", "seed_count", "seed_ids_json", "latent_dimension", "cohort_complete"],
    formula_version="reference-directional-variation-per-prompt-1", experiment=2,
    manuscript_results=["Requested directional revision of Proposition 5 variation"],
    value_column="mean_reference_variation_rmse", value_domain="nonnegative",
    source_scalar_column="direct_prop5_variation_rmse", prediction_domain="positive_noise_transitions",
    input_source=["saved_matched_updates", "saved_initial_terminal_outcomes"],
    reference_law="The declared finite atom-law posterior and saved cross-step numerical integration contract; not an identified complete training law",
    normalization="Arithmetic mean of saved per-seed mathcal{V}_t/sqrt(d); positive part follows each signed integral before seed averaging, and dimensional normalization is not repeated",
    measurement_contract="projected-gap-error-1",
    variation_definition="positive_part_after_integrated_unit_gap_projection",
    zero_gap_convention="mathcal{V}=0 when Delta is exactly zero; no unit direction",
    averaging_measure="Arithmetic mean of individual saved seed estimates on the same complete fixed cohort at every applicable positive-noise transition",
    averaging_unit="Retained prompt-target pair, preserving run/original-index/record/target identity",
    group_rule="All complete eligible prompt-target pairs; no SSCD split or variation-value selection",
    weighting="Equal seed mass within each prompt-target pair; the same seeds define mean terminal SSCD",
    band_definition="None; one seed-mean estimate per pair and transition, with numerical status in caption/audit",
    required_applicability="Complete fixed seed cohort at every applicable positive-noise transition, finite nonnegative directional positive-variation estimates with matching measurement-contract/definition and signed-integral receipts, true direct_prop5_applicable, an accepted integral status and matched terminal SSCD; unresolved condition signs or quadrature budget alone do not exclude estimates; the final t=1 prediction is outside the integral domain",
    measurement_audit_table="audit_data/feedback_endpoints.csv",
    accepted_integral_statuses=["estimated_converged", "numerically_unresolved", "analytic_single_atom_reference", "converged"],
    cohort_audit_table="audit_data/reference_variation_per_prompt_cohort.csv",
    colorbar_label="SSCD", sscd=True, update_index=True, include_zero=True, allow_unavailable=False,
)


_GUIDANCE_FIT = _four_stage(
    "corollary3_initial_cfg_amplification", "corollary3_guidance_scale_vs_loss", "appendix", "four_guidance_fit",
    "How does the fitted initial guidance coefficient vary with genuine conditional forward loss?",
    r"$\sqrt{\mathcal{L}_T(c)/(d\,\mathrm{SNR}_T)}$", r"$\widehat{g}$",
    "Corollary 3, Equation 54: one signed no-intercept least-squares coefficient per prompt-target pair minimizes the sum over the complete evaluation seed cohort of ||(hat{x}_T(c;g)-mu)-a(x_star-mu)||^2. With the same target direction for every seed, this joint coefficient equals the arithmetic mean of the individual signed seed fits. The saved x remains the genuine forward-target loss mean L_T(c)/(d*SNR_T). The display takes one square root of that saved ratio, after the loss has been averaged over draws; it is not a mean of per-draw roots. Color is mean terminal SSCD across exactly the same complete seed cohort. Fits use actual saved first-prediction states, with Gaussian-bank match status audited separately. The selected reference-based mu is used; neither zero centering nor a coefficient fitted to g*Delta_T is substituted. The horizontal reference is the configured guidance scale g; negative fits remain visible and are not clipped.",
    ["run_id", "original_index", "record_id", "target_id", "x", "y", "mean_terminal_sscd",
     "seed_count", "seed_ids_json", "latent_dimension", "snr", "guidance_scale",
     "fit_residual_rmse", "direction_norm_rmse"],
    formula_version="corollary3-guidance-fit-1", experiment=1, manuscript_results=["Corollary 3"],
    requires_theory_mean=True, sscd=True, colorbar_label="SSCD", equal=False, signed=True,
    include_zero=True, allow_unavailable=False,
    input_source=["saved_initial_generated_estimates", "saved_forward_target_loss_summary", "saved_initial_terminal_outcomes"],
    dependencies=["initial_four_stage", "forward_loss_summary", "initial_terminal_outcomes"],
    reference="Configured guidance scale",
    reference_law="The same selected theory-mean vector and receipt as the active mean comparisons; empirical reference scope and the single-target conditional assumption remain explicit.",
    normalization="Saved x is the forward squared-L2 loss mean divided once by d*SNR_T; displayed x is sqrt(saved x), with the root outside the draw mean. y is an unchanged signed dimensionless fitted coefficient; saved residual/direction norms use L2/sqrt(d)",
    saved_x_definition="L_T(c)/(d*SNR_T)", display_x_definition="sqrt(L_T(c)/(d*SNR_T))",
    averaging_measure="Joint no-intercept least squares over the full fixed seed cohort; equivalent to the arithmetic mean of signed seed fits with a common nonzero target direction",
    averaging_unit="One retained prompt-target pair, preserving run/original-index/record/target identity",
    group_rule="All complete eligible prompt-target pairs; no SSCD outcome split or fitted-value selection",
    weighting="Equal seed weight in the joint fit and mean terminal SSCD; genuine independent forward draws determine the pair loss",
    band_definition="None; one joint fit per pair and a horizontal configured-g guide",
    required_applicability="Complete fixed saved initial seed cohort, positive identifiable target-minus-selected-mean direction, valid vector-fit QA, genuine forward-target loss at matching initial SNR and mean provenance",
    auxiliary_definitions="The joint fit uses the full guided clean estimate minus selected mu and target minus selected mu; residual and direction-norm diagnostics retain the same fit population.",
    cohort_audit_table="audit_data/corollary3_guidance_scale_vs_loss_cohort.csv",
)
_GUIDANCE_FIT.pop("geometry_levels", None)


# Frozen descriptors for historical receipts and migration fixtures only.
# Neither the default selector nor diagnostics renders these retired companions.
_ZERO_BASELINE_FIGURES = [
    _four_stage("lemma2_initial_concentration", "initial_unconditional_mean_concentration_zero", "appendix", "ecdf",
        "How concentrated are the same initial unconditional estimates and declared-law atoms around the zero vector?",
        r"$\|\cdot-\mathbf{0}\|/\sqrt{d}$", "CDF",
        "Exact ECDFs of ||hat{x}_T(Z,empty)-0||/sqrt(d) for the same unique Gaussian seeds and ||u_j-0||/sqrt(d) for the same declared atoms. Seed weights are equal; atom weights retain their declared masses. This is an additive baseline comparison; the exact finite-bank mean and posterior remain unchanged independently of the selected reference-based mu.",
        ["distribution", "value", "cdf"], formula_version="zero-baseline-1",
        comparison_centre="zero", declared_law_mean_unchanged=True,
        direct_statement=False, manuscript_results=[], manuscript_label=None,
        manuscript_label_status="descriptive_comparison_not_a_new_manuscript_statement",
        evidence_classification="Descriptive zero-vector baseline on the existing reference law and Gaussian probe bank",
        zero_baseline_summary_table="audit_data/zero_baseline_summary.csv"),
    _four_stage("lemma2_unconditional_baseline", "unconditional_reference_convergence_zero", "appendix", "four_reference",
        "How do the same analytical and learned clean estimates compare with the zero vector across noise levels?",
        r"$\mathrm{SNR}_t$", r"Norm / $\sqrt{d}$",
        "Same saved posterior and Gaussian probe bank as the mean-centred comparison: ||bar{x}_t(Z,empty)-0||/sqrt(d), ||hat{x}_t(Z,empty)-0||/sqrt(d), and the unchanged e_t(Z,empty)/sqrt(d). The lower-SNR extension remains analytical only; native learned evaluations are unchanged. As SNR tends to zero, the reference-to-zero distance tends to the norm of the exact finite-bank weighted atom mean divided by sqrt(d), shown by the Reference limit guide; this is not the norm of the separately selected reference-based mu.",
        ["metric", "source_range", "step_index", "segment_id", "snr", "median", "q25", "q75", "minimum", "maximum"],
        formula_version="zero-baseline-1", comparison_centre="zero", declared_law_mean_unchanged=True,
        direct_statement=False, manuscript_results=[], manuscript_label=None,
        manuscript_label_status="descriptive_comparison_not_a_new_manuscript_statement",
        evidence_classification="Descriptive zero-vector baseline on the existing reference law and Gaussian probe bank",
        zero_baseline_summary_table="audit_data/zero_baseline_summary.csv"),
]

_COUNTERFACTUAL = _four_stage("lemma6_branch_target_errors", "counterfactual_unconditional_response", "appendix", "four_counterfactual",
    "At a fixed matched update, how does the actual empty-prompt learned predictor respond at the two endpoints?",
    "Counterfactual target error (RMSE)", "Actual-next target error (RMSE)",
    "Optional actual-network probe: I_net=(||m_u(next,x_cf)-x_star||-||m_u(next,x_next)-x_star||)/sqrt(d). Both endpoints are evaluated under the recorded compatible inference contract; this is not the empirical posterior reference or a new rollout. Fixed snapshots and exclusions are saved. Equality denotes zero measured I_net.",
    ["x", "y", "terminal_sscd", "step_index", "counterfactual_I_net"], sscd=True, optional_measurement=True,
    input_source=["learned_network_counterfactual_probe"], weighting="Unaggregated common-context learned endpoint pairs", allow_unavailable=True)


def previous_paper_registry(diagnostics=False, *, counterfactual=False):
    """Exactly four main and eleven mandatory appendix entries by default."""
    entries = copy.deepcopy(FOUR_STAGE_MAIN + FOUR_STAGE_APPENDIX)
    if counterfactual:
        entries.append(copy.deepcopy(_COUNTERFACTUAL))
    if diagnostics:
        existing = {entry["stem"] for entry in entries}
        for entry in evidence_registry(True, historical=True):
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


# Presentation-only curation: original source_table/formula_version objects above
# are retained for compatibility receipts and migration fixtures, never dispatch.
_RENDER_RETIREMENT_REASONS = {
    "branch_gap_motion_high_sscd": "Reviewed motion render retired; scalar accounting and identity audits remain mandatory",
    "branch_gap_motion_lower_sscd": "Reviewed motion render retired; scalar accounting and identity audits remain mandatory",
    "initial_unconditional_mean_concentration_zero": "Requested zero-vector concentration companion retired; saved scalar measurements and historical receipts retained; selected theory mean unchanged",
    "unconditional_reference_convergence_zero": "Requested zero-vector convergence companion retired; saved scalar measurements and historical receipts retained; selected theory mean unchanged",
}
RETIRED_RENDER_STEMS = frozenset(_RENDER_RETIREMENT_REASONS)
RENDER_RETIREMENTS = tuple({
    "figure_id": stem, "reason": _RENDER_RETIREMENT_REASONS[stem],
    "rendering_retired": True, "measurements_retained": True,
    "owned_paths": [f"{category}/{stem}.{extension}" for category in ("main", "appendix", "diagnostics") for extension in ("png", "pdf")],
} for stem in sorted(RETIRED_RENDER_STEMS))
MEASUREMENT_MAIN_ORDER = (
    "initial_loss_recovery", "branch_gap_posterior_response",
    "branch_gap_synchronization", "terminal_bound_coverage",
)
MEASUREMENT_APPENDIX_ORDER = (
    "initial_unconditional_mean_concentration", "unconditional_reference_convergence",
    "posterior_feedback_over_time", "posterior_feedback_condition_margin",
    "branch_target_errors", "synchronization_bound", "branch_gap_peak_step",
    "terminal_observable_bound", "final_reproduction_bound", "branch_gap_per_prompt",
    "conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt",
    "target_probability_per_prompt", "reference_branch_gap_per_prompt",
    "corollary3_guidance_scale_vs_loss", "reference_variation_per_prompt",
)
_SUPPORT = {
    "initial_loss_recovery": MEASUREMENT_APPENDIX_ORDER[:2],
    "branch_gap_posterior_response": MEASUREMENT_APPENDIX_ORDER[2:4],
    "branch_gap_synchronization": MEASUREMENT_APPENDIX_ORDER[4:7],
    "terminal_bound_coverage": MEASUREMENT_APPENDIX_ORDER[7:9],
}
_PRESENTATION_NOTES = {
    "reference_variation_per_prompt": "A16 averages saved per-seed mathcal{V}_t/sqrt(d), where mathcal{V} is the positive part after each signed directional integral. One full fixed cohort and mean terminal-SSCD color per pair; t=2,...,T only, with no terminal placeholder. The new scientific contract rejects old norm-integral caches. Numerical statuses remain auditable; estimates are not certified exact integrals.",
    "corollary3_guidance_scale_vs_loss": "A15 plots the full guided-clean least-squares coefficient from Corollary 3 Equation 54 against sqrt of the saved genuine forward loss divided by d*SNR_T, with the root outside the draw mean. The compact x column and scientific formula version are unchanged. One point uses the complete fixed seed cohort and its mean terminal SSCD. Selected mu is unchanged; the configured-g guide is labeled, negative fits remain, and direction degeneracy or incomplete cohorts are excluded with audit reasons.",
    "reference_branch_gap_per_prompt": "A14 is the norm of conditional minus unconditional analytical clean reference at the same saved state and noise. The single-target conditional reference equals x_star, making the saved opposite-sign unconditional-reference-to-target norm identical. Average individual normalized norms once over the complete fixed seed cohort; color uses the same seeds' mean terminal SSCD. No learned-gap, error-difference or radius-tail substitution is used.",
    "conditional_reference_error_per_prompt": "A11 retains saved individual normalized conditional-reference errors before seed averaging. Same fixed full seed cohort and mean terminal-SSCD color at every chronological prediction; the conditional reference retains the single-target assumption.",
    "unconditional_reference_error_per_prompt": "A12 measures learned unconditional clean estimates against their posterior clean reference, not against the target. Same fixed full seed cohort and mean terminal-SSCD color at every chronological prediction; no second dimensional normalization.",
    "target_probability_per_prompt": "A13 averages individual target probabilities obtained by exponentiating saved per-seed log probabilities on the actual generated states. The ordinate is not exponentiated mean log probability. It shares the complete-cohort and fixed mean terminal-SSCD color contract, with a fixed [0,1] probability axis.",
    "initial_loss_recovery": "Conditional and hollow unconditional points share pair identity and x coordinate; connectors do not assert equality. Monte Carlo intervals, initial SNR, and draw/seed counts are recorded. The pair-level color is mean terminal SSCD. The mean-concentration claim has separate supporting evidence in A1 and A2.",
    "branch_gap_posterior_response": "Fixed first-update retained contribution along the same displacement direction, not a change in learned gap norm or a guidance-scale rollout. Signed natural-log probability values use the saved symmetric-log threshold; medians/IQRs are descriptive. Positive initial feedback does not establish the original sufficient condition or positive feedback at every later transition; see A3 and A4.",
    "branch_gap_synchronization": "Common-population prompt-balanced medians of actual vector gap D (dashed) and paired maximum target error Q (solid); only Q has the displayed IQR. A5 first resolves branch timing; A6 supplies the complete bound and A7 individual peak timing. Small gap alone can be agreement away from the target. Display limits cover every plotted curve and band, not unplotted raw extrema.",
    "terminal_bound_coverage": "Each terminal-SSCD group (>0.75 or <=0.75) has three weighted fractions E<=tau, B_obs+delta_sched<=tau, and B_ref+delta_sched<=tau on identical eligible samples and weights within that group. Gold/purple identify groups; solid/dashed/dash-dot identify actual/observable/reference. All six curves share one tolerance scale. Expected ordering F_ref<=F_obs<=F_actual never substitutes for row-level checks; violations remain reported. Independent correction and original-clean versus deterministic/pathwise/probabilistic scope are retained. Latent tolerance is not decoded-copy identity or an SSCD-derived threshold; see paired comparisons A8 and A9.",
    "unconditional_reference_convergence": "The selected mu is estimated from unconditional posterior references at the analytical extension minimum SNR on a separate Gaussian bank by default. The horizontal guide marks the normalized exact-bank-to-selected-mean offset; its value and meaning remain in the caption and audit, without a legend entry. One fixed evaluation Gaussian probe bank is used across genuine native labels. The dotted analytical-only reference segment below SNR_T contains no network evaluations; solid reference and dashed/dotted learned quantities above initialization retain the complete saved native sweep. Empirical reference identification does not identify the checkpoint's complete training marginal.",
    "posterior_feedback_over_time": "Resolved strict-positive feedback and original-condition coverage use the same structural population and weights. Unknown signs remain in the denominator. The condition upper boundary is possible unresolved mass, not observed satisfaction or a statistical confidence interval. A shared zero curve is explicitly labeled only when both saved condition curves are identically zero; exact per-group positive and unresolved sample-transition counts remain in caption metadata.",
    "posterior_feedback_condition_margin": "All finite saved margin/gain pairs are ordinary dots, colored by terminal SSCD where available; no Observed/Unresolved legend, sign gate or sign-dependent marker is used. The combined plot retains all saved timesteps, with additional PNG/PDF views grouped only by chronological step_index k=T-t. Pooled point opacity is one-fiftieth of the shared scatter alpha; per-timestep point opacity is unchanged. Every saved step has a publication status; steps without finite pairs have no fabricated image, and no terminal-output value is inserted. Both zero guides, saved gain scale, numerical classifications and endpoint qualifications remain recorded. The 6% zero-inclusive margin padding is display space only.",
    "branch_target_errors": "Conditional and unconditional target-error medians and their descriptive IQRs use the same paired population at each prediction. Overlapping error norms do not prove vector equality, and group timing is not an every-seed ordering. Full displayed curves/bands share one scale and zero origin.",
    "synchronization_bound": "Only the actual vector gap D (solid) and complete Lemma 6 right-hand side S (dashed) are displayed as shared-weight medians, with both SSCD groups on one nonnegative scale. The paired maximum target-distance curve and its IQR are omitted. The legend sits outside above the plot, with SSCD groups in one column and the two quantities in the other. Raw tails, components and pointwise audits remain saved.",
    "branch_gap_per_prompt": "Each curve averages saved per-seed branch-gap norms over one complete fixed seed cohort; its color averages terminal SSCD over exactly that cohort. The normalized saved norms are not divided by sqrt(d) again. Incomplete or mismatched cohorts are disclosed in the saved cohort audit, never silently filled or reduced during plotting.",
    "branch_gap_peak_step": "Weighted bin mass of each complete trajectory's earliest resolved global maximum, not a density or first local peak. Full prediction range and late mass are retained; exact/near ties, flat/incomplete cases and group overlap remain recorded. A post-initial peak is not proof of unimodality or a categorical outcome dichotomy.",
    "terminal_observable_bound": "The abscissa is (B_obs+delta_sched)/sqrt(d), with B_obs=e_c+(g-1)||Delta||. Every applicable actual endpoint and loose or violated comparison remains visible, with exact zeros and the independent correction scope preserved.",
    "final_reproduction_bound": "The abscissa is (B_ref+delta_sched)/sqrt(d), with B_ref=g e_c+(g-1)[e_u^K+R_K(1-p_K)]. This is distinct from the tighter observable bound. All finite loose points, probability qualifications, zero cases and row-level applicability remain visible.",
}


def measurement_registry(diagnostics=False, *, counterfactual=False):
    """Comprehensive saved measurement inventory; it does not select exports."""
    # Optional learned probes are diagnostics only; saved enablement never adds a paper slot.
    previous = previous_paper_registry(diagnostics, counterfactual=False)
    by_stem = {entry["stem"]: entry for entry in previous}
    by_stem[_PROMPT_BRANCH_GAP["stem"]] = _PROMPT_BRANCH_GAP
    by_stem[_GROUPED_TERMINAL_COVERAGE["stem"]] = _GROUPED_TERMINAL_COVERAGE
    by_stem[_GUIDANCE_FIT["stem"]] = _GUIDANCE_FIT
    by_stem[_PROMPT_REFERENCE_VARIATION["stem"]] = _PROMPT_REFERENCE_VARIATION
    by_stem.update({entry["stem"]: entry for entry in _PROMPT_TRAJECTORY_FIGURES})
    slots = {stem: f"A{index}" for index, stem in enumerate(MEASUREMENT_APPENDIX_ORDER, 1)}
    entries = []
    for category, ordered in (("main", MEASUREMENT_MAIN_ORDER), ("appendix", MEASUREMENT_APPENDIX_ORDER)):
        for order, stem in enumerate(ordered, 1):
            entry = copy.deepcopy(by_stem[stem])
            entry.update(category=category, section=category, order=order,
                paper_slot=f"M{order}" if category == "main" else slots[stem],
                outputs={extension: f"{category}/{stem}.{extension}" for extension in ("png", "pdf")})
            entries.append(entry)
    active = set(MEASUREMENT_MAIN_ORDER + MEASUREMENT_APPENDIX_ORDER)
    for entry in previous:
        if entry["stem"] not in active and entry["stem"] not in RETIRED_RENDER_STEMS:
            entry = copy.deepcopy(entry)
            entry.update(section="supported_optional_diagnostic", order=len(entries) + 1, paper_slot=None)
            entries.append(entry)
    for entry in entries:
        _apply_branch_gap_error_contract(entry)
        stem = entry["stem"]
        entry.update(figure_id=stem, stable_stem=stem, plot_data_key=stem,
            renderer=entry["kind"], formula_id=f"{entry['formula_version']}:{stem}",
            plot_recipe_version=PLOT_RECIPE_VERSION,
            required_applicability=entry.get("required_applicability", "Recorded measurement availability and mathematical applicability; no fabricated fallback"),
            supporting_figure_ids=list(_SUPPORT.get(stem, ())),
            supporting_figure_slots=[slots[value] for value in _SUPPORT.get(stem, ())],
            presentation_note=(_BRANCH_GAP_ERROR_SCOPE if entry.get("measurement_contract") == BRANCH_GAP_ERROR_CONTRACT else _PRESENTATION_NOTES.get(stem, "Saved optional diagnostic; numerical definitions remain unchanged")))
        if stem == "synchronization_bound":
            entry["display_band_definition"] = "None; only the two median curves are displayed within each SSCD group."
            entry["semantic_question"] = "How does the branch-gap norm compare with the right-hand side of Lemma 6?"
            entry["presentation_note"] = _PRESENTATION_NOTES[stem] + " " + _BRANCH_GAP_ERROR_SCOPE
        if stem in active:
            entry["allow_unavailable"] = False
        if stem == "posterior_feedback_over_time":
            entry["required_columns"] = [*entry["required_columns"], "eligible_count"]
        if stem == "posterior_feedback_condition_margin":
            entry["required_columns"] = ["x", "y", "step_index", "terminal_sscd"]
            entry["display_policy"] = "finite_saved_values"
            entry["pooled_scatter_alpha_scale"] = 1.0 / 50.0
            entry["per_timestep_exports"] = {
                "column": "step_index",
                "directory": "appendix/posterior_feedback_condition_margin",
                "filename_template": "step_{step_index:03d}",
                "coordinate": "chronological k=T-t",
                "row_policy": "finite_saved_x_y",
            }
            entry["presentation_note"] = _PRESENTATION_NOTES[stem] + " " + _BRANCH_GAP_ERROR_SCOPE
        if stem in PAPER_AXES:
            entry["axes"] = copy.deepcopy(PAPER_AXES[stem])
            entry["x_definition"], entry["y_definition"] = entry["axes"]["x"], entry["axes"]["y"]
            entry["axis_notation"] = NOTATION_SCOPE
            entry["notation_details"] = NOTATION_DETAILS.get(stem, "All populations, weights and numerical classifications retain their saved definitions.")
            if stem == "initial_loss_recovery":
                entry["colorbar_label"] = "SSCD"
        if stem in {"initial_loss_recovery", "initial_unconditional_mean_concentration", "unconditional_reference_convergence"}:
            entry["requires_theory_mean"] = True
        if entry["kind"] == "four_reference":
            entry["display_range_policy"] = "all_displayed_curves_bands_and_saved_finite_bank_reference_limit"
        elif entry["kind"] == "four_prompt_chronological" and entry.get("value_domain") == "probability":
            entry["display_range_policy"] = "fixed_unit_interval"
        elif entry["kind"] in {"four_chronological", "four_prompt_chronological"}:
            entry["display_range_policy"] = "all_displayed_curves_and_bands_plus_fixed_6_percent_padding"
        elif entry["kind"] == "four_peak":
            entry["display_range_policy"] = "all_displayed_weighted_bin_masses_plus_fixed_6_percent_padding"
    return entries


PAPER_MAIN_ORDER = (
    "initial_loss_recovery",
    "unconditional_reference_convergence",
    "corollary3_guidance_scale_vs_loss",
    "posterior_feedback_condition_margin",
    "synchronization_bound",
    "terminal_bound_coverage",
)
PAPER_APPENDIX_ORDER = ()
PUBLICATION_OUTPUT_ALIASES = {"corollary3_guidance_scale_vs_loss": "guidance_scale_vs_loss"}


def paper_registry(diagnostics=False, *, counterfactual=False):
    """Select exactly six figures; optional measurement flags never add exports.

    Scientific stems, source tables, formula IDs and measurements are unchanged.
    The guidance filename alias is strictly a publication-path convention.
    """
    inventory = {entry["stem"]: entry for entry in measurement_registry()}
    slots = {stem: f"M{index}" for index, stem in enumerate(PAPER_MAIN_ORDER, 1)}
    entries = []
    for order, stem in enumerate(PAPER_MAIN_ORDER, 1):
        entry = copy.deepcopy(inventory[stem])
        output_stem = PUBLICATION_OUTPUT_ALIASES.get(stem, stem)
        support = [name for name in entry.get("supporting_figure_ids", ()) if name in slots]
        entry.update(category="main", section="main", order=order,
                     paper_slot=slots[stem], output_stem=output_stem,
                     outputs={extension: f"figures/{output_stem}.{extension}" for extension in ("png", "pdf")},
                     supporting_figure_ids=support,
                     supporting_figure_slots=[slots[name] for name in support],
                     allow_unavailable=False)
        entry.pop("per_timestep_exports", None)
        entry.pop("pooled_scatter_alpha_scale", None)
        if stem == "posterior_feedback_condition_margin":
            entry["pooled_scatter_alpha"] = 0.1
            entry["presentation_note"] = (
                "One pooled plot contains every finite saved margin/gain pair across all saved timesteps, "
                "with circular points, terminal-SSCD color where available, and fixed opacity 0.1. "
                "There are no status marker classes or per-timestep exports. Numerical classifications, "
                "zero guides and endpoint qualifications remain recorded. " + _BRANCH_GAP_ERROR_SCOPE)
            entry["formula"] = entry["formula"].replace("combined and per-timestep views", "pooled view")
        elif stem == "initial_loss_recovery":
            entry["presentation_note"] = (
                "Conditional and hollow unconditional points share pair identity and x coordinate. "
                "Monte Carlo intervals, initial SNR and draw/seed counts retain their saved definitions. "
                "Color is mean terminal SSCD; the selected-mean comparison is reported by unconditional_reference_convergence.")
        elif stem == "terminal_bound_coverage":
            entry["presentation_note"] = _PRESENTATION_NOTES[stem].replace("; see paired comparisons A8 and A9", "").replace("Gold/purple", "Coral/indigo") + " " + _BRANCH_GAP_ERROR_SCOPE
        elif stem == "synchronization_bound":
            entry["display_band_definition"] = (
                "Saved weighted 25th–75th percentiles for the branch gap and Lemma 6 right-hand side "
                "within each SSCD group and timestep; descriptive spread, not a confidence interval. "
                "Saved endpoints and segment boundaries are used without recomputation.")
            entry["presentation_note"] += (
                " Each displayed median has its saved interquartile band. The vertical range includes "
                "these endpoints under the existing full-display-range policy.")
        elif stem == "corollary3_guidance_scale_vs_loss":
            entry["presentation_note"] = _PRESENTATION_NOTES[stem].replace("A15 plots", "The figure plots")
        entries.append(entry)
    return entries

# Exact runtime migration allowlist. A path match alone never establishes ownership.
FIGURE_RETIREMENTS = (
    {"stem": "terminal_bound_coverage", "old_category": "appendix", "new_category": "main", "reason": "Promoted unchanged coverage measurement to the fourth main slot"},
    {"stem": "final_reproduction_bound", "old_category": "main", "new_category": "appendix", "reason": "Demoted unchanged reference-bound scatter to supporting figure A9"},
    *({"stem": stem, "old_category": category, "new_category": None,
       "reason": _RENDER_RETIREMENT_REASONS[stem]}
      for stem in sorted(RETIRED_RENDER_STEMS) for category in ("appendix", "main", "diagnostics")),
)
