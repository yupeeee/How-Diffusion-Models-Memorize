"""Saved-scalar renderer for four designated figures and optional diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
import numpy as np
import pandas as pd

from .contracts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    TheoryError,
    digest_file,
    read_object,
)

STYLE = {
    "figure.figsize": (4.0, 4.0),
    "savefig.dpi": 150,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 8,
    "text.usetex": False,
}
STYLE_VERSION = "theory-projected-gap-error-10"
DEFAULT_FIGURES = frozenset(
    {
        "initial_recovery",
        "unconditional_center",
        "posterior_feedback",
        "target_synchronization",
    }
)
DIAGNOSTIC_FIGURES = frozenset({"target_injection", "terminal_terms"})
FIGURE_ORDER = (
    "initial_recovery",
    "unconditional_center",
    "posterior_feedback",
    "target_synchronization",
    "target_injection",
    "terminal_terms",
)
KNOWN_STATEMENTS = frozenset(
    DEFAULT_FIGURES
    | {"target_injection", "matched_displacement", "terminal_reproduction"}
)
KEYS = ["run_id", "original_index", "record_id", "target_id", "seed"]
PROMPT_KEYS = KEYS[:-1]
COLOR = Normalize(0.0, 1.0)
FEEDBACK_LINTHRESH = 1e-3
SCATTER = {
    "initial_recovery": (
        "unconditional_target_error_rmse",
        "conditional_target_error_rmse",
        "Initial unconditional target error (RMSE)",
        "Initial conditional target error (RMSE)",
    ),
    "posterior_feedback": (
        "candidate_condition_margin_rmse",
        "candidate_log_probability_gain",
        r"$[\|\boldsymbol{\Delta}_t\|-\mathcal{E}_t-\mathcal{V}_t]/\sqrt{d}$",
        r"$\log[p_{t-1}(\mathbf{x}_{t-1})/p_{t-1}(\mathbf{x}_{t-1}^{\mathrm{cf}})]$",
    ),
    "target_injection": (
        "a_parallel",
        "off_target",
        "Target-direction coefficient",
        "Relative perpendicular component",
    ),
    "terminal_terms": (
        "terminal_A_rmse",
        "terminal_B_rmse",
        "Conditional target error (RMSE)",
        "Amplified branch gap (RMSE)",
    ),
}
FORMULAS = {
    "initial_recovery": {
        "x": "E_u,T = ||m_u,T - x_star|| / sqrt(d)",
        "y": "E_c,T = ||m_c,T - x_star|| / sqrt(d)",
    },
    "unconditional_center": {
        "x": "||m_u,T - mu_ref|| / sqrt(d)",
        "y": "F_n(a) = count(distance <= a) / n for unique evaluation seeds",
    },
    "posterior_feedback": {
        "x": r"$[\|\boldsymbol{\Delta}_t\|-\mathcal{E}_t-\mathcal{V}_t]/\sqrt{d}$",
        "branch_gap_error": r"$\mathcal{E}_t=\mathbf{u}_t^\top(\boldsymbol{\Delta}_t-\bar{\boldsymbol{\Delta}}_t)$",
        "measurement_contract": "projected-gap-error-1",
        "y": r"$\log[p_{t-1}(\mathbf{x}_{t-1})/p_{t-1}(\mathbf{x}_{t-1}^{\mathrm{cf}})]$",
        "variation": "mathcal_V_t = max(0, integral_0^1 (Delta_t/||Delta_t||).(bar_x_next(x_cf + s g kappa_t Delta_t) - bar_x_t(x_t)) ds); t=2,...,T",
    },
    "target_synchronization": {
        "x": "SNR_t = alpha_t^2 / sigma_t^2 at each stored prediction",
        "y": "E_b,t = ||m_b,t - x_star|| / sqrt(d), b in {c,u}",
    },
    "target_injection": {
        "x": "a_parallel = dot(g Delta_T, x_star - mu_ref) / ||x_star - mu_ref||^2",
        "y": "r_perp = ||g Delta_T - a_parallel (x_star - mu_ref)|| / ||x_star - mu_ref||",
    },
    "terminal_terms": {
        "x": "A = ||m_c,1 - x_star|| / sqrt(d)",
        "y": "B = (g - 1) ||Delta_1|| / sqrt(d)",
    },
}
CAPTIONS = {
    "initial_recovery": "Finite-noise initial recovery associated with terminal replication. Every retained prompt and evaluation seed contributes its initial prediction; color is that sample's terminal target SSCD. Equal target errors mean equal distances, not equal branch vectors. This does not measure the pair-specific forward-loss premise or convergence as initial SNR tends to zero.",
    "unconditional_center": "Initial unconditional dispersion around the reference center. One observation per unique evaluation seed around the explicitly declared fixed center. Repeated prompt evaluations do not multiply the sample count. Tight held-out model-output dispersion does not establish agreement with the training-data mean. These observations have no unique prompt-specific SSCD outcome.",
    "posterior_feedback": "Matched posterior feedback under the candidate distribution. The fixed finite candidate law is not the unidentified training law. The condition uses the signed projected branch-gap error mathcal{E}_t=u_t dot (Delta_t-bar{Delta}_t) and the positive part of the integrated signed reference projection; the outcome is a destination-level log-probability gain. Quadrature convergence supports numerical estimates, not formal certification. Unresolved rows are hollow; negative gains and unmet conditions remain. The single-target conditional idealization is not verified by a paired cache.",
    "target_synchronization": "Observed target-specific synchronization, conditional on the fixed seed bank. Branch/outcome-group curves show prompt-balanced medians with descriptive interquartile bands, not confidence intervals. Each represented prompt has equal total weight within an outcome group, divided among its member seeds. SSCD groups do not alter eligibility. Marginal medians do not establish small joint errors on the same samples or verify Lemma 6's Bayes-reference premises. The last point is the last stored prediction, not automatically a clean output; no extra terminal prediction is added.",
    "target_injection": "Finite-noise initial guidance projection around the explicitly recorded fixed center. Both axes are dimensionless. The point (g,0) is the limiting target-aligned reference; positive alignment alone does not establish agreement with that vector. Degenerate target-center directions remain undefined.",
    "terminal_terms": "The two terms of Equation 16 for structurally and numerically clean-update-applicable observations. Their vector sum may exhibit cancellation; the diagram does not verify Theorem 7's training-reference assumptions. A tolerance region is shown only when an independent latent tolerance is supplied.",
}


def _recompute_command(manifest=None):
    command = ["./run_all.sh", "--recompute-experiments"]
    config = (manifest or {}).get("config", {})
    for field, flag in (
        ("model_name", "--model"),
        ("scheduler_name", "--scheduler"),
        ("guidance_scale", "--g"),
        ("num_inference_steps", "--T"),
        ("num_seeds", "--N"),
        ("center", "--center"),
        ("cached_baseline", "--cached-baseline"),
        ("target_error_tolerance", "--target-error-tolerance"),
    ):
        if field in config:
            command.extend([flag, str(config[field])])
    return shlex.join(command)


def _safe_file(bundle, relative):
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise TheoryError(f"Unsafe scalar artifact path: {relative}")
    path = bundle / candidate
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(bundle.resolve())
    ):
        raise TheoryError(
            f"Missing/unsafe scalar artifact: {path}; run {_recompute_command()}."
        )
    if path.suffix not in {".json", ".parquet", ".csv"}:
        raise TheoryError(
            f"Plotting dependencies must be scalar tables or JSON, received: {path}"
        )
    return path


def _entries(registry):
    entries = registry.get("statements")
    if not isinstance(entries, list) or not entries:
        raise TheoryError("statement_registry.json requires a nonempty statements list")
    ids = [entry.get("semantic_id") for entry in entries]
    if len(set(ids)) != len(ids) or set(ids) != KNOWN_STATEMENTS:
        raise TheoryError(
            f"Incomplete/unknown/duplicate registered analysis inventory: {ids}"
        )
    main, diagnostics = set(), set()
    for entry in entries:
        if not entry.get("evidence_classification"):
            raise TheoryError(f"Registry missing evidence: {entry.get('semantic_id')}")
        for key, inventory in (
            ("main_figure", main),
            ("diagnostic_figure", diagnostics),
        ):
            figure = entry.get(key)
            if figure is not None:
                if (
                    not isinstance(figure, dict)
                    or not figure.get("id")
                    or figure["id"] in inventory
                ):
                    raise TheoryError(f"Invalid/duplicate registered figure: {figure}")
                inventory.add(figure["id"])
    if main != DEFAULT_FIGURES or diagnostics != DIAGNOSTIC_FIGURES:
        raise TheoryError(
            f"Incompatible designated figure inventory: main={sorted(main)}, diagnostics={sorted(diagnostics)}; run {_recompute_command()}."
        )
    return entries


def _load_plotdata(bundle, figure_id):
    path = _safe_file(bundle, f"plotdata/{figure_id}.parquet")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as error:
        raise TheoryError(f"Cannot read scalar Parquet {path}: {error}") from error
    required = set(KEYS + ["step_index"])
    if figure_id != "unconditional_center":
        required.add("terminal_sscd")
    if figure_id in SCATTER:
        required.update(SCATTER[figure_id][:2])
    if figure_id != "terminal_terms":
        required.add("snr")
    if figure_id == "unconditional_center":
        required.add("unconditional_center_rmse")
    elif figure_id == "posterior_feedback":
        required.update(
            {
                "candidate_variation_l2",
                "candidate_variation_error_l2",
                "candidate_condition_status",
                "candidate_gain_status",
                "feedback_eligible",
                "feedback_status",
            }
        )
    elif figure_id == "target_synchronization":
        required.update(
            {
                "conditional_target_error_rmse",
                "unconditional_target_error_rmse",
                "joint_target_error_rmse",
            }
        )
    elif figure_id == "terminal_terms":
        required.add("terminal_clean_applicable")
    missing = required - set(frame.columns)
    if missing:
        qualifier = (
            "The new Proposition-5 plot requires projected branch-gap error and directional positive-part V; old candidate margins cannot be relabeled. "
            if figure_id == "posterior_feedback"
            else ""
        )
        raise TheoryError(
            f"{path} lacks scalar columns: {sorted(missing)}. {qualifier}Run {_recompute_command(read_object(bundle / 'manifest.json'))}."
        )
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS + ["step_index"]).any():
        raise TheoryError(f"Duplicate/missing scientific observation keys: {path}")
    if (
        not frame.empty
        and not frame.original_index.map(lambda x: isinstance(x, str)).all()
    ):
        raise TheoryError(f"original_index must remain a string: {path}")
    if (
        figure_id in {"initial_recovery", "target_injection", "unconditional_center"}
        and not frame.step_index.eq(0).all()
    ):
        raise TheoryError(f"Initial figure contains noninitial observations: {path}")
    if (
        figure_id == "unconditional_center"
        and frame.duplicated(["run_id", "seed"]).any()
    ):
        raise TheoryError(
            f"Initial baseline must contain one canonical observation per unique evaluation seed: {path}"
        )
    return frame


def validate_bundle(bundle, expected_config=None):
    """Validate all scalar hashes and six figure contracts before publication."""
    bundle = Path(bundle).resolve()
    manifest = read_object(_safe_file(bundle, "manifest.json"))
    if (
        not manifest.get("complete")
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("formula_version") != FORMULA_VERSION
    ):
        raise TheoryError(
            f"Incomplete/incompatible scalar manifest: {bundle}/manifest.json. Old feedback scalars lack projected branch-gap error and directional positive-part V and cannot be relabeled. Run {_recompute_command(manifest)}."
        )
    if expected_config is not None and manifest.get("config") != expected_config:
        raise TheoryError(f"Scalar bundle configuration differs: {bundle}")
    numerical = manifest.get("numerical_files")
    if not isinstance(numerical, dict):
        raise TheoryError(f"Missing numerical file hashes: {bundle}/manifest.json")
    required = {
        "statement_registry.json",
        "source_records.parquet",
        "schedule.csv",
        "center_metadata.json",
        "support_metadata.json",
        "endpoint_metrics.parquet",
        "initial_metrics.parquet",
        "summary.json",
        "failed.csv",
        "applicability_report.json",
    }
    required.update(f"plotdata/{figure_id}.parquet" for figure_id in FIGURE_ORDER)
    required.update(
        f"summaries/{name}.parquet"
        for name in (
            "synchronization",
            "paired_synchronization",
            "phases",
            "initial_by_prompt",
            "feedback_by_step",
            "feedback_by_prompt",
        )
    )
    if manifest.get("config", {}).get("target_error_tolerance") is not None:
        required.add("summaries/tolerance_entries.parquet")
    if required - numerical.keys():
        raise TheoryError(
            f"Scalar manifest lacks required artifact hashes: {sorted(required - numerical.keys())}; run {_recompute_command(manifest)}."
        )
    for relative, expected in numerical.items():
        path = _safe_file(bundle, relative)
        if not isinstance(expected, str) or digest_file(path) != expected:
            raise TheoryError(
                f"Changed/incompatible numerical artifact: {path}; run {_recompute_command(manifest)}."
            )
    entries = _entries(read_object(bundle / "statement_registry.json"))
    if "statement_ids" in manifest and set(manifest["statement_ids"]) != {
        e["semantic_id"] for e in entries
    }:
        raise TheoryError(
            "Manifest and registry scientific analysis inventories differ"
        )
    config, counts, reference_keys = (
        manifest.get("config", {}),
        manifest.get("expected_counts", {}),
        None,
    )
    for figure_id in FIGURE_ORDER:
        frame = _load_plotdata(bundle, figure_id)
        relative = f"plotdata/{figure_id}.parquet"
        if frame.run_id.nunique() > 1 or (
            manifest.get("scientific_generation_hash")
            and not frame.run_id.eq(manifest["scientific_generation_hash"]).all()
        ):
            raise TheoryError(f"Scientific run identity differs in {relative}")
        baseline = figure_id == "unconditional_center"
        per_sample = figure_id in {
            "initial_recovery",
            "target_injection",
            "terminal_terms",
        }
        count_key = (
            "endpoint_rows"
            if figure_id == "terminal_terms"
            else "initial_rows" if per_sample else "trajectory_rows"
        )
        expected_count = config.get("num_seeds") if baseline else counts.get(count_key)
        if expected_count is not None and len(frame) != expected_count:
            raise TheoryError(
                f"Missing/extra scalar observations in {relative}: expected {expected_count}, found {len(frame)}"
            )
        if not baseline:
            if (
                "prompts" in counts
                and frame[PROMPT_KEYS].drop_duplicates().shape[0] != counts["prompts"]
            ):
                raise TheoryError(f"Missing/extra selected prompts in {relative}")
            sample_keys = set(frame[KEYS].itertuples(index=False, name=None))
            if reference_keys is None:
                reference_keys = sample_keys
            elif sample_keys != reference_keys:
                raise TheoryError(f"Figure observation identities differ: {relative}")
        if "include_prompt" in frame and not frame.include_prompt.eq(True).all():
            raise TheoryError(
                f"Figure contains observations outside frozen selection: {relative}"
            )
        if config.get("num_seeds"):
            expected_seeds = set(range(int(config["num_seeds"])))
            if baseline:
                if set(frame.seed) != expected_seeds:
                    raise TheoryError(f"Incomplete experiment seed bank: {relative}")
            else:
                for _, group in frame.groupby(PROMPT_KEYS, sort=False):
                    if set(group.seed) != expected_seeds:
                        raise TheoryError(
                            f"Incomplete experiment seed bank: {relative}"
                        )
                    if not per_sample:
                        expected_steps = set(range(int(config["num_inference_steps"])))
                        if any(
                            set(rows.step_index) != expected_steps
                            for _, rows in group.groupby("seed")
                        ):
                            raise TheoryError(
                                f"Incomplete trajectory scalar rows: {relative}"
                            )
    return manifest


def trajectory_segments(frame, y_column):
    """Retained diagnostic helper: never connect seeds or bridge missing steps."""
    output = []
    for _, rows in frame.groupby(
        [key for key in KEYS if key in frame], sort=False, dropna=False
    ):
        rows = rows.sort_values("step_index", kind="stable")
        steps, xy = rows.step_index.to_numpy(dtype=int), rows[
            ["snr", y_column]
        ].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] > 0)
        cuts = np.r_[True, np.diff(steps) != 1] | ~valid | np.r_[False, ~valid[:-1]]
        for section in np.split(np.arange(len(rows)), np.flatnonzero(cuts)[1:]):
            section = section[valid[section]]
            if len(section):
                output.append((xy[section], float(rows.iloc[section[0]].terminal_sscd)))
    return output


def _exclusions(frame, valid):
    reasons = {}
    for column in frame:
        if column.endswith("_status") or column.endswith("_reason"):
            for value, count in (
                frame.loc[~valid, column]
                .fillna("missing")
                .astype(str)
                .value_counts()
                .items()
            ):
                reasons[f"{column}: {value}"] = int(count)
    if (~valid).any() and not reasons:
        reasons["nonfinite, unavailable or inapplicable plotted quantity"] = int(
            (~valid).sum()
        )
    return {
        "total_rows": len(frame),
        "plotted_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "reasons": reasons,
    }


def _finite_initial_snr(frame):
    if not {"step_index", "snr"} <= set(frame):
        return []
    return sorted(
        float(value)
        for value in frame.loc[frame.step_index.eq(0), "snr"].unique()
        if np.isfinite(value)
    )


def _annotation(ax, lines):
    ax.text(
        0,
        1.025,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        linespacing=1.1,
    )


def _snr_text(values):
    return (
        r"$\mathrm{SNR}_T$=" + ", ".join(f"{value:.3g}" for value in values)
        if values
        else r"$\mathrm{SNR}_T$: unavailable"
    )


def _feedback_summary(frame, valid):
    eligible = frame.feedback_eligible.fillna(False).astype(bool)
    statuses = frame.candidate_condition_status.fillna("unavailable").astype(str)
    gain = frame.candidate_gain_status.fillna("unavailable").astype(str)
    met = eligible & statuses.isin(
        ["estimated_met", "conservatively_met", "resolved_met"]
    )
    unresolved = eligible & statuses.eq("numerically_unresolved")

    def signs(mask):
        return {
            name: int((mask & gain.eq(name)).sum())
            for name in (
                "positive",
                "negative",
                "zero",
                "numerically_unresolved",
                "unavailable",
            )
        }

    count = int(eligible.sum())
    return {
        "eligible_count": count,
        "total_count": len(frame),
        "condition_met_count": int(met.sum()),
        "condition_coverage": float(met.sum() / count) if count else None,
        "condition_unresolved_count": int(unresolved.sum()),
        "unplottable_count": int((~valid).sum()),
        "eligible_unplottable_count": int((eligible.to_numpy() & ~valid).sum()),
        "ineligible_count": int((~eligible).sum()),
        "gain_sign_counts": signs(eligible),
        "conditional_gain_sign_counts": signs(met),
        "condition_status_counts": {
            str(k): int(v) for k, v in statuses.value_counts().items()
        },
        "coverage_interpretation": (
            "numerically estimated; quadrature convergence is not a formal certificate"
            if statuses.str.startswith("estimated_").any()
            else "see saved condition statuses and numerical policy"
        ),
    }


def _add_colorbar(fig, ax):
    bar = fig.colorbar(
        ScalarMappable(norm=COLOR, cmap="viridis"), ax=ax, pad=0.02, fraction=0.046
    )
    bar.set_label("SSCD", fontsize=12)
    bar.ax.tick_params(labelsize=10)
    if bar.solids is not None:
        bar.solids.set_alpha(1.0)


def _render_synchronization(ax, frame):
    from .summaries import synchronization_summary

    summary = synchronization_summary(frame)
    group_colors = {
        "SSCD > 0.75": plt.get_cmap("viridis")(0.8),
        "SSCD <= 0.75": plt.get_cmap("viridis")(0.2),
    }
    branches = {"conditional": "-", "unconditional": "--"}
    all_steps = (
        np.arange(int(frame.step_index.min()), int(frame.step_index.max()) + 1)
        if len(frame)
        else []
    )
    groups = {}
    for group, color in group_colors.items():
        mask = (
            frame.terminal_sscd.gt(0.75)
            if group == "SSCD > 0.75"
            else frame.terminal_sscd.le(0.75)
        )
        group_rows = frame.loc[mask]
        groups[group] = {
            "prompt_count": int(group_rows[PROMPT_KEYS].drop_duplicates().shape[0]),
            "seed_count": int(group_rows[KEYS].drop_duplicates().shape[0]),
        }
        for branch, linestyle in branches.items():
            rows = summary.loc[
                summary.group.eq(group) & summary.branch.eq(branch)
            ].sort_values("step_index")
            if rows.empty:
                continue
            rows = rows.set_index("step_index").reindex(all_steps)
            x, y = rows.snr.to_numpy(dtype=float), rows.error_median.to_numpy(
                dtype=float
            )
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                lw=1.2,
                label=f"{group.replace('<=', '≤')}; {branch}",
            )
            ax.fill_between(
                x,
                rows.error_q25.to_numpy(dtype=float),
                rows.error_q75.to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
                rasterized=True,
            )
    branch_values = frame[
        ["conditional_target_error_rmse", "unconditional_target_error_rmse"]
    ].to_numpy(dtype=float)
    valid = (
        np.isfinite(frame.snr.to_numpy(dtype=float))
        & frame.snr.gt(0).to_numpy()
        & np.isfinite(frame.terminal_sscd.to_numpy(dtype=float))
        & np.isfinite(branch_values).any(axis=1)
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{SNR}_t$")
    ax.set_ylabel("Target error (RMSE)")
    source = (
        frame.loc[np.isfinite(frame.snr) & frame.snr.gt(0)]
        .groupby("step_index", sort=True)
        .snr.first()
    )
    if len(source) > 1 and source.iloc[0] > source.iloc[-1]:
        ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    references = []
    for snr in _finite_initial_snr(frame):
        if snr > 0:
            label = r"Initial SNR: $\mathrm{SNR}_T$"
            ax.axvline(snr, color="0.35", ls=":", lw=0.9, label=label, zorder=4)
            references.append({"kind": "vertical", "label": label, "value": snr})
    _annotation(
        ax,
        [
            f"{group.replace('<=', '≤')}: {item['prompt_count']} prompts, {item['seed_count']} samples"
            for group, item in groups.items()
        ],
    )
    return (
        valid,
        {
            "outcome_groups": groups,
            "aggregation": "prompt-balanced weighted median and interquartile range",
            "summary_rows": json.loads(summary.to_json(orient="records")),
            "summary_interpretation": "Descriptive; no confidence intervals",
        },
        references,
    )


def build_figure(frame, semantic_id, entry, manifest):
    """Build one saved-scalar design, retaining its formulas and scope in the audit."""
    if semantic_id not in DEFAULT_FIGURES | DIAGNOSTIC_FIGURES:
        raise TheoryError(
            f"No figure is designated for {semantic_id}; matched displacement remains numerical QA."
        )
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.grid(True, alpha=0.18, linewidth=0.5)
        references, details = [], {}
        initial_snr = _finite_initial_snr(frame)
        if semantic_id in SCATTER:
            x, y, xlabel, ylabel = SCATTER[semantic_id]
            xy, colors = frame[[x, y]].to_numpy(
                dtype=float
            ), frame.terminal_sscd.to_numpy(dtype=float)
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(colors)
            if semantic_id == "terminal_terms":
                valid &= frame.terminal_clean_applicable.fillna(False).to_numpy(
                    dtype=bool
                )
            unresolved = np.zeros(len(frame), dtype=bool)
            if semantic_id == "posterior_feedback":
                valid &= frame.feedback_eligible.fillna(False).to_numpy(dtype=bool)
                unresolved = (
                    frame.candidate_condition_status.eq(
                        "numerically_unresolved"
                    ).to_numpy()
                    | frame.candidate_gain_status.eq(
                        "numerically_unresolved"
                    ).to_numpy()
                )
            definite = valid & ~unresolved
            ax.scatter(
                xy[definite, 0],
                xy[definite, 1],
                c=colors[definite],
                cmap="viridis",
                norm=COLOR,
                s=10,
                alpha=0.4,
                linewidths=0,
                rasterized=True,
            )
            if np.any(valid & unresolved):
                ax.scatter(
                    xy[valid & unresolved, 0],
                    xy[valid & unresolved, 1],
                    facecolors="none",
                    edgecolors="0.5",
                    s=12,
                    linewidths=0.5,
                    alpha=0.5,
                    rasterized=True,
                    label="Numerically unresolved",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if semantic_id == "initial_recovery":
                upper = (
                    max(float(xy[valid].max()) * 1.05, 1e-12) if valid.any() else 1.0
                )
                ax.set_xlim(0, upper)
                ax.set_ylim(0, upper)
                ax.set_aspect("equal", adjustable="box")
                label = "Equal target errors"
                ax.plot(
                    [0, upper],
                    [0, upper],
                    color="0.35",
                    lw=0.8,
                    linestyle="--",
                    label=label,
                    zorder=3,
                )
                references.append(
                    {"kind": "diagonal", "label": label, "slope": 1, "intercept": 0}
                )
                _annotation(
                    ax,
                    [
                        f"{_snr_text(initial_snr)}; {len(frame):,} samples, {frame[PROMPT_KEYS].drop_duplicates().shape[0]:,} prompts"
                    ],
                )
            elif semantic_id == "posterior_feedback":
                ax.set_yscale("symlog", base=10, linthresh=FEEDBACK_LINTHRESH)
                for orientation, label in (
                    ("vertical", "Zero condition margin"),
                    ("horizontal", "Zero log-probability gain"),
                ):
                    getattr(ax, "axvline" if orientation == "vertical" else "axhline")(
                        0,
                        color="0.25",
                        lw=0.85,
                        ls=":" if orientation == "vertical" else "--",
                        label=label,
                        zorder=4,
                    )
                    references.append({"kind": orientation, "label": label, "value": 0})
                details = _feedback_summary(frame, valid)
                signs = details["conditional_gain_sign_counts"]
                _annotation(
                    ax,
                    [
                        f"Estimated condition coverage: {details['condition_met_count']:,}/{details['eligible_count']:,}",
                        f"When met: + {signs['positive']:,}, − {signs['negative']:,}, 0 {signs['zero']:,}, ? {signs['numerically_unresolved'] + signs['unavailable']:,}",
                        f"Unresolved condition: {details['condition_unresolved_count']:,}; unplottable: {details['unplottable_count']:,}",
                    ],
                )
                details["symlog_base"], details["symlog_linthresh"] = (
                    10,
                    FEEDBACK_LINTHRESH,
                )
            elif semantic_id == "target_injection":
                guidance = float(manifest["config"]["guidance_scale"])
                label = "Target-aligned reference: $(g,0)$"
                ax.scatter(
                    [guidance],
                    [0],
                    marker="x",
                    color="black",
                    s=35,
                    label=label,
                    zorder=4,
                )
                references.append(
                    {"kind": "point", "label": label, "x": guidance, "y": 0}
                )
                label = "Zero target projection"
                ax.axvline(0, color="0.4", lw=0.8, ls=":", label=label, zorder=4)
                references.append({"kind": "vertical", "label": label, "value": 0})
                ax.set_ylim(bottom=0)
            else:
                if valid.any():
                    ax.set_xlim(
                        min(0, float(xy[valid, 0].min()) * 1.05),
                        max(1e-12, float(xy[valid, 0].max()) * 1.05),
                    )
                    ax.set_ylim(
                        min(0, float(xy[valid, 1].min()) * 1.05),
                        max(1e-12, float(xy[valid, 1].max()) * 1.05),
                    )
                else:
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)
                tau = manifest.get("config", {}).get("target_error_tolerance")
                guidance = float(
                    manifest.get("config", {}).get("guidance_scale", np.nan)
                )
                details["terminal_tolerance_region_status"] = "omitted"
                details["terminal_tolerance_region_reason"] = (
                    "independent_latent_tolerance_not_supplied"
                    if tau is None
                    else (
                        "signed_B_is_not_a_norm_bound_when_guidance_below_one"
                        if not np.isfinite(guidance) or guidance < 1
                        else "invalid_tolerance_or_unavailable_unique_latent_dimension"
                    )
                )
                if (
                    tau is not None
                    and np.isfinite(guidance)
                    and guidance >= 1
                    and "latent_dimension" in frame
                    and frame.latent_dimension.nunique() == 1
                    and np.isfinite(float(tau))
                    and float(tau) >= 0
                    and float(frame.latent_dimension.iloc[0]) > 0
                ):
                    details["terminal_tolerance_region_status"] = "shown"
                    details["terminal_tolerance_region_reason"] = (
                        "independently_supplied_raw_L2_tolerance_and_nonnegative_amplification"
                    )
                    threshold = float(tau) / np.sqrt(
                        float(frame.latent_dimension.iloc[0])
                    )
                    label = r"$A+B=\tau/\sqrt{d}$"
                    ax.plot(
                        [0, threshold],
                        [threshold, 0],
                        ls="--",
                        lw=0.8,
                        color="0.3",
                        label=label,
                    )
                    ax.fill_between(
                        [0, threshold], [threshold, 0], color="0.5", alpha=0.08
                    )
                    references.append(
                        {
                            "kind": "tolerance",
                            "label": label,
                            "rmse_threshold": threshold,
                        }
                    )
            _add_colorbar(fig, ax)
        elif semantic_id == "unconditional_center":
            x, y = "unconditional_center_rmse", "empirical_cumulative_fraction"
            if (
                frame.duplicated(["run_id", "seed"]).any()
                or not frame.step_index.eq(0).all()
            ):
                plt.close(fig)
                raise TheoryError(
                    "Initial baseline requires exactly one canonical initial observation per unique evaluation seed"
                )
            values = frame[x].to_numpy(dtype=float)
            valid = np.isfinite(values)
            distances = np.sort(values[valid], kind="stable")
            if len(distances):
                ax.step(
                    np.r_[0, distances],
                    np.r_[0, np.arange(1, len(distances) + 1) / len(frame)],
                    where="post",
                    color=plt.get_cmap("viridis")(0.35),
                    lw=1.4,
                )
                details["distance_quantiles"] = {
                    str(q): float(np.quantile(distances, q))
                    for q in (0, 0.25, 0.5, 0.75, 0.9, 1)
                }
            ax.set_xlabel("Initial distance to reference center (RMSE)")
            ax.set_ylabel("Empirical cumulative fraction")
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.02)
            _annotation(
                ax,
                [
                    f"N={frame.seed.nunique()} unique evaluation seeds; {_snr_text(initial_snr)}"
                ],
            )
            details["observation_unit"] = (
                "unique evaluation seed at the actual initial state"
            )
            details["unavailable_seed_count"] = int((~valid).sum())
            details["cdf_denominator"] = len(frame)
            if not valid.all():
                ax.text(
                    0,
                    0.98,
                    f"{int((~valid).sum())} unavailable seed observations",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
        else:
            x, y = "snr", [
                "conditional_target_error_rmse",
                "unconditional_target_error_rmse",
            ]
            valid, details, references = _render_synchronization(ax, frame)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="best", framealpha=0.85)
        figure_spec = entry.get("main_figure") or entry.get("diagnostic_figure") or {}
        formulas = dict(FORMULAS[semantic_id])
        caption = CAPTIONS[semantic_id]
        if semantic_id in {"unconditional_center", "target_injection"}:
            center_choice = manifest.get("config", {}).get(
                "center", "reference-initial"
            )
            symbol, definition = {
                "reference-initial": (
                    "mu_ref",
                    "mu_ref is the fixed mean of cached initial unconditional predictions from the disjoint reference seed block, with one observation per distinct seed; it is not identified with the training mean.",
                ),
                "zero": (
                    "0",
                    "The selected center is the literal zero vector; no center is estimated and no zero training mean is asserted.",
                ),
                "cached-baseline": (
                    "mu_saved",
                    "mu_saved denotes the explicitly selected compatible, independently estimated cached model-output center; it is not identified with the training mean.",
                ),
            }[center_choice]
            formulas = {
                key: value.replace("mu_ref", symbol) for key, value in formulas.items()
            }
            formulas["center_definition"] = definition
            details["center_symbol"] = symbol
            details["center_source"] = (
                manifest.get("config", {}).get("cached_baseline")
                if center_choice == "cached-baseline"
                else center_choice
            )
            caption += " " + definition
        audit = _exclusions(frame, valid)
        audit.update(
            xlim=list(ax.get_xlim()),
            ylim=list(ax.get_ylim()),
            xscale=ax.get_xscale(),
            yscale=ax.get_yscale(),
            main_axes_inches=[4, 4],
            evidence_classification=figure_spec.get(
                "evidence_classification",
                entry.get("evidence_classification", "unavailable"),
            ),
            initial_snr=initial_snr,
            prompt_count=int(frame[PROMPT_KEYS].drop_duplicates().shape[0]),
            initial_effective_seed_count=(
                int(frame.seed.nunique()) if initial_snr else None
            ),
            axis_labels={"x": ax.get_xlabel(), "y": ax.get_ylabel()},
            scalar_columns={"x": x, "y": y},
            references=references,
            colorbar_label="SSCD" if semantic_id in SCATTER else None,
            legend_labels=(
                [text.get_text() for text in ax.get_legend().get_texts()]
                if ax.get_legend() is not None
                else []
            ),
            formulas=formulas,
            caption=caption,
            visual_subsampling=False,
            center_choice=(
                manifest.get("config", {}).get("center")
                if semantic_id in {"unconditional_center", "target_injection"}
                else None
            ),
            **details,
        )
        fig.tight_layout(pad=0.5)
        return fig, audit


def _publish_figure(fig, destination):
    fd, name = tempfile.mkstemp(
        prefix="." + destination.stem + "-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(name)
    try:
        fig.savefig(
            temporary,
            format=destination.suffix[1:],
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=[
                label
                for axis in fig.axes
                for label in (axis.xaxis.label, axis.yaxis.label, *axis.texts)
            ],
            pad_inches=0.05,
            metadata=(
                {"CreationDate": None, "ModDate": None}
                if destination.suffix == ".pdf"
                else None
            ),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _unavailable_figure_status(figure_id, frame):
    """Do not conflate a missing reference, an inapplicable premise and numerical uncertainty."""
    if figure_id == "posterior_feedback" and len(frame):
        condition = frame.candidate_condition_status.fillna("unavailable")
        eligible = frame.feedback_eligible.fillna(False).astype(bool)
        if eligible.any() and condition.eq("numerically_unresolved").any():
            return {
                "status": "numerically_unresolved",
                "reason": "Eligible candidate comparisons have no plottable coordinates and unresolved numerical measurements.",
            }
        if not eligible.any() and condition.eq("not_applicable").all():
            return {
                "status": "not_applicable",
                "reason": "The saved feedback applicability conditions are not satisfied; see per-row reasons.",
            }
    if figure_id == "terminal_terms" and len(frame):
        structural = (
            frame.get("terminal_clean_structural", frame.terminal_clean_applicable)
            .fillna(False)
            .astype(bool)
        )
        applicable = frame.terminal_clean_applicable.fillna(False).astype(bool)
        if not structural.any():
            return {
                "status": "not_applicable",
                "reason": "No saved terminal update satisfies the structural clean-update conditions of Equation 5.",
            }
        if (structural & ~applicable).any():
            return {
                "status": "numerically_unresolved",
                "reason": "Structurally clean terminal updates failed their numerical residual checks; no endpoint theorem diagram is substituted.",
            }
    return {
        "status": "unavailable",
        "reason": "No finite eligible observations; inspect per-metric availability and the saved reference-availability report.",
    }


def _old_managed_files(figures):
    path = figures / "manifest.json"
    if not path.is_file() or path.is_symlink():
        return set()
    previous = read_object(path)
    names = set(previous.get("owned_files", []))
    for record in previous.get("figures", {}).values():
        for item in record.get("formats", {}).values():
            if isinstance(item, dict) and "path" in item:
                names.add(item["path"])
    return {
        name
        for name in names
        if isinstance(name, str)
        and Path(name).name == name
        and Path(name).suffix in {".png", ".pdf", ".json"}
        and name != "manifest.json"
    }


def render_bundle(bundle, include_diagnostics=False):
    """Render four designs plus requested diagnostics, staging all writes first.

    Only obsolete images explicitly listed in the previous managed manifest are
    archived. Unrelated files and numerical artifacts are never changed here.
    """
    bundle = Path(bundle).resolve()
    manifest = validate_bundle(bundle)
    entries = _entries(read_object(bundle / "statement_registry.json"))
    by_figure = {
        spec["id"]: entry
        for entry in entries
        for spec in (entry.get("main_figure"), entry.get("diagnostic_figure"))
        if spec is not None
    }
    selected = [
        name for name in FIGURE_ORDER if name in DEFAULT_FIGURES or include_diagnostics
    ]
    figures = bundle / "figures"
    style_hash = hashlib.sha256(
        json.dumps(
            {"style": STYLE, "renderer_sha256": digest_file(Path(__file__))},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result = {
        "analysis_hash": manifest.get("analysis_hash"),
        "style_version": STYLE_VERSION,
        "style_hash": style_hash,
        "statement_registry_sha256": digest_file(bundle / "statement_registry.json"),
        "requested_figures": selected,
        "include_diagnostics": bool(include_diagnostics),
        "figures": {},
        "unavailable_figures": {},
        "owned_files": [],
    }
    with tempfile.TemporaryDirectory(prefix=".figure-stage-", dir=bundle) as name:
        staging = Path(name)
        for figure_id in selected:
            frame = _load_plotdata(bundle, figure_id)
            fig, audit = build_figure(frame, figure_id, by_figure[figure_id], manifest)
            try:
                if audit["plotted_rows"] == 0:
                    result["unavailable_figures"][figure_id] = (
                        audit | _unavailable_figure_status(figure_id, frame)
                    )
                    continue
                formats = {}
                for extension in ("png", "pdf"):
                    destination = staging / f"{figure_id}.{extension}"
                    _publish_figure(fig, destination)
                    formats[extension] = {
                        "path": destination.name,
                        "sha256": digest_file(destination),
                    }
                    result["owned_files"].append(destination.name)
            finally:
                plt.close(fig)
            item = audit | {
                "formats": formats,
                "data_sha256": manifest["numerical_files"][
                    f"plotdata/{figure_id}.parquet"
                ],
                "center_metadata_sha256": manifest["numerical_files"][
                    "center_metadata.json"
                ],
                "support_metadata_sha256": manifest["numerical_files"][
                    "support_metadata.json"
                ],
            }
            result["figures"][figure_id] = item
            sidecar = staging / f"{figure_id}.json"
            sidecar.write_text(
                json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            result["owned_files"].append(sidecar.name)
        (staging / "applicability.json").write_text(
            json.dumps(
                {
                    "requested_figures": selected,
                    "rendered_figures": list(result["figures"]),
                    "unavailable_figures": result["unavailable_figures"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        result["owned_files"].append("applicability.json")
        managed = _old_managed_files(figures)
        collisions = [
            name
            for name in result["owned_files"]
            if (figures / name).exists() and name not in managed
        ]
        if collisions:
            raise TheoryError(
                f"Refusing to overwrite unrelated/unmanaged figure files: {collisions}"
            )
        if figures.is_symlink() or (figures / "manifest.json").is_symlink():
            raise TheoryError(f"Unsafe managed figure directory or manifest: {figures}")
        figures.mkdir(exist_ok=True)
        obsolete = managed - set(result["owned_files"])
        if obsolete:
            archive = figures / "archive" / style_hash[:12]
            archive.mkdir(parents=True, exist_ok=True)
            for filename in sorted(obsolete):
                path = figures / filename
                if path.is_file() and not path.is_symlink():
                    destination = archive / filename
                    if destination.exists():
                        destination = (
                            archive
                            / f"{path.stem}-{digest_file(path)[:12]}{path.suffix}"
                        )
                    if destination.exists():
                        # An identical copy is already retained in the archive.
                        if digest_file(destination) != digest_file(path):
                            raise TheoryError(
                                f"Managed figure archive collision: {destination}"
                            )
                        path.unlink()
                    else:
                        os.replace(path, destination)
            result["archived_obsolete_files"] = sorted(obsolete)
        for filename in result["owned_files"]:
            os.replace(staging / filename, figures / filename)
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(staging / "manifest.json", figures / "manifest.json")
    return result
