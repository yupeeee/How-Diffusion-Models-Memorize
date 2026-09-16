"""Saved-only four-stage artist contracts; tests intentionally not executed here."""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_contracts import load_paper_inputs
from utils.experiments.theory.paper_plotting import _draw
from utils.experiments.theory.paper_registry import paper_registry


def inputs(tmp_path):
    root = compact_fixture(tmp_path / "paper", terminal=False)
    config, summary, _, frames = load_paper_inputs(root)
    return config, summary, frames, {entry["stem"]: entry for entry in paper_registry()}


def draw(data, stem, *, frame=None, metadata=None):
    config, summary, frames, entries = data
    return _draw(entries[stem], frames[stem] if frame is None else frame,
                 summary["figures"][stem] if metadata is None else metadata, config)


def test_dose_uses_H_fixed_symlog_signed_range_and_all_endpoints(tmp_path):
    data = inputs(tmp_path)
    fig, audit = draw(data, "branch_gap_posterior_response")
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "symlog" and audit["symlog_linthresh"] == 1e-3
        assert ax.get_xlim() == (0., 1.) and ax.get_ylim()[0] < -2
        assert not audit["log_odds_substitution"]
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("conditional-only" in text for text in texts)
        assert any("Reconstructed matched CFG" in text for text in texts)
        assert not ax.texts
    finally:
        plt.close(fig)
    frame = data[2]["branch_gap_posterior_response"].copy()
    frame.loc[0, "denominator_weight"] = 99
    with pytest.raises(TheoryError, match="changes population"):
        draw(data, "branch_gap_posterior_response", frame=frame)


def test_chronological_indices_include_initialization_and_no_terminal_prediction(tmp_path):
    fig, audit = draw(inputs(tmp_path), "branch_gap_synchronization")
    try:
        assert audit["chronological_prediction_range"] == [0, 2]
        assert not audit["terminal_output_prediction"]
        assert any("Initialization" in label.get_text() for label in fig.axes[0].get_xticklabels())
        assert fig.axes[0].get_xlim() == (0., 2.)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_reference_preserves_both_native_network_curves_and_reference_only_extension(tmp_path):
    fig, audit = draw(inputs(tmp_path), "unconditional_reference_convergence")
    try:
        assert audit["native_sweep_preserved"] and not audit["network_extrapolation"]
        for color in ("#e67d24", "#3b9955"):
            lines = [line for line in fig.axes[0].lines if line.get_color() == color]
            assert any(np.array_equal(line.get_xdata(), [.01, .1, 10.]) for line in lines)
        assert not fig.axes[0].texts
    finally:
        plt.close(fig)


def test_terminal_log_zeros_use_edge_markers_without_pseudodata(tmp_path):
    data = inputs(tmp_path)
    stem = "final_reproduction_bound"
    frame = data[2][stem].copy()
    frame.loc[1, ["x", "y"]] = [1e-5, 1e-6]
    frame.loc[2, ["x", "y"]] = [np.inf, .8]
    before = frame.copy(deep=True)
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["axis_scale"] = "log"
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "log"
        assert audit["zero_counts"] == {"x": 1, "y": 1}
        assert audit["infinite_bound_count"] == 1
        assert audit["zero_position_policy"].startswith("axes-edge")
        assert fig.axes[0].get_xlim() == fig.axes[0].get_ylim()
        assert fig.axes[1].get_ylabel() == "SSCD"
        assert frame.equals(before)
    finally:
        plt.close(fig)


def test_default_appendix_has_no_small_free_annotations_and_readable_legends(tmp_path):
    from matplotlib.legend import Legend
    data = inputs(tmp_path)
    for entry in data[3].values():
        if entry["category"] != "appendix":
            continue
        fig, _ = draw(data, entry["stem"])
        try:
            ax = fig.axes[0]
            assert not ax.texts and not ax.get_title()
            for legend in (artist for artist in ax.get_children() if isinstance(artist, Legend)):
                assert all(label.get_fontsize() == 10 for label in legend.get_texts())
                assert legend.get_title().get_fontsize() == 10
        finally:
            plt.close(fig)


def test_active_terminal_coverage_keeps_zero_infinity_and_scope_in_legend(tmp_path):
    import pandas as pd
    data = inputs(tmp_path)
    stem = "terminal_bound_coverage"
    frame = pd.DataFrame([
        dict(distribution=name, value=value, cdf=cdf)
        for name in ("actual", "observable", "reference")
        for value, cdf in [(0., .25), (.2, .5), (1., .75), (np.inf, 1.)]])
    metadata = deepcopy(data[1]["figures"][stem])
    metadata["zero_mass"] = {name: .25 for name in ("actual", "observable", "reference")}
    metadata["infinite_mass"] = metadata["zero_mass"].copy()
    fig, audit = draw(data, stem, frame=frame, metadata=metadata)
    try:
        ax = fig.axes[0]
        assert not ax.texts and not ax.get_title()
        labels = [label.get_text() for label in ax.get_legend().get_texts()]
        assert len(labels) == 3 and all("Zero: 0.25; +∞: 0.25" in label for label in labels)
        assert ax.get_legend().get_title().get_text() == "Finite-step extension"
        assert audit["zero_mass_legend_visible"] and audit["infinite_mass_legend_visible"]
        assert all(line.get_ydata()[-1] == .75 for line in ax.lines)
    finally:
        plt.close(fig)
