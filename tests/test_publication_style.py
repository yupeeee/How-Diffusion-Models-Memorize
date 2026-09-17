"""Shared plotting tokens preserve theory appearance; tests are not run here."""
import importlib.util

import matplotlib
import numpy as np
import pytest

from utils.experiments import publication_style as common
from utils.experiments import plotting as shared
from utils.experiments.theory import paper_style as theory
from utils.experiments.theory.paper_registry import GROUPS, measurement_registry, paper_registry


def test_theory_reexports_identical_common_tokens_and_exact_magma_sample():
    assert theory.COLORS is common.COLORS
    assert theory.SSCD_CMAP is common.SSCD_CMAP
    assert theory.SSCD_NORM is common.SSCD_NORM
    assert theory.SSCD_TICKS is common.SSCD_TICKS
    assert theory.PLOT_BOX_INCHES == common.PLOT_BOX_INCHES == 3.1
    expected = matplotlib.colormaps["magma"](np.linspace(.2, .7, 256))
    np.testing.assert_array_equal(common.SSCD_CMAP.colors, expected)
    np.testing.assert_array_equal(theory.GROUP_COLORS[GROUPS[0]], common.SSCD_CMAP(1.0))
    np.testing.assert_array_equal(theory.GROUP_COLORS[GROUPS[1]], common.SSCD_CMAP(0.0))
    assert common.SSCD_CMAP.name == "theory_sscd_magma_020_070"
    assert (common.SSCD_NORM.vmin, common.SSCD_NORM.vmax, common.SSCD_NORM.clip) == (0., 1., True)
    assert common.SSCD_TICKS == (0., .2, .4, .6, .8, 1.)


def test_shared_neutrals_and_theory_local_policies_are_unchanged():
    assert common.COLORS == {
        "background": "#FFFFFF", "text": "#24262B", "axis": "#555C66",
        "reference": "#858B93", "control": "#959BA3", "grid": "#E8EBEF",
        "analytical": "#536B8A", "learned": "#168C91", "reference_error": "#555C66",
        "missing": "#808080", "connector": "#C5CBD2"}
    assert theory.STYLE_VERSION == "six-theory-visual-6"
    assert theory.DENSE_PDF_DPI == shared.FIGURE_DPI == 150
    assert (theory.AXIS_LABEL_SIZE, theory.TICK_LABEL_SIZE, theory.LEGEND_SIZE, theory.ANNOTATION_SIZE) == (18., 15., 12., 10.)
    assert theory.CANVAS_SIZE_INCHES == pytest.approx((5.15, 5.1))
    assert theory.LAYOUT_RECEIPT["export_bounds"] == "per_figure_tight_with_0.05_inch_padding"
    assert all(theory.is_selected(entry) for entry in paper_registry())
    assert not any(theory.is_selected(entry) for entry in measurement_registry(True))
    assert not theory.is_selected({"stem": "proximity_vs_sscd", "outputs": {"pdf": "proximity_vs_sscd.pdf"}})


def test_loading_common_tokens_does_not_mutate_rcparams_or_shared_defaults():
    baseline = dict(matplotlib.rcParams)
    legacy = dict(shared.PLOT_STYLE)
    # Load into an isolated module so the import contract is checked even when
    # normal collection already imported common; live shared object IDs stay put.
    spec = importlib.util.spec_from_file_location("isolated_publication_style", common.__file__)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert dict(matplotlib.rcParams) == baseline
    assert shared.PLOT_STYLE == legacy
    np.testing.assert_array_equal(isolated.SSCD_CMAP.colors, common.SSCD_CMAP.colors)
    with pytest.raises(RuntimeError, match="scoped exit"):
        with matplotlib.rc_context(theory.THEORY_STYLE):
            assert matplotlib.rcParams["axes.labelsize"] == 18.
            raise RuntimeError("scoped exit")
    assert dict(matplotlib.rcParams) == baseline
    assert shared.PLOT_STYLE == legacy
