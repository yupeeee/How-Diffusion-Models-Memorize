"""Shared presentation tokens for explicitly selected publication renderers.

Importing this module sets no rcParams, selects no backend, and performs no data
I/O. Callers retain their own scope, category meanings, typography, layout margins
and export policy. The colormap's historical name and exact 256-color sample are
preserved so extracting these tokens does not alter the approved theory style.
"""
from __future__ import annotations

import matplotlib
from matplotlib.colors import ListedColormap, Normalize
import numpy as np

COLORS = {
    "background": "#FFFFFF", "text": "#24262B", "axis": "#555C66",
    "reference": "#858B93", "control": "#959BA3", "grid": "#E8EBEF",
    "analytical": "#536B8A", "learned": "#168C91", "reference_error": "#555C66",
    "missing": "#808080", "connector": "#C5CBD2",
}

SSCD_CMAP = ListedColormap(
    matplotlib.colormaps["magma"](np.linspace(.20, .70, 256)),
    name="theory_sscd_magma_020_070",
)
SSCD_NORM = Normalize(vmin=0., vmax=1., clip=True)
SSCD_TICKS = tuple(index / 5 for index in range(6))

FIGURE_DPI = 150
PLOT_BOX_INCHES = 3.1
