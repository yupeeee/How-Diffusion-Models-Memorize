# Proximity–SSCD appearance contract

Presentation version **`proximity-publication-2`** applies only to the existing `proximity_vs_sscd_all_prompts` and `proximity_vs_sscd` scatter families. Both evaluation and reference seed roles use the same renderer for SDv1, SDv2 and RealVis, including the supported SDv1/DDPM configuration. The reference `proximity_vs_sscd_gmm_fit` diagnostic now shares this theme through `proximity-gmm-publication-1`. The six theory figures keep `six-theory-visual-6`; other diagnostic families retain their appearance. The existing cached-image examples now share the external PDF publication tree; their prompt selection, montage content and paired targets are unchanged.

## Quantities, populations and categories

The x-axis remains the raw latent Euclidean distance $\|\mathbf{x}_0-\mathbf{x}^{\star}\|$, without division by $\sqrt{d}$, squaring, standardization or an RMSE label. The y-axis remains the saved SSCD score on its existing linear scale, including valid negative scores and finite outliers. All-prompt validity/completeness rules, retained-prompt membership and every eligible seed observation of each retained prompt stay unchanged. Retention does not select only high-SSCD seeds.

The legend preserves the project's monospace mathematics: `$\mathtt{MV}$`, `$\mathtt{RV}$`, `$\mathtt{TV}$` and `$\mathtt{N}$`. These labels identify original categories; they do not encode measured SSCD, GMM components or selection outcomes. In particular, `N` is not reinterpreted as an experimentally established nonmemorizing group.

| Original category | Fixed nominal color |
| --- | --- |
| $\mathtt{MV}$ | Coral: shared $C(1)$ endpoint |
| $\mathtt{RV}$ | Indigo: shared $C(0)$ endpoint |
| $\mathtt{TV}$ | Teal, `#168C91` |
| $\mathtt{N}$ | Slate blue, `#536B8A` |
| Existing unknown/unlabeled category, when present | Neutral gray, `#858B93` |

Here $C(s)=\operatorname{magma}(0.20+0.50s)$. The implementation samples the shared endpoints rather than duplicating approximate hexadecimal colors. Category names determine colors even when a view contains only TV and N. The fixed legend order is MV, RV, TV, N, followed by the existing fallback when represented. Only represented categories receive handles; existing normalization and missing-label behavior remain intact. No SSCD colorbar is added.

The common `publication_style` tokens (`COLORS`, `SSCD_CMAP`, `SSCD_NORM`, `SSCD_TICKS` and `PLOT_BOX_INCHES`) provide palette endpoints, neutrals and geometry. Proximity-specific settings remain scoped to its scatter boundary. Helpers do not alter global plotting defaults or introduce runtime dependencies, scientific imports or I/O at import time.

## Points and paired layout

Each view draws one point collection with per-row category colors, circular markers of area 10 points squared, no outlines and uniform opacity 0.35. A stable observation-identity digest sets the display order. Its inputs exclude category, proximity, SSCD, selection outcome and view name, so shared observations preserve their relative order between the all-prompt and retained-prompt views. Distinct identities and duplicate coordinates retain their multiplicity. Saved frames, CSV order and scientific random streams are unchanged.

For each model and seed role, paired views use identical x/y limits and major ticks derived from the all-prompt view. The retained plot is not rescaled independently. Existing margins and linear scales remain, including the negative-SSCD range; cross-model raw-distance ranges may differ. The physical data rectangle is square, 3.1 inches per side, without equalizing the different x/y data units. Both views use the same compact outer margins; no external header/footer bands are reserved. All four spines remain, with a white background, charcoal text, subtle major grids and no minor grids.

An untitled single-column legend sits inside the upper-right corner, with one row per represented category (MV, RV, TV and N when all four are present). Its equal-sized proxies are opaque and its labels preserve `\mathtt`; an existing unknown/unlabeled category remains visible when present. The original rounded white statistics box is restored inside the lower-left corner at axes coordinates (0.02, 0.02). Its three lines show `#Prompts`, median $\rho$, then $\rho < 0$; the original numerical meaning and evaluable denominator remain unchanged.

The approved current export hierarchy is labels/ticks/legend-and-summary at **18/15/12 points**, using STIX/serif typography. The attachment's 15/12/10 hierarchy describes an older fallback; it does not reduce the current fonts. The manuscript inclusion width is unknown, so these export sizes are not a claim of verified final manuscript readability.

## Reference GMM-fit figure

`proximity_vs_sscd_gmm_fit.pdf` uses the same scoped STIX/serif typography
(18/15/12 points), white background, subtle major grid, light spines, compact
margins and 3.1-inch square plotting rectangle as the proximity scatters.
The low-SSCD fitted component uses the shared indigo endpoint `C(0)`; the
high-SSCD component uses coral `C(1)`. These colors encode the saved GMM
component assignments. All fitted complete observations are shown, including
complete observations from otherwise incomplete prompt groups, as before.

One identity-ordered point collection uses marker area 10 and uniform opacity
0.35. Component labels are exactly **Low SSCD** and **High SSCD**, in regular
font, in an untitled single-column legend inside the upper right. Mean crosses and the original
one-/two-standard-deviation covariance ellipses use their component colors.
Their positions, orientations, widths and heights still come from the saved
full-covariance fit; no refitting, filtering or covariance change is introduced.
Raw latent distance and SSCD remain on linear axes, with all finite points and
ellipse extents included. Existing negative scores remain visible.

The reference PDF is saved at
`figures/<canonical_run>/proximity/reference_S<N>_N<N>/proximity_vs_sscd_gmm_fit.pdf`.
Only its points are rasterized, at the shared 150-DPI setting used by the other
figures. Ellipses, means, legend and axes remain vector artwork. Tight
bounds and 0.05-inch padding remain in effect. The current source-only styling
change has not rendered figures or executed tests.

## Statistical summary

The statistics box reports the existing total prompt count, median $\rho$, and negative-correlation count divided by the **evaluable prompt count**. Here $\rho$ is the existing per-prompt Spearman correlation over the relevant seed observations, not a pooled correlation across all plotted points. Existing validity rules, aggregation and precision remain: median to three decimals and percentage to one decimal. Undefined correlations remain excluded from the evaluable denominator. If none are evaluable, the existing `N/A`/undefined convention is retained rather than reporting zero. The denominator remains explicit when total and evaluable counts differ.

Each role uses its own saved observations and statistics; reference and evaluation correlations can legitimately differ. No displayed values from example PDFs are hardcoded. The negative fraction is not relabeled as significance, validation or memorization rate. Styling adds no new estimand, fitting, smoothing, jitter, thinning, clipping, outcome-based ordering or new point selection.

## Exports and author commands

Scientific tables retain their existing `outputs/` paths. Publication PDFs are saved separately at project-root:

```text
figures/<canonical_run>/proximity/<role>_S<seed_start>_N<N>/
```

`experiment` is the evaluation role and `reference` is the frozen selection role. Each exports only `proximity_vs_sscd_all_prompts.pdf` and `proximity_vs_sscd.pdf`. Reference-role GMM diagnostics are also PDF-only in this publication tree; any active strategy namespace is preserved. Manuscript attachment prefixes do not create renamed or duplicate outputs. Each PDF remains one page; no combined six-panel image is generated.

No PNG is exported. The proximity scatter point collection alone is rasterized in PDF at the shared **150 DPI**; labels, axes, ticks, legend and statistics stay vector-based. The common `publication_style.FIGURE_DPI` setting is 150 for all figures, including proximity, GMM, examples and the six theory figures. Exports use `bbox_inches="tight"` and `pad_inches=0.05`, with no invisible canvas-expanding artist. Explicit PDF format, staging of both views, atomic installation, rollback and figure closing remain in use. Existing cache PNGs/PDFs are left in place; this source change does not move or delete prior artifacts.

The retained/discarded examples are also exported in both normal and `--plot`
execution:

```text
figures/<canonical_run>/proximity/<seed-role>/examples/
  retained/{highest,median,lowest}_l2_{generated,training}.pdf
  discarded/{highest,median,lowest}_l2_{generated,training}.pdf
```

These are the existing examples ranked by prompt-mean terminal L2, with the
existing lower-middle median and reduced number of unique ranks when a group is
small. Generated montages retain every seed in its original order; paired training
images retain their aspect ratio. The existing generated scale 0.75 and training
maximum edge 256 are display-only. Example PDFs use 150 DPI, tight bounds and
0.05-inch padding. Source PNGs, cached gallery PNGs, scientific CSVs, selection and
statistics remain unchanged. No new image generation, VAE decoding or latent
loading is involved: the renderer reads cached images and completion metadata.

The example manifest remains under
`outputs/<canonical_run>/proximity/<seed-role>/examples/manifest.json`; it records
PDF paths relative to its explicit external figure directory, source hashes,
image sizes and chosen prompt identities. All image pairs are validated before
export. PDF bytes and this metadata publish together with rollback; unrelated
files and historical PNGs are preserved. Saved-plot preflight checks example image
availability and decodability without writing files; missing/corrupt caches cause
an error instead of silently regenerating or omitting examples.

For the evaluation seed block:

```bash
bash compute_proximity.sh --model sdv1 --scheduler ddim \
  --g 7.5 --T 50 --N 20 --seed-start 0 --plot
```

For the matching frozen reference block:

```bash
bash compute_proximity.sh --model sdv1 --scheduler ddim \
  --g 7.5 --T 50 --N 20 --seed-start 20 --plot
```

Replace `sdv1` with `sdv2` or `realvis` for their DDIM runs. Use `--model sdv1 --scheduler ddpm` for SDv1/DDPM. In general, evaluation seeds are `0..N-1` and reference seeds are `N..2N-1`, so the reference `--seed-start` equals the saved `N`. `--selection-strategy gmm` remains the default, and the wrapper honors `PYTHON`.

The focused command validates frozen selection, saved proximity CSVs and the cached images needed for example PDFs. It does not require theory bundles or perform inference, raw-trajectory processing, latent-distance/SSCD computation, GMM fitting or selection rebuilding. Missing inputs retain the existing actionable errors. Reference-role plotting also redraws the GMM diagnostic from saved parameters using the shared publication theme. The broader workflow remains:

```bash
bash run_all.sh --plot
```

That command validates and renders its requested saved proximity and theory configurations; its existing GMM diagnostic behavior remains. Palette/layout changes do not enter scientific or selection fingerprints, and plotting leaves scalar tables, masks, statistical results and GMM parameters unchanged.

## Manuscript note and verification status

For color-based prose, original-category MV red becomes coral, RV orange becomes indigo, TV blue becomes teal and N green becomes slate blue. Prefer category names to color references; category meanings are unchanged. The intended manuscript arrangement remains SDv1/SDv2/RealVis by column and all-prompt/retained-prompt by row.

Tests and rendering were **not executed during this proximity coding task**. Plotting-specific regression sources are authored for the author to run. Final-size inspection, grayscale review and a color-vision-deficiency preview remain pending, using existing tooling where available. Check the monospace category labels, omitted-category handling, negative scores/outliers, shared axes and box sizes, statistics-box denominators and export bounds. No readability or accessibility verification is claimed, and no new dependency is added for those checks.
