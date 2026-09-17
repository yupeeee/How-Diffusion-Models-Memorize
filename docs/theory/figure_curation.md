# Six-figure publication and migration

The active paper publisher exports exactly six single-page PDFs under project-root `figures/<experiment>/theory/experiment_S0_N<N>/`. There are no appendix, per-timestep or diagnostic figure exports; `--diagnostics` does not expand this selection. `measurement_registry()` retains the complete 20-measurement inventory and optional scalar audits, while `paper_registry()` selects only these six render entries. Scientific formulas, scalar tables, sample populations and cache identities are unchanged by this curation.

Presentation registry `four-stage-paper-curation-20` uses the existing `four-stage-manuscript-notation-17` labels. The mathematical contract remains `projected-gap-error-1`, including the signed projected error and the positive part after the signed variation integral. The cached empirical source-record prior and selected reference mean are unchanged.

| Order | PDF basename in the theory publication directory | Saved measurement stem |
|---|---|---|
| M1 | `initial_loss_recovery` | `initial_loss_recovery` |
| M2 | `unconditional_reference_convergence` | `unconditional_reference_convergence` |
| M3 | `guidance_scale_vs_loss` | `corollary3_guidance_scale_vs_loss` |
| M4 | `posterior_feedback_condition_margin` | `posterior_feedback_condition_margin` |
| M5 | `synchronization_bound` | `synchronization_bound` |
| M6 | `terminal_bound_coverage` | `terminal_bound_coverage` |

Each selected entry is internally category `main`, M1–M6. Its PDF is directly inside `figures/<experiment>/theory/experiment_S0_N<N>/`; no `main/` subdirectory is created. Registry routes remain presentation identities and do not relocate scientific inputs. The short filename `guidance_scale_vs_loss` is only an output alias. Its scientific stem, compact CSV, cohort audit, saved loss ratio and fitted coefficient retain `corollary3_guidance_scale_vs_loss` and recipe `corollary3-guidance-fit-1`.

`posterior_feedback_condition_margin` shows every finite saved margin/gain pair from all saved timesteps as an ordinary dot, with terminal SSCD colors where available. Its selected dense-layer opacity is 0.1; the selected sparse-scatter opacity is 0.8. Numerical signs do not filter points or create Observed/Unresolved markers or legends. Only the pooled PDF is exported; there are no per-timestep figures. Sign, uncertainty, endpoint-transfer and implication receipts remain unchanged in the audit tables.

Appearance version `six-theory-visual-6` uses the same 24-point marker area and 0.8 opacity for the selected initial-loss and fitted-guidance scatters. The SSCD palette, neutral styling, legend arrangement and marker geometry are presentation settings; finite-row selection and numerical values are unchanged. The pooled margin plot uses a uniform 0.1. The new publisher does not create per-step views or any optional diagnostic plots.

## Retained measurements

`measurement_registry()` keeps the complete 20-measurement inventory, optional measurements, manuscript definitions, compact tables and audit receipts. `paper_registry()` returns only the six entries above, even when diagnostics are requested. Former response, trajectory, shape, branch-error, probability, directional-variation and terminal-scatter designs remain scalar measurements. Publication retirement does not delete their scientific data or bypass their existing QA.

Terminal coverage keeps the same eligible sample population and prompt weights for actual error, corrected observable bound and corrected reference bound within each same-seed terminal-SSCD group. Its full range, exact zeros, infinite-bound masses and correction/probability scope remain intact. The convergence offset guide stays visible without a “Reference limit” legend entry. Selected μ remains the recorded reference-based mean, never a newly selectable zero centre.

## Saved-data migration

The canonical bundle stays at:

```text
outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/
```

Publication stages six PDFs in a separate directory under project-root
`figures/<experiment>/theory/experiment_S0_N<N>/`. Its ownership hashes are recorded
in the scalar bundle's `figure_manifest.json`; captions and all scientific tables
stay in that bundle. The bundle and external figure directory are locked and
published together with rollback. Existing unknown or externally modified theory
PDFs are preserved by ownership checks.

This output-layout change leaves historical cache images and archives in place.
There is no PNG export, broad cleanup, scalar rewrite or inference in plot mode.
Protected generation, target latents, SSCD/proximity data, scalar shards, reference
receipts and numerical audits retain their paths and identities. No existing
artifacts were moved, rendered or deleted during this source-only task.

## Author commands — not executed here

For an otherwise compatible bundle with current scientific receipts, this presentation-only update uses:

```bash
bash run_all.sh --model sdv1 --scheduler ddim --plot
```

Omit `--model` and `--scheduler` to request the full configuration matrix; every requested bundle must pass preflight. Copied bundles use `./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot`. A bundle missing required selected-figure scalars or carrying an outdated scientific contract still requires the reported `--recompute-experiments` command. Plotting does not reconstruct measurements, fit coefficients, evaluate posteriors, or run models.

The six-figure curation fixes selection and output paths. The later `six-theory-visual-6` refresh changes only appearance, the shared loss-axis presentation range, and display of the saved synchronization IQR endpoints; figure identities stay fixed; the current PDF routing separates publication from numerical storage. It performs no new inference, posterior calculation, integration or scalar summary computation in plot mode. Existing source-hash and scientific-contract guards remain in force. The prior empirical-law and signed-error revisions are independent scientific migrations and cannot be applied by relabeling historical scalar results.

See [the six-figure appearance contract](figure_appearance.md) for the scoped palette, typography, display-only ordering, export settings and pending author-side visual inspection.
