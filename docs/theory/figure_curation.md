# Six-figure publication and migration

The active paper publisher exports exactly six PNG/single-page-PDF pairs directly in the bundle’s `figures/` directory (12 images). There are no appendix, per-timestep or diagnostic figure exports; `--diagnostics` does not expand this selection. `measurement_registry()` retains the complete 20-measurement inventory and optional scalar audits, while `paper_registry()` selects only these six render entries. Scientific formulas, scalar tables, sample populations and cache identities are unchanged by this curation.

Presentation registry `four-stage-paper-curation-20` uses the existing `four-stage-manuscript-notation-16` labels. The mathematical contract remains `projected-gap-error-1`, including the signed projected error and the positive part after the signed variation integral. The cached empirical source-record prior and selected reference mean are unchanged.

| Order | File basename in `figures/` | Saved measurement stem |
|---|---|---|
| M1 | `initial_loss_recovery` | `initial_loss_recovery` |
| M2 | `unconditional_reference_convergence` | `unconditional_reference_convergence` |
| M3 | `guidance_scale_vs_loss` | `corollary3_guidance_scale_vs_loss` |
| M4 | `posterior_feedback_condition_margin` | `posterior_feedback_condition_margin` |
| M5 | `synchronization_bound` | `synchronization_bound` |
| M6 | `terminal_bound_coverage` | `terminal_bound_coverage` |

Each selected entry is internally category `main`, M1–M6, but its PNG/PDF paths are directly under `figures/`; no `main/` subdirectory is created. The short filename `guidance_scale_vs_loss` is only an output alias. Its scientific stem, compact CSV, cohort audit, saved loss ratio and fitted coefficient retain `corollary3_guidance_scale_vs_loss` and recipe `corollary3-guidance-fit-1`.

`posterior_feedback_condition_margin` shows every finite saved margin/gain pair from all saved timesteps as an ordinary dot, with terminal SSCD colors where available. Its opacity is exactly 0.01; the shared scatter opacity is 0.8. Numerical signs do not filter points or create Observed/Unresolved markers or legends. Only the pooled PNG/PDF are exported; there are no per-timestep figures. Sign, uncertainty, endpoint-transfer and implication receipts remain unchanged in the audit tables.

The shared scatter opacity is 0.8, including initial-loss and fitted-guidance points. SSCD color limits, point sizes, reference lines, legends, finite-row selection and numerical values are unchanged. Only the pooled margin plot uses 0.01. The new publisher does not create per-step views or any optional diagnostic plots.

## Retained measurements

`measurement_registry()` keeps the complete 20-measurement inventory, optional measurements, manuscript definitions, compact tables and audit receipts. `paper_registry()` returns only the six entries above, even when diagnostics are requested. Former response, trajectory, shape, branch-error, probability, directional-variation and terminal-scatter designs remain scalar measurements. Publication retirement does not delete their scientific data or bypass their existing QA.

Terminal coverage keeps the same eligible sample population and prompt weights for actual error, corrected observable bound and corrected reference bound within each same-seed terminal-SSCD group. Its full range, exact zeros, infinite-bound masses and correction/probability scope remain intact. The convergence offset guide stays visible without a “Reference limit” legend entry. Selected μ remains the recorded reference-based mean, never a newly selectable zero centre.

## Saved-data migration

The canonical bundle stays at:

```text
outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/
```

Publication stages the selected image pairs and their metadata before retiring known old renderer-owned files. Old `main/` and `appendix/` placements, previously exported diagnostics and per-timestep margin views are retired only when their exact prior manifest paths and hashes prove ownership. Selected entries move to `figures/`; unselected entries have no replacement image. No directory-wide purge or unrestricted filename glob is authorized.

Every retirement candidate must be a regular file inside the active bundle at an approved path with a matching prior manifest hash. Unowned, modified, symbolic-link, traversal or unrelated files are preserved and reported as conflicts. Pre-existing historical archives are outside migration scope. Role locking, staged publication and rollback protect PNG/PDF pairs and metadata together. A failed export cannot leave a successful partial publication; the retirement ledger records decisions and conflicts. Repeated plotting does not accumulate duplicate pairs or resurrect unselected designs.

Protected generation, target latents, SSCD/proximity data, all scientific/plot-data CSVs, scalar shards, reference-law receipts and numerical audits are never retirement targets. This source-only change does not move, render or delete existing output files; migration happens when the author runs publication.

## Author commands — not executed here

For an otherwise compatible bundle with current scientific receipts, this presentation-only update uses:

```bash
bash run_all.sh --model sdv1 --scheduler ddim --plot
```

Omit `--model` and `--scheduler` to request the full configuration matrix; every requested bundle must pass preflight. Copied bundles use `./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot`. A bundle missing required selected-figure scalars or carrying an outdated scientific contract still requires the reported `--recompute-experiments` command. Plotting does not reconstruct measurements, fit coefficients, evaluate posteriors, or run models.

This update changes selection, output paths and opacity only. It performs no new inference, posterior calculation, integration or scalar summary computation in plot mode. Existing source-hash and scientific-contract guards remain in force. The prior empirical-law and signed-error revisions are independent scientific migrations and cannot be applied by relabeling historical scalar results.
