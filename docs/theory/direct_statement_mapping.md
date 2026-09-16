> The active figure organization is now [four_stage_experiments.md](four_stage_experiments.md). Seven-primary filenames and roles below describe the preceding design; compatible mathematical measurements and numerical contracts remain reusable. Matching LaTeX is unavailable; historical label strings are author-supplied, not verified source labels.

# Direct statement implementation map

The default `paper` suite now has seven independently exported main figures. These are code contracts, not reports of completed measurements. The full machine-readable mapping is in `statement_registry.json`; `legacy_statement_registry.json` preserves the earlier cache-reader contract.

| Statement | Scalar source and plotted columns | Main figure |
|---|---|---|
| Theorem 1 | `forward_loss_summary.loss_mean_squared_l2`, `snr`, `latent_dimension`; `gaussian_conditional.conditional_error_rmse` | `theorem1_loss_recovery` |
| Lemma 2 | `gaussian_reference`: `reference_mean_error_rmse`, `learned_mean_error_rmse`, `unconditional_reference_error_rmse` versus native `snr` | `lemma2_unconditional_baseline` |
| Corollary 3 | `initial.direct_cor3_injection_bound_rmse` versus `direct_cor3_injection_discrepancy_rmse` | `corollary3_initial_cfg_amplification` |
| Lemma 4 | `matched_updates.direct_lemma4_predicted_rmse` versus `direct_lemma4_displacement_rmse`, with `direct_lemma4_independent` markers | `lemma4_matched_displacement` |
| Proposition 5 | `matched_updates.direct_prop5_margin_rmse` versus `direct_prop5_log_probability_gain`; original variation integration and uncertainty statuses | `proposition5_posterior_feedback` |
| Lemma 6 | `trajectory.direct_lemma6_rhs_rmse` versus `direct_lemma6_gap_rmse`, including the last current prediction | `lemma6_target_specific_synchronization` |
| Theorem 7 | `terminal.direct_theorem7_bound_rmse` versus `direct_theorem7_endpoint_error_rmse`, with `direct_theorem7_applicable` | `theorem7_final_reproduction` |

Each figure has its own PNG and single-page PDF under `outputs/<base_run>/theory/experiment_S0_N<N>/main/`. The nine fixed supporting entries include squared Gaussian recovery, the independent forward identity, plug-in Gaussian transfer at every fixed tolerance, reference-to-learned concentration, optional marginal excess loss, the second CFG vector limit, the update vector residual, posterior concentration, and optional predeclared terminal certification. Optional entries record why they are unavailable or inapplicable. Former behavioral figures retain separate diagnostic names.

`direct_figures.build_direct_plot_inputs` receives measured scalar tables and prepares `plot_data` plus analysis-owned Gaussian and statement summaries. No bound is formed by summing separately aggregated medians. Gaussian forward-loss sample sizes remain pair-level; unique unconditional Gaussian seeds are never duplicated across prompts. Plotting reads those compact CSVs and stored statuses through `paper_contracts`, then uses the existing shared figure publisher. It imports no direct numerical builder.

Theorem 1 has no Gaussian equality line. Its forward-input identity is separate; its probability comparison is a plug-in empirical check, not a rigorous certificate from finite unbounded-loss estimates. Proposition 5 uses the original variation integral, signed gain, zero guides and separate unresolved markers. Theorem 7 keeps all well-defined endpoint/bound pairs, including failures of the clean-update prerequisite; only applicable rows can support its upper-bound interpretation or sufficient event.

All posterior/reference quantities use one declared finite law. Its exact weighted mean is independent of legacy model-output centers. Equation 6 remains an assumed idealization. Scientific schemas and table identities must be refreshed by analysis; old behavioral tables cannot stand in for the new loss, reference or statement-side fields. Runtime retirement operates only on previously owned artifacts after replacement publication; this code-edit task does not move existing outputs.

Tests were added for the author to run after review:

```bash
$PYTHON -m pytest tests/test_theory_direct_figures.py tests/test_theory_paper_plotting.py tests/test_theory_paper_contracts.py tests/test_theory_paper_pipeline.py tests/test_theory_paper_fresh.py
```

No experiments, inference, tests, figures, benchmarks, or data-dependent validations were executed while implementing this change.
