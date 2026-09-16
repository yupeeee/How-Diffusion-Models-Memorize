> The active figure organization is now [four_stage_experiments.md](four_stage_experiments.md). Seven-primary filenames and roles below describe the preceding design; compatible mathematical measurements and numerical contracts remain reusable. Matching LaTeX is unavailable; historical label strings are author-supplied, not verified source labels.

# Fixed seven-statement evidence plan

This plan replaces the primary designs from the first direct-quantity suite. It
was fixed after reviewing SDv1/DDIM results; that configuration is not held-out
confirmation of the design. Apply the same plan to the other supported
configurations and later seeds without selecting favorable plots or snapshots.
No experiments, tests, integration, plotting, downloads, or GPU jobs were run
for this code update. Earlier test receipts do not validate this revised suite.

The mounted `revised.pdf` matches SHA-256
`fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`.
Its text and notation table were inspected. No matching LaTeX source was found;
the source-label identifiers below are the verified identifiers supplied by the
author. The manuscript itself is unchanged.

| Result | Fixed primary design | Mandatory supporting output |
|---|---|---|
| Theorem 1 | One conditional Gaussian RMS and matched unconditional control per pair versus independent forward-loss scale; mean terminal SSCD color | Initial branch comparison; Gaussian transfer including total variation |
| Lemma 2 | Same-law posterior convergence on a fixed lower-SNR grid, with learned and reference-error curves restricted to native labels | Initial unconditional/atom concentration about the same `mu_K` |
| Corollary 3 | Full injection geometry, ideal `(1,0)`, and fixed relative-error contours | Full guided-estimate triangle bound |
| Lemma 4 | Independently constructed matched displacement against predicted displacement | Full vector residual and source precision in audit tables |
| Proposition 5 | Strict feedback prevalence and original strict-condition coverage on identical weighted populations | Original condition/gain scatter; first-update endpoint log odds |
| Lemma 6 | Paired joint target error, actual branch gap, and complete bound on one SNR axis | Branch target errors; bound/gap; posterior concentration |
| Theorem 7 / declared extension | Exact common-population tolerance ECDFs of endpoint error and two independently corrected bounds | Terminal components when supported; original uncorrected audit retained |

## Quantities and uncertainty

All reference quantities use the same declared `D_K`, unchanged atoms/masses,
exact atom mean, and raw squared-L2 Gaussian exponents. Conditional reference
error uses the explicitly assumed single-target idealization. Initial RMS
summaries are descriptive finite-sample moments, not moment convergence implied
by convergence in probability. Pair bootstrap intervals use 1,000 resamples,
seed 0, and central 95% percentiles; these are Monte Carlo bootstrap intervals,
not theorem confidence bounds. Gaussian seed resampling uses a common seed-ID
order across prompts. No pooled bootstrap or significance claim that ignores
shared seeds or shared targets is produced.

Analytical reference evaluation uses 97 logarithmically spaced SNRs from
`SNR_T*10**(-reference_snr_decades)` through `SNR_T`, default six decades. The
same Gaussian bank and fixed saved precision are used at every point. It never
calls a network outside its native labels. The reference-law RMS scale and all
available native measurements remain saved.

Outcome grouping uses the unchanged SSCD threshold 0.75. Prompt-balanced group
weights are common to both sides of each comparison; unresolved sign mass stays
in the strict-coverage denominator. Q is the paired maximum before aggregation;
S is the full per-sample bound before aggregation. Descriptive IQRs, Monte Carlo
bootstrap intervals, numerical ambiguity bands, and Gaussian probability bounds
have separate metadata definitions.

The terminal correction is computed from pre-update quantities. Deterministic
updates use the actual affine defect; supported stochastic updates use an
independently supplied innovation or the declared Gaussian norm bound. The
run-level failure budget defaults to 0.05, divided by the fixed retained final
update count before outcomes are inspected. All three terminal ECDFs share one
population and weighting. Exactly zero mass and all positive change points are
retained; no epsilon replaces zero and no SSCD-fitted tolerance is introduced.
See [submission_alignment.md](submission_alignment.md) for the derivation and
remaining manuscript decisions.

## Files and cache contracts

- `evidence_measurements.py` adds geometry, paired joint errors, pre-update
  terminal corrections, and independent output QA. It leaves original direct
  measurements and the clean terminal gate unchanged.
- `evidence_reference.py` computes analytical-only posterior distances with
  bounded existing support/query evaluation.
- `evidence_reduce.py` first resumes the unchanged direct analysis, then uses
  spawned workers and one parent progress bar for missing supplements/grid
  points. Learned-model replicas have already exited.
- `evidence_figures.py` makes scalar summaries, exact weighted ECDFs, appendix
  inputs, numerical exclusion/consistency audits, and pair bootstrap intervals.
- `paper_registry.py` defines the fixed plan. `evidence_plotting.py` and the
  shared publisher render compact saved inputs only.

Original learned tasks, vector cores, current references, endpoint measurements,
and original variation integrals retain their existing identities. Additional
backing tables live under `theory_measurements/evidence_supplements/<hash>/`
and `theory_measurements/reference_analytical/<hash>/`. A missing geometry or
terminal supplement reads preserved initial/final vectors; it does not repeat
posterior histories or integration. The logical-table manifest keeps original
history paths and records exact joins to additive columns; it does not persist
a second copy of the complete histories. Existing model calls are needed only for
missing incompatible learned tasks in the original direct stage.

The paper bundle schema is 3 and the compact metric schema is `fixed-evidence-1`.
Old primary scalar tables cannot be relabeled into the new designs. New science
identities record the law, formula, fixed analytical grid, aggregation/bootstrap
recipe, terminal correction source, quantile implementation and probability
scope. Renderer source changes remain separate from measurement recipes.

Publication stays at
`outputs/<canonical_base_run>/theory/experiment_S0_N<N>/`. Each registry entry
gets a separate PNG and single-page PDF using the common STIX, 4x4-inch,
150-DPI style. The main stems remain unchanged. The appendix stems are:

1. `theorem1_initial_branch_comparison`
2. `theorem1_gaussian_transfer`
3. `lemma2_initial_concentration`
4. `corollary3_full_guided_error`
5. `proposition5_condition_vs_gain`
6. `proposition5_first_update_comparison`
7. `lemma6_branch_target_errors`
8. `lemma6_bound_vs_gap`
9. `lemma6_posterior_concentration`
10. `theorem7_terminal_components` when supported

Other exact-identity comparisons are optional diagnostics or audit data. No
candidate gallery is regenerated. Original clean applicability, excluded rows,
negative findings, raw Gaussian tails, within-prompt associations, quadrature
components, and original theorem comparisons remain traceable through compact
audit tables and `logical_tables.json`.

On the author's execution, only manifest-owned derived figures are archived
through the existing transaction after a replacement is ready. Old caches,
unrecognized files, and historical validation reports are preserved. No output
migration has been performed during this task.

## Author commands

```bash
# Execute the authored regressions before the new analysis.
python -m pytest -q tests/test_theory_*.py tests/test_generation.py

# Analysis uses all visible devices and reuses compatible prior measurements.
./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto

# Re-render saved compact data without any measurements or integration.
./run_all.sh --model sdv1 --scheduler ddim --plot
./run_all.sh --plot

# Apply the same frozen plan to the whole supported matrix.
./run_all.sh --recompute-experiments --device auto

# Portable compact bundle.
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

`PYTHON` forwarding is retained. Optional predeclared science overrides are
`--reference-snr-decades` and `--terminal-noise-run-alpha`; omission in plot mode
inherits the saved values. These do not optimize a figure or change a sampler.

Before submission the author must align the finite-terminal-update extension
and sampler premise in the manuscript, approve the declared reference/finite-SNR
wording, and preserve the distinction between feedback prevalence and strict
sufficient-condition coverage. An extension figure carries
`manuscript_extension_required=true`; this is not a claim of submission readiness.
