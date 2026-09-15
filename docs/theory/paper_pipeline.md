# Fixed paper pipeline and migration audit

The default theory stage now publishes the reviewed paper selection, using the working candidate measurements. Historical saved suite choices cannot send `run_all.sh` back to the exploratory gallery or old four-figure renderer. No active stage creates `theory_v2`; fresh shared scalar reductions use `theory_measurements` and candidate sufficient statistics remain under `theory_candidates`.

## Commands and files

```bash
./run_all.sh --model sdv1 --scheduler ddim
./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments
./run_all.sh --plot
./run_all.sh --plot --diagnostics
```

The normal matrix is SDv1/DDIM, SDv1/DDPM, SDv2/DDIM, and RealVis/DDIM. The canonical paper directory is `outputs/<model>_<scheduler>_g7.5_T50_N20/theory/experiment_S0_N20/`. `--figure-suite paper` is optional. Historical `main` and `candidates` values now fail before upstream stages.

Main PNG/PDF pairs:

| Legacy design | Stem |
|---|---|
| IR01 | `main/initial_recovery` |
| UB02 | `main/initial_unconditional_concentration` |
| PF03 | `main/posterior_feedback_positive_fraction` |
| TS01 | `main/target_synchronization` |

Core appendix PNG/PDF pairs:

| Legacy design | Stem |
|---|---|
| IR03 | `appendix/initial_recovery_within_prompt` |
| IR04 | `appendix/initial_target_retrieval_rank` |
| PF04, k=0 | `appendix/posterior_feedback_initial_comparison` |
| PF02 | `appendix/posterior_feedback_normalized_gain` |
| PF08 | `appendix/posterior_feedback_increasing_profile_fraction` |
| TS02 | `appendix/branch_gap` |
| TS03, k=10 | `appendix/joint_target_recovery_early` |
| TS03, k=48 | `appendix/joint_target_recovery_late` |

The conditional entries are `appendix/terminal_error_terms` and `appendix/terminal_bound_tightness`. Each manifest distinguishes structural inapplicability from missing metadata or failed reconstruction. T=50 snapshots are fixed at k=10 and k=48; the versioned fallback for other T uses clipped floor(0.2T) and max(0,T-2). Coincident snapshots are aliases.

The data-only registry is `utils/experiments/theory/paper_registry.py`. Each published `figure_manifest.json` records actual output hashes, formula version, applicability, counts, saved scales, source identity, and rendering provenance. Complete definitions and per-run counts are in `figure_captions.md`. PNG/PDF pairs use the same staged Matplotlib figure and the public wrapper around proximity's tested publisher. A failed second export cannot install half a pair; the complete paper role is installed only after validation and successful rendering.

`initial.csv` and `terminal.csv` are consolidated scalar copies. `logical_tables.json` pins the authoritative existing trajectory, feedback, dose, and control shards by path and hash. Large trajectories are not recopied to resemble a suggested layout. Plotting reads only the compact `plot_data/*.csv` plus small manifests; it never opens those backing shards or hashes the large consolidated/audit CSVs. `audit_paper_integrity` provides a separate full scalar-integrity audit. During a plot-only directory transaction, immutable CSV files are linked rather than copied; atomic analysis writers replace links without modifying the old publication. CSV identity strings such as `NA` and leading-zero record IDs survive round trips, and float64 values use round-trip formatting and parsing.

## Manuscript axis notation

The fixed paper renderer follows the printed notation in `revised.pdf`
(Eqs. 2, 8–14, Appendix A, and the matched segment in Eq. 66). Vector symbols
use $\mathbf{x}$, $\mathbf{z}$, $\mathbf{\Delta}$ and $\boldsymbol{\mu}$;
clean estimates and references use $\hat{\mathbf{x}}_t$ and
$\bar{\mathbf{x}}_t$. The Appendix A PDF fonts were checked directly:
Latin vectors and Delta are bold upright, mu is bold italic, and hats/bars
are ordinary accents. Scalar $p_t$, $e_t$, $d$, $g$ and time indices stay
ordinary weight.

The initial prediction uses $T$; a verified clean terminal update uses $1$.
Both joint-recovery CDFs display
$\mathrm{max}_b\|\hat{\mathbf{x}}_t(b)-\mathbf{x}^\star\|/\sqrt{d}$ at their
saved snapshots. Matched posterior axes use $\mathbf{x}_{t-1}^{\mathrm{cf}}$
and the dose coordinate $s$ from
$\mathbf{z}_t(s)=\mathbf{x}_{t-1}^{\mathrm{cf}}+s g\kappa_t\mathbf{\Delta}_t$.
`logit` means $\log[p/(1-p)]$.

Figure axes use $p_t$, $\bar{\mathbf{x}}_t$, $e_t$ and $\boldsymbol{\mu}$
without a support marker. All posterior/reference calculations use their
saved declared law, whose support and masses remain in the measurement
provenance. The caption identifies which center each figure measures:
the fixed independent center for initial concentration and injection, or
the declared law's own mean for the reference diagnostics.

CDF axes say `CDF` or `Weighted CDF`. SSCD outcome legends use
$\leq$ rather than the stored ASCII comparison; the CSV group keys and
membership are unchanged. The main target-synchronization plot places
both legend columns inside its axes: the two SSCD outcomes on the left,
and conditional, unconditional and $\mathrm{SNR}_T$ on the right.

The registry records the mappings in `axis_notation`, exported in
`figure_manifest.json` and `figure_captions.md`. Labels use single-line
slash division and identify reference lines. Descriptive labels remain
for quantities without a manuscript symbol, including retrieval rank,
normalized log-odds gain, coverage, and independent replay diagnostics.
Existing RMSE values, scalar tables, weights, bands, snapshots, and figure
selection are unchanged.

Typography validation: 102 plotting/CLI tests passed. All 25 figure designs
passed PNG/PDF inspection, including inside, disjoint synchronization legend
columns. The current SDv1/DDIM bundle was refreshed (12 PNG/PDF pairs), with
all 37 checked numerical files retaining hashes and modification times.

## Scientific corrections

- **Matched feedback:** both endpoints use the fixed candidate law and the same destination coefficients. Trajectory axes use current SNR. Stable log-odds signs preserve tiny log-probability gains that underflow. Reliable H/G disagreements or beyond-budget endpoint bracket violations block publication and retain offending identities.
- **Fractions:** PF03 and PF08 use prompt-balanced weights normalized over structurally eligible rows, including unresolved rows. The band is resolved-positive mass through resolved-positive plus unresolved mass; it is not a confidence interval. Structurally inapplicable populations have named reasons. Zero displacement has zero raw gain and undefined normalized direction.
- **PF07:** comparisons use identical populations, weights, and strictness. Raw positive estimates, uncertainty-resolved signs, and nonnegative conditions remain distinct. None is renamed the original Proposition 5 condition.
- **Initial center:** UB02 compares unique unconditional evaluation seeds and distinct candidate atoms about exactly the same fixed reference center. The center's reference seeds and support provenance remain pinned. Figures use `mu`, while stored fields retain their scientific identifiers.
- **Synchronization:** TS02 uses the vector gap, and TS03 takes the paired sample maximum before aggregation. The early and late ECDFs share full-tail limits. TS05 recovers a valid final-current reference from the saved base scalar where the newer endpoint-only column is unavailable. It never invents a prediction at the final output tensor.
- **Terminal:** structural clean-update applicability, independent numeric replay, and independent affine accounting remain separate. Nonclean accounting includes the native additive/state discrepancy, both scaled error terms, all cross terms, and reconstruction residual. Missing independent DDPM sampling noise receives an explicit unavailable status. The theorem figure gate also respects the manuscript's guidance domain g>1.
- **Normalization:** coordinate RMSE equals the raw latent L2 norm divided by sqrt(d). Existing RMSE values are not divided a second time. The stored raw norms and dimension remain auditable. Axis equations use single-line division, SSCD colorbars say `SSCD`, and reference lines identify their meaning.

## Reviewed numerical population

The four complete reviewed candidate bundles were available under `outputs/` at execution time. Their original generation logs were independently verified under `_logs/`. Migration explicitly selected each candidate bundle and `_logs`; it did not substitute those records for new generation caches under `logs/`.

| Configuration | Prompts | Evaluation samples | Prediction rows | Conditional terminal figures |
|---|---:|---:|---:|---|
| SDv1/DDIM | 248 | 4,960 | 248,000 | Both omitted: `nonclean_affine_update` |
| SDv1/DDPM | 260 | 5,200 | 260,000 | Both omitted: `nonclean_affine_update` |
| SDv2/DDIM | 86 | 1,720 | 86,000 | Both omitted: `nonclean_affine_update` |
| RealVis/DDIM | 243 | 4,860 | 243,000 | Both omitted: `nonclean_affine_update` |

The three DDIM configurations retain a native final state coefficient about 0.706290, a clean-estimate coefficient kappa about 0.293887, and positive destination sigma about 0.029155. They therefore fail the exact clean-update structure. SDv1/DDPM has state coefficient 0 and kappa 1, but its saved native update retains a positive stochastic standard deviation about 1e-10; the strict zero-additive-noise prerequisite also fails. These settings were preserved.

All 16,740 final predictions have a valid current candidate-reference measurement. Independent affine accounting was computed from saved native coefficients and terminal tensor slices for all 11,540 DDIM samples. The 5,200 DDPM samples retain `unavailable_saved_sampling_noise` for that independent accounting; no output-fitted residual was substituted.

The same-population feedback audits found zero blocking endpoint identities and zero uncertainty-resolved positive-margin/negative-gain contradictions across the four configurations. The maximum per-step excess of the raw M3 positive-coverage estimate over stable positive-gain frequency was about 23.266 percentage points for SDv1/DDIM, 2.093 for SDv2/DDIM, and 26.811 for RealVis/DDIM; DDPM had no positive excess. The mismatching raw estimates were uncertainty-unresolved. Original M0 strict-positive coverage remained zero. These are numerical audit findings, not claims that every theorem assumption was observed.

The inspected `revised.pdf` SHA-256 is `fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`. A matching LaTeX source was not available in the checkout, so source-label verification is not claimed. The existing PDF formula mapping is retained.

## Reuse, retirement, and validation

Fresh paper analysis requests a shared per-record base/candidate pass. The base worker computes current references and independent terminal terms, then produces candidate endpoint/dose/control sufficient statistics while the same protected tensors remain loaded. Candidate publication imports those pinned extension shards after the base manifest completes. The core paper suite defers path integration; optional integrated diagnostics are separately resumable. Workers use the existing generation-style device selection and parent-owned progress reporting.

Compatible paper inputs can be rendered directly. Analysis refreshes reuse valid candidate shards and already audited terminal scalar terms when their source and measurement recipes agree. Changing renderer typography or filenames does not modify numerical tables. Explicit scientific reconfiguration archives the previous compact paper role. Locks and recoverable transaction markers serialize publication.

After successful paper publication, only hash-verified images listed by the selected legacy renderer manifests are retired. Their original paths and hashes are recorded under the model's `theory/archive/legacy-<analysis_hash>/manifest.json`. Numerical shards, old audit tables, proximity, selection, generation, and unrecognized user files stay in place. Repeated plotting does not repeatedly archive files.

Validation receipts are under `outputs/theory_paper_audit/`: migration receipts record candidate scientific hashes and modification-time checks; per-configuration export audits record every PNG and independently rendered PDF page. Synthetic regression tests cover shared-pass parity/read counts, multiworker resume, endpoint numerical contracts, exact weighted populations, tail references, clean/nonclean terminal accounting, CSV identities, unsafe paths, locking, image retirement, and whole-role publication failure.

## Executed checks and remaining environment state

- Broad regression: `python -m pytest -q tests/test_theory_*.py tests/test_proximity.py tests/test_proximity_gmm_core.py` — **568 passed, 2 skipped**, 531.44 seconds. The two skips require CUDA, unavailable on this host.
- Follow-up checks for the final lightweight preflight, cache invalidation, recovery marker, and CLI error handling passed. The direct fresh `run_paper` integration test passed with candidate second reads and quadrature forbidden. Shared-pass tests exercised real spawned CPU workers and exact two-pass numerical parity.
- Explicit cache migration completed for all four configurations. **6,043 authoritative candidate scientific files retained both SHA-256 and modification time.** No learned inference or generation was run for this task.
- All four real paper bundles passed saved-only rendering with raw tensor readers, scientific reducers/builders, noncompact CSV reads, and noncompact CSV hashes forbidden. All 37 checked numerical files per configuration remained unchanged.
- Final visual audit passed for **48 PNGs and 48 single-page PDFs**. All exported pages were inspected; vector fonts/lines and dense-point rasterization were checked. Only the three formerly clipped within-prompt labels required changed image content.
- Hash-verified legacy retirement archived **550 owned PNG/PDF files**, with original paths and hashes. Candidate numerical data, original audit tables, and protected upstream files were preserved.
- `ruff check` and shell syntax checks passed. Matching LaTeX source and real CUDA hardware validation were unavailable.

The actual `./run_all.sh --plot` invocation stopped before any rendering because **SDv1/DDIM's existing saved proximity configuration has a different `selection_hash` from the current frozen selection**. The other three configurations passed their paper-plus-proximity preflights. This is an existing upstream-cache mismatch, not a paper-export failure; analysis-only theory recomputation cannot repair it. No protected proximity or selection data were rewritten to disguise the mismatch.

Run the normal selected pipeline to refresh that configuration's matching upstream outputs, then use the full saved plot matrix:

```bash
./run_all.sh --model sdv1 --scheduler ddim
./run_all.sh --plot
```

The normal pipeline was verified with synthetic protected-cache orchestration tests, but was not executed against real checkpoints during this task. The user-facing normal command retains its existing generation/SSCD/proximity resume semantics.

The completed paper-only bundles can already be rendered independently, for example:

```bash
./theory_validation.sh --model sdv1 --scheduler ddim --plot
```

Receipts: [migration audit](../../outputs/theory_paper_audit/migration_audit.json), [saved-only guards](../../outputs/theory_paper_audit/plot_only_guard.json), [final visual audit](../../outputs/theory_paper_audit/final_visual_audit.md), [final output hash comparison](../../outputs/theory_paper_audit/final_comparison.json), and [broad test results](../../outputs/theory_paper_audit/regression_tests.xml).
