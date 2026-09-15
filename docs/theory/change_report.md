# Cache-only theory implementation change report

## Current corrective patch: schema 3

The current implementation follows the corrective brief and the verified
`revised.pdf` (SHA-256
`fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`).
It supersedes the earlier seven-figure designs and schema-2 measurements
recorded in the historical sections below. Those earlier validation counts
and empirical results are historical, not measurements of Proposition 5's
new integral condition.

The defaults are four designs: paired initial target errors; an initial-only
unconditional ECDF using unique evaluation seeds and a held-out reference
center; matched candidate-distribution posterior feedback using Equation 15;
and branch-resolved, prompt-balanced synchronization summaries. Injection is
optional. The optional terminal-terms figure requires the structural and
numerical clean-update gate. There is no matched-displacement figure or
terminal-reproduction scatter in either current figure set.

The numerical bundle uses schema 3 and formula version
`cache-theory-3.0-proposition5`; its existing `theory_v2` parent directory is
retained for discovery compatibility. Old feedback tables without the
variation integral are rejected. The finite candidate bank now comes from
one declared experiment run, with equal mass on exact distinct atoms, before
prompt selection. Candidate-law diagnostics remain separate from unknown
training-reference quantities. Adaptive float64 quadrature reports estimated
or unresolved conditions, not certified signs.

A failed forced recomputation preserves the previously completed bundle.
Record shards remain resumable, numerical publication is atomic, and normal
analysis invokes the same scalar-only renderer as plot-only mode. The theory
reducer retains generation's device selection/process sharding and one parent
progress bar. The user's corrected runs exercised cuda:0/1/2 for all four
configurations and completed all 837 retained prompts without record failures.
The local audit host has no visible CUDA device; CPU tests and the actual GPU
execution evidence are reported separately.

See [implementation_audit.md](../../implementation_audit.md) for the pre-edit
formula/source audit, [quantity_dictionary.md](quantity_dictionary.md) for the
current schema, and [applicability_report.md](applicability_report.md) for
reference and scheduler availability. Completed corrective-patch measurements
and validation are recorded below.

```bash
./run_all.sh --recompute-experiments
./run_all.sh --plot
./run_all.sh --plot --include-diagnostics
```

Use the existing `--model` and `--scheduler` filters for a subset. An optional
`--target-error-tolerance` is a supplied raw-L2 tolerance; none is inferred
from SSCD or observed errors. Figure axes use RMSE, equal to a raw Euclidean
error divided by `sqrt(d)`, with complete definitions in caption sidecars.

## Corrective-patch verification

| Check | Result |
|---|---|
| `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests` | 999 passed; two obsolete README/module-inventory assertions failed (895.65 seconds) |
| Corrected assertions, focused rerun | 2 passed (29.96 seconds); only test expectations changed |
| `python -m pytest -q tests/test_generation.py` after those corrections | All 46 passed (69.59 seconds), including both original failures |
| `python -m pytest -q tests/test_theory_plot_cli.py` after final center-caption change | All 90 passed (32.74 seconds) |
| `python -m pytest -q tests/test_theory_feedback.py` after CUDA compatibility correction | 23 passed; one actual-CUDA test skipped on the local host |
| Ruff on theory package, CLI and theory tests | Passed |
| `bash -n run_all.sh theory_validation.sh`; `git diff --check` | Passed |

The complete suite was scoped explicitly to `tests` so collection did not
traverse preserved NAS caches. Numerical production files remained fixed during
the final full-suite run and real reductions. The subsequent style-8 change
only corrected descriptions of explicitly selected zero/saved centers; plotting
is excluded from the numerical identity and its entire test file was rerun.
The original suite's two failures are both resolved by the passing generation
file rerun; there is no remaining known failing test.

Coverage includes sharp interior mixture transitions, candidate atom aliases
and weights, estimated/unresolved margins, negative posterior gains,
log-complement saturation, endpoint/integral agreement, independent scheduler
replay and constructed displacement QA, paired synchronization, strict clean
terminal applicability, cancellation/bound terms, source independence,
loader-blocked plot-only rendering, forced-recompute preservation/rollback,
resumable shards, and aggregate multi-device progress. Synthetic cache tests
compare all protected fixture bytes before and after analysis and plotting.

## CUDA regression found during the user's real run

The user's SDv1/DDIM run in bundle
`27d2e537d3aad45948a448929d77883ea8299f8e00567efe7c6e1eafed41dd0f`
started three actual workers on cuda:0/1/2, assigning 83/83/82 prompts. Every
record failed at the same operation: the CUDA implementation of
`scatter_reduce_(reduce="amax")` did not support the Boolean invalid-node
accumulator in `feedback.py`. The CPU test environment had accepted that dtype.
This was a computation bug after device dispatch, not an absence of multi-GPU
sharding.

The accumulator and its 0/1 inputs now use int64. Conversion back to the final
Boolean invalidity flag is preserved. No floating-point measurement formula,
integration tolerance, or eligibility rule changes. A new CPU regression rejects
Boolean scatter reductions and verifies that an invalid node remains flagged
across later chunks. An actual CUDA parity test is also provided.

Validation: **23 feedback tests passed; one CUDA test skipped** because this
validation host has no visible CUDA device. All 40 scalar/status fields matched
bit for bit between old and new implementations on eight synthetic cases and
three real cached SDv2 timesteps (two seeds each). Ruff passed. These checks
establish the fix's CPU parity. The subsequent actual three-GPU retry completed
all 248 retained prompts without failures and published bundle
`81085d1787c9e3f4f09b4f879ff3150cb94b3468ef01ae8b8b48e3104106bd83`.
Workers on cuda:0/1/2 reduced 83/83/82 records in 144.17/145.55/143.82 seconds
each (these are worker durations, not total run time). Its 4,960 initial and
endpoint observations, 248,000 prediction rows, and all four style-8 figures
passed saved-data/visual checks. The retry command was:

```bash
./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments
```

The earlier CPU audits loaded the pre-fix source before this CUDA
failure was reported. Their original source hashes remain recorded accurately;
the exact source snapshot is preserved under
`outputs/theory_v3_audit/cpu_source_before_cuda_fix/`. The dtype fix changes only
integer/Boolean invalidity bookkeeping, with the parity checks above, so those
CPU measurements are not relabeled as having used the later source file.

During the concurrent user run, the earlier SDv1/DDIM output bundles and the
ongoing CPU bundle's headers disappeared, while frozen-selection/proximity
metadata were refreshed. The old CPU run was stopped because it could no longer
publish a valid bundle; a new CPU reduction used the corrected implementation until the user confirmed
the GPU retry was running, at which point the duplicate CPU job was stopped.
Generation and SSCD files were unchanged in the immediate metadata inventory.
Seven of eight changed selection/proximity metadata files remained byte-identical;
only experiment proximity `summary.json` changed contents (1610 to 1611 bytes).
The final integrity report distinguishes this concurrent activity from the
assistant's analysis-only/plot-only writes. Missing historical figures are
recorded separately from successfully archived files.

## Completed primary GPU measurements

All four primary bundles use the corrected source and all three devices. Their
identities, per-worker record counts, and durations are listed in the
[applicability report](applicability_report.md). The original failed bundle is
preserved as the failure audit; current indexes select complete GPU results.

The primary analyses retain **837 prompts, 16,740 evaluation samples, and
837,000 prediction rows**, with no computational record failures. Among 820,260
eligible nonterminal feedback comparisons, zero are `estimated_met`, 786,437
are `estimated_not_met`, and 33,823 are `numerically_unresolved`. Unresolved
conditions remain in the coverage denominator. There are 413,135 positive and
407,125 negative gains, including 70,123 negative gains on high-SSCD trajectories.
Stable log-odds ordering retains signs for 199,435 gains whose plotted H
underflows to zero; no artificial nonzero values are inserted.

The pinned scalar audit checks 71 contracts per configuration, with no hard
contract failures, exhausted quadrature budgets, or proof-slack violations.
All 56 endpoint/integral QA flags (8/17/12/19 by configuration) remain unresolved
and have a separate [numerical investigation](numerical_qa_investigation.md).
These are estimated candidate-law calculations, not formal certificates or
measurements of an identified training distribution. No producer tolerance or
classification was changed after inspecting the results.

All four configurations fail the exact Eq.5 structural terminal-update gate.
The optional terminal-terms figure is therefore unavailable for these caches;
endpoint and last-clean-estimate diagnostics remain recorded separately.
All 16 default figures (32 PNG/PDF files) were rendered and visually inspected
for labels, units, limits, reference lines, counts, and both gain signs.

## Final rendering, protected inputs and obsolete figures

All four primary GPU bundles passed a second render with model loaders, raw
`.pt` access, posterior integration, support construction and center fitting
blocked. All **3,456 numerical files** retained identical SHA-256 and modification
times; all 16 PNGs reproduced byte for byte, and all 32 PNG/PDF hashes matched
their figure manifests. Twenty auxiliary tensors retained their file metadata
without being read. The actual SDv2 root plot-only runner also passed the same
runtime guards in a supplemental CPU-bundle check.

The bounded protected-input comparison inventories 25,542 files (349,315,647,105
bytes before the runs) and hashes 8,008 selected files (494,991,049 bytes before).
It does **not** checksum all 325 GiB. No files were added or removed, and no
generation/SSCD metadata or checked hashes changed. During the concurrent user
runs, 32 selection/proximity files acquired changed metadata; 28 are byte-identical
and four experiment proximity summary JSONs differ in content. All frozen
selection and proximity CSV hashes are unchanged. The inventories cannot assign
each write to a process or recover exact previous JSON fields from hashes alone.
This is a completed comparison with recorded changes, not an unrestricted claim
that all protected contents remained identical.

Once all four replacement GPU publications passed, 42 old managed schema-2 images
were archived with hash verification, manifest updates and an idempotency check.
The missing historical SDv1 bundle is explicitly recorded as absent before
archival; none of its images is counted as moved. Unowned files and supplemental
CPU schema-3 results remain intact.

The [compact rendering/integrity record](validation/render_and_integrity.json)
contains the final checks. Full inventories, per-file hashes, guarded rendering
scripts/logs, all four visual audits, and archival reports are preserved in
[`outputs/theory_v3_audit`](../../outputs/theory_v3_audit).

## Historical implementation and validation record (schema 2)

The default theory pipeline now reduces preserved generation/SSCD/proximity
artifacts once per scientific configuration, saves scalar analysis bundles, and
renders all registered figures from those saved tables. It performs no additional
model evaluations. `--plot` renders saved results; `--recompute-experiments`
bypasses upstream executables and rebuilds only the derived theory analyses.

## Preserved stages and files

The implementation preserves these scientific stages and their existing contracts:

| Stage | Preserved files |
|---|---|
| Dataset preparation and recovery | `download_webster.sh`, `scripts/download_webster.py`, dataset download/recovery helpers |
| Generation and cache publication | `generate.sh`, `scripts/generate.py`, `utils/experiments/generation.py`, `utils/experiments/cache.py` |
| Model, scheduler and latent semantics | Existing `utils/models/` loading, sampling, scheduler, prediction-conversion, schedule-metadata and latent helpers |
| SSCD | `sscd.sh`, `scripts/sscd.py`, `utils/experiments/sscd.py`, `utils/metrics/sscd.py` |
| Proximity and frozen selection | `compute_proximity.sh`, `scripts/compute_proximity.py`, `utils/experiments/proximity.py`, `utils/data/selection.py`, `utils/data/proximity_gmm.py`, existing proximity plotting |

The supported matrix remains SDv1/DDIM, SDv1/DDPM, SDv2/DDIM and RealVis/DDIM.
Defaults remain `g=7.5`, `T=50`, `N=20`. Targets, preprocessing, checkpoints,
inference timesteps, stored canonical-epsilon semantics, experiment/reference
seed blocks, raw-L2 proximity and prompt eligibility are preserved. Every
experiment seed of each selected prompt remains in the analysis.

## Replaced defaults and added products

`run_all.sh` replaces the default independent execution of
`theorem1_loss_recovery`, `lemma2_mean_convergence`, and
`corollary3_cfg_amplification` with `theory_validation.sh` and
`scripts/theory_validation.py`. The old wrappers remain explicit legacy
utilities. Independent forward corruption, the temporary decoded gallery and
independently inferred unconditional baselines are not default prerequisites.

The new `utils/experiments/theory/` package validates source contracts, streams
cache records, constructs a fixed reference-initial center and an explicit
candidate distribution, reconstructs supported matched sampler updates, reduces
shared metrics, evaluates applicability and renders saved scalar tables. Atomic
record shards permit interrupted reductions to resume. Completion manifests
publish only after required observations succeed. Scientific hashes are separate
from rendering-style hashes.

Each numerical bundle includes its source identities, schedule, center/support
metadata, statement registry, initial/trajectory/endpoint scalar tables, failure
audit, summaries and plot-ready Parquet files. One PNG/PDF pair per registered
statement is written with figure provenance and exclusions. A copied scalar
bundle can be plotted without model weights or scientific tensor payloads.

## Verified manuscript mapping

Section 3 and Appendices D.1–D.7 of `revised.pdf` were read. The manuscript has
**seven** formal result statements, rather than the six semantic designs in the
planning brief. Definition 1 supplies the terminal memorization criterion and
does not require an additional result figure.

| Actual result | Stable semantic ID | Main evidence |
|---|---|---|
| Theorem 1 — Initial conditional recovery | `initial_recovery` | Finite initial conditional/unconditional target errors |
| Lemma 2 — Initial unconditional baseline | `unconditional_center` | Finite learned-branch distance to a fixed empirical reference center |
| Corollary 3 — Initial CFG amplification | `target_injection` | Target-direction coefficient and off-target residual |
| Lemma 4 — Matched one-step CFG displacement | `matched_displacement` | Same-state, same-noise displacement identity; algebra QA |
| Proposition 5 — Matched one-step target-posterior feedback | `posterior_feedback` | Explicit empirical-support posterior diagnostic |
| Lemma 6 — Target-specific synchronization | `target_synchronization` | Joint error of both branches to the same paired target |
| Theorem 7 — Final reproduction | `terminal_reproduction` | Endpoint error against a separately named operational scheduler bound |

The full transcription, assumptions, proof references, source SHA-256,
unavailable quantities and figure mapping are in
[statement_registry.json](statement_registry.json). Units, definitions and
evidence classifications are in
[quantity_dictionary.md](quantity_dictionary.md).

## Scientific limits retained in outputs

- Population forward-corruption loss is unavailable. The effective target-noise
  residual identity is labeled `exact_cache_algebra`, never a training loss.
- The initial center is an empirical mean of unique reference-seed unconditional
  estimates. It is not the true training mean, and its seed bank also informed
  frozen selection. One finite schedule does not establish a high-noise limit.
- The single-target training-law condition remains unverified by paired caches.
  True unconditional posterior means, target posteriors and the full training
  support radius remain unidentified. Candidate-distribution values retain
  `empirical_support_surrogate` labels.
- Proposition 5's actual condition is
  `||Delta_t|| >= e_t(c)+e_t(empty)+V_t`, with a continuous cross-step reference
  variation integral. Its unidentified RHS/slack is not replaced by the
  candidate minimum-margin condition.
- Theorem 7's actual bound is
  `g*e_1(c)+(g-1)*(e_1(empty)+R*(1-p_1))`, assuming the clean terminal update
  and `g>1`. It remains unavailable when its training-law inputs are missing.
  The operational affine-scheduler bound has its own name, target-offset
  correction, noise probability and applicability fields.
- Branch agreement can represent synchronized failure; joint target error
  measures proximity to the same fixed target. Suffix maxima cover the saved
  grid and are monotone by construction. Descriptive quantiles are not
  confidence intervals.
- Unsupported scheduler configurations and nonfinite/degenerate quantities
  retain explicit statuses. Casting to float64 does not recover inference or
  storage precision. No new guidance trajectory or SSCD outcome is inferred
  from frozen-state algebra.

## Validation record

All synthetic validation runs use the repository network guard and require no
model downloads. The integration fixture writes real tiny generation tensors,
SSCD markers, saved scheduler metadata, frozen selection and shuffled proximity
observations. It includes duplicate raw prompts, leading-zero string IDs,
nonselected support candidates and unsuccessful experiment seeds.

Final verification used `/nas/home/juyeop/miniconda3/envs/py313/bin/python3.13`.

| Command / check | Observed result |
|---|---|
| `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q` | **878 passed**, 136.31 seconds; complete repository suite, including preserved regressions |
| `python -m pytest -q tests/test_theory_plot_cli.py tests/test_theory_reduction.py` | **72 passed**, 41.04 seconds; rerun after the evidence-footer layout adjustment |
| Mathematical and source-contract component tests | 56 math tests and 22 contract tests passed; also included in the complete suite |
| `python -m ruff check utils/experiments/theory scripts/theory_validation.py tests/test_theory*.py` | Passed |
| `bash -n run_all.sh theory_validation.sh`; `git diff --check` | Passed |
| Source preservation | Git diff contains no changes to the protected files listed above |

Integration validation blocks network access, model loading/calls, JIT model
loading and latent encode/decode. It verifies exact seed joins, failed-sample
retention, target units, atomic failure/resume, failure history, scientific-hash
changes, stale source payloads, scalar tampering, completion inventories and
manuscript applicability. Plot tests remove tensors from copied bundles, block
raw reads/model imports, compare PNGs, and verify unchanged numerical hashes and
timestamps. Metadata-only audit refresh preserves numerical results.

## Actual-cache smoke audit

**Completed on CPU, with network access and model evaluation blocked.**
The final reduction used SDv1/DDIM, `g=7.5`, `T=50`, `N=20`,
`center=reference-initial`, `smoke_record_limit=1`. The canonical eligible prompt
is string `original_index=1730933243` (`sdv1-0000`). All experiment seeds `0..19`
and all 50 actual denoiser-input steps were retained. The saved endpoint is a
separate state; no terminal branch was fabricated.

This is an explicitly scoped smoke prefix. The full 248-selected-prompt run and
the four-configuration numerical matrix were **not** executed. The smoke bundle
is not published as a full-run index entry. To build complete numerical bundles,
use `./run_all.sh --recompute-experiments` (optionally filter model/scheduler);
root `--plot` requires those complete saved bundles.

| Smoke result | Observed value |
|---|---|
| Analysis hash | `ff6a952208e6be54382d18ac9b24bfe284f8703143ebb7b5dffd118b9de31458` |
| Bundle | `outputs/sdv1_ddim_g7.5_T50_N20/theory_v2/ff6a952208e6be54382d18ac9b24bfe284f8703143ebb7b5dffd118b9de31458/` |
| Unique candidate support | 442 exact atoms from 1800 compatible complete target-cache memberships across SDv1 DDIM/DDPM reference/experiment runs; uniform weights; no outcome filtering |
| Reference center | 450 reference records checked; 20 unique seeds `20..39`; repeated initial unconditional outputs exactly equal in this cache |
| Reduced data | 1 prompt, 20 experiment seeds, 1,000 trajectory rows, 20 initial rows, 20 endpoints |
| Designated figures | Seven PNG/PDF pairs with registry evidence, scalar hashes and exclusions |
| Plot exclusions | No nonfinite primary observations excluded in this smoke configuration |
| Protected integrity | Required generation/target/SSCD payload hashes and frozen source metadata validated; no upstream artifact writes |
| Inference | No model evaluations or new decoding/scoring; CPU tensor arithmetic only |

The same final scalar bundle can be rendered directly:

```bash
./theory_validation.sh --bundle outputs/sdv1_ddim_g7.5_T50_N20/theory_v2/ff6a952208e6be54382d18ac9b24bfe284f8703143ebb7b5dffd118b9de31458 --plot
```

This smoke example does not produce a uniformly favorable result. Initial
conditioning reduces target RMSE for **19/20** seeds, with median reduction
**0.0320**, but terminal target SSCD ranges only from **0.0364 to 0.1213**.
The surrogate minimum-margin condition is positive on **24/1,000** updates;
surrogate log-odds gain is positive on **614/1,000**, including 590 updates whose
minimum margin is nonpositive. This is consistent with sufficiency rather than
necessity. The operational endpoint inequality covers **20/20** samples but is
loose: median bound **7.2825 RMSE**, observed endpoint error
**1.5112 RMSE**, median bound/error ratio **4.82**.
These are descriptive measurements of one predeclared smoke prompt, not
population estimates or validation of missing paper assumptions.

The actual initial SNR is **0.00580905**. The reference center's
Monte Carlo standard-error diagnostic is **0.04872 RMSE**
from 20 unique reference seeds; it is not a certified confidence radius.

The final real scalar-only copy was independently rendered with Torch,
Diffusers, Transformers, model modules, reduction/support modules, raw tensor
file reads and network connections blocked. **All seven PNG hashes matched**
the final repository bundle; **22 numerical files retained identical bytes and
modification times**. Torch was never imported. The final bundle's recorded
reducer source hashes were checked against the delivered source files.


## Multi-GPU theory execution

Theory now defaults to `--device auto`, matching generation's public device
convention. Automatic CUDA selection uses every GPU visible to PyTorch;
`CUDA_VISIBLE_DEVICES` restricts that set. `--device cuda` and `cuda:N` select
one GPU, and `--device cpu` selects CPU execution. The theory CLI passes the
requested device unchanged to the reducer. Root normal and cache-only rebuild
modes already forward the same device option used by generation.

Prompt records are the unit of parallel reduction: all seeds and timesteps for
one prompt stay together. Worker count is bounded by visible devices and selected
records. The reducer uses the shared device/sharding helpers and generation's
spawned-process convention, with one shard executed in the parent process.
Workers consume one fixed center and candidate support and publish atomic record
shards. The parent combines and validates saved scalar results before publishing
a complete bundle. Plotting and validation remain independent of CUDA and
reduction-worker imports.

The earlier validation and real-cache audit above describe the original CPU
implementation; they are not GPU performance measurements. Follow-up CLI
validation used the same Python 3.13 environment:

| Command / check | Observed result |
|---|---|
| `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_theory_plot_cli.py` | **68 passed**, 16.00 seconds; automatic and explicit device forwarding, root dispatch and tensor-free plot/validation modes |
| `python -m ruff check scripts/theory_validation.py tests/test_theory_plot_cli.py` | Passed |
| `bash -n run_all.sh theory_validation.sh`; `git diff --check` | Passed |

Example cache-only rebuild on a selected GPU set:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_all.sh --recompute-experiments --device auto
```


The reducer reuses `resolve_devices`, `round_robin_shard`,
`worker_count_for_tasks` and `configure_worker_cpu_threads` from generation's
shared device helpers. Reference initial-state consistency and source-hash checks
use a separate spawned CPU reader pool. Every reduction worker uses the same
saved global experiment initial anchor, reference center and support; those are
never refit inside a shard. Eight-step trajectory blocks reduce repeated device
transfers, and scalar columns are copied to CPU together by device and dtype.
Float64 arithmetic and original integer/boolean scalar types are preserved.

Execution metadata records requested/resolved devices, worker process IDs,
assigned prompt IDs, per-record statuses, CPU thread counts and durations.
Worker reports and atomic per-record completion stamps retain progress across
failures. Device count and CUDA ordinals are execution metadata rather than
scientific identity, so a partial run can resume on a different number of GPUs
with the same backend and reuse completed record files. CPU and CUDA results
remain separate identities. Automatic MPS selection falls back to CPU because
the reductions require float64; an explicit MPS request fails clearly.

Added regression coverage exercises real two-process CPU execution as well as
mocked one-, two- and eight-GPU dispatch. It checks scalar parity, a shared global
center and initial anchor, disjoint prompt coverage, distinct process IDs,
corruption detection in child reference readers, failure recovery, and unchanged
successful shards after switching from two workers to one. Separate tests check
one-, two- and eight-step staging parity and packed scalar type/precision
preservation. These tests do not claim that CUDA kernels or multi-GPU throughput
were measured: this host reports PyTorch `2.11.0+cu130`,
`torch.cuda.is_available() == False` and zero visible CUDA devices.


Final multi-GPU implementation validation:

| Command / check | Observed result |
|---|---|
| `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q` | **905 passed**, 187.59 seconds, including real spawned CPU workers and all retained regressions |
| Final CLI regression after help clarification | **68 passed**, 15.81 seconds |
| Ruff across the theory CLI, package and all theory tests | Passed |
| `bash -n run_all.sh theory_validation.sh`; `git diff --check` | Passed |
| Diff of protected dataset, generation, model, SSCD, selection and proximity files | Empty; existing behavior preserved |
| CUDA execution / speedup | Not measured; no visible CUDA devices on this host |


## Shared theory progress display

Theory reduction now renders one coordinator-owned `[Theory] Records` tqdm bar
across all active devices, following proximity's record units, dynamic terminal
width and visible stderr output. It shows aggregate completion, rate/ETA, device
count and reduced/resumed/failed totals. Workers send small status events after
record completion; they no longer print individual record-progress lines. A
coordinator consumer thread keeps the bar live while the coordinator computes
its own GPU shard. Final results reconcile missing events or worker failures
without counting any prompt twice. The bar and event queue close after the worker
pool finishes, including error paths. Reference validation status precedes the
shared reduction bar.

Validation for this display change:

| Command / check | Observed result |
|---|---|
| `pytest -q tests/test_theory_parallel.py tests/test_theory_reduction.py` | **21 passed**, 68.14 seconds; real spawned reductions, parity, failures and resume, with one bar per run |
| `pytest -q tests/test_theory_progress.py tests/test_theory_plot_cli.py` | **72 passed**, 19.77 seconds; live child updates during coordinator work, deduplication, status reconciliation, cleanup and CLI/plot isolation |
| Ruff on changed Python files; `git diff --check` | Passed |
| Protected proximity, generation and model helper diff | Empty |

Both pytest commands used the same Python 3.13 environment with
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. Progress transport was exercised with real
spawned CPU processes; this change does not add a CUDA throughput measurement.


## Manuscript figure labels

All seven theory figures now use the notation verified from `revised.pdf`:
$\hat{\mathbf{x}}_t$, $\boldsymbol{\Delta}_t$, $\mathbf{x}_{t-1}^{\mathrm{cf}}$
and $\mathrm{SNR}_t$. Normalized Euclidean quantities display the explicit norm
and $\sqrt d$ denominator. The dimensionless projection and log-odds quantities
retain their actual definitions. Colorbars read `SSCD`, and median-line legends
read `Median`. Initial annotations use $\mathrm{SNR}_T$.

Reference-center, candidate-posterior and operational-bound qualifiers remain
explicit. The notation adapts to reference-initial, zero and cached-baseline
center modes; it does not label a surrogate as an unavailable paper quantity.
The renderer records its axis, colorbar and legend strings and auxiliary symbol
definitions in the figure manifest. The style version is now `theory-main-2`.
Additional footer spacing keeps the taller mathematical labels separate from
the evidence notes. The quantity dictionary documents the auxiliary symbols.

Validation:

| Check | Observed result |
|---|---|
| `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_theory_plot_cli.py` | **68 passed**, 22.80 seconds; scalar-only rendering, parity, provenance and CLI checks |
| Manual Matplotlib layout/notation check | All seven x-axis labels clear the evidence footer; all three center modes render the correct center symbol |
| Independent manuscript/metric review | All seven axis pairs agree with their stored quantities and the applicable manuscript notation |
| Ruff on the changed Python files; `git diff --check` | Passed |


The three completed scalar bundles available during this update were re-rendered
with the final labels, producing **21 figures / 42 PNG and PDF files**:

| Configuration (`g=7.5`, `T=50`, `N=20`) | Selected prompts | Trajectory rows | Numerical files with identical hashes and timestamps |
|---|---:|---:|---:|
| `sdv1/ddim` | 248 | 248,000 | 1,017 |
| `sdv1/ddpm` | 260 | 260,000 | 1,065 |
| `sdv2/ddim` | 86 | 86,000 | 369 |

All 42 final image hashes match their figure manifests, and all three figure
manifests record the delivered renderer's style hash. Visual inspection of the
full-data posterior and projection figures confirmed readable formulas and
separate evidence footers. Re-rendering used saved scalar bundles only.


## Remove small plot annotations

The figures now omit evidence footers, inline initial-SNR and prompt/seed-count
notes, empty-data notes and small diagnostic titles. Mathematical axes, SSCD
colorbars, data/reference legends and reference lines remain. Evidence classes,
initial SNR, prompt/seed counts and exclusions are saved in the figure metadata.
This changes presentation only; the evidence classifications and scalar
quantities retain their existing scientific definitions.

The renderer style version is `theory-main-3`; the earlier annotation and footer
layout descriptions above document previous renderings. The final plotting/CLI
regression run passed **68 tests in 18.56 seconds**. Ruff and whitespace checks
passed. Each regenerated figure is also checked for the absence of free text
annotations and titles before publication.


All four completed configurations (`sdv1/ddim`, `sdv1/ddpm`, `sdv2/ddim`,
`realvis/ddim`, each at `g=7.5`, `T=50`, `N=20`) were regenerated: **28 figures
/ 56 PNG and PDF files**. All final image/style hashes match their manifests.
The **3,448 numerical files** across these bundles retained their hashes and
modification times. A full-data posterior figure was visually checked to confirm
that the small annotations and title are absent.


## Inline axis divisions

All axis labels now use inline slash divisions instead of stacked fractions.
Min/max index conditions sit beside their operators, so the equations occupy
one line. The optional manuscript terminal-bound numerator retains visible
brackets around the entire sum before division. The renderer style version is
`theory-main-4`.

Validation used **68 passing plotting/CLI tests in 18.28 seconds**, plus manual
Matplotlib rendering of all seven axis layouts, all three center modes, and the
grouped manuscript-bound label. Ruff and whitespace checks passed. Every
regenerated figure is checked for inline-only axis text and for the continued
absence of free annotations and diagnostic titles.


Regenerated all four completed configuration bundles: **28 figures /
56 PNG and PDF files**. Their final image and style hashes were verified.
All **3,448 numerical files** retained their hashes and modification times.
The full-data posterior plot was visually checked for inline divisions and
side-positioned minimum conditions.


## Plain mu in figure labels

The default center is now displayed as bold mu without the ref subscript in
the unconditional-center and target-injection axes. The quantity dictionary
defines this display shorthand; saved center provenance retains its original
meaning. The style version is `theory-main-5`.

The existing rendering-parity and style/center checks passed (**2 tests**,
7.62 seconds). Ruff and whitespace checks passed. During regeneration, every
axis label is checked for absence of the ref subscript, and reference-initial
center labels are checked to use the plain mu symbol.

All 28 figures / 56 PNG and PDF files were regenerated across the four
completed bundles. Final image/style hashes were verified; all 3,448
numerical files retained their hashes and modification times.


## Manuscript notation audit and reference legends

Rechecked all seven designated figures against `revised.pdf`, their scalar
definitions and all four completed bundles. Removed display-only projection,
operational-bound and candidate-posterior symbols absent from the manuscript.
Target injection now shows the projection directly and expands its perpendicular
ratio using an equivalent Gram-determinant expression; neither coordinate is
renormalized. Candidate log-odds and the scheduler endpoint bound use descriptive
axis labels because the manuscript has no identical quantities. Plain mu is
used for both nonzero center modes, with source provenance retained in metadata.
The per-coordinate errors remain Euclidean norms divided by sqrt(d).

The operational endpoint scalar cannot be relabeled as Theorem 7: the saved
DDIM terminal update retains nonzero state and target-offset contributions,
and the DDPM bound includes its prospective noise term. The literal paper
bound remains guarded by explicit registry selection and identified inputs.
The figure metadata now identifies the actual plotted scalar columns and
describes whichever endpoint bound is selected.

All vertical, horizontal and diagonal references now identify their meaning in
legends. Curve initial-SNR lines read Initial SNR: SNR_T, and aggregate lines
remain Median. The injection marker identifies the target-aligned reference
(g,0). The renderer includes full axis-label extents when exporting so long
inline equations are not clipped. SSCD labels, inline slash divisions and the
absence of diagnostic titles/free annotations are preserved. The style version
is `theory-main-6`.

Validation: **68 plotting/CLI tests passed in 17.75 seconds**; Ruff and whitespace
checks passed. Independent runtime checks covered all seven figures in all three
center modes, every reference legend, and ordinary/partial/unidentified versus
identified paper-bound selection. The perpendicular expression agreed with the
existing residual computation within 1.2e-15 for negative, zero and positive
projections. Full-data images for all seven figure types were visually inspected.

Regenerated **28 figures / 56 PNG and PDF files** across sdv1/ddim, sdv1/ddpm,
sdv2/ddim and realvis/ddim. Every final image/style hash matches its manifest.
All **3,448 numerical files** retained their hashes and modification times.
