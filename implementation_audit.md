# Theory measurement implementation audit

## Pre-edit inspection and authoritative specification

This audit was written before the corrective implementation edits. The active
working tree contains the shared `utils/experiments/theory` package, the
`scripts/theory_validation.py` CLI, `theory_validation.sh`, `run_all.sh`, and
`tests/test_theory_*.py`. The prior theory package is untracked in this working
tree, while README, requirements, run_all.sh and tests/test_generation.py already
contain session changes. Those changes and all protected pipeline behavior are
preserved. This patch corrects that working implementation rather than reviving
legacy theorem scripts.

The authoritative task is the user attachment
`5c7c09b6-a448-4c28-9024-e409a0713438/pasted-text.txt`.
`revised.pdf` was checked against SHA-256
`fd447e263b921e94203558e6b03e2a9aabaa03ab8deb5cec7545427aed4f112f`.
Sections 2–3 and Appendix D define the measurement contracts. In particular,
Proposition 5 requires the cross-step integral V, not a competitor margin.

## Previous axes and intended replacements

In this table m_u/m_c are saved canonical-epsilon clean estimates, Delta=m_c-m_u,
x* is the paired target, d is the number of latent coordinates, and every
replacement candidate-reference quantity is explicitly qualified by K.

| Design | Existing horizontal / vertical quantities | Corrective contract |
|---|---|---|
| initial_recovery | ||m_u,T-x*||/sqrt(d) / ||m_c,T-x*||/sqrt(d) | Same paired k=0 values, unambiguous target-error column names; identical zero-inclusive limits and equal aspect; short RMSE labels, full formula/scope sidecar; equal-target-errors diagonal and finite initial-SNR/count annotation. |
| unconditional_center | saved SNR / ||m_u,t-mu_ref||/sqrt(d), all-step trajectories, continuous SSCD color | Initial-only ECDF: x=||m_u,T-mu_ref||/sqrt(d), y=cumulative fraction of unique evaluation seeds. Exactly N independent initial inputs; no prompt-specific SSCD color. Fixed declared reference center. |
| target_injection | <g Delta_T,v>/||v||² / ||g Delta_T-a_parallel v||/||v||, v=x*-center | Keep scalar values and direct perpendicular reduction; optional diagnostic only, short descriptive labels; add complete relative mismatch and near-zero-direction status. |
| matched_displacement | ||g kappa Delta||/sqrt(d) / ||x_next-x_cf||/sqrt(d) | No default or diagnostic image. Retain vector construction residual and separate independent replay QA, affine positive-kappa gate, shared-noise method and precision provenance. |
| posterior_feedback | minimum target-versus-competitor log-odds improvement / target log-odds gain, equality diagonal | Replace measurements, not labels: x=M_K/sqrt(d), M_K=||Delta||-||m_c-x*||-||m_u-mean_K,current||-V_K; y=H_K=log p_K,next(guided)-log p_K,next(matched). V_K integrates destination posterior-mean distance to the CURRENT posterior mean along the matched segment. Linear x, symlog y, zero lines, unresolved markers/counts; no equality diagonal. |
| target_synchronization | saved SNR / max conditional and unconditional target RMSE, dense seed trajectories | Four prompt-balanced median branch/cohort curves and interquartile bands: conditional solid, unconditional dashed; fixed SSCD>0.75 and <=0.75 descriptive groups, categorical legend, no colorbar. Retain every per-sample paired branch/joint metric and save fixed phase summaries. |
| terminal_reproduction | operational affine scheduler bound / observed endpoint RMSE | Remove default scatter. Always record endpoint and structural Eq.5 applicability. Optional terminal_terms only for applicable observations: x=A=||m_c,1-x*||/sqrt(d), y=B=(g-1)||Delta_1||/sqrt(d). Separately record Eq.16 cancellation, candidate Theorem7 terms and scheduler discrepancy. |

The new default inventory is four figures, each PNG+PDF. Two optional diagnostic
figures are injection and applicable terminal_terms. Seven manuscript statement
entries remain in the registry; algebraic QA is not promoted to a paper plot.
The latest task explicitly replaces earlier long-formula axis conventions and
suppression of essential compact scope/count annotations.

## Current center, support and scheduler findings

The reference center is the mean of initial unconditional estimates from one
canonical complete reference record, selected by minimum
(source_row_number, original_index) independently of selection/outcomes. It uses
reference seeds N..2N-1 once each. Repeated initial states/predictions are checked
across reference records and evaluation records with source-dtype tolerances.
The center is an estimated model-output center, not an identified training mean.
The replacement retains its source and adds explicit center kind/vector and
source hashes, true seed counts and initial-dispersion metadata. Zero and existing
compatible independent baseline modes remain explicit choices.

The old support builder scans ALL compatible experiment/reference/other run
folders. Exact float64-equal latent atoms are deduplicated and equally weighted;
record/image aliases are retained. Existing bundles have 442 atoms for sdv1 and
realvis and 401 for sdv2, each d=16384. This broader implicit source union will be
replaced by a declared source run: all valid complete experiment-run target
latents before retained-prompt filtering. No outcome pruning or frequency claim
is justified. Every exact atom weight, mapping, source hash, candidate mean and
target-specific radius must be recorded. A true training law is unavailable.

The adapter already distinguishes saved current and actual scheduler destination
noise levels and DDIM versus DDPM coefficients. In the four saved default
configurations, DDIM final coefficients are approximately A=0.706290 and
kappa=B=0.293887 with nonzero destination sigma, so Eq.5 is structurally false.
DDPM records A=0, B=1 but has a tiny positive final noise standard deviation
(1e-10); its exact structural applicability must be reported, not inferred from
an accidentally small observed residual. Non-affine clipping/thresholding and
nonpositive kappa cannot qualify for the manuscript matched identity.

The old operational bound sums separate state, unconditional, conditional,
target-offset and prospective-noise terms. It is not Theorem7 and cannot be
renamed as such. Candidate Theorem7 and Eq.16 A+B are separate quantities.

The old initial KL diagnostic uses the initialization-to-forward direction.
Appendix D.1 Eq.31 instead gives the Gaussian-transfer bound
0.5*sqrt(alpha_T²||x*||²+d*(sigma_T²-1-log(sigma_T²))) when initialization is
standard Gaussian and the manuscript noise contract holds. Both directions must
be named, and the bound logged unclipped/clipped; the true pair-specific forward
loss remains unavailable.

## Schema, orchestration and publication findings

Current schema is 2 / cache-theory-2.0, outputs under theory_v2. Current feedback
shards contain no V or Proposition5 margin and cannot be reused as new feedback
measurements. The new formula/schema identity will invalidate derived analysis
only. Plot-only must reject old or incomplete tables with an analysis-only
recomputation command and must never import posterior integration or raw readers.

Record sharding already follows generation's device resolver, round-robin
process convention and one aggregate progress bar. These are retained. The
current environment exposes CPU only (PyTorch 2.11.0+cu130; zero visible CUDA
devices); actual GPU execution cannot be claimed here. Posterior calculations
will use float64 affine segment logits, bounded node/query batches and cached
support matrices/Gram values, with explicit estimated quadrature status.

The current `recompute` path removes an existing same-hash complete manifest
before recomputing. Corrective publication must preserve the valid prior result
until all new required stages succeed. Record shards remain resumable, and
completion/index publication stays atomic. Only obsolete files owned by a
managed figure manifest may be archived; protected/raw/user files are untouched.

## Intended touched modules

- `utils/experiments/theory/contracts.py`, `statements.py`, and
  `docs/theory/statement_registry.json`: schema, evidence, inventory and exact mappings.
- `metrics.py`, `scheduler_adapter.py`, and `centers.py`: branch/initial/terminal
  contracts and source/structural applicability provenance.
- `support.py` and new `feedback.py`: stable finite-reference posterior and Eq.15
  integration with uncertainty, sign/saturation, and proof-identity QA.
- `reduce.py` and new `summaries.py`: one shared streaming reduction, explicit
  support source, scalar tables, balanced/descriptive summaries and safe publication.
- `plotting.py`: four saved-data designs, optional diagnostics, captions,
  numerical-status presentation and managed obsolete-figure archival.
- `scripts/theory_validation.py`, `run_all.sh`, README and theory documentation:
  minimal diagnostics/tolerance options and exact analysis/plot-only contracts.
- Existing/new theory tests: numerical mixture, scheduler, grouping, schema,
  immutable protected caches, resume/publication and loader-isolated plotting.

Implementation and real-run findings will be appended after verification;
planned behavior here is not a claim of completed measurement or validation.

## Pre-edit source hashes

| Module | SHA-256 |
|---|---|
| `utils/experiments/theory/contracts.py` | `a0cc8f3c6b5f6cb56b52355b81fe023a75cfe6487d1a1902344e6d11d759925d` |
| `utils/experiments/theory/cache_reader.py` | `646e89c413b34e8f0bb06ef340cf7d91261826bb2cb4100b907ab328aa0fb62d` |
| `utils/experiments/theory/centers.py` | `75a3fd99d28ac3f5441a0ea6a95764c47efc9745ac0b5645f51d1770b44ad24f` |
| `utils/experiments/theory/metrics.py` | `db4b676723b34eaff50e925b0294d8876ac1acb632468abdace075feaa6787fa` |
| `utils/experiments/theory/scheduler_adapter.py` | `0db9e0b2be249146003ded57b0148cd5835ac96d9a499ee78d22607644b13f17` |
| `utils/experiments/theory/support.py` | `beb31b177142cb811979a697602dd5881d811353f58c2294adc67ca97e2fae14` |
| `utils/experiments/theory/statements.py` | `b5c7486b8454d5bb80ad639e8fc7e3b1fbf4d41682bfa1cf2e20528155f738dd` |
| `utils/experiments/theory/reduce.py` | `99d25bb0f77bf5d6456b1c85e62001e7d9a9727d500ceab602e35cc55e3903c6` |
| `utils/experiments/theory/plotting.py` | `daf7a07d9642705b3c067486db30637bae3ef6b604fc67ec1f73920610630ed5` |
| `utils/experiments/theory/progress.py` | `cb327a6f64a878d50d78c8c362a0b9d08ead72fb07295adb9f67b635ecf30f52` |

## Completed implementation and verification

The corrective implementation uses schema 3 and formula version
`cache-theory-3.0-proposition5`; it preserves the `theory_v2` discovery parent.
Implemented changes cover `contracts.py`, `statements.py`, `metrics.py`,
`scheduler_adapter.py`, `support.py`, new `feedback.py`, `reduce.py`, new
`summaries.py`, `plotting.py`, the statement registry, CLI/runner options,
README, theory documentation, and theory tests. The existing cache reader,
center consistency/seed handling, multi-device progress helper, model/device
utilities and protected generation/SSCD/proximity behavior are reused.
`tests/test_generation.py` has updated documentation/module-inventory
expectations for the authorized new analysis helpers.

The final full repository run passed 999 tests and found two obsolete
documentation/module-list expectations. After correcting only those test
expectations, both focused regressions and all 46 generation tests passed.
All 90 plotting/CLI tests passed after the final center-caption correction.
Ruff, shell syntax and whitespace checks passed. The primary real measurements
are the user's four completed three-GPU runs: 837 prompts, 16,740 samples and
837,000 prediction rows, with zero computational record failures. Earlier CPU
audits remain separate supplemental evidence.

For full-cache immutability checks, the pre-run inventory covers 25,542 files
and 349,315,647,105 bytes. A deterministic bounded subset of 8,008 files
(494,991,049 bytes) has content SHA-256 values; all inventoried files have
size and modification-time records. This is not a full-content checksum of
all 325 GiB. The completed comparison retained the same 25,542 files with no
additions or removals. Generation/SSCD metadata and all checked generation/SSCD
hashes were unchanged. Concurrent selection/proximity refreshes changed metadata
for 32 files; 28 remained byte-identical and four experiment proximity summary
JSONs changed contents. Every frozen-selection CSV and proximity CSV retained its
hash. The comparison records these observations without assigning each write to
a process or claiming exhaustive content immutability. See
`outputs/theory_v3_audit/protected_cache_integrity.json` and the complete before/
after inventories beside it.

## Actual CUDA failure and correction

The user's three-GPU SDv1/DDIM run exposed an unsupported Boolean
`scatter_reduce_(amax)` accumulator in the new feedback integration helper.
The CPU-only test environment had accepted that operation. `feedback.py` now
uses int64 0/1 flags for the accumulator and reduction input and retains the
final Boolean interpretation. A regression rejects Boolean scatter kernels on
CPU and checks invalid-node persistence across chunks. Feedback tests passed
23 cases, with the local actual-CUDA case skipped because CUDA is not visible.
Forty scalar/status fields remained bitwise identical on eight synthetic cases
and three real cached timesteps.

The user's corrected three-GPU SDv1/DDIM retry completed all 248 prompts, with
83/83/82 records assigned to cuda:0/1/2 and no failures. It published the
complete 4,960-sample / 248,000-prediction-row bundle
`81085d1787c9e3f4f09b4f879ff3150cb94b3468ef01ae8b8b48e3104106bd83`.
The original CPU audits retain their accurate pre-fix source identities; the
exact numerical source snapshot is in
`outputs/theory_v3_audit/cpu_source_before_cuda_fix/`. The numerical arithmetic
is unchanged by this dtype correction, and the snapshot/parity checks are
recorded rather than retroactively changing those bundles' source hashes.

All four primary GPU configurations subsequently completed using the corrected
source. Their pinned identities and worker reports are in
`docs/theory/applicability_report.md`. The full scalar audit applies 71 checks
per configuration with no hard contract failures. Of 820,260 eligible feedback
comparisons, zero are estimated met, 786,437 estimated not met and 33,823 unresolved.
The 56 integral QA flags remain unresolved and are investigated separately;
no numerical inconsistency is interpreted as refuting a theorem. The measured
gains include 413,135 positive and 407,125 negative cases. Every preserved
terminal scheduler fails the exact structural Eq.5 gate. All 16 designated
figures have passed visual inspection, including signed feedback data and
caption/reference definitions.

## Final saved-data rendering and managed archival

The final four primary GPU bundles were rerendered with imports of model loaders,
raw tensor readers, feedback integration, support rebuilding and center fitting
blocked. All 3,456 numerical files, including bundle manifests, kept identical
SHA-256 and modification times. Twenty auxiliary tensor files were statted without
being read and retained their metadata. All 16 PNGs reproduced byte for byte;
all 32 PNG/PDF files matched the new figure manifests and retained exactly the
four default designs at style `theory-measurement-contract-8`. The earlier real
SDv2 root `run_all.sh --plot` check also passed with these runtime guards.

After all four current GPU bundles and images passed, 42 obsolete manifest-owned
schema-2 images were archived within their managed figure directories. Every
archived hash was verified, old manifest paths were updated, and a repeated dry
run found zero pending moves. The earlier SDv1 schema-2 bundle was already absent
before archival and is recorded as missing; its 14 images are not claimed as
archived. Unowned files and supplemental schema-3 CPU analyses were preserved.
The reproducible scripts, guarded-render report, visual audits, protected-cache
snapshots and archival records are under `outputs/theory_v3_audit/`; a concise
checked-in record is `docs/theory/validation/render_and_integrity.json`.
