# How Diffusion Models Memorize

Diffusion memorization experiments with preserved generation, SSCD and proximity caches. The default theory stage measures **four mechanism experiments** and eleven fixed appendix figures. Analysis may run missing learned denoiser probes; plotting uses saved compact scalars only.

## Run

```bash
python -m pip install -r requirements.txt
./run_all.sh --download                      # Prepare data and resume the normal pipeline
./run_all.sh                                 # All four supported configurations
./run_all.sh --model sdv1 --scheduler ddim    # One configuration
./run_all.sh --recompute-experiments          # Theory only, including missing learned probes
./run_all.sh --refine-numerics                # Saved evidence arithmetic only; no learned probes
./run_all.sh --plot                          # Saved proximity and paper figures
./run_all.sh --plot --diagnostics            # Also export available behavioral diagnostics
```

Defaults remain guidance 7.5, 50 updates, 20 evaluation seeds and GMM selection. Supported pairs are SDv1/DDIM, SDv1/DDPM, SDv2/DDIM and RealVis/DDIM. Evaluation uses seeds `0..N-1`; reference selection uses `N..2N-1`. Every evaluation seed of each retained prompt remains included. All wrappers honor `PYTHON`.

`--recompute-experiments` bypasses download, generation, SSCD and proximity. It resumes valid independent probe and analytical shards, and computes missing work. `--refine-numerics` resumes numerical signs and original-condition intervals from existing evidence, bypassing learned probes as well as upstream stages. Missing inputs report an exact recomputation command. Normal derived analysis includes the numerical policy automatically. `--overwrite` retains its explicit upstream regeneration meaning. Plot mode rejects these rebuilding options and never invokes models, raw tensor readers, posterior integration or scalar reducers. It validates all requested bundles before rendering.

## Direct measurement options

| Option | Analysis default | Meaning |
|---|---:|---|
| `--num-loss-seeds` | 64 | Independent forward-target noise draws per pair and measured timestep |
| `--loss-seed` | 0 | Root for domain-separated deterministic probe streams |
| `--loss-timesteps` | `initial` | `initial` or all `saved` native prediction timesteps |
| `--num-unconditional-loss-seeds` | 256 | Optional forward-marginal draws; an explicit count enables the stage |
| `--unconditional-loss` / `--no-unconditional-loss` | enabled for `saved` | Override the optional marginal-loss stage |
| `--counterfactual-unconditional` | off | Separate learned empty-prompt comparison at matched next inputs; no rollout |
| `--counterfactual-steps` | `0` | Fixed comma-separated chronological update indices for the enabled supplement |
| `--probe-batch-size` | 8 | Execution batch size, with bounded CUDA OOM retry |
| `--reference-law` | `cached-targets` | Uniform distinct compatible cached targets before selection, or `manifest` |
| `--reference-manifest` | absent | Required local atoms/weights/provenance manifest for `manifest` mode |
| `--reference-snr-decades` | 6 | Depth of the fixed 97-point analytical-only grid below initial SNR; no network calls there |
| `--terminal-noise-run-alpha` | 0.05 | Predeclared simultaneous terminal Gaussian noise failure budget within one scientific run |
| `--numerical-decimal-precision` | 64 | Selective directed arithmetic, with retry at twice the precision |
| `--numerical-max-decimal-products` | 2000000 | Charged Decimal operation budget per row across both attempts; zero disables fallback |
| `--numerical-max-variation-nodes` | 65 | Per-row variation enclosure node budget |
| `--numerical-variation-absolute-width` | 1e-6 | Requested enclosure width in raw latent L2 units |

Gaussian probes use a single fixed bank of the `N` evaluation seeds on every saved native timestep. These are distinct from generated trajectory states and independent forward-target/marginal loss draws. Theorem 1 shows pair-level conditional RMS and a matched unconditional control, with independent forward-loss scale and saved Monte Carlo bootstrap intervals. Pair means use mean terminal SSCD colors; raw seed tails remain in audit tables.

All relevant means, posterior references, target probabilities and radii use one declared law. The cached-target law is qualified as `D_K`; it is not claimed to identify the checkpoint's full training distribution. Exact duplicate target atoms receive equal distinct-atom mass by default. The single-target conditional reference is an assumed idealization; repeated exact prompts with different targets are audited and retained. Legacy `--center` options never redefine this law's exact mean.

## Devices and reuse

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --device auto --recompute-experiments --num-loss-seeds 64 --loss-seed 0
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --model sdv1 --scheduler ddim --device auto --recompute-experiments --loss-timesteps saved --num-loss-seeds 128 --num-unconditional-loss-seeds 256 --loss-seed 0
./run_all.sh --model sdv1 --scheduler ddim --plot
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh --device auto --recompute-experiments
PYTHON=/path/to/python ./run_all.sh --plot
```

`auto` uses all visible CUDA devices with CPU fallback. Workers follow the shared device/sharding conventions and send progress to one parent-owned bar per stage. Fast vector cores and four-stage response/motion shards finish before expensive integration. Inference and analytical pools never compete for full model replicas. The direct stage reads each unfinished protected record once for its vector comparisons and preserves independently resumable Proposition 5 integrals. Missing evidence supplements read the preserved record separately for initial/final vectors, without repeating those histories or integrals.

Numerical policy `fixed-cache-numerics-2` first batches a CPU/CUDA outward-rounded squared comparison proving `D<ec` when its bounds separate. This resolves a negative original condition without square roots, posterior work, or Decimal conversion; it does not certify the separate float64 gain assessment. Only unresolved conditions or numerically ambiguous gains enter bounded CPU Decimal fallback, after budget preflight. Raw H underflow with resolved G and the old absolute-logit subtraction mismatch remain diagnostics rather than automatic fallback triggers. Missing V stays missing even when the condition sign is resolved. Fixed-input arithmetic, source sensitivity and unidentified training-law discrepancy retain separate statuses.

Numerical batch receipts are checked before loading raw witnesses. Completed batches remain reusable inside an unfinished record; raw tensors are loaded lazily for pending work only. The new policy hash does not invalidate compatible sufficient inputs or upstream observations. No new flags or default-budget changes are introduced, and a zero Decimal budget still permits the batched condition screen.

Probe tasks and analytical recipes have independent cache identities. An integrator change or missing integral does not repeat learned probes. Vector cores and integration payloads have separate completion markers. Missing or changed quadrature reads saved sufficient statistics; within the direct stage, only incomplete vector cores require raw-record reloads. Successful records remain reusable. Changing loss draw count gives new probe task identities; the deterministic RNG prefix is preserved, but partial prefix reuse is not implemented. Rendering styles do not invalidate numerical task caches.

Derived recomputation preserves omitted saved loss-draw settings. Plotting inherits omitted scientific settings from the saved bundle, including loss draw counts, timestep mode and reference law. Explicit conflicting settings fail with a recompute command. Repository model/scheduler/g/T/N identities remain checked. A portable bundle needs only its compact scalar tables and manifests:

```bash
./theory_validation.sh --bundle /path/to/experiment_S0_N20 --plot
```

## Paper outputs

Canonical publication remains `outputs/<model>_<scheduler>_g<G>_T<T>_N<N>/theory/experiment_S0_N<N>/`:

```text
run_config.json, summary.json, audit.json
registry.json, figure_manifest.json, figure_captions.md
initial.csv, terminal.csv, failed.csv, logical_tables.json
plot_data/*.csv, audit_data/*.csv
main/*.png, main/*.pdf
appendix/*.png, appendix/*.pdf
diagnostics/                 # Available optional behavioral figures
```

| Experiment | Main figure stem |
|---|---|
| Initial conditional recovery and unconditional baseline | `initial_loss_recovery` |
| Retained branch-gap contribution and matched evidence, fixed first update | `branch_gap_posterior_response` |
| Branch-gap dynamics and joint target accuracy | `branch_gap_synchronization` |
| Actual final latent error versus corrected reference bound | `final_reproduction_bound` |

The eleven appendix figures cover mean concentration/reference convergence,
all-step feedback and original condition margins, branch errors and the full
synchronization bound, individual peak indices, each group's additive branch
motion, the observable terminal bound, and common-population tolerance coverage.
Corollary 3 and Lemma 4 remain mathematical/vector audits rather than extra main
figures. Optional diagnostic figures are available with `--diagnostics`.

The baseline report is saved as `initial_baseline_summary.csv` and `.json`.
The posterior response retains the fixed grid `j/40` plus `1/g`, signed H,
negative responses and fixed prompt-balanced SSCD groups. Synchronization uses
chronological prediction indices (0 is initialization), paired D and Q, and no
fabricated prediction at the final output. Temporal shape is measured, not assumed.

The terminal primary directly compares actual endpoint error with the reference
bound, with the observable bound and tolerance coverage in the appendix. Non-clean
samplers require the independently qualified finite-step correction; supported
Gaussian corrections retain their simultaneous run probability contract.

The optional learned supplement is a separate, explicit inference task:

```bash
./run_all.sh --model sdv1 --scheduler ddim --recompute-experiments --device auto --counterfactual-unconditional
```

It uses matched next inputs at a fixed snapshot, not a new rollout or decode.
Ordinary `--plot` renders it when saved and never backfills inference.

Existing learned tasks and direct analytical/integration caches retain their
identities. New bounded analytical tasks use
`theory_measurements/reference_analytical/<hash>/`; additive geometry/terminal
scalars use `theory_measurements/evidence_supplements/<hash>/`. Missing supplements
read only preserved vectors and existing scalars; they do not repeat compatible
model probes or original variation integration. No active stage creates `theory_v2`.
Numerical sufficient inputs and policy retries are separately resumable under
`theory_measurements/numerical_refinement/`; old classifications remain preserved.
Paper bundle schema 5 / compact metric schema `four-stage-evidence-1` requires fresh
compact evidence tables, while old scientific observations remain preserved.

After the device record bar, theory reports the active aggregation, plot-input, table-saving, and publication stages with elapsed-time updates. Figure preparation has its own progress bar; PNG/PDF export advances once per saved file and shows the current filename.

Role locking, staged publication and paired PNG/PDF export remain in place. Only owned obsolete figure files are retired after successful publication. Plot mode reads compact tables and small metadata, without accessing backing numerical shards. Axes retain bold vector notation, single-line division, named reference lines and compact `SSCD` colorbars (`Mean terminal SSCD` for pair means).

## Author validation

This fixed evidence redesign was prepared under an **edit-code-only** instruction. No new measurements, tests, plots, GPU runs or data-dependent checks were executed. Historical reports document earlier implementations and do not validate this change.

Commands for the author, after review:

```bash
python -m pytest -q tests/test_theory_*.py tests/test_generation.py
```

See [the active four-stage plan and manuscript mapping](docs/theory/four_stage_experiments.md), [the numerical refinement plan](docs/theory/numerical_refinement_plan.md), [the fixed evidence plan](docs/theory/final_evidence_plan.md), [required manuscript alignment](docs/theory/submission_alignment.md), [the earlier implementation report](docs/theory/direct_implementation.md), [the direct pipeline guide](docs/theory/paper_pipeline.md), [the quantity dictionary](docs/theory/quantity_dictionary.md), and [the figure registry](utils/experiments/theory/paper_registry.py). The manuscript mapping is recorded in [statement_registry.json](docs/theory/statement_registry.json).
