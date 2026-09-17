# CUDA theory computation

The active numerical policy is `fixed-cache-numerics-cuda-8`, with four-stage measurement implementation `four-stage-measurements-cuda-2`. Theory analysis requires NVIDIA CUDA. `--device auto` uses all visible CUDA devices with the existing spawned workers and parent-owned progress bars; `cuda:N` chooses one device. CPU/MPS requests and unavailable CUDA fail explicitly. Plot-only and compact-bundle validation do not initialize CUDA.

## Computation and host responsibilities

Learned predictions and their error/loss reductions, finite-law posterior calculations, raw-vector reconstruction for numerical checks, condition screening, interval gains, variation enclosures, and adaptive quadrature value/error reductions stay on their worker GPU. Production measurement modules do not call Python Decimal. The independent Decimal implementation remains only as a test/reference oracle.

Python still runs on the CPU. File validation, serialization, scheduling, scalar control decisions, scalar-table aggregation and figure rendering remain host tasks. Input random draws retain the original seeded CPU generators so that this implementation change does not silently change experimental probes; draws are transferred to CUDA before numerical evaluation. Tensor-to-host transfers used for file hashes, cache payloads and final scalar records are not CPU numerical fallbacks. These distinctions matter: GPU computation cannot mean literal zero CPU utilization.

The active variation is mathcal{V}=[integral signed unit-gap projection of the next-minus-current reference]_+, with positive part applied after integration. Projection and signed quadrature stay on the GPU. Zero gap has an explicit value-zero convention without division. The projected-error recipe/payload version 5 requires signed error values reconstructed from the saved actual gap and reference vectors; compatible independent probes remain reusable. Normal analysis uses point estimates by default (`--numerical-max-products 0`); interval certification is optional via `--refine-numerics` or an explicit positive budget.

## Numerical guarantees

`gpu_intervals.py` uses eager CUDA binary64 arithmetic with outward `nextafter` rounding. It sums through explicit pairwise outward additions. Exponentials use a range-reduced Taylor enclosure with an analytic remainder; logarithms use an atanh-series enclosure after exact binary decomposition. It does not assume correctly rounded vendor exp/log or use unchecked matrix products for interval certificates. Keep this backend eager; a compiler that fuses operations across outward rounding would require a new proof and policy version.

The effective precision is 53 binary significand bits, not the old 64/128 Decimal digits. Intervals enclosing zero remain unresolved. Enlarging the work/node budget can tighten variation enclosures but cannot guarantee resolution or create additional precision. Original saved-tensor certificates and reduced stored-logit certificates retain different scopes. Neither certifies source perturbations or identifies the model's full training law.

The `numerical_decimal_precision` saved field is retained solely for reading historical configurations. Explicit `--numerical-decimal-precision` overrides are rejected during computation. New audits record `numerical_backend`, effective precision, GPU operations, and no CPU fallback; legacy Decimal execution counters remain zero.

## Commands and budgets

Migrate/rebuild derived measurements after this backend change:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_all.sh --model sdv1 --scheduler ddim \
  --recompute-experiments --device auto
```

This skips generation, SSCD and proximity. Source changes deliberately invalidate affected derived task identities; compatible tasks remain reusable and prior caches remain intact. A changed GPU reduction can differ in last bits, so no unverified bitwise compatibility alias is introduced for older CPU reductions.

Once compatible evidence and learned probes exist, increase numerical work without learned inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_all.sh --model sdv1 --scheduler ddim \
  --refine-numerics --device auto \
  --numerical-max-products 200000000 \
  --numerical-max-variation-nodes 257 \
  --numerical-variation-absolute-width 1e-8
```

The operation limit defaults to 2,000,000 per row (per seed across the complete dose grid for response refinements); nodes default to 65 and width to `1e-6` in raw latent L2 units. The 200-million-operation example is a larger trial budget, not a guarantee. `--numerical-max-decimal-products` remains an alias for `--numerical-max-products`. Primitive lane counts include tensor broadcast work and the composed series operations; this new budget is not a wall-time limit or directly comparable with old Decimal timing. A zero limit keeps GPU screening/ordinary assessments but skips interval refinement.

Changing the response policy may rebuild analytical four-stage shards in refinement mode. Those workers read saved observations and never load a denoiser. Missing learned/core prerequisites still request `--recompute-experiments` rather than silently invoking inference.

Inspect `audit_data/proposition5_numerical_resolution_rows.csv` for GPU operation/node counts and stopping reasons. `gpu_operation_budget_exhausted` or its preflight variant calls for more operations; node exhaustion calls for more nodes; a reached variation width may need a smaller requested width. Binary64 arithmetic limitations can persist at any budget. Existing unresolved measurements are never discarded to improve figure appearance.

## Validation status

CUDA regression specifications cover enclosure containment against independent high-precision references, cancellation, subnormals, overflow, budget exhaustion, CPU rejection and scientific status preservation. This source-only revision did not execute tests, scientific imports, plotting, inference, benchmarks or migration. Only source/JSON parsing, shell syntax and whitespace checks were performed. Existing outputs were not modified.
