# Numerical QA investigation

The four complete GPU bundles cover **837,000 trajectory rows, 16,740 evaluation samples, and 820,260 eligible feedback transitions**. The saved-scalar audit found no hard structural or measurement-contract failures. All **56 producer-flagged integral QA discrepancies remain numerically unresolved**. Computational completion, numerical estimates, and formal guarantees are distinct: this audit provides no formal integration certificate and changes no producer tolerance, flag, or result.

The primary evidence below uses the exact four GPU bundles. Earlier CPU audits are supplemental and are listed separately.

| Primary GPU bundle | Prompts | Evaluation samples | Trajectory rows | Eligible feedback | Integral QA flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDv1/DDIM, `81085d1787c9…` | 248 | 4,960 | 248,000 | 243,040 | 8 |
| SDv1/DDPM, `2fe4344cda4b…` | 260 | 5,200 | 260,000 | 254,800 | 17 |
| SDv2/DDIM, `59fd66475728…` | 86 | 1,720 | 86,000 | 84,280 | 12 |
| RealVis/DDIM, `4749db8a1e49…` | 243 | 4,860 | 243,000 | 238,140 | 19 |

The [complete scalar audit](validation/completed_scalar_qa.json) retains every raw finding, sample identity, saved component value, and reconstruction residual. The [read-only audit script](validation/final_scalar_qa.py) requires complete version 3 manifests, verifies scalar-file hashes, and checks **71 contracts per configuration**. It reads no tensors and performs no model inference. Checks include complete prompt/seed/step grids, saved source and destination indices, destination noise levels, SSCD attachment, candidate weights, measurement identities, stable gain signs, integration budgets, and proof-bound slack.

All computational stages completed with zero current failed records. No proof-bound slack fell below its recorded uncertainty allowance. No eligible query exhausted the quadrature evaluation budget. A numerical QA flag is retained separately from a failed computational stage.

| Primary GPU bundle | Estimated not met | Numerically unresolved | Positive gain | Negative gain | Saturated | Underflow with sign retained | Negative gain with SSCD > 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SDv1/DDIM | 230,711 | 12,329 | 123,486 | 119,554 | 84,505 | 64,763 | 30,796 |
| SDv1/DDPM | 243,289 | 11,511 | 138,094 | 116,706 | 87,696 | 67,133 | 7,434 |
| SDv2/DDIM | 83,719 | 561 | 30,848 | 53,432 | 7,045 | 3,192 | 1,414 |
| RealVis/DDIM | 228,718 | 9,422 | 120,707 | 117,433 | 87,380 | 64,347 | 30,479 |

Across all four configurations, the sufficient condition has **0 / 820,260 estimated-positive transitions**. Its conditional conclusion therefore has zero empirical coverage here. There are 786,437 estimated-not-met and 33,823 unresolved conditions. All 413,135 positive and 407,125 negative gain signs remain in the data. Among 199,435 gains that underflow to zero, stable log-odds ordering preserves 123,035 positive and 76,400 negative signs.

## Investigation of the preserved integral QA discrepancies

The diagnostic investigations reconstruct saved trajectories on CPU using protected cached states and predictions, the recorded scheduler, and the candidate bank. They perform no model inference. CPU and CUDA reductions need not agree bitwise, so both the saved-versus-CPU endpoint difference and the CPU endpoint-versus-affine difference are recorded. The first independent SciPy QUADPACK check uses a different quadrature rule but reuses the producer’s partition points; this shared sampling choice matters for the missed-tail case below. All investigated rows retain negative condition margins, positive proof-bound slack, and unresolved producer classifications.

| GPU configuration and full per-row evidence | Small endpoint/reduction discrepancies | Larger quadrature discrepancies | Maximum small absolute residual | Maximum small affine-identity residual under independent integration |
| --- | ---: | ---: | ---: | ---: |
| [SDv1/DDIM](validation/sdv1_gpu_numerical_qa.json) | 5 | 3 | 2.06228e-11 | 3.33067e-16 |
| [SDv1/DDPM](validation/sdv1_ddpm_gpu_numerical_qa.json) | 8 | 9 | 2.90947e-09 | 9.09495e-13 |
| [SDv2/DDIM](validation/sdv2_gpu_numerical_qa.json) | 11 | 1 | 3.07487e-11 | 1.77636e-15 |
| [RealVis/DDIM](validation/realvis_gpu_numerical_qa.json) | 11 | 8 | 8.2192e-12 | 1.77636e-15 |

The small discrepancies have the scale of floating-point endpoint reconstruction and backend reduction differences. They exceed the saved endpoint allowance in these rows; their unresolved classification is preserved. The larger discrepancies expose embedded integration-error estimates that are too small. Tighter numerical checks support the independently evaluated endpoint gains, subject to the sampling and roundoff limitations recorded below.

| GPU configuration / record / seed / step | Absolute saved integral residual | Saved combined allowance | Absolute first independent integral minus saved endpoint | Independent QUADPACK warning |
| --- | ---: | ---: | ---: | --- |
| SDv1/DDIM / `136565496` / 10 / 48 | 0.000339912138 | 0.00031922785 | 4.50745574e-09 | None reported |
| SDv1/DDIM / `283006352` / 1 / 48 | 0.000339944056 | 0.000287910597 | 3.0559022e-10 | None reported |
| SDv1/DDIM / `756633928` / 16 / 48 | 0.000309369004 | 0.000276501534 | 2.77850631e-10 | None reported |
| SDv1/DDPM / `1291771357` / 16 / 48 | 0.000277369665 | 0.000277111134 | 2.22621566e-09 | None reported |
| SDv1/DDPM / `1485909131` / 17 / 48 | 0.000309128881 | 0.000305291554 | 5.30371835e-09 | None reported |
| SDv1/DDPM / `1486909380` / 18 / 48 | 0.000321842535 | 6.11659345e-05 | 1.48838808e-09 | None reported |
| SDv1/DDPM / `1620480710` / 2 / 48 | 0.000335402739 | 6.24709193e-06 | 8.02174327e-10 | None reported |
| SDv1/DDPM / `1683590374` / 0 / 48 | 0.000340120776 | 0.00031768305 | 0.000335409222 | None reported; missed tail, see below |
| SDv1/DDPM / `2026642560` / 1 / 46 | 0.000102657984 | 3.76376172e-05 | 1.69904979e-10 | None reported |
| SDv1/DDPM / `328282297` / 17 / 47 | 0.000304655919 | 0.00023560336 | 1.74622983e-10 | None reported |
| SDv1/DDPM / `487633233` / 7 / 47 | 0.000294380543 | 4.10816776e-05 | 2.31239028e-10 | None reported |
| SDv1/DDPM / `985781054` / 11 / 47 | 0.000259153986 | 0.000155313991 | 4.61568561e-11 | None reported |
| SDv2/DDIM / `776379145` / 19 / 48 | 0.000331677918 | 0.000330642802 | 1.56433089e-09 | Roundoff limit |
| RealVis/DDIM / `1605558551` / 15 / 48 | 0.000326046262 | 0.000265864123 | 0 | None reported |
| RealVis/DDIM / `1704891531` / 12 / 48 | 0.000273304526 | 0.000245493566 | 4.43105819e-09 | Roundoff limit |
| RealVis/DDIM / `1738566173` / 3 / 48 | 0.000338350569 | 0.000271646498 | 7.29187377e-10 | None reported |
| RealVis/DDIM / `2097629300` / 5 / 48 | 0.000299794818 | 0.000163750148 | 1.22126949e-08 | Roundoff limit |
| RealVis/DDIM / `261115606` / 10 / 48 | 0.000306272401 | 0.000305465267 | 2.94051006e-09 | None reported |
| RealVis/DDIM / `283006352` / 18 / 48 | 0.000339322261 | 0.000274603947 | 2.74849299e-09 | None reported |
| RealVis/DDIM / `909615010` / 8 / 47 | 0.00030565211 | 1.94555491e-05 | 2.33043806e-10 | None reported |
| RealVis/DDIM / `972717315` / 9 / 48 | 0.000320181069 | 0.000317527256 | 2.88218871e-09 | None reported |

A QUADPACK roundoff warning means its requested tolerance was not achieved; it must not be treated as a convergence certificate. Exact values, returned warnings, source/destination timesteps, support membership, positive normalized weights, condition margins, and proof-bound slack are retained in the per-row JSON files.

## Confirmed missed-tail mechanism

For SDv1/DDPM record `1683590374`, seed 0, step 48, even the tighter first QUADPACK check misses a transition tail and reports a small error without a warning. The omitted integral is approximately `0.000335406428`, matching `log(1 + exp(-8)) = 0.000335406373`. The original partition includes points at eight transition widths; a sufficiently long adjacent interval can miss the remaining narrow tail.

The [additional diagnostic](validation/sdv1_ddpm_tail_investigation.json) widens the sampling partition without changing production code or saved outputs:

| Diagnostic partition extent | Integral minus affine endpoint | Integral minus saved GPU endpoint |
| --- | ---: | ---: |
| 8 transition widths | -0.000335406428349 | -0.000335409222316 |
| 16 transition widths | -1.12344423542e-07 | -1.15138391266e-07 |
| 32 transition widths | 2.54658516496e-11 | -2.7685018722e-09 |
| 64 transition widths | 2.54658516496e-11 | -2.7685018722e-09 |

This check explains why tightening tolerances alone is insufficient and why the endpoint-versus-integral QA is necessary. The wider-partition result supports the endpoint measurement; it does not retroactively certify the stored integral or change the unresolved status.

## Independently reconstructed condition components

Equivalent high-dimensional reductions take different arithmetic paths: clean-branch differences versus noise-prediction identities, and direct weighted means versus target-relative means. A strict machine-roundoff comparison identifies the following differences when reconstructing the condition margin from separately saved components:

| GPU configuration | Strict roundoff findings | Largest raw L2 difference | Maximum difference / existing saved source sensitivity |
| --- | ---: | ---: | ---: |
| SDv1/DDIM | 799 | 5.59623458685e-10 | 1.09487504613e-10 |
| SDv1/DDPM | 814 | 7.77049535827e-10 | 1.31061522219e-10 |
| SDv2/DDIM | 374 | 1.74577508005e-09 | 4.46605277654e-10 |
| RealVis/DDIM | 758 | 3.2684965845e-10 | 6.36468098585e-11 |

Every such difference lies within the source-sensitivity allowance already saved by the producer. These raw findings remain separate from hard contract failures. The audit introduces no replacement producer tolerance and makes no formal rounding-error guarantee.

## Actual multi-GPU execution and regression evidence

The [execution evidence for all four GPU bundles](validation/all_gpu_execution.json) records three workers on `cuda:0`, `cuda:1`, and `cuda:2`, all assigned records completed, and zero computational failures. All four manifests record the corrected feedback source SHA256 `9cc530d31455370646fbbf1be8459c3940fe52033dea78ac00446aebded77fde`.

The initial GPU attempt exposed an unsupported Boolean `scatter_reduce_(reduce="amax")`. The accumulator now uses integer 0/1 values, preserving Boolean OR semantics across chunks. A regression explicitly rejects Boolean scatter and checks that an invalid node remains invalid after later chunks. The feedback test file reports 23 passed and one CUDA-only test skipped on the CPU audit host; actual execution on three GPUs is established by the complete production bundles.

The [CPU parity report](validation/cuda_scatter_cpu_parity.json) compares the old and corrected accumulator implementations across eight synthetic cases and six real SDv2 transitions. All 40 output fields agree bitwise in every case. The report records both source hashes. Completed CPU measurements retain their original source hash.

## Supplemental CPU evidence

The earlier CPU reports are retained separately: [SDv2 scalar audit](validation/sdv2_cpu_scalar_qa.json), [SDv2 per-row investigation](validation/sdv2_numerical_qa.json), [RealVis scalar audit](validation/realvis_cpu_scalar_qa.json), [RealVis per-row investigation](validation/realvis_cpu_numerical_qa.json), and [SDv1/DDPM scalar audit](validation/sdv1_ddpm_cpu_scalar_qa.json). They use the recorded pre-fix source snapshot and are not included in the primary GPU counts. The CPU SDv2 and RealVis reports preserve 10 and 16 integral QA flags, respectively; the supplemental DDPM scalar audit preserves 14 flags and is superseded by the 17-row primary GPU investigation.

## Reproducing the primary scalar audit

Run from the repository root. Pinning all four hashes avoids ambiguity if an index later points to a different completed bundle.

```bash
python docs/theory/validation/final_scalar_qa.py \
  --analysis sdv1_ddim_g7.5_T50_N20=81085d1787c9e3f4f09b4f879ff3150cb94b3468ef01ae8b8b48e3104106bd83 \
  --analysis sdv1_ddpm_g7.5_T50_N20=2fe4344cda4ba068436d6361d2c8413c265e9c9ba6efb791a719044d7f45e467 \
  --analysis sdv2_ddim_g7.5_T50_N20=59fd66475728e525fa8f189e3ac1ca8a865309214114aa16a046e87ea63df7f7 \
  --analysis realvis_ddim_g7.5_T50_N20=4749db8a1e496cc4419220452e93c90a45aa540d5a7e464a0991843b216411c9 \
  --output /tmp/theory-v3-primary-gpu-audit.json
```

The independent per-row checker is [investigate_numerical_qa.py](validation/investigate_numerical_qa.py). It takes `--audit`, `--config`, and `--output`; unlike the scalar-only audit, it reads the cached tensors for flagged records. The separate [tail diagnostic](validation/investigate_ddpm_tail.py) records the wider-partition check. Neither script writes to generation caches or analysis bundles.
