# How Diffusion Models Memorize

This project recovers the Webster prompt–image benchmark, caches complete
diffusion trajectories, scores generated endpoints with SSCD, and measures
terminal-latent proximity. It also evaluates the initial conditional recovery
of Theorem 1 and the zero-centered forms of Lemma 2 and Corollary 3. Passing
`--use-mu` centers both latter experiments on a reusable model-implied mean.
The scientific cache is deliberately reusable: proximity and plotting never
rerun diffusion.

## Install

Runtime dependencies and test-only dependencies are kept separate:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Figure exports

All Matplotlib figure writers use the shared `FIGURE_DPI = 150` in
`utils/experiments/plotting.py`, including PNG output and embedded raster
content in PDFs. This is a moderate, established export resolution (also the
[MATLAB export default](https://www.mathworks.com/help/matlab/ref/exportgraphics.html)),
not a power-of-two pixel-size convention. PDF text and curves remain vector
graphics and retain their zoom quality. Lower DPI is not guaranteed to be
indistinguishable at high zoom or in high-resolution printing.

Single-panel figures stay 4×4 inches; the 3×10 decoded galleries stay 12×4
inches. STIX fonts, label/tick/legend sizes, `bbox_inches="tight"`, and
`pad_inches=0.05` are unchanged. At the same physical dimensions, 150 DPI
uses one-quarter as many raster pixels as 300 DPI; compressed file-size
savings depend on content. Existing figures adopt this policy when replotted.
The PIL generation montages and proximity example copies already use explicit
pixel dimensions/downscaling; changing their DPI metadata would not reduce
their file sizes, so those scientific/preview image paths are unchanged.

## Workflow

Run the complete experiment matrix from any directory with one command:

```bash
./run_all.sh
```

The default covers 12 configurations:

| Axis | Default values | Restrict the run |
| --- | --- | --- |
| Model | `sdv1`, `sdv2`, `realvis` | `--model MODEL` |
| Scheduler | `ddim`, `ddpm` | `--scheduler NAME` |
| Center | zero, saved model-implied mean | `--no-mu` or `--use-mu` |
| Selection | `gmm` only | `--selection-strategy gmm` |

Each explicit model or scheduler option narrows only its own axis; `all` is
accepted for those two axes. Selection is fixed to GMM. `--use-mu` and
`--no-mu` are mutually exclusive. For example, a single zero-centered
configuration is:

```bash
./run_all.sh --model sdv1 --scheduler ddim --selection-strategy gmm --no-mu
```

Models run sequentially in the listed order, DDIM before DDPM. Within each
model–scheduler pair, the reference and experiment generation/SSCD stages each
run once; their caches are shared across centers. A required baseline also runs
once per pair. The runner rebuilds GMM reference selection and experiment
proximity, runs Theorem 1 once, and runs Lemma 2 and Corollary 3 for each
requested center.
Theorem 1 has no centering dependency and is not redundantly run twice.
The forward-corruptions/generated-states experiment also runs once, after
Theorem 1 and before the center-specific stages, when the pair is SDv1/DDIM
with `--g 7.5`, `--N 20`, and `--T >= 9` (default 50). It reuses evaluation
seeds 0–19 and independent reference seeds 20–39. Other configurations receive
a numbered skip. It includes the two distance plots and the two decoded-state
galleries described in [the experiment guide](FORWARD_CORRUPTIONS_GENERATED_STATES.md).
Defaults give 13 numbered stages per pair, or 78 stages across six pairs,
including explicit skips.

For mean-centered runs, `B = --num-baseline-seeds` Gaussian initial latents
(default `1000`) with seeds `N` through `N+B-1` are evaluated by the empty branch
at the actual initial cached DDIM or DDPM timestep. Their saved mean is
$\widehat{\boldsymbol{\mu}}$. This baseline never traverses prompt records or
reads SSCD, categories, memorization labels, frozen selections, or cached
prompt predictions. It has a separate `S<N>_N<B>` namespace inside the reference
generation run. Zero-only runs skip this stage. Direct experiment wrappers
still default to zero centering and GMM selection; `run_all.sh` runs both
centers and all three models and two schedulers, with GMM selection by default.

All three theory experiments evaluate every timestep in the saved `T`-step
schedule. `--evaluation-source {gaussian,trajectory,both}` selects the input
distribution; the default is `both`:

- `gaussian` evaluates the same $\mathbf{z}_s\sim\mathcal{N}(\mathbf{0},\mathbf{I})$
  at each timestep, independently of the target. This is the input distribution
  in Theorem 1, Lemma 2, and Corollary 3. Model inference is required.
- `trajectory` reads every selected prompt's cached $\mathbf{x}_{t}^{(i,s)}$
  and its matching cached epsilon predictions. After the initial step these
  states are no longer independent Gaussian initializations, so this is a
  separate diagnostic of behavior during denoising.
- `both` saves both evaluations in one source-tagged CSV and separate figures.

The same `N` evaluation seeds `0..N-1` are paired across all timesteps. Lemma 2
Gaussian inference is prompt-independent: it evaluates `N * T` observations
once, without repeating identical draws for each selected prompt. Its trajectory
evaluation retains all `P * N * T` selected prompt–seed–timestep observations.
Corollary 3 evaluates all selected pairs at all timesteps in each source and
supports both schedulers with `g = 7.5`. Theorem 1 retains independent
forward-corruption loss measurements at every timestep in either source mode.

The Lemma 2 and Corollary 3 timestep sweeps use a logarithmic
$\alpha_t^2/\sigma_t^2$ x-axis with lower values on the left. Theorem 1 uses
normalized forward-loss RMSE on x: its primary figure compares initial-step
loss and independent-Gaussian recovery, while its sweep figures use the
corresponding timestep-specific loss. These are finite-noise diagnostics, not an
extrapolation to zero or proof of asymptotic convergence. Theorem 1 requires
its normalized-loss premise to vanish in the high-noise limit, not merely a
decreasing noise ratio. `--use-mu` uses one fixed finite-timestep model-implied baseline;
this is a proxy, not a measured latent-data mean or a separately fitted mean
at each timestep.


Before either generation stage, `generate.sh` performs a fast published-cache
check. A matching complete summary, exact prompt/marker coverage, required
regular artifact files, and marker publication order skip generation without
rereading every large tensor. Any incomplete or uncertain cache falls back to
the full per-record hash validation and resumes only the affected records.
Before either SSCD stage, `sscd.sh` performs its own fast published-cache check.
A matching configuration and completed summary, exact score/marker coverage,
and safely published aggregate and per-prompt files skip SSCD before device,
checkpoint, VAE, or SSCD-model setup. Any incomplete or uncertain cache falls
back to per-record physical validation and resumes only the affected records.
Reference proximity atomically rebuilds GMM selection from the current
validated generation and SSCD caches. Numbered totals reflect the requested
centers. Zero-only runs explicitly skip the baseline;
nonstandard guidance retains a numbered Corollary 3 skip. DDPM no longer skips
any default theory stage. Rebuilding a derived selection cannot silently reuse
stale completion-marker provenance.

Pass `--download` to run or resume the shared Webster preparation once before
the first model. The reference and experiment cache stages and all downstream
analyses receive the same model, scheduler, guidance scale, step count, and
`N`; only the reference and experiment seed blocks differ. All prompt-dependent
measurements use the exact frozen selection. Lemma 2 retains one cached
observation per selected prompt, experiment seed, and timestep, and its Gaussian
evaluation needs no prompts. `--num-baseline-seeds` affects only
`--use-mu` runs and their mean-centered output namespaces.

A downloader exit status of `2` means some records remain retryable; in that
case the orchestrator continues with all currently available prompt–image
pairs.
Other data errors and failures in later stages stop the matrix immediately.

Pass `--overwrite` only to force regeneration of the shared scientific caches.
The orchestrator regenerates trajectories, SSCD, and any required shared
unconditional baseline once per model–scheduler pair, and forwards the flag to
Theorem 1 to recompute its selected-prompt records. On every normal run, with or
without that flag, `run_all.sh` atomically rebuilds the GMM frozen reference and
experiment-proximity outputs from validated caches. Thus a plain run repairs
stale derived selection provenance while complete generation, SSCD, and
Theorem 1 records are reused. A direct `sscd.sh --overwrite` affects only its
requested SSCD seed pool, while a direct Theorem 1 `--overwrite` affects only
its requested result cache. `--overwrite` cannot be combined with `--plot`.

Theorem 1 additionally averages its conditional-loss estimate over
`--num-loss-seeds` independent corruption draws per prompt–target pair and timestep
(default `20`).
`--loss-seed` (default `0`) is the deterministic root used to derive those
draws in a stream disjoint from generation seeds. Both options affect only the
Theorem 1 computation, so changing them does not change generation, SSCD, or
selection cache paths. For example:

```bash
./run_all.sh --model sdv1 --num-loss-seeds 32 --loss-seed 123
```

The frozen-selection hash is part of the Theorem 1 cache and CSV identity.
Derived results remain isolated under
`outputs/<experiment-run>/theorem1_loss_recovery/<selection-hash>/evaluation_<source>/`,
so rebuilding the frozen selection cannot silently reuse or overwrite results
from a different selected prompt set.

The optional shared model-implied baseline is

$\widehat{\boldsymbol{\mu}} = (1/B) \sum_b
(\mathbf{x}_{T,b} - \sigma_T \boldsymbol{\epsilon}_{\emptyset}
(\mathbf{x}_{T,b},T))/\alpha_T$.

It is estimated only for requested mean-centered runs, exactly once at the
actual initial cached scheduler timestep from `B` independent Gaussian initial
latents with seeds `N` through `N+B-1`. Zero-centered runs instead use
$\mathbf{0}$ and have no baseline-artifact dependency. For trajectory input,
Lemma 2 uses every frozen-selected prompt `i`, experiment seed `s`, and cached
DDIM or DDPM timestep `t`:

$\widehat{\mathbf{x}}_{0\mid t,\emptyset}^{(i,s)} =
(\mathbf{x}_{t}^{(i,s)} - \sigma_t
\boldsymbol{\epsilon}_{\emptyset}^{(i,s,t)})/\alpha_t$.

It logs `centered_distance_rmse` relative to $\mathbf{0}$ by default or to
$\widehat{\boldsymbol{\mu}}$ with `--use-mu`. Both $\mathbf{x}_t$ and the
unconditional epsilon prediction come from the validated experiment trajectory
cache, producing exactly `P * N * T` rows for `P` selected prompts. Lemma 2
uses no model inference in `--evaluation-source trajectory` mode. Gaussian mode
instead evaluates the empty branch on the fixed $\mathbf{z}_s$ pool at every
timestep and writes `N * T` rows. With `both`, the table contains
`(P + 1) * N * T` rows. The empirical percentile bands summarize individual
observations; they are not confidence intervals or a norm of the mean prediction.

`--device auto` uses every CUDA device visible to PyTorch. Set
`CUDA_VISIBLE_DEVICES` to restrict that pool, or pass `--device cpu`,
`--device mps`, `--device cuda`, or `--device cuda:N` to select exactly one
device. Model-inference GPU stages run one model replica per device. When
requested, the shared baseline shards Gaussian seeds across workers. Lemma 2
and Corollary 3 distribute their evaluations across requested devices. Cached
trajectory observations use tensor reductions; Gaussian probes use the same
model, scheduler, prediction-conversion, and seeding utilities as generation.
The three models remain sequential.
Cache-only proximity and plotting stay on the CPU.

The GPU stages use independent inference processes, not gradient-oriented
`torch.distributed` DDP: there are no gradients or model updates to synchronize.
With `--use-mu`, the shared baseline reports one `B`-sample unconditional-
inference progress bar. Evaluation progress includes every requested timestep
and source. Corollary 3's progress also counts the `N * T` unconditional
predictions computed once per active worker and reused for that worker's prompts;
these setup work units do not add CSV rows.
The full Gaussian sweep costs more inference than the former
initial-step experiment. Use `--evaluation-source trajectory` to run the cached
Lemma 2 and Corollary 3 diagnostics; Theorem 1 still estimates independent
forward-corruption losses in that mode.

Each worker limits its PyTorch CPU thread pool to its share of the available
CPUs. Generation hashes tensor files during the existing atomic write, and
downstream stages physically validate only the generation artifact they consume.
This preserves hash-checked resume behavior while avoiding redundant reads of
large trajectories and predictions on shared filesystems. Exact linear scaling
is still not guaranteed because model loading, compilation, preview decoding,
cache traffic, and the slowest shard remain part of wall-clock time.

For eligible CUDA workers, the project automatically checks for
`torch.compile` and the Inductor backend, then compiles the validated UNet once
per model replica. Unsupported environments and compilation failures fall back
to eager execution; CPU and MPS execution are unchanged.
CUDA Graph capture remains enabled for fixed shapes, but dynamically shaped
graphs skip capture to avoid recording a separate graph for every input size.
The first compiled
call pays compilation overhead, so short or cache-only runs may not become
faster. Long runs with repeated UNet calls are expected to benefit, but the
speedup depends on the workload, model, GPU, and software environment.

Pass `--plot` to `run_all.sh` to skip every computational stage and regenerate
the theory PNG/PDF figures plus the compatible forward-corruptions figures.
It follows the same 12-configuration matrix by default, with 36 numbered
plotting stages including explicit skips (Theorem 1 and forward corruptions
are independent of centering). `--use-mu` selects only mean-centered results;
`--no-mu` selects only zero-centered results. No generation, SSCD, proximity,
baseline, or VAE decoding is invoked. Every requested CSV and its matching
provenance must exist; forward-corruptions galleries additionally require their
saved decoded-image logs. Run that experiment once without `--plot` if those
logs are missing. Use the axis filters to plot a subset. Both schedulers are
supported by the theory experiments; forward corruptions has the compatibility
requirements above, and nonstandard guidance causes a Corollary 3 skip.

```bash
./theorem1_loss_recovery.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy gmm --num-loss-seeds 20 --loss-seed 0 --plot

# Zero-centered defaults.
./lemma2_mean_convergence.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy gmm --plot
./corollary3_cfg_amplification.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy gmm --plot

# Add both options to either theory wrapper to select its saved mean-centered run.
# --use-mu --num-baseline-seeds 1000
```

Zero-centered plot-only mode never loads or validates a baseline artifact. With
`--use-mu`, it validates the shared-baseline metadata and hash without
deserializing `mu_hat.pt`. Choose the same source for computation and plotting:

```bash
./run_all.sh --model sdv1 --evaluation-source both
./run_all.sh --model sdv1 --evaluation-source both --plot
# Cached diagnostics (Theorem 1 still estimates conditional losses):
./run_all.sh --model sdv1 --evaluation-source trajectory
```

Old initial-step or source-less CSVs must be recomputed. Existing generation
and SSCD caches are reused. New results use an `evaluation_<source>` leaf
directory, so previous results remain available. Plot-only mode performs no
inference or latent-tensor loading. It combines each CSV's recorded centering and
scientific fields with the requested generation-cache and frozen-selection
context; mode-specific output paths keep zero- and mean-centered results
separate.

`N` and `--num-baseline-seeds` must be positive. The baseline count matters only
with `--use-mu`; its seeds `N` through `N+B-1` are disjoint from experiment
trajectory seeds `0` through `N-1`. The prompt-selection reference still uses
`N` through `2N-1`.
`run_all.sh` rejects either seed block when its last seed would exceed the
supported random-seed domain. `--num-loss-seeds` must be positive, and
`--loss-seed` must
be between `0` and `2^63-1`, inclusive. All invoked wrappers use unbuffered
Python output, so their `tqdm` progress remains visible in redirected logs.

To prepare or retry Webster data independently:

```bash
./download_webster.sh
```

Choose one model and sampler configuration, then reuse it for the reference,
experiment, and theory stages. The example below includes the optional baseline
and passes `--use-mu` to both Lemma 2 and Corollary 3. Omit the baseline command,
`--use-mu`, and `--num-baseline-seeds` from those two commands for the
zero-centered default. Only `--seed-start` differs between generation roles:

```bash
MODEL=sdv1
SCHEDULER=ddim
GUIDANCE_SCALE=7.5
STEPS=50
N=20
NUM_BASELINE_SEEDS=1000
NUM_LOSS_SEEDS=20
LOSS_SEED=0
DOWNSCALE=4
DEVICE=auto
SELECTION_STRATEGY=gmm

# Independent selection reference: seeds N through 2N-1.
./generate.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start "$N" \
  --downscale "$DOWNSCALE" \
  --device "$DEVICE"

# Optional --use-mu baseline: B Gaussian seeds; no prompt traversal.
./unconditional_baseline.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --num-baseline-seeds "$NUM_BASELINE_SEEDS" \
  --device "$DEVICE"

./sscd.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start "$N" \
  --device "$DEVICE"

./compute_proximity.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start "$N" \
  --selection-strategy "$SELECTION_STRATEGY"

# Matching experiment: seeds 0 through N-1.
./generate.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start 0 \
  --downscale "$DOWNSCALE" \
  --device "$DEVICE"

./sscd.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start 0 \
  --device "$DEVICE"

./compute_proximity.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --seed-start 0 \
  --selection-strategy "$SELECTION_STRATEGY"

# Theorem 1 uses the selected prompts and the same experiment generation cache.
./theorem1_loss_recovery.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --selection-strategy "$SELECTION_STRATEGY" \
  --num-loss-seeds "$NUM_LOSS_SEEDS" \
  --loss-seed "$LOSS_SEED" \
  --device "$DEVICE"

# Lemma 2 reads every selected cached prompt trajectory.
./lemma2_mean_convergence.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --selection-strategy "$SELECTION_STRATEGY" \
  --use-mu \
  --num-baseline-seeds "$NUM_BASELINE_SEEDS" \
  --device "$DEVICE"

# Corollary 3 uses both cached branches, target latents, and target SSCD.
./corollary3_cfg_amplification.sh \
  --model "$MODEL" \
  --scheduler "$SCHEDULER" \
  --g "$GUIDANCE_SCALE" \
  --T "$STEPS" \
  --N "$N" \
  --selection-strategy "$SELECTION_STRATEGY" \
  --use-mu \
  --num-baseline-seeds "$NUM_BASELINE_SEEDS" \
  --device "$DEVICE"
```

For direct Theorem 1 runs, `--sample-batch-size` remains a performance-only
option (default `8`). It changes inference chunking, not the scientific sample
counts, and is intentionally not a `run_all.sh` option.

The reference proximity call creates the selected strategy's frozen selection
from all seeds `N` through `2N-1`. Run selected-prompt analyses only after it succeeds.
`run_all.sh` checks reference generation, optionally computes or validates the
shared baseline when mean centering is requested, checks reference SSCD, and then
atomically rebuilds the selection and its figures.

Supported models are `sdv1`, `sdv2`, and `realvis`; supported schedulers are
`ddim` and `ddpm`. The RealisticVision dataset directory remains named
`realisticvision`. `sdv2` uses the public
`Manojb/stable-diffusion-2-1-base` repository. At load time, its `main` branch
is resolved to an immutable commit and that revision is recorded in every
generation run configuration.

All theory experiments and the shared baseline support both schedulers.
Corollary 3 requires `--g 7.5`; other guidance values print an explicit skip.

### Data recovery

`download_webster.sh` organizes the three 500-record metadata views
and persists recovery state in
`data/webster/state/recovery.sqlite`. On every rerun it first
performs a fast structural check of saved image paths, headers, dimensions,
and formats. Valid recovered images are skipped without a network request; a
missing or visibly corrupt artifact alone is audited and reactivated. The
final validation phase still performs full dataset checks. Completed misses
stay completed, while only pending or retryable records proceed to network
recovery. An unresolved record is narrowly reopened when
either the versioned audited-mirror strategy or Arquivo.pt strategy is newer
than the one that produced its completed miss.

The ten logical phases cover local validation/cache reuse, official assets,
direct URLs, the audited ground-truth mirror, Wayback, Arquivo.pt, Common
Crawl, final resolution, model-view publishing, and validation. Their `tqdm`
bars show counts, rates,
elapsed time, ETA, and
`MV x/y | TV x/y | RV x/y | N x/y | Total x/y`; download logs do not
dump serialized summaries to the terminal. Detailed request audits remain in
`data/webster/logs/<record_id>.jsonl`.

Failed rows that identify the same exact target are grouped before direct and
archive work. One representative is queried, and a successful image or
archive candidate is fanned out through verified content-addressed references.
Conflicting or incomplete identities remain separate. A verified exact archive
candidate is resolved immediately, so later archive sources do not query a
target that is no longer failed.

Direct origin-URL recovery uses 24 worker threads, interleaves targets across
hosts, caps each host at four concurrent requests, and stops retrying permanent
DNS misses. `--direct-workers`, `--direct-attempts`, and
`--per-host-concurrency` tune only this direct phase.

After Direct, recovery range-reads a third-party ground-truth mirror from
`gdhanuka/memorization_data2` at the immutable revision
`327530503d2c6c16b87691e0fb57c08973a5fc47`. The four exact Parquet LFS objects,
500-row identity set, official-SD1 identity set, and their 352-row exact
`(index, raw prompt, URL)` intersection are pinned by hashes. Parquet projection
reads only `index`, `raw_prompt`, `url`, and embedded `ground_truth.bytes` for
row groups containing accepted identities; generated-image columns and the
embedded path field are never used. Completed projections are hash-checked and
cached beneath `data/webster/state/ground_truth_mirror/`, so reruns fetch only
missing or invalid shards.

The mirror is trusted only after at least 50 unique, independently recovered
exact targets are audited. Every available non-circular overlap is checked;
the collection must have at least 90% agreement at perceptual-hash distance
four or less. Mirror-derived rows, verified duplicates, and source-page guesses
cannot validate the mirror. Independently recovered outliers are reported in
`data/webster/state/ground_truth_mirror_audit.json` and are never overwritten.
Only failed exact identities are imported, and cross-model fan-out requires the
complete target identity.

Wayback and Arquivo.pt are sequential and rate-limited. Arquivo.pt performs
exact-URL version-history queries and samples four metadata pages across the
complete newest-to-oldest history, rather than inspecting only recent captures.
One target-wide budget allows at most 12 replay requests across all safe URL
variants. Replay identity and every recognized MD5 or SHA-1 digest are
verified; malformed reported digests are rejected, and supplied, requested,
and final replay URLs are recorded separately. Arquivo requests do not follow
redirects.

Common Crawl's public CDX lookup is also sequential and rate-limited. The
[Common Crawl FAQ](https://commoncrawl.org/faq) asks clients not to issue
multiple index-request threads from one IP, and the
[URL Index documentation](https://commoncrawl.org/url-index) recommends its
columnar index for bulk analytical jobs. The downloader therefore reuses one
collection catalog per run rather than multiplying public CDX requests.

For Arquivo.pt and Common Crawl, each uncached metadata call gets one attempt.
After three consecutive failures for the same service host, a cache-first
circuit breaker defers further uncached requests until the next run. Cached
responses remain usable and deferred records remain retryable. Their progress
bars distinguish actual network requests, cache hits, deferred unique targets,
deduplicated rows, and open circuits, so fast progress during an outage is not
misreported as completed archive work.

Tune the direct stage without discarding resumable state, for example:

```bash
./download_webster.sh \
  --direct-workers 32 \
  --direct-attempts 2 \
  --per-host-concurrency 4
```

The same options are accepted by `run_all.sh`, but affect it only when
`--download` is present.

### Generation cache

Generation processes every available prompt–target pair, including every TV
prompt. For each original index it stores:

- `latent/<index>.pt`: the complete trajectory `[N, T + 1, C, H, W]`;
- `noise_pred/<index>.pt`: unconditional and conditional predictions, both
  recorded before classifier-free guidance is applied;
- `target_latent/<index>.pt`: the deterministic paired-image VAE latent;
- `image/<index>.png`: a non-scientific preview montage; and
- `record/<index>.json`: the completion marker and artifact hashes.

After actual generation or a record-level repair, `generate.sh` validates and
scans only the small `target_latent` files, one tensor at a time, with a
dedicated `tqdm` bar. A completely published cache reuses this existing
report after hash-checking those small target-latent files. It does not load
them as tensors, read the large trajectories or noise predictions, or rerun
the VAE. The report is written to:

```text
logs/<base-run>/<role>_S<seed-start>_N<N>/target_latent_statistics/
├── report.json
├── mean.pt
└── population_std.pt
```

`report.json` gives the flattened element mean and population standard
deviation (`correction=0`), per-channel moments, and two quantities that map
directly to latent-standardization claims. `mean_squared_l2_per_dimension` is
exactly the empirical `E[||x_0||^2]/d`; `mean_vector_rms` is
`||E[x_0]||/sqrt(d)` and is zero exactly when the empirical coordinate-wise
mean is zero. The full coordinate-wise mean and population standard deviation
are saved as float64 `[C,H,W]` tensors. Every completed prompt--target pair has
one unit of weight; a target-image-deduplicated audit is also included.

The report's scope is the recovered, completed Webster rows for that model,
not the checkpoint's full training distribution. Webster preparation itself
is model-agnostic and loads no VAE, so the first scientifically valid time to
report model-specific image latents is immediately after generation creates
or resumes those cached target tensors.

Every generation and SSCD seed role lives beneath one shared base-run
directory. Experiment caches use `experiment_S0_N<N>`, their matching
selection reference cache uses `reference_S<N>_N<N>`, and any other nonzero
seed block uses `seed_S<seed-start>_N<N>`. GMM selection reuses the
same matching pair:

```text
logs/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/
├── experiment_S0_N<N>/
└── reference_S<N>_N<N>/
    └── unconditional_baseline/
        └── S<N>_N<B>/
            ├── mu_hat.pt
            └── metadata.json
```

`mu_hat.pt` is the reusable float64 `[C,H,W]` baseline tensor. Its companion
`metadata.json` records the model, actual initial timestep, `alpha_T`,
`sigma_T`, `num_baseline_seeds`, `baseline_seed_start`, the exact baseline seed
list, and $\|\widehat{\boldsymbol{\mu}}\|_2/\sqrt{d}$. It also pins the seed fingerprint, input
identity, per-seed estimate-population fingerprint, and both the file and value
hashes of the saved mean tensor, alongside model and schedule provenance.
Different values of `B` cannot overwrite each other because each uses its own
`S<N>_N<B>` directory. A normal baseline invocation validates and reuses these
files; only an explicit `--overwrite` recomputes and atomically replaces them.

A complete-run fast check requires the matching immutable configuration and
summary, exact dataset/marker coverage, every required artifact as a regular
file, and artifacts published no later than their atomic completion marker.
If that check is not conclusive, a record is resumed only when its marker,
configuration identity, physical file hashes, shapes, and dtypes validate.
Existing valid scientific tensors are never overwritten unless
`--overwrite` is explicit. Changing `--downscale` on resume regenerates only the
non-scientific preview montages and their preview provenance; it does not
regenerate trajectories, predictions, target latents, or other scientific
artifacts.

The sampler keeps one explicit full-trajectory loop. It supports DDIM and DDPM,
stores both unguided branches in canonical epsilon space, and then gives the
checkpoint-native guided prediction to the scheduler. Target encoding uses the
VAE posterior mode after fixed RGB/resize preprocessing.

### SSCD

SSCD scores every completed generation record, including TV prompts. It reads
only cached terminal latents, decodes them at full model resolution with the
generation-pinned VAE, and stores one cosine similarity per seed in
`sscd/<index>.pt`. It does not load or run the UNet, tokenizer, text encoder, or
sampler.

On a normal invocation, a complete published SSCD cache returns before device
resolution, checkpoint access, model/VAE construction, trajectory reads, or
physical score-tensor hashing. The fast check verifies the exact scientific
configuration, seed and record coverage, endpoint-relevant generation hashes,
regular nonempty score/marker files, and publication order. An uncertain or
partial cache falls back to physical per-record validation and computes only
missing or invalid scores. Preview-only generation changes therefore retain
SSCD, because preview provenance is audit metadata rather than a score input.
Pass `--overwrite` to `sscd.sh` only to recompute and atomically replace every
score in that requested seed pool.

### Frozen target-pair selection

Target-pair selection happens only during cache-based proximity analysis. The
frozen reference is specific to the selection strategy, model, scheduler,
guidance scale, step count, and `N`. Its generation and SSCD evidence remains
in the shared cache
`logs/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/reference_S<N>_N<N>`.
All N independent observations from seeds `N` through `2N-1` jointly determine
each whole-prompt decision.

The only selection strategy, `gmm`, pools every valid `(l2_norm, sscd)` reference
observation, including valid seeds from otherwise incomplete prompts. It
standardizes both coordinates over that entire population and fits one global,
two-component, diagonal-covariance Gaussian mixture. The diagonal model prevents
a tilted component covariance from extrapolating high-SSCD membership into
the distant low-SSCD cloud. Two deterministic initializations (the existing
first-principal-component k-means and SSCD-only k-means) compete by converged
log likelihood; only the higher-likelihood fit is used. Components are ordered
by fitted SSCD mean. Every valid observation receives exactly one hard
component assignment by maximum posterior probability (exact ties go to the
low component).
There is no trimming, confidence cutoff, or unassigned valid observation.

A complete prompt is included if any reference seed belongs to the high-SSCD
component. A high-component seed retains its prompt even when other seeds
occupy the bottom cloud. All seeds of an included prompt remain in the analysis.
Neither Spearman rho nor an additional SSCD threshold affects this decision. Missing
observations still have explicit error statuses and cannot enter the fit.

Selections and experiment logs must match the current scientific configuration
and schema. Normal `run_all.sh` builds the requested reference selections before
running the experiments.

Missing, duplicate, or non-finite reference observations make a prompt
unusable, and unusable prompts are never silently included. Prompt kind is
retained only for audit; MV, RV, TV, N, and every other kind are treated
identically. Spearman correlation remains a descriptive statistic for the
figures and logs; it never affects selection.

Experimental seeds are an independent pool starting at 0 and never affect
selection. An experiment uses seeds 0 through `N-1` and applies the
already-frozen decisions for its exact strategy, model, and sampler
configuration.
Generation trajectories, noise predictions, target latents, preview montages,
and SSCD tensors remain intact for discarded and unusable prompts; selection
filters only the analysis.

The frozen reference directory contains the three authoritative artifacts,
four derived figures, and the cache-only representative examples:

```text
data/webster/selection/<dataset>/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/reference_S<N>_N<N>/
├── selection.csv
├── config.json
├── summary.json
├── proximity_vs_sscd.png
├── proximity_vs_sscd.pdf
├── proximity_vs_sscd_all_prompts.png
├── proximity_vs_sscd_all_prompts.pdf
└── examples/
    ├── manifest.json
    ├── retained/
    │   ├── {highest,median,lowest}_l2_generated.png
    │   └── {highest,median,lowest}_l2_training.png
    └── discarded/
        ├── {highest,median,lowest}_l2_generated.png
        └── {highest,median,lowest}_l2_training.png
```

`selection.csv` has exactly N seed rows per prompt. It records `prompt`,
`kind`, `l2_norm`, `sscd`, the strategy, per-seed observation status, repeated
prompt-level decision, descriptive `prompt_spearman`, reference provenance,
and each valid seed's fitted GMM component and low-mode posterior. Any
high-component seed retains a complete prompt.
`generated_image_path` points to the existing seed-ascending preview montage;
`generated_image_tile_index` identifies its zero-based row-major tile, so
indices `0` through `N-1` map directly to seeds `N` through `2N-1`. Those cache
paths remain the evidence source; only the chosen files under `examples/` are
derived copies. `config.json` records the model,
sampler values, strategy and frozen policy, seed set, provenance hashes, and
the standardized global GMM fit, while `summary.json` records concise decision
and observation counts. `proximity_vs_sscd.{png,pdf}` is the
selected-prompt view. `proximity_vs_sscd_all_prompts.{png,pdf}` is the
pre-discard view containing every prompt with a complete finite N-seed
reference group, including both selected and discarded prompts. Incomplete or
otherwise unplottable prompts remain in `selection.csv` for audit but cannot
appear in the scatter plots. The two views use identical axis limits for direct
comparison. All four figures are rebuilt from `selection.csv`; they are
derived visualizations rather than part of the frozen scientific identity.

If the selection is missing, experiment proximity exits with the exact three
reference commands required to create it. A direct reference proximity call
rejects a frozen selection whose evidence or policy differs unless
`compute_proximity.sh --overwrite` is explicit; that option recomputes the
reference evidence and atomically replaces only that exact run's frozen
directory. The orchestrated `run_all.sh` path always uses that overwrite only
for reference and experiment proximity. Raw generation and SSCD records remain
under their normal cache-reuse rules unless the top-level `--overwrite` flag is
passed.

Later mechanism experiments use the same API:

```python
from utils.data.selection import load_target_pair_selection

selection = load_target_pair_selection(
    root,
    model_name=MODEL,
    scheduler_name=SCHEDULER,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=STEPS,
    num_seeds=N,
    selection_strategy=SELECTION_STRATEGY,
)
```

The returned object exposes `selection_strategy`, its model and sampler
identity, `num_seeds`, `included_indices`, `excluded_indices`, `prompt_frame`,
`frame`, `configuration`, and `sha256`.

### Proximity outputs

The successful reference invocation writes no second proximity-output tree;
the frozen selection directory contains its three authoritative data and
provenance files plus the derived figures and examples shown above.
For an experiment, `compute_proximity.sh` reads cached terminal and target
latents, joins every seed with its cached SSCD score, applies the frozen
selection, and writes:

```text
outputs/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/proximity/
└── experiment_S0_N<N>/
    ├── proximity.csv
    ├── proximity_vs_sscd.png
    ├── proximity_vs_sscd.pdf
    ├── proximity_vs_sscd_all_prompts.png
    ├── proximity_vs_sscd_all_prompts.pdf
    ├── run_config.json
    ├── summary.json
    └── examples/
        ├── manifest.json
        ├── retained/{highest,median,lowest}_l2_generated.png
        ├── retained/{highest,median,lowest}_l2_training.png
        ├── discarded/{highest,median,lowest}_l2_generated.png
        └── discarded/{highest,median,lowest}_l2_training.png
```

`proximity.csv` retains every experimental seed row, including rows from
included, discarded, and unusable prompts, together with the frozen decision
and its strategy-specific evidence and reason for auditability. It also records
`experiment_prompt_spearman`, repeated on each prompt's rows and left blank
when the experimental L2 or SSCD vector is constant or has fewer than two
observations. `proximity_vs_sscd.{png,pdf}` shows the included prompts, while
`proximity_vs_sscd_all_prompts.{png,pdf}` shows the same experiment before the
frozen decisions are applied. Both views use identical axis limits and show
unconnected seed-level scatter points. Category colors and legend order are
MV (C3), RV (C1), TV (C0), and N (C2), with gray for other/unlabeled samples.
Legend markers are opaque and category labels use typewriter text. SSCD is
shown on the vertical axis; category-colored figures have no SSCD colorbar.
The selected view's annotation and `summary.json` aggregate one experimental
Spearman value per included prompt: selected and evaluable prompt counts, the
count and fraction of evaluable prompts with rho below zero, and median rho.
No pooled seed-row correlation is reported. `run_config.json` pins the
generation, SSCD, selection-strategy, and frozen-selection identities. Plotting
reloads `proximity.csv`; it does not rerun diffusion.

#### Automatic representative examples

Every successful `compute_proximity.sh` invocation refreshes an `examples/`
directory beside that invocation's existing proximity figures. Experiment
examples therefore live inside `outputs/.../proximity/experiment_S0_N<N>/`.
Reference examples live beside the frozen-selection figures shown above; they
do not create a second reference output tree. Re-running reference proximity
refreshes these derived examples even when the compatible frozen selection is
reused.

The exporter considers retained and discarded prompts separately. A prompt is
eligible only when it has usable, complete terminal-L2 observations for all
`N` requested seeds (normally `20`). Within each group, prompts are ordered by
their mean terminal L2 across those seeds, then deterministically by
`source_row_number` and `original_index`. The chosen representatives are the
actual lowest prompt, the actual lower-middle prompt at index
`(number_of_prompts - 1) // 2`, and the actual highest prompt; the median is not
an interpolated value. A two-prompt group emits only its distinct lowest and
highest representatives, a one-prompt group emits that prompt only as the
median, and an empty group emits none. Thus small groups never duplicate an
example under multiple rank names.

For each representative, the exporter saves the existing cached montage of
all `N` seed-ordered generated images to
`examples/<retained|discarded>/<highest|median|lowest>_l2_generated.png` at 75% of
its width and height. The paired normalized training target is saved to the
matching `..._l2_training.png` name with its longest edge capped at 256 pixels.
These display-only copies use Lanczos downsampling, preserve aspect ratios,
never upscale, and use optimized PNG compression. Training targets already
within the cap keep their original PNG bytes if re-encoding would not make
them smaller. The original cached montage and training image remain unchanged.
This cache-only export performs no diffusion inference.
Incomplete or unusable prompts are excluded from this
gallery, but that exclusion is not a new selection filter and does not alter
the frozen GMM decision.

`examples/manifest.json` records the ranking definition, lower-middle median
rule, `N`, and source CSV. Each exported entry records its retained/discarded group,
rank, prompt and identifiers, mean terminal L2 (`mean_l2_norm`), mean SSCD
(`mean_sscd`), and individual terminal L2 norms (`l2_norms`, aligned with
`seeds`). Both means use all contributing generation seeds for that prompt.
Source and exported paths, dimensions, and separate file hashes trace every
resized image back to its unmodified cache. `target_image_sha256` retains the
source target identity; `generated_image_sha256` and `training_image_sha256`
describe the exported PNGs.

The GMM diagnostic uses the same diagonal fit and any-high-seed prompt rule;
Spearman sign is descriptive only. The GMM/k-means diagnostic scripts write
`gmm_k2{,_spearman,_kind}.{png,pdf}` and
`kmeans_k2{,_spearman,_kind}.{png,pdf}` views. Color encodes SSCD; line style
encodes component, Spearman sign, or prompt kind. On membership views, each
segment's style represents its right endpoint in L2 order. These diagnostic
plots do not change the selection rule.

### Theorem 1 selected-pair outputs

Theorem 1 loads the exact frozen selection before computation and evaluates
only its included prompt–target indices at all saved timesteps. Each source has
one row per pair and timestep (`P * T` rows). Independent forward-corruption
draws are paired across timesteps and kept disjoint from the Gaussian evaluation
seeds. The measured forward-corruption loss $\mathcal{L}_t(c)$ is shared between
sources at the same pair and timestep. Gaussian recovery evaluates the
conditional branch on the same independent $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$
samples at each timestep; trajectory recovery uses each cached latent and its
conditional prediction. Both average squared recovery errors over generation
seeds before taking a square root.

The primary `theorem1_loss_recovery.{png,pdf}` figure uses only the actual
initial timestep $T$. Each selected prompt is one scatter point: x is
`normalized_loss_rmse`,
$\sqrt{\mathcal{L}_T(c)/[d(\alpha_T^2/\sigma_T^2)]}$, and y is the independently
measured Gaussian recovery RMSE,
$\sqrt{\mathbb{E}_{\mathbf{x}_T}[\|\widehat{\mathbf{x}}_{0\mid T,c}(\mathbf{x}_T)-\mathbf{x}^{\star}\|^2]/d}$.
Prompts are not joined by lines, and this primary figure has no median or
reference line. It compares the theorem's normalized-loss premise with
recovery at a finite initialization; it neither proves convergence in
probability nor asserts an exact $y=x$ identity. The loss is measured on
forward-corrupted targets, not defined from the generated-state recovery error.

The `theorem1_loss_recovery_trajectory.{png,pdf}` figure retains the
selected-prompt cached-trajectory diagnostic. It uses
$\sqrt{\mathcal{L}_t(c)/[d(\alpha_t^2/\sigma_t^2)]}$ on x and recovery RMSE on y.
It connects each selected prompt's observations in timestep order, even when
its loss is nonmonotonic. It does not join unrelated prompts or retain the old
timestep-aggregate bands and median-loss reference: normalized loss now varies
by prompt on the x-axis. With both sources, the exact Gaussian initial scatter
from the primary figure is overlaid with opaque markers above the curves,
using the same x, y, and SSCD values. In trajectory-only mode, where Gaussian
measurements are unavailable, the overlay instead uses each trajectory's own
cached initial-timestep observation. Cached-trajectory and independently
evaluated Gaussian initial recovery can differ slightly because of numerical
precision; the trajectory curves are not snapped to the Gaussian markers.

Later trajectory states have already been influenced by
prompt conditioning and CFG, so improved trajectory recovery is not a
replacement for the initial Gaussian test. The primary plot and trajectory
diagnostic are generated from existing logged measurements without changing
computation or cache configuration; `--plot` requires no model inference.
Prompt eligibility remains the frozen selected subset. The fixed-Gaussian
all-timestep view is removed, but its logged measurements and their computation
are unchanged. Plot regeneration safely removes only the obsolete
`theorem1_loss_recovery_noise_sweep.png` and
`theorem1_loss_recovery_noise_sweep.pdf` figure files from the requested output
directory; it does not delete cached measurements.

The SSCD color remains the mean target-specific endpoint score over those same
generation seeds; it is not an intermediate-image score.
SSCD is a cosine similarity, so negative values are valid and remain unchanged
in the CSV. Only the displayed colors are clipped to the fixed `[0, 1]` range;
negative scores use the bottom viridis color and are not discarded.
The authoritative resumable cache is:

```text
logs/<experiment-run>/experiment_S0_N<N>/theorem1_loss_recovery/
└── <selection-hash>/
    └── loss_S<loss-seed>_N<num-loss-seeds>/
        └── evaluation_<source>/
            ├── run_config.json
            ├── theorem1_loss_recovery.csv
            └── record/
                └── <original_index>.json
```

`run_config.json` pins the model, scheduler, guidance scale, `T`, `N`, frozen
selection, loss stream and its seeds, requested evaluation source, generation
identity, physical schedule identity, and SSCD identity. An incompatible global
configuration is rejected with an instruction to use `--overwrite`; it is
never silently reused. Each record marker fingerprints the current paired
generation trajectory and SSCD evidence, allowing regenerated inputs to make
only the affected prompt stale without rereading their large tensors during
the initial cache check.

A normal run validates and reuses completed prompt records before device
resolution or model construction, then retries only missing, failed, or stale
prompts. A record is published atomically only after that whole prompt is
complete for every saved timestep and every source requested by its namespace.
After interruption, completed prompts resume; the current incomplete prompt is
recomputed in full. Resume is deliberately prompt-level, not cell-level, and
different `evaluation_<source>` namespaces do not share checkpoints.
`--max-records`, `--output-dir`, `--device`, and `--sample-batch-size` do not
invalidate compatible records because they are scope, presentation, or
execution controls rather than scientific inputs. A full cache hit therefore
returns on the CPU before GPU, model, or device setup.

On the first normal run after this cache format is introduced, a matching
current-schema output CSV may seed the new records only after full configuration,
schedule, selected-pair trajectory, and same-pair SSCD provenance validation.
This is strict bootstrap of current results, not migration of an old CSV format.

The aggregate cache CSV records source, timestep, and frozen-selection
provenance. The output directory remains a derived publication containing that
CSV and its figures. With both sources it contains one derived CSV and four
figure files:

```text
outputs/<experiment-run>/theorem1_loss_recovery/<selection-hash>/evaluation_both/
├── theorem1_loss_recovery.csv
├── theorem1_loss_recovery.png
├── theorem1_loss_recovery.pdf
├── theorem1_loss_recovery_trajectory.png
└── theorem1_loss_recovery_trajectory.pdf
```

Plot-only mode is CPU-only. It reads the aggregate CSV from the logs cache and
verifies the frozen selection and full selection hash against every row, then
republishes the derived CSV and figures without computation or inferred
bootstrap from `outputs/`. It cannot silently plot a CSV produced from a
different selected prompt set.

### Lemma 2 convergence outputs

Lemma 2 is zero-centered by default. Pass `--use-mu` to load the shared
`mu_hat.pt` estimated from `B` Gaussian seeds `N` through `N+B-1` and instead
center on $\widehat{\boldsymbol{\mu}}$. All sources use that same fixed center
and the saved DDIM or DDPM schedule. Gaussian mode runs fresh unconditional inference
once per evaluation seed and timestep; prompts do not enter the empty branch.
Its `N * T` rows have empty prompt IDs and trajectory hashes. Trajectory mode
loads the exact frozen GMM, GMM-evidence, or Spearman selection and reconstructs
the unconditional estimate from every cached latent and matching prediction.
It retains all `P * N * T` observations. Both sources compute
`centered_distance_rmse` in float64. No mean is re-estimated at later timesteps.

The CSV has one row per evaluation-seed–timestep in the Gaussian source and
one row per selected-prompt–generation-seed–timestep in the trajectory source.
`evaluation_source` records which input distribution produced each measurement:

```text
selection_strategy,selection_hash,model_name,scheduler_name,guidance_scale,num_inference_steps,evaluation_source,centering_mode,num_baseline_seeds,evaluation_generation_scientific_config_hash,evaluation_schedule_sha256,baseline_generation_scientific_config_hash,baseline_mu_hat_sha256,record_id,original_index,generation_seed,step_index,timestep,alpha_t,sigma_t,snr_t,latent_dimension,centered_distance_rmse,target_sscd,trajectory_sha256,is_initial_timestep,status,error
```

The `target_sscd` column is populated for trajectory rows from the matching
prompt–seed endpoint score; it is blank for prompt-independent Gaussian rows.
Plot-only mode requires the current CSV schema and does not reconstruct missing
columns from other caches.

The `is_initial_timestep` column marks `step_index == 0`, the actual first cached
timestep; `scheduler_name` identifies DDIM versus DDPM.

`centering_mode` is `zero` by default and `mu_hat` with `--use-mu`. Zero mode
logs `num_baseline_seeds=0`, leaves both baseline-provenance fields empty, and
never loads baseline metadata or tensor data. Mean-centered mode records `B`
and the exact baseline scientific configuration and tensor hash.

`evaluation_schedule_sha256` pins the physical SHA-256 of the experiment
generation run's `schedule.pt`. With `--use-mu`, Lemma 2 also requires this
digest to match the shared baseline/reference source's
`source_schedule_sha256`; plot-only mode repeats that file-hash and cross-source
check without deserializing the schedule tensor. Zero mode has no baseline
cross-source check.

For each selected prompt record, `trajectory_sha256` is the canonical hash of
`{"latent": latent_file_sha256, "noise_prediction": prediction_file_sha256}`,
using the two tensor hashes published by the generation completion marker. Its
prompt-level value is repeated across that record's `N * T` rows. Plot-only
mode rereads each selected lightweight completion marker and validates its
identity and digest, but neither deserializes nor rehashes the tensor bytes.

The two default namespaces are:

```text
# Default: active center is zero.
outputs/<experiment-run>/lemma2_mean_convergence/centering_zero/<selection-hash>/evaluation_<source>/

# With --use-mu.
outputs/<experiment-run>/lemma2_mean_convergence/centering_mu_hat/baseline_S<N>_N<B>/<selection-hash>/evaluation_<source>/
```

Each contains `lemma2_mean_convergence.csv` plus a PNG and PDF per requested
source. The existing figure names identify Gaussian probes; the `_trajectory`
suffix identifies the cached diagnostic. `--output-dir` replaces the final
default directory. Figures plot `alpha_t^2/sigma_t^2` on a logarithmic x-axis
and show individual connected timestep curves, a black median, and nested
5th–95th, 25th–75th, and 40th–60th percentile bands. Cached prompt–seed curves
use matching endpoint target SSCD colors (viridis, fixed [0, 1]); Gaussian
seed curves stay neutral because they have no unique prompt–target SSCD.
There are no scatter points,
legends, or initial-timestep vertical markers. The y-axis shows distance to zero
by default or to $\widehat{\boldsymbol{\mu}}$ with `--use-mu`. The 95th percentile
is the outer band's upper edge, exposing the distribution's high-error tail.

### Corollary 3 CFG-amplification outputs

Corollary 3 uses the same active-center contract: $\mathbf{c}=\mathbf{0}$ by
default and $\mathbf{c}=\widehat{\boldsymbol{\mu}}$ with `--use-mu`. It
loads the exact frozen selection and matching experiment generation and SSCD
caches. For every included prompt–target pair and experiment seed, it uses the
same Gaussian $\mathbf{z}_s$ at each saved timestep for the Gaussian source, or
the corresponding cached latent and both epsilon branches for the trajectory
source. It loads the target latent and attaches same-seed target SSCD. The
endpoint SSCD is repeated across timesteps and never describes an intermediate
image. It writes `P * N * T` rows per source without averaging across seeds
or saving latent tensors. The empty Gaussian prediction is reused across
prompts within each worker.

With $\mathbf{v}_{\star}=\mathbf{x}^{\star}-\mathbf{c}$ and
$\mathbf{v}_g=\widehat{\mathbf{x}}_{0\mid t,g}-\mathbf{c}$, the experiment solves
the no-intercept least-squares problem separately for each prompt, seed, and
timestep:

$\widehat{g}_t=\mathop{\mathrm{argmin}}_a
\|\mathbf{v}_g-a\mathbf{v}_{\star}\|_2^2
=\langle\mathbf{v}_g,\mathbf{v}_{\star}\rangle/
\|\mathbf{v}_{\star}\|_2^2$,

using float64 `torch.linalg.lstsq` with the target direction as a one-column
design matrix and one right-hand side per generation seed. There is no fitted
intercept; $\mathbf{c}$ remains either zero or the shared saved baseline. The
logged metrics include

`residual_rmse` $=\|\mathbf{r}_t\|_2/\sqrt{d}$, where
$\mathbf{r}_t=\mathbf{v}_g-\widehat{g}_t\mathbf{v}_{\star}$,

`guided_target_rmse` $=\|\widehat{\mathbf{x}}_{0\mid t,g}
-[\mathbf{c}+g(\mathbf{x}^{\star}-\mathbf{c})]\|_2/\sqrt{d}$, and

`unconditional_rmse` $=\|\widehat{\mathbf{x}}_{0\mid t,\emptyset}
-\mathbf{c}\|_2/\sqrt{d}$.

`conditional_recovery_rmse` remains the distance between the conditional clean
estimate and $\mathbf{x}^{\star}$. The implementation independently verifies
the exact CFG identity and never fits the tautological branch-difference
coefficient.

The CSV has exactly these columns:

```text
record_id,generation_seed,evaluation_source,step_index,timestep,alpha_t,sigma_t,snr_t,guidance_scale,centering_mode,baseline_generation_scientific_config_hash,baseline_schedule_sha256,baseline_mu_hat_sha256,num_baseline_seeds,fitted_guidance_scale,residual_rmse,guided_target_rmse,conditional_recovery_rmse,unconditional_rmse,target_sscd,status,error
```

Zero mode logs blank baseline hashes and `num_baseline_seeds=0`; `--use-mu` logs
the shared baseline provenance and `B`. Results are isolated as follows:

```text
outputs/<experiment-run>/corollary3_cfg_amplification/centering_zero/<selection-hash>/evaluation_<source>/
outputs/<experiment-run>/corollary3_cfg_amplification/centering_mu_hat/baseline_S<N>_N<B>/<selection-hash>/evaluation_<source>/
```

Each directory contains one CSV and separate single-panel PNG/PDF figures
for each source and fitted quantity. The existing Gaussian/`_trajectory`
basenames show the fitted guidance scale $\widehat{g}_t$, with a dashed
$g=7.5$ reference; an additional `_residual` suffix shows
$\|\mathbf{r}_t\|_2/\sqrt{d}$. Each is a 4-by-4 STIX figure with a logarithmic
$\alpha_t^2/\sigma_t^2$ axis, without subplots. In `--use-mu` mode the labels
indicate the shared $\widehat{\boldsymbol{\mu}}$ used for centering. The corollary
predicts convergence of the fitted scale toward $g$ and of the residual toward
zero. The direct error at prescribed $g$ and both premise errors remain in the
CSV, but do not substitute for the fitted quantities in the figure. Existing
CSVs computed with the equivalent closed-form fit can be replotted without
inference. Black medians and nested percentile regions show both seedwise
distributions. Individual prompt–seed timestep lines use the fixed `[0, 1]`
viridis scale for same-seed target SSCD, with an opaque `SSCD` colorbar. Plot-only
mode validates the requested centering and source and the complete timestep
grid before plotting. All figures retain STIX fonts, 15-point labels, 12-point
axis numbers, a 4-by-4-inch size, and PNG/PDF output with 0.05-inch padding.

## Offline validation

The unit suite uses synthetic tensors and mocked components; it performs no
network requests and loads no real diffusion or SSCD model:

```bash
python -m compileall scripts utils
pytest -q
bash -n run_all.sh download_webster.sh generate.sh sscd.sh \
  compute_proximity.sh theorem1_loss_recovery.sh unconditional_baseline.sh \
  lemma2_mean_convergence.sh corollary3_cfg_amplification.sh
```
