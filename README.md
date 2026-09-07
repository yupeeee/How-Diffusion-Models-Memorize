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

## Workflow

Run all modeling and analysis stages for all three supported models from any
directory with one command:

```bash
./run_all.sh \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --selection-strategy spearman \
  --num-loss-seeds 20 \
  --loss-seed 0 \
  --downscale 4 \
  --device auto
```

When `--model` is omitted, `run_all.sh` runs `sdv1`, `sdv2`, and
`realvis` sequentially in that order. Pass, for example, `--model sdv1` to
run only one model. The orchestrator has ten numbered stage positions:

1. reference generation for seeds `N` through `2N-1`;
2. optional shared unconditional-baseline computation (skipped by default);
3. reference SSCD;
4. cache-only reference proximity and frozen prompt selection;
5. experiment generation for seeds `0` through `N-1`;
6. experiment SSCD;
7. cache-only experiment proximity;
8. Theorem 1 loss–recovery;
9. Lemma 2 convergence; and
10. Corollary 3 CFG amplification.

The default keeps stage 2 as an explicit skip and sets the Lemma 2 and
Corollary 3 center to $\mathbf{0}$. Pass `--use-mu` to make stage 2 draw
`B = --num-baseline-seeds` Gaussian initial latents (default `1000`) with seeds
`N` through `N+B-1`, evaluate the empty-condition branch at the actual initial
DDIM timestep, and save their mean as $\widehat{\boldsymbol{\mu}}$. This
baseline stage never traverses prompt records or reads SSCD, categories,
memorization labels, frozen selections, or cached prompt predictions.

`run_all.sh` uses `spearman` prompt selection by default; set
`--selection-strategy {gmm,gmm-evidence,spearman}` to choose the
posterior-mean global mixture, the GMM evidence-and-correlation hybrid, or the
within-prompt rank-correlation rule. Each strategy has separate theory-result
directories while reusing generation and SSCD caches. With `--use-mu`, the
unconditional baseline is shared across selection strategies and lives in its
own `S<N>_N<B>` namespace inside the reference generation run.

Lemma 2 loads the exact frozen selection and evaluates every included prompt
with all `N` experiment seeds at all `T` cached DDIM timesteps. For each
prompt–seed–timestep cell it reads the cached $\mathbf{x}_t$ and unconditional
epsilon prediction, reconstructs the unconditional clean estimate, and measures
its distance from the active center: $\mathbf{0}$ by default or
$\widehat{\boldsymbol{\mu}}$ with `--use-mu`. It draws no fresh Gaussian
latent and runs no model inference. Corollary 3 uses the same centering choice
for every selected prompt–target pair and all `N` experiment seeds at the actual
initial DDIM timestep. It remains restricted to DDIM with `g = 7.5`.


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
Reference proximity atomically rebuilds the selected strategy from the current
validated generation and SSCD caches. Every model keeps the same ten numbered
stage positions, including explicit skips when a configuration is incompatible.
The optional baseline position is also explicitly skipped unless `--use-mu` is
passed. A default all-model DDIM run therefore has 30 model stage positions
without performing baseline inference. Rebuilding a derived selection cannot
silently reuse stale completion-marker provenance.

Pass `--download` to run or resume the shared Webster preparation once before
the first model. The reference and experiment cache stages and all downstream
analyses receive the same model, scheduler, guidance scale, step count, and
`N`; only the reference and experiment seed blocks differ. Theorem 1, Lemma 2,
and Corollary 3 use only prompt–target pairs included by the exact frozen
selection. Lemma 2 retains one cached observation per selected prompt,
experiment seed, and timestep. `--num-baseline-seeds` affects only
`--use-mu` runs and their mean-centered output namespaces.

A downloader exit status of `2` means some records remain retryable; in that
case the orchestrator continues with all currently available prompt–image
pairs.
Other data errors and failures in later stages stop the sequence before the
next model.

Pass `--overwrite` only when every trajectory should be regenerated. The
orchestrator then also recomputes SSCD and, only with `--use-mu`, the shared
unconditional baseline. On every normal run, with or without that flag,
`run_all.sh` atomically rebuilds only the selected strategy's frozen reference
and experiment-proximity outputs from validated caches. Thus a plain run repairs
stale derived selection provenance while complete generation and SSCD caches
are reused. A direct `sscd.sh --overwrite` recomputes only the requested SSCD
seed pool. `--overwrite` cannot be combined with `--plot`.

Theorem 1 additionally averages its conditional-loss estimate over
`--num-loss-seeds` independent corruption draws per prompt–target pair
(default `20`).
`--loss-seed` (default `0`) is the deterministic root used to derive those
draws in a stream disjoint from generation seeds. Both options affect only the
Theorem 1 computation, so changing them does not change generation, SSCD, or
selection cache paths. For example:

```bash
./run_all.sh --model sdv1 --num-loss-seeds 32 --loss-seed 123
```

The selected strategy and frozen-selection hash are part of the Theorem 1 CSV
identity. Default results are isolated under
`outputs/<experiment-run>/theorem1_loss_recovery/<strategy>/<selection-hash>/`,
so changing the strategy or rebuilding its frozen selection cannot silently
reuse or overwrite results from a different selected prompt set.

The optional shared model-implied baseline is

$\widehat{\boldsymbol{\mu}} = (1/B) \sum_b
(\mathbf{x}_{T,b} - \sigma_T \boldsymbol{\epsilon}_{\emptyset}
(\mathbf{x}_{T,b},T))/\alpha_T$.

It is estimated only with `--use-mu`, exactly once at the actual initial DDIM
timestep from `B` independent Gaussian initial latents with seeds `N` through
`N+B-1`. The default instead sets the active center to $\mathbf{0}$ and has no
baseline-artifact dependency. In either mode, Lemma 2 uses every
frozen-selected prompt `i`, experiment seed `s`, and cached DDIM timestep `t`:

$\widehat{\mathbf{x}}_{0\mid t,\emptyset}^{(i,s)} =
(\mathbf{x}_{t}^{(i,s)} - \sigma_t
\boldsymbol{\epsilon}_{\emptyset}^{(i,s,t)})/\alpha_t$.

It logs `centered_distance_rmse` relative to $\mathbf{0}$ by default or to
$\widehat{\boldsymbol{\mu}}$ with `--use-mu`. Both $\mathbf{x}_t$ and the
unconditional epsilon prediction come from the validated experiment trajectory
cache, producing exactly `P * N * T` rows for `P` selected prompts. Lemma 2
performs no fresh sampling, model load, UNet evaluation, or prompt encoding.

`--device auto` uses every CUDA device visible to PyTorch. Set
`CUDA_VISIBLE_DEVICES` to restrict that pool, or pass `--device cpu`,
`--device mps`, `--device cuda`, or `--device cuda:N` to select exactly one
device. Model-inference GPU stages run one model replica per device. When
requested, the shared baseline shards Gaussian seeds across workers. Lemma 2
and Corollary 3 shard whole selected prompts across devices; Lemma 2 workers
perform only cached tensor loading and reductions, while Corollary 3 also loads
its cached target and SSCD evidence. The three models remain sequential.
Cache-only proximity and plotting stay on the CPU.

The GPU stages use independent inference processes, not gradient-oriented
`torch.distributed` DDP: there are no gradients or model updates to synchronize.
With `--use-mu`, the shared baseline reports one `B`-sample unconditional-
inference progress bar. Lemma 2 reports one `P * N * T` cached
prompt–seed–timestep progress bar, and Corollary 3 reports one `P * N`
prompt–experiment-seed progress bar.

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
to eager execution; CPU and MPS execution are unchanged. The first compiled
call pays compilation overhead, so short or cache-only runs may not become
faster. Long runs with repeated UNet calls are expected to benefit, but the
speedup depends on the workload, model, GPU, and software environment.

Pass `--plot` to `run_all.sh` to skip every computational stage and regenerate
the Theorem 1, Lemma 2, and Corollary 3 PNG and PDF figures from their saved
CSVs. With no centering option, it selects the zero-centered Lemma 2 and
Corollary 3 outputs. Pass `--use-mu` as well to select their model-mean-centered
outputs. The baseline command is never invoked in plot-only mode. Lemma 2 is
explicitly skipped for non-DDIM configurations, and Corollary 3 is skipped
unless the scheduler is DDIM and `g = 7.5`. The same all-model default applies.

```bash
./theorem1_loss_recovery.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy spearman --num-loss-seeds 20 --loss-seed 0 --plot

# Zero-centered defaults.
./lemma2_mean_convergence.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy spearman --plot
./corollary3_cfg_amplification.sh \
  --model sdv1 --scheduler ddim --g 7.5 --T 50 --N 20 \
  --selection-strategy spearman --plot

# Add both options to either theory wrapper to select its saved mean-centered run.
# --use-mu --num-baseline-seeds 1000
```

Zero-centered plot-only mode never loads or validates a baseline artifact. With
`--use-mu`, it validates the shared-baseline metadata and hash without
deserializing `mu_hat.pt`. Neither mode performs inference or rereads latent
tensors. Plot-only validation combines each CSV's recorded centering and
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
SELECTION_STRATEGY=spearman

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
from all seeds `N` through `2N-1`. Run the experiment only after it succeeds.
`run_all.sh` checks reference generation, optionally computes or validates the
shared baseline when `--use-mu` is passed, checks reference SSCD, and then
atomically rebuilds the selection and its figures.

Supported models are `sdv1`, `sdv2`, and `realvis`; supported schedulers are
`ddim` and `ddpm`. The RealisticVision dataset directory remains named
`realisticvision`. `sdv2` uses the public
`Manojb/stable-diffusion-2-1-base` repository. At load time, its `main` branch
is resolved to an immutable commit and that revision is recorded in every
generation run configuration.

Lemma 2 requires `--scheduler ddim`; Corollary 3 additionally requires
`--g 7.5`. The optional baseline also requires DDIM. Other valid configurations
keep all ten numbered positions and print explicit skips where needed.

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
seed block uses `seed_S<seed-start>_N<N>`. All selection strategies reuse the
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

The `gmm` strategy pools the `(l2_norm, sscd)` observations from every
prompt with all N complete reference observations. It standardizes each of the
two coordinates over that pooled population and fits one global,
two-component, full-covariance Gaussian mixture. The components are ordered by
their fitted SSCD means. For each prompt, the strategy averages the N posterior
probabilities of membership in the low-SSCD component. The whole prompt is
included exactly when this mean is below 0.5, so the high-SSCD component carries
more than half of that prompt's total soft mixture membership. Spearman rho and
the fitted marginal SSCD boundary are audit values, not inputs to this decision.

With `--selection-strategy gmm-evidence`, the same global mixture fit is used
to derive the weighted marginal SSCD-density boundary between the component
means; no SSCD threshold is hardcoded. For each prompt, the strategy counts
reference seeds for which SSCD is above that boundary and L2 is strictly below
that prompt's median L2. The whole prompt is included exactly when this count is
at least one and the Spearman correlation over all N paired observations is
finite and negative. Thus an intermittent prompt is not discarded merely
because most of its seeds occupy the low-SSCD cloud.

The `spearman` strategy—the `run_all.sh` default—computes, for each complete prompt,
Spearman's rank correlation between terminal latent L2 distance and
paired-target SSCD. The whole prompt is included exactly when the correlation
is finite and `prompt_spearman < 0`; a zero or positive correlation is
discarded. A constant L2 or SSCD vector makes this correlation undefined and
the prompt unusable.

Missing, duplicate, or non-finite reference observations make a prompt
unusable under any strategy, and unusable prompts are never silently
included. Prompt kind is retained only for audit; MV, RV, TV, N, and every
other kind are treated identically by all three rules.

Experimental seeds are an independent pool starting at 0 and never affect
selection. An experiment uses seeds 0 through `N-1` and applies the
already-frozen decisions for its exact strategy, model, and sampler
configuration.
Generation trajectories, noise predictions, target latents, preview montages,
and SSCD tensors remain intact for discarded and unusable prompts; selection
filters only the analysis.

The frozen reference directory contains the three authoritative artifacts and
four derived figures:

```text
data/webster/selection/<dataset>/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/<strategy>/reference_S<N>_N<N>/
├── selection.csv
├── config.json
├── summary.json
├── proximity_vs_sscd.png
├── proximity_vs_sscd.pdf
├── proximity_vs_sscd_all_prompts.png
└── proximity_vs_sscd_all_prompts.pdf
```

`selection.csv` has exactly N seed rows per prompt. It records `prompt`,
`kind`, `l2_norm`, `sscd`, the strategy, per-seed observation status, repeated
prompt-level decision, `prompt_spearman`, and reference provenance. Under
both `gmm` and `gmm-evidence`, it also records each seed's fitted component and
low-mode posterior; these fields are blank under `spearman`.
`prompt_gmm_evidence_seed_count` is populated only under `gmm-evidence` and is
blank under `gmm` and `spearman`. Under `gmm`, the mean low-mode posterior alone
determines selection. Under `gmm-evidence`, `prompt_spearman` and the evidence
count jointly determine selection; under `spearman`, only `prompt_spearman`
does.
`generated_image_path` points to the existing seed-ascending preview montage;
`generated_image_tile_index` identifies its zero-based row-major tile, so
indices `0` through `N-1` map directly to seeds `N` through `2N-1`. No image
is copied into the selection directory. `config.json` records the model,
sampler values, strategy and frozen policy, seed set, provenance hashes, and
the standardized global GMM fit and derived marginal SSCD boundary when
applicable, while `summary.json` records concise decision and observation
counts. `proximity_vs_sscd.{png,pdf}` is the
selected-prompt view. `proximity_vs_sscd_all_prompts.{png,pdf}` is the
pre-discard view containing every prompt with a complete finite N-seed
reference group, including both selected and discarded prompts. Incomplete or
otherwise unplottable prompts remain in `selection.csv` for audit but cannot
appear as scatter points. The two views use identical axis limits for direct
comparison. All four figures are rebuilt from `selection.csv`; they are
derived visualizations rather than part of the frozen scientific identity.

If the selection is missing, experiment proximity exits with the exact three
reference commands required to create it. A direct reference proximity call
rejects a frozen selection whose evidence or policy differs unless
`compute_proximity.sh --overwrite` is explicit; that option recomputes the
reference evidence and atomically replaces only that exact strategy's frozen
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
provenance files plus the four derived figures shown above.
For an experiment, `compute_proximity.sh` reads cached terminal and target
latents, joins every seed with its cached SSCD score, applies the frozen
selection, and writes only:

```text
outputs/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/proximity/<strategy>/
└── experiment_S0_N<N>/
    ├── proximity.csv
    ├── proximity_vs_sscd.png
    ├── proximity_vs_sscd.pdf
    ├── proximity_vs_sscd_all_prompts.png
    ├── proximity_vs_sscd_all_prompts.pdf
    ├── run_config.json
    └── summary.json
```

`proximity.csv` retains every experimental seed row, including rows from
included, discarded, and unusable prompts, together with the frozen decision
and its strategy-specific evidence and reason for auditability. It also records
`experiment_prompt_spearman`, repeated on each prompt's rows and left blank
when the experimental L2 or SSCD vector is constant or has fewer than two
observations. `proximity_vs_sscd.{png,pdf}` shows the included prompts, while
`proximity_vs_sscd_all_prompts.{png,pdf}` shows the same experiment before the
frozen decisions are applied. Both views use identical axis limits. The
selected view's annotation and `summary.json` aggregate one experimental
Spearman value per included prompt: selected and evaluable prompt counts, the
count and fraction of evaluable prompts with rho below zero, and median rho.
No pooled seed-row correlation is reported. `run_config.json` pins the
generation, SSCD, selection-strategy, and frozen-selection identities. Plotting
reloads `proximity.csv`; it does not rerun diffusion or create duplicate
tables, image galleries, or copied previews.

### Theorem 1 selected-pair outputs

Theorem 1 loads the exact frozen selection before computation and evaluates
only its included prompt–target indices. The CSV contains no discarded or
unusable prompt rows and records the constant `selection_strategy` and
`selection_hash` needed to audit its population. Results contain only:

```text
outputs/<experiment-run>/theorem1_loss_recovery/<strategy>/<selection-hash>/
├── theorem1_loss_recovery.csv
├── theorem1_loss_recovery.png
└── theorem1_loss_recovery.pdf
```

Plot-only mode reloads that same frozen selection and verifies the strategy and
full selection hash against every CSV row before replacing both figure formats.
It cannot silently plot a CSV produced from a different selected prompt set.

### Lemma 2 convergence outputs

Lemma 2 is zero-centered by default. Pass `--use-mu` to load the shared
`mu_hat.pt` estimated from `B` Gaussian seeds `N` through `N+B-1` and instead
center on $\widehat{\boldsymbol{\mu}}$. Both modes load the exact frozen GMM,
GMM-evidence, or Spearman selection. For each of its `P` included prompts,
every experiment seed `0` through `N-1`, and every timestep in the cached
`T`-step DDIM trajectory, the experiment reconstructs the unconditional clean
estimate from cached $\mathbf{x}_t$ and unconditional epsilon prediction. It
computes `centered_distance_rmse` in float64 relative to the selected center and
performs no fresh sampling or model, UNet, or prompt-encoder inference.

The CSV has exactly one row per selected-prompt–generation-seed–timestep
measurement and these columns:

```text
selection_strategy,selection_hash,model_name,scheduler_name,guidance_scale,num_inference_steps,centering_mode,num_baseline_seeds,evaluation_generation_scientific_config_hash,evaluation_schedule_sha256,baseline_generation_scientific_config_hash,baseline_mu_hat_sha256,record_id,original_index,generation_seed,step_index,timestep,alpha_t,sigma_t,snr_t,latent_dimension,centered_distance_rmse,trajectory_sha256,is_actual_ddim_initial_timestep,status,error
```

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
outputs/<experiment-run>/lemma2_mean_convergence/centering_zero/<strategy>/<selection-hash>/

# With --use-mu.
outputs/<experiment-run>/lemma2_mean_convergence/centering_mu_hat/baseline_S<N>_N<B>/<strategy>/<selection-hash>/
```

Each contains only `lemma2_mean_convergence.csv` plus the PNG and PDF figure.
`--output-dir` replaces the mode-specific default location. Both figures plot
`alpha_t^2/sigma_t^2` on a logarithmic x-axis and summarize the `P * N`
distances at each timestep with a black median curve and nested 5th–95th,
25th–75th, and 40th–60th percentile bands. The y-axis shows distance to zero
by default or to $\widehat{\boldsymbol{\mu}}$ with `--use-mu`.

### Corollary 3 CFG-amplification outputs

Corollary 3 uses the same active-center contract: $\mathbf{c}=\mathbf{0}$ by
default and $\mathbf{c}=\widehat{\boldsymbol{\mu}}$ with `--use-mu`. It
loads the exact frozen selection and matching experiment generation and SSCD
caches. For every included prompt–target pair and experiment seed, it uses the
cached initial latent and both cached epsilon-prediction branches, loads the
target latent, and attaches same-seed target SSCD. It writes exactly `P * N`
rows without averaging across seeds or saving latent tensors.

With $\mathbf{v}_{\star}=\mathbf{x}^{\star}-\mathbf{c}$ and
$\mathbf{v}_g=\widehat{\mathbf{x}}_{0\mid T,g}-\mathbf{c}$, the fit and
logged metrics are

$\widehat{g}=\langle\mathbf{v}_g,\mathbf{v}_{\star}\rangle/
\|\mathbf{v}_{\star}\|_2^2$,

`residual_rmse` $=\|\mathbf{v}_g-\widehat{g}\mathbf{v}_{\star}\|_2/\sqrt{d}$,

`guided_target_rmse` $=\|\widehat{\mathbf{x}}_{0\mid T,g}
-[\mathbf{c}+g(\mathbf{x}^{\star}-\mathbf{c})]\|_2/\sqrt{d}$, and

`unconditional_rmse` $=\|\widehat{\mathbf{x}}_{0\mid T,\emptyset}
-\mathbf{c}\|_2/\sqrt{d}$.

`conditional_recovery_rmse` remains the distance between the conditional clean
estimate and $\mathbf{x}^{\star}$. The implementation independently verifies
the exact CFG identity and never fits the tautological branch-difference
coefficient.

The CSV has exactly these columns:

```text
record_id,generation_seed,timestep,alpha_t,sigma_t,snr_t,guidance_scale,centering_mode,baseline_generation_scientific_config_hash,baseline_schedule_sha256,baseline_mu_hat_sha256,num_baseline_seeds,fitted_guidance_scale,residual_rmse,guided_target_rmse,conditional_recovery_rmse,unconditional_rmse,target_sscd,status,error
```

Zero mode logs blank baseline hashes and `num_baseline_seeds=0`; `--use-mu` logs
the shared baseline provenance and `B`. Results are isolated as follows:

```text
outputs/<experiment-run>/corollary3_cfg_amplification/centering_zero/<strategy>/<selection-hash>/
outputs/<experiment-run>/corollary3_cfg_amplification/centering_mu_hat/baseline_S<N>_N<B>/<strategy>/<selection-hash>/
```

Each directory contains only the CSV and its PNG/PDF figure. The 4-by-4 STIX
figure plots the fitted guidance scale against residual RMSE for the active
center. Points use the fixed `[0, 1]` viridis scale for same-seed target SSCD,
with an opaque `SSCD` colorbar and a reference line at `g = 7.5`. Plot-only
mode selects and validates the requested centering mode without computation.

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
