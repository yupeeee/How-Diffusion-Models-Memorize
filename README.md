# How Diffusion Models Memorize

This project recovers the Webster prompt–image benchmark, caches complete
diffusion trajectories, scores generated endpoints with SSCD, and measures
terminal-latent proximity. The scientific cache is deliberately reusable:
proximity and plotting never rerun diffusion.

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
  --downscale 4
```

When `--model` is omitted, `run_all.sh` runs `sdv1`, `sdv2`, and
`realvis` sequentially in that order. Pass, for example, `--model sdv1` to
run only one model. For each model, the canonical orchestrator uses the
existing Webster dataset and makes seven stage invocations in order: reference
generation for seeds 20–39; reference SSCD; cache-only reference proximity,
which freezes the target-pair selection; experiment generation for seeds 0
through `N-1`; experiment SSCD; cache-only experiment proximity; and the
Theorem 1 loss–recovery experiment. Thus the default all-model run has 21
model stages. Pass `--download` to run or resume the shared Webster
preparation once before the first model. The Theorem 1 stage requires the
canonical DDIM/g7.5/T50/N20 experiment and is explicitly skipped for other
sampling settings. A downloader exit status of `2` means some records remain
retryable; in that case the orchestrator continues with all currently
available prompt–image pairs. Other data errors and failures in later stages
stop the sequence before the next model.

Pass `--plot` to `run_all.sh` to skip every computational stage and regenerate
only the Theorem 1 PDFs from saved CSVs. It follows the same all-model default;
combine it with `--model` to plot one model. The experiment wrapper exposes
the same mode directly (with `--plot-only` retained as a compatibility alias):

```bash
./theorem1_loss_recovery.sh --model sdv1 --plot
```

`N` must be positive and at most 20, so experimental randomness never overlaps
the fixed reference pool. All invoked stage wrappers use unbuffered Python
output, and their `tqdm` progress stays visible in redirected logs: generation
records, denoising steps, and previews; SSCD records and checkpoint bytes; and
proximity records, analysis views, and the held-out TV and N preview galleries.
When `--download` is present, Webster's ten local phases are also shown.

To prepare or retry Webster data independently:

```bash
./download_webster.sh
```

Then create or resume the fixed selection reference. Its model follows
`--model`, but its sampling configuration and seed block are not tunable:

```bash
./generate.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 20 \
  --downscale 4
```

```bash
./sscd.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 20
```

```bash
./compute_proximity.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 20
```

That cache-only proximity call creates or validates the frozen selection. Run
the experiment only after it succeeds. This example uses all experimental
seeds 0–19; a smaller `N` uses the prefix 0 through `N-1`:

```bash
./generate.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 0 \
  --downscale 4
```

```bash
./sscd.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 0
```

```bash
./compute_proximity.sh \
  --model sdv1 \
  --scheduler ddim \
  --g 7.5 \
  --T 50 \
  --N 20 \
  --seed-start 0
```

Supported models are `sdv1`, `sdv2`, and `realvis`; supported schedulers are
`ddim` and `ddpm`. The RealisticVision dataset directory remains named
`realisticvision`. `sdv2` uses the public
`Manojb/stable-diffusion-2-1-base` repository. At load time, its `main` branch
is resolved to an immutable commit and that revision is recorded in every
generation run configuration.

### Data recovery

`download_webster.sh` organizes the three 500-record metadata views
and persists recovery state in
`data/webster/state/recovery.sqlite`. On every rerun it first
performs a fast structural check of saved image paths, headers, dimensions,
and formats. Valid recovered images are skipped without a network request; a
missing or visibly corrupt artifact alone is audited and reactivated. The
final validation phase still performs full dataset checks. Completed misses
stay completed, while only pending or retryable records proceed to network
recovery. Old `unresolved` records are narrowly reopened when either the versioned
audited-mirror strategy or Arquivo.pt strategy is newer than the one that
produced their prior miss.

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
variants. Replay identity and every recognized legacy MD5 or modern SHA-1
digest are verified; malformed reported digests are rejected, and supplied,
requested, and final replay URLs are recorded separately. Arquivo requests do
not follow redirects.

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
`--download` is present. `--per-host` is an alias for
`--per-host-concurrency` in both entry points.

### Generation cache

Generation processes every available prompt–target pair, including every TV
prompt. For each original index it stores:

- `latent/<index>.pt`: the complete trajectory `[N, T + 1, C, H, W]`;
- `noise_pred/<index>.pt`: unconditional and conditional predictions, both
  recorded before classifier-free guidance is applied;
- `target_latent/<index>.pt`: the deterministic paired-image VAE latent;
- `image/<index>.png`: a non-scientific preview montage; and
- `record/<index>.json`: the completion marker and artifact hashes.

After every generation or fully cached resume, `generate.sh` validates and
scans only the small `target_latent` files, one tensor at a time, with a
dedicated `tqdm` bar. It does not read the large trajectories or noise
predictions and does not rerun the VAE. The report is written to:

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

Every seed role lives beneath one shared base-run directory. Experiment caches
use `experiment_S0_N<N>`, the fixed selection reference uses
`reference_S20_N20`, and any other nonzero seed block uses
`seed_S<seed-start>_N<N>`. The canonical pair is therefore:

```text
logs/<model>_ddim_g7.5_T50_N20/
├── experiment_S0_N20/
└── reference_S20_N20/
```

A record is resumed only when its marker, configuration identity, file hashes,
shapes, and dtypes validate. Existing valid scientific tensors are never
overwritten.

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

### Frozen target-pair selection

Target-pair selection happens only during cache-based proximity analysis. The
frozen reference selection is model-specific, while its boundary and rules are
fixed across all supported models:

- a TV prompt is included iff its arithmetic mean paired-target SSCD over
  reference seeds 20–29 is greater than or equal to `0.25`;
- an N prompt is included iff its arithmetic mean paired-target SSCD over
  reference seeds 20–29 is greater than or equal to `0.25`;
- MV and RV retain their prior unconditional-inclusion behavior;
- reference seeds 30–39 compute validation diagnostics only and never affect
  inclusion;
- experimental seeds 0–19 affect neither selection nor reference validation;
  and
- terminal latent proximity, latent-distance thresholds, Pearson/Spearman
  correlation, and plot appearance never affect inclusion.

Thus both TV and N prompts exactly at `0.25` are included. The boundary is
never re-optimized per model or per run.
Selection artifacts record schema version 4 and policy
`target_pair_selection_tv_ge_0_25_n_ge_0_25`; N decisions use
`included_n_target_supported` and `excluded_n_target_unsupported` status
vocabulary.
Each model builds one frozen selection only from the exact reference run
`<model>_ddim_g7.5_T50_S20_N20`, containing seeds 20–39, and stores it in
`data/webster/selection/<dataset-model>/reference_S20_N20/`. Experiment runs
starting at seed 0 with `N <= 20` reuse that frozen model-specific selection.
If it is missing, proximity exits with the exact three reference commands
required to create it.

The selection unit is the whole prompt–target pair. Excluded TV and N prompts
below the boundary remain intact as target-unsupported diagnostics:
generation trajectories, noise predictions, target latents, generated
previews, and SSCD tensors are all preserved. Every seed for those prompts is
removed only from the target-supported selected analysis. Older or wrong-policy
frozen selections—including schema-2 TV-only selections and schema-3 selections
with the reversed N rule—are rejected rather than overwritten. Through
proximity, the incompatibility reports only the exact frozen-selection and
current derived proximity directories that must be archived or removed before
rebuilding; no raw cache directory should be changed.

Later mechanism experiments use the same API:

```python
from utils.data.selection import load_target_pair_selection

selection = load_target_pair_selection(root, model_name="sdv1")
```

The returned object exposes `included_indices`, `excluded_indices`,
`selected_tv_indices`, `excluded_tv_indices`, `selected_n_indices`,
`excluded_n_indices`, `frame`, `configuration`, and `sha256`.

### Proximity outputs

`compute_proximity.sh` reads cached terminal and target latents, joins every
seed with its cached SSCD score, freezes or validates the reference selection,
and writes:

```text
outputs/<model>_<scheduler>_g<guidance>_T<steps>_N<N>/proximity/
├── experiment_S0_N<N>/
│   └── ...
└── reference_S20_N20/
    ├── records/<original_index>.pt
    ├── paired_all.{csv,parquet}
    ├── paired_selected.{csv,parquet}
    ├── selection.csv
    ├── selected_tv.csv
    ├── excluded_tv.csv
    ├── selected_n.csv
    ├── excluded_n.csv
    ├── threshold_diagnostics.csv
    ├── held_out_tv/
    │   ├── gallery.html
    │   ├── manifest.csv
    │   ├── config.json
    │   ├── summary.json
    │   └── prompts/<original_index>/
    │       ├── generated.png
    │       └── prompt.txt
    ├── held_out_n/
    │   ├── gallery.html
    │   ├── manifest.csv
    │   ├── config.json
    │   ├── summary.json
    │   └── prompts/<original_index>/
    │       ├── generated.png
    │       └── prompt.txt
    ├── proximity_vs_sscd_all.{png,pdf}
    ├── proximity_vs_sscd_selected.{png,pdf}
    ├── failed.csv
    ├── run_config.json
    └── summary.json
```

The fixed reference namespace is `reference_S20_N20`; experiment namespaces
are `experiment_S0_N<N>`. Both live under the experiment run's single
`proximity/` directory. Their generation caches follow the same grouping under
`logs/<model>_ddim_g7.5_T50_N20/`.

`paired_all` contains every completed prompt and seed; selection never removes
rows from `paired_all`. For an experiment, `paired_selected` contains every
experiment seed for every included prompt. It is the sole paper-facing,
target-supported selected table and correlation view. There is no
`paired_evaluation` artifact:
reference seeds 30–39 are diagnostics, not an experimental half-split.
Reference proximity output itself is also not paper-facing. The frozen
selection hash and its explicit TV/N category rules, fixed boundary, selection
seeds, and validation seeds are recorded in `run_config.json` inside each
seed-role namespace.

`held_out_tv/` and `held_out_n/` are non-scientific visual diagnostics for,
respectively, whole excluded TV prompts and whole excluded N prompts. They do
not refer to the validation seeds as “held out.” Each `gallery.html` shows the
exact prompt beside its seed-ascending generated montage. Each `manifest.csv`
records identity, normalized category, selection reason, selection and
validation SSCD means, seeds, source path, and hashes. A reference run therefore
shows seeds 20–39, while an experiment run shows that experiment's seeds
(normally 0–19). Missing or copied previews and gallery creation never affect
selection, paired tables, correlations, or scientific hashes.

## Offline validation

The unit suite uses synthetic tensors and mocked components; it performs no
network requests and loads no real diffusion or SSCD model:

```bash
python -m compileall scripts utils
pytest -q
bash -n run_all.sh download_webster.sh generate.sh sscd.sh \
  compute_proximity.sh theorem1_loss_recovery.sh
```
