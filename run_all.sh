#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="all"
SCHEDULER="all"
GUIDANCE_SCALE="7.5"
NUM_INFERENCE_STEPS="50"
NUM_SEEDS="20"
CENTER="reference-initial"
CENTER_EXPLICIT=0
CACHED_BASELINE=""
SELECTION_STRATEGY="gmm"
DOWNSCALE_FACTOR="4"
DEVICE="auto"
DIRECT_WORKERS="${WEBSTER_DIRECT_WORKERS:-24}"
DIRECT_ATTEMPTS="${WEBSTER_DIRECT_ATTEMPTS:-2}"
PER_HOST_CONCURRENCY="${WEBSTER_PER_HOST_CONCURRENCY:-4}"
DOWNLOAD_WEBSTER=0
PLOT_ONLY=0
OVERWRITE=0
RECOMPUTE=0
REFINE_NUMERICS=0
INCLUDE_DIAGNOSTICS=0
FIGURE_SUITE="paper"
TARGET_ERROR_TOLERANCE=""
MEAN_SOURCE=""
NUM_MEAN_SAMPLES=""
MEAN_SEED=""
NUM_LOSS_SEEDS=""
LOSS_SEED=""
LOSS_TIMESTEPS=""
NUM_UNCONDITIONAL_LOSS_SEEDS=""
MEASURE_UNCONDITIONAL_LOSS=""
PROBE_BATCH_SIZE=""
REFERENCE_LAW=""
REFERENCE_MANIFEST=""
REFERENCE_SNR_DECADES=""
TERMINAL_NOISE_RUN_ALPHA=""
NUMERICAL_DECIMAL_PRECISION=""
NUMERICAL_MAX_DECIMAL_PRODUCTS=""
NUMERICAL_MAX_VARIATION_NODES=""
NUMERICAL_VARIATION_ABSOLUTE_WIDTH=""
COUNTERFACTUAL_UNCONDITIONAL=0
COUNTERFACTUAL_STEPS=""

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Run sdv1/ddim, sdv1/ddpm, sdv2/ddim, and realvis/ddim.
Explicit model/scheduler options filter these supported pairs; requests with
no matching pair fail before any stage. --scheduler ddpm selects sdv1 only.
Protected generation, SSCD, and GMM proximity retain their existing defaults.
Theory resumes shared denoiser probes and one analytical trajectory pass per
model/scheduler, then renders six selected theory figures from saved scalar inputs
under project-root figures/<experiment>/theory/<seed-role>/. Proximity PDFs use
figures/<experiment>/proximity/<seed-role>/. Only PDFs are exported; scalar tables
and metadata stay under outputs/. Probe workers use all visible GPUs.
The reference-only SNR extension and terminal bounds use separate reusable caches.

Options:
  --download            Run/resume shared Webster preparation once first
  --plot                Render saved proximity/theory PDFs and cached-image examples
  --figure-suite paper  Optional alias for the default fixed paper suite
  --diagnostics         Compatibility flag; exports the same six theory figures
                        Diagnostic scalar measurements remain available
  --target-error-tolerance FLOAT
                        Independently supplied raw latent L2 tolerance (optional)
  --recompute-experiments
                        Resume/rebuild theory, including missing learned probes;
                        bypass download, generation, SSCD and proximity rebuilding
  --refine-numerics     Opt in to interval refinement of saved numerics only;
                        saved positive budget or 2000000 when omitted; no learned probes
  --numerical-max-products INT
                        Positive budget explicitly enables CUDA interval refinement
                        Normal/recompute default: 0 (numerical estimates only)
                        --numerical-max-decimal-products remains an alias
  --numerical-decimal-precision INT
                        Legacy saved-policy field; unavailable for GPU computation
  --numerical-max-variation-nodes INT
                        Per-row interval node budget (default: 65)
  --numerical-variation-absolute-width FLOAT
                        Requested raw-L2 enclosure width (default: 1e-6)
  --overwrite           Explicitly regenerate protected generation/SSCD caches
                        and rebuild derived theory (normal pipeline only)
  --model MODEL         sdv1, sdv2, realvis, or all (default: all three)
  --scheduler NAME      ddim, ddpm, or all (default: all supported pairs)
  --g FLOAT             Guidance scale (default: 7.5; finite values supported)
  --T INTEGER           Inference steps (default: 50)
  --N INTEGER           Experiment seeds 0..N-1; reference N..2N-1 (default: 20)
  --mean-source NAME    reference-min-snr (default), reference-initial, or cached-targets
  --num-mean-samples INT Independent analytical reference estimates for mu (default: 10000)
  --mean-seed INT        Dedicated mean-estimation RNG root (default: 0)
  --num-loss-seeds INT   Forward-target draws (analysis default: 64)
  --loss-seed INT        Independent probe RNG root (analysis default: 0)
  --loss-timesteps NAME  initial (default) or saved
  --num-unconditional-loss-seeds INT
                        Forward-marginal draws (default: 256); enables this stage
  --unconditional-loss  Enable optional forward-marginal losses
  --no-unconditional-loss Disable optional forward-marginal losses
  --counterfactual-unconditional
                        Optional learned unconditional comparison at matched inputs
  --counterfactual-steps LIST
                        Fixed comma-separated update indices (default: 0); requires opt-in
  --probe-batch-size INT Denoiser batch size, execution only (default: 8)
  --reference-law NAME   cached-targets (default) or manifest
  --reference-manifest PATH Required with --reference-law manifest
  --reference-snr-decades FLOAT
                        Analytical-only 97-point grid depth (default: 6)
  --terminal-noise-run-alpha FLOAT
                        Simultaneous terminal Gaussian noise failure budget (default: 0.05)
  --center NAME         Legacy diagnostics only: reference-initial, cached-baseline
  --cached-baseline PATH Existing independent baseline for cached-baseline center
  --use-mu              Deprecated alias for --center cached-baseline;
                        requires --cached-baseline, never estimates a new center
  --selection-strategy NAME
                        gmm only (default: gmm)
  --downscale INTEGER   Upstream preview downscale factor (default: 4)
  --device DEVICE       auto, cuda, or cuda:N; computation requires CUDA (default: auto)
                        auto uses all visible CUDA GPUs for generation and theory;
                        theory fails clearly when CUDA is unavailable
  --direct-workers INT  Webster direct-URL workers (default: 24)
  --direct-attempts INT Direct attempts per URL (default: 2)
  --per-host-concurrency INT Concurrent requests per host (default: 4)
  -h, --help            Show help

Posterior comparisons use one declared reference law: by default, uniform
distinct compatible cached targets before selection. The selected mu defaults
to the average of analytical unconditional references at minimum analytical SNR.
This finite law is not asserted to be the full training law. Legacy center
options do not redefine the reference law or the selected mu.
All selected experiment seeds remain in the analysis, including failed recovery.

--plot validates every requested scalar bundle and saved proximity metadata
before writing figures. It never reads raw trajectory tensors, evaluates models,
fits selection, runs VAE decoding, or modifies numerical logs. Proximity examples
read their existing cached PNGs and completion metadata. Copied scalar bundles
can also be plotted using theory_validation.sh --bundle PATH --plot.
--plot cannot be combined with --download, --overwrite, or --recompute-experiments.
--recompute-experiments cannot be combined with --download or --overwrite.
--refine-numerics is a separate cache-only analysis mode; missing inputs report
an explicit recomputation command. Normal analysis saves numerical estimates without
interval refinement. A positive --numerical-max-products opts in for that invocation;
normal/recompute modes do not inherit a previously saved positive interval budget.
Plot mode inherits omitted measurement settings from each saved bundle.
--loss-timesteps saved enables marginal losses unless explicitly disabled.
--evaluation-source and --num-baseline-seeds remain unsupported.
Legacy theorem/gallery/baseline wrappers remain explicitly invokable utilities.
The PYTHON environment variable is honored. Any failed stage stops the matrix.
EOF
}

missing_value() {
    printf 'run_all.sh: %s requires a value\n' "$1" >&2
    exit 2
}

invalid_value() {
    printf 'run_all.sh: invalid %s: %s\n' "$1" "$2" >&2
    exit 2
}

decimal_greater_than() {
    local left="$1"
    local right="$2"
    if ((${#left} != ${#right})); then
        ((${#left} > ${#right}))
    else
        [[ "$left" > "$right" ]]
    fi
}

normalize_positive_integer() {
    local option="$1"
    local value="$2"
    local destination="$3"
    local maximum="${4:-}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        invalid_value "$option" "$value (expected a positive integer)"
    fi
    value="${value#"${value%%[!0]*}"}"
    if [[ -z "$value" ]]; then
        invalid_value "$option" "0 (expected a positive integer)"
    fi
    if [[ -n "$maximum" ]] && decimal_greater_than "$value" "$maximum"; then
        invalid_value "$option" "$value (maximum: $maximum)"
    fi
    printf -v "$destination" '%s' "$value"
}

normalize_nonnegative_integer() {
    local option="$1"
    local value="$2"
    local destination="$3"
    local maximum="${4:-}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        invalid_value "$option" "$value (expected a nonnegative integer)"
    fi
    value="${value#"${value%%[!0]*}"}"
    if [[ -z "$value" ]]; then
        value="0"
    fi
    if [[ -n "$maximum" ]] && decimal_greater_than "$value" "$maximum"; then
        invalid_value "$option" "$value (maximum: $maximum)"
    fi
    printf -v "$destination" '%s' "$value"
}

normalize_finite_float() {
    local option="$1"
    local value="$2"
    local destination="$3"
    local normalized
    if [[ ! "$value" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
        invalid_value "$option" "$value (expected a finite float)"
    fi
    if ! normalized="$(LC_ALL=C awk -v value="$value" 'BEGIN {
        number = value + 0.0
        rendered = sprintf("%.15g", number)
        lowered = tolower(rendered)
        if (lowered ~ /nan|inf/) exit 1
        if (number == 0.0) rendered = "0"
        print rendered
    }')"; then
        invalid_value "$option" "$value (expected a finite float)"
    fi
    printf -v "$destination" '%s' "$normalized"
}

set_center() {
    if ((CENTER_EXPLICIT)) && [[ "$CENTER" != "$1" ]]; then
        invalid_value "--center" "conflicting centering options are mutually exclusive"
    fi
    CENTER="$1"
    CENTER_EXPLICIT=1
}

while (($# > 0)); do
    case "$1" in
        --download) DOWNLOAD_WEBSTER=1; shift; continue ;;
        --plot) PLOT_ONLY=1; shift; continue ;;
        --recompute-experiments) RECOMPUTE=1; shift; continue ;;
        --refine-numerics) REFINE_NUMERICS=1; shift; continue ;;
        --counterfactual-unconditional) COUNTERFACTUAL_UNCONDITIONAL=1; shift; continue ;;
        --diagnostics|--include-diagnostics) INCLUDE_DIAGNOSTICS=1; shift; continue ;;
        --overwrite) OVERWRITE=1; shift; continue ;;
        --unconditional-loss|--no-unconditional-loss)
            if [[ -n "$MEASURE_UNCONDITIONAL_LOSS" && "$MEASURE_UNCONDITIONAL_LOSS" != "$1" ]]; then
                invalid_value "$1" "conflicting unconditional-loss flags"
            fi
            MEASURE_UNCONDITIONAL_LOSS="$1"; shift; continue ;;
        --use-mu) set_center cached-baseline; shift; continue ;;
        --no-mu) invalid_value "$1" "zero centering is retired; use the saved reference mean for mu" ;;
        -h|--help) usage; exit 0 ;;
    esac
    option="${1%%=*}"
    case "$option" in
        --figure-suite)
            if [[ "$1" == *=* ]]; then value="${1#*=}"; else (($# >= 2)) || missing_value "$option"; value="$2"; shift; fi
            [[ "$value" == paper ]] || invalid_value "$option" "$value (discovery modes retired; use paper or omit this option)"
            FIGURE_SUITE="$value"; shift; continue ;;
        --model) destination=MODEL ;;
        --scheduler) destination=SCHEDULER ;;
        --g) destination=GUIDANCE_SCALE ;;
        --T) destination=NUM_INFERENCE_STEPS ;;
        --N) destination=NUM_SEEDS ;;
        --center) destination=CENTER_VALUE ;;
        --cached-baseline) destination=CACHED_BASELINE ;;
        --target-error-tolerance) destination=TARGET_ERROR_TOLERANCE ;;
        --mean-source) destination=MEAN_SOURCE ;;
        --num-mean-samples) destination=NUM_MEAN_SAMPLES ;;
        --mean-seed) destination=MEAN_SEED ;;
        --num-loss-seeds) destination=NUM_LOSS_SEEDS ;;
        --loss-seed) destination=LOSS_SEED ;;
        --loss-timesteps) destination=LOSS_TIMESTEPS ;;
        --num-unconditional-loss-seeds) destination=NUM_UNCONDITIONAL_LOSS_SEEDS ;;
        --probe-batch-size) destination=PROBE_BATCH_SIZE ;;
        --counterfactual-steps) destination=COUNTERFACTUAL_STEPS ;;
        --reference-law) destination=REFERENCE_LAW ;;
        --reference-manifest) destination=REFERENCE_MANIFEST ;;
        --reference-snr-decades) destination=REFERENCE_SNR_DECADES ;;
        --terminal-noise-run-alpha) destination=TERMINAL_NOISE_RUN_ALPHA ;;
        --numerical-decimal-precision) destination=NUMERICAL_DECIMAL_PRECISION ;;
        --numerical-max-products|--numerical-max-decimal-products) destination=NUMERICAL_MAX_DECIMAL_PRODUCTS ;;
        --numerical-max-variation-nodes) destination=NUMERICAL_MAX_VARIATION_NODES ;;
        --numerical-variation-absolute-width) destination=NUMERICAL_VARIATION_ABSOLUTE_WIDTH ;;
        --selection-strategy) destination=SELECTION_STRATEGY ;;
        --downscale) destination=DOWNSCALE_FACTOR ;;
        --device) destination=DEVICE ;;
        --direct-workers) destination=DIRECT_WORKERS ;;
        --direct-attempts) destination=DIRECT_ATTEMPTS ;;
        --per-host-concurrency) destination=PER_HOST_CONCURRENCY ;;
        --evaluation-source|--num-baseline-seeds)
            invalid_value "$option" "unsupported; use the direct probe measurement options" ;;
        *) printf 'run_all.sh: unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    if [[ "$1" == *=* ]]; then
        value="${1#*=}"; shift
    else
        (($# >= 2)) || missing_value "$option"
        value="$2"; shift 2
    fi
    printf -v "$destination" '%s' "$value"
    if [[ "$destination" == CENTER_VALUE ]]; then set_center "$CENTER_VALUE"; fi
done

case "$MODEL" in all|sdv1|sdv2|realvis) ;; *) invalid_value "--model" "$MODEL" ;; esac
case "$SCHEDULER" in all|ddim|ddpm) ;; *) invalid_value "--scheduler" "$SCHEDULER" ;; esac
case "$CENTER" in
    reference-initial|cached-baseline) ;;
    zero) invalid_value "--center" "zero centering is retired; use the saved reference mean for mu" ;;
    *) invalid_value "--center" "$CENTER" ;;
esac
[[ "$SELECTION_STRATEGY" == "gmm" ]] || invalid_value "--selection-strategy" "$SELECTION_STRATEGY"
if [[ "$CENTER" == cached-baseline && -z "$CACHED_BASELINE" && "$PLOT_ONLY" == 0 && "$REFINE_NUMERICS" == 0 ]]; then
    invalid_value "--cached-baseline" "required for the existing independent baseline center"
fi
if [[ "$CENTER" != cached-baseline && -n "$CACHED_BASELINE" && "$PLOT_ONLY" == 0 && "$REFINE_NUMERICS" == 0 ]]; then
    invalid_value "--cached-baseline" "requires --center cached-baseline"
fi
MODEL_SCHEDULER_PAIRS=()
for pair in sdv1:ddim sdv1:ddpm sdv2:ddim realvis:ddim; do
    pair_model="${pair%%:*}"; pair_scheduler="${pair#*:}"
    if [[ "$MODEL" != all && "$MODEL" != "$pair_model" ]]; then continue; fi
    if [[ "$SCHEDULER" != all && "$SCHEDULER" != "$pair_scheduler" ]]; then continue; fi
    MODEL_SCHEDULER_PAIRS+=("$pair")
done
if ((${#MODEL_SCHEDULER_PAIRS[@]} == 0)); then
    invalid_value "--model/--scheduler" "$MODEL/$SCHEDULER (no supported pair; choose sdv1/ddim, sdv1/ddpm, sdv2/ddim, or realvis/ddim)"
fi
DEVICE="${DEVICE#"${DEVICE%%[![:space:]]*}"}"
DEVICE="${DEVICE%"${DEVICE##*[![:space:]]}"}"
DEVICE="${DEVICE,,}"
[[ "$DEVICE" =~ ^(auto|cpu|mps|cuda|cuda:[0-9]+)$ ]] || invalid_value "--device" "$DEVICE"
if ((!PLOT_ONLY)) && [[ "$DEVICE" == cpu || "$DEVICE" == mps ]]; then
    invalid_value --device "theory computation requires CUDA; CPU/MPS fallback is disabled"
fi
if ((!PLOT_ONLY)) && [[ -n "$NUMERICAL_DECIMAL_PRECISION" ]]; then
    invalid_value --numerical-decimal-precision "CUDA uses outward binary64 enclosures; increase --numerical-max-products or --numerical-max-variation-nodes instead"
fi
normalize_positive_integer "--N" "$NUM_SEEDS" NUM_SEEDS 4611686018427387904
normalize_positive_integer "--T" "$NUM_INFERENCE_STEPS" NUM_INFERENCE_STEPS
normalize_finite_float "--g" "$GUIDANCE_SCALE" GUIDANCE_SCALE
if ((PLOT_ONLY)); then
    ((!DOWNLOAD_WEBSTER)) || invalid_value "--plot" "cannot be combined with --download"
    ((!OVERWRITE)) || invalid_value "--plot" "cannot be combined with --overwrite"
    ((!RECOMPUTE)) || invalid_value "--plot" "cannot be combined with --recompute-experiments"
fi
if ((RECOMPUTE)); then
    ((!DOWNLOAD_WEBSTER)) || invalid_value "--recompute-experiments" "cannot be combined with --download"
    ((!OVERWRITE)) || invalid_value "--recompute-experiments" "cannot be combined with --overwrite"
fi
if ((REFINE_NUMERICS)); then
    ((!PLOT_ONLY && !RECOMPUTE && !DOWNLOAD_WEBSTER && !OVERWRITE)) || invalid_value "--refine-numerics" "cannot be combined with --plot, --recompute-experiments, --download, or --overwrite"
fi
if ((!PLOT_ONLY && !RECOMPUTE && !REFINE_NUMERICS)); then
    normalize_positive_integer "--downscale" "$DOWNSCALE_FACTOR" DOWNSCALE_FACTOR
fi
if ((DOWNLOAD_WEBSTER)); then
    normalize_positive_integer "--direct-workers" "$DIRECT_WORKERS" DIRECT_WORKERS 64
    normalize_positive_integer "--direct-attempts" "$DIRECT_ATTEMPTS" DIRECT_ATTEMPTS 5
    normalize_positive_integer "--per-host-concurrency" "$PER_HOST_CONCURRENCY" PER_HOST_CONCURRENCY 8
    if decimal_greater_than "$PER_HOST_CONCURRENCY" "$DIRECT_WORKERS"; then
        invalid_value "--per-host-concurrency" "$PER_HOST_CONCURRENCY (cannot exceed --direct-workers $DIRECT_WORKERS)"
    fi
fi
CACHE_OVERWRITE_ARGUMENTS=()
if ((OVERWRITE)); then CACHE_OVERWRITE_ARGUMENTS=(--overwrite); fi
THEORY_CENTER_ARGUMENTS=()
if ((CENTER_EXPLICIT)); then THEORY_CENTER_ARGUMENTS=(--center "$CENTER"); fi
THEORY_MEASUREMENT_ARGUMENTS=()
if ((COUNTERFACTUAL_UNCONDITIONAL)); then
    THEORY_MEASUREMENT_ARGUMENTS+=(--counterfactual-unconditional)
fi
if [[ -n "$COUNTERFACTUAL_STEPS" ]]; then
    ((COUNTERFACTUAL_UNCONDITIONAL || PLOT_ONLY || REFINE_NUMERICS)) || invalid_value --counterfactual-steps "requires --counterfactual-unconditional"
    [[ "$COUNTERFACTUAL_STEPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || invalid_value --counterfactual-steps "expected comma-separated nonnegative integers"
    THEORY_MEASUREMENT_ARGUMENTS+=(--counterfactual-steps "$COUNTERFACTUAL_STEPS")
fi
for option_variable in "num-mean-samples:NUM_MEAN_SAMPLES" "num-loss-seeds:NUM_LOSS_SEEDS" "num-unconditional-loss-seeds:NUM_UNCONDITIONAL_LOSS_SEEDS" "probe-batch-size:PROBE_BATCH_SIZE" "numerical-decimal-precision:NUMERICAL_DECIMAL_PRECISION" "numerical-max-variation-nodes:NUMERICAL_MAX_VARIATION_NODES"; do
    option="--${option_variable%%:*}"; variable="${option_variable#*:}"
    if [[ -n "${!variable}" ]]; then
        normalize_positive_integer "$option" "${!variable}" "$variable"
        THEORY_MEASUREMENT_ARGUMENTS+=("$option" "${!variable}")
    fi
done
if [[ -n "$NUMERICAL_MAX_DECIMAL_PRODUCTS" ]]; then
    normalize_nonnegative_integer --numerical-max-decimal-products "$NUMERICAL_MAX_DECIMAL_PRODUCTS" NUMERICAL_MAX_DECIMAL_PRODUCTS 9223372036854775807
    THEORY_MEASUREMENT_ARGUMENTS+=(--numerical-max-products "$NUMERICAL_MAX_DECIMAL_PRODUCTS")
fi
if [[ -n "$NUMERICAL_DECIMAL_PRECISION" ]] && decimal_greater_than 32 "$NUMERICAL_DECIMAL_PRECISION"; then
    invalid_value --numerical-decimal-precision "must be at least 32"
fi
if [[ -n "$NUMERICAL_VARIATION_ABSOLUTE_WIDTH" ]]; then
    normalize_finite_float --numerical-variation-absolute-width "$NUMERICAL_VARIATION_ABSOLUTE_WIDTH" NUMERICAL_VARIATION_ABSOLUTE_WIDTH
    LC_ALL=C awk -v value="$NUMERICAL_VARIATION_ABSOLUTE_WIDTH" 'BEGIN {exit !(value > 0)}' || invalid_value --numerical-variation-absolute-width "must be positive"
    THEORY_MEASUREMENT_ARGUMENTS+=(--numerical-variation-absolute-width "$NUMERICAL_VARIATION_ABSOLUTE_WIDTH")
fi
if [[ -n "$NUM_MEAN_SAMPLES" ]] && ! decimal_greater_than "$NUM_MEAN_SAMPLES" 1; then
    invalid_value --num-mean-samples "must be at least 2"
fi
if [[ -n "$MEAN_SOURCE" ]]; then
    case "$MEAN_SOURCE" in reference-min-snr|reference-initial|cached-targets) ;; *) invalid_value --mean-source "$MEAN_SOURCE" ;; esac
    THEORY_MEASUREMENT_ARGUMENTS+=(--mean-source "$MEAN_SOURCE")
fi
if [[ -n "$MEAN_SEED" ]]; then
    normalize_nonnegative_integer --mean-seed "$MEAN_SEED" MEAN_SEED 9223372036854775807
    THEORY_MEASUREMENT_ARGUMENTS+=(--mean-seed "$MEAN_SEED")
fi
if [[ -n "$LOSS_SEED" ]]; then
    normalize_nonnegative_integer --loss-seed "$LOSS_SEED" LOSS_SEED 9223372036854775807
    THEORY_MEASUREMENT_ARGUMENTS+=(--loss-seed "$LOSS_SEED")
fi
if [[ -n "$LOSS_TIMESTEPS" ]]; then
    case "$LOSS_TIMESTEPS" in initial|saved) ;; *) invalid_value --loss-timesteps "$LOSS_TIMESTEPS" ;; esac
    THEORY_MEASUREMENT_ARGUMENTS+=(--loss-timesteps "$LOSS_TIMESTEPS")
fi
if [[ -n "$REFERENCE_LAW" ]]; then
    case "$REFERENCE_LAW" in cached-targets|manifest) ;; *) invalid_value --reference-law "$REFERENCE_LAW" ;; esac
    THEORY_MEASUREMENT_ARGUMENTS+=(--reference-law "$REFERENCE_LAW")
fi
if [[ -n "$REFERENCE_MANIFEST" ]]; then
    if [[ "$REFERENCE_LAW" == cached-targets || ( -z "$REFERENCE_LAW" && "$PLOT_ONLY" == 0 && "$REFINE_NUMERICS" == 0 ) ]]; then
        invalid_value --reference-manifest "requires --reference-law manifest"
    fi
    THEORY_MEASUREMENT_ARGUMENTS+=(--reference-manifest "$REFERENCE_MANIFEST")
fi
if [[ "$REFERENCE_LAW" == manifest && -z "$REFERENCE_MANIFEST" && "$PLOT_ONLY" == 0 && "$REFINE_NUMERICS" == 0 ]]; then
    invalid_value --reference-law "manifest requires --reference-manifest PATH"
fi
if [[ -n "$REFERENCE_SNR_DECADES" ]]; then
    normalize_finite_float --reference-snr-decades "$REFERENCE_SNR_DECADES" REFERENCE_SNR_DECADES
    LC_ALL=C awk -v value="$REFERENCE_SNR_DECADES" 'BEGIN {exit !(value > 0 && value <= 12)}' || invalid_value --reference-snr-decades "must be in (0,12]"
    THEORY_MEASUREMENT_ARGUMENTS+=(--reference-snr-decades "$REFERENCE_SNR_DECADES")
fi
if [[ -n "$TERMINAL_NOISE_RUN_ALPHA" ]]; then
    normalize_finite_float --terminal-noise-run-alpha "$TERMINAL_NOISE_RUN_ALPHA" TERMINAL_NOISE_RUN_ALPHA
    LC_ALL=C awk -v value="$TERMINAL_NOISE_RUN_ALPHA" 'BEGIN {exit !(value > 0 && value < 1)}' || invalid_value --terminal-noise-run-alpha "must be in (0,1)"
    THEORY_MEASUREMENT_ARGUMENTS+=(--terminal-noise-run-alpha "$TERMINAL_NOISE_RUN_ALPHA")
fi
if [[ -n "$MEASURE_UNCONDITIONAL_LOSS" ]]; then
    THEORY_MEASUREMENT_ARGUMENTS+=("$MEASURE_UNCONDITIONAL_LOSS")
fi
if [[ -n "$TARGET_ERROR_TOLERANCE" ]]; then
    normalize_finite_float --target-error-tolerance "$TARGET_ERROR_TOLERANCE" TARGET_ERROR_TOLERANCE
    [[ "$TARGET_ERROR_TOLERANCE" != -* ]] || invalid_value --target-error-tolerance "must be nonnegative raw latent L2"
    THEORY_CENTER_ARGUMENTS+=(--target-error-tolerance "$TARGET_ERROR_TOLERANCE")
fi
THEORY_FIGURE_ARGUMENTS=()
if ((INCLUDE_DIAGNOSTICS)); then THEORY_FIGURE_ARGUMENTS=(--diagnostics); fi
if [[ -n "$FIGURE_SUITE" ]]; then THEORY_FIGURE_ARGUMENTS+=(--figure-suite "$FIGURE_SUITE"); fi
if [[ -n "$CACHED_BASELINE" ]]; then THEORY_CENTER_ARGUMENTS+=(--cached-baseline "$CACHED_BASELINE"); fi

run_stage() {
    printf '[%s/%s] %s\n' "$STAGE_INDEX" "$STAGE_TOTAL" "$1"
    shift
    "$@"
    STAGE_INDEX=$((STAGE_INDEX + 1))
}
run_model_scheduler() {
    local STAGE_INDEX=1 STAGE_TOTAL=7
    local COMMON_ARGUMENTS=(--model "$MODEL" --scheduler "$SCHEDULER" --g "$GUIDANCE_SCALE" --T "$NUM_INFERENCE_STEPS" --N "$NUM_SEEDS")
    local REFERENCE_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" --seed-start "$NUM_SEEDS")
    local EXPERIMENT_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" --seed-start 0)
    local THEORY_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" "${THEORY_CENTER_ARGUMENTS[@]}" "${THEORY_FIGURE_ARGUMENTS[@]}" "${THEORY_MEASUREMENT_ARGUMENTS[@]}")
    if ((PLOT_ONLY)); then
        STAGE_TOTAL=3
        run_stage 'Plotting frozen reference proximity' "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" --selection-strategy gmm --plot
        run_stage 'Plotting saved experiment proximity' "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" --selection-strategy gmm --plot
        run_stage 'Plotting saved theory scalars' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --plot
        return
    fi
    if ((REFINE_NUMERICS)); then
        STAGE_TOTAL=1
        run_stage 'Refining saved theory numerics and rendering derived scalars' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --device "$DEVICE" --refine-numerics
        return
    fi
    if ((RECOMPUTE)); then
        STAGE_TOTAL=1
        run_stage 'Resuming direct theory measurements and missing learned probes' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --device "$DEVICE" --recompute-experiments
        return
    fi
    run_stage 'Checking/resuming reference trajectories' "$PROJECT_ROOT/generate.sh" "${REFERENCE_ARGUMENTS[@]}" --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming reference SSCD' "$PROJECT_ROOT/sscd.sh" "${REFERENCE_ARGUMENTS[@]}" --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming experiment trajectories' "$PROJECT_ROOT/generate.sh" "${EXPERIMENT_ARGUMENTS[@]}" --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming experiment SSCD' "$PROJECT_ROOT/sscd.sh" "${EXPERIMENT_ARGUMENTS[@]}" --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Rebuilding frozen GMM reference proximity' "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" --selection-strategy gmm --overwrite
    run_stage 'Rebuilding saved experiment proximity' "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" --selection-strategy gmm --overwrite
    if ((OVERWRITE)); then THEORY_ARGUMENTS+=(--recompute-experiments); fi
    run_stage 'Measuring four-stage mechanism suite and plotting saved scalars' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --device "$DEVICE"
}

cd "$PROJECT_ROOT"
PAIR_TOTAL=${#MODEL_SCHEDULER_PAIRS[@]}
printf 'Experiment matrix: %s model/scheduler pairs, center %s\n' "$PAIR_TOTAL" "$CENTER"
RUN_CONTEXT="matrix preflight"
trap 'status=$?; printf "run_all.sh: pipeline failed for %s (exit %s)\n" "$RUN_CONTEXT" "$status" >&2; exit "$status"' ERR
# Validate the entire matrix before any renderer can write a figure.
if ((PLOT_ONLY)); then
    PREFLIGHT_STATUS=0
    PREFLIGHT_FAILED_PAIRS=()
    for pair in "${MODEL_SCHEDULER_PAIRS[@]}"; do
        printf '[Plot preflight] %s / %s\n' "${pair%%:*}" "${pair#*:}"
        if "$PROJECT_ROOT/theory_validation.sh" --model "${pair%%:*}" --scheduler "${pair#*:}" --g "$GUIDANCE_SCALE" --T "$NUM_INFERENCE_STEPS" --N "$NUM_SEEDS" "${THEORY_CENTER_ARGUMENTS[@]}" "${THEORY_FIGURE_ARGUMENTS[@]}" "${THEORY_MEASUREMENT_ARGUMENTS[@]}" --validate-only --validate-proximity; then
            :
        else
            PREFLIGHT_PAIR_STATUS=$?
            if ((PREFLIGHT_STATUS == 0)); then PREFLIGHT_STATUS=$PREFLIGHT_PAIR_STATUS; fi
            PREFLIGHT_FAILED_PAIRS+=("$pair")
        fi
    done
    if ((PREFLIGHT_STATUS)); then
        printf 'run_all.sh: plot preflight failed for %s/%s configurations; no figures were changed:\n' "${#PREFLIGHT_FAILED_PAIRS[@]}" "$PAIR_TOTAL" >&2
        printf '  %s\n' "${PREFLIGHT_FAILED_PAIRS[@]}" >&2
        printf 'Use the repair commands above, or select compatible saved bundles with --model and --scheduler. An unrestricted --plot requires all four configurations.\n' >&2
        exit "$PREFLIGHT_STATUS"
    fi
fi
if ((DOWNLOAD_WEBSTER)); then
    if "$PROJECT_ROOT/download_webster.sh" --direct-workers "$DIRECT_WORKERS" --direct-attempts "$DIRECT_ATTEMPTS" --per-host-concurrency "$PER_HOST_CONCURRENCY"; then
        :
    else
        download_status=$?
        if ((download_status != 2)); then exit "$download_status"; fi
    fi
fi
# Preflight before any generation or analytical work; plot-only is host-only.
if ((!PLOT_ONLY)); then
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python}" -B - "$DEVICE" <<'PYGPU'
import sys
import torch
from utils.models.devices import resolve_devices
try:
    devices = resolve_devices(sys.argv[1])
except (ValueError, RuntimeError) as error:
    raise SystemExit(f"run_all.sh: CUDA preflight failed: {error}") from error
if not devices or any(device.type != "cuda" for device in devices) or getattr(torch.version, "hip", None):
    raise SystemExit("run_all.sh: CUDA is required for computation; CPU/MPS fallback is disabled. Saved --plot remains available.")
print("[Pipeline] Compute devices: " + ", ".join(map(str, devices)), flush=True)
PYGPU
fi
PAIR_INDEX=0
for pair in "${MODEL_SCHEDULER_PAIRS[@]}"; do
    MODEL="${pair%%:*}"; SCHEDULER="${pair#*:}"
    PAIR_INDEX=$((PAIR_INDEX + 1))
    RUN_CONTEXT="model $MODEL / scheduler $SCHEDULER"
    printf '\nModel/scheduler %s/%s: %s / %s\n' "$PAIR_INDEX" "$PAIR_TOTAL" "$MODEL" "$SCHEDULER"
    run_model_scheduler
done
printf 'Experiment matrix complete (%s configurations).\n' "$PAIR_TOTAL"
