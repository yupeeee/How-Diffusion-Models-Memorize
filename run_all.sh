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
INCLUDE_DIAGNOSTICS=0
FIGURE_SUITE=""
TARGET_ERROR_TOLERANCE=""

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Run sdv1/ddim, sdv1/ddpm, sdv2/ddim, and realvis/ddim.
Explicit model/scheduler options filter these supported pairs; requests with
no matching pair fail before any stage. --scheduler ddpm selects sdv1 only.
Protected generation, SSCD, and GMM proximity retain their existing defaults.
Theory reduces existing caches once per model/scheduler, then reloads scalar
outputs to render the registered semantic figures. No theory stage performs inference.

Options:
  --download            Run/resume shared Webster preparation once first
  --plot                Render saved proximity and theory scalar outputs only
  --figure-suite NAME   main or candidates; omitted reloads the saved suite
  --include-diagnostics Also render saved injection and applicable terminal terms
  --target-error-tolerance FLOAT
                        Independently supplied raw latent L2 tolerance (optional)
  --recompute-experiments
                        Rebuild theory from existing protected caches only;
                        bypass download, generation, SSCD and proximity rebuilding
  --overwrite           Explicitly regenerate protected generation/SSCD caches
                        and rebuild derived theory (normal pipeline only)
  --model MODEL         sdv1, sdv2, realvis, or all (default: all three)
  --scheduler NAME      ddim, ddpm, or all (default: all supported pairs)
  --g FLOAT             Guidance scale (default: 7.5; finite values supported)
  --T INTEGER           Inference steps (default: 50)
  --N INTEGER           Experiment seeds 0..N-1; reference N..2N-1 (default: 20)
  --center NAME         reference-initial (default), zero, or cached-baseline
  --cached-baseline PATH Existing independent baseline for cached-baseline center
  --use-mu              Deprecated alias for --center cached-baseline;
                        requires --cached-baseline, never estimates a new center
  --no-mu               Deprecated alias for --center zero
  --selection-strategy NAME
                        gmm only (default: gmm)
  --downscale INTEGER   Upstream preview downscale factor (default: 4)
  --device DEVICE       auto, cpu, mps, cuda, or cuda:N (default: auto)
                        auto uses all visible CUDA GPUs for generation and theory;
                        theory falls back to CPU when CUDA is unavailable
  --direct-workers INT  Webster direct-URL workers (default: 24)
  --direct-attempts INT Direct attempts per URL (default: 2)
  --per-host-concurrency INT Concurrent requests per host (default: 4)
  -h, --help            Show help

The default center is the fixed mean of initial unconditional clean estimates
from unique reference seeds. It is not the known training-distribution mean.
All selected experiment seeds remain in the analysis, including failed recovery.

--plot validates every requested scalar bundle and saved proximity metadata
before writing figures. It never reads raw trajectory tensors, evaluates models,
fits selection, decodes images, or modifies numerical logs. Copied scalar bundles
can also be plotted using theory_validation.sh --bundle PATH --plot.
--plot cannot be combined with --download, --overwrite, or --recompute-experiments.
--recompute-experiments cannot be combined with --download or --overwrite.
Fresh Gaussian sweeps and independent loss draws are legacy-only utilities;
--evaluation-source, --num-loss-seeds, --loss-seed, and --num-baseline-seeds error.
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
        --include-diagnostics) INCLUDE_DIAGNOSTICS=1; shift; continue ;;
        --overwrite) OVERWRITE=1; shift; continue ;;
        --use-mu) set_center cached-baseline; shift; continue ;;
        --no-mu) set_center zero; shift; continue ;;
        -h|--help) usage; exit 0 ;;
    esac
    option="${1%%=*}"
    case "$option" in
        --figure-suite)
            if [[ "$1" == *=* ]]; then value="${1#*=}"; else (($# >= 2)) || missing_value "$option"; value="$2"; shift; fi
            [[ "$value" == main || "$value" == candidates ]] || invalid_value "$option" "$value (expected main or candidates)"
            FIGURE_SUITE="$value"; shift; continue ;;
        --model) destination=MODEL ;;
        --scheduler) destination=SCHEDULER ;;
        --g) destination=GUIDANCE_SCALE ;;
        --T) destination=NUM_INFERENCE_STEPS ;;
        --N) destination=NUM_SEEDS ;;
        --center) destination=CENTER_VALUE ;;
        --cached-baseline) destination=CACHED_BASELINE ;;
        --target-error-tolerance) destination=TARGET_ERROR_TOLERANCE ;;
        --selection-strategy) destination=SELECTION_STRATEGY ;;
        --downscale) destination=DOWNSCALE_FACTOR ;;
        --device) destination=DEVICE ;;
        --direct-workers) destination=DIRECT_WORKERS ;;
        --direct-attempts) destination=DIRECT_ATTEMPTS ;;
        --per-host-concurrency) destination=PER_HOST_CONCURRENCY ;;
        --evaluation-source|--num-loss-seeds|--loss-seed|--num-baseline-seeds)
            invalid_value "$option" "removed from cache-only theory; invoke the explicit legacy experiment wrapper for independent inference" ;;
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
case "$CENTER" in reference-initial|zero|cached-baseline) ;; *) invalid_value "--center" "$CENTER" ;; esac
[[ "$SELECTION_STRATEGY" == "gmm" ]] || invalid_value "--selection-strategy" "$SELECTION_STRATEGY"
if [[ "$CENTER" == cached-baseline && -z "$CACHED_BASELINE" ]]; then
    invalid_value "--cached-baseline" "required for the existing independent baseline center"
fi
if [[ "$CENTER" != cached-baseline && -n "$CACHED_BASELINE" ]]; then
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
if ((!PLOT_ONLY && !RECOMPUTE)); then
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
THEORY_CENTER_ARGUMENTS=(--center "$CENTER")
if [[ -n "$TARGET_ERROR_TOLERANCE" ]]; then
    normalize_finite_float --target-error-tolerance "$TARGET_ERROR_TOLERANCE" TARGET_ERROR_TOLERANCE
    [[ "$TARGET_ERROR_TOLERANCE" != -* ]] || invalid_value --target-error-tolerance "must be nonnegative raw latent L2"
    THEORY_CENTER_ARGUMENTS+=(--target-error-tolerance "$TARGET_ERROR_TOLERANCE")
fi
THEORY_FIGURE_ARGUMENTS=()
if ((INCLUDE_DIAGNOSTICS)); then THEORY_FIGURE_ARGUMENTS=(--include-diagnostics); fi
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
    local THEORY_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" "${THEORY_CENTER_ARGUMENTS[@]}" "${THEORY_FIGURE_ARGUMENTS[@]}")
    if ((PLOT_ONLY)); then
        STAGE_TOTAL=3
        run_stage 'Plotting frozen reference proximity' "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" --selection-strategy gmm --plot
        run_stage 'Plotting saved experiment proximity' "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" --selection-strategy gmm --plot
        run_stage 'Plotting saved theory scalars' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --plot
        return
    fi
    if ((RECOMPUTE)); then
        STAGE_TOTAL=1
        run_stage 'Rebuilding theory from protected caches only' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --device "$DEVICE" --recompute-experiments
        return
    fi
    run_stage 'Checking/resuming reference trajectories' "$PROJECT_ROOT/generate.sh" "${REFERENCE_ARGUMENTS[@]}" --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming reference SSCD' "$PROJECT_ROOT/sscd.sh" "${REFERENCE_ARGUMENTS[@]}" --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming experiment trajectories' "$PROJECT_ROOT/generate.sh" "${EXPERIMENT_ARGUMENTS[@]}" --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Checking/resuming experiment SSCD' "$PROJECT_ROOT/sscd.sh" "${EXPERIMENT_ARGUMENTS[@]}" --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    run_stage 'Rebuilding frozen GMM reference proximity' "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" --selection-strategy gmm --overwrite
    run_stage 'Rebuilding saved experiment proximity' "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" --selection-strategy gmm --overwrite
    if ((OVERWRITE)); then THEORY_ARGUMENTS+=(--recompute-experiments); fi
    run_stage 'Reducing theory once and plotting saved scalars' "$PROJECT_ROOT/theory_validation.sh" "${THEORY_ARGUMENTS[@]}" --device "$DEVICE"
}

cd "$PROJECT_ROOT"
PAIR_TOTAL=${#MODEL_SCHEDULER_PAIRS[@]}
printf 'Experiment matrix: %s model/scheduler pairs, center %s\n' "$PAIR_TOTAL" "$CENTER"
RUN_CONTEXT="matrix preflight"
trap 'status=$?; printf "run_all.sh: pipeline failed for %s (exit %s)\n" "$RUN_CONTEXT" "$status" >&2; exit "$status"' ERR
# Validate the entire matrix before any renderer can write a figure.
if ((PLOT_ONLY)); then
    for pair in "${MODEL_SCHEDULER_PAIRS[@]}"; do
        "$PROJECT_ROOT/theory_validation.sh" --model "${pair%%:*}" --scheduler "${pair#*:}" --g "$GUIDANCE_SCALE" --T "$NUM_INFERENCE_STEPS" --N "$NUM_SEEDS" "${THEORY_CENTER_ARGUMENTS[@]}" "${THEORY_FIGURE_ARGUMENTS[@]}" --validate-only --validate-proximity
    done
fi
if ((DOWNLOAD_WEBSTER)); then
    if "$PROJECT_ROOT/download_webster.sh" --direct-workers "$DIRECT_WORKERS" --direct-attempts "$DIRECT_ATTEMPTS" --per-host-concurrency "$PER_HOST_CONCURRENCY"; then
        :
    else
        download_status=$?
        if ((download_status != 2)); then exit "$download_status"; fi
    fi
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
