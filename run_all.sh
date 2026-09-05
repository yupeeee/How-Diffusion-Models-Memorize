#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGUMENTS=("$@")

MODEL="sdv1"
MODEL_OPTION_PROVIDED=0
SCHEDULER="ddim"
GUIDANCE_SCALE="7.5"
NUM_INFERENCE_STEPS="50"
NUM_SEEDS="20"
SELECTION_STRATEGY="gmm"
NUM_LOSS_SEEDS="20"
LOSS_SEED="0"
DOWNSCALE_FACTOR="4"
DEVICE="auto"
DIRECT_WORKERS="${WEBSTER_DIRECT_WORKERS:-24}"
DIRECT_ATTEMPTS="${WEBSTER_DIRECT_ATTEMPTS:-2}"
PER_HOST_CONCURRENCY="${WEBSTER_PER_HOST_CONCURRENCY:-4}"
DOWNLOAD_WEBSTER=0
PLOT_ONLY=0
OVERWRITE=0

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Build proximity-rule prompt selection from an independent reference, then run
the matching experiment and Theorem 1 analysis. Webster data preparation is
opt-in.

Options:
  --download            Run/resume Webster data preparation first
  --plot                Plot Theorem 1 from its saved CSV; run no computation
  --overwrite           Regenerate trajectories and SSCD caches
  --model MODEL         Run only sdv1, sdv2, or realvis (default: all three)
  --scheduler NAME      ddim or ddpm (default: ddim)
  --g FLOAT             Classifier-free guidance scale (default: 7.5)
  --T INTEGER           Number of inference steps (default: 50)
  --N INTEGER           Seeds per pool: experiment 0..N-1, reference N..2N-1
                        (default: 20)
  --selection-strategy NAME
                        gmm, gmm-evidence, or spearman (default: gmm)
  --num-loss-seeds K    Conditional-loss draws per pair (default: 20)
  --loss-seed SEED      Root seed for independent loss draws (default: 0)
  --downscale INTEGER   Preview downscale factor (default: 4)
  --device DEVICE       auto, cpu, mps, cuda, or cuda:N (default: auto)
  --direct-workers INT  Parallel Webster direct-URL workers (default: 24)
  --direct-attempts INT Direct attempts per URL before archives (default: 2)
  --per-host-concurrency INT
                        Concurrent requests to one host (default: 4)
  -h, --help            Show this help message

Without --model, the pipeline runs sequentially for sdv1, sdv2, and realvis.
The selection reference uses the requested model, scheduler, guidance, steps,
and seed count. Its seeds are N..2N-1; the experiment uses 0..N-1, so the two
caches stay disjoint while all other sampling variables remain aligned.
GMM, GMM-evidence, and Spearman selections have separate frozen and
experiment-output paths, but reuse the same generation and SSCD caches.
Complete published trajectory caches are checked structurally and skipped;
--overwrite is the only option that forces their regeneration. SSCD caches
follow the same rule. The reference selection and experiment proximity outputs
are derived from those validated caches and rebuilt on every normal pipeline
run; this does not rerun diffusion or SSCD inference. Theorem 1 receives no
overwrite option.
Download tuning options have effect only when --download is present. The
Theorem 1 stage uses the same requested selection strategy, scheduler,
guidance, steps, and seed count. --num-loss-seeds and --loss-seed configure
only its independent conditional-loss draws. The PYTHON environment variable
is honored by each wrapper.
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

while (($# > 0)); do
    case "$1" in
        --download)
            DOWNLOAD_WEBSTER=1
            shift
            ;;
        --plot)
            PLOT_ONLY=1
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --model)
            (($# >= 2)) || missing_value "$1"
            MODEL="$2"
            MODEL_OPTION_PROVIDED=1
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            MODEL_OPTION_PROVIDED=1
            shift
            ;;
        --scheduler)
            (($# >= 2)) || missing_value "$1"
            SCHEDULER="$2"
            shift 2
            ;;
        --scheduler=*)
            SCHEDULER="${1#*=}"
            shift
            ;;
        --g)
            (($# >= 2)) || missing_value "$1"
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --g=*)
            GUIDANCE_SCALE="${1#*=}"
            shift
            ;;
        --T)
            (($# >= 2)) || missing_value "$1"
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --T=*)
            NUM_INFERENCE_STEPS="${1#*=}"
            shift
            ;;
        --N)
            (($# >= 2)) || missing_value "$1"
            NUM_SEEDS="$2"
            shift 2
            ;;
        --N=*)
            NUM_SEEDS="${1#*=}"
            shift
            ;;
        --selection-strategy)
            (($# >= 2)) || missing_value "$1"
            SELECTION_STRATEGY="$2"
            shift 2
            ;;
        --selection-strategy=*)
            SELECTION_STRATEGY="${1#*=}"
            shift
            ;;
        --num-loss-seeds)
            (($# >= 2)) || missing_value "$1"
            NUM_LOSS_SEEDS="$2"
            shift 2
            ;;
        --num-loss-seeds=*)
            NUM_LOSS_SEEDS="${1#*=}"
            shift
            ;;
        --loss-seed)
            (($# >= 2)) || missing_value "$1"
            LOSS_SEED="$2"
            shift 2
            ;;
        --loss-seed=*)
            LOSS_SEED="${1#*=}"
            shift
            ;;
        --downscale)
            (($# >= 2)) || missing_value "$1"
            DOWNSCALE_FACTOR="$2"
            shift 2
            ;;
        --downscale=*)
            DOWNSCALE_FACTOR="${1#*=}"
            shift
            ;;
        --device)
            (($# >= 2)) || missing_value "$1"
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --direct-workers)
            (($# >= 2)) || missing_value "$1"
            DIRECT_WORKERS="$2"
            shift 2
            ;;
        --direct-workers=*)
            DIRECT_WORKERS="${1#*=}"
            shift
            ;;
        --direct-attempts)
            (($# >= 2)) || missing_value "$1"
            DIRECT_ATTEMPTS="$2"
            shift 2
            ;;
        --direct-attempts=*)
            DIRECT_ATTEMPTS="${1#*=}"
            shift
            ;;
        --per-host-concurrency)
            (($# >= 2)) || missing_value "$1"
            PER_HOST_CONCURRENCY="$2"
            shift 2
            ;;
        --per-host-concurrency=*)
            PER_HOST_CONCURRENCY="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'run_all.sh: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SELECTION_STRATEGY" in
    gmm|gmm-evidence|spearman) ;;
    *)
        invalid_value \
            "--selection-strategy" \
            "$SELECTION_STRATEGY (expected gmm, gmm-evidence, or spearman)"
        ;;
esac

if ((!MODEL_OPTION_PROVIDED)); then
    ALL_MODELS=(sdv1 sdv2 realvis)
    ARGUMENTS_WITHOUT_DOWNLOAD=()
    for argument in "${ORIGINAL_ARGUMENTS[@]}"; do
        if [[ "$argument" != "--download" ]]; then
            ARGUMENTS_WITHOUT_DOWNLOAD+=("$argument")
        fi
    done

    for position in "${!ALL_MODELS[@]}"; do
        model="${ALL_MODELS[$position]}"
        child_arguments=("${ORIGINAL_ARGUMENTS[@]}")
        if ((position > 0 && DOWNLOAD_WEBSTER)); then
            child_arguments=("${ARGUMENTS_WITHOUT_DOWNLOAD[@]}")
        fi
        if "$PROJECT_ROOT/run_all.sh" --model "$model" "${child_arguments[@]}"; then
            :
        else
            status=$?
            printf 'run_all.sh: pipeline failed for model %s (exit %s)\n' "$model" "$status" >&2
            exit "$status"
        fi
    done
    printf 'All-model pipeline complete.\n'
    exit 0
fi

case "$MODEL" in
    sdv1|sdv2|realvis) ;;
    *)
        printf 'run_all.sh: unsupported model: %s\n' "$MODEL" >&2
        exit 2
        ;;
esac

DEVICE="${DEVICE#"${DEVICE%%[![:space:]]*}"}"
DEVICE="${DEVICE%"${DEVICE##*[![:space:]]}"}"
DEVICE="${DEVICE,,}"
if [[ ! "$DEVICE" =~ ^(auto|cpu|mps|cuda|cuda:[0-9]+)$ ]]; then
    invalid_value "--device" "$DEVICE (expected auto, cpu, mps, cuda, or cuda:N)"
fi

normalize_positive_integer \
    "--N" "$NUM_SEEDS" NUM_SEEDS 4611686018427387904
normalize_positive_integer \
    "--num-loss-seeds" "$NUM_LOSS_SEEDS" NUM_LOSS_SEEDS
normalize_nonnegative_integer \
    "--loss-seed" "$LOSS_SEED" LOSS_SEED 9223372036854775807

case "$SCHEDULER" in
    ddim|ddpm) ;;
    *)
        printf 'run_all.sh: unsupported scheduler: %s\n' "$SCHEDULER" >&2
        exit 2
        ;;
esac

normalize_finite_float "--g" "$GUIDANCE_SCALE" GUIDANCE_SCALE

normalize_positive_integer "--T" "$NUM_INFERENCE_STEPS" NUM_INFERENCE_STEPS

if ((PLOT_ONLY)); then
    if ((DOWNLOAD_WEBSTER)); then
        invalid_value "--plot" "cannot be combined with --download"
    fi
    if ((OVERWRITE)); then
        invalid_value "--plot" "cannot be combined with --overwrite"
    fi
    cd "$PROJECT_ROOT"
    printf '[1/1] Plotting Theorem 1 loss–recovery from saved results\n'
    exec "$PROJECT_ROOT/theorem1_loss_recovery.sh" \
        --model "$MODEL" \
        --scheduler "$SCHEDULER" \
        --g "$GUIDANCE_SCALE" \
        --T "$NUM_INFERENCE_STEPS" \
        --N "$NUM_SEEDS" \
        --selection-strategy "$SELECTION_STRATEGY" \
        --num-loss-seeds "$NUM_LOSS_SEEDS" \
        --loss-seed "$LOSS_SEED" \
        --device "$DEVICE" \
        --plot
fi
normalize_positive_integer "--downscale" "$DOWNSCALE_FACTOR" DOWNSCALE_FACTOR
if ((DOWNLOAD_WEBSTER)); then
    normalize_positive_integer "--direct-workers" "$DIRECT_WORKERS" DIRECT_WORKERS 64
    normalize_positive_integer "--direct-attempts" "$DIRECT_ATTEMPTS" DIRECT_ATTEMPTS 5
    normalize_positive_integer \
        "--per-host-concurrency" "$PER_HOST_CONCURRENCY" PER_HOST_CONCURRENCY 8
    if decimal_greater_than "$PER_HOST_CONCURRENCY" "$DIRECT_WORKERS"; then
        invalid_value \
            "--per-host-concurrency" \
            "$PER_HOST_CONCURRENCY (cannot exceed --direct-workers $DIRECT_WORKERS)"
    fi
fi

REFERENCE_ARGUMENTS=(
    --model "$MODEL"
    --scheduler "$SCHEDULER"
    --g "$GUIDANCE_SCALE"
    --T "$NUM_INFERENCE_STEPS"
    --N "$NUM_SEEDS"
    --seed-start "$NUM_SEEDS"
)

EXPERIMENT_ARGUMENTS=(
    --model "$MODEL"
    --scheduler "$SCHEDULER"
    --g "$GUIDANCE_SCALE"
    --T "$NUM_INFERENCE_STEPS"
    --N "$NUM_SEEDS"
    --seed-start 0
)

SELECTION_ARGUMENTS=(
    --selection-strategy "$SELECTION_STRATEGY"
)

CACHE_OVERWRITE_ARGUMENTS=()
if ((OVERWRITE)); then
    CACHE_OVERWRITE_ARGUMENTS=(--overwrite)
fi

cd "$PROJECT_ROOT"

REFERENCE_LAST_SEED=$((NUM_SEEDS - 1 + NUM_SEEDS))
REFERENCE_STAGE_COUNT=3

STAGE_INDEX=1
STAGE_TOTAL=$((4 + REFERENCE_STAGE_COUNT))
if ((DOWNLOAD_WEBSTER)); then
    STAGE_TOTAL=$((STAGE_TOTAL + 1))
    printf '[%s/%s] Preparing Webster data\n' "$STAGE_INDEX" "$STAGE_TOTAL"
    if "$PROJECT_ROOT/download_webster.sh" \
        --direct-workers "$DIRECT_WORKERS" \
        --direct-attempts "$DIRECT_ATTEMPTS" \
        --per-host-concurrency "$PER_HOST_CONCURRENCY"; then
        :
    else
        download_status=$?
        if ((download_status != 2)); then
            exit "$download_status"
        fi
    fi
    STAGE_INDEX=$((STAGE_INDEX + 1))
else
    printf 'Webster data preparation skipped; pass --download to run it.\n'
fi

if ((OVERWRITE)); then
    printf '[%s/%s] Regenerating proximity-selection reference (seeds %s-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$NUM_SEEDS" "$REFERENCE_LAST_SEED"
else
    printf '[%s/%s] Checking/resuming proximity-selection reference (seeds %s-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$NUM_SEEDS" "$REFERENCE_LAST_SEED"
fi
"$PROJECT_ROOT/generate.sh" \
    "${REFERENCE_ARGUMENTS[@]}" \
    --device "$DEVICE" \
    --downscale "$DOWNSCALE_FACTOR" \
    "${CACHE_OVERWRITE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

if ((OVERWRITE)); then
    printf '[%s/%s] Computing proximity-selection SSCD (seeds %s-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$NUM_SEEDS" "$REFERENCE_LAST_SEED"
else
    printf '[%s/%s] Checking/resuming proximity-selection SSCD (seeds %s-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$NUM_SEEDS" "$REFERENCE_LAST_SEED"
fi
"$PROJECT_ROOT/sscd.sh" \
    "${REFERENCE_ARGUMENTS[@]}" \
    --device "$DEVICE" \
    "${CACHE_OVERWRITE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Rebuilding prompt selection with %s from %s-seed reference evidence\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$SELECTION_STRATEGY" "$NUM_SEEDS"
"$PROJECT_ROOT/compute_proximity.sh" \
    "${REFERENCE_ARGUMENTS[@]}" "${SELECTION_ARGUMENTS[@]}" \
    --overwrite
STAGE_INDEX=$((STAGE_INDEX + 1))

if ((OVERWRITE)); then
    printf '[%s/%s] Regenerating experiment trajectories (seeds 0-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
else
    printf '[%s/%s] Checking/resuming experiment trajectories (seeds 0-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
fi
"$PROJECT_ROOT/generate.sh" \
    "${EXPERIMENT_ARGUMENTS[@]}" \
    --device "$DEVICE" \
    --downscale "$DOWNSCALE_FACTOR" \
    "${CACHE_OVERWRITE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

if ((OVERWRITE)); then
    printf '[%s/%s] Recomputing experiment SSCD (seeds 0-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
else
    printf '[%s/%s] Checking/resuming experiment SSCD (seeds 0-%s)\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
fi
"$PROJECT_ROOT/sscd.sh" \
    "${EXPERIMENT_ARGUMENTS[@]}" \
    --device "$DEVICE" \
    "${CACHE_OVERWRITE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Rebuilding cache-only experiment proximity with %s\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$SELECTION_STRATEGY"
"$PROJECT_ROOT/compute_proximity.sh" \
    "${EXPERIMENT_ARGUMENTS[@]}" "${SELECTION_ARGUMENTS[@]}" \
    --overwrite
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Running Theorem 1 loss–recovery experiment\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL"
"$PROJECT_ROOT/theorem1_loss_recovery.sh" \
    --model "$MODEL" \
    --scheduler "$SCHEDULER" \
    --g "$GUIDANCE_SCALE" \
    --T "$NUM_INFERENCE_STEPS" \
    --N "$NUM_SEEDS" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --num-loss-seeds "$NUM_LOSS_SEEDS" \
    --loss-seed "$LOSS_SEED" \
    --device "$DEVICE"

printf 'Pipeline complete.\n'
