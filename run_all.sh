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
DOWNSCALE_FACTOR="4"
DIRECT_WORKERS="${WEBSTER_DIRECT_WORKERS:-24}"
DIRECT_ATTEMPTS="${WEBSTER_DIRECT_ATTEMPTS:-2}"
PER_HOST_CONCURRENCY="${WEBSTER_PER_HOST_CONCURRENCY:-4}"
DOWNLOAD_WEBSTER=0
PLOT_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Build the fixed selection reference, proximity analysis, and compatible
Theorem 1 experiment. Webster data preparation is opt-in.

Options:
  --download            Run/resume Webster data preparation first
  --plot                Plot Theorem 1 from its saved CSV; run no computation
  --model MODEL         Run only sdv1, sdv2, or realvis (default: all three)
  --scheduler NAME      ddim or ddpm (default: ddim)
  --g FLOAT             Classifier-free guidance scale (default: 7.5)
  --T INTEGER           Number of inference steps (default: 50)
  --N INTEGER           Experiment seeds 0..N-1; must be 1..20 (default: 20)
  --downscale INTEGER   Preview downscale factor (default: 4)
  --direct-workers INT  Parallel Webster direct-URL workers (default: 24)
  --direct-attempts INT Direct attempts per URL before archives (default: 2)
  --per-host INT        Concurrent requests to one host (default: 4)
  -h, --help            Show this help message

Without --model, the pipeline runs sequentially for sdv1, sdv2, and realvis.
The selection reference is always ddim/g7.5/T50 with seeds 20..39. The
experiment always starts at seed 0, so its cache stays disjoint from the
reference cache. Download tuning options have effect only when --download is
present. The Theorem 1 stage runs only for the canonical ddim/g7.5/T50/N20
experiment. The PYTHON environment variable is honored by each wrapper.
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
        --downscale)
            (($# >= 2)) || missing_value "$1"
            DOWNSCALE_FACTOR="$2"
            shift 2
            ;;
        --downscale=*)
            DOWNSCALE_FACTOR="${1#*=}"
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
        --per-host|--per-host-concurrency)
            (($# >= 2)) || missing_value "$1"
            PER_HOST_CONCURRENCY="$2"
            shift 2
            ;;
        --per-host=*|--per-host-concurrency=*)
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

if ((PLOT_ONLY)); then
    if ((DOWNLOAD_WEBSTER)); then
        invalid_value "--plot" "cannot be combined with --download"
    fi
    cd "$PROJECT_ROOT"
    printf '[1/1] Plotting Theorem 1 loss–recovery from saved results\n'
    exec "$PROJECT_ROOT/theorem1_loss_recovery.sh" --model "$MODEL" --plot
fi

case "$SCHEDULER" in
    ddim|ddpm) ;;
    *)
        printf 'run_all.sh: unsupported scheduler: %s\n' "$SCHEDULER" >&2
        exit 2
        ;;
esac

if [[ ! "$GUIDANCE_SCALE" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
    invalid_value "--g" "$GUIDANCE_SCALE (expected a finite float)"
fi

normalize_positive_integer "--T" "$NUM_INFERENCE_STEPS" NUM_INFERENCE_STEPS
normalize_positive_integer "--N" "$NUM_SEEDS" NUM_SEEDS 20
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

RUN_THEOREM1=0
if [[ "$SCHEDULER" == "ddim" \
    && "$NUM_INFERENCE_STEPS" == "50" \
    && "$NUM_SEEDS" == "20" ]] \
    && awk -v value="$GUIDANCE_SCALE" \
        'BEGIN { exit !((value + 0.0) == 7.5) }'; then
    RUN_THEOREM1=1
fi

REFERENCE_ARGUMENTS=(
    --model "$MODEL"
    --scheduler ddim
    --g 7.5
    --T 50
    --N 20
    --seed-start 20
)

EXPERIMENT_ARGUMENTS=(
    --model "$MODEL"
    --scheduler "$SCHEDULER"
    --g "$GUIDANCE_SCALE"
    --T "$NUM_INFERENCE_STEPS"
    --N "$NUM_SEEDS"
    --seed-start 0
)

cd "$PROJECT_ROOT"

STAGE_INDEX=1
STAGE_TOTAL=$((6 + RUN_THEOREM1))
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

printf '[%s/%s] Generating selection reference (seeds 20-39)\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL"
"$PROJECT_ROOT/generate.sh" \
    "${REFERENCE_ARGUMENTS[@]}" \
    --downscale "$DOWNSCALE_FACTOR"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Computing selection-reference SSCD (seeds 20-39)\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL"
"$PROJECT_ROOT/sscd.sh" "${REFERENCE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Building frozen selection from the reference cache\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL"
"$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Generating experiment trajectories (seeds 0-%s)\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
"$PROJECT_ROOT/generate.sh" \
    "${EXPERIMENT_ARGUMENTS[@]}" \
    --downscale "$DOWNSCALE_FACTOR"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Computing experiment SSCD (seeds 0-%s)\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL" "$((NUM_SEEDS - 1))"
"$PROJECT_ROOT/sscd.sh" "${EXPERIMENT_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

printf '[%s/%s] Computing cache-only experiment proximity\n' \
    "$STAGE_INDEX" "$STAGE_TOTAL"
"$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}"
STAGE_INDEX=$((STAGE_INDEX + 1))

if ((RUN_THEOREM1)); then
    printf '[%s/%s] Running Theorem 1 loss–recovery experiment\n' \
        "$STAGE_INDEX" "$STAGE_TOTAL"
    "$PROJECT_ROOT/theorem1_loss_recovery.sh" --model "$MODEL"
else
    printf '%s\n' \
        'Theorem 1 loss–recovery skipped; it requires ddim/g7.5/T50/N20.'
fi

printf 'Pipeline complete.\n'
