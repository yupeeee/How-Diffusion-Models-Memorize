#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL="all"
SCHEDULER="all"
GUIDANCE_SCALE="7.5"
NUM_INFERENCE_STEPS="50"
NUM_SEEDS="20"
NUM_BASELINE_SEEDS="1000"
CENTERING_MODE="both"
SELECTION_STRATEGY="gmm"
EVALUATION_SOURCE="both"
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

Run sdv1/ddim, sdv1/ddpm, sdv2/ddim, and realvis/ddim, each with
zero/mu_hat centering and GMM selection (8 configurations).
Explicit model/scheduler options filter these supported pairs; requests with
no matching pair fail before any stage. --scheduler ddpm selects sdv1 only.
Generation, SSCD, and the shared baseline run once per model/scheduler;
Theorem 1 runs once per model/scheduler because it does not depend on the center.
Forward corruptions vs generated states also runs once, only for compatible
sdv1/DDIM, g=7.5, N=20, T>=9 configurations; other combinations are skipped.

Options:
  --download            Run/resume shared Webster preparation once first
  --plot                Only plot saved proximity/theory results and decoded-state
                        galleries
  --overwrite           Regenerate each shared trajectory/SSCD cache, Theorem 1
                        cache, and any required baseline once per model/scheduler;
                        also rebuild the compatible forward/state experiment cache
  --model MODEL         sdv1, sdv2, realvis, or all (default: all three)
  --scheduler NAME      ddim, ddpm, or all (default: all supported pairs)
  --g FLOAT             Classifier-free guidance scale (default: 7.5)
  --T INTEGER           Inference steps per cached trajectory (default: 50)
  --N INTEGER           Experiment seeds 0..N-1; independent selection
                        reference seeds N..2N-1 (default: 20)
  --num-baseline-seeds B
                        Baseline Gaussian seeds N..N+B-1 (default: 1000)
  --use-mu              Run only saved-mu centering for Lemma 2/Corollary 3
  --no-mu               Run only zero centering for Lemma 2/Corollary 3
                        (default: both; these two flags are mutually exclusive)
  --selection-strategy NAME
                        gmm only (default: gmm)
  --evaluation-source SOURCE
                        gaussian, trajectory, or both (default: both)
  --num-loss-seeds DRAWS
                        Conditional-loss draws per pair and timestep (default: 20)
  --loss-seed SEED      Root seed for independent loss draws (default: 0)
  --downscale INTEGER   Preview downscale factor (default: 4)
  --device DEVICE       auto, cpu, mps, cuda, or cuda:N (default: auto)
  --direct-workers INT  Parallel Webster direct-URL workers (default: 24)
  --direct-attempts INT Direct attempts per URL before archives (default: 2)
  --per-host-concurrency INT
                        Concurrent requests to one host (default: 4)
  -h, --help            Show this help message

Models run sequentially in sdv1, sdv2, realvis order, with DDIM before DDPM.
Each model/scheduler's generation and SSCD caches are shared across centers.
Complete published caches are checked and skipped; --overwrite is the only option
that forces their regeneration. Cache-only GMM reference selection and experiment
proximity are rebuilt once per model/scheduler on normal runs.

For mu centering, the shared baseline is estimated once at the actual initial
cached scheduler timestep from B independent Gaussian latents with seeds
N..N+B-1. It evaluates the empty-condition model branch directly, shards across
visible CUDA devices, and never traverses prompts or uses SSCD, categories,
memorization labels, or frozen selection. Zero centering has no baseline
dependency. The same saved mu is shared across selections, Lemma 2, and
Corollary 3; it is never refitted per timestep or prompt.

All three theory experiments evaluate every saved timestep. Gaussian evaluation
uses the same independent initial noises at every saved timestep; trajectory
evaluation uses the exact frozen selection and every included prompt,
experiment seed 0..N-1, and cached timestep. Lemma 2 Gaussian evaluation has
N*T observations; Corollary 3 has P*N*T observations per source. Theorem 1 loss
draws and Gaussian evaluations require inference; trajectory reductions reuse
cached predictions. Corollary 3 uses same-seed target SSCD and guidance scale 7.5;
other guidance values retain an explicit numbered Corollary 3 skip. Both DDIM
and DDPM are supported by all theory experiments and the shared baseline.
Figures plot alpha_t^2/sigma_t^2 logarithmically, lower noise ratios to the left.

The forward/state experiment runs forward_corruptions_generated_states.sh once
outside the centering loop. It uses reference seeds 20..39 to fix its two pairs
and all evaluation seeds 0..19, independently of GMM selection and centering.
It reuses trajectories and SSCD, decoding gallery inputs with only the VAE when
needed. Incompatible model, scheduler, guidance, seed count, or step count gives
an explicit numbered skip. Its --plot mode requires the decoded gallery cache;
it never regenerates trajectories, scores, or decoded images.

--plot never invokes generation, SSCD, selection rebuilding, or baseline
computation. It rebuilds reference and experiment proximity figures from their
frozen/saved CSVs, then plots the theory and compatible forward/state results.
It requires matching saved provenance for the entire requested matrix, including
precomputed decoded galleries for the compatible forward/state experiment.
--plot cannot be combined with --download or --overwrite.
Download tuning options apply only with --download. The PYTHON environment
variable is honored by every wrapper. Any failed stage stops the matrix.
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
        --download) DOWNLOAD_WEBSTER=1; shift; continue ;;
        --plot) PLOT_ONLY=1; shift; continue ;;
        --overwrite) OVERWRITE=1; shift; continue ;;
        --use-mu)
            [[ "$CENTERING_MODE" != "zero" ]] || invalid_value "$1" "cannot be combined with --no-mu"
            CENTERING_MODE="mu_hat"
            shift
            continue
            ;;
        --no-mu)
            [[ "$CENTERING_MODE" != "mu_hat" ]] || invalid_value "$1" "cannot be combined with --use-mu"
            CENTERING_MODE="zero"
            shift
            continue
            ;;
        -h|--help) usage; exit 0 ;;
    esac
    option="${1%%=*}"
    case "$option" in
        --model) destination=MODEL ;;
        --scheduler) destination=SCHEDULER ;;
        --g) destination=GUIDANCE_SCALE ;;
        --T) destination=NUM_INFERENCE_STEPS ;;
        --N) destination=NUM_SEEDS ;;
        --num-baseline-seeds) destination=NUM_BASELINE_SEEDS ;;
        --selection-strategy) destination=SELECTION_STRATEGY ;;
        --evaluation-source) destination=EVALUATION_SOURCE ;;
        --num-loss-seeds) destination=NUM_LOSS_SEEDS ;;
        --loss-seed) destination=LOSS_SEED ;;
        --downscale) destination=DOWNSCALE_FACTOR ;;
        --device) destination=DEVICE ;;
        --direct-workers) destination=DIRECT_WORKERS ;;
        --direct-attempts) destination=DIRECT_ATTEMPTS ;;
        --per-host-concurrency) destination=PER_HOST_CONCURRENCY ;;
        *) printf 'run_all.sh: unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    if [[ "$1" == *=* ]]; then
        value="${1#*=}"
        shift
    else
        (($# >= 2)) || missing_value "$option"
        value="$2"
        shift 2
    fi
    printf -v "$destination" '%s' "$value"
done

case "$MODEL" in
    all|sdv1|sdv2|realvis) ;;
    *) invalid_value "--model" "$MODEL" ;;
esac
case "$SCHEDULER" in
    all|ddim|ddpm) ;;
    *) invalid_value "--scheduler" "$SCHEDULER" ;;
esac

MODEL_SCHEDULER_PAIRS=()
for pair in sdv1:ddim sdv1:ddpm sdv2:ddim realvis:ddim; do
    pair_model="${pair%%:*}"
    pair_scheduler="${pair#*:}"
    if [[ "$MODEL" != "all" && "$MODEL" != "$pair_model" ]]; then
        continue
    fi
    if [[ "$SCHEDULER" != "all" && "$SCHEDULER" != "$pair_scheduler" ]]; then
        continue
    fi
    MODEL_SCHEDULER_PAIRS+=("$pair")
done
if ((${#MODEL_SCHEDULER_PAIRS[@]} == 0)); then
    invalid_value "--model/--scheduler" \
        "$MODEL/$SCHEDULER (no supported pair; choose sdv1/ddim, sdv1/ddpm, sdv2/ddim, or realvis/ddim)"
fi
[[ "$SELECTION_STRATEGY" == "gmm" ]] || invalid_value "--selection-strategy" "$SELECTION_STRATEGY"
case "$CENTERING_MODE" in
    both) CENTERS=(zero mu_hat) ;;
    *) CENTERS=("$CENTERING_MODE") ;;
esac
case "$EVALUATION_SOURCE" in
    gaussian|trajectory|both) ;;
    *) invalid_value "--evaluation-source" "$EVALUATION_SOURCE" ;;
esac

DEVICE="${DEVICE#"${DEVICE%%[![:space:]]*}"}"
DEVICE="${DEVICE%"${DEVICE##*[![:space:]]}"}"
DEVICE="${DEVICE,,}"
if [[ ! "$DEVICE" =~ ^(auto|cpu|mps|cuda|cuda:[0-9]+)$ ]]; then
    invalid_value "--device" "$DEVICE (expected auto, cpu, mps, cuda, or cuda:N)"
fi
normalize_positive_integer "--N" "$NUM_SEEDS" NUM_SEEDS 4611686018427387904
MAX_BASELINE_SEEDS=$((9223372036854775807 - NUM_SEEDS + 1))
normalize_positive_integer \
    "--num-baseline-seeds" "$NUM_BASELINE_SEEDS" NUM_BASELINE_SEEDS "$MAX_BASELINE_SEEDS"
normalize_positive_integer "--num-loss-seeds" "$NUM_LOSS_SEEDS" NUM_LOSS_SEEDS
normalize_nonnegative_integer "--loss-seed" "$LOSS_SEED" LOSS_SEED 9223372036854775807
normalize_positive_integer "--T" "$NUM_INFERENCE_STEPS" NUM_INFERENCE_STEPS
normalize_finite_float "--g" "$GUIDANCE_SCALE" GUIDANCE_SCALE

if ((PLOT_ONLY)); then
    ((!DOWNLOAD_WEBSTER)) || invalid_value "--plot" "cannot be combined with --download"
    ((!OVERWRITE)) || invalid_value "--plot" "cannot be combined with --overwrite"
else
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

EVALUATION_ARGUMENTS=()
if [[ "$EVALUATION_SOURCE" != "both" ]]; then
    EVALUATION_ARGUMENTS=(--evaluation-source "$EVALUATION_SOURCE")
fi
PLOT_ARGUMENTS=()
if ((PLOT_ONLY)); then
    PLOT_ARGUMENTS=(--plot)
fi
CACHE_OVERWRITE_ARGUMENTS=()
if ((OVERWRITE)); then
    CACHE_OVERWRITE_ARGUMENTS=(--overwrite)
fi

run_stage() {
    printf '[%s/%s] %s\n' "$STAGE_INDEX" "$STAGE_TOTAL" "$1"
    shift
    "$@"
    STAGE_INDEX=$((STAGE_INDEX + 1))
}

run_model_scheduler() {
    local STAGE_INDEX=1
    local STAGE_TOTAL=$((2 + 2 * ${#CENTERS[@]}))
    if ((PLOT_ONLY)); then
        STAGE_TOTAL=$((STAGE_TOTAL + 2))
    fi
    local selection="gmm"
    local center
    local COMMON_ARGUMENTS=(
        --model "$MODEL" --scheduler "$SCHEDULER" --g "$GUIDANCE_SCALE"
        --T "$NUM_INFERENCE_STEPS" --N "$NUM_SEEDS"
    )
    local REFERENCE_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" --seed-start "$NUM_SEEDS")
    local EXPERIMENT_ARGUMENTS=("${COMMON_ARGUMENTS[@]}" --seed-start 0)
    local CENTERING_ARGUMENTS=()
    local FORWARD_DEVICE_ARGUMENTS=()
    if [[ "$DEVICE" != "auto" ]]; then
        FORWARD_DEVICE_ARGUMENTS=(--device "$DEVICE")
    fi
    local cache_action="Checking/resuming"
    if ((OVERWRITE)); then
        cache_action="Regenerating"
    fi

    if ((!PLOT_ONLY)); then
        STAGE_TOTAL=$((STAGE_TOTAL + 7))
        run_stage "$cache_action proximity-selection reference (seeds $NUM_SEEDS-$((NUM_SEEDS - 1 + NUM_SEEDS)))" \
            "$PROJECT_ROOT/generate.sh" "${REFERENCE_ARGUMENTS[@]}" \
            --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
        if [[ "$CENTERING_MODE" != "zero" ]]; then
            run_stage "$cache_action shared unconditional baseline ($NUM_BASELINE_SEEDS Gaussian seeds $NUM_SEEDS-$((NUM_SEEDS - 1 + NUM_BASELINE_SEEDS)))" \
                "$PROJECT_ROOT/unconditional_baseline.sh" "${COMMON_ARGUMENTS[@]}" \
                --num-baseline-seeds "$NUM_BASELINE_SEEDS" --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
        else
            run_stage 'Skipping shared unconditional baseline: zero-only centering' true
        fi
        run_stage "$cache_action proximity-selection SSCD (seeds $NUM_SEEDS-$((NUM_SEEDS - 1 + NUM_SEEDS)))" \
            "$PROJECT_ROOT/sscd.sh" "${REFERENCE_ARGUMENTS[@]}" \
            --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
        run_stage "$cache_action experiment trajectories (seeds 0-$((NUM_SEEDS - 1)))" \
            "$PROJECT_ROOT/generate.sh" "${EXPERIMENT_ARGUMENTS[@]}" \
            --device "$DEVICE" --downscale "$DOWNSCALE_FACTOR" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
        run_stage "$cache_action experiment SSCD (seeds 0-$((NUM_SEEDS - 1)))" \
            "$PROJECT_ROOT/sscd.sh" "${EXPERIMENT_ARGUMENTS[@]}" \
            --device "$DEVICE" "${CACHE_OVERWRITE_ARGUMENTS[@]}"
    fi

        RUN_CONTEXT="model $MODEL / scheduler $SCHEDULER / selection $selection"
        if ((PLOT_ONLY)); then
            run_stage "Plotting frozen prompt selection with $selection" \
                "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" \
                --selection-strategy "$selection" --plot
            run_stage "Plotting saved experiment proximity with $selection" \
                "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" \
                --selection-strategy "$selection" --plot
        else
            run_stage "Rebuilding prompt selection with $selection from $NUM_SEEDS-seed reference evidence" \
                "$PROJECT_ROOT/compute_proximity.sh" "${REFERENCE_ARGUMENTS[@]}" \
                --selection-strategy "$selection" --overwrite
            run_stage "Rebuilding cache-only experiment proximity with $selection" \
                "$PROJECT_ROOT/compute_proximity.sh" "${EXPERIMENT_ARGUMENTS[@]}" \
                --selection-strategy "$selection" --overwrite
        fi
        run_stage "Theorem 1 loss–recovery ($selection; independent of centering)" \
            "$PROJECT_ROOT/theorem1_loss_recovery.sh" "${COMMON_ARGUMENTS[@]}" \
            --selection-strategy "$selection" --num-loss-seeds "$NUM_LOSS_SEEDS" \
            --loss-seed "$LOSS_SEED" --device "$DEVICE" \
            "${EVALUATION_ARGUMENTS[@]}" "${PLOT_ARGUMENTS[@]}" \
            "${CACHE_OVERWRITE_ARGUMENTS[@]}"

        RUN_CONTEXT="model $MODEL / scheduler $SCHEDULER / forward corruptions vs generated states"
        if [[ "$MODEL" == "sdv1" && "$SCHEDULER" == "ddim" \
            && "$GUIDANCE_SCALE" == "7.5" && "$NUM_SEEDS" == "20" ]] \
            && decimal_greater_than "$NUM_INFERENCE_STEPS" 8; then
            run_stage "Forward corruptions vs generated states (independent of centering)" \
                "$PROJECT_ROOT/forward_corruptions_generated_states.sh" "${COMMON_ARGUMENTS[@]}" \
                --seed-start 0 --reference-seed-start "$NUM_SEEDS" \
                "${FORWARD_DEVICE_ARGUMENTS[@]}" "${PLOT_ARGUMENTS[@]}" \
                "${CACHE_OVERWRITE_ARGUMENTS[@]}"
        else
            run_stage "Skipping forward corruptions vs generated states: requires --model sdv1, --scheduler ddim, --g 7.5, --N 20, --T >= 9 (received $MODEL, $SCHEDULER, g=$GUIDANCE_SCALE, N=$NUM_SEEDS, T=$NUM_INFERENCE_STEPS)" true
        fi

        for center in "${CENTERS[@]}"; do
            RUN_CONTEXT="model $MODEL / scheduler $SCHEDULER / selection $selection / center $center"
            CENTERING_ARGUMENTS=()
            if [[ "$center" == "mu_hat" ]]; then
                CENTERING_ARGUMENTS=(--use-mu --num-baseline-seeds "$NUM_BASELINE_SEEDS")
            fi
            run_stage "Lemma 2 across noise levels ($selection, $center)" \
                "$PROJECT_ROOT/lemma2_mean_convergence.sh" "${COMMON_ARGUMENTS[@]}" \
                --selection-strategy "$selection" "${CENTERING_ARGUMENTS[@]}" \
                --device "$DEVICE" "${EVALUATION_ARGUMENTS[@]}" "${PLOT_ARGUMENTS[@]}"
            if [[ "$GUIDANCE_SCALE" == "7.5" ]]; then
                run_stage "Corollary 3 CFG amplification ($selection, $center)" \
                    "$PROJECT_ROOT/corollary3_cfg_amplification.sh" "${COMMON_ARGUMENTS[@]}" \
                    --selection-strategy "$selection" "${CENTERING_ARGUMENTS[@]}" \
                    --device "$DEVICE" "${EVALUATION_ARGUMENTS[@]}" "${PLOT_ARGUMENTS[@]}"
            else
                run_stage "Skipping Corollary 3: requires --g 7.5 (received $GUIDANCE_SCALE; $selection, $center)" true
            fi
        done
}

cd "$PROJECT_ROOT"
PAIR_TOTAL=${#MODEL_SCHEDULER_PAIRS[@]}
MATRIX_SIZE=$((PAIR_TOTAL * ${#CENTERS[@]}))
printf 'Experiment matrix: %s model/scheduler pairs x %s centers x 1 selection (%s configurations)\n' \
    "$PAIR_TOTAL" "${#CENTERS[@]}" "$MATRIX_SIZE"

if ((DOWNLOAD_WEBSTER)); then
    printf '[setup] Preparing Webster data once for the entire matrix\n'
    if "$PROJECT_ROOT/download_webster.sh" \
        --direct-workers "$DIRECT_WORKERS" --direct-attempts "$DIRECT_ATTEMPTS" \
        --per-host-concurrency "$PER_HOST_CONCURRENCY"; then
        :
    else
        download_status=$?
        if ((download_status != 2)); then
            exit "$download_status"
        fi
    fi
elif ((!PLOT_ONLY)); then
    printf 'Webster data preparation skipped; pass --download to run it.\n'
fi

trap 'status=$?; printf "run_all.sh: pipeline failed for %s (exit %s)\n" "$RUN_CONTEXT" "$status" >&2; exit "$status"' ERR
PAIR_INDEX=0
for pair in "${MODEL_SCHEDULER_PAIRS[@]}"; do
    MODEL="${pair%%:*}"
    SCHEDULER="${pair#*:}"
    PAIR_INDEX=$((PAIR_INDEX + 1))
    RUN_CONTEXT="model $MODEL / scheduler $SCHEDULER"
    printf '\nModel/scheduler %s/%s: %s / %s\n' "$PAIR_INDEX" "$PAIR_TOTAL" "$MODEL" "$SCHEDULER"
    run_model_scheduler
done
printf 'Experiment matrix complete (%s configurations).\n' "$MATRIX_SIZE"
