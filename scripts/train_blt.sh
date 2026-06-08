#!/bin/bash
#SBATCH --job-name=blt-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# A100 partition rejects >8 CPU cores per 1 GPU; keep this conservative.
#SBATCH --cpus-per-task=4
#SBATCH --time=01:55:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"

if [[ "$PWD" != /scratch/"$USER"/* ]]; then
    echo "Error: Neuron jobs must be submitted from /scratch/$USER."
    echo "Current directory: $PWD"
    exit 1
fi

mkdir -p logs outputs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Submit Dir: $SLURM_SUBMIT_DIR"
echo "Partition: amd_a100nv_8"
NUM_GPUS="${NUM_GPUS:-1}"
echo "Requested GPUs: $NUM_GPUS"
echo "Requested CPU cores per task: ${SLURM_CPUS_PER_TASK:-4}"

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

# Python 환경 설정:
# - 기본 conda 환경은 phdq
# - 필요할 때만 PYTHON_BIN=/path/to/python 또는 CONDA_ENV=<env>로 override
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-phdq}"
if [[ -n "${CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
        PYTHON_BIN="python"
    else
        echo "Error: CONDA_ENV=$CONDA_ENV was set but conda is not available."
        exit 1
    fi
fi

echo "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
if ! "$PYTHON_BIN" -c "import torch; import numpy; print('Torch:', torch.__version__); print('NumPy:', numpy.__version__)" >/tmp/phdq_blt_torch_check.txt 2>&1; then
    cat /tmp/phdq_blt_torch_check.txt
    echo "Error: PyTorch/NumPy is not installed in the selected Python environment."
    echo "Install it first, or submit with PYTHON_BIN=/path/to/python or CONDA_ENV=<env_name>."
    echo "Default CONDA_ENV is phdq."
    exit 1
fi
cat /tmp/phdq_blt_torch_check.txt

if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Error: NUM_GPUS must be >= 1, got $NUM_GPUS."
    exit 1
fi

ACTUAL_GPUS="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
echo "Visible CUDA device count: $ACTUAL_GPUS"
if [[ "$ACTUAL_GPUS" -lt "$NUM_GPUS" ]]; then
    echo "Error: NUM_GPUS=$NUM_GPUS but only $ACTUAL_GPUS CUDA device(s) are visible."
    echo "Submit with: NUM_GPUS=$NUM_GPUS sbatch --gres=gpu:$NUM_GPUS --cpus-per-task=16 scripts/train_blt.sh"
    exit 1
fi
if [[ "$NUM_GPUS" -gt 1 && "${SLURM_NNODES:-1}" -ne 1 ]]; then
    echo "Error: this script supports single-node DDP only. SLURM_NNODES=${SLURM_NNODES:-1}"
    exit 1
fi

REFERENCE_BLT_DIR="${REFERENCE_BLT_DIR:-$PROJECT_HOME/reference_code/blt}"
if [[ ! -d "$REFERENCE_BLT_DIR/bytelatent" ]]; then
    echo "Error: reference BLT code not found: $REFERENCE_BLT_DIR/bytelatent"
    exit 1
fi

if ! "$PYTHON_BIN" - "$REFERENCE_BLT_DIR" <<'PY' >/tmp/phdq_blt_ref_check.txt 2>&1
import sys
from pathlib import Path

ref = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ref))
from bytelatent.hf import BltTokenizerAndPatcher  # noqa: F401
from bytelatent.model.blt import ByteLatentTransformer  # noqa: F401
from bytelatent.transformer import LMTransformer  # noqa: F401
print("Reference BLT import path:", ref)
PY
then
    cat /tmp/phdq_blt_ref_check.txt
    echo "Error: reference BLT package cannot be imported."
    echo "Install reference_code/blt requirements in the selected conda env."
    exit 1
fi
cat /tmp/phdq_blt_ref_check.txt

DATA_DIR="$PROJECT_HOME/data"
DATASET_TYPE="${DATASET_TYPE:-native}"
DATASET_DIR_NAME="$DATASET_TYPE"
if [[ "$DATASET_TYPE" == "learner" ]]; then
    DATASET_DIR_NAME="korean_learner"
fi

MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/blt_gec}"
BLT_REPO="${BLT_REPO:-facebook/blt-1b}"
ENTROPY_REPO="${ENTROPY_REPO:-facebook/blt-entropy}"
PRECISION="${PRECISION:-bf16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
BLT_NUM_BEAMS="${BLT_NUM_BEAMS:-4}"
MAX_GEN_LEN="${MAX_GEN_LEN:-256}"
EVAL_GENERATION="${EVAL_GENERATION:-0}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
M2_SOURCE_GOLD_PATH="${M2_SOURCE_GOLD_PATH:-}"
RUN_TEST_ON_END="${RUN_TEST_ON_END:-0}"
TEST_ONLY="${TEST_ONLY:-0}"

PREPROCESSED_DIR="$DATA_DIR/Preprocessed/$DATASET_DIR_NAME"
TRAIN_DATA="${TRAIN_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_train.txt}"
VAL_DATA="${VAL_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_val.txt}"
TEST_DATA="${TEST_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_test.txt}"

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -f "$TEST_DATA" ]]; then
    echo "Error: Train/Val/Test data not found."
    echo "DATASET_TYPE=$DATASET_TYPE"
    echo "DATASET_DIR_NAME=$DATASET_DIR_NAME"
    echo "Expected:"
    echo "  $TRAIN_DATA"
    echo "  $VAL_DATA"
    echo "  $TEST_DATA"
    exit 1
fi

checkpoint_is_loadable() {
    local checkpoint_path="$1"
    "$PYTHON_BIN" - "$checkpoint_path" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"Invalid checkpoint {path}: {exc}")
    raise SystemExit(1)

if not isinstance(checkpoint, dict) or "model" not in checkpoint:
    print(f"Invalid checkpoint {path}: missing model state")
    raise SystemExit(1)

print(f"Checkpoint is loadable: {path}")
PY
}

quarantine_checkpoint() {
    local checkpoint_path="$1"
    local quarantined="${checkpoint_path}.corrupt.$(date +%Y%m%d%H%M%S)"
    mv "$checkpoint_path" "$quarantined"
    echo "Moved corrupt checkpoint to: $quarantined"
}

RESUME_CKPT="${RESUME_CKPT-auto}"
RESUME_ARGS=()
if [[ "$RESUME_CKPT" == "auto" ]]; then
    BEST_CKPT="$OUTPUT_DIR/${DATASET_TYPE}/best.ckpt"
    LAST_CKPT="$OUTPUT_DIR/${DATASET_TYPE}/last.ckpt"

    if [[ -f "$BEST_CKPT" ]]; then
        if checkpoint_is_loadable "$BEST_CKPT"; then
            RESUME_ARGS=(--resume_ckpt_path "$BEST_CKPT")
            echo "Auto-best resume enabled: $BEST_CKPT"
        else
            quarantine_checkpoint "$BEST_CKPT"
        fi
    fi

    if [[ ${#RESUME_ARGS[@]} -eq 0 && -f "$LAST_CKPT" ]]; then
        if checkpoint_is_loadable "$LAST_CKPT"; then
            RESUME_ARGS=(--resume_ckpt_path "$LAST_CKPT")
            echo "Auto-best checkpoint not found or invalid; falling back to last checkpoint: $LAST_CKPT"
        else
            quarantine_checkpoint "$LAST_CKPT"
        fi
    fi

    if [[ ${#RESUME_ARGS[@]} -eq 0 ]]; then
        echo "Auto-resume enabled: no loadable checkpoint found; starting fresh."
    fi
elif [[ -n "$RESUME_CKPT" ]]; then
    if [[ ! -f "$RESUME_CKPT" ]]; then
        echo "Error: RESUME_CKPT not found: $RESUME_CKPT"
        exit 1
    fi
    if ! checkpoint_is_loadable "$RESUME_CKPT"; then
        echo "Error: explicit RESUME_CKPT is not loadable: $RESUME_CKPT"
        exit 1
    fi
    RESUME_ARGS=(--resume_ckpt_path "$RESUME_CKPT")
    echo "Resuming from explicit checkpoint: $RESUME_CKPT"
fi

LOCAL_FILES_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
    LOCAL_FILES_ARGS=(--local_files_only)
fi
GEN_EVAL_ARGS=()
if [[ "$EVAL_GENERATION" == "1" || "$EVAL_GENERATION" == "true" ]]; then
    GEN_EVAL_ARGS=(--eval_generation)
fi
M2_ARGS=()
if [[ -n "$M2_SOURCE_GOLD_PATH" ]]; then
    if [[ ! -f "$M2_SOURCE_GOLD_PATH" ]]; then
        echo "Error: M2_SOURCE_GOLD_PATH not found: $M2_SOURCE_GOLD_PATH"
        exit 1
    fi
    M2_ARGS=(--m2_source_gold_path "$M2_SOURCE_GOLD_PATH")
fi
TEST_ARGS=()
if [[ "$RUN_TEST_ON_END" == "1" || "$RUN_TEST_ON_END" == "true" ]]; then
    TEST_ARGS+=(--run_test_on_end)
fi
if [[ "$TEST_ONLY" == "1" || "$TEST_ONLY" == "true" ]]; then
    TEST_ARGS+=(--test_only)
fi

echo "Starting reference BLT-GEC fine-tuning on $DATASET_TYPE dataset..."
echo "BLT lr: $LR"
echo "BLT weight_decay: $WEIGHT_DECAY"
echo "BLT Adam betas: ($ADAM_BETA1, $ADAM_BETA2)"
echo "BLT scheduler: $SCHEDULER"
echo "BLT warmup_steps: $WARMUP_STEPS"
echo "BLT num_beams: $BLT_NUM_BEAMS"
echo "BLT NUM_GPUS: $NUM_GPUS"

export PYTHONUNBUFFERED=1

if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCHER=(
        srun "$PYTHON_BIN" -m torch.distributed.run
        --standalone
        --nproc_per_node "$NUM_GPUS"
        -m blt_gec.train
    )
else
    LAUNCHER=(srun "$PYTHON_BIN" -m blt_gec.train)
fi

"${LAUNCHER[@]}" \
    --data "$DATASET_TYPE" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --reference_code_dir "$REFERENCE_BLT_DIR" \
    --blt_repo "$BLT_REPO" \
    --entropy_repo "$ENTROPY_REPO" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta1 "$ADAM_BETA1" \
    --adam_beta2 "$ADAM_BETA2" \
    --adam_eps "$ADAM_EPS" \
    --precision "$PRECISION" \
    --scheduler "$SCHEDULER" \
    --warmup_steps "$WARMUP_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --num_beams "$BLT_NUM_BEAMS" \
    --max_gen_len "$MAX_GEN_LEN" \
    --eval_max_examples "$EVAL_MAX_EXAMPLES" \
    --eval_every_steps "$EVAL_EVERY_STEPS" \
    --max_steps "$MAX_STEPS" \
    --log_every_steps "$LOG_EVERY_STEPS" \
    --checkpoint_interval_minutes 20 \
    --max_time "00:01:50:00" \
    "${LOCAL_FILES_ARGS[@]}" \
    "${GEN_EVAL_ARGS[@]}" \
    "${M2_ARGS[@]}" \
    "${TEST_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

echo "End Time: $(date)"
