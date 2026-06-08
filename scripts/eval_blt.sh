#!/bin/bash
#SBATCH --job-name=eval-blt-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# A100 partition rejects high CPU requests per 1 GPU; keep this conservative.
#SBATCH --cpus-per-task=4
#SBATCH --time=01:55:00

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
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Submit Dir: $SLURM_SUBMIT_DIR"
echo "Partition: amd_a100nv_8"
echo "Requested GPUs: 1"
echo "Requested CPU cores per task: ${SLURM_CPUS_PER_TASK:-4}"

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-phdq_blt}"
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
if ! "$PYTHON_BIN" -c "import torch; import numpy; print('Torch:', torch.__version__); print('NumPy:', numpy.__version__)" >/tmp/phdq_blt_eval_env_check.txt 2>&1; then
    cat /tmp/phdq_blt_eval_env_check.txt
    echo "Error: PyTorch/NumPy is not installed in the selected Python environment."
    exit 1
fi
cat /tmp/phdq_blt_eval_env_check.txt

REFERENCE_BLT_DIR="${REFERENCE_BLT_DIR:-$PROJECT_HOME/reference_code/blt}"
if [[ ! -d "$REFERENCE_BLT_DIR/bytelatent" ]]; then
    echo "Error: reference BLT code not found: $REFERENCE_BLT_DIR/bytelatent"
    exit 1
fi

DATA_DIR="$PROJECT_HOME/data"
DATASET_TYPE="${DATASET_TYPE:-native}"
DATASET_DIR_NAME="$DATASET_TYPE"
if [[ "$DATASET_TYPE" == "learner" ]]; then
    DATASET_DIR_NAME="korean_learner"
fi
SPLIT="${SPLIT:-val}"
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Error: SPLIT must be val or test, got $SPLIT."
    exit 1
fi

PREPROCESSED_DIR="$DATA_DIR/Preprocessed/$DATASET_DIR_NAME"
VAL_DATA="${VAL_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_val.txt}"
TEST_DATA="${TEST_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_test.txt}"
if [[ "$SPLIT" == "val" ]]; then
    SPLIT_FILE="$VAL_DATA"
else
    SPLIT_FILE="$TEST_DATA"
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
    echo "Error: split data not found: $SPLIT_FILE"
    exit 1
fi

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: CKPT_PATH is required."
    echo "Example: CKPT_PATH=outputs/blt_gec/native/best.ckpt sbatch --array=0-39 scripts/eval_blt.sh"
    exit 1
fi
CKPT_NAME="${CKPT_NAME:-$(basename "$CKPT_PATH" .ckpt)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/blt_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_GEN_LEN="${MAX_GEN_LEN:-256}"
BLT_NUM_BEAMS="${BLT_NUM_BEAMS:-4}"
PRECISION="${PRECISION:-bf16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
BLT_REPO="${BLT_REPO:-facebook/blt-1b}"
ENTROPY_REPO="${ENTROPY_REPO:-facebook/blt-entropy}"
M2_SOURCE_GOLD_PATH="${M2_SOURCE_GOLD_PATH:-}"
AGGREGATE="${AGGREGATE:-0}"

LOCAL_FILES_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
    LOCAL_FILES_ARGS=(--local_files_only)
fi
M2_ARGS=()
if [[ -n "$M2_SOURCE_GOLD_PATH" ]]; then
    if [[ ! -f "$M2_SOURCE_GOLD_PATH" ]]; then
        echo "Error: M2_SOURCE_GOLD_PATH not found: $M2_SOURCE_GOLD_PATH"
        exit 1
    fi
    M2_ARGS=(--m2_source_gold_path "$M2_SOURCE_GOLD_PATH")
fi

if [[ "$AGGREGATE" == "1" || "$AGGREGATE" == "true" ]]; then
    echo "Aggregating BLT generation shards..."
    srun "$PYTHON_BIN" -m blt_gec.eval \
        --aggregate \
        --output_dir "$OUTPUT_DIR" \
        --data "$DATASET_TYPE" \
        --split "$SPLIT" \
        --val_data_path "$VAL_DATA" \
        --test_data_path "$TEST_DATA" \
        --checkpoint_name "$CKPT_NAME" \
        "${M2_ARGS[@]}"
    echo "End Time: $(date)"
    exit 0
fi

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

TOTAL_EXAMPLES="$("$PYTHON_BIN" - "$SPLIT_FILE" <<'PY'
import sys
from pathlib import Path

count = 0
with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip() and len(line.rstrip("\n").split("\t")) == 2:
            count += 1
print(count)
PY
)"
if [[ "$TOTAL_EXAMPLES" -lt 1 ]]; then
    echo "Error: no valid examples found in $SPLIT_FILE"
    exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-40}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SIZE=$(( (TOTAL_EXAMPLES + NUM_SHARDS - 1) / NUM_SHARDS ))
START_INDEX="${START_INDEX:-$(( TASK_ID * SHARD_SIZE ))}"
MAX_EXAMPLES="${MAX_EXAMPLES:-$SHARD_SIZE}"

if [[ "$START_INDEX" -ge "$TOTAL_EXAMPLES" ]]; then
    echo "Shard start index $START_INDEX >= total examples $TOTAL_EXAMPLES; nothing to do."
    exit 0
fi

echo "Evaluating BLT checkpoint: $CKPT_PATH"
echo "Dataset: $DATASET_TYPE"
echo "Split: $SPLIT"
echo "Total examples: $TOTAL_EXAMPLES"
echo "NUM_SHARDS: $NUM_SHARDS"
echo "TASK_ID: $TASK_ID"
echo "START_INDEX: $START_INDEX"
echo "MAX_EXAMPLES: $MAX_EXAMPLES"

srun "$PYTHON_BIN" -m blt_gec.eval \
    --checkpoint "$CKPT_PATH" \
    --checkpoint_name "$CKPT_NAME" \
    --data "$DATASET_TYPE" \
    --split "$SPLIT" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --reference_code_dir "$REFERENCE_BLT_DIR" \
    --blt_repo "$BLT_REPO" \
    --entropy_repo "$ENTROPY_REPO" \
    --max_length "$MAX_LENGTH" \
    --max_gen_len "$MAX_GEN_LEN" \
    --num_beams "$BLT_NUM_BEAMS" \
    --precision "$PRECISION" \
    --start_index "$START_INDEX" \
    --max_examples "$MAX_EXAMPLES" \
    "${LOCAL_FILES_ARGS[@]}" \
    "${M2_ARGS[@]}"

echo "End Time: $(date)"
