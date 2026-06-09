#!/bin/bash
#SBATCH --job-name=eval-bart-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# A100 partition rejects high CPU requests per 1 GPU; keep this conservative.
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

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
echo "Requested GPUs: 1"
echo "Requested CPU cores per task: 4"

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

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
if ! "$PYTHON_BIN" -c "import torch; import numpy; import lightning; import transformers; print('Torch:', torch.__version__); print('NumPy:', numpy.__version__)" >/tmp/phdq_bart_eval_env_check.txt 2>&1; then
    cat /tmp/phdq_bart_eval_env_check.txt
    echo "Error: required BART evaluation packages are not installed in the selected Python environment."
    echo "Install baseline/requirements.txt first, or submit with PYTHON_BIN=/path/to/python or CONDA_ENV=<env_name>."
    exit 1
fi
cat /tmp/phdq_bart_eval_env_check.txt

# 경로 변수 설정
DATA_DIR="$PROJECT_HOME/data"

# 사용법:
#   DATASET_TYPE=native BART_RUN_NAME=native_clean sbatch scripts/eval_bart.sh
#   DATASET_TYPE=learner BART_RUN_NAME=learner_clean SPLIT=test sbatch scripts/eval_bart.sh
#   MODEL_CKPT=outputs/native_clean/best.ckpt DATASET_TYPE=native sbatch scripts/eval_bart.sh
DATASET_TYPE="${DATASET_TYPE:-${2:-native}}"
DATASET_DIR_NAME="$DATASET_TYPE"
if [[ "$DATASET_TYPE" == "learner" ]]; then
    DATASET_DIR_NAME="korean_learner"
fi

BART_RUN_NAME="${BART_RUN_NAME:-${DATASET_TYPE}_clean}"
MODEL_CKPT="${MODEL_CKPT:-${1:-outputs/${BART_RUN_NAME}/best.ckpt}}"
SPLIT="${SPLIT:-test}"
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Error: SPLIT must be val or test, got: $SPLIT"
    exit 1
fi

PREPROCESSED_DIR="$DATA_DIR/Preprocessed/$DATASET_DIR_NAME"
TRAIN_DATA="${TRAIN_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_train.txt}"
VAL_DATA="${VAL_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_val.txt}"
TEST_DATA="${TEST_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_test.txt}"

if [[ "$SPLIT" == "test" ]]; then
    EVAL_DATA="$TEST_DATA"
else
    EVAL_DATA="$VAL_DATA"
fi

if [[ ! -f "$MODEL_CKPT" ]]; then
    echo "Error: checkpoint not found: $MODEL_CKPT"
    exit 1
fi

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

echo "Evaluating Checkpoint: $MODEL_CKPT"
echo "Dataset: $DATASET_TYPE"
echo "BART run name: $BART_RUN_NAME"
echo "Split: $SPLIT"
echo "Eval data: $EVAL_DATA"

NUM_WORKERS="${NUM_WORKERS:-4}"
BART_NUM_BEAMS="${BART_NUM_BEAMS:-4}"
M2_SOURCE_GOLD_PATH="${M2_SOURCE_GOLD_PATH:-}"
M2_ARGS=()
if [[ -n "$M2_SOURCE_GOLD_PATH" ]]; then
    if [[ ! -f "$M2_SOURCE_GOLD_PATH" ]]; then
        echo "Error: M2_SOURCE_GOLD_PATH not found: $M2_SOURCE_GOLD_PATH"
        exit 1
    fi
    M2_ARGS=(--m2_source_gold_path "$M2_SOURCE_GOLD_PATH")
fi

# baseline/run.py는 model_ckpt_path가 있으면 validate()를 실행한다.
# SPLIT=test일 때 test file을 val_data_path에 넣어 test generation GLEU를 계산한다.
srun "$PYTHON_BIN" baseline/run.py \
    --name "eval-${BART_RUN_NAME}-${SPLIT}" \
    --data "${BART_RUN_NAME}_${SPLIT}" \
    --model_ckpt_path "$MODEL_CKPT" \
    --num_workers "$NUM_WORKERS" \
    --num_beams "$BART_NUM_BEAMS" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$EVAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --max_epochs 0 \
    "${M2_ARGS[@]}"

echo "Evaluation End: $(date)"
