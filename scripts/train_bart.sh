#!/bin/bash
#SBATCH --job-name=bart-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# A100 partition rejects high CPU requests per 1 GPU; keep this conservative.
#SBATCH --cpus-per-task=4
#SBATCH --time=01:55:00
#SBATCH --signal=B:TERM@300

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

# Python 환경 설정:
# - PYTHON_BIN=/path/to/python 으로 명시 가능
# - 또는 CONDA_ENV=phdq_gec 로 conda 환경 활성화 가능
PYTHON_BIN="${PYTHON_BIN:-python}"
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
if ! "$PYTHON_BIN" -c "import torch; import numpy; import lightning; import transformers; print('Torch:', torch.__version__); print('NumPy:', numpy.__version__)" >/tmp/phdq_bart_env_check.txt 2>&1; then
    cat /tmp/phdq_bart_env_check.txt
    echo "Error: required BART training packages are not installed in the selected Python environment."
    echo "Install baseline/requirements.txt first, or submit with PYTHON_BIN=/path/to/python or CONDA_ENV=<env_name>."
    echo "Example: CONDA_ENV=phdq_gec sbatch scripts/train_bart.sh"
    exit 1
fi
cat /tmp/phdq_bart_env_check.txt

# 경로 변수 설정
DATA_DIR="$PROJECT_HOME/data"
CHECKPOINT_DIR="$PROJECT_HOME/baseline/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# 사용하려는 데이터셋 설정: DATASET_TYPE=native sbatch scripts/train_bart.sh
DATASET_TYPE="${DATASET_TYPE:-native}"
DATASET_DIR_NAME="$DATASET_TYPE"
if [[ "$DATASET_TYPE" == "learner" ]]; then
    DATASET_DIR_NAME="korean_learner"
fi

PREPROCESSED_DIR="$DATA_DIR/Preprocessed/$DATASET_DIR_NAME"
TRAIN_DATA="${TRAIN_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_train.txt}"
VAL_DATA="${VAL_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_val.txt}"
TEST_DATA="${TEST_DATA:-$PREPROCESSED_DIR/${DATASET_DIR_NAME}_test.txt}"

# 데이터 파일 존재 여부 검증
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

echo "Starting KoBART GEC Training on $DATASET_TYPE dataset..."

RESUME_CKPT="${RESUME_CKPT-auto}"
RESUME_ARGS=()
if [[ "$RESUME_CKPT" == "auto" ]]; then
    if [[ -f "outputs/${DATASET_TYPE}/last.ckpt" ]]; then
        RESUME_ARGS=(--resume_ckpt_path "outputs/${DATASET_TYPE}/last.ckpt")
        echo "Auto-resume enabled: outputs/${DATASET_TYPE}/last.ckpt"
    else
        echo "Auto-resume enabled: no existing last.ckpt found; starting fresh."
    fi
elif [[ -n "$RESUME_CKPT" ]]; then
    if [[ ! -f "$RESUME_CKPT" ]]; then
        echo "Error: RESUME_CKPT not found: $RESUME_CKPT"
        exit 1
    fi
    RESUME_ARGS=(--resume_ckpt_path "$RESUME_CKPT")
    echo "Resuming from explicit checkpoint: $RESUME_CKPT"
fi

# 학습 실행 (PL 2.x 호환 run.py)
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
srun "$PYTHON_BIN" baseline/run.py \
    --name "kobart-${DATASET_TYPE}" \
    --data "$DATASET_TYPE" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size 32 \
    --num_workers "$NUM_WORKERS" \
    --lr 5e-5 \
    --max_seq_len 128 \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --checkpoint_interval_minutes 20 \
    --max_time "00:01:50:00" \
    "${RESUME_ARGS[@]}"

echo "End Time: $(date)"
