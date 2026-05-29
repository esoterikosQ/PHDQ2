#!/bin/bash
#SBATCH --job-name=bart-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
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

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

# 가상환경 활성화 (필요 시 주석 해제 및 수정)
# module load anaconda3/2021.05
# source activate phdq_gec

# 경로 변수 설정
DATA_DIR="$PROJECT_HOME/data"
CHECKPOINT_DIR="$PROJECT_HOME/baseline/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# 사용하려는 데이터셋 설정: DATASET_TYPE=native sbatch scripts/train_bart.sh
DATASET_TYPE="${DATASET_TYPE:-native}"
TRAIN_DATA="$DATA_DIR/${DATASET_TYPE}_train.tsv"
VAL_DATA="$DATA_DIR/${DATASET_TYPE}_dev.tsv"
TEST_DATA="$DATA_DIR/${DATASET_TYPE}_test.tsv"

# 데이터 파일 존재 여부 검증
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -f "$TEST_DATA" ]]; then
    echo "Error: Train/Val/Test data not found at $DATA_DIR"
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
srun python baseline/run.py \
    --name "kobart-${DATASET_TYPE}" \
    --data "$DATASET_TYPE" \
    --max_epochs 20 \
    --batch_size 32 \
    --lr 5e-5 \
    --max_seq_len 128 \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --checkpoint_interval_minutes 20 \
    --max_time "00:01:50:00" \
    "${RESUME_ARGS[@]}"

echo "End Time: $(date)"
