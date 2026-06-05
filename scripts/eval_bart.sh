#!/bin/bash
#SBATCH --job-name=eval-bart-gec
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p eme_h200nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
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

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

# TODO: 가상환경 활성화
# source activate phdq_gec

DATA_DIR="$PROJECT_HOME/data"
MODEL_CKPT="${1:-}" # 실행 시 $1로 체크포인트 경로 전달
DATASET_TYPE="${2:-native}"

if [ -z "$MODEL_CKPT" ]; then
    echo "Usage: sbatch scripts/eval_bart.sh <path_to_checkpoint.ckpt> [dataset_type]"
    exit 1
fi

TRAIN_DATA="$DATA_DIR/${DATASET_TYPE}_train.tsv"
VAL_DATA="$DATA_DIR/${DATASET_TYPE}_dev.tsv"
TEST_DATA="$DATA_DIR/${DATASET_TYPE}_test.tsv"

if [[ ! -f "$MODEL_CKPT" ]]; then
    echo "Error: checkpoint not found: $MODEL_CKPT"
    exit 1
fi

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -f "$TEST_DATA" ]]; then
    echo "Error: Train/Val/Test data not found at $DATA_DIR"
    echo "Expected:"
    echo "  $TRAIN_DATA"
    echo "  $VAL_DATA"
    echo "  $TEST_DATA"
    exit 1
fi

echo "Evaluating Checkpoint: $MODEL_CKPT"
echo "Dataset: $DATASET_TYPE"

# 추론 및 생성 (test 파이프라인) -- test 모드 스크립트 작성 필요 시 활용
# 현재 run.py 내에 test() 과정이 포함될 수 있으나, 보통 generation은 별도 스크립트화

# 임시: run.py에 --model_ckpt_path 전달하여 평가
srun python baseline/run.py \
    --name "eval-kobart-${DATASET_TYPE}" \
    --data "$DATASET_TYPE" \
    --model_ckpt_path "$MODEL_CKPT" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --max_epochs 0

echo "Evaluation End: $(date)"
