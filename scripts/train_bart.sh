#!/bin/bash
#SBATCH --job-name=bart-gec
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# SLURM 환경 로그 출력
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# 프로젝트 홈 설정 (실행하는 곳이 프로젝트 루트여야 함)
PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

# 가상환경 활성화 (필요 시 주석 해제 및 수정)
# module load anaconda3/2021.05
# source activate phdq_gec

# 경로 변수 설정
DATA_DIR="$PROJECT_HOME/data"
CHECKPOINT_DIR="$PROJECT_HOME/baseline/checkpoints"
mkdir -p logs "$CHECKPOINT_DIR"

# 사용하려는 데이터셋 설정 (native / lang8 / learner)
DATASET_TYPE="native"
TRAIN_DATA="$DATA_DIR/${DATASET_TYPE}_train.tsv"
VAL_DATA="$DATA_DIR/${DATASET_TYPE}_dev.tsv"
TEST_DATA="$DATA_DIR/${DATASET_TYPE}_test.tsv"

# 데이터 파일 존재 여부 검증
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "Error: Train or Val data not found at $DATA_DIR"
    exit 1
fi

echo "Starting KoBART GEC Training on $DATASET_TYPE dataset..."

# 학습 실행 (PL 2.x 호환 run.py)
python baseline/run.py \
    --name "kobart-${DATASET_TYPE}" \
    --data "$DATASET_TYPE" \
    --max_epochs 20 \
    --batch_size 32 \
    --lr 5e-5 \
    --max_seq_len 128 \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --test_data_path "$TEST_DATA"

echo "End Time: $(date)"
