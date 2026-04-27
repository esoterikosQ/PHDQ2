#!/bin/bash
#SBATCH --job-name=eval-bart-gec
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

PROJECT_HOME="$PWD"
cd "$PROJECT_HOME"

# TODO: 가상환경 활성화
# source activate phdq_gec

DATA_DIR="$PROJECT_HOME/data"
CHECKPOINT_DIR="$PROJECT_HOME/baseline/checkpoints"
MODEL_CKPT="$1" # 실행 시 $1로 체크포인트 경로 전달

if [ -z "$MODEL_CKPT" ]; then
    echo "Usage: sbatch scripts/eval_bart.sh <path_to_checkpoint.ckpt>"
    exit 1
fi

echo "Evaluating Checkpoint: $MODEL_CKPT"

# 추론 및 생성 (test 파이프라인) -- test 모드 스크립트 작성 필요 시 활용
# 현재 run.py 내에 test() 과정이 포함될 수 있으나, 보통 generation은 별도 스크립트화

# 임시: run.py에 --model_ckpt_path 전달하여 평가
python baseline/run.py \
    --name "eval-kobart" \
    --model_ckpt_path "$MODEL_CKPT" \
    --train_data_path "$DATA_DIR/native_train.tsv" \
    --val_data_path "$DATA_DIR/native_dev.tsv" \
    --test_data_path "$DATA_DIR/native_test.tsv" \
    --max_epochs 0 # 테스트만 수행할 경우

echo "Evaluation End: $(date)"
