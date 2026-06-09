# GPU 자원 확인

```bash
# 전체 파티션의 자원 상황 일람
sinfo

# 특정 파티션의 자원 상황
sinfo -p [파티션] -Nel

# 특정 노드의 GPU 잔여량 확인
scontron show node [노드명]

scontrol show node [노드명] | egrep "NodeName=|State=|Gres=|CfgTRES=|AllocTRES=|CPUAlloc"

```

# 작업 확인

```bash
# 사용자별 작업 목록
squeue -u [ID]

# 작업이 진행중인 경우에는 gpu 노드에 접속해서 작업 상황을 확인할 수 있음
ssh gpu[GPU 노드 번호]
nvidia-smi -l 2

# 작업 삭제
scancel [작업번호]

```

# 작업 요청

```bash
# 작업 요청 방법
sbatch [쉘스크립트 파일명]

# 플래그 설명
sbatch -J [작업명] --time [최대작업시간] -o [작업로그파일명 (1출력)] -e [에러로그 파일명(2출력)] -p [파티션 이름] --comment [애플리케이션 이름] --nodes [작업 노드 수] --gres=gpu:[GPU 요청 수] --cpus-per-task=[CPU 요청 수] 

# 특정 노드에 작업 요청
sbatch -p [파티션 이름] --nodelist=[노드명] --gres=gpu:[GPU 요청 수] --cpus-per-task=[CPU 요청 수] [쉘스크립트 파일명]
```

# BLT H200 1GPU 재개

```bash
# native 이어서 학습
CONDA_ENV=phdq_blt DATASET_TYPE=native NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/native/last.ckpt \
  sbatch -p amd_h200nv_8 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

# learner 이어서 학습: 남은 GPU가 있는 노드를 확인한 뒤 --nodelist만 바꿔서 사용
CONDA_ENV=phdq_blt DATASET_TYPE=learner NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/learner/last.ckpt \
  sbatch -p amd_h200nv_8 --nodelist=[노드명] --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

# union clean start
CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=1 MAX_EPOCHS=10 RESUME_CKPT="" \
  sbatch -p amd_h200nv_8 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

# 특정 작업이 끝난 뒤에 작업을 시작
CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/union/last.ckpt \
  sbatch --dependency=afterok:753653:753644 \
  -p amd_h200nv_8 --nodelist=gpu55 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh
```


union

CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/union/last.ckpt \
  sbatch --dependency=afterok:753723 \
  -p amd_h200nv_8 --nodelist=gpu55 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

native

CONDA_ENV=phdq_blt DATASET_TYPE=native NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/native/last.ckpt \
  sbatch --dependency=afterok:753644 \
  -p amd_a100nv_8 --nodelist=gpu41 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

learner

CONDA_ENV=phdq_blt DATASET_TYPE=learner NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/learner/last.ckpt \
  sbatch --dependency=afterok:753647 \
  -p amd_a100nv_8 --nodelist=gpu43 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh


bart test

DATASET_TYPE=native BART_RUN_NAME=native_clean SPLIT=test \
sbatch scripts/eval_bart.sh

DATASET_TYPE=learner BART_RUN_NAME=learner_clean SPLIT=test \
sbatch scripts/eval_bart.sh

DATASET_TYPE=union BART_RUN_NAME=union_clean SPLIT=test \
sbatch scripts/eval_bart.sh