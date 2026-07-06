# GPU 자원 확인

```bash
# 전체 파티션의 자원 상황 일람
sinfo

# 특정 파티션의 자원 상황
sinfo -p [파티션] -Nel

# 특정 노드의 GPU 잔여량 확인
scontrol show node [노드명]

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

# BLT 48시간 학습

`scripts/train_blt.sh`는 기본적으로 SLURM job의 `TimeLimit`을 읽어서 Python 학습 코드의 `--max_time`에 자동 반영한다.
따라서 48시간으로 돌릴 때는 `sbatch --time=2-00:00:00`만 override하면 된다.

```bash
# union 4GPU, 48시간 이어서 학습
CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=4 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/union/last.ckpt \
  GRAD_ACCUM_STEPS=2 \
  sbatch --time=2-00:00:00 -p amd_a100nv_8 --nodelist=[노드명] \
  --gres=gpu:4 --cpus-per-task=8 scripts/train_blt.sh

# native 2GPU, 48시간 이어서 학습
CONDA_ENV=phdq_blt DATASET_TYPE=native NUM_GPUS=2 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/native/last.ckpt \
  GRAD_ACCUM_STEPS=4 \
  sbatch --time=2-00:00:00 -p amd_a100nv_8 --nodelist=[노드명] \
  --gres=gpu:2 --cpus-per-task=8 scripts/train_blt.sh

# learner 2GPU, 48시간 이어서 학습
CONDA_ENV=phdq_blt DATASET_TYPE=learner NUM_GPUS=2 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/learner/last.ckpt \
  GRAD_ACCUM_STEPS=4 \
  sbatch --time=2-00:00:00 -p amd_a100nv_8 --nodelist=[노드명] \
  --gres=gpu:2 --cpus-per-task=8 scripts/train_blt.sh
```

수동으로 내부 학습 제한을 지정해야 하면 `MAX_TIME=DD:HH:MM:SS`를 같이 준다.
예: `MAX_TIME=01:23:50:00 sbatch --time=2-00:00:00 ...`


union

CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/union/last.ckpt \
  sbatch \
  -p amd_h200nv_8 --nodelist=gpu55 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

native

CONDA_ENV=phdq_blt DATASET_TYPE=native NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/native/last.ckpt \
  sbatch \
  -p amd_a100nv_8 --nodelist=gpu41 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh

learner

CONDA_ENV=phdq_blt DATASET_TYPE=learner NUM_GPUS=1 MAX_EPOCHS=10 \
  RESUME_CKPT=outputs/blt_gec/learner/last.ckpt \
  sbatch  \
  -p amd_a100nv_8 --nodelist=gpu43 --gres=gpu:1 --cpus-per-task=4 scripts/train_blt.sh


bart test

DATASET_TYPE=native BART_RUN_NAME=native_clean SPLIT=test \
sbatch scripts/eval_bart.sh

DATASET_TYPE=learner BART_RUN_NAME=learner_clean SPLIT=test \
sbatch scripts/eval_bart.sh

DATASET_TYPE=union BART_RUN_NAME=union_clean SPLIT=test \
sbatch scripts/eval_bart.sh


753881 : union / done
753878 : learner / done
753882 : native / done

753911 : learner / cancel
753945 : union4 / cancel
753950 : native2

753976 : union4

learner 작업 819735 종료 후

CONDA_ENV=phdq_blt DATASET_TYPE=learner SPLIT=test \
  CKPT_PATH=outputs/blt_gec/learner/best.ckpt \
  NUM_SHARDS=80 PRECISION=fp16 \
  sbatch -p cas_v100_4 --array=4-7 \
  --gres=gpu:1 --cpus-per-task=4 scripts/eval_blt.sh

819897 union epoch0 이어서

819902 learner validation array 4-7 [done]

820154 learner 16-19/79
820158 native 4-7/39
820188 learner 20-23/79
820248 learner 24-27/79
820344 learner 28-31/79

820341 union epoch0 이어서

820374 learner 32-35/79
823115 learner 40-43/79
823126 union epoch1

CONDA_ENV=phdq_blt DATASET_TYPE=union NUM_GPUS=4 MAX_EPOCHS=2 \
  RESUME_CKPT=outputs/blt_gec/union/last.ckpt \
  GRAD_ACCUM_STEPS=4 \
  sbatch --time=12:00:00 -p amd_h200nv_8 \
  --nodes=1 --gres=gpu:4 --cpus-per-task=8 scripts/train_blt.sh

823131 native 8-11/39
CONDA_ENV=phdq_blt DATASET_TYPE=native SPLIT=test \
  CKPT_PATH=outputs/blt_gec/native/best.ckpt NUM_SHARDS=40 \
  sbatch --array=8-11 scripts/eval_blt.sh

823336 learner 44-47/79

823433 learner 48-51/79
823437 native 12-15/39
823656 learner 52-55/79

823988 learner 56-59/79
823992 native 16-19/39

824070 learner 60-63/79
824208 learner 64-67/79

824537 learner 68-71/79
824540 native 20-23/39

824546 union epoch 2-7

824587 learner 72-75/79
824602 learner 76-79/79


834647 native 28-31/39