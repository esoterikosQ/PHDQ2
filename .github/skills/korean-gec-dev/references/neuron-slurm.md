# Neuron SLURM 실행 레퍼런스

기준 문서: https://docs-ksc.gitbook.io/neuron-user-guide/system/running-jobs-through-scheduler-slurm

## 핵심 규칙

- 작업 제출은 `/scratch/$USER` 아래에서만 수행한다.
- 작업 제출은 `sbatch <script>`로 수행하고, 상태 확인은 `squeue`, 취소는 `scancel <JOB_ID>`를 사용한다.
- Neuron은 SBATCH `--comment` 옵션으로 사용 애플리케이션을 명시해야 한다.
  - 본 프로젝트의 PyTorch 학습/평가 스크립트는 `--comment=pytorch`를 사용한다.
- Neuron 공유 노드 정책에서는 `--mem` 옵션을 사용하지 않는다.
  - 메모리는 `--ntasks-per-node`와 `--cpus-per-task`에 의해 자동 산정된다.
- 일반 GPU 파티션은 시스템 상태에 따라 조정될 수 있으므로 제출 전 `sinfo`로 확인한다.

## PHDQ 학습 스크립트 기본값

`scripts/train_bart.sh`, `scripts/eval_bart.sh`, `scripts/train_blt.sh`는 Neuron 가이드에 맞춰 다음 기본값을 사용한다.

- BART 학습/평가: `eme_h200nv_8`
- BLT-GEC scaffold 학습: `amd_a100nv_8`

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --comment=pytorch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
```

학습 스크립트는 2시간 제한에 대비해 다음 안전장치를 포함한다.

- SLURM 제한은 `#SBATCH --time=01:55:00`으로 요청한다.
- Lightning 내부 `--max_time`은 `00:01:50:00`으로 설정해 스케줄러 강제 종료 전에 스스로 멈춘다.
- 학습 중 20분마다 metric과 무관한 재개용 checkpoint를 `outputs/<dataset>/resume/`에 저장한다.
- validation GLEU 기준 best checkpoint는 `outputs/<dataset>/model_ckpt/`에 별도로 저장한다.
- 학습 종료 시 `outputs/<dataset>/last.ckpt`를 한 번 더 저장한다.
- 다음 제출 시 `outputs/<dataset>/last.ckpt`가 있으면 자동으로 이어서 학습한다.

BLT-GEC scaffold는 같은 정책을 사용하되 checkpoint 경로가 `outputs/blt_gec/<dataset>/last.ckpt`다.

다른 GPU 파티션을 사용할 경우 스크립트의 `#SBATCH -p ...` 줄을 수정한다.
예: `cas_v100_4`, `cas_v100nv_4`, `amd_a100nv_8`, `eme_h200nv_8`, `gh200_1`.

## 제출 전 체크리스트

```bash
cd /scratch/$USER/PHDQ2
git pull origin main
mkdir -p data logs

# 데이터 파일 확인
ls data/Preprocessed/native/native_train.txt
ls data/Preprocessed/native/native_val.txt
ls data/Preprocessed/native/native_test.txt

# 파티션 상태 확인
sinfo

# 학습 제출
sbatch scripts/train_bart.sh

# BLT-GEC scaffold 학습 제출
sbatch scripts/train_blt.sh

# 이어서 학습할 때도 같은 명령 사용: last.ckpt가 있으면 자동 resume
sbatch scripts/train_bart.sh
sbatch scripts/train_blt.sh

# 자동 resume을 끄고 새로 시작
RESUME_CKPT="" sbatch scripts/train_bart.sh

# 특정 checkpoint에서 재개
RESUME_CKPT=outputs/native/model_ckpt/<checkpoint>.ckpt sbatch scripts/train_bart.sh
RESUME_CKPT=outputs/blt_gec/native/best.ckpt sbatch scripts/train_blt.sh

# 체크포인트 평가 제출
sbatch scripts/eval_bart.sh outputs/native/model_ckpt/<checkpoint>.ckpt native

# 상태 확인
squeue -u $USER
```

## 모니터링

```bash
squeue -u $USER
sinfo -Nel
scancel <JOB_ID>
```

작업이 배정된 노드는 `squeue`의 `NODELIST`에서 확인하고, 필요하면 해당 계산 노드에 `ssh`로 접속해 `top` 또는 `nvidia-smi`로 진행 상태를 확인한다.

## 중단/재개 운영 원칙

- 2시간 제한이 있는 학습은 하나의 긴 job이 아니라 여러 번의 짧은 job으로 누적 실행한다.
- 시간 제한으로 중단되었거나 `CANCELLED`, `TIMEOUT` 상태가 되었으면 동일 명령으로 다시 제출한다.
- `outputs/<dataset>/last.ckpt`가 있으면 `scripts/train_bart.sh`가 자동으로 `--resume_ckpt_path`를 전달한다.
- `outputs/blt_gec/<dataset>/last.ckpt`가 있으면 `scripts/train_blt.sh`가 자동으로 `--resume_ckpt_path`를 전달한다.
- 새 실험을 시작하려면 기존 `last.ckpt`를 다른 위치로 옮기거나 `RESUME_CKPT="" sbatch scripts/train_bart.sh` / `RESUME_CKPT="" sbatch scripts/train_blt.sh`를 사용한다.
- 각 job의 `JOB_ID`, 종료 상태, 마지막 checkpoint, validation GLEU를 `LOG.md` 또는 별도 실험 노트에 기록한다.
