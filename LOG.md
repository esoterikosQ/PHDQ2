# 한국어 GEC 프로젝트 진행 로그

> 이 파일은 `/korean-gec-dev` 스킬에 의해 자동 관리됩니다.
> 최신 항목이 위에 오도록 역순으로 기록합니다.

---
## [2026-06-09] BART union resume max_time 타이머 상태 방어

### 목표
- BART union clean run을 `last.ckpt`에서 이어서 학습할 때 짧은 시간만 돌고 중단되는 원인 확인 및 방어

### 원인
- `runtimelog/slurm-bart-gec-752910.*` 확인 결과 실제 SLURM 실행 시간은 약 16분이었지만 Lightning이 `Time limit reached. Elapsed time is 1:50:00`으로 판단
- `last.ckpt`로 Trainer 전체 상태를 복원하면서 Lightning `Timer` callback의 누적 `max_time` 상태까지 복원된 것으로 판단
- 이 상태에서는 새 SLURM job이어도 이전 job의 elapsed time을 이어받아 조기 종료될 수 있음

### 수행 내용
- `baseline/run.py`에 `strip_timer_state_from_resume_checkpoint()` 추가
  - resume checkpoint의 `callbacks`에서 key에 `timer`가 포함된 callback state 제거
  - 같은 디렉토리에 `*.resume_no_timer.ckpt` 임시 checkpoint 저장
  - Trainer resume은 이 sanitized checkpoint를 사용
- `init_ckpt_path` 로딩도 PyTorch 2.6+ 호환을 위해 `weights_only=False` 명시

### 결과
- BART는 optimizer/epoch/trainer state는 유지하면서 SLURM job마다 `max_time` 타이머를 새로 시작할 수 있음

### 다음 단계
- [ ] 최신 코드 반영 후 `RESUME_CKPT=outputs/union_clean/last.ckpt`로 BART union 재개

---
## [2026-06-08] BLT generation 평가 분리 및 single-node DDP 준비

### 목표
- SLURM 2시간 제한에서 BLT 학습과 generation 평가가 서로 막지 않도록 작업 분리
- single-node multi-GPU DDP 학습을 위한 코드 경로 추가

### 수행 내용
- `blt_gec/metrics.py` 추가
  - GLEU/M2 계산을 학습/평가 공용 유틸로 분리
- `blt_gec/eval.py` 추가
  - checkpoint 기반 val/test generation 평가를 standalone CLI로 분리
  - `start_index`/`max_examples` shard 평가와 aggregate 모드 지원
  - shard range gap/overlap 및 source/reference/hypothesis 누락 검사 추가
- `scripts/eval_blt.sh` 추가
  - SLURM array 기반 BLT generation shard 평가 지원
  - aggregate 모드 지원
- `scripts/train_blt.sh` 수정
  - 기본 `EVAL_GENERATION=0`으로 변경
  - `NUM_GPUS>1`일 때 `torch.distributed.run` 기반 single-node DDP 실행
  - 실제 CUDA visible GPU 수와 `NUM_GPUS` 불일치 fail-fast 추가
- `blt_gec/train.py` 수정
  - `DistributedSampler`, DDP, `no_sync()` gradient accumulation 지원
  - rank 0 전용 logging/checkpoint/validation 처리
  - DDP checkpoint 저장/로드 호환성 확보
- `blt_gec/generate.py` 수정
  - DDP checkpoint의 `module.` prefix 제거 후 로드

### 결과
- 학습 job은 validation loss 중심으로 2시간 제한 안에서 돌리고, GLEU generation은 별도 shard job으로 분리 가능
- single GPU 기존 실행과 single-node multi-GPU DDP 실행 경로가 공존

### 다음 단계
- [ ] `CONDA_ENV=phdq_blt EVAL_GENERATION=0 sbatch scripts/train_blt.sh`로 단일 GPU 하위 호환 확인
- [ ] `CONDA_ENV=phdq_blt CKPT_PATH=outputs/blt_gec/native/best.ckpt sbatch --array=0-1 scripts/eval_blt.sh`로 shard 평가 smoke test
- [ ] `CONDA_ENV=phdq_blt NUM_GPUS=2 sbatch --gres=gpu:2 scripts/train_blt.sh`로 DDP smoke test

---
## [2026-06-07] BLT auto-resume corrupt checkpoint 방어 추가

### 목표
- SLURM 중단/부분 저장으로 깨진 BLT checkpoint가 있을 때 auto-resume이 `PytorchStreamReader failed reading zip archive`로 중단되는 문제 해결

### 원인
- `scripts/train_blt.sh`가 `outputs/blt_gec/<dataset>/best.ckpt` 또는 `last.ckpt` 파일 존재만 확인하고 `--resume_ckpt_path`로 전달
- 깨진 zip checkpoint를 `torch.load`가 읽으면서 `failed finding central directory` 발생

### 수행 내용
- `scripts/train_blt.sh`에 checkpoint 사전 검증 추가
  - `torch.load(..., map_location="cpu", weights_only=False)`로 load 가능 여부 확인
  - auto-resume에서 깨진 `best.ckpt`는 `*.corrupt.<timestamp>`로 이동 후 `last.ckpt` 후보 확인
  - `last.ckpt`도 깨졌으면 격리하고 fresh start
  - 명시적 `RESUME_CKPT`가 깨졌으면 즉시 실패
- `blt_gec/train.py`의 실제 checkpoint load도 `weights_only=False`를 명시해 PyTorch 2.6+ 호환성 확보

### 결과
- 깨진 auto checkpoint 때문에 BLT smoke run 전체가 중단되는 상황을 방지
- 명시적 resume 경로가 잘못된 경우에는 조용히 새로 시작하지 않고 오류로 알려줌

### 다음 단계
- [ ] 클러스터에서 최신 코드 pull 후 `CONDA_ENV=phdq_blt EVAL_MAX_EXAMPLES=20 sbatch scripts/train_blt.sh` 재실행

---
## [2026-06-07] BLT attention 호환성 실행 플래그 추가

### 목표
- Neuron SLURM 환경의 PyTorch/xformers/flex_attention 조합 차이로 BLT reference 실행이 중단되는 문제를 완화

### 수행 내용
- `scripts/train_blt.sh`에 BLT attention 호환성 환경변수 기본값 추가
  - `BLT_SUPPRESS_ATTN_ERROR=1`
  - `BLT_ALLOW_MISSING_FLEX_ATTENTION=1`
- `blt_gec/model.py`의 reference BLT 로딩 후 attention 구현을 `sdpa`로 강제하고, flex attention 기반 cross-attention 경로를 우회하도록 한 기존 fallback과 맞춤
- 쉘 구문 확인
  - `bash -n scripts/train_blt.sh scripts/train_bart.sh scripts/eval_bart.sh scripts/train_byte_prefix_lm.sh`

### 결과
- BLT용 torch/xformers/flex_attention 세부 구현이 클러스터 환경과 완전히 일치하지 않아도, 가능한 경우 PyTorch SDPA 경로로 우회해 smoke run을 시도할 수 있음
- 여전히 BLT 전용 환경은 `phdq_blt`로 분리하고, torch nightly/xformers reference 조합을 우선 맞추는 것이 기본 방침

### 다음 단계
- [ ] `CONDA_ENV=phdq_blt EVAL_MAX_EXAMPLES=20 sbatch scripts/train_blt.sh`로 실제 SLURM smoke run에서 attention fallback 동작 확인

---
## [2026-06-07] BLT optimizer/test/generation 정리

### 목표
- BLT 논문 optimizer 설정과 fine-tuning 안정성 지적 반영
- generation/test 평가 유지보수 리스크 축소

### 수행 내용
- BLT-GEC AdamW 설정 변경
  - `betas=(0.9, 0.95)`, `eps=1e-8`
  - bias/norm/RMSNorm 계열 파라미터를 weight decay에서 제외
- `blt_gec/generation.py` 공용 generation helper 추가
  - `train.py` validation generation과 `generate.py` CLI가 같은 beam search 구현 사용
- `GecBltCollator`에서 사용하지 않는 `attention_mask` 제거
- BLT test 평가 경로 추가
  - `TEST_ONLY=1`로 checkpoint test-only 평가
  - `RUN_TEST_ON_END=1`로 학습 종료 후 test 평가

### 결과
- BLT optimizer 기본값이 논문 설정과 더 가까워짐
- generation 구현이 단일화되어 BART/BLT decoding 비교 조건 관리가 쉬워짐

### 다음 단계
- [ ] `TEST_ONLY=1 RESUME_CKPT=<best.ckpt> EVAL_MAX_EXAMPLES=100 sbatch scripts/train_blt.sh`로 test 평가 smoke run 확인

---
## [2026-06-07] 비교 신뢰성 개선 항목 반영

### 목표
- BART/BLT 비교가 loss-only 또는 서로 다른 decoding 조건에 의존하지 않도록 평가/학습 기본값 정리

### 수행 내용
- BLT-GEC에 generation 기반 GLEU 평가 추가
  - validation source/reference/hypothesis 파일 저장
  - baseline `run_gleu` 재사용
  - optional `M2_SOURCE_GOLD_PATH`가 있으면 m2scorer 실행
  - SLURM 기본은 epoch 끝 평가이며, step 중 평가는 `EVAL_EVERY_STEPS`로 명시
- BLT-GEC decoding에 beam search 추가
  - `BLT_NUM_BEAMS` 기본값 4
  - `blt_gec/generate.py`도 동일 beam search 사용
- BLT-GEC optimizer schedule 추가
  - `SCHEDULER=cosine`, `WARMUP_STEPS=2000`, `WEIGHT_DECAY=0.1` 기본값
- BART baseline profile 분리
  - `BART_PROFILE=paper`: lr 3e-5, batch 64, epoch 10
  - `BART_PROFILE=tuned`: lr 5e-5, batch 32, epoch 40
  - output run name을 `native_paper`처럼 profile별로 분리
- BART baseline의 `num_beams`, `M2_SOURCE_GOLD_PATH`, AdamW `correct_bias` 옵션 반영
- Serving repetition penalty 기본값을 2.0에서 1.0으로 변경

### 결과
- BART/BLT의 1차 비교는 beam=4 GLEU 기준으로 맞출 수 있음
- M2는 source-gold 파일이 있을 때만 계산하며, 없으면 unavailable로 기록

### 다음 단계
- [ ] BLT reference 환경에서 `EVAL_MAX_EXAMPLES`를 작게 둔 smoke run으로 generation 평가 속도 확인
- [ ] KAGAS로 validation/test source-gold M2 파일 생성 후 `M2_SOURCE_GOLD_PATH` 연결

---
## [2026-06-07] BLT 구현 구조 정정 및 byte-only baseline 격하

### 목표
- dynamic patching 없는 byte-only 모델을 BLT로 오인하지 않도록 코드와 실행 경로 정정

### 수행 내용
- 기존 `blt_gec`의 causal byte Prefix-LM 코드를 `byte_prefix_lm/` 패키지로 이동해 baseline/smoke-test 용도로 격하
- `scripts/train_byte_prefix_lm.sh` 추가
  - 기존 byte-only 학습은 이 스크립트로만 실행
  - 출력 기본 경로를 `outputs/byte_prefix_lm`로 변경
- `blt_gec/`를 reference BLT wrapper로 재작성
  - `reference_code/blt/bytelatent` import 경로 연결
  - `ByteLatentTransformer.from_pretrained`
  - `LMTransformer` entropy model
  - `BltTokenizerAndPatcher` 기반 realtime entropy patcher
  - batch마다 `patcher.patch(..., include_next_token=True)`로 `patch_lengths`를 계산해 forward에 전달
- reference BLT vocabulary와 충돌하지 않도록 별도 SEP token을 만들지 않고 textual separator를 byte로 인코딩
- `scripts/train_blt.sh`에서 mini 모델 크기 인자(`MODEL_DIM`, `NUM_LAYERS` 등)를 제거하고 reference BLT/HF repo 설정으로 교체

### 결과
- `blt_gec`는 dynamic entropy patching이 없으면 실행 실패
- 기존 512x8 결과는 BLT 결과가 아니라 byte Prefix-LM baseline 결과로 재분류해야 함

### 다음 단계
- [ ] 클러스터 BLT 전용 conda 환경에 `reference_code/blt/requirements.txt` 설치
- [ ] Hugging Face `facebook/blt-1b`, `facebook/blt-entropy` 접근 승인 및 cache 확인
- [ ] `sbatch scripts/train_blt.sh`로 reference BLT smoke run 실행

---
## [2026-06-06] BART checkpoint callback 충돌 수정

### 목표
- `Found more than one stateful callback of type ModelCheckpoint` 오류 해결

### 수행 내용
- `outputs/<dataset>/best.ckpt` 별칭 저장을 두 번째 `ModelCheckpoint`로 처리하던 방식을 제거
- 기존 top-k `ModelCheckpoint`가 선택한 best file을 state 없는 `SaveBestAlias` callback이 복사하도록 수정

### 결과
- Lightning `ModelCheckpoint` state key 충돌 없이 best checkpoint 별칭을 유지할 수 있게 됨

### 다음 단계
- [ ] `sbatch scripts/train_bart.sh` 재제출 후 callback 충돌이 사라졌는지 확인

---
## [2026-06-06] 배치 기본 실행 정책 단순화

### 목표
- `CONDA_ENV`, `RESUME_CKPT`, `INIT_CKPT`를 매번 명령어에 붙이지 않아도 되도록 배치 스크립트 기본 동작 정리

### 수행 내용
- `scripts/train_bart.sh`, `scripts/train_blt.sh`, `scripts/eval_bart.sh`의 기본 conda 환경을 `phdq`로 내장
- BART 학습은 기본적으로 해당 dataset의 best checkpoint를 자동 탐색해 resume
- 해당 dataset best가 없고 `DATASET_TYPE`이 `native`가 아니면 native best checkpoint 가중치로 자동 초기화
- BLT 학습은 `best.ckpt`를 우선 resume하고 없으면 `last.ckpt`로 fallback
- Neuron SLURM 레퍼런스의 제출 예시를 짧은 `sbatch scripts/...` 기준으로 갱신

### 결과
- native BART 이어학습: `sbatch scripts/train_bart.sh`
- learner/union BART 학습: `DATASET_TYPE=learner sbatch scripts/train_bart.sh`
- BLT 이어학습: `sbatch scripts/train_blt.sh`

### 다음 단계
- [ ] 클러스터에서 `sbatch scripts/train_bart.sh` 제출 시 `Auto-best resume enabled` 로그 확인
- [ ] learner/union 첫 제출 시 native best 초기화 로그 확인

---
## [2026-06-06] BART best checkpoint 기반 재개/초기화 분리

### 목표
- native BART 추가 학습에서 `last.ckpt`가 아니라 best checkpoint를 기준으로 이어가야 하는 문제 반영
- 다른 데이터셋 학습에서는 trainer state resume이 아니라 best model weight 초기화로 시작하도록 구분

### 수행 내용
- `baseline/run.py`에 `--init_ckpt_path` 추가
- `--init_ckpt_path`는 모델 가중치만 로드하고 optimizer/epoch/trainer state는 복원하지 않도록 구현
- BART best checkpoint 별칭 `outputs/<dataset>/best.ckpt` 저장 로직 추가
- `scripts/train_bart.sh`에서 `INIT_CKPT` 환경변수 지원
- `scripts/train_bart.sh`에서 `RESUME_CKPT=best` 지원
- Neuron SLURM 레퍼런스에 best resume과 init fine-tune 사용법 추가

### 결과
- 같은 데이터셋 추가 학습: `RESUME_CKPT=best`
- 다른 데이터셋 전이 학습: `RESUME_CKPT="" INIT_CKPT=<best checkpoint>`

### 다음 단계
- [ ] 기존 native run은 `outputs/native/best.ckpt`가 없으므로 `RESUME_CKPT=outputs/native/model_ckpt/native_5e-05_epoch=24_step=9625.ckpt`처럼 명시 경로 사용
- [ ] learner/union은 native best를 `INIT_CKPT`로 넘기고 `RESUME_CKPT=""`로 새 학습 시작

---
## [2026-06-06] BLT 모델 크기 조절 인자 노출

### 목표
- BLT scaffold 학습이 정상 진행됨을 확인한 뒤 더 큰 모델 크기로 실험할 수 있게 수정

### 수행 내용
- `scripts/train_blt.sh`에 모델 크기 관련 환경변수 추가
  - `MODEL_DIM` 기본값 256
  - `NUM_LAYERS` 기본값 4
  - `NUM_HEADS` 기본값 8
  - `DROPOUT` 기본값 0.1
- `NUM_WORKERS`도 환경변수로 노출
- `OUTPUT_DIR` 환경변수 추가
- auto-resume checkpoint 탐색 경로도 `OUTPUT_DIR` 기준으로 변경
- 기존 `blt_gec.train`의 `--dim`, `--num_layers`, `--num_heads`, `--dropout`, `--num_workers` 인자로 전달

### 결과
- `OUTPUT_DIR=outputs/blt_gec_512x8 MODEL_DIM=512 NUM_LAYERS=8 ... sbatch scripts/train_blt.sh`처럼 제출 시점에 크기 조절 가능
- 모델 크기를 바꾸면 기존 checkpoint와 호환되지 않으므로 큰 모델 실험은 별도 `OUTPUT_DIR`로 새로 시작해야 함

### 다음 단계
- [ ] 기존 256/4/8 모델은 `MAX_EPOCHS=40`으로 resume 학습
- [ ] 큰 모델은 `OUTPUT_DIR=outputs/blt_gec_512x8 MODEL_DIM=512 NUM_LAYERS=8 NUM_HEADS=8`로 별도 run 실험

---
## [2026-06-06] 학습 epoch 기본값 확장

### 목표
- 749696(BLT)와 749714(BART) 로그에서 20 epoch 도달로 정상 종료된 뒤 추가 학습 여지가 있는 상황 반영

### 수행 내용
- `scripts/train_blt.sh`에 `MAX_EPOCHS` 환경변수 추가, 기본값 40으로 설정
- `scripts/train_bart.sh`에 `MAX_EPOCHS` 환경변수 추가, 기본값 40으로 설정
- 기존 `last.ckpt`가 있으면 자동 resume 정책으로 20 epoch 이후부터 이어서 학습 가능

### 결과
- `MAX_EPOCHS=60 sbatch ...`처럼 제출 시점에 총 목표 epoch을 쉽게 늘릴 수 있게 됨

### 다음 단계
- [ ] BLT는 validation loss가 마지막 epoch까지 개선 중이므로 `MAX_EPOCHS=40` 이상으로 이어서 학습
- [ ] BART는 GLEU가 15~19 epoch에서 plateau라 `MAX_EPOCHS=30` 또는 40으로 짧게 추가 확인

---
## [2026-06-05] BART 학습 모드 복구 보강

### 목표
- BART 학습 로그의 `Modules in train mode: 0`, `Modules in eval mode: 182` 상태가 반복되지 않도록 수정

### 수행 내용
- `baseline/model.py`에서 KoBART wrapper 초기화 시 `self.model.train()` 호출
- `on_train_epoch_start()`에서 내부 BART 모델을 명시적으로 train mode로 전환
- validation generation 중에는 `torch.no_grad()`와 eval mode를 사용하되, generation 전 상태가 train mode였으면 복구하도록 수정

### 결과
- 다음 BART 제출부터 validation generation이 학습 모드를 계속 eval 상태로 남기는 위험을 줄임

### 다음 단계
- [ ] 다음 BART 재제출 로그에서 `Modules in train mode`가 0으로 남는지 확인

---
## [2026-06-05] BART DataLoader worker 수 조정

### 목표
- `--cpus-per-task=4` 환경에서 BART DataLoader가 8 worker를 생성한다는 경고 제거

### 수행 내용
- `baseline/run.py`에 `--num_workers` 인자 추가
- 명시값이 없으면 `SLURM_CPUS_PER_TASK`를 우선 사용하도록 수정
- `scripts/train_bart.sh`, `scripts/eval_bart.sh`에서 기본 `NUM_WORKERS=4`를 전달

### 결과
- 다음 BART 제출부터 DataLoader worker 수가 SLURM CPU 요청량과 맞도록 정리됨

### 다음 단계
- [ ] 현재 실행 중인 job은 그대로 두고, 다음 제출부터 최신 스크립트 사용

---
## [2026-06-05] NumPy 의존성 사전 검사 추가

### 목표
- BLT 학습 로그의 `Failed to initialize NumPy: No module named 'numpy'` 경고 제거

### 수행 내용
- `blt_gec/requirements.txt`에 `numpy>=1.23.0` 추가
- `scripts/train_blt.sh`의 환경 사전 검사에 `import numpy` 추가
- BART 코드도 dataset/metric에서 NumPy를 사용하므로 `scripts/train_bart.sh`, `scripts/eval_bart.sh` 사전 검사에 `import numpy` 추가

### 결과
- NumPy가 없는 환경에서는 학습 본문에 진입하기 전에 명확히 실패하도록 정리됨

### 다음 단계
- [ ] 클러스터 환경에서 `pip install -r blt_gec/requirements.txt` 또는 `pip install numpy` 실행

---
## [2026-06-05] BART PartitionConfig 대응

### 목표
- `eme_h200nv_8`에서 BART job이 `PENDING (PartitionConfig)`로 고정되는 문제 회피

### 수행 내용
- `scripts/train_bart.sh`: 기본 파티션을 `eme_h200nv_8` → `amd_a100nv_8`로 변경
- `scripts/eval_bart.sh`: 기본 파티션을 `eme_h200nv_8` → `amd_a100nv_8`로 변경
- BART 학습/평가 CPU 요청량을 `--cpus-per-task=4`로 낮춤
- 제출 로그에 partition/GPU/CPU 요청량을 출력하도록 추가
- Neuron SLURM 레퍼런스에 `PENDING (PartitionConfig)` 발생 시 A100 기본값으로 제출하도록 기록

### 결과
- BART도 현재 실행이 확인된 A100 파티션의 보수적 리소스 요청으로 제출되도록 정리됨

### 다음 단계
- [ ] 기존 `PartitionConfig` 상태의 BART job을 `scancel <JOBID>`로 취소
- [ ] 최신 코드 pull 후 `CONDA_ENV=phdq_gec sbatch scripts/train_bart.sh`로 재제출

---
## [2026-06-05] BART CPU 요청량 제한 반영

### 목표
- BART 계열 job도 GPU 1개당 CPU 8 core 제한을 넘지 않도록 수정

### 수행 내용
- `scripts/train_bart.sh`: 기존 10 core 요청 → `--cpus-per-task=8`
- `scripts/eval_bart.sh`: 기존 10 core 요청 → `--cpus-per-task=8`
- Neuron SLURM 레퍼런스의 BART/H200 CPU 요청량을 8 core로 갱신

### 결과
- BART 학습/평가 스크립트에 10 core 요청이 남지 않도록 정리됨

### 다음 단계
- [ ] 클러스터에서 `grep -n "cpus-per-task\\|#SBATCH -p" scripts/train_bart.sh scripts/eval_bart.sh`로 최신 스크립트 반영 여부 확인

---
## [2026-06-05] SLURM Python 환경 선택 및 사전 검사 추가

### 목표
- 시스템 Python에 `torch`가 없어 BLT 학습이 실패한 문제를 방지

### 수행 내용
- `scripts/train_blt.sh`
  - `PYTHON_BIN` 또는 `CONDA_ENV`로 Python 환경을 지정할 수 있게 수정
  - 학습 실행 전 `import torch` 사전 검사 추가
  - `srun "$PYTHON_BIN" -m blt_gec.train`로 선택된 Python 사용
- `scripts/train_bart.sh`, `scripts/eval_bart.sh`
  - 동일한 `PYTHON_BIN`/`CONDA_ENV` 선택 방식 추가
  - `torch`, `lightning`, `transformers` 사전 검사 추가
- Neuron SLURM 레퍼런스의 제출 예시를 `CONDA_ENV=phdq_gec` / `CONDA_ENV=phdq_blt` 기준으로 갱신

### 결과
- `ModuleNotFoundError: No module named 'torch'`가 학습 traceback으로 터지기 전에 환경 설정 오류로 명확히 실패하도록 정리됨

### 다음 단계
- [ ] SLURM에서 `phdq_blt` 또는 원하는 Python 환경에 PyTorch 설치
- [ ] `CONDA_ENV=phdq_blt sbatch scripts/train_blt.sh`로 재제출

---

## [2026-06-05] BLT A100 요청 리소스 보수화

### 목표
- `amd_a100nv_8` 제출 오류가 계속 나는 상황에 대비해 BLT-GEC job의 CPU 요청량을 더 낮춤

### 수행 내용
- `scripts/train_blt.sh`: `--cpus-per-task=8` → `--cpus-per-task=4`
- 제출 후 로그에서 partition/GPU/CPU 요청량을 바로 확인할 수 있도록 echo 추가
- Neuron SLURM 레퍼런스와 환경 체크리스트의 BLT/A100 CPU 요청량 갱신

### 결과
- A100 파티션의 GPU 1개당 CPU 제한을 더 보수적으로 만족하도록 조정됨

### 다음 단계
- [ ] 클러스터에서 `grep -n "cpus-per-task\\|#SBATCH -p" scripts/train_blt.sh`로 최신 스크립트 반영 여부 확인
- [ ] 여전히 거절되면 `sbatch` 에러 전문을 기록해 파티션/계정/시간/옵션 문제를 분리

---

## [2026-06-05] A100 파티션 CPU 제한 반영

### 목표
- `amd_a100nv_8` 파티션의 GPU 1개당 CPU core 제한에 맞게 BLT-GEC job 수정

### 수행 내용
- `scripts/train_blt.sh`: 기존 10 core 요청 → `--cpus-per-task=8`
- Neuron SLURM 레퍼런스와 환경 체크리스트에 BART/H200=10 cores, BLT/A100=8 cores로 분리 명시

### 결과
- `amd_a100nv_8`에서 `Requested CPU cores per node (10) exceed the allowed limit (8)` 오류가 나지 않도록 수정됨

### 다음 단계
- [ ] 최신 코드 pull 후 `sbatch scripts/train_blt.sh` 재제출

---

## [2026-06-05] Neuron 파티션 정책 갱신

### 목표
- BART와 BLT-GEC 학습 job을 지정 GPU 파티션으로 분리

### 수행 내용
- `scripts/train_bart.sh`: BART 학습 파티션을 `eme_h200nv_8`로 변경
- `scripts/eval_bart.sh`: BART 평가 파티션을 `eme_h200nv_8`로 변경
- `scripts/train_blt.sh`: BLT-GEC scaffold 학습 파티션을 `amd_a100nv_8`로 변경
- Neuron SLURM 레퍼런스와 환경 체크리스트의 기본 파티션 설명 갱신

### 결과
- BART 계열 job은 H200 파티션, BLT-GEC scaffold job은 A100 파티션으로 제출되도록 정리됨

### 다음 단계
- [ ] Neuron에서 `sinfo`로 두 파티션 상태 확인
- [ ] 기존 job이 있으면 취소 후 최신 스크립트로 재제출

---

## [2026-06-05] BLT-GEC 학습 scaffold 구현

### 목표
- 현재 데이터 구조(`data/<dataset>_train/dev/test.tsv`)와 Neuron SLURM 운영 방식에 맞는 BLT-GEC 학습용 코드 작성
- 2시간 제한에 대비한 checkpoint/resume 정책을 BLT-GEC 학습에도 적용

### 수행 내용
- `blt_gec/data_adapter.py`
  - 특수 토큰을 `BOS=256`, `EOS=257`, `SEP=258`, `PAD=259`로 정리
  - Prefix-LM causal label 구성으로 수정
  - `attention_mask` 반환 추가
- `blt_gec/model.py`
  - repo-local byte Prefix-LM causal Transformer scaffold 추가
  - 공식 BLT 연결 전 데이터/학습 루프 검증용 모델로 명시
- `blt_gec/train.py`
  - train/val/test TSV 경로 처리
  - masked byte-level cross entropy 학습
  - validation loss, `best.ckpt`, `last.ckpt`, 자동 resume 지원
  - `--max_time 00:01:50:00`, 20분 checkpoint 간격 지원
- `blt_gec/generate.py`
  - 학습 checkpoint에서 greedy byte generation으로 교정문 생성
- `scripts/train_blt.sh`
  - Neuron SLURM용 BLT-GEC 학습 job 추가
  - `outputs/blt_gec/<dataset>/last.ckpt` 자동 resume 적용
- `blt_gec/architecture.md`, pipeline/env 레퍼런스 문서 갱신

### 결과
- `sbatch scripts/train_blt.sh`로 BLT-GEC scaffold 학습을 제출할 수 있는 구조가 마련됨
- 공식 BLT fine-tuning 전에도 데이터 경로, loss masking, checkpoint/resume, 생성 루프를 end-to-end로 검증할 수 있음

### 다음 단계
- [ ] SLURM에서 작은 native subset으로 smoke run 실행
- [ ] `outputs/blt_gec/native/last.ckpt` 생성 및 재제출 resume 확인
- [ ] 공식 `ByteLatentTransformer` backend 연결 설계

---

## [2026-05-29] 2시간 SLURM 제한 대비 checkpoint/resume 안전장치

### 목표
- Neuron SLURM 2시간 제한으로 학습 job이 중단되어도 이어서 재개 가능하도록 수정
- 에이전트 지침과 레퍼런스 문서에 짧은 job 반복 제출 운영 원칙 반영

### 수행 내용
- `baseline/run.py`
  - `--resume_ckpt_path` 추가: 학습 재개용 checkpoint 경로
  - `--checkpoint_interval_minutes` 추가: 기본 20분 간격 재개용 checkpoint 저장
  - `--max_time` 추가: 기본 `00:01:50:00`으로 SLURM 제한 전에 Lightning이 먼저 정지
  - validation GLEU best checkpoint와 시간 기반 재개용 checkpoint 분리
  - 학습 종료 시 `outputs/<dataset>/last.ckpt`를 저장하는 콜백 추가
- `scripts/train_bart.sh`
  - `#SBATCH --time=01:55:00`으로 2시간 제한에 맞춤
  - `#SBATCH --signal=B:TERM@300`으로 종료 전 신호 요청
  - `outputs/<dataset>/last.ckpt`가 있으면 자동으로 `--resume_ckpt_path` 전달
  - `RESUME_CKPT=""`로 자동 재개 비활성화, `RESUME_CKPT=<path>`로 특정 checkpoint 재개 가능
- `.github/skills/korean-gec-dev/` 레퍼런스와 `SKILL.md`에 checkpoint/resume 운영 지침 추가

### 결과
- 동일한 `sbatch scripts/train_bart.sh` 명령을 반복 제출하면 마지막 checkpoint부터 이어서 학습하는 구조가 됨
- 2시간 제한 job에서도 최대 손실 구간을 약 20분 이내로 제한하고, 정상적인 Lightning 종료 시에는 마지막 상태를 한 번 더 저장

### 다음 단계
- [ ] Neuron에서 smoke run 제출 후 `outputs/native/last.ckpt` 생성 여부 확인
- [ ] 첫 재제출에서 `Auto-resume enabled` 로그와 global step 재개 여부 확인
- [ ] 필요하면 checkpoint 간격을 10분으로 줄이거나 batch size를 조정

---

## [2026-05-27] Neuron SLURM 가이드 반영

### 목표
- Neuron SLURM 작업 제출 가이드를 프로젝트 레퍼런스에 추가
- Baseline 학습/평가 스크립트를 Neuron 정책에 맞게 수정

### 수행 내용
- `.github/skills/korean-gec-dev/references/neuron-slurm.md` 추가
  - `/scratch/$USER` 제출 위치
  - `--comment=pytorch` 필수
  - `--mem` 미사용
  - `sinfo`, `squeue`, `scancel` 기반 모니터링 명령 정리
- `scripts/train_bart.sh`, `scripts/eval_bart.sh` 수정
  - 기본 파티션 `cas_v100_4`
  - `--ntasks-per-node=1`, CPU 요청 10 core, `--gres=gpu:1`
  - `srun python ...` 실행 방식 적용
  - `/scratch/$USER` 밖에서 제출하면 오류 처리
- `env-checklist.md`, `pipeline-reference.md`, `SKILL.md`, `data/README.md`의 SLURM 경로/레퍼런스 설명 갱신

### 결과
- 현재 SLURM 스크립트가 Neuron 사용자 가이드의 제출 규칙과 맞도록 정리됨
- `slurm-*.out`, `slurm-*.err` 런타임 로그를 `.gitignore`에 추가해 실험 로그 파일의 실수 커밋을 방지

### 다음 단계
- [ ] Neuron 로그인 노드에서 `/scratch/$USER/PHDQ2` 경로로 repo 배치
- [ ] `sinfo`로 사용 가능한 GPU 파티션 확인 후 필요 시 스크립트의 `#SBATCH -p` 수정
- [ ] `sbatch scripts/train_bart.sh`로 baseline smoke run 실행

---

## [2026-05-27] 문서 상태 정합성 수정 및 data ignore 규칙 정리

### 목표
- 실제 코드 상태와 `PROJECT_MASTER_PLAN.md`의 현재 상태 설명 불일치 해소
- `data/README.md`의 설명과 실제 `.gitignore` 정책 일치
- 로컬 데이터/실험 산출물이 git 추적 대상으로 잡히지 않도록 정리

### 수행 내용
- `PROJECT_MASTER_PLAN.md`: 2026-05-27 기준 현재 상태로 갱신
  - Track 1 Serving 코드(`infer.py`, `app.py`, `requirements.txt`) 구현 완료 반영
  - Track 2 SLURM 스크립트(`train_bart.sh`, `eval_bart.sh`) 작성 완료 반영
  - Track 3 BLT 데이터 어댑터(`data_adapter.py`) 초안 작성 완료 반영
  - 다음 실행 To-do를 배포 테스트, baseline 실행, BLT 학습 루프 연결 중심으로 수정
- `data/README.md`: 현재 로컬 데이터 구조(`Raw/`, `Preprocessed/`, `generated_outputs/`)와 git 추적 제외 방침 명시
- `.gitignore`: `data/` 제외 규칙 추가

### 결과
- master plan의 오래된 미완료 항목과 실제 구현 상태가 일치하도록 정리됨
- 로컬 데이터 및 실험 산출물이 기본적으로 git에 추가되지 않도록 설정됨

### 다음 단계
- [ ] Ubuntu RTX 5090에서 Serving 앱 실제 실행 검증
- [ ] SLURM에서 `scripts/train_bart.sh` 기준 baseline 1차 학습 실행
- [ ] `blt_gec/data_adapter.py`를 BLT 원본 학습 루프 입력 형식과 맞춰 확장

---

## [2026-04-27] Track 2 & 3 스캐폴딩 (SLURM 스크립트 및 데이터 어댑터)

### 목표
- PROJECT_MASTER_PLAN에 따른 다음 단계(Track 2, Track 3 준비) 수행
- SLURM 환경을 위한 Baseline 학습 스크립트 작성
- BLT 모델을 위한 데이터 어댑터 프로토타입 구현

### 수행 내용
- **Track 2 (Baseline)**
  - `scripts/train_bart.sh`: SLURM 작업 제출용 배치 스크립트 생성 (노드/메모리/시간/GPU 설정 및 파라미터 매핑)
  - `scripts/eval_bart.sh`: 모델 체크포인트 평가용 SLURM 스크립트 생성
  - `data/README.md`: 데이터셋 형태(TSV) 및 파일 명명 규칙(native/lang8/learner) 문서화
- **Track 3 (BLT-GEC)**
  - `blt_gec/data_adapter.py`: TSV 파일을 읽어 Prefix-LM용 UTF-8 바이트 시퀀스와 loss mask를 생성하는 `GecBltDataset` 어댑터 구현

### 결과
- SLURM 환경에서 곧바로 Baseline 실험을 돌릴 수 있는 실행 환경 세팅 완료
- 데이터 디렉토리 구조 확립 완료
- BLT 모델 어댑터의 뼈대 구성 완료

### 다음 단계
- [ ] 실제 학습 데이터(Kor-Native 등)를 `data/` 디렉토리에 배치
- [ ] SLURM 클러스터에서 `sbatch scripts/train_bart.sh` 실행 및 동작 확인
- [ ] `blt_gec/` 디렉토리 내에 모델 래핑 코드 추가 구현

---


## [2026-04-22] Track 1 Serving 코드 구현 + 종합 계획서

### 목표
- Track 1(Serving) 실행 코드 구현
- 전체 프로젝트 종합 실행 계획서 작성

### 수행 내용
- 체크포인트 확보 방법 확정: **HuggingFace `Soyoung97/gec_kr`** (논문 저자 공개)
  - 별도 다운로드 불필요, `from_pretrained()` 직접 사용
  - 논문 README 샘플 코드 기반 추론 파라미터: num_beams=4, repetition_penalty=2.0
- `serving/infer.py` 구현: GecInferenceEngine 클래스, diff 생성 유틸
- `serving/app.py` 구현: Gradio UI (단일/다중 문장 탭, diff 하이라이트)
- `serving/requirements.txt` 생성
- `serving/architecture.md` 업데이트: HF 모델 정보 반영, 실행 방법 갱신
- `PROJECT_MASTER_PLAN.md` 작성: 3트랙 12주 실행 계획, 주차별 로드맵

### 결과
- Serving 코드 완성 — Ubuntu RTX 5090에서 `python app.py` 한 줄로 실행 가능
- 종합 계획서 완성 — 마일스톤 6개, 주차별 작업 배분

### 다음 단계
- [ ] Ubuntu RTX 5090에서 serving 실행 테스트 (pull → pip install → python app.py)
- [ ] SLURM baseline 데이터 준비 착수

---
## [2026-04-27] Track 2 & 3 스캐폴딩 (SLURM 스크립트 및 데이터 어댑터)

### 목표
- PROJECT_MASTER_PLAN에 따른 다음 단계(Track 2, Track 3 준비) 수행
- SLURM 환경을 위한 Baseline 학습 스크립트 작성
- BLT 모델을 위한 데이터 어댑터 프로토타입 구현

### 수행 내용
- **Track 2 (Baseline)**
  - `scripts/train_bart.sh`: SLURM 작업 제출용 배치 스크립트 생성 (노드/메모리/시간/GPU 설정 및 파라미터 매핑)
  - `scripts/eval_bart.sh`: 모델 체크포인트 평가용 SLURM 스크립트 생성
  - `data/README.md`: 데이터셋 형태(TSV) 및 파일 명명 규칙(native/lang8/learner) 문서화
- **Track 3 (BLT-GEC)**
  - `blt_gec/data_adapter.py`: TSV 파일을 읽어 Prefix-LM용 UTF-8 바이트 시퀀스와 loss mask를 생성하는 `GecBltDataset` 어댑터 구현

### 결과
- SLURM 환경에서 곧바로 Baseline 실험을 돌릴 수 있는 실행 환경 세팅 완료
- 데이터 디렉토리 구조 확립 완료
- BLT 모델 어댑터의 뼈대 구성 완료

### 다음 단계
- [ ] 실제 학습 데이터(Kor-Native 등)를 `data/` 디렉토리에 배치
- [ ] SLURM 클러스터에서 `sbatch scripts/train_bart.sh` 실행 및 동작 확인
- [ ] `blt_gec/` 디렉토리 내에 모델 래핑 코드 추가 구현

---


## [2026-04-20] 3-트랙 구조 재편 (Serving 트랙 추가)

### 목표
- 프로젝트를 3트랙으로 확장: (1) Serving, (2) Baseline, (3) BLT-GEC

### 수행 내용
- `serving/` 디렉토리 생성 + `architecture.md` 작성
  - 기학습 KoBART GEC 체크포인트를 RTX 5090에서 로드 → Gradio 웹 UI
  - 추론 파이프라인, UI 레이아웃, 구현 단계 문서화
- SKILL.md 전면 업데이트: 2-트랙 → 3-트랙 구조, Phase 번호 재배정
  - Phase 3: GEC 서빙 (Track 1), Phase 4: Baseline (Track 2), Phase 5: BLT-GEC (Track 3)
- pipeline-reference.md: 디렉토리 구조도에 `serving/` 반영
- .gitignore: `serving/checkpoints/` 제외 추가

### 트랙 구조
| # | 트랙 | 디렉토리 | 실행 머신 | 상태 |
|---|------|---------|----------|------|
| 1 | Serving | `serving/` | Ubuntu RTX 5090 | 계획 완료 |
| 2 | Baseline | `baseline/` | SLURM | 코드 이식 완료 |
| 3 | BLT-GEC | `blt_gec/` | SLURM | 설계 완료 |

### 다음 단계
- [ ] Track 1: 체크포인트 확보 방법 확정, infer.py + app.py 구현
- [ ] Track 2: SLURM에서 baseline 학습 실행
- [ ] Track 3: 데이터 어댑터 구현

---
## [2026-04-27] Track 2 & 3 스캐폴딩 (SLURM 스크립트 및 데이터 어댑터)

### 목표
- PROJECT_MASTER_PLAN에 따른 다음 단계(Track 2, Track 3 준비) 수행
- SLURM 환경을 위한 Baseline 학습 스크립트 작성
- BLT 모델을 위한 데이터 어댑터 프로토타입 구현

### 수행 내용
- **Track 2 (Baseline)**
  - `scripts/train_bart.sh`: SLURM 작업 제출용 배치 스크립트 생성 (노드/메모리/시간/GPU 설정 및 파라미터 매핑)
  - `scripts/eval_bart.sh`: 모델 체크포인트 평가용 SLURM 스크립트 생성
  - `data/README.md`: 데이터셋 형태(TSV) 및 파일 명명 규칙(native/lang8/learner) 문서화
- **Track 3 (BLT-GEC)**
  - `blt_gec/data_adapter.py`: TSV 파일을 읽어 Prefix-LM용 UTF-8 바이트 시퀀스와 loss mask를 생성하는 `GecBltDataset` 어댑터 구현

### 결과
- SLURM 환경에서 곧바로 Baseline 실험을 돌릴 수 있는 실행 환경 세팅 완료
- 데이터 디렉토리 구조 확립 완료
- BLT 모델 어댑터의 뼈대 구성 완료

### 다음 단계
- [ ] 실제 학습 데이터(Kor-Native 등)를 `data/` 디렉토리에 배치
- [ ] SLURM 클러스터에서 `sbatch scripts/train_bart.sh` 실행 및 동작 확인
- [ ] `blt_gec/` 디렉토리 내에 모델 래핑 코드 추가 구현

---


## [2026-04-19] Baseline 코드 이식 및 Skills 업데이트

### 목표
- baseline/ 디렉토리에 BART GEC 실행 코드 이식 (PL 2.x 마이그레이션)
- skills 문서에 미반영 사항 반영 (PHDQ2 repo 상태, 논문 파일명)

### 수행 내용
- `baseline/run.py` 작성: PL 2.x Trainer 직접 생성, strategy='auto'
- `baseline/model.py` 작성: on_train_epoch_end, on_validation_epoch_end 등 PL 2.x API
- `baseline/dataset.py` 작성: GecDataModule + KoBARTGecDataset (변경 최소)
- `baseline/metric/` 복사: gleu.py, gleumodule.py, m2scorer/ (deprecated 제거)
- `baseline/requirements.txt` 작성: torch>=2.0, lightning>=2.0, transformers>=4.30
- Skills 업데이트:
  - SKILL.md: 논문 파일명 반영 (korean_gec.pdf, blt.pdf), phdq=PHDQ2 메타 추가
  - pipeline-reference.md: phdq/가 PHDQ2 repo임을 반영한 구조도 갱신

### PL 1.x → 2.x 마이그레이션 요약
| 원본 (PL 1.x) | 마이그레이션 (PL 2.x) |
|---|---|
| `training_epoch_end(outputs)` | `on_train_epoch_end()` |
| `validation_epoch_end(outputs)` | `on_validation_epoch_end()` + 수동 loss 축적 |
| `Trainer.from_argparse_args()` | `L.Trainer()` 직접 생성 |
| `strategy='dp'` | `strategy='auto'` |
| `transformers.optimization.AdamW` | `torch.optim.AdamW` |
| `import pytorch_lightning as pl` | `import lightning as L` |

### 다음 단계
- [ ] SLURM 환경에서 baseline 동작 확인
- [ ] 학습 데이터 배치 및 전처리
- [ ] blt_gec/ 데이터 어댑터 구현

---
## [2026-04-27] Track 2 & 3 스캐폴딩 (SLURM 스크립트 및 데이터 어댑터)

### 목표
- PROJECT_MASTER_PLAN에 따른 다음 단계(Track 2, Track 3 준비) 수행
- SLURM 환경을 위한 Baseline 학습 스크립트 작성
- BLT 모델을 위한 데이터 어댑터 프로토타입 구현

### 수행 내용
- **Track 2 (Baseline)**
  - `scripts/train_bart.sh`: SLURM 작업 제출용 배치 스크립트 생성 (노드/메모리/시간/GPU 설정 및 파라미터 매핑)
  - `scripts/eval_bart.sh`: 모델 체크포인트 평가용 SLURM 스크립트 생성
  - `data/README.md`: 데이터셋 형태(TSV) 및 파일 명명 규칙(native/lang8/learner) 문서화
- **Track 3 (BLT-GEC)**
  - `blt_gec/data_adapter.py`: TSV 파일을 읽어 Prefix-LM용 UTF-8 바이트 시퀀스와 loss mask를 생성하는 `GecBltDataset` 어댑터 구현

### 결과
- SLURM 환경에서 곧바로 Baseline 실험을 돌릴 수 있는 실행 환경 세팅 완료
- 데이터 디렉토리 구조 확립 완료
- BLT 모델 어댑터의 뼈대 구성 완료

### 다음 단계
- [ ] 실제 학습 데이터(Kor-Native 등)를 `data/` 디렉토리에 배치
- [ ] SLURM 클러스터에서 `sbatch scripts/train_bart.sh` 실행 및 동작 확인
- [ ] `blt_gec/` 디렉토리 내에 모델 래핑 코드 추가 구현

---


## [2026-04-19] 프로젝트 초기 구조 수립

### 목표
- 프로젝트 디렉토리 구조 확립
- 레퍼런스 코드 확보 (Standard_Korean_GEC, BLT)
- Baseline(BART)과 BLT-GEC 양쪽의 아키텍처 계획 문서화

### 수행 내용
- `reference_code/` 디렉토리 생성, 원본 코드 클론
  - `reference_code/Standard_Korean_GEC/` — BART GEC + KAGAS 어노테이션
  - `reference_code/blt/` — Byte Latent Transformer (facebookresearch)
- `baseline/architecture.md` 작성
  - KoBART GEC 재현 계획: 모델 구조, 데이터 파이프라인, 학습 루프, 평가 지표
  - 환경 마이그레이션 전략 (원본 torch==1.7.1 → 현대 버전)
  - 단계별 계획: 코드 이식 → 마이그레이션 → 학습 → 전 데이터셋 실험
- `blt_gec/architecture.md` 작성
  - Prefix-LM 방식으로 GEC 적용 결정 (BLT 구조 변경 최소화)
  - 데이터 어댑터 설계: TSV → UTF-8 바이트 → [BOS][src][SEP][tgt][EOS]
  - 한국어 바이트 특성 분석: 3bytes/char, 띄어쓰기=0x20, 엔트로피 기반 패칭 가설
  - 구현 5단계 계획: 데이터 → 모델래핑 → 학습 → 추론평가 → 최적화
- `.github/skills/korean-gec-dev/` 스킬 파일 업데이트
  - reference_code 로컬 경로 반영
  - 3-Machine 워크플로우 (Mac → SLURM → Ubuntu RTX 5090)
  - 환경 체크리스트 머신별 분리

### 결과
- 프로젝트 구조 확립:
  ```
  phdq/
  ├── reference_code/Standard_Korean_GEC/  (원본 참조)
  ├── reference_code/blt/                  (원본 참조)
  ├── baseline/architecture.md             (BART 재현 계획)
  ├── blt_gec/architecture.md              (BLT-GEC 설계 계획)
  ├── .github/skills/korean-gec-dev/       (코딩 가이드 스킬)
  └── LOG.md                               (이 로그)
  ```
- BLT-GEC 핵심 설계 결정: **Prefix-LM 방식** 채택 (1차)
- 원본 GEC 코드 분석 완료: run.py, model.py, dataset.py 전체 구조 파악

### 다음 단계
- [ ] PHDQ2 리포 초기화 및 첫 push
- [ ] baseline/ 코드 이식 시작 (reference → baseline)
- [ ] blt_gec/ 데이터 어댑터 구현 시작
- [ ] 논문 PDF에서 베이스라인 수치 확인

### 메모
- 원본 GEC 코드의 M2 scorer 부분이 주석 처리 상태 — 별도 실행 필요
- BLT 사전학습 가중치는 HuggingFace 접근 승인 필요 (facebook/blt-1b)
- BLT의 generate_nocache()가 beam search 미지원 — GEC용 beam search 구현 필요할 수 있음

---
## [2026-04-27] Track 2 & 3 스캐폴딩 (SLURM 스크립트 및 데이터 어댑터)

### 목표
- PROJECT_MASTER_PLAN에 따른 다음 단계(Track 2, Track 3 준비) 수행
- SLURM 환경을 위한 Baseline 학습 스크립트 작성
- BLT 모델을 위한 데이터 어댑터 프로토타입 구현

### 수행 내용
- **Track 2 (Baseline)**
  - `scripts/train_bart.sh`: SLURM 작업 제출용 배치 스크립트 생성 (노드/메모리/시간/GPU 설정 및 파라미터 매핑)
  - `scripts/eval_bart.sh`: 모델 체크포인트 평가용 SLURM 스크립트 생성
  - `data/README.md`: 데이터셋 형태(TSV) 및 파일 명명 규칙(native/lang8/learner) 문서화
- **Track 3 (BLT-GEC)**
  - `blt_gec/data_adapter.py`: TSV 파일을 읽어 Prefix-LM용 UTF-8 바이트 시퀀스와 loss mask를 생성하는 `GecBltDataset` 어댑터 구현

### 결과
- SLURM 환경에서 곧바로 Baseline 실험을 돌릴 수 있는 실행 환경 세팅 완료
- 데이터 디렉토리 구조 확립 완료
- BLT 모델 어댑터의 뼈대 구성 완료

### 다음 단계
- [ ] 실제 학습 데이터(Kor-Native 등)를 `data/` 디렉토리에 배치
- [ ] SLURM 클러스터에서 `sbatch scripts/train_bart.sh` 실행 및 동작 확인
- [ ] `blt_gec/` 디렉토리 내에 모델 래핑 코드 추가 구현

---


<!-- 아래 템플릿을 복사하여 새 세션을 추가하세요 -->
<!--
## [YYYY-MM-DD] 작업 제목

### 목표
-

### 수행 내용
-

### 결과
-

### 다음 단계
-

### 메모
-
-->
