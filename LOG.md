# 한국어 GEC 프로젝트 진행 로그

> 이 파일은 `/korean-gec-dev` 스킬에 의해 자동 관리됩니다.
> 최신 항목이 위에 오도록 역순으로 기록합니다.

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
  - `--ntasks-per-node=1`, `--cpus-per-task=10`, `--gres=gpu:1`
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
