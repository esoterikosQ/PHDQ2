# 한국어 GEC 프로젝트 진행 로그

> 이 파일은 `/korean-gec-dev` 스킬에 의해 자동 관리됩니다.
> 최신 항목이 위에 오도록 역순으로 기록합니다.

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
