# 한국어 GEC 프로젝트 진행 로그

> 이 파일은 `/korean-gec-dev` 스킬에 의해 자동 관리됩니다.
> 최신 항목이 위에 오도록 역순으로 기록합니다.

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
