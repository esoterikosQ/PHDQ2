# 한국어 GEC 파이프라인 레퍼런스

## 프로젝트 구조 (두 트랙, 3-Machine)

> **phdq/ 워크스페이스 = esoterikosQ/PHDQ2 git repo** (origin → GitHub)
> `reference_code/`는 `.gitignore`로 제외 (로컬 전용 참조)

```
phdq/  (= PHDQ2 git repo)
├── .github/skills/korean-gec-dev/      # 이 스킬
├── reference_code/                     # ★ 원본 코드 (로컬 전용, .gitignore)
│   ├── Standard_Korean_GEC/            #   BART GEC + KAGAS 원본
│   └── blt/                            #   BLT 원본 (facebookresearch)
├── baseline/                           # [Track A] BART 베이스라인 (마이그레이션 코드)
│   ├── architecture.md                 #   아키텍처·재현 계획
│   ├── run.py                          #   학습 진입점
│   ├── model.py                        #   KoBART GEC 모델
│   ├── dataset.py                      #   데이터 로더
│   ├── requirements.txt                #   현대 버전 의존성
│   └── metric/                         #   GLEU, M2 scorer
├── blt_gec/                            # [Track B] BLT-GEC 개발 모델
│   └── architecture.md                 #   설계 계획
├── papers/                             # 논문 PDF
│   ├── korean_gec.pdf                  #   GEC 논문
│   ├── blt.pdf                         #   BLT 논문
│   └── s10994-021-06034-2.pdf          #   참고 논문
├── LOG.md                              # 진행 로그
├── .gitignore                          # reference_code/, 체크포인트 등 제외
└── (향후 추가)
    ├── scripts/                        # SLURM job 스크립트
    ├── ui/                             # UI 서빙 코드
    ├── results/                        # 학습·평가 결과
    ├── configs/                        # 학습 설정 파일
    └── eval/                           # 공통 평가 파이프라인
```

### 머신별 작업 영역
| 머신 | 주 작업 디렉토리 | 역할 |
|------|------------------|------|
| Mac (로컬) | phdq/ 전체 편집 | 코드 작성 → `git push` (origin = PHDQ2) |
| SLURM 노드 | `baseline/`, `blt_gec/`, `scripts/` | `git pull` → 학습/추론 → `results/` push |
| Ubuntu (RTX 5090) | `ui/`, `results/` | `git pull` → 모델 서빙·UI |

---

## Baseline 전체 구조 (reference_code/Standard_Korean_GEC)

```
Standard_Korean_GEC/
├── get_data/                    # [Phase A] 데이터 수집·전처리
│   ├── filter.py                # Lang8 데이터 필터링 (KoBERT 활용)
│   ├── process_for_training.py  # train/val/test 분할, union 파일 생성
│   └── korean_learner/          # Kor-Learner 코퍼스 빌드
├── KAGAS/                       # [Phase B] 어노테이션 (m2 파일 생성)
│   ├── parallel_to_m2_korean.py # ★ 진입점: orig+cor → m2 변환
│   ├── scripts/
│   │   ├── align_text_korean.py # ★ 핵심: 정렬·분류·병합 로직
│   │   └── rdlextra.py          # Damerau-Levenshtein 거리 계산
│   ├── aff-dic/                 # Hunspell 한국어 사전 (ko.aff, ko.dic)
│   └── sample_test_data/        # 테스트용 샘플
├── src/KoBART-gec/              # [Phase C-원본] 모델 코드 (deprecated)
├── dataset.py                   # [Phase C] 데이터셋 로더
├── model.py                     # [Phase C] KoBART GEC 모델 정의
├── run.py                       # [Phase C] ★ 학습 진입점
├── eval/                        # [Phase D] 평가 (GLEU)
├── metric/                      # [Phase D] 메트릭 (m2scorer)
└── requirements.txt             # 의존성
```

## 데이터 흐름 (A → B → C → D)

```
[A] 데이터 수집 (공통)
    Lang8 raw → filter.py → kor_lang8.txt
    NIKL corpus → korean_learner/ → kor_learner.txt
    Native corpus → (수동 다운로드) → native.txt
         │
         ▼
[A-2] 학습 데이터 분할 (공통)
    process_for_training.py
    ├── {data}_train.txt  (orig \t corrected 쌍)
    ├── {data}_val.txt
    └── {data}_test.txt
         │
    ┌────┴──────────────────────┐
    │                          │
    ▼                          ▼
[B] KAGAS 어노테이션         [C-BLT] BLT 학습
    (평가용 m2 파일 생성)      ├─ 데이터 어댑터: txt → bytes
    orig + cor → output.m2     ├─ 엔트로피 모델로 패치 생성
    │                          ├─ ByteLatentTransformer 학습
    │                          └─ 출력: 교정된 바이트 → 텍스트
    │                          │
    ▼                          ▼
[C-BART] BART 학습            [D] 평가 (공통)
    run.py → model.py            ├─ GLEU
    + dataset.py                  ├─ M2 Scorer (P/R/F0.5)
    │                            └─ BART vs BLT 비교
    │                              │
    └─────────▶ [D]              ▼
                               esoterikosQ/PHDQ2 push
```

## 핵심 파일별 역할

### `run.py` (학습 진입점)
- argparse로 하이퍼파라미터 수신
- `dataset.py`에서 데이터 로드
- `model.py`에서 모델 생성
- PyTorch Lightning Trainer로 학습 실행
- 에폭마다 validation 시 GLEU 평가 자동 수행

### `model.py` (모델 정의)
- `BartForConditionalGeneration` 래핑
- KoBART pretrained 가중치 로드
- `training_step`, `validation_step` 정의

### `dataset.py` (데이터 로더)
- txt 파일에서 `orig \t corrected` 쌍 읽기
- KoBART 토크나이저로 인코딩
- PyTorch DataLoader 반환

### `KAGAS/parallel_to_m2_korean.py` (어노테이션 진입점)
- orig, cor 파일을 읽어 문장 쌍 구성
- `align_text_korean.py`의 함수들 호출
- m2 포맷으로 출력 (오류 위치·유형·교정)

### `KAGAS/scripts/align_text_korean.py` (핵심 정렬 로직)
- `getAutoAlignedEdits()`: 원문-교정문 정렬 (비용 기반)
- `token_substitution_korean()`: 토큰 치환 비용 계산
- `classify_output()` in `ErrorAnalyzer`: 오류 유형 분류
- `merge_ws()`: 띄어쓰기 오류 병합 처리

## 데이터셋 정보

| 데이터셋 | 출처 | 특성 | 라이선스 |
|----------|------|------|----------|
| Kor-Lang8 | Lang-8 학습자 코퍼스 | 외국인 학습자 오류, 노이즈 많음 | 연구용 |
| Kor-Learner | NIKL 코퍼스 | 한국어 학습자 체계적 오류 | 비상업적 |
| Kor-Native | 교육원+국립국어원 | 원어민 오류 (띄어쓰기 등) | 비상업적 |

## 평가 지표

- **GLEU**: 문장 수준 평가 (GEC 특화 BLEU 변형)
- **M2 Scorer**: 편집 수준 평가
  - Precision: 올바른 교정 / 전체 교정
  - Recall: 올바른 교정 / 전체 오류
  - F0.5: Precision 가중 조화 평균 (GEC에서는 Precision 중시)

## 알려진 의존성

### Baseline (BART) 환경
- `transformers` (BartForConditionalGeneration, PreTrainedTokenizerFast)
- `pytorch-lightning`
- `kobart-transformers` (KoBART 가중치)
- `KoNLPy` (KKma 형태소 분석기, Java 필요)
- `cyhunspell` (Hunspell 파이썬 래퍼)
- `KoBERT` (데이터 필터링에 사용)

### Development (BLT) 환경
- Python 3.12
- PyTorch nightly (CUDA 12.1)
- `xformers` (specific commit: `de742ec3d64bd83b1184cc043e541f15d270c148`)
- `ninja` (xformers 빌드 필요)
- `bytelatent` (메인 BLT 패키지, `facebookresearch/blt`)
- HuggingFace 계정 + 가중치 접근 승인 (`facebook/blt-1b`, `facebook/blt-entropy`)

> ⚠️ **환경 주의**: Baseline(Python 3.7~3.8)과 BLT(Python 3.12)는 별도 가상환경 필수
