# Baseline: KoBART GEC 재현 아키텍처

## 목표

Standard_Korean_GEC 원본 코드를 기반으로 BART 베이스라인을 **재현 가능한 상태**로 구성한다.
이 결과가 BLT-GEC 모델과의 성능 비교 기준점이 된다.

---

## 1. 모델 아키텍처

### 1-1. 모델 구성
```
입력 (오류 문장)
    │
    ▼ KoBART Tokenizer (PreTrainedTokenizerFast)
    │
    ▼ [BOS] + token_ids + [EOS] → padding (max_len=128)
    │
┌───┴───────────────────────────────┐
│  BartForConditionalGeneration      │
│  (pretrained: hyunwoongko/kobart)  │
│                                    │
│  Encoder: 오류 문장 인코딩          │
│  Decoder: 교정 문장 생성 (AR)       │
│  Loss: CrossEntropy (labels=-100   │
│         로 패딩 무시)               │
└───┬───────────────────────────────┘
    │
    ▼ Beam Search (num_beams=4, eos_token_id=1)
    │
    ▼ Tokenizer Decode
    │
출력 (교정 문장)
```

### 1-2. 핵심 하이퍼파라미터

| 파라미터 | 기본값 | 비고 |
|---------|--------|------|
| `max_seq_len` | 128 | 토큰 기준 최대 길이 |
| `batch_size` | 32 | |
| `lr` | 5e-05 | AdamW |
| `max_epochs` | 100 | (논문 실험에서는 10 사용) |
| `warmup_ratio` | 0.0 | |
| `weight_decay` | 0.01 | bias, LayerNorm 제외 |
| `num_beams` | 4 | 추론 시 beam search |

### 1-3. 옵티마이저/스케줄러
- **Optimizer**: AdamW (weight_decay=0.01, bias/LayerNorm 제외)
- **Scheduler**: Linear warmup → linear decay
- `num_train_steps = data_len * max_epochs / batch_size`

---

## 2. 데이터 파이프라인

### 2-1. 데이터 형식
```
# TSV (탭 구분): source\ttarget
한국어는어렵다.	한국어는 어렵다.
나는 학교에갔다	나는 학교에 갔다.
```

### 2-2. 전처리 흐름
```
원본 데이터 (native.txt, lang8, learner)
    │
    ▼ process_for_training.py → train_split
    │   (70/15/15 split, random_state=1)
    │
    ├── {data}_train.txt
    ├── {data}_val.txt
    └── {data}_test.txt
    │
    ▼ KoBARTGecDataset.__getitem__()
    │
    ├── input_ids:         pad(tokenize(source), max_len)
    ├── decoder_input_ids: pad([0] + tokenize(target) + [eos], max_len)  # shifted right
    └── labels:            pad(tokenize(target) + [eos], max_len, fill=-100)
```

### 2-3. 데이터셋

| 데이터셋 | 용도 | 특성 |
|---------|------|------|
| Kor-Native | 주 실험 | 원어민 오류 (띄어쓰기 중심) |
| Kor-Lang8 | 보조 | 외국인 학습자 오류, 노이즈 |
| Kor-Learner | 보조 | 한국어 학습자 체계적 오류 |
| Union | 종합 | 위 3개 합본 |

---

## 3. 학습 루프

### 3-1. 실행 흐름 (run.py)
```python
# 1. 모델·토크나이저 로드
model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')
tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')

# 2. 데이터 모듈 생성
dm = GecDataModule(args, tokenizer, KoBARTGecDataset)

# 3. Lightning 모듈 래핑
lit_model = KoBARTConditionalGeneration(args, model, tokenizer, dm)

# 4. Trainer 설정
trainer = pl.Trainer(
    accelerator='gpu', strategy='dp',
    max_epochs=args.max_epochs,
    callbacks=[ModelCheckpoint(monitor='val_gleu')]
)

# 5. 학습
trainer.fit(lit_model, dm)
```

### 3-2. Validation 단계 (매 에폭)
1. 모델 generate → hypothesis 텍스트 생성
2. `outputs/generation/epoch{N}/val/` 에 hyp/ref/src 파일 저장
3. GLEU 점수 계산 (`metric/gleumodule.py`)
4. (선택) M2 scorer 실행 → Precision/Recall/F0.5

---

## 4. 평가 지표

### 4-1. GLEU
- n-gram(n=4) 기반, source를 고려한 BLEU 변형
- `metric/gleumodule.py` → `run_gleu(ref, src, hyp)`
- GEC 표준 문장 수준 평가 지표

### 4-2. M2 Scorer
- `metric/m2scorer/` 활용
- KAGAS로 생성한 `.m2` 파일 기반
- Precision, Recall, F0.5 산출
- 원본 코드에서는 주석 처리 상태 (별도 실행 필요)

---

## 5. 재현 계획

### 5-1. 환경 호환성 전략

원본 코드는 매우 오래된 패키지 버전에 의존한다:
```
torch==1.7.1, transformers==4.8.1, pytorch-lightning==1.1.0
```

**전략 선택지:**

| 전략 | 장점 | 단점 |
|------|------|------|
| A. 원본 Docker (`msyoon8/default:gec`) | 최소 수정 | Docker 접근 필요, CUDA 호환 문제 |
| B. 패키지 버전 고정 + 패치 | 환경 제어 가능 | 다수 API 변경 대응 필요 |
| C. 현대 버전으로 마이그레이션 | 장기 유지보수 | 코드 수정량 많음 |

> **권장**: 전략 C (마이그레이션). BLT-GEC와 동일 환경에서 비교하려면 현대적 코드 필요.

### 5-2. 마이그레이션 시 변경 포인트

| 모듈 | 변경 필요 사항 |
|------|--------------|
| `model.py` | `pytorch-lightning` 2.x API 대응 (`training_epoch_end` → `on_train_epoch_end` 등) |
| `run.py` | `strategy='dp'` → `'auto'`, Trainer API 업데이트 |
| `dataset.py` | 큰 변경 없음 (순수 PyTorch) |
| `requirements.txt` | 전면 재작성 |
| `metric/` | GLEU 코드는 순수 Python, 그대로 사용 가능 |
| pretrained | `hyunwoongko/kobart` → HuggingFace 직접 로드 확인 |

### 5-3. 단계별 계획

```
Phase 1: 코드 이식
  └── reference_code/Standard_Korean_GEC → baseline/ 핵심 파일 복사·정리

Phase 2: 환경 마이그레이션
  └── 현대 패키지 버전에 맞춰 코드 수정
  └── SLURM 환경에서 동작 확인

Phase 3: 베이스라인 학습
  └── native 데이터로 학습 실행
  └── GLEU, M2 점수 기록

Phase 4: 전 데이터셋 실험
  └── lang8, learner, union으로 확대
  └── 논문 수치와 비교
```

---

## 6. 디렉토리 구조 (계획)

```
baseline/
├── architecture.md          # 이 파일
├── run.py                   # 학습 진입점 (마이그레이션)
├── model.py                 # KoBART GEC 모델
├── dataset.py               # 데이터 로더
├── requirements.txt         # 현대 버전 의존성
├── metric/
│   ├── gleu.py              # GLEU 점수 계산
│   ├── gleumodule.py        # GLEU 래퍼
│   └── m2scorer/            # M2 scorer
├── scripts/
│   └── train_bart.sh        # SLURM 학습 스크립트
└── configs/
    └── native.yaml          # 학습 설정
```

---

## 7. 재현 목표 수치 (논문 참조)

| 데이터셋 | GLEU | M2 F0.5 | 비고 |
|---------|------|---------|------|
| Kor-Native | TBD | TBD | 논문 Table 참조 |
| Kor-Lang8 | TBD | TBD | |
| Kor-Learner | TBD | TBD | |
| Union | TBD | TBD | |

> 논문 수치는 학습 후 채워 넣을 것. 논문 PDF(`korean_gec.pdf`)에서 확인.
