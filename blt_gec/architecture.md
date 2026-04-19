# BLT-GEC: Byte Latent Transformer 기반 한국어 GEC 모델 아키텍처

## 목표

BLT(Byte Latent Transformer)를 한국어 GEC 태스크에 적용하여
KoBART(BART) 베이스라인 대비 성능 향상을 검증한다.

---

## 1. 핵심 설계 결정

### 1-1. 왜 BLT인가?

| 특성 | BART (Baseline) | BLT (제안) | GEC에서의 이점 |
|------|----------------|------------|--------------|
| 입력 단위 | 서브워드 토큰 | 바이트 (UTF-8) | OOV 없음, 오타/신조어에 강건 |
| 전처리 | 토크나이저 필수 | 없음 | 전처리 오류 제거 |
| 연산 배분 | 모든 토큰 동일 | 엔트로피 기반 동적 할당 | **오류 부분에 더 많은 연산** |
| 띄어쓰기 | 토큰 경계에 암묵적 | 바이트 0x20으로 명시적 | 띄어쓰기 오류 직접 모델링 |
| 한글 자모 | 서브워드에 섞임 | UTF-8 바이트로 자연 분리 | 자모 수준 오류 포착 |

### 1-2. BLT의 GEC 적용 방식: Prefix-LM

BLT는 원래 **autoregressive causal LM** (다음 바이트 예측).
GEC는 **seq2seq** (오류문 → 교정문).

**선택: Prefix-LM 방식**

```
입력 형식:
[오류문 바이트들] [SEP] [교정문 바이트들] [EOS]

학습 시:
- [오류문] 부분: loss 계산하지 않음 (prefix, mask 처리)
- [SEP] 이후: 교정문 바이트 예측으로 loss 계산

추론 시:
- [오류문 바이트들] [SEP] 까지 입력
- 이후 autoregressive하게 교정문 바이트 생성
- [EOS] 나올 때까지 생성
```

**근거:**
- BLT 구조 변경 최소화 (encoder-decoder 분리 불필요)
- 원본 BLT의 학습 코드/가중치를 그대로 활용 가능
- Prefix 부분이 오류문의 "인코딩" 역할을 수행
- 엔트로피 기반 패칭이 오류/정상 부분을 자연스럽게 구분

### 1-3. 대안 (Encoder-Decoder) — 보류

| 항목 | Prefix-LM | Encoder-Decoder |
|------|-----------|-----------------|
| 구현 난이도 | 낮음 | 높음 (BLT 구조 수정) |
| 사전학습 활용 | 그대로 | 불완전 (decoder만 활용) |
| 양방향 인코딩 | 불가 (causal) | 가능 |
| 긴 입력 효율 | 전체가 하나의 시퀀스 | 인코더/디코더 분리 |

> 1차 실험은 Prefix-LM으로 진행. 성능이 불충분하면 Encoder-Decoder 검토.

---

## 2. 모델 아키텍처

### 2-1. 전체 흐름

```
오류 문장: "한국어는어렵다."
    │
    ▼ UTF-8 인코딩 (토크나이저 없음)
    │
    ▼ bytes: [0xed,0x95,0x9c, 0xea,0xb5,0xad, ...] + [SEP bytes]
    │
    ▼ Entropy Model (facebook/blt-entropy)
    │   각 바이트의 next-byte 엔트로피 계산
    │   → 엔트로피 높은 곳 = 패치 경계
    │
    ▼ Dynamic Patching
    │   "한국"=[patch1], "어는"=[patch2], "어렵"=[patch3], ...
    │   오류 부분 → 짧은 패치 (더 세밀)
    │   정상 부분 → 긴 패치 (효율적)
    │
┌───┴─────────────────────────────────────┐
│  ByteLatentTransformer                   │
│                                          │
│  ┌──────────────┐                       │
│  │ Local Encoder │  바이트 → 패치 임베딩  │
│  │ (byte attn)   │                       │
│  └──────┬───────┘                       │
│         ▼                                │
│  ┌──────────────────┐                   │
│  │ Global Transformer │  패치 단위 처리   │
│  │ (main compute)     │                  │
│  └──────┬─────────────┘                  │
│         ▼                                │
│  ┌──────────────┐                       │
│  │ Local Decoder │  패치 → 바이트 출력    │
│  │ (byte attn)   │                       │
│  └──────┬───────┘                       │
│         ▼                                │
│  Byte-level logits (vocab_size=260)      │
└───┬─────────────────────────────────────┘
    │
    ▼ Autoregressive Generation (SEP 이후)
    │
    ▼ UTF-8 디코딩
    │
교정 문장: "한국어는 어렵다."
```

### 2-2. 모델 설정 (초기 계획)

| 파라미터 | 값 | 근거 |
|---------|------|------|
| 사전학습 모델 | `facebook/blt-1b` | 가장 작은 공개 모델, fine-tuning 적합 |
| 엔트로피 모델 | `facebook/blt-entropy` | 패치 경계 결정 |
| `patching_mode` | `"entropy"` | 동적 패치 (GEC에 최적) |
| `vocab_size` | 260 | 256 bytes + 4 special tokens |
| `max_seqlen` | 2048 | 한글 3bytes/char × ~300자 + 교정문 |
| `dim_global` | 원본 유지 | fine-tuning이므로 구조 변경 없음 |
| `cross_attn_encoder` | `true` | 원본 유지 |
| `cross_attn_decoder` | `true` | 원본 유지 |

### 2-3. 특수 토큰 설계

```python
BYTE_VOCAB_SIZE = 256  # 0x00 ~ 0xFF
BOS_ID = 256           # 시퀀스 시작
EOS_ID = 257           # 시퀀스 끝
SEP_ID = 258           # 오류문 ↔ 교정문 경계
PAD_ID = 259           # 패딩

# 입력 형식:
# [BOS] [오류문 bytes...] [SEP] [교정문 bytes...] [EOS] [PAD...]
```

---

## 3. 데이터 파이프라인

### 3-1. GEC 데이터 → BLT 입력 변환

```
원본 GEC TSV:
  "한국어는어렵다.\t한국어는 어렵다."

           ▼ GecBltDataset

BLT 입력:
  x: [BOS, 0xed,0x95,0x9c, ..., SEP, 0xed,0x95,0x9c, ..., 0x2e, EOS, PAD, PAD, ...]
  y: [0xed,0x95,0x9c, ..., SEP, 0xed,0x95,0x9c, ..., 0x2e, EOS, PAD, PAD, ...]  # x를 1칸 shift
  mask: [0,0,...,0, 1,1,...,1, 0,0,...]  # SEP 이후 교정문만 loss 계산
  patch_lengths: entropy model로 동적 계산
```

### 3-2. 데이터 어댑터 설계

```python
class GecBltDataset:
    """GEC TSV 데이터를 BLT 바이트 시퀀스로 변환"""

    def __init__(self, filepath, max_seq_len=2048):
        # TSV 파일에서 (source, target) 쌍 로드
        ...

    def __getitem__(self, idx):
        source, target = self.pairs[idx]

        # UTF-8 바이트로 변환
        src_bytes = list(source.encode('utf-8'))
        tgt_bytes = list(target.encode('utf-8'))

        # [BOS] + src_bytes + [SEP] + tgt_bytes + [EOS]
        x = [BOS_ID] + src_bytes + [SEP_ID] + tgt_bytes + [EOS_ID]

        # y = x shifted left by 1
        y = x[1:] + [PAD_ID]

        # mask: SEP 이후만 1 (교정문 부분만 loss 계산)
        sep_pos = len(src_bytes) + 1  # BOS 포함
        mask = [0] * (sep_pos + 1) + [1] * (len(tgt_bytes) + 1)

        # 패딩
        ...

        return {"x": x, "y": y, "mask": mask}
```

### 3-3. 패칭 통합

두 가지 방식 선택 가능:

| 방식 | 설명 | 장점 |
|------|------|------|
| **Realtime patching** | forward() 안에서 entropy model로 패치 계산 | 구현 단순 |
| **Pre-computed patching** | 데이터 전처리 시 패치 미리 계산·저장 | 학습 속도 빠름 |

> 1차: Realtime patching (`patch_in_forward=True`)으로 시작.
> 데이터가 크면 pre-computed로 전환.

---

## 4. 학습 전략

### 4-1. Fine-tuning 접근

```
사전학습된 BLT (facebook/blt-1b, 영어 중심)
    │
    ▼ GEC 데이터로 Fine-tuning
    │
    ├── Stage 1: 전체 파라미터 동결, decoder 마지막 레이어만 학습 (warmup)
    ├── Stage 2: 전체 모델 fine-tuning (낮은 lr)
    └── (선택) Stage 3: GEC 데이터 증강 후 추가 학습
```

### 4-2. 학습 하이퍼파라미터 (초기)

| 파라미터 | 값 | 비고 |
|---------|------|------|
| `lr` | 1e-05 | fine-tuning이므로 낮게 |
| `batch_size` | 4~8 | GPU 메모리에 따라 조절 |
| `max_steps` | TBD | 데이터 크기에 따라 |
| `grad_acc_steps` | 4~8 | effective batch size 확보 |
| `warmup_ratio` | 0.1 | |
| `weight_decay` | 0.01 | |
| `gradient_clipping` | 1.0 | |
| `precision` | bf16 | 메모리 절약 |

### 4-3. Loss 함수

```python
# Prefix-LM 스타일: SEP 이후 바이트만 loss 계산
def compute_gec_loss(logits, targets, mask):
    """
    logits: (batch, seq_len, vocab_size=260)
    targets: (batch, seq_len)
    mask: (batch, seq_len) — 1 for correction bytes, 0 for source/padding
    """
    loss = F.cross_entropy(
        logits.view(-1, 260),
        targets.view(-1),
        reduction='none'
    )
    loss = (loss * mask.view(-1)).sum() / mask.sum()
    return loss
```

---

## 5. 추론 (Generation)

### 5-1. 추론 흐름

```python
def generate_correction(model, patcher, error_text, max_gen_len=512):
    # 1. 오류문을 바이트로 변환
    src_bytes = [BOS_ID] + list(error_text.encode('utf-8')) + [SEP_ID]

    # 2. Autoregressive generation
    generated = []
    input_bytes = src_bytes
    for _ in range(max_gen_len):
        # 패칭
        patch_lengths = patcher.patch(input_bytes)

        # Forward
        logits = model(input_bytes, patch_lengths=patch_lengths)

        # 마지막 바이트의 다음 예측
        next_byte = logits[-1].argmax()

        if next_byte == EOS_ID:
            break

        generated.append(next_byte)
        input_bytes.append(next_byte)

    # 3. 바이트 → 텍스트
    correction = bytes(generated).decode('utf-8', errors='replace')
    return correction
```

### 5-2. 디코딩 전략

| 전략 | 설명 | 사용 시점 |
|------|------|----------|
| Greedy | argmax | 빠른 테스트 |
| Beam Search | BLT에 맞게 구현 필요 | 최종 평가 |
| Sampling | temperature + top-k/top-p | 다양성 필요 시 |

> BLT의 `generate_nocache()` 함수를 GEC에 맞게 수정하여 사용.

---

## 6. 평가

### 6-1. 공통 평가 파이프라인 (baseline과 동일)

```
BLT-GEC 출력
    │
    ▼ UTF-8 디코딩 → 교정 텍스트
    │
    ├── GLEU 점수 (metric/gleumodule.py)
    │
    ├── KAGAS → .m2 파일 생성
    │     └── M2 Scorer → Precision/Recall/F0.5
    │
    └── (추가) 바이트 수준 정확도, 생성 길이 분포
```

### 6-2. BLT 고유 분석

| 분석 항목 | 목적 |
|----------|------|
| 패치 길이 분포 | 한국어에서 패치가 어떻게 형성되는지 |
| 오류 위치 vs 패치 경계 | 오류 부분에 패치가 세밀한지 |
| 바이트 생성 정확도 | UTF-8 유효성 (깨진 문자 비율) |
| 추론 속도 비교 | BART 대비 생성 시간 |

---

## 7. 한국어 특수 고려사항

### 7-1. UTF-8 바이트 구조

```
한글 1글자 = 3 바이트 (U+AC00 ~ U+D7A3)
예: "한" = U+D55C → 0xED 0x95 0x9C

ASCII (영문, 숫자, 구두점) = 1 바이트
예: "." = 0x2E, " " = 0x20

한글 300자 문장 ≈ 900 바이트 + 구두점/공백
오류문 + 교정문 = ~1800 바이트 + 특수토큰
→ max_seqlen=2048이면 충분
```

### 7-2. 띄어쓰기 오류와 바이트

```
오류: "한국어는어렵다"     → [..., 0x9C, 0xEB, ...]  (공백 없음)
정답: "한국어는 어렵다"    → [..., 0x9C, 0x20, 0xEB, ...]  (0x20 삽입)

BLT는 바이트 0x20의 삽입/삭제를 직접 학습 가능
→ 띄어쓰기 오류에 대해 BART보다 유리할 수 있음
```

### 7-3. 엔트로피와 오류

```
정상 텍스트: 엔트로피 낮음 → 긴 패치 → 빠른 처리
오류 텍스트: 엔트로피 높음 → 짧은 패치 → 세밀한 처리

가설: BLT의 엔트로피 기반 패칭이 GEC에서
     "오류 부분에 자동으로 집중"하는 효과를 가짐
→ 실험으로 검증 필요
```

---

## 8. 구현 단계 (계획)

```
Phase 1: 데이터 어댑터
  ├── GecBltDataset 구현
  ├── TSV → 바이트 시퀀스 변환
  ├── Prefix-LM 형식 마스킹
  └── 단위 테스트 (바이트 변환 정확성)

Phase 2: 모델 래핑
  ├── BLT 가중치 로드
  ├── GEC용 loss 함수 (마스크 적용)
  ├── 학습 루프 작성
  └── 소량 데이터로 overfitting 테스트

Phase 3: 학습
  ├── SLURM 스크립트 작성
  ├── native 데이터로 fine-tuning
  ├── 에폭별 GLEU 모니터링
  └── 체크포인트 저장

Phase 4: 추론·평가
  ├── 생성 함수 구현
  ├── GLEU / M2 평가
  ├── baseline(BART)과 비교
  └── 오류 유형별 분석

Phase 5: 최적화 (성능에 따라)
  ├── 하이퍼파라미터 탐색
  ├── 데이터 증강
  ├── Encoder-Decoder 방식 검토
  └── blt-7b 스케일업
```

---

## 9. 디렉토리 구조 (계획)

```
blt_gec/
├── architecture.md          # 이 파일
├── data/
│   ├── dataset.py           # GecBltDataset
│   └── utils.py             # 바이트 변환 유틸리티
├── model/
│   ├── blt_gec.py           # BLT-GEC 모델 래퍼
│   └── loss.py              # GEC용 loss 함수
├── train.py                 # 학습 진입점
├── generate.py              # 추론/생성
├── evaluate.py              # 평가 파이프라인
├── configs/
│   ├── gec_native.yaml      # native 데이터 학습 설정
│   └── gec_union.yaml       # union 데이터 학습 설정
├── scripts/
│   └── train_blt.sh         # SLURM 학습 스크립트
└── requirements.txt         # BLT 환경 의존성
```

---

## 10. 리스크 및 대안

| 리스크 | 영향 | 대안 |
|--------|------|------|
| BLT 사전학습이 영어 중심 | 한국어 성능 저하 | 한국어 데이터로 continued pretraining |
| Prefix-LM이 seq2seq에 비효율 | 긴 입력에서 성능 하락 | Encoder-Decoder 방식으로 전환 |
| UTF-8 깨진 문자 생성 | 디코딩 실패 | 바이트 유효성 검사 + constrained generation |
| GPU 메모리 부족 (1B 모델) | 학습 불가 | gradient checkpointing, LoRA, QLoRA |
| 엔트로피 모델 한국어 부정확 | 패치 경계 잘못 설정 | space patching으로 대체, 또는 한국어로 재학습 |
