# BLT (Byte Latent Transformer) 아키텍처 레퍼런스

## 개요

BLT는 **토크나이저 없이 바이트(byte) 수준에서 직접 동작**하는 LLM 아키텍처.
기존 토크나이저 기반 모델과 동등한 성능을 달성하면서 추론 효율성과 robustness를 개선.

**핵심 아이디어**: 바이트를 엔트로피 기반으로 **동적 패치(dynamic patch)**로 그룹화하여
복잡한 부분에 더 많은 연산을 할당하고, 단순한 부분은 빠르게 처리.

## 아키텍처 구성 요소

```
입력 바이트 시퀀스
    │
    ▼
┌─────────────────────────┐
│  Entropy Model           │  ← facebook/blt-entropy
│  (LMTransformer)         │  각 바이트의 next-byte 엔트로피 계산
│  → 패치 경계 결정         │
└──────────┬──────────────┘
           │ 엔트로피 기반 패치 분할
           ▼
┌─────────────────────────┐
│  Local Encoder           │  바이트 → 패치 인코딩
│  (byte-level attention)  │  패치 내 바이트 간 attention
└──────────┬──────────────┘
           │ 패치 임베딩
           ▼
┌─────────────────────────┐
│  Global Latent           │  패치 단위 Transformer
│  Transformer             │  (핵심 연산, 가장 큰 모델)
└──────────┬──────────────┘
           │ 패치 표현
           ▼
┌─────────────────────────┐
│  Local Decoder           │  패치 → 바이트 디코딩
│  (byte-level attention)  │  바이트 단위 출력 생성
└──────────┬──────────────┘
           │
           ▼
출력 바이트 시퀀스
```

## 코드 구조 (facebookresearch/blt)

```
blt/
├── bytelatent/
│   ├── model/
│   │   └── blt.py              # ★ ByteLatentTransformer 메인 클래스
│   ├── transformer.py          # LMTransformer (엔트로피 모델 등)
│   ├── hf.py                   # HuggingFace 연동 (BltTokenizerAndPatcher)
│   ├── configs/
│   │   └── debug.yaml          # 디버그용 학습 설정
│   ├── train.py                # ★ 학습 진입점 (torchrun)
│   └── stool.py                # SLURM 실행 도구
├── apps/                       # 응용 코드
├── demo.py                     # 추론 데모
├── download_blt_weights.py     # 가중치 다운로드
└── requirements.txt
```

## 핵심 클래스/함수

### `ByteLatentTransformer` (bytelatent/model/blt.py)
- BLT 전체 모델 (encoder + global transformer + decoder)
- `from_pretrained()`: HuggingFace에서 가중치 로드
- 입력: 바이트 시퀀스 + 패치 정보
- 출력: 바이트 단위 확률 분포

### `LMTransformer` (bytelatent/transformer.py)
- 엔트로피 모델로 사용
- 각 위치의 next-byte 엔트로피를 계산
- 엔트로피가 높은 지점에서 패치 경계를 설정

### `BltTokenizerAndPatcher` (bytelatent/hf.py)
- `tokenizer_args`: 바이트 단위 "토크나이저" 설정
- `patcher_args`: 패치 분할 설정 (엔트로피 임계값 등)
- `from_pretrained()`: HuggingFace 리포에서 설정 로드

## GEC 적용 시 핵심 고려사항

### 1. seq2seq 구조 매핑
BLT는 원래 **autoregressive LM** (다음 바이트 예측). GEC는 **seq2seq** (오류문 → 교정문).
적용 방식 선택지:

| 방식 | 설명 | 장단점 |
|------|------|--------|
| **Prefix-LM** | `[오류문][SEP][교정문]` 연결 후 교정문만 생성 | 가장 단순, BLT 구조 변경 최소 |
| **Encoder-Decoder** | BLT encoder로 오류문 인코딩, decoder로 교정문 생성 | 성능 좋을 수 있으나 구조 수정 필요 |
| **Fine-tuning** | 사전학습된 BLT를 GEC 데이터로 fine-tune | Prefix-LM과 결합 |

### 2. 한국어 바이트 특성
- 한글 1글자 = UTF-8 3바이트 (예: "한" = `\xed\x95\x9c`)
- 영어 대비 동일 텍스트가 ~3배 긴 바이트 시퀀스
- BLT의 동적 패치가 이를 자연스럽게 처리할 수 있는지 검증 필요
- 자모 분리가 바이트 수준에서 자연스럽게 이루어짐 (초성·중성·종성)

### 3. 패치 분할과 GEC
- 오류 부분: 엔트로피 높음 → 짧은 패치 → 더 많은 연산 할당
- 정상 부분: 엔트로피 낮음 → 긴 패치 → 효율적 처리
- 이 특성이 GEC에 유리할 수 있음 (오류에 집중)

### 4. 모델 크기
| 모델 | 파라미터 | GPU 메모리 (추론) | GPU 메모리 (학습) |
|------|---------|-------------------|-------------------|
| blt-1b | 1B | ~4GB | ~16GB+ |
| blt-7b | 7B | ~14GB | ~56GB+ (multi-GPU) |

GEC fine-tuning은 blt-1b부터 시작 권장.

## 사전학습 가중치 로드 코드

```python
from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.hf import BltTokenizerAndPatcher

# 엔트로피 모델 (패치 경계 결정)
entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")

# BLT 메인 모델
blt_model = ByteLatentTransformer.from_pretrained("facebook/blt-1b")

# 토크나이저 & 패처
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-1b")
tokenizer = tok_and_patcher.tokenizer_args.build()
patcher = tok_and_patcher.patcher_args.build()
```

## BART vs BLT 비교표

| 항목 | BART (Baseline) | BLT (Development) |
|------|----------------|-------------------|
| 입력 단위 | 서브워드 토큰 | 바이트 (UTF-8) |
| 전처리 | 토크나이저 필수 | 없음 (raw bytes) |
| 패치/토큰 | 고정 길이 토큰 | 가변 길이 패치 (엔트로피 기반) |
| 어텐션 | 토큰 간 self-attention | byte↔byte (local) + patch↔patch (global) |
| 한국어 처리 | 서브워드 분할 (어휘 의존) | 바이트 분할 (어휘 독립) |
| OOV 처리 | UNK 토큰 가능 | 불가능 (모든 바이트 커버) |
| 사전학습 | KoBART (한국어 특화) | BLT (다국어 바이트) |
| GEC 적용 | seq2seq (native) | 구조 적응 필요 |
| 라이선스 | MIT (modified) | CC-BY-NC-4.0 |
