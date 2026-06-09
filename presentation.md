# 발표 계획: 한국어 GEC에서 KoBART 재현과 BLT 적용

## 발표 방향

이 발표는 신경망 구조를 한국어 문법 오류 교정(GEC)에 적용하고 확장하는 과정을 중심으로 구성한다. 다음 네 가지를 발표의 축으로 둔다.

1. KAGAS가 한국어 GEC 데이터를 어떻게 구성하고 평가 가능한 형태로 만드는지 설명한다.
2. KoBART 기반 baseline을 어떤 모델 구조와 fine-tuning 방식으로 재현했는지 설명한다.
3. BLT(Byte Latent Transformer)를 한국어 GEC에 적용한 구조, 학습 방식, 자원 비용, 현재 성능 상태를 설명한다.
4. 학습한 모델을 UI로 감싸 실제 교정 시스템으로 사용하는 흐름과 이후 구현 과제를 제시한다.

전체 발표 시간은 약 15분을 목표로 한다. 질문이나 데모 설명을 포함하면 17-18분까지 확장 가능하다.

핵심 메시지:

> 한국어 GEC benchmark의 데이터/평가 체계(KAGAS)를 기준으로 KoBART baseline을 재현하고, byte-level dynamic patching을 갖는 BLT를 한국어 GEC 모델로 확장하는 실험 파이프라인을 구축했다.

---

## 발표 제목 후보

- 한국어 문법 오류 교정에서 KoBART 재현과 BLT 적용
- KAGAS 기반 한국어 GEC 평가와 BLT 모델 적용 실험
- Subword 기반 KoBART와 Byte Latent Transformer의 한국어 GEC 적용 비교

---

## 전체 구성

| 파트 | 주제 | 권장 시간 |
|---|---:|---:|
| 1 | KAGAS와 한국어 GEC 데이터/평가 구조 | 3분 |
| 2 | KoBART baseline 재현과 fine-tuning | 4분 |
| 3 | BLT 구조와 한국어 GEC 적용 | 5분 |
| 4 | UI 래핑, 시연 화면, 이후 구현 | 3분 |

---

## Part 1. KAGAS와 한국어 GEC Benchmark

### 1. 문제 정의: 한국어 GEC

목표:
- 입력: 문법/띄어쓰기/철자/조사/어미 오류가 포함된 한국어 문장
- 출력: 자연스럽고 문법적으로 교정된 문장

예시:

```text
입력: 한국어는어렵다.
출력: 한국어는 어렵다.
```

한국어 GEC가 어려운 이유:
- 띄어쓰기 규칙이 복잡하다.
- 교착어 특성상 조사, 어미, 활용 오류가 빈번하다.
- 표면 문자열 변화가 작아도 의미/문법 변화가 크다.
- 영어 ERRANT 같은 기존 자동 오류 주석 체계를 그대로 쓰기 어렵다.

발표 포인트:
- 한국어 GEC는 단순 번역/요약이 아니라, 작은 edit를 정확히 선택해야 하는 correction task다.

### 2. 기준 논문과 KAGAS의 역할

기준 논문:
- Soyoung Yoon et al., “Towards standardizing Korean Grammatical Error Correction: Datasets and Annotation”

논문의 핵심 기여:
- Kor-Learner, Kor-Native, Kor-Lang8 병렬 코퍼스 구축
- 한국어 오류 유형에 맞춘 KAGAS 제안
- KoBART baseline과 Hanspell 비교

KAGAS:
- Korean Automatic Grammatical error Annotation System
- 오류문과 정답문 pair를 입력받아 edit alignment 수행
- 각 edit에 한국어 오류 유형을 자동 부여
- M2 scorer에 넣을 수 있는 annotation을 생성

발표 포인트:
- KAGAS는 neural model이 아니라, 한국어 GEC 데이터를 분석하고 평가 가능한 형태로 바꾸는 자동 주석/평가 도구다.

### 3. KAGAS 구조와 구현 방식

KAGAS 처리 흐름:

```text
오류 문장 / 정답 문장 pair
    │
    ▼
문장 단위 alignment
    │
    ▼
word-level edit 추출
    │
    ▼
한국어 형태소/POS 기반 edit classification
    │
    ▼
14개 오류 유형 + UNK 부여
    │
    ▼
M2 형식 annotation 생성
```

KAGAS 오류 유형:
- `INS`: 삽입
- `DEL`: 삭제
- `WS`: 띄어쓰기
- `WO`: 어순
- `SPELL`: 철자
- `PUNCT`: 문장부호
- `SHORT`: 형태소 구조를 크게 바꾸지 않는 축약/표기 변화
- `VERB`, `ADJ`, `NOUN`, `PART`, `END`, `MOD`, `CONJ`

논문에서 보고한 KAGAS 품질:

| 데이터셋 | Coverage | Overall acceptance rate |
|---|---:|---:|
| Kor-Learner | 81.56% | 87.34% |
| Kor-Native | 90.92% | 93.93% |
| Kor-Lang8 | 82.52% | 87.06% |

발표 포인트:
- KAGAS는 완전한 정답 생성기가 아니라, 오류 유형별 평가와 M2 계산을 가능하게 하는 benchmark infrastructure다.

### 4. 데이터 구성과 평가 지표

논문 데이터셋:

| 데이터셋 | 성격 | 문장 pair 수 |
|---|---|---:|
| Kor-Learner | 한국어 학습자 말뭉치 기반 | 28,426 |
| Kor-Native | 원어민 dictation 기반 오류 | 17,559 |
| Kor-Lang8 | Lang-8 한국어 학습자 문장 정제 | 109,559 |

현재 프로젝트 데이터 대응:
- `native` -> Kor-Native 대응
- `korean_learner` -> Kor-Learner 대응
- `union` -> 현재 파일 기준 Kor-Native + Kor-Learner + Kor-Lang8 합본 규모

주의:
- 논문의 Kor-Union은 Kor-Learner + Kor-Native + Kor-Lang8이다.
- 현재 프로젝트의 `union`은 line count 기준으로 논문 Kor-Union 구성과 거의 같은 규모지만, split과 전처리가 완전히 동일한지는 별도 확인이 필요하다.

평가지표:
- GLEU: 문장 생성 결과의 n-gram 기반 품질 평가
- M2 Precision/Recall/F0.5: KAGAS로 만든 edit annotation 기준의 edit-level 평가
- F0.5: precision을 recall보다 더 중시하므로, 과잉 수정이 문제가 되는 GEC에 적합하다.

논문 Table 6 test set 핵심 점수:

| 모델 | Kor-Learner GLEU / F0.5 | Kor-Native GLEU / F0.5 | Kor-Lang8 GLEU / F0.5 | Kor-Union GLEU / F0.5 |
|---|---:|---:|---:|---:|
| Hanspell | 30.36 / 15.46 | 57.08 / 71.50 | 22.94 / 19.88 | 28.82 / 25.85 |
| KoBART | 45.06 / 37.58 | 67.24 / 70.45 | 28.48 / 25.93 | 33.70 / 31.70 |
| KoBART + Kor-Union | 42.66 / 41.00 | 59.71 / 73.63 | 28.65 / 26.78 | - |

발표 포인트:
- 논문에서 KoBART는 Hanspell보다 넓은 오류 유형에서 안정적인 성능을 보인다.
- 이번 프로젝트의 BART/BLT 비교도 궁극적으로 GLEU와 KAGAS 기반 M2/F0.5로 평가해야 한다.

---

## Part 2. KoBART Baseline 재현

### 5. 사용한 BART 모델 구조

논문 KoBART와 재현 KoBART의 구조 비교:

| 항목 | 논문 KoBART | 현재 재현 KoBART |
|---|---|---|
| 기본 구조 | BART encoder-decoder | BART encoder-decoder |
| 구현 클래스 | `BartForConditionalGeneration` 계열 | `BartForConditionalGeneration` |
| pretrained KoBART | KoBART pretrained model | `hyunwoongko/kobart` |
| tokenizer | KoBART tokenizer | `hyunwoongko/kobart` tokenizer |
| 입력 단위 | subword token | subword token |
| 학습 방식 | 오류문 -> 교정문 seq2seq fine-tuning | 오류문 -> 교정문 seq2seq fine-tuning |
| trainable parameters | 약 123M | 약 123M |
| 추정 모델 크기 | 약 495MB 규모 | 약 495MB |

발표에서 명확히 말할 점:
- 논문 KoBART와 현재 재현 KoBART는 모두 pretrained KoBART를 GEC 병렬 데이터에 fine-tuning한 encoder-decoder 모델이다.
- 현재 로그 기준 재현 모델은 trainable parameter가 약 123M이고, 모델 메모리 추정 크기는 약 495MB다.
- 따라서 native clean run의 낮은 점수는 “더 작은 모델을 썼기 때문”이 아니라, 학습 조건/optimizer 구현/seed 평균/전처리/평가 경로 차이에서 원인을 찾아야 한다.

구조:

```text
오류 문장
    │
    ▼ KoBART tokenizer
Encoder
    │
    ▼
Decoder autoregressive generation
    │
    ▼
교정 문장
```

학습 입력:
- source: 오류 문장
- target: 교정 문장
- encoder-decoder seq2seq fine-tuning

발표 포인트:
- KoBART는 subword tokenizer 기반 encoder-decoder 모델이다.
- 오류 문장을 encoder가 읽고, decoder가 교정 문장을 생성한다.
- 논문 모델과 현재 재현 모델은 구조와 파라미터 수 면에서는 같은 계열이므로, 점수 차이는 구조 차이보다 실험 조건/구현 세부 차이에서 설명해야 한다.

### 6. Fine-tuning 방식

논문 실험 조건:

| 항목 | 논문 조건 |
|---|---:|
| max epoch | 10 |
| learning rate | 3e-5 |
| split | train/test/valid = 70/15/15 |
| checkpoint 선택 | validation GLEU 최고 모델 |
| 결과 보고 | 3 seed 평균 |
| test 점수 | validation best checkpoint로 test 평가 |

현재 clean 재현 설정:

| 항목 | 값 |
|---|---:|
| profile | paper |
| learning rate | 3e-5 |
| batch size | 64 |
| max epochs | 10 |
| max sequence length | 128 |
| beam size | 4 |
| optimizer | AdamW |
| validation metric | GLEU |

조건 비교:

| 항목 | 논문 KoBART | 현재 clean run | native tuned/40 run |
|---|---:|---:|---:|
| epoch | 10 | 10 | max 40 |
| learning rate | 3e-5 | 3e-5 | 5e-5 |
| batch size | 64 계열 재현 예시 | 64 | 32 |
| seed | 3 seed 평균 | seed 0 단일 run | seed 0 단일 run |
| optimizer | AdamW, `correct_bias=False` | 환경상 PyTorch AdamW fallback 가능 | 환경상 PyTorch AdamW fallback 가능 |
| 성격 | 논문 보고 조건 | 보수적 재현 run | 논문 조건 밖의 탐색 run |

학습 절차:

```text
1. KoBART pretrained weights 로드
2. GEC train split으로 teacher-forcing fine-tuning
3. 매 epoch validation set generation
4. GLEU 기준 best checkpoint 저장
5. 필요 시 KAGAS M2 파일로 F0.5 평가 확장
```

역할 구분:
- KAGAS: 데이터 annotation / M2 평가 도구
- BART fine-tuning: 교정 문장을 생성하는 neural model 학습
- GLEU/M2/F0.5: 모델 출력 품질을 측정하는 metric

발표 포인트:
- KAGAS와 BART는 경쟁 관계가 아니라, KAGAS가 평가 체계를 제공하고 BART가 correction model 역할을 한다.
- 논문은 10 epoch 조건이므로, 40 epoch 결과는 논문 재현 수치가 아니라 학습 설정 민감도를 보여주는 별도 분석으로 다룬다.

### 7. KoBART 재현 결과와 자원 사용

논문 reported GLEU와 현재 재현 validation GLEU 비교:

| 데이터셋 | 논문 KoBART test GLEU | 논문 KoBART validation GLEU | 현재 clean validation GLEU | 차이: 논문 validation 대비 |
|---|---:|---:|---:|---:|
| Kor-Native / `native` | 67.24 | 69.37 | 49.76 | -19.61 |
| Kor-Learner / `learner` | 45.06 | 46.94 | 38.97 | -7.97 |
| Kor-Union / `union` | 33.70 | 34.07 | 27.57 | -6.50 |

주의:
- 위 표의 현재 점수는 validation GLEU다. test GLEU를 직접 비교하려면 validation best checkpoint로 test split generation 평가를 별도로 수행해야 한다.
- 논문 test GLEU는 validation GLEU보다 낮게 보고된다. 논문 기준 하락폭은 Native -2.13, Learner -1.88, Union -0.37이다.
- 따라서 현재 validation 점수만으로 논문 test 점수와 직접 비교하면 재현 모델 성능을 과대평가할 수 있다.

현재 확보한 clean run validation GLEU:

| 데이터셋 | 상태 | Best validation GLEU | Best epoch |
|---|---|---:|---:|
| native_clean | 완료 | 49.76 | 8 |
| learner_clean | 완료 | 38.97 | 6 |
| union_clean | 부분 진행 | 27.57 | 3 |

추가로 확인한 native tuned/40 epoch run:

| run | 설정 | 시작점 | Best validation GLEU | 관찰 |
|---|---|---|---:|---|
| native_clean | lr 3e-5, batch 64, 10 epoch | fresh | 49.76 | 보수적 clean 재현 |
| native tuned/40 | lr 5e-5, batch 32, max 40 epoch | `outputs/native/last.ckpt` resume | 63.39 | 4 epoch 부근 최고, 이후 62-63대 plateau |
| 논문 KoBART validation | 3 seed 평균 | 논문 설정 | 69.37 | Appendix D.1 기준 |

해석:
- 같은 KoBART 구조라도 learning rate, batch size, resume checkpoint, 학습 epoch 설정에 따라 validation GLEU가 크게 달라진다.
- native 기준으로 tuned/40 run은 clean run보다 약 +13.6 GLEU 높다.
- 그래도 논문 validation GLEU 69.37보다는 낮으므로, optimizer 구현 차이, seed 평균, checkpoint 선택, 세부 전처리 차이를 추가로 확인해야 한다.
- 논문 조건은 10 epoch이므로, `native_clean`과 직접 비교하는 것이 원칙이고, `native tuned/40`은 “추가 학습으로 성능이 얼마나 회복되는지”를 보여주는 보조 결과다.

학습 자원:

| 데이터셋 | GPU | epoch | wall time | GPU-hour |
|---|---:|---:|---:|---:|
| native_clean | 1 x A100 | 10 | 약 22분 | 약 0.37 |
| learner_clean | 1 x A100 | 10 | 약 29분 | 약 0.48 |
| union_clean | 1 x A100 | 7/10 진행 | 약 1시간 50분 제한 전 중단 | 약 1.8 |
| native tuned/40 | 1 x A100 | 40 target | 약 37분 관측 로그 | 약 0.62 |

해석:
- native는 논문에서도 상대적으로 높은 점수를 보이는 데이터셋이고, 현재 재현에서도 가장 높은 validation GLEU를 보인다.
- learner는 오류 유형이 더 다양해 native보다 낮은 GLEU를 보인다.
- 현재 `union`은 아직 완료 run이 아니다.

발표 포인트:
- KoBART baseline은 현재 가장 정량 결과가 안정적으로 확보된 기준 모델이다.
- 발표에서는 clean run과 tuned/40 run을 분리해, 모델 구조보다 학습 설정이 성능에 미치는 영향을 명확히 보여준다.

---

## Part 3. BLT 구조와 한국어 GEC 적용

### 8. 왜 BLT를 비교 대상으로 삼는가

KoBART의 한계:
- subword tokenizer에 의존한다.
- 띄어쓰기/철자/조사처럼 표면 변화가 작은 오류에서 tokenizer 경계가 영향을 줄 수 있다.
- OOV나 드문 표기, 오타, 신조어 처리에서 tokenizer 품질에 의존한다.

BLT의 가능성:
- byte-level 모델이므로 OOV가 없다.
- UTF-8 바이트 단위로 모든 한국어 문자열을 표현할 수 있다.
- 띄어쓰기 오류는 실제 space byte 변화로 직접 모델링된다.
- dynamic patching을 통해 예측이 어려운 구간에 더 세밀한 계산을 배분할 수 있다.

한국어에서 기대되는 장점:
- 띄어쓰기 오류(`WS`)
- 철자/오타(`SPELL`)
- 조사(`PART`)
- 어미(`END`)
- 활용/결합 오류(`CONJ`)

발표 포인트:
- BLT가 우수할 수 있다는 가설은 “byte-level 표현 + dynamic patching이 한국어 표면 오류에 더 직접적으로 반응할 수 있다”는 데 있다.

### 9. BLT 모델 구조

구현한 모델:
- reference BLT 기반
- `facebook/blt-1b`
- `facebook/blt-entropy`
- 약 1B scale 모델
- KoBART 123M 대비 약 8배 규모

BLT 핵심 구조:

```text
UTF-8 bytes
    │
    ▼
Entropy model
    │  예측 난도가 높은 위치를 patch boundary로 선택
    ▼
Dynamic patching
    │
    ▼
Local Encoder
    │  byte -> patch representation
    ▼
Global Transformer
    │  patch 단위 주요 연산
    ▼
Local Decoder
    │  patch -> byte logits
    ▼
Next-byte generation
```

핵심 차이:
- KoBART: subword token sequence를 encoder-decoder로 변환
- BLT: byte sequence를 dynamic patch로 압축하고, patch 단위 latent transformer로 처리

발표 포인트:
- 단순히 byte 단위로 Transformer를 돌리는 것은 BLT가 아니다.
- BLT의 핵심은 entropy 기반 dynamic patching과 local/global 구조다.

### 10. GEC 태스크로 변환한 학습 방식

BLT는 기본적으로 autoregressive causal LM이다.
GEC는 오류문을 입력받아 교정문을 생성하는 seq2seq 문제다.

이번 구현의 변환 방식:

```text
[BOS] 오류문 bytes [SEP] 교정문 bytes [EOS]
```

loss mask:
- `[BOS] 오류문 [SEP]` 구간은 loss 제외
- `교정문 bytes [EOS]` 구간만 next-byte prediction loss 계산

즉, BLT를 prefix-LM 방식으로 fine-tuning한다.

학습 설정:

| 항목 | 값 |
|---|---:|
| model | `facebook/blt-1b` |
| patching | entropy-based dynamic patching |
| lr | 1e-5 |
| weight decay | 0.1 |
| AdamW betas | `(0.9, 0.95)` |
| scheduler | cosine |
| warmup steps | 2000 |
| beam size | 4 |

발표 포인트:
- encoder-decoder로 BLT 구조를 뜯어고치지 않고, pretrained causal LM의 구조를 유지한 채 GEC를 prefix-LM 문제로 바꿨다.

### 11. BLT가 BART보다 자원 소모가 큰 이유

모델 크기:
- KoBART: 약 123M parameters
- BLT: 약 1B scale
- 단순 parameter 기준으로도 약 8배 이상 크다.

시퀀스 길이:
- KoBART는 subword token 기준으로 처리한다.
- BLT는 UTF-8 byte 기준으로 시작한다.
- 한글 한 글자는 보통 UTF-8 3 bytes이므로, raw sequence가 길어진다.

태스크 변환 비용:
- KoBART는 source와 target을 encoder-decoder로 분리한다.
- BLT prefix-LM은 오류문과 교정문을 하나의 causal sequence로 이어 붙인다.
- 학습 sequence가 길어지고, generation도 byte 단위로 진행된다.

dynamic patching 비용:
- entropy model로 patch boundary를 계산한다.
- local encoder/global transformer/local decoder 단계를 거친다.
- patching이 계산을 줄이는 효과도 있지만, fine-tuning pipeline에서는 patch 계산 자체가 추가 비용이다.

발표 포인트:
- BLT의 비용 증가는 구현 문제가 아니라, byte-level modeling과 1B scale architecture에서 자연스럽게 발생하는 구조적 비용이다.

### 12. 현재까지 확보한 BLT 학습 결과

최신 reference BLT 구조 기준:

| 데이터셋 | GPU | epoch | validation loss | wall time | GPU-hour |
|---|---:|---:|---:|---:|---:|
| native | 4 x A100 | 3 | 0.0597 | 약 39분 | 약 2.6 |
| learner | 4 x A100 | 3 | best 0.1310 | 약 56분 | 약 3.7 |
| union | 진행 중 | 진행 중 | 미확정 | 미확정 | 미확정 |

KoBART와 현재 비교 가능한 상태:

| 항목 | KoBART | BLT |
|---|---:|---:|
| 모델 크기 | 123M | 약 1B |
| native 결과 | validation GLEU 49.76 | validation loss 0.0597 |
| learner 결과 | validation GLEU 38.97 | best validation loss 0.1310 |
| generation GLEU | 산출 완료 | 아직 full generation 평가 필요 |
| 결론 가능 여부 | baseline으로 사용 가능 | 학습 가능성 확인, 최종 GLEU 비교 전 |

해석:
- BLT는 loss 기준으로 정상 학습과 checkpoint 저장을 확인했다.
- 하지만 BART와 같은 GLEU/F0.5 기준의 최종 성능 비교는 아직 generation 평가가 끝나야 가능하다.
- 따라서 현재 발표에서는 “BLT가 BART보다 높다/낮다”가 아니라 “BLT를 한국어 GEC에 적용 가능한 구조로 구현했고, 학습 수렴을 확인했다”가 정확한 결론이다.

발표 포인트:
- 현재 성능표에서 BART는 정량 baseline, BLT는 구조 적용 및 학습 가능성 검증 단계다.

---

## Part 4. UI 래핑과 실제 사용 화면

### 13. 모델을 UI로 감싼 이유

목표:
- 학습한 correction model을 실제 사용 가능한 형태로 확인한다.
- 단일 문장/여러 문장 입력을 받아 교정 결과를 즉시 확인한다.
- 향후 KoBART와 BLT를 같은 입력으로 비교할 수 있는 데모 기반을 만든다.

현재 구현:
- `serving/app.py`
- Gradio 기반 웹 UI
- 기본 모델: `Soyoung97/gec_kr`
- 향후 직접 fine-tuned KoBART checkpoint 또는 BLT checkpoint로 교체 가능

UI 기능:
- 단일 문장 교정
- 여러 문장 batch 교정
- 원문/교정문 diff 표시
- 예제 문장 버튼

발표 포인트:
- 모델 성능은 숫자로 평가하지만, 실제 correction task에서는 UI를 통해 수정 품질을 직관적으로 확인하는 과정도 중요하다.

### 14. 실제 화면 슬라이드

슬라이드 구성:
- 왼쪽: 입력 문장
- 오른쪽: 교정 결과
- 아래: diff highlight

넣을 예시:

```text
입력: 한국어는어렵다.
출력: 한국어는 어렵다.
오류 유형 후보: WS
```

```text
입력: 나는 학교에갔다
출력: 나는 학교에 갔다
오류 유형 후보: WS
```

```text
입력: 오늘날시가 좋습니다.
출력: 오늘 날씨가 좋습니다.
오류 유형 후보: SPELL / WS
```

발표 자료 제작 시 필요한 이미지:
- Gradio 단일 문장 탭 캡처
- 여러 문장 탭 캡처
- diff 표시 캡처

발표 포인트:
- 같은 UI에 KoBART와 BLT를 모두 붙이면 모델 구조 차이가 실제 출력 차이로 어떻게 나타나는지 비교할 수 있다.

### 15. 이후 구현해야 할 내용

평가:
- BLT validation/test generation 평가 완료
- KAGAS 기반 M2/F0.5 평가 파일 준비
- KoBART vs BLT를 동일 데이터셋, 동일 split, 동일 metric으로 비교

모델:
- BLT 10 epoch 이상 학습 완료
- union 데이터 완료
- 필요 시 BLT fine-tuning strategy 조정
  - layer freezing
  - learning rate schedule 조정
  - generation decoding 설정 조정

분석:
- KAGAS 오류 유형별 성능 비교
- 특히 `WS`, `SPELL`, `PART`, `END`, `CONJ`에서 BLT가 KoBART보다 나은지 확인
- 입력 길이/오류 유형별 generation time 비교

UI:
- KoBART/BLT 모델 선택 탭
- 같은 입력에 대한 A/B 출력 비교
- KAGAS 오류 유형 annotation 표시
- correction confidence 또는 edit-level diff 개선

발표 포인트:
- 이후 작업의 핵심은 단순히 학습을 더 돌리는 것이 아니라, KAGAS 오류 유형별로 BLT의 장점이 실제로 나타나는지 확인하는 것이다.

### 16. 결론

정리:
- KAGAS는 한국어 GEC를 데이터/오류 유형/평가지표 관점에서 표준화하는 핵심 도구다.
- KoBART baseline은 123M parameter encoder-decoder 모델로 재현했고, validation GLEU 기준 결과를 확보했다.
- BLT는 1B scale byte-level dynamic patching 모델로 구현했으며, prefix-LM 방식으로 한국어 GEC fine-tuning을 수행했다.
- BLT는 자원 소모가 크지만, tokenizer에 의존하지 않고 한국어 표면 오류를 byte 단위로 직접 모델링할 수 있다는 장점이 있다.
- 현재는 BLT의 학습 수렴을 확인한 단계이며, 최종 비교는 generation GLEU와 KAGAS 기반 F0.5 평가가 완료되어야 한다.

마무리 문장:

> 이번 프로젝트의 핵심 성과는 KoBART baseline을 재현한 것과, BLT를 한국어 GEC에 적용 가능한 neural architecture로 변환하고 학습 가능성을 확인한 것이다. 다음 단계는 KAGAS 오류 유형별 평가를 통해 BLT가 실제로 어떤 한국어 오류에서 장점을 갖는지 검증하는 것이다.

---

## 발표 시 강조할 표현

- “KAGAS는 모델이 아니라 한국어 GEC 평가 체계다.”
- “KoBART는 현재 정량 baseline이다.”
- “BLT는 byte-level dynamic patching 구조를 한국어 GEC에 적용한 확장 실험이다.”
- “BLT의 최종 성능 결론은 generation 평가와 KAGAS 기반 M2 평가 이후 가능하다.”
- “현재 단계의 핵심은 구조 구현, 학습 수렴, 비교 가능한 평가 파이프라인 구축이다.”

## 피해야 할 표현

- “BLT가 KoBART보다 좋다.”
- “BLT 성능 비교가 완료됐다.”
- “union 결과가 논문 Kor-Union과 동일하다.”
- “KAGAS와 BART를 비교한다.”

대신:

- “KAGAS는 평가 체계이고, KoBART/BLT는 correction model이다.”
- “현재 BLT는 loss 수렴을 확인했으며 GLEU/F0.5 비교는 산출 중이다.”
