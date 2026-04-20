# Track 1: GEC 서빙 아키텍처

## 목표

기학습된 KoBART GEC 모델을 **RTX 5090 Ubuntu 서버**에서 로드하여
웹 UI를 통해 한국어 문법 교정 서비스를 제공한다.
학습 없이 추론만 수행하므로 가장 빠르게 작동하는 결과물을 만들 수 있다.

---

## 1. 시스템 구성

```
┌─────────────────────────────────────────────┐
│  Ubuntu 서버 (RTX 5090)                      │
│                                              │
│  ┌──────────┐     ┌───────────────────────┐  │
│  │ Web UI   │────▶│ Inference Engine      │  │
│  │ (Gradio) │◀────│                       │  │
│  │ :7860    │     │ KoBART GEC Model      │  │
│  └──────────┘     │ (BartForConditional   │  │
│       ▲           │  Generation)          │  │
│       │           │                       │  │
│  사용자 브라우저   │ GPU: RTX 5090         │  │
│                   └───────────────────────┘  │
└─────────────────────────────────────────────┘
```

## 2. 체크포인트 확보

### 2-1. 옵션

| 옵션 | 소스 | 비고 |
|------|------|------|
| A. 논문 공개 체크포인트 | Standard_Korean_GEC 리포 | README에 다운로드 링크 확인 필요 |
| B. HuggingFace Hub | `hyunwoongko/kobart` + fine-tune | 사전학습만 공개, GEC fine-tuned는 별도 |
| C. 직접 학습 후 사용 | Track 2 (baseline/) 결과 | Track 2 완료 후 가능 |

> **우선 전략**: 옵션 A를 먼저 시도. 체크포인트가 없으면 옵션 C로 전환 (Track 2 의존).

### 2-2. 모델 파일 위치
```
serving/
└── checkpoints/          # .gitignore — 대용량 모델 파일
    └── kobart_gec.ckpt   # 또는 pytorch_model.bin
```

## 3. 추론 파이프라인

### 3-1. 추론 흐름

```python
# 1. 모델 로드 (서버 시작 시 1회)
model = BartForConditionalGeneration.from_pretrained(ckpt_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
model.to('cuda').eval()

# 2. 추론 (요청마다)
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
output_ids = model.generate(input_ids, num_beams=4, max_length=128, eos_token_id=1)
corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 3-2. 핵심 파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `num_beams` | 4 | 논문과 동일 |
| `max_length` | 128 | 토큰 기준 |
| `eos_token_id` | 1 | KoBART EOS |
| device | `cuda` | RTX 5090 |

## 4. 웹 UI 설계

### 4-1. 프레임워크 선택

| 후보 | 장점 | 단점 |
|------|------|------|
| **Gradio** | ML 데모 특화, 빠른 구현 | 커스텀 UI 한계 |
| Streamlit | 유연한 레이아웃 | 실시간 응답 약간 느림 |
| FastAPI + 프론트 | 완전 커스텀 | 개발량 많음 |

> **선택: Gradio** — ML 추론 서빙에 최적화, 코드 최소화.

### 4-2. UI 구성

```
┌─────────────────────────────────────┐
│  한국어 문법 교정기 (Korean GEC)      │
│                                      │
│  입력:                               │
│  ┌─────────────────────────────────┐ │
│  │ 한국어는어렵다.                   │ │
│  └─────────────────────────────────┘ │
│                                      │
│  [교정하기]                          │
│                                      │
│  결과:                               │
│  ┌─────────────────────────────────┐ │
│  │ 한국어는 어렵다.                  │ │
│  └─────────────────────────────────┘ │
│                                      │
│  변경 사항:                          │
│  "한국어는어렵다." → "한국어는 어렵다."│
│  [WS] 띄어쓰기 추가: "는" 뒤         │
└─────────────────────────────────────┘
```

### 4-3. 추가 기능 (선택)
- 원문/교정 diff 하이라이트 (삽입=초록, 삭제=빨강)
- 배치 입력 (여러 문장 한번에)
- 교정 신뢰도 표시
- 향후 Track 3(BLT) 모델과의 A/B 비교 탭

## 5. 디렉토리 구조

```
serving/
├── architecture.md      # 이 파일
├── app.py               # Gradio 웹 UI 진입점
├── infer.py             # 추론 엔진 (모델 로드 + generate)
├── requirements.txt     # 서빙 전용 의존성
└── checkpoints/         # 모델 체크포인트 (.gitignore)
```

## 6. 의존성

```
torch>=2.0.0
transformers>=4.30.0
gradio>=4.0.0
```

## 7. 실행 방법

```bash
# Ubuntu RTX 5090 서버에서
cd serving/
pip install -r requirements.txt

# 체크포인트 배치 후
python app.py --checkpoint checkpoints/kobart_gec.ckpt --port 7860

# 브라우저에서 접속
# http://<서버IP>:7860
```

## 8. 구현 단계

```
Phase 1: 추론 엔진
  └── infer.py: 모델 로드, generate 래핑, 단일 문장 교정 함수

Phase 2: 웹 UI
  └── app.py: Gradio 인터페이스, 교정 결과 + diff 표시

Phase 3: 서버 배포
  └── Ubuntu RTX 5090에서 실행 확인, 외부 접속 테스트

Phase 4: 확장 (선택)
  └── Track 3 BLT 모델 추가 시 A/B 비교 탭 추가
```
