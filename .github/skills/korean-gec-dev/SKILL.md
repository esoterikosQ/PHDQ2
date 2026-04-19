---
name: korean-gec-dev
description: '한국어 GEC(문법 오류 교정) 코드 분석·개량 보조. 파이프라인 이해, 환경 호환성 관리, 성능 개선, 진행 로그 기록. Standard_Korean_GEC 기반 KoBART seq2seq 모델 작업 시 사용.'
argument-hint: '작업 내용을 간단히 설명 (예: "데이터 전처리 파이프라인 분석", "환경 설정 점검", "학습 실행 및 평가")'
---

<!-- 리포지토리 -->
<!-- Baseline: https://github.com/soyoung97/Standard_Korean_GEC.git -->
<!-- BLT 원본: https://github.com/facebookresearch/blt.git -->
<!-- 작업 결과물: https://github.com/esoterikosQ/PHDQ2.git -->
<!-- 레퍼런스 코드 로컬: reference_code/Standard_Korean_GEC, reference_code/blt -->
<!-- 논문 PDF: papers/korean_gec.pdf (GEC), papers/blt.pdf (BLT), papers/s10994-021-06034-2.pdf -->
<!-- 워크스페이스(phdq/) = PHDQ2 git repo (origin → esoterikosQ/PHDQ2) -->

# 한국어 GEC 코드 분석·개량 보조 스킬

## When to Use
- Standard_Korean_GEC(BART) 베이스라인 코드를 분석·재현할 때
- BLT 기반 GEC 모델을 설계·구현할 때
- 학습 데이터 파이프라인을 이해하거나 개선할 때
- 환경 호환성 문제(CUDA, Python, 패키지 버전)를 진단할 때
- BART vs BLT 성능을 비교·평가할 때
- 작업 진행 로그를 남길 때

## 프로젝트 개요

**최종 목표**: KoBART(BART) 기반 한국어 GEC 모델을 **BLT(Byte Latent Transformer)**로 대체하여 성능 향상을 검증한다.

### 두 트랙 구조

| 트랙 | 모델 | 리포지토리 | 역할 |
|------|------|-----------|------|
| **Baseline** | KoBART (BART seq2seq) | `soyoung97/Standard_Korean_GEC` | 기존 성능 재현·비교 기준 |
| **Development** | BLT (Byte Latent Transformer) | `facebookresearch/blt` | GEC 프레임워크에 BLT 결합 |
| **Output** | BLT-GEC | `esoterikosQ/PHDQ2` | 작업 중간/최종 결과물 업로드 |

### 실행 환경 (3-Machine)

| 환경 | 역할 | 상세 |
|------|------|------|
| **macOS (로컬)** | 코드 작성 전용 | 코딩, 스킬 사용, git push |
| **SLURM 공유 노드** | 학습·추론 실행 | GPU 클러스터, sbatch/srun으로 job 제출 |
| **Ubuntu 서버 (RTX 5090)** | UI 구현·서빙 | 실험 결과 시각화, 데모 서빙 |

> ⚠️ **3대 머신 간 코드 동기화는 `esoterikosQ/PHDQ2` git repo를 통해 수행한다.**
> Mac에서 코드 작성 → push → SLURM/Ubuntu에서 pull → 실행. 모든 코드 변경은 반드시 push 후 원격에서 pull.

### Git 동기화 워크플로우
```
[Mac] 코드 작성 → git push
                          ↘
                    esoterikosQ/PHDQ2 (GitHub)
                          ↙              ↘
         [SLURM] git pull → 학습/추론    [Ubuntu] git pull → UI 서빙
                    ↘                        ↘
              결과 push (logs, metrics)   결과 push (UI 코드)
```

### Baseline: Standard_Korean_GEC
- **로컬 경로**: `reference_code/Standard_Korean_GEC/`
- **논문**: `papers/korean_gec.pdf` — "Towards Standardizing Korean Grammatical Error Correction" (ACL 2023)
- **모델**: `BartForConditionalGeneration` (KoBART pretrained)
- **Docker(원본)**: `msyoon8/default:gec`
- ⚠️ 원본 README 경고: "The original model code was implemented years ago, so it may not work well in current CUDA environments."

### Development: BLT
- **로컬 경로**: `reference_code/blt/`
- **논문**: `papers/blt.pdf` — "Byte Latent Transformer: Patches Scale Better Than Tokens" (Meta, 2024)
- **핵심 차이**: 토크나이저 없이 바이트 단위로 입력 처리, 엔트로피 기반 동적 패치 분할
- **사전학습 가중치**: `facebook/blt-1b`, `facebook/blt-7b` (HuggingFace)
- **엔트로피 모델**: `facebook/blt-entropy` (패치 경계 결정용)
- **라이선스**: CC-BY-NC-4.0 (비상업적 연구 사용 가능)
- **아키텍처 참조**: [blt-architecture.md](./references/blt-architecture.md)

---

## Phase 1: 파이프라인 구조 파악

새 작업 세션을 시작하거나, 코드 흐름이 불분명할 때 이 절차를 따른다.

1. **현재 상태 확인**: 진행 로그(`LOG.md`)를 읽고 이전 작업 내역 확인
2. **전체 구조 참조**: [pipeline-reference.md](./references/pipeline-reference.md) 참조
3. **레퍼런스 코드 읽기**: `reference_code/Standard_Korean_GEC/` 또는 `reference_code/blt/`에서 원본 코드 확인
4. **데이터 흐름 추적**: 입력 → 전처리 → 학습 → 평가까지 데이터가 어떻게 변환되는지 추적
5. **의존성 그래프 확인**: 해당 모듈이 다른 모듈과 어떻게 연결되는지 파악

### 설명 원칙
- 코드 블록 단위로 역할을 설명하되, 한국어로 작성
- 함수/클래스 간 호출 관계를 명시
- 데이터 포맷 변환 지점을 명확히 표시 (예: `txt → m2`, `pair → tensor`)

---

## Phase 2: 환경 호환성 관리

코드를 새 환경에서 실행하거나 패키지 충돌이 발생했을 때.
**3대 머신 각각에 대해 별도로 환경을 구성해야 한다.**

### 머신별 환경 역할
| 머신 | 설치 필요 항목 | 설치 불필요 |
|------|---------------|-------------|
| **Mac (로컬)** | git, 에디터, linter, Python (코드 검증용) | CUDA, GPU 드라이버, 대용량 모델 |
| **SLURM 노드** | CUDA, PyTorch, BLT/BART 전체 의존성, 데이터 | UI 관련 패키지 |
| **Ubuntu (RTX 5090)** | CUDA, PyTorch, 추론 모델, UI 프레임워크 | 학습용 대규모 데이터 |

### 점검 절차
1. **환경 스냅샷 기록** (머신별로 각각)
   - Python 버전, CUDA 버전, GPU 정보 수집
   - `pip list` 또는 `conda list`로 현재 패키지 목록 확인
   - 원본 `requirements.txt`와 비교

2. **충돌 진단 체크리스트** → [env-checklist.md](./references/env-checklist.md) 참조

3. **해결 전략 선택**
   | 상황 | 전략 |
   |------|------|
   | 패키지 버전만 다름 | 호환 버전 핀 설정 |
   | API 변경 (transformers 등) | 코드 패치 (최소 변경) |
   | 근본적 구조 불일치 | 해당 모듈 재작성 검토 |
   | Docker 사용 가능 | `msyoon8/default:gec` 또는 커스텀 이미지 |

4. **변경 사항 기록**: 어떤 환경 변경을 했는지 `LOG.md`에 **머신 이름과 함께** 기록

---

## Phase 3: BART 베이스라인 확립

1. Standard_Korean_GEC 원본 코드를 `esoterikosQ/PHDQ2`에 베이스라인으로 구성
2. 원본 코드로 베이스라인 결과 재현 (또는 논문 수치 참조)
3. 평가 지표 기록: **GLEU**, **M2 Precision/Recall/F0.5**
4. 데이터셋별 결과 분리: Kor-Lang8, Kor-Learner, Kor-Native
5. 베이스라인 결과를 `LOG.md`에 기록 (이후 모든 비교의 기준점)

---

## Phase 4: BLT-GEC 모델 설계·구현

BLT를 GEC 태스크에 적용하는 핵심 단계.

### 4-1. 아키텍처 설계 결정
BART → BLT 전환 시 핵심 설계 차이점:

| 항목 | BART (Baseline) | BLT (Development) |
|------|----------------|-------------------|
| 입력 단위 | 서브워드 토큰 | 바이트 (UTF-8) |
| 토크나이저 | KoBART tokenizer | 없음 (byte-level) |
| 인코딩 | 토큰 → embedding | 바이트 → 동적 패치 → latent |
| 패치 분할 | N/A | 엔트로피 기반 동적 분할 |
| 디코딩 | 토큰 생성 | 패치 → 바이트 복원 |

### 4-2. 구현 순서
1. **BLT 코드 분석**: `facebookresearch/blt` 구조 파악 → [blt-architecture.md](./references/blt-architecture.md)
2. **데이터 어댑터**: GEC 학습 데이터(orig\tcorrected txt)를 BLT 입력 형식으로 변환
3. **모델 래핑**: BLT를 seq2seq GEC 태스크에 맞게 래핑 (encoder-decoder 또는 prefix-LM)
4. **학습 루프**: BLT 학습 스크립트를 GEC 데이터에 맞게 수정
5. **평가 통합**: BART 베이스라인과 동일한 평가 파이프라인(GLEU, M2) 사용

### 4-3. 한국어 특수 고려사항
- 한국어 UTF-8: 한 글자당 3바이트 → 패치 길이/엔트로피 분포가 영어와 다름
- 띄어쓰기 오류(WS): 바이트 수준에서 공백(0x20) 삽입/삭제로 직접 모델링 가능
- 자모 분리: 바이트 수준에서 초성/중성/종성 자연스럽게 구분됨 (한글 유니코드 블록)

---

## Phase 5: 성능 비교 및 분석

### 5-1. 실험 설계
1. **통제 변수**: 동일 데이터셋, 동일 split, 동일 평가 지표
2. **비교 축**:
   - 전체 성능: GLEU, M2 F0.5
   - 오류 유형별: WS, SPELL, PUNCT 각각
   - 추론 속도 / 모델 크기
3. **한 번에 하나의 변수만 변경** (controlled experiment)

### 5-2. 결과 기록
- 모든 실험 결과를 `LOG.md`에 수치와 함께 기록
- `esoterikosQ/PHDQ2`에 체크포인트·결과 push

### 5-3. 코드 재작성 판단 기준
다음 중 2개 이상 해당되면 해당 모듈 재작성을 검토:
- [ ] 원본 코드가 deprecated API에 의존
- [ ] 환경 패치로는 해결 불가능한 구조적 문제
- [ ] 재작성 시 명확한 성능/유지보수 이점 존재
- [ ] 원본 코드의 핵심 로직을 완전히 이해한 상태

---

## Phase 6: 진행 로그 기록 및 결과물 관리

**모든 작업 세션의 시작과 끝에 로그를 업데이트한다.**

### 로그 파일 위치
`LOG.md` (프로젝트 루트)

### 로그 형식
```markdown
## [YYYY-MM-DD] 작업 제목

### 목표
- 이번 세션에서 달성하려는 것

### 수행 내용
- 실제로 한 일 (코드 변경, 실험, 분석 등)

### 결과
- 실험 결과, 발견한 문제점, 해결한 이슈

### 다음 단계
- 이어서 해야 할 작업

### 메모
- 기타 참고 사항, 환경 이슈, 아이디어
```

### 로그 원칙
- 세션 시작 시: 목표를 먼저 작성
- 세션 종료 시: 수행 내용, 결과, 다음 단계를 채움
- 실험 결과는 수치를 포함 (예: `GLEU: 0.45 → 0.48`)
- 환경 변경 사항은 반드시 기록

### 결과물 관리 및 Git 동기화
- **모든 코드 변경은 Mac에서 작성 후 반드시 `esoterikosQ/PHDQ2`에 push**
- SLURM/Ubuntu에서는 실행 전 항상 `git pull` 먼저 수행
- 커밋 메시지 규칙: `[P{n}][{머신}] 작업 설명`
  - 예: `[P4][mac] BLT 데이터 어댑터 구현`
  - 예: `[P5][slurm] native 데이터 BLT 학습 결과`
  - 예: `[P6][ubuntu] 데모 UI 초기 구현`
- SLURM에서 생성된 결과(logs, metrics, 작은 체크포인트)도 push
- 대용량 체크포인트·모델 가중치는 git-lfs 또는 외부 저장소 활용
- `.gitignore`에 대용량 데이터·임시 파일 패턴 등록

### 원격 실행 팁
```bash
# SLURM에서 job 제출 패턴
ssh slurm-node
cd ~/PHDQ2 && git pull
sbatch scripts/train_bart.sh   # 또는 BLT 학습
# 완료 후
git add results/ logs/ && git commit -m "[P5][slurm] 학습 결과" && git push

# Ubuntu 서버에서 UI 서빙
ssh ubuntu-server
cd ~/PHDQ2 && git pull
python ui/serve.py
```

---

## Quick Reference: 주요 명령어

### Baseline (BART)
```bash
# 데이터 전처리
cd get_data && python3 process_for_training.py

# 모델 학습
python3 run.py --data native --default_root_dir ../../output \
  --max_epochs 10 --lr 3e-05 --SEED 0 \
  --train_file_path ../../get_data/native_train.txt \
  --valid_file_path ../../get_data/native_val.txt \
  --test_file_path ../../get_data/native_test.txt

# KAGAS (m2 파일 생성)
cd KAGAS && python3 parallel_to_m2_korean.py \
  -orig sample_test_data/orig.txt \
  -cor sample_test_data/corrected.txt \
  -out sample_test_data/output.m2 \
  -hunspell ./aff-dic
```

### BLT
```bash
# 환경 설정
conda create -n blt python=3.12 && conda activate blt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@de742ec3d64bd83b1184cc043e541f15d270c148
pip install -r requirements.txt  # BLT requirements

# 가중치 다운로드
python download_blt_weights.py

# 추론 테스트
python demo.py "A BLT has"

# BLT 모델 로드 (코드에서)
from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.hf import BltTokenizerAndPatcher
entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")
blt_model = ByteLatentTransformer.from_pretrained("facebook/blt-1b")

# 학습 (torchrun)
torchrun --nproc-per-node 8 -m bytelatent.train config=bytelatent/configs/debug.yaml
```

### 결과물 관리
```bash
# PHDQ2 리포에 push
cd PHDQ2 && git add -A && git commit -m "[P4] 작업 설명" && git push origin main
```
