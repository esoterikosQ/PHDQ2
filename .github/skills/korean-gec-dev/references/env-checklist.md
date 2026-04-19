# 환경 호환성 체크리스트

> ⚠️ 3대 머신에서 각각 다른 역할을 수행. 머신별 환경을 별도로 관리한다.

## 머신별 역할 요약

| 머신 | OS | GPU | 역할 | Python 환경 |
|------|-----|-----|------|-------------|
| **Mac (로컬)** | macOS | 없음 | 코드 작성, git push | 린팅·타입체크용만 |
| **SLURM 노드** | Linux | 공유 GPU | 학습·추론 | BART venv + BLT venv |
| **Ubuntu 서버** | Ubuntu | RTX 5090 | UI 서빙·데모 | 추론 + UI 프레임워크 |

> 모든 머신은 `esoterikosQ/PHDQ2` git repo를 통해 코드 동기화.

---

## Machine 1: Mac (로컬) — 코드 작성 전용

### 필수 설치
- [ ] git (코드 동기화)
- [ ] Python 3.12 (린팅·타입체크·로컬 테스트용)
- [ ] 에디터 (VS Code + Copilot)

### 불필요
- CUDA, GPU 드라이버, 대용량 모델, 학습 데이터

### Git 설정
```bash
cd ~/path/to/PHDQ2
git remote -v  # origin → esoterikosQ/PHDQ2
# push 전 항상 확인
git status && git diff --stat
```

---

## Machine 2: SLURM 공유 노드 — 학습·추론

### Track A: Baseline (BART) 환경

### 1. Python & CUDA
- [ ] Python 버전 확인 (`python3 --version`) — 원본은 3.7~3.8 기준
- [ ] CUDA 버전 확인 (`nvcc --version` 또는 `nvidia-smi`)
- [ ] PyTorch CUDA 호환성 확인 (`torch.cuda.is_available()`)
- [ ] GPU 메모리 확인 (`nvidia-smi`)

### 2. 핵심 패키지 버전 충돌 확인
| 패키지 | 원본 기준 | 현재 환경 | 상태 |
|--------|-----------|-----------|------|
| `transformers` | ~4.x 초기 | | |
| `pytorch-lightning` | ~1.x | | |
| `torch` | ~1.x | | |
| `kobart-transformers` | 0.2.x | | |
| `KoNLPy` | | | |
| `cyhunspell` | | | |
| `KoBERT` | (SKTBrain) | | |

### 3. 시스템 의존성
- [ ] Java 설치 여부 (KoNLPy의 KKma 실행에 필요)
  - macOS: `brew install openjdk`
  - Linux: `apt-get install openjdk-8-jdk`
- [ ] Hunspell 라이브러리
  - macOS: `brew install hunspell`
  - Linux: `cyhunspell` pip 패키지로 대체 가능

### 4. 데이터 파일
- [ ] Lang8 raw 데이터 다운로드 여부
- [ ] Kor-Learner, Kor-Native 데이터 다운로드 여부
- [ ] `get_data/` 하위에 데이터 파일 배치 여부
- [ ] Hunspell 사전 파일 (`KAGAS/aff-dic/ko.aff`, `ko.dic`)

## 자주 발생하는 문제와 해결

### `transformers` 버전 불일치
- **증상**: `from_pretrained` 호출 시 인자 오류, 모델 구조 변경
- **해결**: `pip install transformers==4.11.3` 또는 코드 패치

### `pytorch-lightning` API 변경
- **증상**: `Trainer`, `LightningModule` 인터페이스 변경
- **해결**: `pip install pytorch-lightning==1.5.10` 또는 코드 마이그레이션

### `kobart-transformers` 설치 실패
- **증상**: pip 설치 시 의존성 충돌
- **해결**: 직접 `transformers`에서 KoBART 로드 (`gogamza/kobart-base-v2`)

### KoNLPy / KKma Java 오류
- **증상**: `JavaError`, `JVMNotFoundException`
- **해결**: `JAVA_HOME` 환경변수 설정, Java 설치 확인

### cyhunspell 빌드 실패
- **증상**: C 컴파일러 오류
- **해결**: macOS → `brew install hunspell` 후 재시도, 또는 `pyhunspell` 대안

## 환경 재작성 판단 기준

아래 중 2개 이상이면 requirements 전면 재구성 고려:
- [ ] 주요 패키지 3개 이상 버전 불일치
- [ ] deprecated API 의존 코드가 5곳 이상
- [ ] Docker 이미지를 사용할 수 없는 환경
- [ ] 원본 requirements.txt로는 pip resolve 불가

---

## Track B: BLT 환경

### 1. Python & CUDA
- [ ] Python 3.12 확인 (`python3 --version`)
- [ ] CUDA 12.1 확인 (`nvcc --version`)
- [ ] H100 GPU 권장 (BLT 공식 테스트 환경)
- [ ] GPU 메모리 최소 24GB (1B 모델 기준)

### 2. 핵심 패키지
| 패키지 | 필요 버전 | 현재 환경 | 상태 |
|--------|-----------|-----------|------|
| `torch` | nightly (cu121) | | |
| `ninja` | latest | | |
| `xformers` | `de742ec3` commit | | |
| `bytelatent` | (blt repo) | | |
| `fairscale` | (if needed) | | |

### 3. HuggingFace 가중치 접근
- [ ] HuggingFace 계정 생성
- [ ] `facebook/blt-1b` 가중치 접근 신청 및 승인
- [ ] `facebook/blt-entropy` 가중치 접근 신청 및 승인
- [ ] `huggingface-cli login` 완료
- [ ] `python download_blt_weights.py` 성공

### 4. BLT 설치 확인
```bash
conda create -n blt python=3.12 && conda activate blt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@de742ec3d64bd83b1184cc043e541f15d270c148
pip install -r requirements.txt
python demo.py "A BLT has"  # 정상 출력 확인
```

### BLT 자주 발생하는 문제

#### xformers 빌드 실패
- **증상**: CUDA 미스매치, 컴파일러 오류
- **해결**: CUDA 12.1 정확히 설치, `ninja` 사전 설치 확인

#### BLT 가중치 로드 실패
- **증상**: `401 Unauthorized`, `gated repo`
- **해결**: HF 가중치 접근 승인 확인, `huggingface-cli login` 재실행

#### 메모리 부족 (OOM)
- **증상**: `CUDA out of memory`
- **해결**: blt-1b(1B)부터 시작, batch size 축소, gradient checkpointing

### SLURM Job 제출 패턴
```bash
# SLURM 노드에서 실행
ssh slurm-node
cd ~/PHDQ2 && git pull origin main

# BART 학습 job 제출
sbatch scripts/train_bart.sh

# BLT 학습 job 제출
sbatch scripts/train_blt.sh

# job 상태 확인
squeue -u $USER
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS

# 완료 후 결과 push
git add results/ logs/ && git commit -m "[P5][slurm] 학습 결과" && git push
```

### SLURM 스크립트 템플릿 (참고)
```bash
#!/bin/bash
#SBATCH --job-name=blt-gec
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

conda activate blt
cd ~/PHDQ2
python -m bytelatent.train config=configs/gec_blt.yaml
```

---

## Machine 3: Ubuntu 서버 (RTX 5090) — UI 서빙

### 환경 체크리스트
- [ ] Ubuntu 버전 확인 (`lsb_release -a`)
- [ ] NVIDIA 드라이버 + CUDA 확인 (`nvidia-smi`)
- [ ] RTX 5090 인식 확인
- [ ] Python 3.12 설치
- [ ] PyTorch (CUDA 호환 버전) 설치
- [ ] 추론용 모델 가중치 배치 (SLURM에서 학습된 체크포인트)
- [ ] UI 프레임워크 설치 (Gradio, Streamlit 등)

### Git 동기화
```bash
ssh ubuntu-server
cd ~/PHDQ2 && git pull origin main
# 추론 실행 또는 UI 서빙
python ui/serve.py
```

### 주의사항
- 대용량 모델 가중치는 git-lfs 또는 직접 scp로 전송
- SLURM 결과를 먼저 push → Ubuntu에서 pull 순서 준수
- RTX 5090 CUDA 호환 버전 확인 필수 (최신 드라이버 필요)
