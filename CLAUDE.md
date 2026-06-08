# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korean Grammatical Error Correction (GEC) research comparing KoBART (BART) baseline against Byte Latent Transformer (BLT). Three parallel tracks:

- **Track 1 (Serving)** — `serving/`: Pre-trained KoBART GEC web UI via Gradio on Ubuntu RTX 5090
- **Track 2 (Baseline)** — `baseline/`: KoBART GEC training reproduction using PyTorch Lightning
- **Track 3 (BLT-GEC)** — `blt_gec/`: Reference BLT (`facebookresearch/blt`) fine-tuning for GEC with dynamic entropy patching

A demoted `byte_prefix_lm/` package exists for byte-only causal Transformer experiments (not BLT).

## 3-Machine Workflow

Code is written on **Mac**, trained on **SLURM** (Neuron cluster, A100), served on **Ubuntu RTX 5090**. All sync via `esoterikosQ/PHDQ2` git repo. Code changes happen on Mac; SLURM/Ubuntu always `git pull` before execution.

## Neuron SLURM Cluster Reference

### GPU Partitions

| Partition | GPU | GPUs/Node | CPUs/Node | Mem | Nodes | Max Active/Running |
|-----------|-----|-----------|-----------|-----|-------|--------------------|
| **amd_a100nv_8** | A100 80GB | 8 | 64 | 768GB | gpu[35-43] | 20 / 10 |
| amd_a100_4 | A100 80GB | 4 | 64 | 768GB | gpu45 | 2 / 2 |
| amd_h100_2 | H100 | 2 | 48 | 576GB | gpu[57-59] | 2 / 1 |
| eme_h200nv_8 | H200 | 8 | 96 | 1.1TB | gpu47 | 4 / 2 |
| amd_h200nv_8 | H200 | 8 | 96 | 1.1TB | gpu56 | 4 / 2 |
| gh200_1 | GH200 | 1 | 72 | 864GB | gpu52 | 12 / 6 |
| cas_v100nv_8 | V100 | 8 | 32 | 384GB | gpu[01-05] | 12 / 10 |
| cas_v100nv_4 | V100 | 4 | 40 | 480GB | gpu[06-09] | 8 / 4 |
| cas_v100_4 | V100 | 4 | 40 | 480GB | gpu[10-20] | 20 / 10 |
| cas_v100_2 | V100 | 2 | 32 | 384GB | gpu[25-26] | 12 / 6 |

**현재 사용 중**: `amd_a100nv_8` (A100 80GB × 8, 64 cores, 768GB)

### CPU-per-GPU 할당 규칙

노드의 CPU/GPU 비율로 자동 계산: `cpus-per-gpu = (총 코어 / 총 GPU) × 요청 GPU 수`

| Partition | 계산 | GPU 1개당 CPU |
|-----------|------|---------------|
| amd_a100nv_8 | 64 / 8 | **8** |
| amd_a100_4 | 64 / 4 | 16 |
| cas_v100nv_8 | 32 / 8 | 4 |
| cas_v100nv_4 | 40 / 4 | 10 |
| cas_v100_4 | 40 / 4 | 10 |

`amd_a100nv_8`에서 GPU 1개 요청 시 최대 **8 CPU cores**. 스크립트의 `--cpus-per-task=4`는 이 범위 내.

### 시간 제한 및 정책

- 기본 wall time: **48시간** (배치), 24시간(Jupyter), 8시간(인터랙티브)
- 현재 스크립트: `--time=01:55:00` (안전 마진 포함)
- 노드 공유 정책: 한 노드에 여러 작업 동시 실행 (shared node)
- `--exclusive` 사용 시 노드 독점 가능하나 대기 시간 증가

### 스토리지

- **작업 디렉토리**: `/scratch/$USER/` — 모든 연산은 여기서 수행, sbatch 제출도 여기서만 가능
- **홈 디렉토리**: `/home01/$USER/` — 설정 파일, conda 환경 기본 경로
- **자동 삭제**: /scratch에서 **15일간 미접근 파일 자동 삭제** (ToBeDelete_ 접두어 후 20-30일 유예)
- **백업 없음**: Neuron 시스템에 별도 백업 없음
- 용량 확인: `quotainfo`

### Conda 환경

```bash
# x86_64 초기화 (최초 1회)
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda init && source ~/.bashrc

# AArch64 (gh200_1 노드)
source /apps/ARM_node/ARM_applications/Miniconda/24.5.0/etc/profile.d/conda.sh
```

기본 저장 경로: `~/.conda/envs/`, `~/.conda/pkgs/`. scratch로 변경 가능:
```bash
export CONDA_ENVS_PATH=/scratch/$USER/.conda/envs
export CONDA_PKGS_DIRS=/scratch/$USER/.conda/pkgs
```

### SBATCH 필수 지시자

- `--comment=pytorch` — 애플리케이션 유형 필수 명시 (pytorch, tensorflow 등)
- `-p <partition>` — 파티션 선택
- `--gres=gpu:<N>` — GPU 수 요청
- `--cpus-per-task` — 파티션의 CPU-per-GPU 한도 이내로 설정

### 주요 명령어

```bash
sbatch script.sh          # 작업 제출
squeue -u $USER           # 내 작업 확인
scancel <JOB_ID>          # 작업 취소
sinfo -Nel                # 노드/파티션 상태
scontrol show job <ID>    # 작업 상세 정보
```

## Commands

### SLURM Training (from `/scratch/$USER/PHDQ2`)

```bash
# BART baseline paper profile (default: native dataset, lr=3e-5, batch=64, 10 epochs, beam=4)
sbatch scripts/train_bart.sh

# Previous exploratory profile (lr=5e-5, batch=32, 40 epochs)
BART_PROFILE=tuned sbatch scripts/train_bart.sh

# Different dataset
DATASET_TYPE=learner sbatch scripts/train_bart.sh
DATASET_TYPE=union sbatch scripts/train_bart.sh

# BLT-GEC (requires reference BLT env + HF access to facebook/blt-1b, facebook/blt-entropy)
sbatch scripts/train_blt.sh

# Byte-only baseline (not BLT)
sbatch scripts/train_byte_prefix_lm.sh
```

SLURM jobs are 1:55:00 with `--signal=B:TERM@300` (saves checkpoint 5 min before timeout). Default conda env is `phdq`. BART run outputs are profile-separated, e.g. `outputs/native_paper` and `outputs/native_tuned`. BLT outputs default to `outputs/blt_gec/<dataset>`.

### Serving (on Ubuntu RTX 5090)

```bash
pip install -r serving/requirements.txt
python serving/app.py --port 7860
```

### Evaluation Metrics

- **GLEU**: computed via `baseline/metric/gleumodule.py` during BART and BLT validation generation
- **M2**: optional; requires KAGAS preprocessing (`parallel_to_m2_korean.py`) to generate source-gold `.m2` files, then pass `M2_SOURCE_GOLD_PATH`

## Architecture

### Baseline (Track 2)

`baseline/run.py` → Lightning `Trainer` with `KoBARTConditionalGeneration` (`baseline/model.py`). Uses `hyunwoongko/kobart` pretrained weights and `BartForConditionalGeneration`. Data via `GecDataModule` (`baseline/dataset.py`) reading TSV files. Outputs hypotheses/references per epoch to `outputs/generation/` for GLEU scoring.

Checkpoint strategy: `ModelCheckpoint` saves top-3 by `val_gleu`; `SaveBestAlias` copies best to `outputs/<dataset>/best.ckpt`; `SaveLastOnTrainEnd` writes `last.ckpt` for SLURM resume.

### BLT-GEC (Track 3)

`blt_gec/train.py` is a manual training loop (no Lightning). It loads the full reference BLT stack via `blt_gec/model.py`:
- `ByteLatentTransformer.from_pretrained("facebook/blt-1b")` — main model
- `LMTransformer.from_pretrained("facebook/blt-entropy")` — frozen entropy model for patch boundary decisions
- `BltTokenizerAndPatcher` — tokenizer + dynamic patcher

The patcher computes `patch_lengths` per batch via entropy-based dynamic patching, passed to `model(input_ids, patch_lengths=...)`. Reference BLT code is imported from `reference_code/blt/bytelatent/` (added to `sys.path` at runtime).

`blt_gec/data_adapter.py` encodes GEC pairs as `[BOS] source \n<BLT_GEC_SEP>\n target [EOS]` in byte IDs. Loss is computed only on the target portion (prefix-LM). No special SEP token is added to the vocabulary; the separator is encoded as ordinary bytes.

`blt_gec/generate.py` does autoregressive beam search by default (`num_beams=4`) with per-step patching. Use `--num_beams 1` for greedy comparison.
The same generation helper is shared by training validation and CLI inference.
BLT optimizer defaults follow the paper-oriented setting: AdamW betas `(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`, with bias/norm parameters excluded from decay.
Use `TEST_ONLY=1 RESUME_CKPT=<ckpt> sbatch scripts/train_blt.sh` for test-only evaluation, or `RUN_TEST_ON_END=1` for final test evaluation after training.

**Critical**: `blt_gec/` intentionally has no fallback to a simpler model. If reference BLT or entropy patching is unavailable, it fails rather than silently producing non-BLT results.

### Data Format

Tab-separated `source\ttarget` text files, no header. Located at `data/Preprocessed/<dataset>/<dataset>_{train,val,test}.txt`. Datasets: `native`, `korean_learner`, `union`. The `data/` directory is gitignored.

## Conventions

- Commit messages: `[P{phase}][{machine}] description` (e.g., `[P4][mac] BLT data adapter`)
- Progress log: `LOG.md` (reverse chronological, updated each session)
- Language: Korean for documentation/comments, English for code identifiers
- Reference code in `reference_code/` is gitignored and read-only (cloned from upstream repos)
- Large files (checkpoints, model weights, data) are never committed; use `.gitignore` patterns
