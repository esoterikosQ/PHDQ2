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
