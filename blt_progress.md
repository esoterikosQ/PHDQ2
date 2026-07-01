# BART vs BLT 학습 현황 (2026-06-10)

## BART (KoBART) — 전체 학습 완료

Paper profile: `lr=3e-5, batch=64, max_epochs=10, num_beams=4`

| 데이터셋 | Job ID | Best GLEU | Best Epoch | Val Loss (최종) | 소요시간 | 상태 |
|----------|--------|-----------|------------|----------------|----------|------|
| native | 752393 | **49.76** | epoch 8 | 0.22 | 22min (1 job) | 완료 |
| learner | 752394 | **38.97** | epoch 6 | 0.55 | 29min (1 job) | 완료 |
| union | 752395→752910→753328 | **27.57** | epoch 3 | 0.73 | 160min (3 jobs) | 완료 |

### BART 소요시간 상세

| Job | 데이터셋 | Epochs | 소요시간 | 비고 |
|-----|----------|--------|----------|------|
| 752393 | native | 0-9 (전체) | 22min | 단일 job 완료 |
| 752394 | learner | 0-9 (전체) | 29min | 단일 job 완료 |
| 752395 | union | 0-6 | 110min | SLURM 시간제한으로 중단 |
| 752910 | union | 7-8 | 17min | Lightning Timer 버그로 조기 종료 |
| 753328 | union | 8-9 | 33min | Timer 수정 후 완료 |

### BART Native GLEU 추이

| Epoch | Val Loss | GLEU |
|-------|----------|------|
| 0 | 0.33 | 36.21 |
| 1 | 0.27 | 39.76 |
| 2 | 0.25 | 43.53 |
| 3 | 0.23 | 45.69 |
| 4 | 0.23 | 47.70 |
| 5 | 0.22 | 48.51 |
| 6 | 0.22 | 47.64 |
| 7 | 0.22 | **49.42** |
| 8 | 0.22 | **49.76** |
| 9 | 0.22 | 49.66 |

### BART Learner GLEU 추이

| Epoch | Val Loss | GLEU |
|-------|----------|------|
| 0 | 0.55 | 35.06 |
| 1 | 0.52 | 35.06 |
| 2 | 0.51 | 36.73 |
| 3 | 0.52 | 37.39 |
| 4 | 0.53 | 37.87 |
| 5 | 0.54 | 38.15 |
| 6 | 0.54 | **38.97** |
| 7 | 0.55 | 38.50 |
| 8 | 0.55 | 38.32 |
| 9 | 0.55 | 38.18 |

### BART Union GLEU 추이

| Epoch | Val Loss | GLEU | Job |
|-------|----------|------|-----|
| 0 | 0.67 | 25.56 | 752395 |
| 1 | 0.64 | 26.78 | 752395 |
| 2 | 0.64 | 27.08 | 752395 |
| 3 | 0.65 | **27.57** | 752395 |
| 4 | 0.67 | 27.21 | 752395 |
| 5 | 0.68 | 27.40 | 752395 |
| 6 | 0.70 | 27.35 | 752395 |
| 7 | 0.71 | 27.84 | 752910 |
| 8 | 0.72 | 27.14 | 753328 |
| 9 | 0.73 | 26.93 | 753328 |

---

## BLT (facebook/blt-1b) — 학습 진행 중

`lr=1e-5, batch=1, grad_accum=4 (effective batch 4), weight_decay=0.1, cosine schedule, warmup=2000`

### BLT Native (~3,073 steps/epoch, 12,291 samples)

| Epoch | Val Loss | Best? | Job |
|-------|----------|-------|-----|
| 0 | 0.0774 | | 752408 |
| 1 | 0.0598 | | 752408 |
| 2 | 0.0597 | | 752408 |
| 3 | **0.0562** | best | 753647 |
| 4 | 0.0570 | | 753729 |
| 5 | 0.0613~0.0664 | | 753644 / 753861 |
| 6 | 0.0631~0.0692 | | 753644 / 753861 |
| 7 | (미완) | | 753861 |

소요시간 상세:

| Job | 구간 | 소요시간 | 비고 |
|-----|------|----------|------|
| 752408 | fresh → epoch 2 | 32min | 단일 GPU |
| 753647 | epoch 3 → 4 | 105min | 단일 GPU |
| 753729 | epoch 4 → 5 | 105min | 단일 GPU |
| 753861 | epoch 5 → 7 | 105min | DDP 2GPU |

누적 소요시간: **~347min (~5.8h)**, epoch 7 step 18,460 (last.ckpt 저장됨)

Epoch 3 이후 val_loss 상승 — **overfitting**. Best checkpoint는 epoch 3 (val_loss 0.0562).

### BLT Learner (~4,974 steps/epoch, 19,897 samples)

| Epoch | Val Loss | Best? | Job |
|-------|----------|-------|-----|
| 0 | 0.1419 | | 752409 |
| 1 | 0.1310 | | 752409 |
| 2 | 0.1317 | | 752409 |
| 3 | **0.1293** | best | 753730 |
| 4 | 0.1306 | | 753864 |
| 5 | (진행 중) | | 753878 (DDP 2GPU) |

소요시간 상세:

| Job | 구간 | 소요시간 | 비고 |
|-----|------|----------|------|
| 752409 | fresh → epoch 2 | 50min | 단일 GPU |
| 753730 | epoch 3 → 4 | 105min | 단일 GPU |
| 753864 | epoch 4 (val) | 56min | DDP 2GPU |
| 753878 | epoch 5 (진행 중) | 41min+ | DDP 2GPU |

누적 소요시간: **~252min (~4.2h)**, epoch 5 step 18,820

Epoch 3 이후 val_loss 상승 — **overfitting**. Best checkpoint는 epoch 3 (val_loss 0.1293).

### BLT Union (~27,221 steps/epoch, 108,883 samples)

| Job | Resume Step | Last Step | 소요시간 | 비고 |
|-----|-------------|-----------|----------|------|
| 753653 | 0 (fresh) | 3,160 | 105min | 단일 GPU |
| 753723 | 3,168 | 10,560 | 105min | 단일 GPU |
| 753744 | 10,565 | 17,870 | 105min | 단일 GPU |
| 753865 | 17,872 | 20,980 | 105min | 단일 GPU |
| 753881 | 20,981 | 21,000+ | running | 단일 GPU |

누적 소요시간: **~420min (~7.0h)**, epoch 0 step ~21,000 / 27,221 (약 77%)

Validation loss 없음 (첫 epoch 미완료). 남은 ~6,200 steps 예상 소요: 약 20분.

---

## 소요시간 비교 요약

| 모델 | 데이터셋 | 총 소요시간 | Jobs | Epochs 완료 | 시간/Epoch |
|------|----------|------------|------|-------------|-----------|
| BART | native | 22min | 1 | 10/10 | ~2min |
| BART | learner | 29min | 1 | 10/10 | ~3min |
| BART | union | 160min | 3 | 10/10 | ~16min |
| BLT | native | ~347min (~5.8h) | 4 | 7/10 진행 중 | ~50min |
| BLT | learner | ~252min (~4.2h) | 4 | 5/10 진행 중 | ~50min |
| BLT | union | ~420min (~7.0h) | 5 | 0/3 진행 중 | (미완) |

BLT는 BART 대비 epoch당 **약 15~25배** 느림. 바이트 단위 처리 + entropy patching + 큰 모델(1B vs 123M) 영향.

---

## 관찰 및 다음 단계

### BART
- 3개 데이터셋 모두 10 epoch 완료. 추가 학습 불필요.
- Union GLEU (27.57)가 native (49.76)에 비해 현저히 낮음. Val loss도 수렴하지 않고 상승 추세.

### BLT
- **Native/Learner**: epoch 3이 best. 이후 overfitting 확인. 추가 학습보다 best ckpt 기준 **GLEU generation 평가**가 우선.
- **Union**: epoch 0 완료까지 SLURM job 1회 추가 필요 (~20분). 이후 첫 val_loss 확인 가능.
- GLEU 평가는 `scripts/eval_blt.sh` (shard 기반) 또는 `TEST_ONLY=1` 모드로 수행 가능.
