"""Generation helpers shared by BLT training and CLI inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_patch_lengths(patcher, input_ids: torch.Tensor) -> torch.Tensor:
    patch_lengths, _ = patcher.patch(input_ids, include_next_token=True)
    if patch_lengths.numel() == 0:
        raise RuntimeError("BLT patcher returned empty patch_lengths")
    return patch_lengths


@torch.no_grad()
def generate_correction(
    model,
    tokenizer,
    patcher,
    source: str,
    *,
    separator: str,
    max_length: int,
    max_gen_len: int,
    num_beams: int,
    device,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(source + separator, add_bos=True, add_eos=False)
    if len(prompt_ids) >= max_length:
        prompt_ids = prompt_ids[: max_length - 1]

    beams: list[tuple[list[int], float, bool]] = [(prompt_ids, 0.0, False)]
    beam_size = max(num_beams, 1)
    for _ in range(max_gen_len):
        candidates: list[tuple[list[int], float, bool]] = []
        for token_ids, score, done in beams:
            if done or len(token_ids) >= max_length:
                candidates.append((token_ids, score, True))
                continue

            tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
            patch_lengths = build_patch_lengths(patcher, tokens)
            logits = model(tokens, patch_lengths=patch_lengths)[:, -1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(log_probs[0], k=beam_size)
            for value, index in zip(top_values.tolist(), top_indices.tolist()):
                next_ids = token_ids + [int(index)]
                candidates.append((next_ids, score + float(value), int(index) == tokenizer.eos_id))

        def normalized(candidate: tuple[list[int], float, bool]) -> float:
            token_ids, score, _ = candidate
            generated_len = max(len(token_ids) - len(prompt_ids), 1)
            return score / generated_len

        beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
        if all(done for _, _, done in beams):
            break

    best_ids, _, _ = max(beams, key=lambda candidate: candidate[1] / max(len(candidate[0]) - len(prompt_ids), 1))
    return tokenizer.decode(best_ids[len(prompt_ids):], cut_at_eos=True).replace("\n", "")
