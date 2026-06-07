"""GEC dataset adapter for the reference Byte Latent Transformer.

The reference BLT tokenizer uses ids 0..3 for special symbols and represents
raw bytes as byte + OFFSET. We therefore do not invent a new SEP token. GEC
source and target are separated with a textual sentinel that is encoded as
ordinary bytes by the BLT tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
DEFAULT_GEC_SEPARATOR = "\n<BLT_GEC_SEP>\n"


class BltTokenizerProtocol(Protocol):
    bos_id: int
    eos_id: int

    def encode(
        self,
        text: str,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> list[int]:
        ...

    def decode(self, tokens: list[int], cut_at_eos: bool = False) -> str:
        ...


@dataclass(frozen=True)
class GecBltExample:
    source: str
    target: str


class GecBltDataset(Dataset):
    """TSV(source<TAB>target) dataset encoded with the reference BLT tokenizer."""

    def __init__(
        self,
        tsv_path: str | Path,
        tokenizer: BltTokenizerProtocol,
        *,
        max_length: int = 2048,
        separator: str = DEFAULT_GEC_SEPARATOR,
        strict_tsv: bool = False,
    ):
        super().__init__()
        self.tsv_path = Path(tsv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        self.strict_tsv = strict_tsv
        self.data = self._load_tsv()

    def _load_tsv(self) -> list[GecBltExample]:
        samples: list[GecBltExample] = []
        with self.tsv_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    samples.append(GecBltExample(parts[0], parts[1]))
                elif self.strict_tsv:
                    raise ValueError(
                        f"{self.tsv_path}:{line_no}: expected 2 tab-separated columns, got {len(parts)}"
                    )
        if not samples:
            raise ValueError(f"No valid TSV examples found in {self.tsv_path}")
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def _encode_pair(self, source: str, target: str) -> tuple[list[int], list[int]]:
        prefix_ids = self.tokenizer.encode(
            source + self.separator,
            add_bos=True,
            add_eos=False,
        )
        target_ids = self.tokenizer.encode(target, add_bos=False, add_eos=True)

        if len(prefix_ids) >= self.max_length:
            prefix_ids = prefix_ids[: self.max_length - 1]
            target_ids = [self.tokenizer.eos_id]

        max_target_len = max(self.max_length - len(prefix_ids), 1)
        target_ids = target_ids[:max_target_len]
        if target_ids[-1] != self.tokenizer.eos_id:
            target_ids[-1] = self.tokenizer.eos_id

        input_ids = prefix_ids + target_ids
        labels = [IGNORE_INDEX] * len(input_ids)
        target_start = len(prefix_ids)
        for pos in range(max(target_start - 1, 0), len(input_ids) - 1):
            labels[pos] = input_ids[pos + 1]
        return input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.data[idx]
        input_ids, labels = self._encode_pair(example.source, example.target)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class GecBltCollator:
    """Pads BLT batches with a valid token id while masking padded labels."""

    def __init__(self, pad_token_id: int):
        # Reference BLT uses PAD_ID=-1, which cannot be passed to embeddings.
        # EOS is a valid id; labels keep padding out of the loss.
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].numel()) for item in batch)
        input_batch = []
        label_batch = []
        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - int(input_ids.numel())
            if pad_len > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
                labels = torch.cat(
                    [
                        labels,
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    ]
                )
            input_batch.append(input_ids)
            label_batch.append(labels)

        return {
            "input_ids": torch.stack(input_batch),
            "labels": torch.stack(label_batch),
        }
