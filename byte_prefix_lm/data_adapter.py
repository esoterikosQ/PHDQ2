"""Dataset utilities for byte-level Prefix-LM baseline training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


BYTE_VOCAB_SIZE = 256
BOS_ID = 256
EOS_ID = 257
SEP_ID = 258
PAD_ID = 259
VOCAB_SIZE = 260
IGNORE_INDEX = -100


@dataclass(frozen=True)
class GecBltExample:
    source: str
    target: str


class GecBltDataset(Dataset):
    """
    Byte Prefix-LM 방식의 GEC 데이터셋 어댑터.
    이 클래스는 BLT용이 아니다. 레거시 byte-only baseline 보존용이다.

    TSV 파일(오류문 \t 교정문)을 읽어들여 UTF-8 바이트 시퀀스로 변환.
    [BOS] + [오류문 바이트들] + [SEP] + [교정문 바이트들] + [EOS] 형태로 구성.
    """
    def __init__(
        self,
        tsv_path: str | Path,
        max_length: int = 1024,
        strict_tsv: bool = False,
    ):
        super().__init__()
        self.tsv_path = Path(tsv_path)
        self.max_length = max_length
        self.strict_tsv = strict_tsv
        self.data = self._load_tsv()

    def _load_tsv(self):
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

    @staticmethod
    def text_to_bytes(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def bytes_to_text(byte_ids: list[int]) -> str:
        valid_bytes = [x for x in byte_ids if 0 <= x < BYTE_VOCAB_SIZE]
        return bytes(valid_bytes).decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.data)

    def _build_sequence(self, source: str, target: str) -> tuple[list[int], list[int]]:
        src_bytes = self.text_to_bytes(source)
        tgt_bytes = self.text_to_bytes(target)

        # Keep at least BOS, SEP, one target-or-EOS slot. If the source is too
        # long, truncate it first; then fit as much target as possible.
        max_src_len = max(self.max_length - 3, 0)
        src_bytes = src_bytes[:max_src_len]
        max_tgt_len = max(self.max_length - len(src_bytes) - 3, 0)
        tgt_bytes = tgt_bytes[:max_tgt_len]

        input_ids = [BOS_ID] + src_bytes + [SEP_ID] + tgt_bytes + [EOS_ID]
        labels = [IGNORE_INDEX] * len(input_ids)

        sep_pos = 1 + len(src_bytes)
        for pos in range(sep_pos, len(input_ids) - 1):
            labels[pos] = input_ids[pos + 1]

        return input_ids, labels

    def __getitem__(self, idx):
        example = self.data[idx]
        input_seq, labels = self._build_sequence(example.source, example.target)

        pad_len = self.max_length - len(input_seq)
        if pad_len > 0:
            input_seq.extend([PAD_ID] * pad_len)
            labels.extend([IGNORE_INDEX] * pad_len)

        attention_mask = [0 if token_id == PAD_ID else 1 for token_id in input_seq]

        return {
            "input_ids": torch.tensor(input_seq, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_path = Path(tmpdir) / "dummy.tsv"
        with dummy_path.open("w", encoding="utf-8") as f:
            f.write("안뇽하세요\t안녕하세요.\n이거슨 테스트\t이것은 테스트.\n")

        dataset = GecBltDataset(dummy_path, max_length=32)
        sample = dataset[0]

        print("Source \\t Target : 안뇽하세요 \\t 안녕하세요.")
        print(f"input_ids: {sample['input_ids']}")
        print(f"labels:    {sample['labels']}")
