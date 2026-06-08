"""Standalone shardable generation evaluation for reference BLT-GEC."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from blt_gec.data_adapter import DEFAULT_GEC_SEPARATOR, GecBltDataset, GecBltExample
from blt_gec.generation import generate_correction
from blt_gec.metrics import compute_gleu, compute_m2
from blt_gec.model import ReferenceBltUnavailable, load_reference_blt_components


SHARD_RE = re.compile(r"^hypothesis_(\d{5})_(\d{5})\.txt$")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLT-GEC generation in shards")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--data", type=str, default="native")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="outputs/blt_eval")
    parser.add_argument("--reference_code_dir", type=str, default="reference_code/blt")
    parser.add_argument("--blt_repo", type=str, default="facebook/blt-1b")
    parser.add_argument("--entropy_repo", type=str, default="facebook/blt-entropy")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--separator", type=str, default=DEFAULT_GEC_SEPARATOR)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--m2_source_gold_path", type=str, default="")
    return parser.parse_args()


def resolve_split_path(args) -> Path:
    if args.split == "val" and args.val_data_path:
        return Path(args.val_data_path)
    if args.split == "test" and args.test_data_path:
        return Path(args.test_data_path)
    data_dir = Path(args.data_dir)
    return data_dir / f"{args.data}_{args.split}.tsv"


def checkpoint_name(args) -> str:
    if args.checkpoint_name:
        return args.checkpoint_name
    if args.checkpoint:
        return Path(args.checkpoint).stem
    raise ValueError("--checkpoint_name or --checkpoint is required")


def eval_output_dir(args) -> Path:
    return Path(args.output_dir) / args.data / args.split / checkpoint_name(args)


def count_valid_examples(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if len(line.rstrip("\n").split("\t")) == 2:
                count += 1
    if count == 0:
        raise ValueError(f"No valid TSV examples found in {path}")
    return count


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def write_lines(path: Path, lines: list[str]) -> None:
    if not lines:
        path.write_text("", encoding="utf-8")
        return
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def parse_shard_range(path: Path) -> tuple[int, int]:
    match = SHARD_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected shard filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def aggregate_shards(args) -> dict[str, float | int | None]:
    split_path = resolve_split_path(args)
    expected_total = count_valid_examples(split_path)
    out_dir = eval_output_dir(args)
    shard_files = sorted(out_dir.glob("hypothesis_[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9].txt"))
    if not shard_files:
        raise FileNotFoundError(f"No hypothesis shard files found in {out_dir}")

    ranges = [(parse_shard_range(path), path) for path in shard_files]
    ranges.sort(key=lambda item: item[0])
    if ranges[0][0][0] != 0 or ranges[-1][0][1] != expected_total:
        raise RuntimeError(
            f"Shard range mismatch: covers {ranges[0][0][0]}..{ranges[-1][0][1]}, "
            f"expected 0..{expected_total}"
        )
    for index in range(1, len(ranges)):
        prev_range = ranges[index - 1][0]
        current_range = ranges[index][0]
        if current_range[0] != prev_range[1]:
            raise RuntimeError(f"Shard gap/overlap: {prev_range} -> {current_range}")

    all_hyps: list[str] = []
    all_refs: list[str] = []
    all_srcs: list[str] = []
    for (start, end), hyp_path in ranges:
        ref_path = out_dir / f"reference_{start:05d}_{end:05d}.txt"
        src_path = out_dir / f"source_{start:05d}_{end:05d}.txt"
        hyps = read_lines(hyp_path)
        refs = read_lines(ref_path)
        srcs = read_lines(src_path)
        expected_len = end - start
        if not (len(hyps) == len(refs) == len(srcs) == expected_len):
            raise RuntimeError(
                f"Shard length mismatch for {start}..{end}: "
                f"hyp={len(hyps)} ref={len(refs)} src={len(srcs)} expected={expected_len}"
            )
        all_hyps.extend(hyps)
        all_refs.extend(refs)
        all_srcs.extend(srcs)

    hyp_path = out_dir / "hypothesis.txt"
    ref_path = out_dir / "reference.txt"
    src_path = out_dir / "source.txt"
    write_lines(hyp_path, all_hyps)
    write_lines(ref_path, all_refs)
    write_lines(src_path, all_srcs)

    gleu = compute_gleu(reference=ref_path, source=src_path, hypothesis=hyp_path)
    precision, recall, f_score = compute_m2(hyp_path, args.m2_source_gold_path)
    metrics = {
        "split": args.split,
        "examples": len(all_hyps),
        "gleu": gleu,
        "precision": precision,
        "recall": recall,
        "f0.5": f_score,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


@torch.no_grad()
def run_generation(args) -> dict[str, float | int | None]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --aggregate is set")
    split_path = resolve_split_path(args)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        components = load_reference_blt_components(
            reference_code_dir=args.reference_code_dir,
            blt_repo=args.blt_repo,
            entropy_repo=args.entropy_repo,
            local_files_only=args.local_files_only,
            device=device,
            precision=args.precision,
        )
    except ReferenceBltUnavailable as exc:
        raise SystemExit(f"Error: {exc}") from exc

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    components.model.load_state_dict(strip_module_prefix(checkpoint["model"]), strict=False)

    dataset = GecBltDataset(
        split_path,
        components.tokenizer,
        max_length=args.max_length,
        separator=args.separator,
    )
    total = len(dataset.data)
    start_index = max(args.start_index, 0)
    end_index = min(start_index + args.max_examples, total) if args.max_examples > 0 else total
    if start_index >= total:
        print(f"Shard start_index={start_index} is outside total={total}; nothing to do.")
        return {"split": args.split, "start_index": start_index, "end_index": total, "examples": 0}

    examples: list[GecBltExample] = dataset.data[start_index:end_index]
    out_dir = eval_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    hyps: list[str] = []
    refs: list[str] = []
    srcs: list[str] = []
    start_time = time.time()
    for example in examples:
        hyps.append(
            generate_correction(
                components.model,
                components.tokenizer,
                components.patcher,
                example.source,
                separator=args.separator,
                max_length=args.max_length,
                max_gen_len=args.max_gen_len,
                num_beams=args.num_beams,
                device=device,
            )
        )
        refs.append(example.target)
        srcs.append(example.source)

    hyp_path = out_dir / f"hypothesis_{start_index:05d}_{end_index:05d}.txt"
    ref_path = out_dir / f"reference_{start_index:05d}_{end_index:05d}.txt"
    src_path = out_dir / f"source_{start_index:05d}_{end_index:05d}.txt"
    write_lines(hyp_path, hyps)
    write_lines(ref_path, refs)
    write_lines(src_path, srcs)

    elapsed = time.time() - start_time
    metrics = {
        "split": args.split,
        "start_index": start_index,
        "end_index": end_index,
        "examples": len(examples),
        "generation_time": elapsed,
        "seconds_per_example": elapsed / max(len(examples), 1),
    }
    (out_dir / f"metrics_{start_index:05d}_{end_index:05d}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main():
    args = parse_args()
    if args.aggregate:
        aggregate_shards(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
