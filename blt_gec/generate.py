"""Generate Korean GEC corrections with the reference BLT backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from blt_gec.data_adapter import DEFAULT_GEC_SEPARATOR
from blt_gec.generation import generate_correction
from blt_gec.model import ReferenceBltUnavailable, load_reference_blt_components


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correction with reference BLT")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--reference_code_dir", type=str, default="reference_code/blt")
    parser.add_argument("--blt_repo", type=str, default="facebook/blt-1b")
    parser.add_argument("--entropy_repo", type=str, default="facebook/blt-entropy")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--separator", type=str, default=DEFAULT_GEC_SEPARATOR)
    return parser.parse_args()


def main():
    args = parse_args()
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

    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location=device)
        components.model.load_state_dict(checkpoint["model"], strict=False)

    print(
        generate_correction(
            components.model,
            components.tokenizer,
            components.patcher,
            args.text,
            separator=args.separator,
            max_length=args.max_length,
            max_gen_len=args.max_gen_len,
            num_beams=args.num_beams,
            device=device,
        )
    )


if __name__ == "__main__":
    main()
