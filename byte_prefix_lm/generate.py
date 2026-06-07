"""Generate Korean GEC corrections from a trained byte Prefix-LM baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from byte_prefix_lm.data_adapter import BOS_ID, EOS_ID, SEP_ID, GecBltDataset
from byte_prefix_lm.model import BytePrefixTransformerConfig, BytePrefixTransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correction with byte Prefix-LM baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_gen_len", type=int, default=256)
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint):
    ckpt_args = checkpoint.get("args", {})
    config = BytePrefixTransformerConfig(
        max_length=int(ckpt_args.get("max_length", 1024)),
        dim=int(ckpt_args.get("dim", 256)),
        num_layers=int(ckpt_args.get("num_layers", 4)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        ffn_dim=int(ckpt_args.get("dim", 256)) * 4,
    )
    model = BytePrefixTransformerLM(config)
    model.load_state_dict(checkpoint["model"])
    return model


@torch.no_grad()
def generate(model, text: str, max_gen_len: int, device):
    model.eval()
    prefix = [BOS_ID] + GecBltDataset.text_to_bytes(text) + [SEP_ID]
    input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []

    for _ in range(max_gen_len):
        if input_ids.size(1) >= model.config.max_length:
            break
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits = model(input_ids, attention_mask=attention_mask)
        next_id = int(torch.argmax(logits[0, -1]).item())
        if next_id == EOS_ID:
            break
        if 0 <= next_id < 256:
            generated.append(next_id)
        else:
            break
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

    return GecBltDataset.bytes_to_text(generated)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    print(generate(model, args.text, args.max_gen_len, device))


if __name__ == "__main__":
    main()
