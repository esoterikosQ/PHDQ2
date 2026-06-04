"""Train a byte-level Prefix-LM GEC model.

The default backend is a small repo-local byte Transformer for end-to-end
pipeline validation. It uses the same data format, checkpoint policy, and SLURM
resume behavior intended for the future official BLT backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from blt_gec.data_adapter import GecBltDataset, IGNORE_INDEX, PAD_ID
from blt_gec.model import BytePrefixTransformerConfig, BytePrefixTransformerLM


STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will save checkpoint after current step.")


def parse_max_time(value: str) -> int:
    parts = [int(x) for x in value.split(":")]
    if len(parts) != 4:
        raise ValueError("--max_time must use DD:HH:MM:SS")
    days, hours, minutes, seconds = parts
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def resolve_data_paths(args) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir)
    train_path = Path(args.train_data_path) if args.train_data_path else data_dir / f"{args.data}_train.tsv"
    val_path = Path(args.val_data_path) if args.val_data_path else data_dir / f"{args.data}_dev.tsv"
    test_path = Path(args.test_data_path) if args.test_data_path else data_dir / f"{args.data}_test.tsv"
    return train_path, val_path, test_path


def parse_args():
    parser = argparse.ArgumentParser(description="BLT-GEC byte Prefix-LM training")
    parser.add_argument("--data", type=str, default="native")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/blt_gec")
    parser.add_argument("--backend", type=str, default="mini", choices=["mini"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--log_every_steps", type=int, default=20)
    parser.add_argument("--checkpoint_interval_minutes", type=int, default=20)
    parser.add_argument("--max_time", type=str, default="00:01:50:00")
    parser.add_argument("--resume_ckpt_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model(args) -> BytePrefixTransformerLM:
    config = BytePrefixTransformerConfig(
        max_length=args.max_length,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ffn_dim=args.dim * 4,
    )
    return BytePrefixTransformerLM(config)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    print(f"Loaded checkpoint: {path} (epoch={epoch}, global_step={global_step})")
    return epoch, global_step, best_val_loss


@torch.no_grad()
def evaluate(model, dataloader, device, precision: str) -> float:
    model.eval()
    losses = []
    autocast_enabled = precision == "bf16" and device.type == "cuda"
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = compute_loss(logits, labels)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main():
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    args = parse_args()
    torch.manual_seed(args.seed)

    train_path, val_path, test_path = resolve_data_paths(args)
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    run_dir = Path(args.output_dir) / args.data
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")

    train_dataset = GecBltDataset(train_path, max_length=args.max_length)
    val_dataset = GecBltDataset(val_path, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    if args.resume_ckpt_path:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(args.resume_ckpt_path), model, optimizer, device
        )

    max_seconds = parse_max_time(args.max_time)
    checkpoint_interval = args.checkpoint_interval_minutes * 60
    start_time = time.time()
    last_checkpoint_time = start_time
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.max_epochs):
        total_batches = len(train_loader)
        for step_in_epoch, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = compute_loss(logits, labels) / args.grad_accum_steps

            loss.backward()

            if step_in_epoch % args.grad_accum_steps == 0 or step_in_epoch == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every_steps == 0:
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={loss.item() * args.grad_accum_steps:.4f}"
                    )

                if global_step % args.eval_every_steps == 0:
                    val_loss = evaluate(model, val_loader, device, args.precision)
                    print(f"validation step={global_step} loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            run_dir / "best.ckpt",
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            args=args,
                        )

                now = time.time()
                if now - last_checkpoint_time >= checkpoint_interval:
                    save_checkpoint(
                        run_dir / "last.ckpt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        args=args,
                    )
                    last_checkpoint_time = now

                if STOP_REQUESTED or now - start_time >= max_seconds:
                    save_checkpoint(
                        run_dir / "last.ckpt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        args=args,
                    )
                    print("Stopping early after saving resumable checkpoint.")
                    return

        val_loss = evaluate(model, val_loader, device, args.precision)
        print(f"validation epoch={epoch} loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                run_dir / "best.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
                args=args,
            )
        save_checkpoint(
            run_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_loss=best_val_loss,
            args=args,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
