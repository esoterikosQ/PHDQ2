"""Fine-tune the reference Byte Latent Transformer for Korean GEC.

This module intentionally depends on facebookresearch/blt's bytelatent code.
It has no byte-only fallback: if dynamic entropy patching is unavailable, the
job should fail rather than produce results mislabeled as BLT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from blt_gec.data_adapter import (
    DEFAULT_GEC_SEPARATOR,
    IGNORE_INDEX,
    GecBltCollator,
    GecBltDataset,
)
from blt_gec.generation import build_patch_lengths, generate_correction
from blt_gec.metrics import compute_gleu, compute_m2
from blt_gec.model import ReferenceBltUnavailable, load_reference_blt_components


STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will save checkpoint after current step.")


def setup_distributed() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def print_main(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def sync_should_stop(should_stop: bool, device: torch.device) -> bool:
    if not is_distributed():
        return should_stop
    stop_tensor = torch.tensor([int(should_stop)], device=device)
    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
    return bool(stop_tensor.item())


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
    parser = argparse.ArgumentParser(description="Reference BLT Korean GEC fine-tuning")
    parser.add_argument("--data", type=str, default="native")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/blt_gec")
    parser.add_argument("--reference_code_dir", type=str, default="reference_code/blt")
    parser.add_argument("--blt_repo", type=str, default="facebook/blt-1b")
    parser.add_argument("--entropy_repo", type=str, default="facebook/blt-entropy")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--eval_generation", action="store_true")
    parser.add_argument("--eval_max_examples", type=int, default=0,
                        help="Limit generation-based validation examples. 0 means full validation set.")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--checkpoint_interval_minutes", type=int, default=20)
    parser.add_argument("--max_time", type=str, default="00:01:50:00")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--m2_source_gold_path", type=str, default="")
    parser.add_argument("--resume_ckpt_path", type=str, default="")
    parser.add_argument("--run_test_on_end", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--separator", type=str, default=DEFAULT_GEC_SEPARATOR)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop training after this many optimizer steps. 0 means no limit.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_val_gleu: float,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_gleu": best_val_gleu,
        "args": vars(args),
    }
    torch.save(payload, path)
    print_main(f"Saved checkpoint: {path}")


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(strip_module_prefix(checkpoint["model"]), strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    best_val_gleu = float(checkpoint.get("best_val_gleu", -math.inf))
    print_main(f"Loaded checkpoint: {path} (epoch={epoch}, global_step={global_step})")
    return epoch, global_step, best_val_loss, best_val_gleu


@torch.no_grad()
def evaluate(model, patcher, dataloader, device, precision: str) -> float:
    model.eval()
    losses = []
    autocast_enabled = precision in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        patch_lengths = build_patch_lengths(patcher, input_ids)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            logits = model(input_ids, patch_lengths=patch_lengths)
            loss = compute_loss(logits, labels)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def build_scheduler(optimizer, args, train_loader_len: int):
    optimizer_steps_per_epoch = max(math.ceil(train_loader_len / args.grad_accum_steps), 1)
    total_steps = max(optimizer_steps_per_epoch * args.max_epochs, 1)
    warmup_steps = args.warmup_steps
    if args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    if args.scheduler == "constant":
        return None
    if args.scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def build_optimizer(model, args):
    no_decay = ("bias", "norm", "Norm")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    grouped = []
    if decay_params:
        grouped.append({"params": decay_params, "weight_decay": args.weight_decay})
    if no_decay_params:
        grouped.append({"params": no_decay_params, "weight_decay": 0.0})
    return AdamW(
        grouped,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )


@torch.no_grad()
def evaluate_generation(model, tokenizer, patcher, dataset, args, device, epoch: int, mode: str = "val"):
    examples = dataset.data
    if args.eval_max_examples > 0:
        examples = examples[: args.eval_max_examples]

    out_dir = Path(args.output_dir) / args.data / "generation" / f"epoch{epoch}" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / "hypothesis.txt"
    ref_path = out_dir / "reference.txt"
    src_path = out_dir / "source.txt"

    hyps = []
    refs = []
    srcs = []
    start = time.time()
    for example in examples:
        hyps.append(
            generate_correction(
                model,
                tokenizer,
                patcher,
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

    hyp_path.write_text("".join(f"{line}\n" for line in hyps), encoding="utf-8")
    ref_path.write_text("".join(f"{line}\n" for line in refs), encoding="utf-8")
    src_path.write_text("".join(f"{line}\n" for line in srcs), encoding="utf-8")

    gleu = compute_gleu(reference=ref_path, source=src_path, hypothesis=hyp_path)
    p, r, f_score = compute_m2(hyp_path, args.m2_source_gold_path)
    elapsed = time.time() - start
    metrics = {"gleu": gleu, "precision": p, "recall": r, "f0.5": f_score, "generation_time": elapsed}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    model.train()
    return metrics


def main():
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    try:
        args = parse_args()
        distributed = "LOCAL_RANK" in os.environ
        if distributed:
            local_rank = setup_distributed()
            device = torch.device(f"cuda:{local_rank}")
        else:
            local_rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        train_path, val_path, test_path = resolve_data_paths(args)
        for path in (train_path, val_path, test_path):
            if not path.exists():
                raise FileNotFoundError(path)

        run_dir = Path(args.output_dir) / args.data
        if is_main_process():
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "train_args.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
        if is_distributed():
            dist.barrier()

        print_main(f"Device: {device}")
        print_main(f"Distributed: {is_distributed()} rank={get_rank()} world_size={get_world_size()}")
        print_main("Backend: reference_blt")
        print_main(f"Reference BLT code: {args.reference_code_dir}")
        print_main(f"BLT repo: {args.blt_repo}")
        print_main(f"Entropy repo: {args.entropy_repo}")
        print_main(f"Train: {train_path}")
        print_main(f"Val:   {val_path}")
        print_main(f"Test:  {test_path}")

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

        model = components.model
        tokenizer = components.tokenizer
        patcher = components.patcher
        if not getattr(patcher, "realtime_patching", False) or getattr(patcher, "entropy_model", None) is None:
            raise RuntimeError("Reference BLT dynamic entropy patcher is not active.")

        if is_distributed():
            world_size = get_world_size()
            original_accum = args.grad_accum_steps
            args.grad_accum_steps = max(original_accum // world_size, 1)
            effective_batch = args.batch_size * args.grad_accum_steps * world_size
            print_main(
                f"DDP: world_size={world_size}, "
                f"grad_accum {original_accum} -> {args.grad_accum_steps}, "
                f"effective_batch_size={effective_batch}"
            )

        train_dataset = GecBltDataset(
            train_path,
            tokenizer,
            max_length=args.max_length,
            separator=args.separator,
        )
        val_dataset = GecBltDataset(
            val_path,
            tokenizer,
            max_length=args.max_length,
            separator=args.separator,
        )
        test_dataset = GecBltDataset(
            test_path,
            tokenizer,
            max_length=args.max_length,
            separator=args.separator,
        )
        collator = GecBltCollator(pad_token_id=tokenizer.eos_id)
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if is_distributed() else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )

        if is_distributed():
            model = DDP(model, device_ids=[local_rank])
        model_for_eval = raw_model(model)

        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, len(train_loader))
        start_epoch = 0
        global_step = 0
        best_val_loss = math.inf
        best_val_gleu = -math.inf
        if args.resume_ckpt_path:
            start_epoch, global_step, best_val_loss, best_val_gleu = load_checkpoint(
                Path(args.resume_ckpt_path), model_for_eval, optimizer, scheduler, device
            )

        if args.test_only:
            if is_main_process():
                test_loss = evaluate(model_for_eval, patcher, test_loader, device, args.precision)
                print(f"test loss={test_loss:.4f}")
                if args.eval_generation:
                    test_metrics = evaluate_generation(
                        model_for_eval,
                        tokenizer,
                        patcher,
                        test_dataset,
                        args,
                        device,
                        epoch=start_epoch,
                        mode="test",
                    )
                    print(f"test generation GLEU={test_metrics['gleu']:.4f} F0.5={test_metrics['f0.5']}")
            if is_distributed():
                dist.barrier()
            return

        max_seconds = parse_max_time(args.max_time)
        checkpoint_interval = args.checkpoint_interval_minutes * 60
        start_time = time.time()
        last_checkpoint_time = start_time
        autocast_enabled = args.precision in {"bf16", "fp16"} and device.type == "cuda"
        autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, args.max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            total_batches = len(train_loader)
            for step_in_epoch, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                patch_lengths = build_patch_lengths(patcher, input_ids)

                is_accum = step_in_epoch % args.grad_accum_steps != 0 and step_in_epoch != total_batches
                sync_context = model.no_sync() if is_distributed() and is_accum else nullcontext()
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                        logits = model(input_ids, patch_lengths=patch_lengths)
                        loss = compute_loss(logits, labels) / args.grad_accum_steps
                    loss.backward()

                if step_in_epoch % args.grad_accum_steps == 0 or step_in_epoch == total_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_every_steps == 0 and is_main_process():
                        elapsed_min = (time.time() - start_time) / 60
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={loss.item() * args.grad_accum_steps:.4f} "
                            f"elapsed={elapsed_min:.1f}m"
                        )

                    if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                        if is_main_process():
                            val_loss = evaluate(model_for_eval, patcher, val_loader, device, args.precision)
                            print(f"validation step={global_step} loss={val_loss:.4f}")
                            should_save_best = val_loss < best_val_loss
                            if args.eval_generation:
                                gen_metrics = evaluate_generation(
                                    model_for_eval,
                                    tokenizer,
                                    patcher,
                                    val_dataset,
                                    args,
                                    device,
                                    epoch=epoch,
                                    mode=f"val_step{global_step}",
                                )
                                val_gleu = gen_metrics["gleu"]
                                print(
                                    f"generation validation step={global_step} "
                                    f"GLEU={val_gleu:.4f} F0.5={gen_metrics['f0.5']}"
                                )
                                should_save_best = val_gleu > best_val_gleu
                                if should_save_best:
                                    best_val_gleu = val_gleu
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                            if should_save_best:
                                save_checkpoint(
                                    run_dir / "best.ckpt",
                                    model=model_for_eval,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    global_step=global_step,
                                    best_val_loss=best_val_loss,
                                    best_val_gleu=best_val_gleu,
                                    args=args,
                                )
                        if is_distributed():
                            dist.barrier()

                    now = time.time()
                    if sync_should_stop(now - last_checkpoint_time >= checkpoint_interval, device):
                        if is_main_process():
                            save_checkpoint(
                                run_dir / "last.ckpt",
                                model=model_for_eval,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                best_val_loss=best_val_loss,
                                best_val_gleu=best_val_gleu,
                                args=args,
                            )
                        last_checkpoint_time = now
                        if is_distributed():
                            dist.barrier()

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        if is_main_process():
                            print(f"Reached max_steps={args.max_steps}. Stopping.")
                            save_checkpoint(
                                run_dir / "last.ckpt",
                                model=model_for_eval,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                best_val_loss=best_val_loss,
                                best_val_gleu=best_val_gleu,
                                args=args,
                            )
                        if is_distributed():
                            dist.barrier()
                        return

                    elapsed = now - start_time
                    if sync_should_stop(STOP_REQUESTED or elapsed > max_seconds - 300, device):
                        if is_main_process():
                            save_checkpoint(
                                run_dir / "last.ckpt",
                                model=model_for_eval,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                best_val_loss=best_val_loss,
                                best_val_gleu=best_val_gleu,
                                args=args,
                            )
                            print("Stopping early to preserve checkpoint before time limit.")
                        if is_distributed():
                            dist.barrier()
                        return

            if is_main_process():
                val_loss = evaluate(model_for_eval, patcher, val_loader, device, args.precision)
                print(f"epoch={epoch} validation loss={val_loss:.4f}")
                should_save_best = val_loss < best_val_loss
                if args.eval_generation:
                    gen_metrics = evaluate_generation(
                        model_for_eval,
                        tokenizer,
                        patcher,
                        val_dataset,
                        args,
                        device,
                        epoch=epoch,
                        mode="val",
                    )
                    val_gleu = gen_metrics["gleu"]
                    print(f"epoch={epoch} generation GLEU={val_gleu:.4f} F0.5={gen_metrics['f0.5']}")
                    should_save_best = val_gleu > best_val_gleu
                    if should_save_best:
                        best_val_gleu = val_gleu
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if should_save_best:
                    save_checkpoint(
                        run_dir / "best.ckpt",
                        model=model_for_eval,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        best_val_gleu=best_val_gleu,
                        args=args,
                    )
                save_checkpoint(
                    run_dir / "last.ckpt",
                    model=model_for_eval,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    best_val_gleu=best_val_gleu,
                    args=args,
                )
            if is_distributed():
                dist.barrier()

        print_main("Training complete.")
        if args.run_test_on_end:
            if is_main_process():
                test_loss = evaluate(model_for_eval, patcher, test_loader, device, args.precision)
                print(f"test loss={test_loss:.4f}")
                if args.eval_generation:
                    test_metrics = evaluate_generation(
                        model_for_eval,
                        tokenizer,
                        patcher,
                        test_dataset,
                        args,
                        device,
                        epoch=args.max_epochs,
                        mode="test",
                    )
                    print(f"test generation GLEU={test_metrics['gleu']:.4f} F0.5={test_metrics['f0.5']}")
            if is_distributed():
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
