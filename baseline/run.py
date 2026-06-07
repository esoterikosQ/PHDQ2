"""
KoBART GEC 학습 진입점.
원본: https://github.com/soyoung97/Standard_Korean_GEC (Modified MIT License)
마이그레이션: pytorch-lightning 2.x → lightning 호환
주요 변경:
  - Trainer.from_argparse_args → 직접 Trainer 생성
  - strategy='dp' → strategy='auto'
  - wandb logger 제거 (필요 시 재추가)
"""
import argparse
import datetime
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from pprint import pprint

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

from dataset import GecDataModule, KoBARTGecDataset
from model import KoBARTConditionalGeneration


class SaveLastOnTrainEnd(Callback):
    """Write a final resumable checkpoint when Lightning stops before SLURM timeout."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step <= 0:
            return
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(self.checkpoint_path)
        print(f"Saved final resumable checkpoint: {self.checkpoint_path}")


class SaveBestAlias(Callback):
    """Copy the current best ModelCheckpoint file to a stable best.ckpt path."""

    def __init__(self, checkpoint_callback: ModelCheckpoint, alias_path: str):
        self.checkpoint_callback = checkpoint_callback
        self.alias_path = Path(alias_path)
        self._last_source = ""

    def _sync_alias(self):
        source = self.checkpoint_callback.best_model_path
        if not source or source == self._last_source:
            return

        source_path = Path(source)
        if not source_path.exists():
            return

        self.alias_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != self.alias_path.resolve():
            shutil.copy2(source_path, self.alias_path)
        self._last_source = source
        print(f"Saved best checkpoint alias: {self.alias_path}")

    def on_validation_end(self, trainer, pl_module):
        self._sync_alias()

    def on_train_end(self, trainer, pl_module):
        self._sync_alias()


def parse_args():
    parser = argparse.ArgumentParser(description='KoBART GEC Training')
    parser.add_argument('--data', type=str, default='native')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--name', type=str, default='default_name')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--every_n_epochs', type=int, default=1)
    parser.add_argument('--model_ckpt_path', type=str, default='')
    parser.add_argument('--init_ckpt_path', type=str, default='',
                        help='Model-weight checkpoint to initialize from without restoring trainer state.')
    parser.add_argument('--resume_ckpt_path', type=str, default='',
                        help='Training checkpoint to resume from. Use for interrupted SLURM jobs.')
    parser.add_argument('--checkpoint_interval_minutes', type=int, default=20,
                        help='Save a resumable checkpoint every N minutes during training.')
    parser.add_argument('--max_time', type=str, default='00:01:50:00',
                        help='Lightning max_time in DD:HH:MM:SS format. Default leaves time before 2h SLURM limit.')
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Beam size for validation/test generation. Use 1 for greedy comparison.')
    parser.add_argument('--adamw_correct_bias', action='store_true',
                        help='Use AdamW bias correction. Default off to match the reference Korean GEC code.')
    parser.add_argument('--m2_source_gold_path', type=str, default='',
                        help='Optional M2 source-gold file for m2scorer. If omitted, M2 is reported as unavailable.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Defaults to SLURM_CPUS_PER_TASK when available.')
    return parser.parse_args()


def get_device_count():
    if not torch.cuda.is_available():
        return 0
    try:
        devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    except KeyError:
        devices = torch.cuda.device_count()
    return devices


def write_command_log():
    command_line = "python3 " + ' '.join(sys.argv)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        devices = ','.join(str(x) for x in range(torch.cuda.device_count()))
    os.makedirs("logs", exist_ok=True)
    with open("logs/command_logs.txt", 'a') as f:
        f.write(f"[{cur_time}]: CUDA_VISIBLE_DEVICES={devices} {command_line}\n")


def main():
    write_command_log()
    args = parse_args()

    if args.debug:
        args.num_workers = 0
        args.batch_size = 16
    else:
        if args.num_workers is None:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus:
                args.num_workers = int(slurm_cpus)
            else:
                args.num_workers = min(int(multiprocessing.cpu_count() / 2), 8)

    args.best = {'gleu': 0, 'prec': 0, 'rec': 0, 'f0.5': 0}

    L.seed_everything(args.seed)
    print("Arguments:")
    pprint(vars(args))

    # ---- 모델 & 토크나이저 ----
    bart_model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')

    dm = GecDataModule(args, tokenizer, KoBARTGecDataset)
    model = KoBARTConditionalGeneration(args, bart_model, tokenizer, dm)

    if args.init_ckpt_path and args.resume_ckpt_path:
        raise ValueError("--init_ckpt_path and --resume_ckpt_path cannot be used together.")

    if args.init_ckpt_path:
        checkpoint = torch.load(args.init_ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Initialized model weights from {args.init_ckpt_path}")
        print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # ---- 콜백 ----
    checkpoint_interval = datetime.timedelta(minutes=args.checkpoint_interval_minutes)
    best_ckpt_callback = ModelCheckpoint(
        monitor='val_gleu',
        dirpath=f'outputs/{args.data}/',
        mode='max',
        verbose=True,
        save_last=False,
        save_top_k=3,
        every_n_epochs=args.every_n_epochs,
        filename=f'model_ckpt/{args.data}_{args.lr}_' + '{epoch:02d}_{step}')
    best_alias_callback = SaveBestAlias(best_ckpt_callback, f'outputs/{args.data}/best.ckpt')
    resumable_ckpt_callback = ModelCheckpoint(
        dirpath=f'outputs/{args.data}/',
        verbose=True,
        save_last=True,
        save_top_k=-1,
        train_time_interval=checkpoint_interval,
        every_n_epochs=0,
        filename=f'resume/{args.data}_{args.lr}_' + '{epoch:02d}_{step}')
    save_last_on_end = SaveLastOnTrainEnd(f'outputs/{args.data}/last.ckpt')

    # log_every_n_steps 조정
    data_len = len(dm.train_dataloader().dataset)
    log_steps = max(min(int(data_len / (args.batch_size * 2)), args.log_every_n_steps), 1)

    # ---- Trainer ----
    device_count = get_device_count()
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=device_count if device_count > 0 else 'auto',
        strategy='auto',
        callbacks=[best_ckpt_callback, best_alias_callback, resumable_ckpt_callback, save_last_on_end],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=log_steps,
        num_sanity_val_steps=0,
        max_time=args.max_time,
    )

    if args.model_ckpt_path == '':
        ckpt_path = args.resume_ckpt_path or None
        if ckpt_path:
            print(f"Resuming training from {ckpt_path}...")
        trainer.fit(model, dm, ckpt_path=ckpt_path)
    else:
        print(f"Loading model from {args.model_ckpt_path}...")
        model = KoBARTConditionalGeneration.load_from_checkpoint(
            checkpoint_path=args.model_ckpt_path,
            args=args, model=bart_model, tokenizer=tokenizer, datamodules=dm)
        trainer.validate(model, dataloaders=dm)


if __name__ == '__main__':
    main()
