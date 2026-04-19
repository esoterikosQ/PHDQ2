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
import sys
from pprint import pprint

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

from dataset import GecDataModule, KoBARTGecDataset
from model import KoBARTConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description='KoBART GEC Training')
    parser.add_argument('--data', type=str, default='native')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--name', type=str, default='default_name')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--every_n_epochs', type=int, default=10)
    parser.add_argument('--model_ckpt_path', type=str, default='')
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
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

    # ---- 콜백 ----
    ckpt_callback = ModelCheckpoint(
        monitor='val_gleu',
        dirpath=f'outputs/{args.data}/',
        mode='max',
        verbose=True,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=args.every_n_epochs,
        filename=f'model_ckpt/{args.data}_{args.lr}_' + '{epoch:02d}')

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
        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=log_steps,
        num_sanity_val_steps=0,
    )

    if args.model_ckpt_path == '':
        trainer.fit(model, dm)
    else:
        print(f"Loading model from {args.model_ckpt_path}...")
        model = KoBARTConditionalGeneration.load_from_checkpoint(
            checkpoint_path=args.model_ckpt_path,
            args=args, model=bart_model, tokenizer=tokenizer, datamodules=dm)
        trainer.validate(model, dataloaders=dm)


if __name__ == '__main__':
    main()
