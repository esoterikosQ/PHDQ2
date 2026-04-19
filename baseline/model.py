"""
KoBART GEC 모델 (LightningModule).
원본: https://github.com/soyoung97/Standard_Korean_GEC (Modified MIT License)
마이그레이션: pytorch-lightning 2.x → lightning 호환
주요 변경:
  - training_epoch_end → on_train_epoch_end
  - validation_epoch_end(outputs) → on_validation_epoch_end (수동 축적)
  - AdamW: transformers.optimization → torch.optim
  - M2 scorer 호출부 정리 (원본 코드 버그 수정)
"""
import logging
import os
import time
from pathlib import Path
from pprint import pprint

import lightning as L
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from metric.gleumodule import run_gleu


class KoBARTConditionalGeneration(L.LightningModule):
    def __init__(self, args, model, tokenizer, datamodules):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.dm = datamodules

        self.pad_token_id = 0
        self.current_epoch_idx = 0
        self.step = 0
        self.generation_time = 0

        # 에폭별 축적 버퍼 (PL 2.x에서는 수동 관리)
        self.outputs = []
        self.decoded_labels = []
        self.origs = []
        self._val_losses = []

        self.scores = {}

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optimizer = AdamW(grouped, lr=self.args.lr)

        data_len = len(self.dm.train_dataloader().dataset)
        num_train_steps = max(
            int(data_len * self.args.max_epochs / self.args.batch_size),
            self.args.max_epochs)
        num_warmup_steps = int(num_train_steps * self.args.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        return self.model(
            input_ids=inputs['input_ids'],
            attention_mask=attention_mask,
            decoder_input_ids=inputs['decoder_input_ids'],
            decoder_attention_mask=decoder_attention_mask,
            labels=inputs['labels'],
            return_dict=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=False)
        self.step += 1
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch_idx in self.scores:
            self.scores[self.current_epoch_idx]['generation_time'] = self.generation_time
        print(f"\nGeneration time: {self.generation_time}")
        self.generation_time = 0
        self.current_epoch_idx += 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _generate(self, input_ids, labels):
        self.model.eval()
        start = time.time()
        output = self.model.generate(
            input_ids, eos_token_id=1,
            max_length=self.args.max_seq_len, num_beams=4)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.generation_time += time.time() - start

        decoded_label = self.tokenizer.batch_decode(
            labels.masked_fill(labels == -100, 1), skip_special_tokens=True)

        self.outputs += [x.replace('\n', '') for x in output]
        self.decoded_labels += decoded_label
        self.origs += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self._val_losses.append(loss)
        self._generate(batch['input_ids'], batch['labels'])
        return loss

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end(mode='val')

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self._val_losses.append(loss)
        self._generate(batch['input_ids'], batch['labels'])
        return loss

    def on_test_epoch_end(self):
        self._on_eval_epoch_end(mode='test')

    # ------------------------------------------------------------------
    # 평가 공통 로직
    # ------------------------------------------------------------------
    def _on_eval_epoch_end(self, mode='val'):
        total_loss = torch.stack(self._val_losses).mean()
        self._val_losses.clear()

        directory = f"outputs/generation/epoch{self.current_epoch_idx}/{mode}"
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        hyp_path = os.path.join(directory, f"hypothesis_{total_loss}.txt")
        ref_path = os.path.join(directory, f"reference_{total_loss}.txt")
        src_path = os.path.join(directory, f"source_{total_loss}.txt")

        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.outputs))
        with open(ref_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.decoded_labels))
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.origs))

        self.outputs.clear()
        self.decoded_labels.clear()
        self.origs.clear()

        gleu_out = run_gleu(reference=ref_path, source=src_path, hypothesis=hyp_path)
        gleuscore = float(gleu_out) * 100
        logging.info(f"\ngleu_value: {gleu_out}")

        with open(os.path.join(directory, f"gleu_{gleu_out}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"data: {self.args.data}, epoch: {self.current_epoch_idx}, "
                    f"gleu_out: {gleu_out}, val_loss: {total_loss}\n")

        # M2 scorer (별도 실행 — 원본 코드에서도 주석 처리 상태)
        # m2 파일이 필요하며, KAGAS를 통해 사전 생성해야 함
        p, r, f_score = 0, 0, 0

        self.scores[self.current_epoch_idx] = {
            'precision': p, 'recall': r, 'f_score': f_score,
            'gleu': gleuscore, 'loss': total_loss.item()}

        print(f"\n\nEPOCH {self.current_epoch_idx} / {mode.upper()}_LOSS "
              f"{round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n")

        self.log(f'{mode}_loss', total_loss, prog_bar=False)
        self.log(f'{mode}_gleu', gleuscore)

        if self.args.best['gleu'] < gleuscore:
            self.args.best['gleu'] = gleuscore
            self.args.best['f0.5'] = f_score
            self.args.best['prec'] = p
            self.args.best['rec'] = r

        pprint(self.scores)
