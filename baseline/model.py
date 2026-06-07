"""
KoBART GEC 모델 (LightningModule).
원본: https://github.com/soyoung97/Standard_Korean_GEC (Modified MIT License)
마이그레이션: pytorch-lightning 2.x → lightning 호환
주요 변경:
  - training_epoch_end → on_train_epoch_end
  - validation_epoch_end(outputs) → on_validation_epoch_end (수동 축적)
  - M2 scorer optional 연결
"""
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from pprint import pprint

import lightning as L
import torch
from transformers import get_linear_schedule_with_warmup

try:
    from transformers.optimization import AdamW as TransformersAdamW
except ImportError:  # pragma: no cover - depends on transformers version
    TransformersAdamW = None
from torch.optim import AdamW as TorchAdamW

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
        self.model.train()

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
        if self.args.adamw_correct_bias:
            optimizer = TorchAdamW(grouped, lr=self.args.lr)
        elif TransformersAdamW is not None:
            optimizer = TransformersAdamW(grouped, lr=self.args.lr, correct_bias=False)
        else:
            logging.warning(
                "transformers.optimization.AdamW is unavailable; falling back to "
                "torch.optim.AdamW with bias correction enabled."
            )
            optimizer = TorchAdamW(grouped, lr=self.args.lr)

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

    def on_train_epoch_start(self):
        self.model.train()

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
        was_training = self.model.training
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids, eos_token_id=1,
                max_length=self.args.max_seq_len, num_beams=self.args.num_beams)
        if was_training:
            self.model.train()
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

        p, r, f_score = self._run_m2_scorer(hyp_path)

        self.scores[self.current_epoch_idx] = {
            'precision': p, 'recall': r, 'f_score': f_score,
            'gleu': gleuscore, 'loss': total_loss.item()}

        print(f"\n\nEPOCH {self.current_epoch_idx} / {mode.upper()}_LOSS "
              f"{round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n")

        self.log(f'{mode}_loss', total_loss, prog_bar=False)
        self.log(f'{mode}_gleu', gleuscore)

        if self.args.best['gleu'] < gleuscore:
            self.args.best['gleu'] = gleuscore
            if f_score is not None:
                self.args.best['f0.5'] = f_score
                self.args.best['prec'] = p
                self.args.best['rec'] = r

        pprint(self.scores)

    def _run_m2_scorer(self, hyp_path):
        source_gold = getattr(self.args, 'm2_source_gold_path', '')
        if not source_gold:
            return None, None, None

        source_gold_path = Path(source_gold)
        if not source_gold_path.exists():
            logging.warning(f"M2 source-gold file does not exist: {source_gold_path}")
            return None, None, None

        scorer = Path(__file__).parent / "metric" / "m2scorer" / "scripts" / "m2scorer.py"
        cmd = [sys.executable, str(scorer), hyp_path, str(source_gold_path)]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            logging.warning(f"M2 scorer failed: {exc.stderr or exc.stdout}")
            return None, None, None

        metrics = {}
        for line in completed.stdout.splitlines():
            match = re.match(r"(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)", line.strip())
            if match:
                metrics[match.group(1)] = float(match.group(2))

        if not {"Precision", "Recall", "F_0.5"} <= set(metrics):
            logging.warning(f"Could not parse M2 scorer output: {completed.stdout}")
            return None, None, None
        return metrics["Precision"], metrics["Recall"], metrics["F_0.5"]
