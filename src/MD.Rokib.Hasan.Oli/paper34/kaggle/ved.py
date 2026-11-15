"""Transformer OCR: Hugging Face VisionEncoderDecoder (image encoder + text decoder) on plate crops (ocr/{split}.csv).

  python ved.py --encoder google/vit-base-patch16-384 --decoder sagorsarker/bangla-bert-base --name vit_bbert ...
  python ved.py --encoder google/vit-base-patch16-384 --decoder xlm-roberta-base --name vit_xlmr ...
  python ved.py --trocr microsoft/trocr-base-stage1 --name trocr ...

Paper #1 settings (Table I): dropout 0.3 (encoder, decoder, attention), max length 20, beam 3, no-repeat n-gram 0,
length penalty 1.0, weight decay 0.01, fp16, batch 6. Paper #32: random flip + rotation ±5°, AdamW (flip is an option
here only to measure its effect). Images are padded to a square with grey (default) or stretched (--stretch).
Writes results.json + ocr_preds_test.csv and ocr_preds_val.csv (for fusion, paper #24).
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab
from seq import load_rows


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, processor, tokenizer, max_len, train, stretch, flip):
        self.rows, self.proc, self.tok, self.max_len, self.train, self.stretch, self.flip = rows, processor, tokenizer, max_len, train, stretch, flip

    def __len__(self):
        return len(self.rows)

    def image(self, path):
        im = Image.open(path).convert('RGB')
        if self.train:
            im = im.rotate(random.uniform(-5, 5), resample=Image.BILINEAR, expand=False, fillcolor=(127, 127, 127))
            if self.flip and random.random() < 0.5:
                im = ImageOps.mirror(im)
        if not self.stretch:
            w, h = im.size
            s = max(w, h)
            canvas = Image.new('RGB', (s, s), (127, 127, 127))
            canvas.paste(im, ((s - w) // 2, (s - h) // 2))
            im = canvas
        return im

    def __getitem__(self, i):
        r = self.rows[i]
        px = self.proc(images=self.image(r['path']), return_tensors='pt').pixel_values[0]
        text = vocab.N(r['text'])
        ids = self.tok(text, padding='max_length', max_length=self.max_len, truncation=True).input_ids
        labels = torch.tensor([t if t != self.tok.pad_token_id else -100 for t in ids])
        return px, labels, i


def build(a):
    from transformers import (AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel)
    if a.trocr:
        proc = TrOCRProcessor.from_pretrained(a.trocr)
        model = VisionEncoderDecoderModel.from_pretrained(a.trocr)
        image_proc, tok = proc.image_processor, proc.tokenizer
        model.config.decoder_start_token_id = model.config.decoder_start_token_id or tok.cls_token_id or tok.bos_token_id
    else:
        image_proc = AutoImageProcessor.from_pretrained(a.encoder)
        kw = {}
        if 'bert' in a.decoder.lower() and 'roberta' not in a.decoder.lower():
            kw = {'do_lower_case': False, 'strip_accents': False}   # BERT's accent stripping deletes Bangla vowel signs
        tok = AutoTokenizer.from_pretrained(a.decoder, **kw)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            a.encoder, a.decoder,
            encoder_hidden_dropout_prob=a.dropout, encoder_attention_probs_dropout_prob=a.dropout,
            decoder_hidden_dropout_prob=a.dropout, decoder_attention_probs_dropout_prob=a.dropout)
        model.config.decoder.is_decoder = True
        model.config.decoder.add_cross_attention = True
        model.config.decoder_start_token_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        model.config.eos_token_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gc = model.generation_config
    gc.decoder_start_token_id = model.config.decoder_start_token_id
    gc.eos_token_id = model.config.eos_token_id
    gc.pad_token_id = tok.pad_token_id
    gc.max_length, gc.num_beams, gc.no_repeat_ngram_size, gc.length_penalty, gc.early_stopping = a.max_len, a.beams, 0, 1.0, True
    return model, image_proc, tok


@torch.no_grad()
def predict(model, tok, ds, device, bs=16):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=min(2, common.WORKERS))
    out = []
    for px, _, ids in dl:
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            gen = model.generate(px.to(device), output_scores=True, return_dict_in_generate=True)
        texts = tok.batch_decode(gen.sequences, skip_special_tokens=True)
        scores = gen.sequences_scores.exp().tolist() if getattr(gen, 'sequences_scores', None) is not None else [0.0] * len(texts)
        for i, t, s in zip(ids.tolist(), texts, scores):
            r = ds.rows[i]
            out.append({'file': r['file'], 'true': r['text'], 'pred': vocab.N(t.replace(' ##', '').strip()),
                        'conf': round(float(s), 4), 'source': r['source']})
    ms = []
    for i in range(min(60, len(ds))):
        px = ds[i][0][None].to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            model.generate(px)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t0))
    return out, float(np.mean(ms[10:])) if len(ms) > 10 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', default='google/vit-base-patch16-384')
    p.add_argument('--decoder', default='sagorsarker/bangla-bert-base')
    p.add_argument('--trocr', default='', help='a full TrOCR checkpoint instead of encoder+decoder')
    p.add_argument('--name', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=6)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--max-len', type=int, default=20)
    p.add_argument('--beams', type=int, default=3)
    p.add_argument('--stretch', action='store_true')
    p.add_argument('--flip', action='store_true')
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-test', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--note', default='')
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    model, proc, tok = build(a)
    model.to(device)
    tr, va, te = load_rows('train')[:a.max_train], load_rows('val')[:a.max_test], load_rows('test')[:a.max_test]
    ds_tr = DS(tr, proc, tok, a.max_len, True, a.stretch, a.flip)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=common.WORKERS, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(dl) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / max(1, steps)))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    t0, log, step = time.time(), [], 0
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        for k, (px, labels, _) in enumerate(dl):
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                loss = model(pixel_values=px.to(device), labels=labels.to(device)).loss / a.accum
            scaler.scale(loss).backward()
            if (k + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            tot += float(loss) * a.accum * px.shape[0]; n += px.shape[0]
        vpred, _ = predict(model, tok, DS(va[:300], proc, tok, a.max_len, False, a.stretch, False), device)
        vs = metrics.score_rows((x['pred'], x['true']) for x in vpred)
        log.append({'epoch': ep + 1, 'train_loss': round(tot / max(1, n), 4), 'val_CER_%': vs['CER_%'],
                    'val_exact_%': vs['plate_exact_%']})
        print(log[-1], flush=True)
        if not any(l['val_CER_%'] < vs['CER_%'] for l in log[:-1]):
            model.save_pretrained(run / 'best'); tok.save_pretrained(run / 'best')
    train_min = (time.time() - t0) / 60
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(run / 'best').to(device)
    preds, ms = predict(model, tok, DS(te, proc, tok, a.max_len, False, a.stretch, False), device)
    vpred, _ = predict(model, tok, DS(va, proc, tok, a.max_len, False, a.stretch, False), device)
    r = {'run': a.name, 'kind': 'vision-encoder-decoder', 'note': a.note,
         'settings': {'encoder': a.encoder if not a.trocr else a.trocr, 'decoder': a.decoder if not a.trocr else a.trocr,
                      'epochs': a.epochs, 'batch': a.batch * a.accum, 'lr': a.lr, 'weight_decay': a.wd,
                      'dropout': a.dropout, 'max_len': a.max_len, 'beams': a.beams, 'pad_to_square': not a.stretch,
                      'flip': a.flip, 'steps': step, 'train_rows': len(tr)},
         'train_minutes': round(train_min, 1), 'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
         'val_ocr': metrics.score_rows((x['pred'], x['true']) for x in vpred),
         'ocr_test': {'rows_decoder': metrics.score_rows((x['pred'], x['true']) for x in preds), 'ms_per_plate': round(ms, 1),
                      'by_source': {s: metrics.score_rows((x['pred'], x['true']) for x in preds if x['source'] == s)
                                    for s in sorted({x['source'] for x in preds})}},
         'log': log, 'env': common.env_info()}
    root = common.find_dataset()
    r['ocr_test']['by_condition'] = metrics.condition_scores(preds, root)
    r['ocr_test']['extra_tests'] = {}
    for sp in metrics.extra_splits(root):            # real fog crops, synthetic rain copies
        ep, _ = predict(model, tok, DS(load_rows(sp)[:a.max_test], proc, tok, a.max_len, False, a.stretch, False), device)
        r['ocr_test']['extra_tests'][sp] = metrics.score_rows((x['pred'], x['true']) for x in ep)
    common.save_json(r, run / 'results.json')
    for name, rows in (('test', preds), ('val', vpred)):
        with open(run / f'ocr_preds_{name}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print('DONE', a.name, r['ocr_test']['rows_decoder'], f'{train_min:.1f} min')


if __name__ == '__main__':
    main()
