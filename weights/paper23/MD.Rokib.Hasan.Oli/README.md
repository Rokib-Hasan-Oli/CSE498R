# Paper #23: Do We Train on Test Data? Near-duplicates in Bangla LPR (leaky vs fair split, dataset v1.1)

> **Project:** 498R BanglaLPR. Re-run of the idea of *"Do We Train on Test Data? The Impact of Near-Duplicates on
> License Plate Recognition"* (Laroca et al., IJCNN 2023, arXiv 2304.04653) on the KUET Bangladeshi plate dataset
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Run:** 22 October 2026, Kaggle 2× Tesla T4, BanglaLPR All-in-One **v1.1** (replaces the v1.0 run of 18 October)
> **Full study report:** [`Paper23_Study_Report.pdf`](Paper23_Study_Report.pdf)
>
> **Other files for this paper in this repo:** [finished Kaggle notebook](../../../notebooks/MD.Rokib.Hasan.Oli/paper23_run.ipynb) and [its log](../../../notebooks/MD.Rokib.Hasan.Oli/paper23_run.log) · [Kaggle kit + START_HERE guide](../../../src/MD.Rokib.Hasan.Oli/paper23/) · [reading notes + report source](../../../report/MD.Rokib.Hasan.Oli/paper23/). The full `paper23_outputs.zip` (with every `last.pt`) is on the team Drive in `498R / weights / paper23 /`.

The question: if a dataset's published train/test split puts other photos (or Roboflow copies) of the **same plate**
on both sides, how much does that inflate the score? We trained the same models twice on KUET:

- **leaky**: KUET's own published train/valid/test split, with every Roboflow copy (how papers #1 and #6 trained)
- **fair**: the same KUET images in our grouped split (seed 42): one copy per photo, and no test plate in training

Both plate readers are tested on **the same 194 KUET test plates** (KUET's published test crops). This is the paper's
CCPD-Fair design: one test set, two training sets.

---

## 1. At a glance

| | |
|---|---|
| **Models** | YOLOv8n plate detector (640 px) and YOLOv8n character detector used as plate reader (320 px), seed 42, SGD |
| **Training length** | about equal optimiser steps: det leaky 50 / fair 160 epochs; reader leaky 80 / fair 377 epochs (early stop, best 277) |
| **Leakage in KUET's published split** | reading: 114 of 194 test plates (58.8 %) are in train; detection: 79 of 553 test photos (14.3 %) have a copy in train |
| **Plate reading, same 194 plates** | leaky **85.57 %** vs fair **77.84 %**: error 14.4 % → 22.2 % (×1.54), exact McNemar **p = 0.008** (22 vs 7 discordant plates) |
| **Plates the leaky reader had seen (114)** | 90.35 % vs 79.82 % (+10.5, p = 0.017) |
| **Plates neither reader saw (80)** | 78.75 % vs 75.00 % (+3.75, p = 0.45, not significant) |
| **Other datasets, unseen by both (468 plates)** | 50.21 % vs 50.00 % (p = 1.0) |
| **Detection mAP50-95, duplicated test photos (79)** | leaky 83.06 vs fair 68.51 (+14.6); clean photos (474): 76.04 vs 73.06 (+3.0) |
| **Training time (T4)** | det_leaky 49.8 min · det_fair 41.1 min · rec_leaky 28.5 min · rec_fair 28.3 min (wall 96.3 min) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · dataset_version 1.1 |

The leaky models give exactly the same numbers as in the v1.0 run (deterministic training), so the result is reproducible.

## 2. Files

| File | Size | What it is |
|---|---|---|
| `README.md` | - | this guide |
| `best_det_leaky.pt` | 6.0 MiB | plate detector trained on KUET's published split (with copies) |
| `best_det_fair.pt` | 6.0 MiB | plate detector trained on our grouped split |
| `best_rec_leaky.pt` | 6.6 MiB | plate reader (character YOLO), published split |
| `best_rec_fair.pt` | 6.6 MiB | plate reader (character YOLO), grouped split |
| `paper23_results.zip` | 3.3 MiB | all results, logs, curves and code, no weights |
| `results_vs_paper.md` | - | the kit's ours-vs-paper table (with the seen/unseen table) |
| `paired_test_v11.json` | - | McNemar tests and 95 % intervals of the reading result (re-scored from the two best.pt) |
| `Paper23_Study_Report.pdf` | - | 14-page study report: paper, results, health check, how to use it |

The full output with `last.pt` files (`paper23_outputs.zip`, 48.9 MiB) is on the team Drive in `498R / weights / paper23 /`.

## 3. Quick start

The plate-text decoder is `code/vocab.py` inside `paper23_results.zip`. This snippet was tested on a KUET test photo and crop:

```python
from ultralytics import YOLO
import vocab                                   # code/vocab.py from paper23_results.zip

det = YOLO('best_det_fair.pt')                 # plate detector
boxes = det.predict('car.jpg', imgsz=640, conf=0.25)[0].boxes.xyxy

reader = YOLO('best_rec_fair.pt')              # character YOLO on a plate crop
r = reader.predict('plate_crop.jpg', imgsz=320, conf=0.25)[0]
toks = [[r.names[int(c)]] + b.tolist() for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]
text, valid = vocab.plate_text(toks)           # e.g. 'ঢাকা মেট্রো-ট ২৬-০২৯৫', True
```

## 4. Main results (KUET only)

**Plate reading, whole plate correct (%), the same 194 test plates**

| Which plates | plates | leaky | fair | only leaky right / only fair right | McNemar p |
|---|---|---|---|---|---|
| all | 194 | 85.57 | 77.84 | 22 / 7 | 0.008 |
| B: plate in the leaky training set (other photos) | 114 | 90.35 | 79.82 | 17 / 5 | 0.017 |
| C: plate in neither training set | 80 | 78.75 | 75.00 | 5 / 2 | 0.45 |
| other datasets (anpr, blpd, lpdba, nightlpbd) | 468 | 50.21 | 50.00 | 26 / 25 | 1.0 |

**Detection, mAP50-95 (%)** on KUET's published test set and its two parts (mAP50 is 98.5-99.5 % everywhere)

| Model | all (553) | clean (474) | duplicated (79) | our fair test (540) |
|---|---|---|---|---|
| det_leaky | 76.86 | 76.04 | 83.06 | 77.45* |
| det_fair (160 epochs) | 71.99 | 73.06 | 68.51 | 73.31 |

\* det_leaky trained on 20 of these 540 photos; on the 520 others it scores 76.78.

## 5. Known limits

- One dataset (KUET) and one seed. The 95 % intervals on 194 plates are about ±5 points (±9 on the 80 unseen plates).
- The fair detector overfits with 160 epochs (best epoch 102). The 50-epoch v1.0 fair detector scored 75.93 on clean photos, so
  the clean-photo difference (+3.0) depends on the recipe. The duplicated-photo advantage (+14.6; +14.8 in v1.0) does not.
- In `results_vs_paper.md`, the per-condition reading table **with** KUET crops is not a fair comparison: both readers trained on
  part of the main OCR test's KUET crops. Use the table "without the KUET crops" (50.21 % vs 50.00 %).
- The seen/unseen table prints each row twice, because the leaky and fair tests are now the same 194 crops.
- The v1.0 run's headline (85.57 % vs 72.13 %) compared different test plates and is replaced by this run.

## 6. Reproduce

Notebook `kaggle/paper23_kaggle.ipynb` (paper folder, rebuilt 21-22 Oct), Kaggle GPU T4 x2, Internet on, `FAST = False`.
It downloads dataset v1.1 itself (`extras/leaky_kuet/`, `*/fair_kuet_*.txt`). Run time 96 min.
The McNemar numbers come from `report_src/paired_v11.py` in the paper folder.

## 7. File integrity (SHA-256)

```text
ccd6050c273d3af45151bf3843ad480db708872cdd199461ebfb3851a9040339  best_det_leaky.pt
82fc9469464657bc682bdd8ed93faee02afc68c20c7874dfbd7be446689d105e  best_det_fair.pt
d8b93458f1549fb971caae88c5757dc8eb2d6a0d3c6b10c996a85c2ae654c1ee  best_rec_leaky.pt
2b4d79bdf2c7bf1edc9453f459f76336a9c0806033307b7902ddade55ac1ba09  best_rec_fair.pt
35d43e97cb2680f8cc0a4687d32df00b395d60aefce8009f7005a1ce7c70bd7a  paper23_results.zip
```

## 8. Citation

```bibtex
@inproceedings{laroca2023do,
  title     = {Do We Train on Test Data? {T}he Impact of Near-Duplicates on License Plate Recognition},
  author    = {Laroca, Rayson and Estevam, Valter and Britto Jr., Alceu S. and Minetto, Rodrigo and Menotti, David},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--8},
  year      = {2023},
  doi       = {10.1109/IJCNN54540.2023.10191584}
}
```
