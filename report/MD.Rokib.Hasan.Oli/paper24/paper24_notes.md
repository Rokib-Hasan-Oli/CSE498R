# Paper #24: Leveraging Model Fusion for Improved License Plate Recognition

*Reader: MD. Rokib Hasan Oli · Board group: General LPR state of the art*
*PDF: `Papers/6_3 General LPR state of the art/24_2023_Leveraging Model Fusion for Improved LPR.pdf`
(author version, arXiv 2309.04331v2, 15 pages; CIARP 2023, LNCS pp. 60–75, DOI 10.1007/978-3-031-49249-5_5). Page
numbers below are the PDF pages.*

---

## Leveraging Model Fusion for Improved License Plate Recognition (2023)
```
Problem:   Single OCR models are strong on some datasets and weak on others. Does simply combining the outputs of
           several different recognition models give better and more stable plate reading, and how many/which?
Method:    Train 12 OCR models once on 8 datasets (+ 772 web plates + character-permuted synthetic plates):
           RARE, R2AM, STAR-Net, CRNN, GRCNN, Rosetta, TRBA, ViTSTR-Base (PyTorch), Holistic-CNN, Multi-Task-LR
           (Keras), CR-NET, Fast-OCR (Darknet YOLO) (p.5-7). Fuse the top-N models by Highest Confidence (HC),
           sequence Majority Vote (MV) or Majority Vote per Character Position (MVCP); ties broken by highest
           confidence (-HC) or by the best single model (-BM) (p.7-8). Plates are cut by a YOLOv4 detector,
           not from ground truth (p.8).
Data:      12 public datasets (Caltech Cars, EnglishLP, UCSD-Stills, ChineseLP, AOLP, SSIG-SegPlate, UFPR-ALPR,
           RodoSol-ALPR for train/test; OpenALPR-EU, PKU, CD-HARD, CLPD for cross-dataset tests) (p.5-6, Table 2).
Results:   Intra-dataset mean: best single model ViTSTR-Base 92.4 % → MV-HC (top 8) 97.6 % (p.9, Table 3).
           Cross-dataset mean: best single Fast-OCR 87.6 % → MV-BM (top 8) 90.3 % (p.11, Table 5). HC is the
           weakest strategy (96.0 % intra, 84.4 % cross). Speed: top-8 ensemble 59.7 ms (17 FPS); fusing the
           4-6 fastest models gives 95.4-96.6 % at 93-48 FPS (p.12, Table 6; Quadro RTX 8000).
Limits:    Latin/Chinese one- and two-row plates only; models trained once each (no seeds/variance); "best model"
           chosen on test performance across datasets (validation ranking said to be nearly the same, p.9);
           ensembles of 8 models are heavy for real time; confidence calibration across models not solved.
For us:    Cheap accuracy/robustness gain for our OCR: combine 3-4 different, fast Bangla OCR models (character-
           YOLO, CRNN, slot-CNN, transformer) by sequence majority vote; report speed vs. accuracy like Table 6.
Code:      no fusion code; models from https://github.com/roatienza/deep-text-recognition-benchmark and Darknet;
           splits at https://raysonlaroca.github.io/supp/lpr-model-fusion/
BibTeX:    laroca2023leveraging  (from the authors' page; full entry at the bottom)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** can fusing several OCR models raise and stabilise plate recognition?
2. **How:** 12 models, 12 datasets; fuse top-N outputs by highest confidence, sequence majority vote, or per-position vote.
3. **How well:** mean 92.4 → 97.6 % intra-dataset and 87.6 → 90.3 % cross-dataset (p.9, p.11); 4–6 fast models are the
   best speed/accuracy trade-off (p.12).

**Useful for us?** Yes. Peer-reviewed (CIARP, LNCS), simple to copy, and it directly targets our ≥ 90 % OCR goal:
fusion mostly removes the bad cases of each single model.

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | CIARP 2023 (Springer LNCS), pp. 60–75 | p.1 |
| Models | R2AM, RARE, STAR-Net, CRNN, GRCNN, Rosetta, TRBA, ViTSTR-Base (PyTorch fork of Clova AI); Holistic-CNN, Multi-Task-LR (Keras); CR-NET, Fast-OCR (Darknet) | p.5 |
| Training | Darknet: SGD, 90 k iterations, batch 64, lr 1e-3 → 1e-5; Keras: Adam 1e-3, batch 64, patience 11; PyTorch: Adadelta ρ 0.99, 300 k iterations, batch 128; input aspect 3:1 | p.5 |
| Data | 8 train/test datasets + 4 cross-dataset; 772 extra web plates; augmentation (crop, shadows, grey, HSV); character-shuffle synthetic plates | p.5–7 |
| Fusion | HC (most confident string), MV (most frequent string), MVCP (most frequent character per position); ties → highest confidence (HC) or best model (BM); raw confidences (normalising them did not help) | p.7–8 |
| Input crops | YOLOv4 detections (recall > 99.7 % intra, 97.8 % cross at IoU ≥ 0.7) | p.8 |
| Metric | plate recognition rate (all characters right) | p.8 |
| Hardware | Threadripper 1920X, 96 GB, Quadro RTX 8000 | p.5 |

**Every result number** (main ones)
| Result | Value | Page |
|---|---|---|
| Single models, intra mean | ViTSTR-Base 92.4, STAR-Net 92.2, TRBA 91.1, CR-NET 90.8, RARE 90.7, Fast-OCR 90.2, Holistic-CNN 89.4, Rosetta 89.4, GRCNN 88.2, R2AM 87.4, CRNN 87.1, Multi-Task-LR 85.9 % | p.9, Table 3 |
| Fusion, intra mean | HC (top 6) 96.0; MV-BM (top 8) 97.5; **MV-HC (top 8) 97.6**; MVCP-BM (top 9) 97.2; MVCP-HC (top 9) 97.5 % | p.9, Table 3 |
| Top-N curve (MV-HC) | top 2: 94.1, top 3: 94.9, top 4: 96.3, top 5: 96.6, top 6: 97.0, top 8: 97.6, top 12: 97.0 % | p.9, Table 4 |
| Single models, cross mean | Fast-OCR 87.6, CR-NET 87.1, STAR-Net 84.4, RARE 80.7, TRBA 79.6, CRNN 78.1, ViTSTR-Base 75.9 % | p.11, Table 5 |
| Fusion, cross mean | HC 84.4; **MV-BM 90.3**; MV-HC 90.1; MVCP-BM 86.3; MVCP-HC 87.6 % | p.11, Table 5 |
| Speed (ms per plate / FPS) | single: Multi-Task-LR 2.3 / 427, Holistic-CNN 2.5 / 399, CRNN 2.9 / 343, Fast-OCR 3.0 / 330, ViTSTR-Base 7.3 / 137, TRBA 16.9 / 59 | p.12, Table 6 |
| Speed-ranked fusion (MV-HC) | 3 fastest 91.1 % / 129 FPS; 4: 95.4 % / 93 FPS; 5: 96.0 % / 65 FPS; 6: 96.6 % / 48 FPS | p.12, Table 6 |
| Training time | not reported | – |

Terms I looked up:
- **Ensemble / model fusion:** run several models on the same input and combine their answers; it helps when the
  models make different mistakes.
- **MV vs. MVCP:** MV votes on whole strings. MVCP votes on each position, so it breaks when a model drops or adds a
  character (the positions shift). That is why MV won here.

**Weak points I found:**
- **Test-informed ranking.** "Best model" and the top-N order come from test results across datasets. The authors say
  the validation order is almost the same, but strictly the selection should use validation only.
- **No variance:** one training run per model.
- **Heavy for real time:** 8 models cost 59.7 ms per plate on a 48 GB GPU; on a T4/CPU this is much slower.
- **No two-row Bangla or other complex scripts.**

## Pass 3: rebuild it
```
plate crop ─┬─► model 1 (e.g. char-YOLO)  ─► "ঢাকা মেট্রো-গ ১২-৩৪৫৬" (0.91)
            ├─► model 2 (CRNN-CTC)         ─► "ঢাকা মেট্রো-গ ১২-৩৪৫৬" (0.85)
            ├─► model 3 (slot CNN)         ─► "ঢাকা মেট্রো-ঘ ১২-৩৪৫৬" (0.97)
            └─► model 4 (transformer)      ─► "ঢাকা মেট্রো-গ ১২-৩৪৫৮" (0.66)
                       └─► MV-HC: most frequent full string (ties → most confident) ─► final
```
**Assumptions:** the models are diverse (different architectures make different errors); confidences are roughly
comparable; there is time budget for several models.

**What would fail on our Bangladeshi plates:**
- **MVCP on Unicode characters** is fragile for Bangla (vowel signs and conjuncts change the length). I vote per
  **token** (district / মেট্রো / letter / 6 digits), which has fixed positions.
- **Correlated models:** if all models were trained on the same small set, they share errors on rare districts.
  Fusion does not invent knowledge.
- **Real time:** an ensemble only fits our ≥ 10 FPS goal with 2–4 fast models, run only on new tracks.

**How I reproduce it:**
- **Four fast OCR models**, different families, all trained on our OCR split:
  - GPU 0: character-YOLOv8n (CR-NET / Fast-OCR family), then the fast-plate-ocr slot CNN (Holistic / Multi-Task family)
  - GPU 1: CRNN-CTC, then a small ViTSTR-style transformer
- **Fusion on the test crops:** HC, MV-HC, MV-BM and token-MVCP-HC, for every subset of 2–4 models.
- **Report:** plate exact and CER for each model and each fusion, ms per plate on the T4 (Table-6 style), and the "best
  model" chosen on **validation**, not test.
- See the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper24/Kaggle_Training_Guide.md).

## Check yourself (2-minute version)
> They train 12 different OCR models on 12 public datasets and combine their outputs. Taking the string most models agree
> on (majority vote, ties broken by confidence) lifts the mean accuracy from 92.4 % for the best single model to 97.6 %
> (and 87.6 → 90.3 % on unseen datasets); trusting the single most confident model is the worst choice. Fusing 4–6 fast
> models gives most of the gain at ~50–90 FPS. For us: combine 3–4 fast, different Bangla OCR models by majority vote.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| Leveraging Model Fusion for Improved LPR (Laroca et al.) | 2023 | CIARP 2023 (Springer LNCS) | 12 public datasets (8 train/test, 4 cross) + 772 web plates + synthetic | YOLOv4 (crops for OCR) | 12 OCR models fused by HC / MV / MVCP | Mean 92.4 → 97.6 % intra (p.9), 87.6 → 90.3 % cross (p.11); 4–6 fast models best trade-off (p.12) | Splits only (models from public repos) | Latin/Chinese only; ranking uses test results; heavy ensembles; 1 run/model | Majority-vote fusion of 3–4 fast Bangla OCR models; speed-vs-accuracy table |

## Code check → Research board
- **Official code:** no fusion code is published. The split lists are at https://raysonlaroca.github.io/supp/lpr-model-fusion/.
  The models come from https://github.com/roatienza/deep-text-recognition-benchmark (PyTorch), Darknet
  (CR-NET / Fast-OCR) and Keras implementations.
- **Re-implementation:** fusion is ~40 lines (HC / MV / MVCP with tie-breaks) in my kit. The member models are our
  shared character-YOLO (Ultralytics), CRNN-CTC (PyTorch), fast-plate-ocr CNN and a ViTSTR-style transformer.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/roatienza/deep-text-recognition-benchmark
```
```
No fusion code (splits only). My T4x2 kit: 4 fast Bangla OCR models (char-YOLOv8n, CRNN, slot-CNN, ViTSTR-small) fused by HC / MV-HC / MV-BM / token-MVCP, ms per plate. Paper: mean 92.4→97.6 % with MV-HC top-8 (p.9).
```

## BibTeX
```bibtex
@inproceedings{laroca2023leveraging,
  title     = {Leveraging Model Fusion for Improved License Plate Recognition},
  author    = {Laroca, Rayson and Zanlorensi, Luiz A. and Estevam, Valter and Minetto, Rodrigo and Menotti, David},
  booktitle = {Iberoamerican Congress on Pattern Recognition (CIARP)},
  pages     = {60--75},
  year      = {2023},
  doi       = {10.1007/978-3-031-49249-5_5}
}
```
*From the authors' own BibTeX file (raysonlaroca.github.io/bibtex/laroca2023leveraging.txt).*

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper24_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper24/kaggle/paper24_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper24/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 4 h for the full run |

| Run | GPU | What |
|---|---|---|
| `chars_v8n` | 0 | character YOLOv8n, 100 epochs, 320 px |
| `slot` | 0 (after) | slot CNN, 100 epochs |
| `crnn` | 1 | CRNN-CTC, 60 epochs |
| `vitstr` | 1 (after) | ViTSTR-tiny, 60 epochs |
| `fusion` | 0 (after) | fusion table: paper rules + MV-BM+format + fallback, McNemar vs best single |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper24_outputs.zip` (team Drive `498R / weights / paper24 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #24: ours vs. the paper

| Metric | Paper (page) | chars_v8n | slot | crnn | vitstr | fusion |
|---|---|---|---|---|---|---|
| Single model plate exact (%) | best single 92.4 (ViTSTR-Base, intra) (p.9, Table 3) | 86.79 | 57.60 | 73.43 | 20.89 | – |
| Single model ms per plate | - | 9.36 | 1.15 | 6.47 | 10.18 | – |
| Best fusion of the paper's rules (chosen on val), test exact (%) | 97.6 (MV-HC, top 8) (p.9, Table 3) | – | – | – | – | 86.33 |
| Paper-rule fusion vs best single: exact McNemar p | - | – | – | – | – | 0.58 |
| Best fusion, any rule incl. MV-BM+format / fallback (chosen on val), test exact (%) | not in the paper | – | – | – | – | 88.63 |
| Chosen fusion rule | - | – | – | – | – | fallback |
| Gain over the best single model (pp) | +5.2 intra, +2.7 cross (p.9-11) | – | – | – | – | 1.84 |
| Plates the chosen fusion fixes vs the best single | - | – | – | – | – | 12 |
| Plates it breaks vs the best single | - | – | – | – | – | 0 |
| Chosen fusion vs best single: exact McNemar p | - | – | – | – | – | 0.0005 |
| Fusion ms per plate (sum; fallback: average) | 59.7 (top 8, RTX 8000) (p.12, Table 6) | – | – | – | – | 10.54 |
| Fallback: test plates that run the helpers (%) | - | – | – | – | – | 7.07 |
| Oracle: any model right (%) | - | – | – | – | – | 90.02 |
| Training time (min) | not reported | 52.10 | 23.70 | 23.90 | 24.50 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| chars_v8n | 86.79 | 90.83 | 80.00 | 74.11 | 80.77 | 75.88 |
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |
| slot | 57.60 | 62.29 | 27.69 | 54.46 | 42.31 | 51.46 |
| vitstr | 20.89 | 23.33 | 4.62 | 19.64 | 0.00 | 17.97 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

### How to read this table (hand-written 24 Oct 2026; add_results.py does not touch this part)
- **Which run:** the v1.1 Kaggle run of 22 Oct 2026 (dataset v1.1, 5 jobs code=0), checked on 24 Oct. It replaces the
  v1.0 run of 20 Oct, whose OCR test had 19 BLPD whole car photos (v1.0: char-YOLO 85.56, paper rule 86.18, fallback
  87.40 %). Full study: [Paper24_Study_Report.pdf](../../../weights/paper24/MD.Rokib.Hasan.Oli/Paper24_Study_Report.pdf); the v1.0 report is kept as
  `Paper24_Study_Report_v1.0_run.pdf`.
- **The "fusion" column** is the kit's row chosen on validation: char-YOLO + CRNN + ViTSTR as a fallback. The reader we
  recommend and share on GitHub is the two-model version (char-YOLO, then CRNN only when the BRTA layout fails). It has
  the same 88.63 % (12 fixed / 0 broken, McNemar p = 0.0005) at 9.82 ms; ViTSTR only adds one validation plate.
  The fusion run is not a row in `experiments.csv` (add_results.py skips analysis runs); it lives in this table.
- **The paper's own rules do not help:** 86.33 % (−0.46; 5 fixed / 8 broken, p = 0.58), as in the v1.0 run.
- **In the team table** (`docs/results/results_table.md`, "Plate readers v1.1"): paper #2's `lpnr_v12n` (87.71 %) sits
  above our `chars_v8n` (86.79 %). On the same 651 plates, 24 plates are read only by #2's model and 18 only by ours
  (McNemar p = 0.44), so the two character YOLOs are tied. Two trainings of the same model moved by 1-3 points in our
  v1.0/v1.1 runs, so do not rank readers this close from one run.
- **Cross-paper fusion (local, 24 Oct, `report_src/cross_fusion_p02_p24.md`):** `p24_fusion.py` on #2's `lpnr_v12n`
  + our `chars_v8n` + `crnn`. Chosen on validation: char-YOLOv8n, then #2's model only when the layout fails:
  88.79 % (13 fixed / 0 broken, p = 0.0002) at 11.0 ms. That is one plate more than with CRNN as the helper, so not a
  clear win. HC over the two character YOLOs reads 89.71 % on test, but it lost on validation (85.38 vs 86.55 %), so
  we do not claim it. The oracle of the three models is 91.71 % (90.02 % for our four), so a second character
  detector adds more diversity than our CRNN does.

## Original-paper check (the paper's own data)
<!-- check:start -->
*Not possible: split lists are public, but most of its 12 datasets need a request.* The main kit above runs the paper's method on our data instead.
<!-- check:end -->
