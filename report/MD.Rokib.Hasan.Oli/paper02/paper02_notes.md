# Paper #2: Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition

*Reader: MD. Rokib Hasan Oli · Board group: Bangla / Bangladeshi LPR*
*PDF: `Papers/6_1 Bangla_Bangladeshi LPR (core)/02_2025_Cascaded YOLO Bangladeshi LP Identification and Number Recognition.pdf`
(PLOS One 21(8): e0355858, published 10 Aug 2026, 42 pages). Page numbers below are the PDF pages (= journal pages).*

---

## Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition (2026)
```
Problem:   Read the full Bangladeshi plate (district, "Metro", class letter, 2 + 4 digits) without an OCR
           engine, with enough classes to cover every district name.
Method:    Three YOLOv12 models in a cascade (p.14-16, Algorithm 1): (1) COCO-pretrained YOLO12l finds the
           vehicle, (2) a custom YOLOv12 "LPD" model finds the plate inside the vehicle crop, (3) a custom
           YOLOv12 "LPNR" model detects 104 classes (digits, letters, "Metro", district WORDS) inside the plate
           crop; the labels are sorted left to right and joined. Ultralytics, 640 px, batch 16, lr0 0.01,
           optimizer "auto", no augmentation listed, 100 (LPD) / 70 (LPNR) epochs on a Colab T4 (p.15-18).
Data:      Dataset-01 (plates): 2,140 images, 2,251 boxes, 1,502 / 425 / 213. Dataset-02 (characters): 7,217
           plate images, 104 classes, 63,879 boxes, 5,058 / 1,438 / 721. Sources: own Khulna phone photos
           (daylight), web crawling and Roboflow (p.6, Table 2; p.16). External: 150 own images (p.32).
           Public: github.com/prosensarker4/BLPD (data + weights) and Kaggle ferozahmedds/bangladeshi-license-
           plate-detection-and-recognition (Apache 2.0).
Results:   LPD test: P 0.986, R 0.965, mAP@50 0.963, mAP@50-95 0.685 (p.23, Table 7); val mAP@50 0.974
           (p.21, Table 6). LPNR test: P 0.950, R 0.898, mAP@50 0.949, mAP@50-95 0.767 (p.25, Table 9).
           External 150 images: plate detection 100 %, full-plate exact 95.33 %, char-wise 98.82 %,
           Levenshtein similarity 99.57 % (p.34, Table 10). OCR (EasyOCR-type) baseline on the same 150: 15.33 %
           (p.37-38, Table 15). Latency per image, 3 models: GPU desktop (RTX 3050, p.17) mean 117.16 ms
           (≈ 8.5 FPS), max 284.31 ms; CPU laptop i5-8265U mean 2,311.53 ms; Jetson Orin Nano CPU mean 10.37 s
           (p.34-36, Tables 11-13). Training: LPD 100 epochs ≈ 2.42 h (87.3 s/epoch); LPNR 70 epochs ≈ 4.35 h on
           Colab T4 (p.17-18, p.24).
Limits:    External test is small (150) and chosen by the authors. Strong class imbalance (Munshiganj 2
           samples, p.9) and duplicated labels (105 names in LPNR_data.yaml: "Cox-s Bazar"/"Cox-s-Bazar",
           "DA"/"Da", "THA"/"Tha"). The abstract and Table 17 report the MAX GPU latency (284.31 ms) and the MAX
           Jetson time (25.4 s) next to the MEAN CPU time. Data from Roboflow/web may contain near-duplicates
           across splits (no de-duplication described). Jetson was run on CPU only.
For us:    The strongest public Bangla baseline and dataset: character/word-level YOLO (no OCR engine) reached
           95 % full-plate accuracy; we reuse its 104-class scheme idea, its data, and its numbers to compare.
Code:      https://github.com/prosensarker4/BLPD (official, linked on p.2; data, weights, eval notebook; no licence file)
BibTeX:    sarker2026cascaded  (full entry at the bottom; confirm on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** full Bangladeshi plate reading, including the many district names, without a general OCR.
2. **How:** YOLO12l (vehicle) → YOLOv12 plate detector → YOLOv12 detector of 104 plate "components"; join the labels.
3. **How well:** plate mAP@50 0.963 and component mAP@50 0.949 on test (p.23, p.25); 95.33 % full-plate exact on 150
   external images vs 15.33 % for OCR (p.34, p.38).

**Useful for us?** Yes, very. It is a peer-reviewed journal paper (PLOS One), with public data and weights, and its
method is the "character-level YOLO" in our plan. It is our main OCR-free baseline.

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | PLOS One 21(8): e0355858, received 14 Mar 2026, accepted 27 Jul 2026 (peer-reviewed, open access) | p.1 |
| Plate format | City code, "Metro" (metro areas), vehicle-type letter, 2-digit class + 4-digit number | p.2 |
| Dataset-01 (LPD) | 2,140 images, 2,251 boxes (Fig. 3 shows 1,561 instances), 1 class, Roboflow; 1,502 / 425 / 213 | p.6–7, Table 2 |
| Dataset-02 (LPNR) | 7,217 images, 104 classes, 63,879 boxes; 5,058 / 1,438 / 721; heavy imbalance ("1": 7,396 vs Munshiganj: 2) | p.7–10, Tables 2–3 |
| Sources | own phone photos in Khulna (Gollamari, Sonadanga, Zero Point), daylight only; web crawl; Roboflow | p.6 |
| Split | 70 / 20 / 10 for both datasets; external 150 images never used in training/tuning | p.6 |
| Models | stage 1: Ultralytics `yolo12l.pt` COCO (26.4 M params, 88.9 GFLOPs); stages 2–3: YOLOv12 (size not named; the repo's LPD_Model.pt is 40.8 MB ≈ "m", LPNR_Model.pt 5.7 MB ≈ "n", per the Kaggle card) | p.15, Kaggle card |
| Training | 640×640, batch 16, LPD 100 epochs, LPNR 70 epochs, optimizer auto, lr0 0.01, wd 5e-4, NMS IoU 0.7, "Data augmentation: No", cosine LR off, deterministic, patience 100 | p.18, Table 5 |
| Hardware | training: Colab T4 16 GB (p.15); testing: i5-8265U laptop (CPU), Ryzen 9 7900X + RTX 3050 (p.17), Jetson Orin Nano (CPU only) | p.13, p.17 |
| Decoding | sort detected components left to right, join: "Dhaka Metro Ga 328301" | p.16 |
| Metrics | P, R, F1, mAP@50, mAP@50-95; full-number exact match; character-wise mean accuracy; Levenshtein similarity | p.16–17 |

**Every result number**
| Result | Value | Page |
|---|---|---|
| LPD last training epoch | P 0.99259, R 0.96983, mAP@50 0.98057, mAP@50-95 0.697 | p.21, p.39 Table 16 |
| LPD validation (425 imgs, 464 plates) | P 0.993, R 0.970, mAP@50 0.974, mAP@50-95 0.674; best F1 0.98 at conf 0.518 | p.21–22, Table 6 |
| **LPD test (213 imgs, 226 plates)** | **P 0.986, R 0.965, mAP@50 0.963, mAP@50-95 0.685**; 219 TP, 7 FN, 6 FP; F1 0.97 at conf 0.669 | p.22–23, Table 7 |
| LPNR validation (1,438 imgs, 12,488 boxes) | P 0.952, R 0.905, mAP@50 0.950, mAP@50-95 0.77 | p.24, Table 8 |
| **LPNR test (721 imgs, 6,292 boxes)** | **P 0.950, R 0.898, mAP@50 0.949, mAP@50-95 0.767**; digits mAP@50 ≥ 0.985; weakest: Sa 0.346, Joypurhat 0.42, Pa 0.60 (mAP@50-95) | p.25–27, Table 9 |
| LPNR YOLO-version comparison (test mAP@50) | v8 0.936, v9 0.938, v10 0.897, v11 0.931, **v12 0.949** | p.37, Table 14 |
| **External 150 images** | plate detection 100 %, **full-number exact 95.33 % (143/150)**, char-wise mean 98.82 %, Levenshtein similarity 99.57 % | p.34, Table 10 |
| OCR-based pipeline on the same 150 | 23 correct = 15.33 % | p.37–38, Table 15 |
| Latency, laptop CPU (mean per image) | vehicle 911.39 + plate 1,091.19 + chars 232.92 = **2,311.53 ms** | p.35, Table 11 |
| Latency, RTX 3050 desktop (mean) | 39.63 + 45.18 + 19.51 = **117.16 ms** (std 37.74, min 69.00, max 284.31) | p.35, Table 12 |
| Latency, Jetson Orin Nano CPU (mean) | 4.535 + 5.145 + 0.656 = **10.367 s** (max 25.439) | p.36, Table 13 |
| Training time | LPD ≈ 8,730.75 s (2.42 h), 87.3 s/epoch; LPNR ≈ 15,654.2 s (4.35 h), ~223.6 s/epoch; Colab T4 | p.17–18, p.24 |

Terms I looked up:
- **Cascade:** each model works only inside the crop from the previous one, which shrinks the search area and
  cuts false positives.
- **Levenshtein similarity:** 1 − (edit distance / length). It rewards nearly-right plates.
- **YOLOv12:** the "attention-centric" YOLO (area attention + R-ELAN), see paper #26.

**Weak points I found:**
- **Reported latency.** The abstract and Table 17 quote 284.31 ms for the GPU desktop, which is the **maximum** in
  Table 12 (the mean is 117.16 ms), and 25.439 s for the Jetson, also the maximum (mean 10.367 s). The CPU number
  (2,311.53 ms) is the mean. So the summary mixes max and mean.
- **Instances.** Dataset-01 has 2,251 boxes in the text (p.6), but Fig. 3 shows 1,561.
- **Class list.** The yaml has 105 names with near-duplicate classes ("Cox-s Bazar" and "Cox-s-Bazar", "DA" and
  "Da", "THA" and "Tha", lower-case "panchagarh"). Duplicate classes split the training signal and make some
  errors count twice.
- **Leakage risk.** Roboflow and web images, 70/20/10 split, and no de-duplication step described. Our paper #23
  shows this can inflate scores.
- **No augmentation and no early stopping** (Table 5). With Ultralytics, "no augmentation" needs every aug set to
  0; the paper does not show that.
- **Section 2.2** describes YOLOv1 (S×S grid, anchors, 448 px). That is not how YOLOv12 works.

## Pass 3: rebuild it
```
frame ─► YOLO12l (COCO, vehicles) ─► vehicle crop ─► YOLOv12 LPD (1 class) ─► plate box (mapped back) ─► plate crop
                                                                                                      │
                                YOLOv12 LPNR (104 classes: 0-9, letters, Metro, 64 districts) ◄────────┘
                                                    │ sort by x (left → right), join labels
                                                    ▼
                                          "Dhaka Metro Ga 32 8301"
```
**Assumptions:** the vehicle detector always finds the vehicle (else the plate is lost); a plate has one reading
order (left to right); district names are seen often enough in training to be detected as single objects.

**What would fail on our Bangladeshi plates:**
- **Two lines.** Sorting only by x mixes the lines when the bottom row starts left of the district word. A
  row-split (cluster by y, then sort by x) is safer. I add that in our decoder and report both.
- **Rare districts.** With 2–50 training samples, rare districts will be missed or confused. Our merged recognition
  set adds KUET, LPDB-A and Foggy plates, but most of them are Dhaka too.
- **Night / blur / fog.** Their photos are daylight only (p.6). We test on our NightLPBD and Foggy parts separately.
- **Vehicle stage.** Background vehicles, CNG or rickshaw types that COCO does not know may be missed by YOLO12l.
  I measure the plate detector with and without the vehicle stage.
- **Speed.** 3 models ≈ 8.5 FPS on an RTX 3050. That is under our ≥ 10 FPS goal unless we drop the vehicle stage or
  use smaller models.

**How I reproduce it:** GPU 0 trains the LPD model (YOLOv12s, 100 epochs, 640, batch 16, paper settings) on our
merged plate split. GPU 1 trains the LPNR model (YOLOv12n, 70 epochs, 640, batch 16) on our merged character split.
Then the cascade (YOLO12l COCO → LPD → LPNR) runs on our test images. I score detection mAP, component mAP, and
full-plate exact / char-wise / Levenshtein accuracy on our text test set, plus latency on one T4. I also
re-evaluate the authors' own weights on their own test split. See the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper02/Kaggle_Training_Guide.md).

## Check yourself (2-minute version)
> Three YOLOv12 models in a row: COCO vehicle detector, a plate detector inside the vehicle, and a 104-class detector
> that finds digits, letters, "Metro" and whole district names inside the plate. Joining the labels left to right
> gives the plate text. Test mAP@50 is 0.963 for plates and 0.949 for components; on 150 external photos 95.3 % of
> plates are read exactly, against 15.3 % for an OCR engine. Weak spots: small external set, max-vs-mean latency in
> the summary, duplicated classes, and only ~8.5 FPS on an RTX 3050. For us it is the main OCR-free baseline.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded YOLO-based approach for Bangladeshi LP identification and number recognition (Sarker & Alam) | 2026 | PLOS One (journal) | LPD 2,140 imgs; LPNR 7,217 plate imgs / 104 classes; external 150; public | YOLO12l (vehicle) → YOLOv12 plate | YOLOv12 component detector (104 classes), left-to-right join, no OCR engine | Plate mAP@50 0.963 (p.23); comp. mAP@50 0.949 (p.25); exact 95.33 %, char 98.82 % (p.34); 117 ms/img RTX 3050 (p.35) | Yes (GitHub + Kaggle, data + weights) | Small external set; duplicate classes; max/mean latency mixed; no dedup; daylight only | Main OCR-free baseline; dataset in our merged set; numbers to beat |

## Code check → Research board
- **Official code and data:** https://github.com/prosensarker4/BLPD (paper p.2, "Data availability").
  - Last push: 25 Jun 2026. No licence file in the repo; the Kaggle copy (`ferozahmedds/bangladeshi-license-plate-detection-and-recognition`) says Apache 2.0.
  - Contents: `LPD_Model.pt` (40.8 MB), `LPNR_Model.pt` (5.7 MB), `LPD_data.yaml`, `LPNR_data.yaml` (105 names),
    `LPD_Dataset/`, `LPNR_Dataset/`, `BLPD_Result_Reproducibility.ipynb` (only `model.val` on val/test).
  - No training script and no `requirements.txt`. "Install ultralytics" is the only setup.
- **How I use it:**
  - Its detection and recognition images go into our merged dataset, after de-duplication and our own split.
  - Its weights are evaluated once on its own test split in the Kaggle notebook (a check that the paper's numbers
    reproduce). They are not used for training.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/prosensarker4/BLPD
```
```
Official data + YOLOv12 weights + val notebook (Ultralytics; no train script, no licence file; Kaggle copy Apache-2.0). My T4x2 kit retrains LPD+LPNR on our split and re-checks their weights. Paper: plate mAP50 0.963 (p.23), exact 95.33% (p.34).
```

## BibTeX
```bibtex
@article{sarker2026cascaded,
  title   = {Cascaded {YOLO}-based approach for {Bangladeshi} license plate identification and number recognition},
  author  = {Sarker, Proshenjit and Alam, S. M. Shamsul},
  journal = {PLOS One},
  volume  = {21},
  number  = {8},
  pages   = {e0355858},
  year    = {2026},
  doi     = {10.1371/journal.pone.0355858}
}
```
*Check the key on Google Scholar before the report.*

## Data audit (from building the 498R All-in-One dataset, 12 Oct 2026)
Checking BLPD's published split (Kaggle ferozahmedds copy) with verified near-duplicate search (perceptual hash + pixel
correlation, Roboflow base names):
- **LPD (plates):** 1,502 train photos are only 828 distinct photos. 94 of the 213 test photos (44.1 %) have a
  near-duplicate in train.
- **LPNR (characters):** 5,058 train crops are 3,350 distinct crops. 270 of the 721 test crops (37.4 %) are
  duplicates, and 345 test crops show a plate whose text also appears in train.
- **Labels:**
  - The plate boxes are loose (median width/height ratio 1.05, i.e. they often cover the plate surround).
  - Some images are mosaics with masked digits.
  - The class list has duplicate names ("Cox-s Bazar" / "Cox-s-Bazar").
- **External validation photos:** these are near-duplicates of public Bangla LPDB-A images.

So Tables 7, 9 and 10 are partly measured on seen data. In our dataset, BLPD's loose boxes are used only as hints
for auto-labels, and the external photos are kept as an end-to-end test (`e2e_test/`, 144 photos). The kit scores
them separately (`by_source.blpd_ext`).

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper02_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper02/kaggle/paper02_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper02/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 7 h for the full run |

| Run | GPU | What |
|---|---|---|
| `lpd_v12s` | 0 | YOLO12s plate detector, 60 epochs (paper: 100) |
| `lpnr_v12n` | 1 | YOLO12n character detector, 70 epochs, 640 px |
| `cascade_ours` | 0 (after) | cascade with our weights on e2e_test |
| `cascade_blpd_weights` | 1 (after) | cascade with the authors' LPD/LPNR weights |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper02_outputs.zip` (team Drive `498R / weights / paper02 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #2: ours vs. the paper

| Metric | Paper (page) | lpd_v12s | lpnr_v12n | cascade_ours | cascade_blpd_weights |
|---|---|---|---|---|---|
| LPD precision | 0.986 (p.23, Table 7) | 96.7 | – | – | – |
| LPD recall | 0.965 (p.23, Table 7) | 96.2 | – | – | – |
| LPD mAP50 | 0.963 (p.23, Table 7) | 98.9 | – | – | – |
| LPD mAP50-95 | 0.685 (p.23, Table 7) | 46.2 | – | – | – |
| LPNR mAP50 | 0.949 (p.25, Table 9) | – | 97.1 | – | – |
| LPNR mAP50-95 | 0.767 (p.25, Table 9) | – | 66.6 | – | – |
| OCR test crops: plate exact (%) | - | – | 87.71 | – | 74.35 |
| Cascade: plate found (%) | 100 (p.34, Table 10) | – | – | 93.14 | 88.24 |
| Cascade: full-plate exact (%) | 95.33 (p.34, Table 10) | – | – | 85.29 | 78.43 |
| Cascade: char-wise accuracy (%) | 98.82 (p.34, Table 10) | – | – | 92.78 | 88.18 |
| Cascade: Levenshtein similarity (%) | 99.57 (p.34, Table 10) | – | – | 90.84 | 84.65 |
| Cascade ms per photo (mean) | 117.16 (RTX 3050) (p.34, Table 11) | – | – | 93.30 | 96.40 |
| Without the vehicle stage: exact (%) | - | – | – | 86.76 | 77.94 |
| LPD FPS, T4 TensorRT FP16, batch 1 | - | 131.60 | – | – | – |
| Training time (min) | LPD 145, LPNR 261 (Colab T4) (p.17-18, p.24) | 178.40 | 94.40 | – | – |

Speed rows re-read from results.json by add_results.py: this zip comes from a kit before 22 Oct 2026, whose table showed the PyTorch FP16 mean timed without warming up every image shape (up to 8x too slow; paper #2: 7.1 FPS vs TensorRT 131.6). (fp32) = PyTorch FP32, the run has no TensorRT number; – = not measured (old kits had no FP16 median).

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

`cascade_blpd_weights` -> `e2e.*.by_source.blpd_ext` is the paper's own external set read with the paper's own weights (direct check of Table 10). Note: BLPD's external photos are near-duplicates of public LPDB-A images.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| lpd_v12s | 98.9 | 99.2 | 98.4 | 99.2 | 93.2 | 83.1 | 97.5 | 10.00 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| lpnr_v12n | 87.71 | 90.42 | 83.08 | 79.46 | 76.92 | 74.50 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

**Read with the table above (v1.1 run of 21 Oct 2026, checked 22 Oct):**
- Speed: the run's own table said "LPD FPS, T4 FP16 = 7.10", a timing artefact of the old kit's PyTorch FP16 loop.
  add_results.py now re-reads the real speed: **131.6 FPS with TensorRT FP16**, 52.9 FPS in PyTorch FP32 on the T4
  (`runs/lpd_v12s/results.json` → `speed`).
- The low LPD mAP50-95 (46.2) comes from one test source: ridoykj draws tighter boxes (24.2 there; **64.3** without ridoykj).
- Same 204 photos, our weights vs the authors': daylight tie (93.1 vs 95.1 %, McNemar p 0.58), night 66.7 vs 38.3 %
  (p < 0.001), all photos 85.3 vs 78.4 % (p 0.02). Full analysis and v1.0 vs v1.1: [Paper02_Study_Report.pdf](../../../weights/paper02/MD.Rokib.Hasan.Oli/Paper02_Study_Report.pdf).

## Original-paper check (the paper's own data)
<!-- check:start -->
The paper's data / weights are public, so a second notebook checks the printed numbers on the SAME data:
[`kaggle/paper02_original_check.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper02/kaggle/paper02_original_check.ipynb), steps in
[Original_Check_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper02/Original_Check_Guide.md). Runs: `official`, `lpd_repro`, `lpnr_repro`, `e2e_repro` (about 7.5 h).
After the run: `python add_results.py paper02_check_outputs.zip --member "Rokib Hasan Oli"` replaces this table.

| Metric | Paper (page) | Ours, same data |
|---|---|---|
| Authors' LPD, val mAP50 (%) | 97.4 (p.21, Table 6) | not run yet |
| Authors' LPD, test precision (%) | 98.6 (p.23, Table 7) | not run yet |
| Authors' LPD, test recall (%) | 96.5 (p.23, Table 7) | not run yet |
| Authors' LPD, test mAP50 (%) | 96.3 (p.23, Table 7) | not run yet |
| Authors' LPD, test mAP50-95 (%) | 68.5 (p.23, Table 7) | not run yet |
| Authors' LPNR, test precision (%) | 95.0 (p.25, Table 9) | not run yet |
| Authors' LPNR, test recall (%) | 89.8 (p.25, Table 9) | not run yet |
| Authors' LPNR, test mAP50 (%) | 94.9 (p.25, Table 9) | not run yet |
| Authors' LPNR, test mAP50-95 (%) | 76.7 (p.25, Table 9) | not run yet |
| Retrained LPD, test mAP50 (%) | 96.3 (p.23, Table 7) | not run yet |
| Retrained LPD, test mAP50-95 (%) | 68.5 (p.23, Table 7) | not run yet |
| Retrained LPNR, test mAP50 (%) | 94.9 (p.25, Table 9) | not run yet |
| Retrained LPNR, test mAP50-95 (%) | 76.7 (p.25, Table 9) | not run yet |
| External 150: plate found (%) | 100 (p.34, Table 10) | not run yet |
| External 150: full-plate exact (%) | 95.33 (p.34, Table 10) | not run yet |
| External 150: char-wise accuracy (%) | 98.82 (p.34, Table 10) | not run yet |
| External 150: Levenshtein similarity (%) | 99.57 (p.34, Table 10) | not run yet |
| Cascade ms per photo, mean | 117.16 (RTX 3050) (p.34, Table 11) | not run yet |
| Cascade ms per photo, max | 284.31 (RTX 3050) (p.34, Table 11) | not run yet |
| Training time LPD (min) | 145 (2.42 h, Colab T4) (p.17-18) | not run yet |
| Training time LPNR (min) | 261 (4.35 h, Colab T4) (p.24) | not run yet |
<!-- check:end -->

<!-- preview:start -->
### Local preview of the original-paper check (13 Oct 2026, my PC: RTX 5060 Ti, Ultralytics 8.4.170)
Our retraining with the paper's settings on the paper's own split, then the full cascade on the authors' 150 external photos. Times are on my GPU, not the paper's.

| BLPD own split, paper settings (YOLO12s LPD 100 ep., YOLO12n LPNR 70 ep., 640, batch 16) | Ours | Paper (page) |
|---|---|---|
| LPD val mAP50 (%) | 97.4 | 97.4 (p.21, Table 6) |
| LPD test P / R (%) | 99.1 / 96.4 | 98.6 / 96.5 (p.23, Table 7) |
| LPD test mAP50 / mAP50-95 (%) | 97.2 / 68.1 | 96.3 / 68.5 (p.23, Table 7) |
| LPNR test P / R (%) | 96.5 / 93.5 | 95.0 / 89.8 (p.25, Table 9) |
| LPNR test mAP50 / mAP50-95 (%) | 96.7 / 80.0 | 94.9 / 76.7 (p.25, Table 9) |
| Cascade, 150 external photos: plate found (%) | 100 | 100 (p.34, Table 10) |
| Cascade: full-plate exact (%) | 90.67 | 95.33 (p.34, Table 10) |
| Cascade: char-wise accuracy (%) | 98.05 | 98.82 (p.34, Table 10) |
| Cascade: Levenshtein similarity (%) | 98.42 | 99.57 (p.34, Table 10) |
| Cascade ms per photo, mean / max | 53.1 / 185.3 (RTX 5060 Ti) | 117.16 / 284.31 (RTX 3050, p.34, Table 11) |
| Training time LPD / LPNR (min) | 33 / 49 (RTX 5060 Ti) | 145 / 261 (Colab T4, p.17-18, p.24) |

The detectors reproduce Tables 6, 7 and 9 (LPNR even a little higher). End to end on the external photos we get 90.67 % full-plate exact vs 95.33 %: 7 more plates wrong out of 150. The authors' own weights are checked in the Kaggle notebook. Remember the data audit above: 44 % of the published LPD test photos have a near-duplicate in train, so these test numbers are optimistic for new photos.
<!-- preview:end -->
