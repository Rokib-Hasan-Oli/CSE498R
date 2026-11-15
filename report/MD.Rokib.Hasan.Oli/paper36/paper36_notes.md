# Paper #36: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)

*Reader: MD. Rokib Hasan Oli · new5 (added 27 Oct 2026, replaces #15) · Board group: Model backbones / OCR methods (small objects)*
*PDF: `Papers/6_6 Added papers (new5)/36_2022_SAHI Slicing Aided Hyper Inference Small Object Detection.pdf`
(arXiv 2202.06934 v5, 5 pages; IEEE ICIP 2022). Page numbers below are the PDF pages.*

---

## Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (2022)
```
Problem:   Detectors pre-trained on COCO-like photos (640 x 480, large objects) miss small and far objects in
           high-resolution surveillance / drone images: after resizing to the network input they are only a few pixels
           (p.1).
Method:    SAHI = slicing aided hyper inference: cut the image into overlapping M x N patches, resize each to the
           network input, detect, shift the boxes back, optionally add full-image inference (FI), merge with NMS
           (IoU 0.5) (p.2-3, Fig. 2). SF = slicing aided fine-tuning: add patches (480-640 px for VisDrone) cut from the
           training images to the training set (p.2-3). Works on top of any detector; open source (sahi package).
Data:      VisDrone2019-DET (8,599 drone photos, 10 classes; test-dev 1,610) and xView (satellite, 60 classes) (p.3).
           FCOS, VFNet, TOOD in MMDetection; SGD lr 0.01, decay at epochs 16 and 22; 640 px slices with 25 % overlap
           for VisDrone, 400 px for xView; AP50 with max 500 detections (p.4).
Results:   VisDrone AP50: TOOD 29.4 (FI) -> 34.7 (SAHI + FI + PO) -> 43.5 (SF + SAHI + FI + PO); small objects
           18.1 -> 23.8 -> 31.7 (p.3, Table 1). FCOS 25.8 -> 31.0 -> 38.5, VFNet 28.8 -> 33.9 -> 42.2. xView: FI 2.1-2.2
           -> SF + SAHI 17.0-20.6 (p.3, Table 2). Overlap (PO) helps small / medium objects, FI helps large ones (p.4).
Limits:    Drone / satellite data only; old mmdetection detectors; no speed numbers (only "increases linearly" with
           the slices, p.4); no comparison with simply training at a higher input resolution; two abstract numbers
           do not match Table 1.
For us:    845 of our 1,153 test photos are 1280 px and every kit reads them at 640 px: SAHI / SF are the cheapest
           ways to see far plates bigger, on our own data (it replaces #15, whose Kaggle data was never found).
Code:      https://github.com/obss/sahi (MIT, pip sahi); results / weights: https://github.com/fcakyon/small-object-detection-benchmark
BibTeX:    akyon2022sahi  (full entry at the bottom; confirm on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** small, far objects in large photos vanish when the photo is shrunk to the detector's input size.
2. **How:** detect on overlapping slices (inference) and/or train on slices (fine-tuning), merge with NMS.
3. **How well:** +5.1 to +5.3 AP50 from slicing alone and +12.7 to +14.1 with slice fine-tuning on VisDrone (p.3, Table 1).

**Useful for us?** Yes. Peer-reviewed (ICIP 2022), very widely used, MIT-licensed, a pip package that wraps Ultralytics
models directly. Our photos are 1280 px and far plates are a known weak spot (small-plate recall in #11, #27, #30,
#33), so it is a direct, cheap test. It needs nothing but our dataset.

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | IEEE ICIP 2022 (OBSS AI, METU, Ankara) | p.1 |
| SF | each training image cut into overlapping patches (VisDrone 480-640 px, xView 300-500 px), patches resized to width 800-1333 and added to the original images | p.2-3, p.4 |
| SAHI | M x N overlapping patches (VisDrone 640 x 640, xView 400 x 400), each resized, detected, shifted back; optional FI on the whole image | p.3 |
| Merge | NMS, matching threshold Tm 0.5 (IoU); low-score boxes removed | p.3, p.4 |
| PO | 25 % overlap between patches | p.4 |
| Detectors | FCOS, VarifocalNet, TOOD (MMDetection), SGD lr 0.01, momentum 0.9, wd 1e-4, 500-iteration warm-up, decay at epochs 16 and 22 | p.3-4 |
| Metric | COCO AP at IoU 0.5, overall and small / medium / large, max 500 detections | p.4 |
| Data | VisDrone2019-DET: 6,471 train / 548 val, Table 1 on test-dev; xView: random 75 / 25 split | p.3 |

**Every result number** (AP50; p.3, Table 1 VisDrone test-dev and Table 2 xView)
| Setting | FCOS | VFNet | TOOD | TOOD small / medium / large |
|---|---|---|---|---|
| FI (baseline) | 25.8 | 28.8 | 29.4 | 18.1 / 44.1 / 50.0 |
| SAHI (no overlap) | – | – | 31.9 | 22.6 / 44.0 / 45.2 |
| SAHI + PO | 29.0 | 32.0 | 32.5 | 22.8 / 45.2 / 43.6 |
| SAHI + FI | – | – | 34.6 | 23.8 / 48.5 / 53.1 |
| SAHI + FI + PO | 31.0 | 33.9 | 34.7 | 23.8 / 48.9 / 50.3 |
| SF + FI | – | – | 36.8 | 24.4 / 53.8 / 66.4 |
| SF + SAHI | – | – | 42.5 | 31.6 / 58.0 / 61.1 |
| SF + SAHI + PO | 38.1 | 41.9 | 43.1 | 31.7 / 59.0 / 60.2 |
| SF + SAHI + FI | – | – | 43.4 | 31.7 / 59.6 / 65.6 |
| **SF + SAHI + FI + PO** | **38.5** | **42.2** | **43.5** | 31.7 / 59.8 / 65.4 |
| xView FI -> best SF + SAHI | 2.2 -> 17.1 | 2.1 -> 17.7 | 2.1 -> 20.6 | |
| Speed / training time | not reported ("increases the computational time linearly", p.4) | | | |

Terms I looked up:
- **FI (full inference):** the normal way, one pass on the whole (resized) image.
- **PO (patch overlap):** neighbouring slices overlap by 25 %, so an object on a slice border is whole in at least one.
- **AP50_s / m / l:** AP at IoU 0.5 for objects < 32², 32²-96², > 96² pixels (COCO areas).

**Weak points I found:**
- **Abstract vs table.** The abstract says slicing adds 6.8 / 5.1 / 5.3 AP and with SF 12.7 / 13.4 / 14.5 (p.1); Table 1
  gives 5.2 / 5.1 / 5.3 and 12.7 / 13.4 / 14.1. FCOS's +6.8 and TOOD's +14.5 are not in the table.
- **No bigger-input baseline.** The obvious alternative, training / testing at 1280 px instead of slicing, is never
  tested; "memory stays fixed" is the only argument (p.4).
- **No speed.** Slicing multiplies the forward passes (a 1280 x 720 photo = 6 slices + FI) but no ms are given.
- **Different data.** Drone and satellite views (tiny objects everywhere) are not road photos with one or two plates.

## Pass 3: rebuild it
```
1280 px photo ─► 640 x 640 slices, 25 % overlap (+ the whole photo resized to 640 = FI)
             ─► YOLOv8s on every slice (batch 1) ─► boxes shifted back to photo coordinates ─► NMS IoU 0.5 ─► plates
training (SF): train photos + 640 x 640 slices of every photo > 800 px (plates >= 50 % inside kept)
```
**Assumptions:** objects are small relative to the photo; a slice keeps enough context; extra passes are affordable.

**What would fail on our Bangladeshi plates:**
- **Close-up plates** (most KUET and Roboflow photos) are large: slices cut them in half and add false boxes; FI is
  needed for them (the paper's large-object drop with PO, p.4).
- **640 px photos** (Roboflow exports) gain nothing: one slice = the whole photo.
- **Video / real time:** 6-10 passes per 1280 px frame instead of 1.

**How I reproduce it:**
- **Same data and split as every kit**, YOLOv8s: 640 px (60 epochs, the usual), SF (the 8,574 photos + about 11 k
  slices: 3,329 photos are > 800 px, giving 9,110 slices with a plate + 1,926 kept without; 26 epochs on 2.29x the
  images = the optimiser steps of 60 epochs), and 1280 px (20 epochs, batch 8), the bigger-input baseline the paper
  never ran.
- **Inference with the `sahi` package (pinned 0.12.8)** and the paper's merge (NMS, IoU 0.5): FI, SAHI (640 slices,
  25 % overlap) and SAHI + FI; AP50 all / small / medium / large (pycocotools, conf 0.01, max 500), plates found at conf
  0.25, small / night plates, false alarms on plate-free photos, ms per photo; the e2e photos read by our CRNN.
  **Kit update 29 Oct 2026** (after the v1.1 run): a fourth setting, SAHI + FI with cut-off slice boxes dropped (a
  slice box that touches an inner slice edge is removed before the merge; the validation-best merge of the run's
  study), and per-plate / per-photo files (`per_plate.csv`, `e2e_reads.csv`, `boxes.json`) for paired tests.
- **Original check:** the authors' 20 released VisDrone result files re-scored against the official test-dev labels.
- See the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper36/Kaggle_Training_Guide.md) and the [original-check guide](../../../src/MD.Rokib.Hasan.Oli/paper36/Original_Check_Guide.md).

## Check yourself (2-minute version)
> SAHI finds small, far objects by running the detector on overlapping slices of a big image (and optionally on
> slices during training), then merging the boxes with NMS. On VisDrone it lifts TOOD from 29.4 to 34.7 AP50 by slicing
> alone and to 43.5 with slice fine-tuning, mostly through small objects. It never reports speed, never compares with a
> simply larger input, and two abstract numbers disagree with its own table. For us: our test photos are 1280 px but
> read at 640, so slicing vs a 1280 px detector is a direct, cheap test for far plates.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection — SAHI (Akyon et al.) | 2022 | ICIP 2022 | VisDrone2019-DET (8,599 drone photos), xView | FCOS / VFNet / TOOD + slicing (any detector) | – | TOOD AP50 29.4 FI -> 34.7 SAHI -> 43.5 SF+SAHI; small 18.1 -> 31.7 (p.3) | Yes (sahi, MIT; result files public) | Aerial data only; no speed; no higher-resolution baseline; abstract vs table | Far / small plates in our 1280 px photos without new data |

## Code check → Research board
- **Code:** https://github.com/obss/sahi (MIT, 5.6 k stars, last push 16 Oct 2026); the kit pins `sahi==0.12.8` and uses
  `AutoDetectionModel.from_pretrained(model_type='ultralytics', ...)` with `get_sliced_prediction(...,
  postprocess_type='NMS', postprocess_match_metric='IOU', postprocess_match_threshold=0.5)`, the paper's merge (the
  package's default is GREEDYNMM with IOS).
- **Paper's results:** footnote 1 (`fcakyon/sahi-benchmark`) now redirects to
  https://github.com/fcakyon/small-object-detection-benchmark (last push 17 Jan 2025): mmdetection configs, release
  v0.0.1 with `result.json` + `eval.json` for every Table 1 / 2 row, release v0.0.2 with some SF weights.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/obss/sahi
```
```
MIT pip package. My T4x2 kit: YOLOv8s at 640 vs slice fine-tuning vs 1280 px, each with full-image / SAHI slices: small-plate AP50, plates found, ms, e2e read. Paper: TOOD AP50 29.4 -> 43.5 on VisDrone (p.3); check re-scores their 20 result files.
```

## BibTeX
```bibtex
@inproceedings{akyon2022sahi,
  title     = {Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection},
  author    = {Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  booktitle = {2022 IEEE International Conference on Image Processing (ICIP)},
  pages     = {966--970},
  year      = {2022},
  doi       = {10.1109/ICIP46576.2022.9897990}
}
```
*Pages and DOI as given in the sahi README citation; confirm on IEEE Xplore before the report.*

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper36_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper36/kaggle/paper36_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper36/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170, sahi==0.12.8, pycocotools; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 7.5 h for the full run |

| Run | GPU | What |
|---|---|---|
| `sf_slices` | CPU | SF training set: all train photos + 640 x 640 slices (20 % overlap) of every photo > 800 px |
| `det_v8s_640` | 0 | YOLOv8s at 640 px, 60 epochs (the usual detector, also used by SAHI) |
| `crnn` | 0 (after) | CRNN plate reader, 60 epochs (reads the e2e photos) |
| `det_v8s_sf` | 1 | YOLOv8s, slicing aided fine-tuning: 8,574 photos + ~11 k slices, 26 epochs (= the steps of 60 on photos) |
| `det_v8s_1280` | 1 (after) | YOLOv8s at 1280 px, 20 epochs, batch 8 (bigger input instead of slicing) |
| `sahi` | 0 (after) | FI / SAHI / SAHI+FI (640 and SF models), FI (1280 model): AP50 by size, plates, ms, e2e (the kit of 29 Oct adds SAHI+FI with cut-off slice boxes dropped and per_plate.csv) |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper36_outputs.zip` (team Drive `498R / weights / paper36 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #36: ours vs. the paper

| Metric | Paper (page) | det_v8s_640 | det_v8s_sf | det_v8s_1280 | crnn | sahi |
|---|---|---|---|---|---|---|
| Test mAP50, Ultralytics val on the full photo (%) | - | 98.5 | 97.5 | 98.0 | – | – |
| AP50, 640 model, full image (FI) (%) | TOOD + FI 29.4 (VisDrone) (p.3, Table 1) |  |  |  |  | 97.33 |
| AP50, 640 model, SAHI slices only (%) | TOOD + SAHI + PO 32.5 (p.3, Table 1) |  |  |  |  | 88.78 |
| AP50, 640 model, SAHI + FI (%) | TOOD + SAHI + FI + PO 34.7 (p.3, Table 1) |  |  |  |  | 91.63 |
| AP50, SF model, full image (%) | TOOD + SF + FI 36.8 (p.3, Table 1) |  |  |  |  | 96.65 |
| AP50, SF model, SAHI + FI (%) | TOOD + SF + SAHI + FI + PO 43.5 (p.3, Table 1) |  |  |  |  | 93.62 |
| AP50, 1280 px model, full image (%) | not in the paper |  |  |  |  | 97.10 |
| AP50 small plates: 640 FI (%) | TOOD + FI 18.1 (p.3, Table 1) |  |  |  |  | 26.01 |
| AP50 small plates: 640 SAHI + FI (%) | TOOD + SAHI + FI + PO 23.8 (p.3, Table 1) |  |  |  |  | 16.60 |
| AP50 small plates: SF SAHI + FI (%) | TOOD + SF + SAHI + FI + PO 31.7 (p.3, Table 1) |  |  |  |  | 12.58 |
| AP50 small plates: 1280 FI (%) | - |  |  |  |  | 16.07 |
| AP50 large plates: 640 FI / SAHI + FI (%) | TOOD 50.0 -> 50.3 (FI helps large objects) (p.3-4, Table 1) |  |  |  |  | 93.94 |
| Plates found at conf 0.25: 640 FI (%) | - |  |  |  |  | 98.12 |
| Plates found: 640 SAHI + FI (%) | - |  |  |  |  | 97.23 |
| Plates found: SF SAHI + FI (%) | - |  |  |  |  | 97.23 |
| Plates found: 1280 FI (%) | - |  |  |  |  | 97.63 |
| Small plates found: 640 FI / SAHI + FI gain (pp) | - |  |  |  |  | 40.00 |
| Small plates found: 1280 FI (%) | - |  |  |  |  | 60.00 |
| Boxes correct (precision) at conf 0.25: 640 SAHI + FI (%) | slices add false positives (p.4) (p.4) |  |  |  |  | 60.72 |
| False alarms on plate-free photos: 640 SAHI + FI (%) | - |  |  |  |  | 20.00 |
| ms per photo: 640 FI | - |  |  |  |  | 33.80 |
| ms per photo: 640 SAHI + FI | "increases linearly" with the slices (p.4) |  |  |  |  | 118.20 |
| ms per photo: 1280 FI | - |  |  |  |  | 46.10 |
| e2e photos plate exact: 640 FI (%) | - |  |  |  |  | 72.06 |
| e2e photos plate exact: 640 SAHI + FI (%) | - |  |  |  |  | 68.14 |
| e2e photos plate exact: 1280 FI (%) | - |  |  |  |  | 72.55 |
| Detector FPS, T4 TensorRT FP16 (640 / 1280 model) | - | 127.90 | – | 83.20 | – | – |
| Training time (min) | not reported (SGD lr 0.01, decay at epochs 16 and 22) (p.4) | 145.60 | 137.20 | 134.80 | 34.30 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's numbers are AP50 on VisDrone test-dev (drone photos, 10 classes, mmdetection FCOS / VFNet / TOOD); ours are one class (plate) on our test photos with YOLOv8s. Read the gains (FI -> SAHI -> SF), not the absolute values. AP50 here = pycocotools on the sahi predictions (conf 0.01, max 500 boxes); 'plates found' = conf 0.25, IoU 0.5 as in every kit.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_v8s_1280 | 98.0 | 98.4 | 94.3 | 97.7 | 91.4 | 76.4 | 97.6 | 5.62 |
| det_v8s_640 | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |
| det_v8s_sf | 97.5 | 97.5 | 96.5 | 97.4 | 96.3 | 85.5 | 97.8 | 6.25 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

### How to read this table (run checked 29 Oct 2026)
- **The run is sound:** dataset v1.1, 7 / 7 jobs `code=0` in 5.15 h, every training ran all its epochs (60 / 26 / 20,
  best 55 / 24 / 19), no NaN, code = the kit. On my PC (RTX 5060 Ti, same sahi 0.12.8 / Ultralytics 8.4.170) the same
  weights give the same numbers for all 7 SAHI settings (within 0.03), and every condition mAP50 above is reproduced
  exactly. The 640 detector is tensor for tensor the HVI-CIDNet (#35) run's; the CRNN is the #24 file.
- **"AP50 small plates" and "Small plates found" rows:** only **5** of the 1,011 test plates are smaller than 32 x 32 px
  in the photo, so these rows are noise (the "+40 pp gain" is 2 of 5 plates). The "AP50 large plates: 640 FI / SAHI + FI"
  row shows only the SAHI + FI value (FI: 97.61). Use the team's small-plate list (24 plates) or the plate width at 640
  px instead (report section 13). The kit of 29 Oct splits the large-plate row into FI and SAHI + FI, labels the
  COCO-small rows as such, and reports the small-plate list instead.
- **The paper's SAHI hurts on our plates:** 640 model AP50 97.33 -> 91.63 with SAHI + FI (bootstrap −5.7, CI −7.5 to
  −4.1), boxes correct 90.6 -> 60.7 %, plate-free photos with a box 10 -> 32 of 160 (22 vs 0, p < 0.0001), 3.5x the time.
  Cause: slice edges cut our (large) plates; the paper's NMS at IoU 0.5 keeps the boxes on plate parts (8 -> 278).
- **Fix chosen on val:** drop slice boxes that touch an inner slice edge, then NMS IoU 0.5 → AP50 97.51 (= FI, n.s.),
  997 vs 992 plates found (7 vs 2, p = 0.18), 24 / 24 small-list plates (FI 22); false alarms stay (29 vs 10 photos).
- **Labels leave out far plates:** in a seeded sample of 48 slice-only "false" boxes, 25 are real plates on far /
  secondary vehicles (6 unclear, 17 signs or painted text); 8 of the 10 sampled boxes on "plate-free" photos are plates.
- **SF (training on slices):** small-plate photos mAP50 84.5 -> 96.3 (+11.8, CI +2.8 to +20.5), clear photos −1.5
  (CI −2.1 to −0.1), overall −1.0 (n.s.); the 1280 px model is no better and 1.5x slower (only 20 epochs).
- **e2e rows:** my local reads are 1 photo lower than Kaggle in 5 of 7 settings: 33 of 198 e2e readings flip when the
  plate box moves by 1 px, so e2e differences of a few photos are noise.
- Full study: [SAHI_Study_Report.pdf](../../../weights/paper36/MD.Rokib.Hasan.Oli/SAHI_Study_Report.pdf) (19 pages; sources in [`source/`](source/)).

## Original-paper check (the paper's own data)
<!-- check:start -->
### Paper #36 on its own data: ours vs. the paper

| Metric | Paper (page) | visdrone_rescore |
|---|---|---|
| FCOS + FI: AP50 (%) | 25.8 (p.3, Table 1) | 25.80 |
| FCOS + SAHI + PO: AP50 (%) | 29.0 (p.3, Table 1) | 29.00 |
| FCOS + SAHI + FI + PO: AP50 (%) | 31.0 (p.3, Table 1) | 31.00 |
| FCOS + SF + SAHI + PO: AP50 (%) | 38.1 (p.3, Table 1) | 38.10 |
| FCOS + SF + SAHI + FI + PO: AP50 (%) | 38.5 (p.3, Table 1) | 38.50 |
| VFNet + FI: AP50 (%) | 28.8 (p.3, Table 1) | 28.80 |
| VFNet + SAHI + PO: AP50 (%) | 32.0 (p.3, Table 1) | 32.00 |
| VFNet + SAHI + FI + PO: AP50 (%) | 33.9 (p.3, Table 1) | 33.90 |
| VFNet + SF + SAHI + PO: AP50 (%) | 41.9 (p.3, Table 1) | 41.90 |
| VFNet + SF + SAHI + FI + PO: AP50 (%) | 42.2 (p.3, Table 1) | 42.20 |
| TOOD + FI: AP50 (%) | 29.4 (p.3, Table 1) | 29.40 |
| TOOD + SAHI: AP50 (%) | 31.9 (p.3, Table 1) | 31.90 |
| TOOD + SAHI + PO: AP50 (%) | 32.5 (p.3, Table 1) | 32.50 |
| TOOD + SAHI + FI: AP50 (%) | 34.6 (p.3, Table 1) | 34.60 |
| TOOD + SAHI + FI + PO: AP50 (%) | 34.7 (p.3, Table 1) | 34.70 |
| TOOD + SF + FI: AP50 (%) | 36.8 (p.3, Table 1) | 36.80 |
| TOOD + SF + SAHI: AP50 (%) | 42.5 (p.3, Table 1) | 42.50 |
| TOOD + SF + SAHI + PO: AP50 (%) | 43.1 (p.3, Table 1) | 43.10 |
| TOOD + SF + SAHI + FI: AP50 (%) | 43.4 (p.3, Table 1) | 43.40 |
| TOOD + SF + SAHI + FI + PO: AP50 (%) | 43.5 (p.3, Table 1) | 43.50 |
| TOOD + FI: AP50 small (%) | 18.1 (p.3, Table 1) | 18.10 |
| TOOD + SF + SAHI + FI + PO: AP50 small (%) | 31.7 (p.3, Table 1) | 31.70 |
| FCOS: SAHI gain over FI, AP50 (pp) | 6.8 (abstract) (p.1) | 5.20 |
| FCOS: SF + SAHI gain over FI (pp) | 12.7 (abstract) (p.1) | 12.70 |
| VFNet: SAHI / SF + SAHI gain (pp) | 5.1 / 13.4 (abstract) (p.1) | 13.40 |
| TOOD: SAHI gain over FI (pp) | 5.3 (abstract) (p.1) | 5.30 |
| TOOD: SF + SAHI gain over FI (pp) | 14.5 (abstract) (p.1) | 14.10 |
| Largest |re-scored - Table 1| AP50 (pp) | - | 0.00 |

Same data and split as the paper (its own published data / weights), so this IS a direct check of
the paper's numbers. Remaining differences come from software versions, hardware and randomness.

Same data and the authors' own predictions: a direct check of Table 1. 'SAHI gain' = best SAHI row without SF minus FI; 'SF + SAHI gain' = best row of that detector minus FI (the abstract's wording).

<!-- check:end -->

*This check was run on my PC (CPU, 27 Oct 2026) with the kit's `c36_sahi.py`, not on Kaggle; the zip was packed with
the check notebook's own report cell (`report_src/make_check_zip.py`). "VFNet: SAHI / SF + SAHI gain" shows only the SF
gain (13.40); the SAHI gain is 5.10. At a working threshold (score 0.3) the authors' own TOOD predictions lose precision
with slicing too (61.1 -> 47.5 %) but gain 11.8 points of recall, because 68 % of VisDrone objects are under 32 x 32 px
(report section 15).*

<!-- preview:start -->
### Local preview of the original-paper check (27 Oct 2026, my PC, the kit's `c36_sahi.py`, CPU)
The authors' 20 released VisDrone result files (release v0.0.1) re-scored with pycocotools against the official
VisDrone2019-DET test-dev labels (1,610 photos), AP at IoU 0.5 with max 500 detections:

| Check | Result |
|---|---|
| Table 1, all 20 rows x 4 columns (AP50, small, medium, large) | **80 / 80 numbers reproduced exactly** (largest difference 0.0) |
| Their own `eval.json` files | identical to Table 1 and to our re-score |
| Abstract "SAHI +6.8 / +5.1 / +5.3 AP" (FCOS / VFNet / TOOD) | table: **+5.2** / +5.1 / +5.3 (FCOS's +6.8 is not in the table) |
| Abstract "with SF +12.7 / +13.4 / +14.5 AP" | table: +12.7 / +13.4 / **+14.1** (TOOD's +14.5 is not in the table) |

So Table 1 is exactly what the released predictions give; only two numbers in the abstract are off. The image ids of
the result files follow the authors' own (unpublished) file order; it was recovered from box overlap and is frozen in
`c36_visdrone_map.py` (with it, the authors' eval.json is reproduced for every file). xView (Table 2) was not checked
(registration needed).
<!-- preview:end -->
