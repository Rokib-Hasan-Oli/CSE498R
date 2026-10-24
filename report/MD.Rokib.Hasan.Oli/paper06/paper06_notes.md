# Paper #6: YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles

*Reader: MD. Rokib Hasan Oli · Week 1 · Board group: Bangla / Bangladeshi LPR*
*PDF: `Papers/6_1 Bangla_Bangladeshi LPR (core)/06_2025_YOLOv8-Based LPR for Bangladeshi Vehicles.pdf` (22 pages; text on
p.1–17, references p.17–21). Page numbers below are the PDF pages. The PDF has no author names (template placeholders);
the authors come from the DOI record.*

---

## YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles (2025)
```
Problem:   Find Bangladeshi number plates in street / CCTV images and read their Bangla text, including
           low-resolution, dirty, partly hidden plates and bad weather.
Method:    COCO-pretrained YOLOv8 fine-tuned as a one-class plate detector ("Bangladeshi License Plate"),
           then EasyOCR's Bangla model reads the cropped plate. Boxes drawn and images augmented (blur, noise,
           weather) in Roboflow. Model size, image size, batch and learning rate are not given; the training
           plot (Fig. 6, p.11) is a standard Ultralytics run of about 118 epochs.
Data:      Own set of 2,600 photos (different light, angles, rain/dust; ~90 % taken from the left or right
           side), boxes drawn by hand in Roboflow, split 70/20/10, test set 260 photos (p.6-7, p.9).
           Not public, no link.
Results:   Detection "accuracy" 94.8 %, precision 93.5 %, recall 92.3 %, mAP 94.0 % (p.9, Table 1; the mAP
           matches mAP@50 in Fig. 6, p.11, where mAP@50-95 ends near 0.59). 78 FPS (p.12, Table 3; GPU not
           named; trained on Google Colab, p.8). EasyOCR character accuracy 92.7 % (p.11); by case: clean 97.5,
           low-res 92.0, damaged 85.6, bad weather 89.3 % (p.12, Table 2). Training time: not reported.
Limits:    Metrics are not defined ("detection accuracy", "character accuracy"). Augmentation was done before
           the split (p.7), so copies of one photo can sit in train and test. No data, no code, no settings.
           The text contradicts itself and the model (TensorFlow, anchor boxes, "object loss"; see below).
           Comparison table uses other papers' numbers on other data. Preprint, not peer-reviewed.
For us:    The simplest full baseline for our W2 plan, a YOLOv8 plate detector plus stock EasyOCR, with
           reference numbers (mAP 94.0 %, 78 FPS, 92.7 % char acc). Treat them as weak evidence.
           Verdict after the v1.1 run (24 Oct): KEEP the YOLOv8n detector (97.4 % mAP50, 183 FPS TensorRT;
           train it on human + auto boxes), DROP stock EasyOCR from the system (13.5 % of plates exact) and keep
           it only as the baseline row; the same crops read by a char-YOLO reader give 84.8-88.2 % exact.
Code:      None in the paper. Re-implemented with Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)
           + EasyOCR (https://github.com/JaidedAI/EasyOCR). Closest public Bangla version of the same pipeline:
           https://github.com/ShahinurAlamBhuiyan/Bangla-Number-Plate-Detection (YOLOv8n plate weights included).
BibTeX:    ahamed2025yolov8  (full entry at the bottom; confirm on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** detect and read two-line Bangladeshi plates from street/CCTV images, including bad conditions.
2. **How:** COCO-pretrained YOLOv8 fine-tuned to find plates (one class), then EasyOCR (Bangla) on the crop.
3. **How well:** P 93.5 %, R 92.3 %, mAP 94.0 % on 260 test photos (p.9), 78 FPS (p.12), 92.7 % character accuracy (p.11).

**Useful for us?** Yes, as a recipe: it is exactly the "YOLOv8n plate detector + EasyOCR baseline" in our W2 plan.
It is not strong evidence: an unreviewed preprint with undefined metrics, a leak-prone split, no data and no code.
Cite stronger papers for results.

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | Authorea preprint (Wiley), posted 21 Jan 2025, DOI 10.22541/au.173748276.69471512/v1; laid out in an Inderscience journal template ("Int. J. xxxx") | p.1 |
| Dataset | 2,600 raw photos, light / angle / rain / dust / low light; ~90 % from the left or right side | p.6–7 |
| Labels | boxes drawn by hand in Roboflow, one class "Bangladeshi License Plate" | p.7 |
| Augmentation | blur, noise, weather in Roboflow, **before** splitting; "three dataset propositions", each trained, best one kept | p.7 |
| Split | ~70 % train / 20 % val / 10 % test; test = 260 photos | p.7, p.9 |
| Model | YOLOv8, COCO-pretrained, fine-tuned (size n/s/m/l/x not given) | p.7 |
| Training | Google Colab GPU; dev PC Intel i5, 16 GB RAM; "TensorFlow", PyCharm, OpenCV, EasyOCR; epochs/batch/lr/img size not given; Fig. 6 shows ~118 epochs | p.8, p.11 |
| OCR | YOLOv8 box → "segment characters" → preprocess → EasyOCR (Bangla); no fine-tuning, no details of the preprocessing | p.8 |
| Metric words | "detection accuracy", "character recognition accuracy": never defined | p.9, p.11 |

**Every result number**
| Result | Value | Page |
|---|---|---|
| Detection accuracy / precision / recall / mAP | 94.8 / 93.5 / 92.3 / 94.0 % | p.9, Table 1 |
| mAP curve | > 0.90 by epoch 20, then 0.93–0.95 | p.10, Fig. 4 |
| Ultralytics training plot, last epoch (read off) | mAP@50 ≈ 0.95, mAP@50-95 ≈ 0.59, P ≈ 0.92, R ≈ 0.90, ~118 epochs | p.11, Fig. 6 |
| Character recognition accuracy | 92.7 % | p.11 |
| … by case (plates) | clean 97.5 % (100), low-res 92.0 % (80), damaged 85.6 % (50), weather 89.3 % (30). The 4 groups add up to 260 plates; their weighted mean is 92.6 % | p.12, Table 2 |
| Comparison (detection acc / FPS / mAP) | YOLOv8 94.8 / 78 / 94.0; YOLOv7 93.2 / 65 / 92.5; Faster R-CNN 92.1 / 42 / 93.3; SSD 89.5 / 85 / 90.2. Other rows are cited from other papers, so they are not on the same data | p.12, Table 3 |
| Adverse conditions | low light 91.5, rain 89.3, dust 87.2, partly hidden 85.0 % (metric not named) | p.13, Table 4 |
| Example OCR | two lines read as two separate results; top (letters) line confidence 0.52, number line 0.96 | p.14, Fig. 8 |
| Training time, model size, GPU for FPS | not reported | – |

Terms I looked up:
- **mAP@50 / mAP@50-95:** average precision with a box counted as right at IoU ≥ 0.5, or averaged over IoU 0.5–0.95 (stricter). Ultralytics plots both.
- **YOLOv8 losses:** box (CIoU), class (BCE) and DFL (distribution focal loss). YOLOv8 is **anchor-free** and has **no objectness loss**.
- **EasyOCR:** CRAFT text detector + CRNN recogniser. It reads whole words or lines, not single characters. The `bn` model is trained on general Bangla text, not plate fonts.

**Inconsistencies (they lower trust in the numbers):**
- It says YOLOv8 was implemented in **TensorFlow** (p.8), but YOLOv8 is PyTorch/Ultralytics, and Fig. 6 (p.11) is an Ultralytics `results.png`.
- It says "anchor boxes" were tuned (p.7). YOLOv8 has no anchors.
- "Object Loss" is plotted (p.10, Fig. 5), but its values match the val **DFL** loss in Fig. 6.
- The discussion says Faster R-CNN has a slightly higher mAP than YOLOv8 (p.13). Table 3 (p.12) shows the opposite.
- The conclusion says the work focuses "on detection rather than full character recognition" (p.17). The abstract claims recognition accuracy.
- In Table 3, the YOLOv7 row cites Redmon et al. 2015, which is YOLOv1.

## Pass 3: rebuild it
```
photo ──► YOLOv8 (COCO-pretrained, 1 class "plate") ──► plate box(es) ──► crop (+ preprocessing?)
                                                                               │
                                       EasyOCR 'bn':  CRAFT finds text lines ──┤
                                                      CRNN reads each line  ───┘
                                                                               ▼
                                             "ঢাকা মেট্রো-গ"  +  "১২-৩৪৫৬"   (two separate results)
```
**Assumptions:** one plate box gives the whole plate, and EasyOCR's general Bangla model can read plate fonts. The
paper also assumes the two lines come back in the right order, though it never says how it joins them. Its test
photos are assumed to be independent of its training photos.

**What would fail on our Bangladeshi plates (498R All-in-One: street, CCTV and whole-vehicle photos):**
- **Small plates.** In bus, truck and CNG photos the plate is only about 30–80 px wide, so the letters are 8–12 px tall.
  YOLOv8 at 640 should still find most plates, but EasyOCR on a tiny crop will miss letters. We need upscaling, a
  larger `imgsz`, or tiling. The kit reports the `low_res` OCR tag and the small-plate detection subset separately.
- **Two lines.** EasyOCR returns each line separately. If they are joined in the wrong order, a correct read still
  scores wrong. Our `p06_easyocr.py` joins the lines top to bottom.
- **Bangla letters.** Stock `bn` is weakest on the letters line (confidence 0.52 vs 0.96 for digits in the paper's
  own Fig. 8, p.14). Conjuncts in district names (চট্ট, ব্রাহ্মণবাড়িয়া) and the series letter are the risk.
- **Night, blur and weather.** Our test split has night photos (NightLPBD and dark photos), blurry ones and real fog
  copies of test plates, so the paper's low-light and weather claims (p.12-13) can be checked. There is no real rain
  data: the rain set is synthetic.
- **Stretched photos.** Part of our data (the DIU and other Roboflow exports) is 640×640, so those plates are squashed.
- **Several plates per photo.** Background vehicles have plates too. All of them are labelled; the whole-system test
  reads the most confident plate of each `e2e_test` photo.

**How I reproduce it:**
- **Code and settings:** Ultralytics 8.4.170, YOLOv8n and YOLOv8s (COCO-pretrained), 120 epochs, 640 px, batch 16,
  SGD, default augmentation, seed 42, then EasyOCR 1.7.2 `bn` (no fine-tuning, like the paper).
- **Data:** the 498R All-in-One dataset (downloaded by the notebook): the **human** plate boxes of the grouped
  70/15/15 split (like the paper's hand-drawn boxes), the 651 human-labelled OCR test crops, and the 204 `e2e_test`
  whole photos with their plate text.
- **Hardware:** Kaggle T4 x2, one model per GPU at the same time.
- **Steps:** [START_HERE.md](../../../src/MD.Rokib.Hasan.Oli/paper06/START_HERE.md) and the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper06/Kaggle_Training_Guide.md). The first kit (team DIU
  photos, draft boxes, typed plates) is kept in `_old_kit_2026-10-10/` for reference.

## Check yourself (2-minute version)
> They fine-tune a COCO-pretrained YOLOv8 to find Bangladeshi plates in 2,600 of their own photos, then read each
> plate with EasyOCR's Bangla model. They report mAP 94.0 % and 78 FPS for detection and 92.7 % "character
> accuracy" on 260 test photos. But no metric is defined, the photos were augmented before splitting (copies can
> leak into the test set), and there is no data or code. Some statements don't fit YOLOv8 (TensorFlow, anchors).
> For us it is the baseline recipe, YOLOv8 plate detector + EasyOCR, not a number to trust.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles (Ahamed & Ismail) | 2025 | Authorea preprint (not peer-reviewed) | Own Roboflow set, 2,600 photos, 70/20/10 (test 260), not public | YOLOv8, COCO-pretrained, 1 class (size not given) | Stock EasyOCR `bn` on the plate crop | P 93.5 %, R 92.3 %, mAP 94.0 % (p.9); 78 FPS (p.12); char acc 92.7 % (p.11) | No (re-implemented with Ultralytics + EasyOCR) | Undefined metrics; augmented before split (leak risk); no data/code/settings; text contradicts itself | Baseline recipe for our W2 detector + EasyOCR; reference numbers to compare against |

## Code check → Research board
- **Official code:** none. The PDF links no repository (its only URLs are statistics sources, p.18), and the DOI's
  Crossref record lists no code or data links.
- **Trusted libraries used for the re-implementation:**
  - Ultralytics YOLOv8 (`ultralytics==8.4.170`, AGPL-3.0, released 30 Sep 2026)
  - EasyOCR (`easyocr==1.7.2`, Apache-2.0)
- **Closest public Bangla versions of the same pipeline:**
  | Repository | Last push | Licence | What it has | Caveats |
  |---|---|---|---|---|
  | [ShahinurAlamBhuiyan/Bangla-Number-Plate-Detection](https://github.com/ShahinurAlamBhuiyan/Bangla-Number-Plate-Detection) | 4 Jul 2025 | no licence file; README: "educational and research purposes only" | YOLOv8n plate weights in the repo (`licensePlatemodel/best (1).pt`: one class, 100 epochs, 640 px, Ultralytics 8.0.3) + EasyOCR | scripts run on hard-coded local video paths |
  | [renzhamin/bengali-alpr](https://github.com/renzhamin/bengali-alpr) | Sep 2023 | GPL-3.0 | YOLOv8 plate model + an EasyOCR `bn` model fine-tuned on plates, both on Google Drive | |
- **How I use them:** not at all any more. The first kit (10 Oct) used the Shahinur detector only to draw draft
  boxes on the team DIU photos; the All-in-One dataset now has human plate boxes, so the current kit needs neither
  repo. I never ran either repo's own scripts.
- **Runs?**
  - The current notebook (`kaggle/paper06_kaggle.ipynb`, built by `docs/_kit` like the other papers) ran end to end
    locally in quick-test mode on 21 Oct (RTX 5060 Ti, 1 epoch, every job `code=0`, 5.6 min). EasyOCR is not
    installed on that PC, so a stub reader stood in; the real EasyOCR runs on Kaggle.
  - Ultralytics 8.3.100 crashed in validation on NumPy 2.5 (`np.trapz` no longer exists). That is why 8.4.170 is
    pinned; 8.4's `optimizer=auto` would pick MuSGD, so SGD is set explicitly.
  - Real GPU numbers come from the Kaggle run.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/ultralytics/ultralytics
```
```
No official code (none linked). Re-implemented: Ultralytics YOLOv8n/s 8.4.170 + stock EasyOCR 1.7.2 'bn', Kaggle T4x2 notebook on the 498R All-in-One set (human boxes, by-condition OCR, whole-photo test). Paper: mAP 94.0% (p.9), 78 FPS (p.12), char acc 92.7% (p.11).
```

## BibTeX
```bibtex
@misc{ahamed2025yolov8,
  title        = {{YOLOv8}-Based License Plate Recognition for {Bangladeshi} Vehicles},
  author       = {Ahamed, Istiak and Ismail, Gazi Mohammad},
  year         = {2025},
  month        = jan,
  howpublished = {Authorea preprint},
  doi          = {10.22541/au.173748276.69471512/v1},
  note         = {Not peer-reviewed}
}
```
*Authors and date are from the Crossref record of the DOI (posted 21 Jan 2025). Check the key on Google Scholar
before the report. If a journal version appears later, cite that instead.*

## Expected gaps vs. the paper (written before the numbers, 11 Oct; updated for the All-in-One data on 21 Oct)
These are the gaps I expect, and why. I wrote them down in advance so I can't fit the explanation to the numbers
afterwards. After the Kaggle run, mark each one as confirmed or not.

| Metric | Expected vs. the paper | Why |
|---|---|---|
| mAP@50, P, R | **lower**, by a few points or more | Our test photos are fully separate: near-duplicates are kept in one split by the grouped split. The paper augmented before splitting (p.7), so its test set can contain copies of training photos, which inflates its numbers. Our test also mixes 7 sources, night photos and small plates (bus, truck and CNG plates are ~30–80 px wide). |
| mAP@50-95 | **lower** than ≈ 59 | Small plates make tight boxes hard, and the sources draw boxes slightly differently. A few pixels of error cost a lot of IoU. |
| Plates found % vs. 94.8 "detection accuracy" | not comparable | The paper never defines its metric. Ours is recall at conf 0.25, IoU 0.5. |
| FPS | YOLOv8n **faster** than 78, YOLOv8s about the same or faster with FP16 / TensorRT | The paper names neither its GPU nor its model size. Ours is batch 1 on one T4, timing pre-processing + inference + NMS. |
| Training time | no comparison possible | The paper gives none. Ours: two jobs sharing one Kaggle VM (4 CPU cores), so each is slower than it would be alone. |
| EasyOCR char accuracy vs. 92.7 % | **much lower**, and plate exact far lower still | Stock EasyOCR `bn` was not trained on plate fonts; its weakest part is the letters line (confidence 0.52 even in the paper's own example, p.14). The paper's 92.7 % is undefined and was measured on its own plates. |
| By condition | clear > small / night / blurry / fog | Like its Table 2, but with our fixed condition subsets. |
| YOLOv8n vs. YOLOv8s | s slightly higher mAP, n ~2× faster | It's the usual size trade-off. If they tie, our data (not the model size) is the limit. |

Teammates' rows on the same data and split appear in part 2 of [`results_table.md`](../../../results/results_table.md)
after `add_results.py`; for each big gap, write one reason (model size, epochs, image size, hardware, boxes).

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper06_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper06/kaggle/paper06_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper06/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170, easyocr==1.7.2; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 3.5 h for the full run |

| Run | GPU | What |
|---|---|---|
| `det_v8n` | 0 | YOLOv8n (COCO), 120 epochs, 640 px, batch 16, human boxes |
| `det_v8s` | 1 | YOLOv8s (COCO), 120 epochs, 640 px, batch 16, human boxes |
| `easyocr_crops` | 0 (after) | stock EasyOCR bn on the 651 OCR test crops (raw + preprocessed), per condition |
| `e2e_v8n / e2e_v8s` | 1 (after) | whole photos: detector -> best plate -> EasyOCR (204 e2e_test photos) |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper06_outputs.zip` (team Drive `498R / weights / paper06 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #6: ours vs. the paper

| Metric | Paper (page) | det_v8n | det_v8s | easyocr_crops | e2e_v8n | e2e_v8s |
|---|---|---|---|---|---|---|
| Precision (%) | 93.5 (p.9, Table 1) | 96.6 | 96.2 | – | – | – |
| Recall (%) | 92.3 (p.9, Table 1) | 94.1 | 94.7 | – | – | – |
| mAP50 (%) | 94.0 ("mAP"; = mAP@50 in Fig. 6) (p.9, Table 1) | 97.4 | 97.6 | – | – | – |
| mAP50-95 (%) | about 59 (read off Fig. 6) (p.11, Fig. 6) | 45.0 | 45.9 | – | – | – |
| Plates found (%), conf 0.25, IoU 0.5 | 94.8 "detection accuracy" (not defined) (p.9, Table 1) | 96.44 | 97.13 | – | – | – |
| Detector FPS, T4 PyTorch FP32, batch 1 | 78 (GPU and precision not named) (p.12, Table 3) | 113.80 | 109.60 | – | – | – |
| Detector FPS, T4 TensorRT FP16, batch 1 | - | 182.90 | 162.50 | – | – | – |
| Training time (min) | not reported | 130.90 | 144.50 | – | – | – |
| EasyOCR alone (human crops): char accuracy (%) | 92.7 ("character recognition accuracy", not defined) (p.11) | – | – | 60.45 | – | – |
| EasyOCR alone: plate exact (%) | not reported | – | – | 13.52 | – | – |
| EasyOCR alone: CER (%) | not reported | – | – | 39.55 | – | – |
| EasyOCR + preprocessing (4x, grey, CLAHE): char accuracy (%) | "preprocessing", no details (p.8) | – | – | 41.80 | – | – |
| EasyOCR, clear plates: char accuracy (%) | 97.5 (clean plates) (p.12, Table 2) | – | – | 66.84 | – | – |
| EasyOCR, small plates (low_res tag): char accuracy (%) | 92.0 (low resolution) (p.12, Table 2) | – | – | – | – | – |
| EasyOCR, night: char accuracy (%) | 91.5 (low light; metric not named) (p.13, Table 4) | – | – | 49.69 | – | – |
| EasyOCR, blurry: char accuracy (%) | - | – | – | 38.74 | – | – |
| EasyOCR, real fog copies: char accuracy (%) | 89.3 (bad weather) (p.12, Table 2) | – | – | 59.53 | – | – |
| EasyOCR, synthetic rain: char accuracy (%) | 89.3 (rain; metric not named) (p.13, Table 4) | – | – | 44.04 | – | – |
| EasyOCR ms per plate (T4) | not reported | – | – | 90.86 | – | – |
| Whole system (e2e photos): plate found (%) | - | – | – | – | 94.61 | 96.08 |
| Whole system: char accuracy (%) | - | – | – | – | 58.34 | 57.64 |
| Whole system: plate exact (%) | - | – | – | – | 11.76 | 10.29 |
| Whole system: ms per photo (detector + EasyOCR) | - | – | – | – | 68.70 | 69.50 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's 2,600 photos are not public and were augmented BEFORE the split (p.7), so its test set can hold copies of training photos; ours is the grouped All-in-One test split with human boxes. Char accuracy = 100 - CER after normalisation (the paper never defines its metric). Its case groups (clean / low-res / damaged / weather) are matched to our closest condition tags; 'damaged' has no counterpart.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_v8n | 97.4 | 97.7 | 94.3 | 98.6 | 94.8 | 77.5 | 94.8 | 6.25 |
| det_v8s | 97.6 | 98.0 | 93.4 | 98.4 | 85.9 | 80.4 | 93.2 | 6.25 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| easyocr_crops | 13.52 | 17.71 | 1.54 | 1.79 | 15.38 | 5.68 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

**How to read this table (hand note, 24 Oct; kept outside the markers so a re-run of add_results does not remove it).**
Run checked: 7 jobs code=0, dataset v1.1, 120/120 epochs, best epochs 90 (v8n) / 79 (v8s). Full write-up:
[Paper06_Study_Report.pdf](../../../weights/paper06/MD.Rokib.Hasan.Oli/Paper06_Study_Report.pdf) (sources in `report_src/`).
- **mAP50-95 45 vs the paper's ≈59 is a label effect.** The ridoykj source (274 of 1,153 test photos) boxes the plate
  exactly at its edge; the model's boxes are 1.20x wider and 1.36x higher there. ridoykj alone: 22.1; test without
  ridoykj: 64.3 (v8n) / 64.5 (v8s). Paper #2's YOLO12s shows the same (24.2). Local re-test reproduces Kaggle exactly.
- **Same test plates (McNemar):** v8s vs v8n 17 vs 10 plates (p .25, tie); #2 YOLO12s (human + auto boxes) vs v8n
  24 vs 5 (p .0005). Yousuf's #3 YOLOv8n with auto boxes finds 98.12 % vs our 96.44 % (not paired).
- **EasyOCR, same 651 crops:** #2 char-YOLO12n 87.71 % vs EasyOCR 13.52 % (487 vs 4 plates). Letters line right on
  19.2 %, digits on 32.7 %; 39 % of readings contain English digits (৪ read as "8" 55 times). The 4x/CLAHE
  preprocessing loses 69 plates and gains 13. Local EasyOCR re-run: same 60.45 % char accuracy; allowlist 13.98 %
  (p .81) and 2x upscale 13.06 %: no cheap fix. No OCR test crop is under 24 px, so the "small plates" row stays empty.
- **Whole system, 204 photos, same crops, reader swapped (local):** v8n 11.76 → 84.80 %, v8s 10.29 → 88.24 %. In #2's
  cascade, #6 YOLOv8s 88.24 % vs #2's own YOLO12s 86.76 % (7 vs 4, p .55). All photos without a plate box are NightLPBD.
- **Expected gaps (section above):** mAP50/P/R not lower but higher (not confirmed); mAP50-95, FPS, EasyOCR and the
  condition order confirmed; v8n vs v8s only partly (tie on accuracy, n only 1.04x faster in FP32 at batch 1).

## Original-paper check (the paper's own data)
<!-- check:start -->
*Not possible: no code and no data published (its own 2,600 Roboflow photos are not public; Authorea preprint).* The main kit above runs the paper's method on our data instead.
<!-- check:end -->
