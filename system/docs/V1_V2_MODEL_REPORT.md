# BanglaLPR V1 and V2: how they were built and what they score

**Project:** CSE 498R, *Real-Time Bangla License Plate Recognition and Detection using Deep Learning* (NSU, Fall 2026)
**Date of this build:** 31 October 2026 · **Data:** 498R BanglaLPR All-in-One v1.1 · **Hardware for all numbers:** this PC
(NVIDIA RTX 5060 Ti 16 GB, Intel Core Ultra 7 265K, Windows 11, PyTorch 2.13 + CUDA 13, Ultralytics 8.4.170)

> **What is real here.** No model was trained for V1 or V2. Both are built only from weight files that the team
> trained on Kaggle in the paper studies (12-30 October 2026). What is new is how they are combined, and every
> combination was chosen on the **validation** split before the **test** split was scored once. Every number below was
> re-measured on this PC by the scripts in `eval/`. The JSON files in `eval/results/` are the source of every number.
> Nothing was copied from the papers.

---

## 1. Result against the proposal targets

| Proposal target | V1 | V2 | Where measured |
|---|---|---|---|
| Detection ≥ 95 % mAP@50 | **98.96 %** | **98.96 %** (same detector, plus a retry chain) | detection test, 1,153 photos (`detectors.json`) |
| Character recognition ≥ 90 % | **97.97 %** crops · **94.74 %** whole photos | **98.71 %** crops · **96.79 %** whole photos | 651 test crops; 204 whole test photos (`system_gpu.json`) |
| Real time ≥ 10 FPS | **48.8 FPS** GPU · **17.7 FPS** CPU only | **41.7 FPS** GPU · **20.0 FPS** CPU only | whole photo in → plates out, batch 1, mean over 204 photos |
| *(stricter)* Whole plate exactly right | 90.02 % crops · 89.22 % photos | **91.71 %** crops · **92.65 %** photos | same sets |

- All targets are met **on this PC**. The CPU figure is a desktop CPU (Core Ultra 7 265K, 8 threads), not an edge
  board. Speed on a Raspberry Pi 5 or a Jetson has **not** been measured (see `IMPROVEMENT_GUIDE.md`, step 6).
- "Character recognition accuracy" is 100 − CER (Unicode characters, after normalising spaces and hyphens), the same
  definition as the team's kit. Whole-plate exact is the stricter number and is reported next to it everywhere.
- V2 vs V1 on the same plates:
  - **Test crops:** V2 fixes 12 plates and breaks 1. Exact McNemar p = 0.003, so the gain is significant.
  - **Whole photos:** 9 fixed and 2 broken, p = 0.065, so the gain is **not** significant on only 204 photos. All of
    it comes from the 60 night photos (71.67 % → 83.33 %); the 144 day photos are unchanged (96.53 %).

---

## 2. What each paper contributes

Papers are named as in the team's notes. "Used" means the component runs in the system; "tested, not used" means it
was measured on validation data and lost (the number shows why).

### V1: papers 3, 20, 18, 11, 2, 8, 10

| # | Paper (short name) | Used in V1 | Tested, not used (why) |
|---|---|---|---|
| 18 | Embedded LPR (YOLOv8n-576 + slot OCR) | **Plate detector** `p18_v8n_576` (val mAP50 96.92, the best of 6 V1 detectors; 3.0 M parameters). **Slot OCR** as the second fallback reader | Dynamic INT8 OCR (−20 points in its Kaggle run) |
| 11 | Video LPR (low-resource) | **Primary reader**: char-YOLOv8n (86.18 % alone). **Video tracking + vote per vehicle** (`lpr/video.py`) | Haar gate (2.47 % recall in its run); 320 px detector (val mAP50 94.80) |
| 2 | Cascaded YOLO (BLPD) | **First fallback reader**: LPNR char-YOLO12n at 640 px (87.71 % alone). **BRTA layout decoder** (district, মেট্রো, letter / 6 digits by token kind). **8 % crop padding** | YOLO12s plate detector (val 96.80 < 96.92); vehicle stage (no gain in its study) |
| 20 | Plate super-resolution | **Status reader**: its CRNN reads every plate as an independent second opinion (75.73 % alone) | The SR network: on val plates shrunk to 24 / 32 px, SR 12.9 / 28.9 % vs plain bicubic ×2 19.6 / 34.3 %, so V1 upscales tiny crops with bicubic |
| 8 | CNN + GFP-GAN | **Character verifier**: its 28×28 CNN re-classifies every character box | GFP-GAN (205 ms per plate, no exact gain); contour segmentation (finds 52 % of symbols) |
| 10 | BLPnet | **Second character verifier**: its 32×32 CNN (rebuilt from the team's `paper10.ipynb`) | Its YOLOv8n detector (val 96.73 < 96.92); Inception box regressor (11.3 % IoU > 0.5) |
| 3 | Cloud hybrid (YOLO tiny + Haar) | **Architecture**: plates read at the edge, only small results go to a server (the web app's design, event messages of ~170 bytes) | Haar cascade (F1 4.43 % in our run); its YOLOv8n 416 / 640 detectors (val 96.66 / 96.22) |

The two character CNNs give an agreement score; their mean is the best error signal (val AUROC 0.765). Plates with less
than 80 % agreement are only 33 % right, so they go to **review** (`char_verifier_val.json`).

### V2: papers 27, 28, 21, 35, 23, 6, 19, 34 (V2 keeps V1 as its base)

| # | Paper (short name) | Used in V2 | Tested, not used (why) |
|---|---|---|---|
| 27 | YOLOv10 | **Always-on second reader**: the char-YOLOv8n trained in that run (86.02 % alone) | YOLOv10n / v10s plate detectors: val mAP50 96.87 / 97.28, but they **find fewer plates** on val (94.94 / 95.41 % vs 96.35 %); char-YOLOv10n reader (83.72 %) |
| 34 | SVTRv2 | **Two helper readers** SVTRv2-S and -T, exported to ONNX (81.87 / 81.26 % alone, but they make different mistakes) | — |
| 6 | YOLOv8 + EasyOCR | **Retry detector**: YOLOv8s runs only when V1's detector finds nothing | EasyOCR (13.52 % exact) |
| 35 | HVI-CIDNet (low light) | **Retry on a CLAHE-enhanced copy**, only on a miss. Its own study showed that enhancing every photo adds false boxes and does not help reading | HVI-CIDNet network itself (no gain on night plates in its run) |
| 19 | LPLCv2 (legibility) | **Status from reader agreement**: its study found reader confidence/agreement beats a legibility classifier as a gate | Legibility classifier (gate 81.0 % vs 84.45 % for confidence at 80 % coverage) |
| 23 | Near-duplicates | **Evaluation protocol**: grouped split with no plate in two splits, choose on val, test once. This is why the numbers above can be trusted | KUET-only "fair" reader (62.37 % on the full test set) |
| 28 | YOLOv9 | — (evaluated) | YOLOv9s: val mAP50 97.11 but val plates found 95.41 %; as a retry detector 96.71 % vs 97.18 % for the chosen chain |
| 21 | Advanced DL ALPR | — (evaluated) | Its YOLOv9s / v10s / v8s / v5su detectors give the same scores as the #28 / #27 runs (deterministic kit); Tesseract (0.77 %) |

**Plainly:** papers 28 and 21 do not contribute a component that runs in V2. Their detectors were measured and lost on
validation. Putting one in anyway would have meant choosing on the test set, which paper 23 warns against. If the team
wants YOLOv9s in the system anyway, change `retry[0].file` in `models/v2/config.json` to `p28_v9s.pt` (val plates
found 96.71 % instead of 97.18 %).

---

## 3. The two pipelines

```
V1  photo ──► YOLOv8n-576 (#18) ──► crop +8 % (#2) ──► [h < 32 px: bicubic ×2 (#20 study)]
              ──► char-YOLOv8n (#11) ── BRTA layout OK? ── yes ──► text
                                         └─ no ──► + LPNR char-YOLO12n (#2) + slot OCR (#18) ──► vote (layout-valid first)
              status: CRNN (#20) agrees → verified · else char-CNN agreement (#8, #10) ≥ 80 % → likely · else review

V2  photo ──► YOLOv8n-576 (#18) ── nothing found? ──► YOLOv8s (#6) ──► nothing? ──► YOLOv8s on CLAHE copy (#35)
              ──► crop +8 % ──► char-YOLOv8n (#11)  +  char-YOLOv8n (#27)   (both always run)
                                ── same layout-valid plate? ── yes ──► text  (86 % of test plates stop here)
                                   └─ no ──► + LPNR (#2) + SVTRv2-S + SVTRv2-T (#34) ──► vote (layout-valid first)
              status: ≥ 2 readers agree → verified · otherwise → review (#19 study)
```

Fusion rules (`lpr/fusion.py`): **fallback** = helpers only when the primary reading breaks the BRTA layout;
**verify** = primary + second reader always, helpers only when they disagree; the vote is MV-BM+format (majority of
layout-valid readings, tie → primary). In the team's earlier studies no reader was ever right with an invalid layout.

---

## 4. How the choices were made (no peeking at test)

| Choice | Search | Picked on | Rule |
|---|---|---|---|
| Readers + rule | V1: 4 readers; V2: V1's 2 char-YOLO readers + 6 V2 readers; every ordered subset up to 5 readers × fallback / vote / verify (thousands of combinations) | OCR **val** 766 crops | best val plate exact; tie → fewer ms |
| Detector | 14 detectors alone, ±CLAHE retry, V1 detector → each V2 detector on a miss (30 strategies) | detection **val** 896 photos | most val plates found (IoU ≥ 0.5, conf 0.25); tie → fewer extra boxes |
| Tiny-plate upscaling | none / bicubic ×2 / #20 SR ×2 on val crops shrunk to 24 and 32 px | OCR **val** | best val plate exact |
| Character verifier | #8 CNN, #10 CNN, their mean | OCR **val** | best AUROC for "this reading is wrong" |

Scripts: `eval/reader_preds.py` → `eval/select_fusion.py` → `eval/detectors.py` + `eval/detector_combo.py` →
`eval/system_eval.py` (GPU and `--cpu`). Many combinations were searched, so some of the val winner's lead is luck. On
test, the 8 best V2 combinations by val score 91.71-91.86 % and the 8 best V1 combinations 89.86-90.32 %
(`fusion_selection.json`), so the chosen rows are typical of their neighbours, not lucky outliers.

---

## 5. Detailed results

### 5.1 Every candidate reader (OCR crops, batch 1 on the RTX 5060 Ti)

| Reader | Paper | Val exact | Test exact | Test char acc | ms / plate |
|---|---|---|---|---|---|
| LPNR char-YOLO12n (640 px) | 2 | 80.55 | **87.71** | 97.86 | 8.9 |
| char-YOLOv8n | 11 | 80.81 | 86.18 | 97.29 | 5.4 |
| char-YOLOv8n | 27 | 80.29 | 86.02 | 97.66 | 5.5 |
| char-YOLOv10n | 27 | 77.55 | 83.72 | 97.13 | 5.8 |
| SVTRv2-S (ONNX, CPU) | 34 | 75.33 | 81.87 | 96.92 | 14.0 |
| SVTRv2-T (ONNX, CPU) | 34 | 73.76 | 81.26 | 97.51 | 8.2 |
| CRNN-hr | 20 | 65.01 | 75.73 | 94.46 | 4.5 |
| CRNN (deployment reader of the LPLCv2 study) | 19 | 62.27 | 73.43 | 94.48 | 4.4 |
| KUET-only "fair" char-YOLOv8n | 23 | 41.12 | 62.37 | 86.19 | 5.2 |
| Slot OCR | 18 | 47.65 | 57.60 | 88.58 | 1.0 |

Each local test score equals the Kaggle run of that paper (for example 87.71, 86.18, 81.26, 73.43, 57.60), so the
weights load and run correctly here. The #20 CRNN scores 75.73 % on the v1.1 test (its v1.0 Kaggle run reported
74.04 % on the v1.0 test, which had 19 whole-photo BLPD rows).

### 5.2 Every candidate detector (test = 1,153 photos)

| Detector | Paper | px | Val mAP50 | Test mAP50 | Test mAP50-95 | Test recall | Params |
|---|---|---|---|---|---|---|---|
| **YOLOv8n-576** | 18 | 576 | **96.92** | **98.96** | 46.08 | 96.4 | 3.0 M |
| YOLO12s | 2 | 640 | 96.80 | 98.87 | 46.18 | 96.2 | 9.3 M |
| YOLOv8n | 10 | 640 | 96.73 | 98.74 | 45.71 | 96.4 | 3.1 M |
| YOLOv8n | 3 | 416 | 96.66 | 98.80 | 45.76 | 96.2 | 3.0 M |
| YOLOv8n | 3 | 640 | 96.22 | 98.73 | 45.22 | 96.7 | 3.0 M |
| YOLOv8n | 11 | 320 | 94.80 | 97.51 | 44.23 | 96.4 | 3.0 M |
| YOLOv8n (human boxes) | 6 | 640 | 97.70 | 97.39 | 45.00 | 94.2 | 3.0 M |
| YOLOv8s (human boxes) | 6 | 640 | 97.61 | 97.62 | 45.88 | 94.8 | 11.1 M |
| YOLOv10s | 27 | 640 | 97.28 | 98.25 | 45.40 | 95.4 | 8.1 M |
| YOLOv9s | 28 / 21 | 640 | 97.11 | 98.79 | 46.35 | 97.2 | 7.3 M |
| YOLOv8s | 35 | 640 | 97.07 | 98.50 | 45.83 | 96.9 | 11.1 M |
| YOLOv10n | 27 | 640 | 96.87 | 98.66 | 45.10 | 96.4 | 2.7 M |
| YOLOv9t | 28 | 640 | 96.21 | 98.30 | 45.23 | 95.7 | 2.0 M |

mAP50-95 is low for every model because one source (ridoykj) labels boxes tight to the plate edge (found in the
Cascaded YOLO and YOLOv8 studies). V2 kept V1's detector because mAP50 on val did not translate into more plates found:
the human-box #6 models had the highest val mAP50 but found fewer val plates alone than the #18 model.

### 5.3 Where the systems still fail (651 test crops)

| Subset | V1 exact | V2 exact |
|---|---|---|
| clear (480) | 93.75 | 95.00 |
| night (65) | 80.00 | 84.62 |
| blurry (112) | 80.36 | 82.14 |

Remaining V2 errors (54 plates): a wrong digit in a valid layout 25, a wrong series letter 13, a wrong district 8,
invalid layout 3, mixed 5. The oracle (any of V2's five readers right) is 93.7 %, so better voting can add at most
about 2 points. Bigger gains need better readers or more frames (see `IMPROVEMENT_GUIDE.md`).

### 5.4 Status levels (what the operator sees)

| Version | verified | likely | review |
|---|---|---|---|
| V1 | 78.0 % of plates, **96.06 %** exact | 15.8 %, 82.52 % exact | 6.1 %, 32.50 % exact |
| V2 | 94.9 % of plates, **95.31 %** exact | — | 5.1 %, 24.24 % exact |

V2 accepts 95 % of plates automatically and sends 5 % to a person. V1 needs a person for more plates.

### 5.5 Speed

| | V1 GPU | V2 GPU | V1 CPU | V2 CPU |
|---|---|---|---|---|
| ms per whole photo (mean / median) | 20.5 / 18.7 | 24.0 / 17.4 | 56.5 / 51.7 | 49.9 / 37.4 |
| FPS (mean) | 48.8 | 41.7 | 17.7 | 20.0 |
| ms per plate, reader part only | 14.5 | 15.6 | – | – |

V2 is faster than V1 on the CPU because V1 runs its CRNN status reader and two character CNNs on every plate. In the
video mode the reader runs only until 3 readings of a vehicle agree. On the synthetic demo clip (8 vehicles, 400
frames) V2 needed 24 reads and processed 68.9 frames per second.

---

## 6. Limitations

- **No new training.** If the team retrains any reader or detector, re-run the scripts in section 4 so that the
  choices stay validation-based.
- **Small whole-photo test.** There are 204 photos: 144 BLPD external day photos and 60 NightLPBD night photos, one plate
  each. The BLPD photos come from the same streets as public LPDB-A images in training (no same plate, by the grouped
  split), so 96.5 % on them is likely optimistic for new cities. The night set is small (60).
- **No real traffic video** is in the dataset. The video mode was checked on a *synthetic* clip made from still test
  photos (`tools/make_demo_video.py`), which proves the mechanics, not the accuracy on real video.
- **Edge speed not measured.** The FPS figures are from a desktop GPU and a desktop CPU.
- **Two-line rear plates only.** BRTA's new single-line motorcycle front plates (mandatory from 15 Nov 2026) are not in
  the data.
- **SVTRv2 runs on the CPU through ONNX Runtime**, because the installed onnxruntime has no CUDA provider. That is
  fine because it runs on about 14 % of plates.
