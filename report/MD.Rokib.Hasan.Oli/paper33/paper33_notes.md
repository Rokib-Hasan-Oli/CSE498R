# Paper #33: YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection

*Reader: MD. Rokib Hasan Oli · new5 (added 25 Oct 2026) · Board group: Model backbones / OCR methods*
*PDF: `Papers/6_6 Added papers (new5)/33_2025_YOLO26 Key Architectural Enhancements and Benchmarking.pdf`
(arXiv 2509.25164 v5, 16 Mar 2026, 15 pages; review / preprint). Page numbers below are the PDF pages.*

---

## YOLO26: Key Architectural Enhancements and Performance Benchmarking (2025)
```
Problem:   YOLOv8 to v13 kept adding accuracy at the price of deployment friction: NMS post-processing (latency,
           thresholds) and Distribution Focal Loss (DFL, export trouble) (p.4, p.6-7). Edge devices need simple, fast,
           quantisable detectors.
Method:    A review of Ultralytics YOLO26 (released Sept 2025): (1) DFL removed (plain box regression, reg_max 1);
           (2) end-to-end NMS-free one-to-one head next to the classic one-to-many head; (3) ProgLoss (progressive
           loss balancing) and STAL (small-target-aware label assignment); (4) MuSGD, a hybrid of SGD and the Muon
           optimiser (p.6-7, Fig. 4). Multi-task: detect, segment, pose, OBB, classify (p.5).
Data:      MS COCO (detection, segmentation, pose), ImageNet (classification), DOTA v1 (OBB) — Ultralytics' own runs.
Results:   COCO val, 640 px (p.8, Table 2): 26n 40.9 mAP50-95 (40.1 NMS-free), 38.9 ms CPU ONNX, 1.7 ms T4 TensorRT10,
           2.4 M params, 5.4 GFLOPs; 26s 48.6 (47.8), 87.2 ms, 2.5 ms, 9.5 M; 26m 53.1 (52.5), 4.7 ms; 26l 55.0;
           26x 57.5 (56.9), 11.8 ms. INT8 / FP16 "minimal impact on accuracy" (p.10, no number).
Limits:    No experiment of its own: Table 2 and Fig. 1 are Ultralytics' published numbers; the abstract promises
           Jetson Nano / Orin benchmarks against YOLOv8-v13 and transformers, but the paper contains no such table;
           claims about INT8 robustness and small objects come without numbers; no ablation of DFL / STAL / MuSGD.
For us:    The default model family of our pinned Ultralytics (8.4.170): we must know if it beats YOLO11 / v8 on
           plates, what NMS-free buys in ms, and whether it survives INT8 for the edge demo (#18).
Code:      https://github.com/ultralytics/ultralytics (AGPL-3.0), YOLO26 in ultralytics==8.4.170
BibTeX:    sapkota2025yolo26  (full entry at the bottom; confirm on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** make YOLO simpler and faster to deploy on edge hardware without losing accuracy.
2. **How:** drop DFL, add an NMS-free head, new loss balancing and label assignment for small objects, MuSGD.
3. **How well:** YOLO26-N 40.9 % COCO mAP at 1.7 ms (T4 TensorRT) and 38.9 ms on a CPU (p.8, Table 2).

**Useful for us?** Yes, but read it as documentation, not evidence. It is a review written around Ultralytics'
numbers. The model itself matters to us: it is what `optimizer=auto` and the newest weights in our kit pull in, and
nobody has measured it on Bangla plates. Our kit supplies the missing experiments (both heads, MuSGD, P2, INT8).

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | arXiv review (Cornell, Kansas State), v5 March 2026 | p.1 |
| Lineage | YOLOv1 to v13 + YOLO26: innovations, tasks, frameworks | p.3, Table 1 |
| DFL removal | box regression without the distribution head: faster export, fewer ops | p.6-7, Fig. 4a |
| NMS-free | one-to-one head gives final boxes directly; no IoU / score thresholds after the network | p.7, Fig. 4b |
| ProgLoss + STAL | progressive balancing of loss terms; label assignment that favours small targets | p.7, Fig. 4c |
| MuSGD | SGD + Muon (from LLM training) for stable, fast convergence | p.7, Fig. 4d |
| Deployment | ONNX, TensorRT, CoreML, TFLite, OpenVINO; FP16 / INT8 | p.10 |

**Every result number** (Ultralytics, COCO val2017, 640 px, p.8 Table 2)
| Model | mAP50-95 | mAP50-95 (e2e) | CPU ONNX ms | T4 TRT10 ms | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLO26n | 40.9 | 40.1 | 38.9 | 1.7 | 2.4 | 5.4 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 | 20.7 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 | 68.2 |
| YOLO26l | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 | 86.4 |
| YOLO26x | 57.5 | 56.9 | 525.8 | 11.8 | 55.7 | 193.9 |
| Classification / pose / OBB tables | ImageNet top-1 71.4-79.9; COCO pose 57.2-71.6; DOTA 52.4-56.7 | | | | | p.9, Tables 3-5 |
| INT8, Jetson, small objects | claimed, no numbers | | | | | p.1, p.7, p.10 |

Terms I looked up:
- **One-to-one vs one-to-many head:** the one-to-many head predicts several boxes per object (NMS keeps one); the
  one-to-one head is trained to predict exactly one, so no NMS is needed (the YOLOv10 idea, paper #27).
- **DFL (Distribution Focal Loss):** box edges predicted as a probability distribution over bins (reg_max 16); YOLO26
  uses reg_max 1, i.e. direct regression.
- **Muon:** an optimiser that orthogonalises the update of weight matrices; MuSGD mixes it with SGD.

**Weak points I found:**
- **No own experiment.** Every number is copied from Ultralytics; there are no ablations of DFL / STAL / ProgLoss /
  MuSGD, so the paper cannot say which change gives what.
- **Missing benchmarks.** The abstract promises Jetson Nano / Orin comparisons with YOLOv8-v13 and transformers (p.1);
  the paper has none.
- **What "default" means in practice:** in Ultralytics 8.4.170 a YOLO26 model runs its one-to-many head with NMS
  unless `nms=False` is passed (we checked: `nms=None` → one-to-many). So "NMS-free" is opt-in there.

## Pass 3: rebuild it
```
image 640 ─► backbone + neck (C3k2 / attention blocks) ─► Detect head, reg_max 1 (no DFL)
            ├─ one-to-many branch ─► many boxes ─► NMS ─► plates        (nms=None, Ultralytics 8.4.170 default)
            └─ one-to-one branch  ─► top-k boxes ─► plates, no NMS      (nms=False, the paper's "e2e")
training: one-to-many + one-to-one losses, ProgLoss balancing, STAL assignment, MuSGD (lr 0.01)
```
**Assumptions:** the one-to-one head is nearly as accurate as one-to-many + NMS (-0.8 mAP on COCO); STAL helps small
objects; quantisation does not hurt.

**What would fail on our Bangladeshi plates:**
- **Two plates close together** (front of a bus, two-row parking) are where one-to-one heads can drop a box.
- **Tiny plates:** STAL's gain is unmeasured; the P2 head (stride 4) may matter more for far plates.
- **Small data:** MuSGD was tuned for COCO-scale training; on ~8.5 k photos for 60 epochs SGD may do as well.

**How I reproduce it:**
- **Same split and settings as all detector kits** (640 px, 60 epochs, batch 16, seed 42): YOLO26-N and -S with
  MuSGD (lr 0.01), YOLO11-N with SGD, YOLO26-N with SGD, YOLO26-N with the P2 head (the COCO backbone is
  transferred, 360 of 902 tensors; the P2 neck and head start new).
- **Both heads of each YOLO26 model:** test mAP, plates found, false alarms, PyTorch FP16, TensorRT FP16 and INT8
  (with test mAP), CPU ONNX; YOLO11-N gets the same speed and INT8 tests.
- **Original check:** the released YOLO26-N/S/M and YOLO11-N on COCO val2017, both heads, T4 TensorRT.
- See the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper33/Kaggle_Training_Guide.md) and the [original-check guide](../../../src/MD.Rokib.Hasan.Oli/paper33/Original_Check_Guide.md).

## Check yourself (2-minute version)
> YOLO26 simplifies YOLO for edge devices: no DFL, a second NMS-free head, small-object label assignment and the
> MuSGD optimiser. Ultralytics' numbers give YOLO26-N 40.9 % COCO mAP (40.1 without NMS) at 1.7 ms on a T4. This paper
> is a review that repeats those numbers without new experiments and without the Jetson results it promises. For us:
> measure both heads, MuSGD, the P2 head and INT8 on our plates before choosing it as the demo detector.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26: Key Architectural Enhancements and Performance Benchmarking (Sapkota et al.) | 2025 | arXiv review | MS COCO (Ultralytics' runs) | YOLO26 n-x (NMS-free head, no DFL, STAL, MuSGD) | – | 26n 40.9 mAP (40.1 e2e), 1.7 ms T4 TRT, 38.9 ms CPU (p.8) | Yes (Ultralytics, AGPL-3.0) | No own experiments; promised Jetson results missing | Newest detector in our pinned Ultralytics; NMS-free + INT8 for the edge demo |

## Code check → Research board
- **Code:** https://github.com/ultralytics/ultralytics (AGPL-3.0). YOLO26 configs `cfg/models/26/yolo26.yaml`
  (`end2end: True`, `reg_max: 1`) and `yolo26-p2.yaml`; weights `yolo26{n,s,m,l,x}.pt` from the v8.4.0 assets. The kit
  pins `ultralytics==8.4.170` like every kit.
- **Head switch (8.4.170):** `model.val(..., nms=False)` / `predict(..., nms=False)` = one-to-one NMS-free head;
  default (`nms` unset) = one-to-many head + NMS; `export(..., nms=False)` builds an end-to-end engine.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/ultralytics/ultralytics
```
```
Review of Ultralytics YOLO26 (no own tests). My T4x2 kit: YOLO26-N/S (MuSGD) vs YOLO11-N, SGD and P2 ablations, NMS-free vs NMS head of one model (mAP, ms, TensorRT INT8). Paper/Ultralytics: 26n 40.9 COCO mAP, 1.7 ms T4 (p.8).
```

## BibTeX
```bibtex
@article{sapkota2025yolo26,
  title   = {{YOLO26}: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection},
  author  = {Sapkota, Ranjan and Cheppally, Rahul Harsha and Sharda, Ajay and Karkee, Manoj},
  journal = {arXiv preprint arXiv:2509.25164},
  year    = {2025}
}
```
*Cite the Ultralytics software too (Jocher et al., Ultralytics YOLO, GitHub) when the report uses the model.*

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper33_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper33/kaggle/paper33_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper33/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 8 h for the full run |

| Run | GPU | What |
|---|---|---|
| `y26n` | 0 | YOLO26-N, MuSGD lr 0.01, 60 epochs |
| `y26s` | 0 (after) | YOLO26-S, MuSGD lr 0.01, 60 epochs |
| `y11n` | 1 | YOLO11-N, SGD, 60 epochs (team baseline) |
| `y26n_sgd` | 1 (after) | YOLO26-N, SGD (optimiser ablation) |
| `y26n_p2` | 1 (after) | YOLO26-N + P2 head, MuSGD; COCO yolo26n backbone transferred (the P2 neck / head start new) |
| `heads_y26n / heads_y26s / heads_y26n_p2` | 0 (after) | NMS-free vs NMS head: mAP, plates, FP16 / TensorRT FP16 + INT8 / CPU ms |
| `speed_y11n` | 0 (after the heads) | YOLO11-N: FP32 / FP16 / TensorRT FP16 + INT8 / CPU ONNX |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper33_outputs.zip` (team Drive `498R / weights / paper33 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #33: ours vs. the paper

| Metric | Paper (page) | y26n | y26s | y11n | y26n_sgd | y26n_p2 | heads_y26n | heads_y26s | heads_y26n_p2 |
|---|---|---|---|---|---|---|---|---|---|
| mAP50 (%) | - | 98.7 | 99.1 | 97.8 | 98.5 | 98.0 | – | – | – |
| mAP50-95 (%) | COCO val: 26n 40.9, 26s 48.6 (p.8, Table 2) | 45.4 | 46.4 | 44.7 | 45.8 | 46.2 | – | – | – |
| mAP50-95, NMS-free one-to-one head (%) | COCO e2e: 26n 40.1, 26s 47.8 (p.8, Table 2) | – | – | – | – | – | 45.0 | 47.1 | 45.4 |
| mAP50-95, one-to-many head + NMS (%) | COCO: 26n 40.9, 26s 48.6 (p.8, Table 2) | – | – | – | – | – | 45.4 | 46.4 | 46.2 |
| NMS-free minus NMS head, mAP50-95 (pp) | -0.8 (26n and 26s, COCO) (p.8, Table 2) | – | – | – | – | – | -0.40 | 0.68 | -0.80 |
| Plates found at conf 0.25 (%) | - | 97.82 | 97.73 | 97.73 | 97.63 | 98.42 | – | – | – |
| Plates found, NMS-free head (%) | - | – | – | – | – | – | 96.83 | 97.53 | 98.12 |
| Small plates: recall (%) | STAL "boosts small objects", no number (p.7) | 40.00 | 20.00 | 60.00 | 60.00 | 80.00 | – | – | – |
| Small plates, NMS-free head: recall (%) | - | – | – | – | – | – | 40.00 | 20.00 | 80.00 |
| False alarms on plate-free photos, NMS-free head (%) | - | – | – | – | – | – | 5.00 | 5.00 | 3.75 |
| T4 PyTorch FP16 ms, median (YOLO26: NMS-free head; YOLO11: with NMS) | - | – | – | 12.06 (fp16) | – | – | 14.26 | 12.86 | 14.83 |
| T4 PyTorch FP16 ms, one-to-many + NMS (median) | - | – | – | – | – | – | 17.65 | 13.23 | 15.27 |
| T4 TensorRT FP16 ms (YOLO26: NMS-free; YOLO11: with NMS) | 26n 1.7, 26s 2.5 (TensorRT 10, model only) (p.8, Table 2) | – | – | 6.28 (tensorrt_fp16) | – | – | 5.55 | 5.78 | 5.85 |
| TensorRT INT8: test mAP50 (%) (YOLO26: NMS-free; YOLO11: with NMS) | "minimal impact on accuracy" (no number) (p.10) | – | – | 96.9 (tensorrt_int8) | – | – | 97.4 | 97.8 | 97.4 |
| TensorRT INT8 ms | - | – | – | 7.77 (tensorrt_int8) | – | – | 5.62 | 5.76 | 5.80 |
| CPU ONNX ms (YOLO26: NMS-free; YOLO11: with NMS) | 26n 38.9, 26s 87.2 (p.8, Table 2) | – | – | 90.98 (cpu_onnx) | – | – | 43.83 | 117.43 | 98.36 |
| Params (M) | 26n 2.4, 26s 9.5 (p.8, Table 2) | 2.50 | 9.95 | 2.59 | 2.50 | 2.52 | – | – | – |
| GFLOPs | 26n 5.4, 26s 20.7 (p.8, Table 2) | 5.90 | 22.70 | 6.50 | 5.90 | 7.60 | – | – | – |
| Best epoch | - | 50 | 49 | 55 | 55 | 55 | – | – | – |
| Training time (min) | not reported | 141.00 | 175.10 | 113.30 | 127.80 | 134.40 | – | – | – |

A value marked with (...) comes from `speed.fp16.ms_median`: that run has no number at `heads.one2one_e2e.pytorch_fp16.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_fp16.ms_median`: that run has no number at `heads.one2one_e2e.tensorrt_fp16.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_int8.test_mAP50`: that run has no number at `heads.one2one_e2e.tensorrt_int8.test_mAP50` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_int8.ms_median`: that run has no number at `heads.one2one_e2e.tensorrt_int8.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.cpu_onnx.ms_median`: that run has no number at `heads.one2one_e2e.cpu_onnx.ms_median` (e.g. the TensorRT export failed or was not run).

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's Table 2 is Ultralytics' own COCO table (the paper runs no experiment of its own); its speed column is model-only TensorRT, ours is pre + inference + post at batch 1. In Ultralytics 8.4.170 a YOLO26 model runs its one-to-many head with NMS unless `nms=False` is given: the training runs (y26n, y26s, ...) are scored that way, the heads_* runs score both heads of the same weights.

*26 Oct 2026: the TensorRT INT8 mAP50 row was printed as a fraction by the notebook (0.97 = 97 %); it is shown here in % (report.py fix). Every other line is the run's own output; all results.json files are unchanged.*

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| y11n | 97.8 | 97.9 | 96.0 | 98.5 | 93.9 | 71.6 | 95.9 | 5.62 |
| y26n | 98.7 | 99.0 | 93.0 | 95.7 | 92.4 | 88.7 | 97.7 | 5.62 |
| y26n_p2 | 98.0 | 98.0 | 94.7 | 96.4 | 93.5 | 87.6 | 96.6 | 5.62 |
| y26n_sgd | 98.5 | 98.7 | 94.0 | 97.8 | 97.3 | 72.5 | 97.5 | 4.38 |
| y26s | 99.1 | 99.3 | 94.1 | 97.9 | 92.1 | 86.5 | 97.9 | 5.00 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

**How to read this table (hand note, 26 Oct; kept outside the markers so a re-run of add_results does not remove it).**
Run checked: 9 jobs code=0, dataset v1.1, no NaN; local re-scoring (RTX 5060 Ti) gives the same plates found for all
8 model/head pairs. Full write-up: [Paper33_Study_Report.pdf](../../../weights/paper33/MD.Rokib.Hasan.Oli/Paper33_Study_Report.pdf) (sources in [`source/`](source/)).
- **`y26n` (MuSGD) stopped at epoch 50 of 60** (2.3 h time cap; MuSGD steps 13.6 vs 1.0 ms + a shared CPU): base lr
  0.0019 instead of 0.00027, no mosaic-free last 10 epochs, ProgLoss one-to-many weight 0.21 instead of 0.10. Its row
  is a lower bound. Kit fixed 26 Oct (caps 3.2 / 3.6 / 2.5 / 2.7 / 3.0 h, speed job after the heads on one GPU, INT8 in %).
- **Clean test = tie:** 97.6-98.4 % found; YOLO26-N vs YOLO11-N 10 vs 9 plates (p 1.0), P2 vs N 13 vs 7 (p .26); 5
  plates missed by all. mAP50-95 without ridoykj 64.6-65.0 for all five (YOLO11-N highest): the mAP50-95 ranking above
  is the ridoykj box style.
- **NMS-free head:** net 2-3 plates lost on fully trained models (n.s.), 11 vs 1 on the stopped `y26n` (p .006); 13-24
  double boxes per 1,153 photos (NMS 0-1); CPU post-processing 7.5 ms (N) / 80.4 ms (P2) → 0.27 ms.
- **Rain (synthetic):** MuSGD models 78.6-82.4 % found, SGD models 60.6-63.0 %; `y26n` vs `y26n_sgd` 232 vs 16 plates
  (p < 1e-40). Architecture alone: rain n.s. (85 vs 109), fog +2.4 points (34 vs 10, p .0004). One seed each.
- **Speed (row above mixes jobs):** `heads_y26n` and `speed_y11n` ran at the same time, so their T4 / CPU ms are upper
  bounds. Clean local CPU (4 threads): network 15.9 (YOLO26-N NMS-free) vs 18.5 ms (YOLO11-N); + post 16.2 vs 26.2 ms.
  INT8 keeps accuracy (NMS-free -0.04 to -0.52 mAP50 vs TRT FP16) but is not faster at batch 1.
- **For us:** detection is saturated; use YOLO26-N(-P2) NMS-free (+ a box merge) for the CPU / edge pipeline, keep
  YOLO11-N as the baseline row, and test MuSGD vs SGD with 3 seeds + real rain photos.

## Original-paper check (the paper's own data)
<!-- check:start -->
### Paper #33 on its own data: ours vs. the paper

| Metric | Paper (page) | coco_gpu0 | coco_gpu1 |
|---|---|---|---|
| YOLO26-N: COCO AP50-95, NMS head (%) | 40.9 (p.8, Table 2) | 40.81 |  |
| YOLO26-N: COCO AP50-95, NMS-free head (%) | 40.1 (e2e) (p.8, Table 2) | 40.06 |  |
| YOLO26-N: T4 TensorRT FP16, NMS-free, inference ms | 1.7 (p.8, Table 2) | 2.52 |  |
| YOLO26-N: CPU ONNX ms (end to end) | 38.9 (p.8, Table 2) | 96.64 |  |
| YOLO26-S: COCO AP50-95, NMS head (%) | 48.6 (p.8, Table 2) | 48.62 |  |
| YOLO26-S: COCO AP50-95, NMS-free head (%) | 47.8 (e2e) (p.8, Table 2) | 47.78 |  |
| YOLO26-S: T4 TensorRT FP16, NMS-free, inference ms | 2.5 (p.8, Table 2) | 2.64 |  |
| YOLO26-S: CPU ONNX ms (end to end) | 87.2 (p.8, Table 2) | 134.05 |  |
| YOLO26-M: COCO AP50-95, NMS head (%) | 53.1 (p.8, Table 2) |  | 53.09 |
| YOLO26-M: COCO AP50-95, NMS-free head (%) | 52.5 (e2e) (p.8, Table 2) |  | 52.50 |
| YOLO26-M: T4 TensorRT FP16, NMS-free, inference ms | 4.7 (p.8, Table 2) |  | 4.92 |
| YOLO11-N (baseline): COCO AP50-95 (%) | 39.5 (Ultralytics docs; not in the paper) |  | 39.45 |
| YOLO11-N: T4 TensorRT FP16, inference ms | 1.5 (Ultralytics docs) |  | 2.19 |

Same data and split as the paper (its own published data / weights), so this IS a direct check of
the paper's numbers. Remaining differences come from software versions, hardware and randomness.

The paper copies these numbers from Ultralytics; this check is therefore a check of the released weights. TensorRT 'inference ms' is the model only (the paper's column); `end_to_end_ms` in results.json adds pre- and post-processing (NMS for the one-to-many head).

<!-- check:end -->

**Hand note (26 Oct).** Accuracy reproduced: all 7 AP numbers within 0.1 of the paper; NMS-free penalty -0.75 / -0.84 /
-0.59 (paper -0.8 / -0.8 / -0.6). T4: S and M within 0.25 ms, the nano models about 0.7-0.8 ms slower (fixed per-call
cost, TensorRT 11.4, two jobs at once); YOLO11-N is faster than YOLO26-N on the GPU here and in Ultralytics' table. The
CPU rows came from 4 shared Kaggle cores with the other job running: not a check of the paper's CPU column (the kit now
runs the two jobs one after the other and reports model-only CPU ms).

## Follow-up: the rain experiment (notebook ready 26 Oct; not run yet)
Question: does the MuSGD rain effect (synthetic rain: 232 vs 16 plates, one seed) survive 3 seeds, and which of MuSGD's
two changes causes it? Notebook [`kaggle/paper33rain_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper33/kaggle/paper33rain_kaggle.ipynb), steps in
[Rain_Experiment_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper33/Rain_Experiment_Guide.md), code `kaggle/p33_rain.py` (built by `_kit/build_kits.py 33rain`
from `_kit/specs_exp.py`).

| Arm | Muon term | 3x head lr | Part (about 8 h each) |
|---|---|---|---|
| `sgd` (Ultralytics SGD) | off | off | 1 |
| `musgd` (Ultralytics MuSGD) | on | on | 1 |
| `muon` | on | off | 2 |
| `headlr` | off | on | 2 |

YOLO26-N, the #33 recipe, seeds 1-3 (same seed = same data order for every arm). A self-test checks that MuSGD with
the Muon step off and no 3x head equals SGD after one step (difference 0.0 on 26 Oct). Every run is scored plate by
plate (both heads) on clean test, night, synthetic rain and fog; `rain_analysis.md` gives per-seed found %, arm-vs-arm
tests (Welch on seeds, sign test over plates, per-seed McNemar) and, with both parts, the 2 x 2 factorial. The MuSGD
runs (seeds 1-3, 60 epochs) also replace the planned re-run of the stopped `y26n`.
