<div align="center">

# Real-Time Bangladeshi License Plate Detection & Recognition

### A Deep Learning–Based Framework for Bangladeshi Traffic Scenarios

<p>
  <img src="https://img.shields.io/badge/Course-CSE%20498R-1f6feb?style=flat-square" alt="Course: CSE 498R" />
  <img src="https://img.shields.io/badge/Semester-Fall%202026-0e8a16?style=flat-square" alt="Semester: Fall 2026" />
  <img src="https://img.shields.io/badge/Research-Computer%20Vision-8250df?style=flat-square" alt="Research: Computer Vision" />
  <img src="https://img.shields.io/badge/Focus-Real--Time%20LPR-f0883e?style=flat-square" alt="Focus: Real-Time LPR" />
  <img src="https://img.shields.io/badge/Papers%20uploaded-19%20of%2025-informational?style=flat-square" alt="Papers uploaded: 19 of 25" />
</p>

**Course:** CSE 498R — Research  
**Faculty:** Dr. Md Mansur ul Mulk (MSLK)  
**Semester:** Fall 2026 · North South University

</div>

---

## 📌 Project overview

This research builds a practical **License Plate Detection and Recognition (LPR)** system for Bangladeshi traffic:
a **YOLO-based detector** finds the plate, and an **OCR module** reads the Bangla text (district, "মেট্রো", letter
and digits).

**Current phase (October 2026): re-implementing 20 related papers.** Each member re-runs five published methods on
our shared dataset, the **498R BanglaLPR All-in-One** (10,623 detection photos and 7,024 plate crops from 15 public
sources; see [data/](data/README.md)), on Kaggle (2× Tesla T4), and compares the result with what the paper reports.
The best ideas feed our own pipeline.

```mermaid
flowchart LR
    A[Image or video frame] --> B[YOLO plate detection]
    B --> C[Plate crop]
    C --> D[Bangla OCR]
    D --> E[Plate text]
    E --> F[Evaluation: accuracy and speed]
```

## 🗂️ Repository map

```text
CSE498R/
├── README.md            ← you are here: project, paper tracker, where things are
├── CONTRIBUTING.md      ← how to add your results (which file goes where, how to upload)
├── data/                ← dataset card: sources, licences, statistics, sample images
├── weights/             ← ONE FOLDER PER PAPER AND MEMBER: weights/paperNN/<Member>/
│                          README (start here), results_vs_paper.md, paperNN_results.zip, best*.pt, study report PDF
├── notebooks/<Member>/  ← finished Kaggle notebooks: paperNN_run.ipynb (+ .log)
├── src/<Member>/        ← code to run a paper again: paperNN/START_HERE.md, paperNN/kaggle/paperNN_kaggle.ipynb
├── report/<Member>/     ← written reports, reading notes, report sources
├── results/             ← team results table (ours vs. the papers, all members)
├── output/              ← full Kaggle output zips (Git LFS, 40–250 MB each)
└── requirements.txt     ← packages to load the models locally
```

**To look at one paper's results, open `weights/paperNN/<Member>/README.md`.** It links to everything else.

## 📄 Paper tracker

Status: ✅ done on dataset v1.1 · ☑️ done on v1.0 and still valid · 🔁 uploaded, re-run on v1.1 needed ·
🟡 partly uploaded · ⬜ not uploaded yet

### MD. Rokib Hasan Oli — `MD.Rokib.Hasan.Oli`

| # | Paper | Status | Main result (our data) | Files |
|---|---|---|---|---|
| 2 | Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition | ✅ | Character-YOLO reader: **87.71 %** of 651 plates exact; full pipeline 86.76 %; at night 66.7 % vs 38.3 % for the authors' weights | [results](weights/paper02/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper02_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper02/) · [notes](report/MD.Rokib.Hasan.Oli/paper02/) |
| 6 | YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles | ✅ | YOLOv8n/s mAP50 **97.39 / 97.61 %**; the paper's stock EasyOCR reads only 13.52 % of plates exactly | [results](weights/paper06/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper06_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper06/) · [notes](report/MD.Rokib.Hasan.Oli/paper06/) |
| 8 | Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN | ✅ | Symbol CNN **93.08 %**, but the paper's full plate reader gets 0.15 % of plates exact (its contour step finds 52 % of symbols) | [results](weights/paper08/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper08_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper08/) · [notes](report/MD.Rokib.Hasan.Oli/paper08/) |
| 19 | LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification | ✅ | A legibility classifier is **not** a better reject-or-read gate than the OCR's own confidence: keeping 80 % of plates, CRNN confidence 84.5 % right vs classifier 81.0 % (CI −5.6 to −1.3); confidence + BRTA layout + reader agreement gives **96.7 %** for the #24 fusion. Proxy labels: 4-class 51.9 % / Quality Filter 71.7 % (paper 89.8 / 96.7); EMA loss: no gain on 4 classes (3 seeds) | [results](weights/paper19/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper19_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper19/) · [notes](report/MD.Rokib.Hasan.Oli/paper19/) |
| 23 | Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition | ✅ | KUET's leaky split inflates plate reading from **77.84 % to 85.57 %** on the same 194 plates (p = 0.008) | [results](weights/paper23/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper23_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper23/) · [notes](report/MD.Rokib.Hasan.Oli/paper23/) |
| 24 | Leveraging Model Fusion for Improved License Plate Recognition | ✅ | Format fallback (char-YOLO, then CRNN): **88.63 %** exact, +1.84 over the best single model (12 fixed, 0 broken) | [results](weights/paper24/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper24_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper24/) · [notes](report/MD.Rokib.Hasan.Oli/paper24/) |
| 33 | YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection | ✅ | Five detectors tie on clean plates (**97.6-98.4 %** found; COCO numbers reproduced to 0.1); NMS-free head removes CPU NMS (7.5 → 0.3 ms); MuSGD models find ~20 points more plates in synthetic rain (232 vs 16) | [results](weights/paper33/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper33_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper33/) · [notes](report/MD.Rokib.Hasan.Oli/paper33/) |
| 34 | SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition | ✅ | SVTRv2-T (official code, English start) reads **81.26 %** of test plates vs CRNN 73.43 % (p < 0.0001), below char-YOLO 86.79 %; from scratch 61.75 %. As a format fallback behind char-YOLO: **90.17 %** (22 fixed / 0 broken); 4-reader vote chosen on val **90.94 %** | [results](weights/paper34/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper34_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper34/) · [notes](report/MD.Rokib.Hasan.Oli/paper34/) |
| 35 | HVI: A New Color Space for Low-light Image Enhancement (HVI-CIDNet) | ✅ | Zero-shot low-light enhancement does **not** help night plates: 64 / 71 night plates found with or without it; night crops read 61.5 → 55.4 %; end to end +4 / 60 (n.s., same as CLAHE); 155 ms per photo on a T4. LOL numbers reproduced (GT mean / lost weights flagged) | [results](weights/paper35/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper35_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper35/) · [notes](report/MD.Rokib.Hasan.Oli/paper35/) |
| 36 | SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection | ✅ | The paper's slicing **hurts** our plates: AP50 97.33 → 91.64 (3x false alarms, 3.5x time; slice edges cut large plates). Dropping cut-off slice boxes (chosen on val): **97.51** (= full image, n.s.). Slice training (SF): small-plate photos mAP50 84.5 → **96.3**; 1280 px no gain. Paper Table 1 reproduced 80 / 80 | [results](weights/paper36/MD.Rokib.Hasan.Oli/) · [notebook](notebooks/MD.Rokib.Hasan.Oli/paper36_run.ipynb) · [kit](src/MD.Rokib.Hasan.Oli/paper36/) · [notes](report/MD.Rokib.Hasan.Oli/paper36/) |

### Md. Yousuf — `Md.Yousuf`

| # | Paper | Status | Main result (our data) | Files |
|---|---|---|---|---|
| 3 | A Cloud-Based Hybrid Model for Real-Time Detection of BRTA-Approved Licence Plates Using YOLO Tiny and Haar Cascade | ☑️ | YOLOv8n plate detector mAP50 **98.80 %** (416 px); the paper's Haar-cascade design F1 4.19 % | [results](weights/paper03/Md.Yousuf/) · [notebook](notebooks/Md.Yousuf/paper03_run.ipynb) · [report](report/Md.Yousuf/Paper03_Report.md) |
| 20 | Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark | 🔁 | CRNN reader 74.04 % exact; ×2 super-resolution 24.12 % vs bicubic 23.81 % on low-res plates | [results](weights/paper20/Md.Yousuf/) · [notebook](notebooks/Md.Yousuf/paper20_run.ipynb) · [report](report/Md.Yousuf/Paper20_Report.md) |
| 26 | YOLOv12: Attention-Centric Real-Time Object Detectors | ✅ | Plate detectors: YOLO11-S mAP50 **99.0 %**, Ultralytics YOLO12-S 98.9 %, official YOLOv12-S 98.6 % (mAP50-95 48.5 %); character readers YOLO11-N 83.56 % vs YOLO12-N 82.80 % plates exact | [results](weights/paper26/Md.Yousuf/) · [notebook](notebooks/Md.Yousuf/paper26_run.ipynb) · [report](report/Md.Yousuf/Paper26_Report.md) · [outputs zip](output/paper26_outputs.zip) |
| 27 | YOLOv10: Real-Time End-to-End Object Detection | 🔁 (character models only) | YOLOv10-N and YOLOv8-N plate mAP50 **98.7 %**; character reader v8n 84.02 % vs v10n 81.57 % | [results](weights/paper27/Md.Yousuf/) · [notebook](notebooks/Md.Yousuf/paper27_run.ipynb) · [report](report/Md.Yousuf/Paper27_Report.md) |
| 28 | YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information | ☑️ | YOLOv9-S best mAP50 **98.79 %**; YOLOv8-N fastest (6.38 ms TensorRT FP16) | [results](weights/paper28/Md.Yousuf/) · [notebook](notebooks/Md.Yousuf/paper28_run.ipynb) · [report](report/Md.Yousuf/Paper28_Report.md) |

### Md Rakibul Hasan — `Md.Rakibul.Hasan`

| # | Paper | Status | Main result (our data) | Files |
|---|---|---|---|---|
| 11 | Real-time Bangla License Plate Recognition System for Low Resource Video-based Applications | 🟡 notebook + outputs zip; `weights/paper11/Md.Rakibul.Hasan/` still missing | YOLOv8n plate detector AP50 **98.7 %** (640 px); character reader 86.18 % plates exact; the paper's Haar gate skips 97.7 % of photos (recall 2.47 %) | [notebook](notebooks/Md.Rakibul.Hasan/paper11_run.ipynb) · [outputs zip](output/paper11_outputs.zip) · [team table](results/results_table.md) |
| 17 | Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion | ⬜ | – | – |
| 18 | An Embedded Real-Time License Plate Recognition System for Complex Traffic Scenes | 🟡 notebook + outputs zip; `weights/paper18/Md.Rakibul.Hasan/` still missing | YOLOv8n detector AP50 **99.0 %** (5.6 ms INT8 on a T4); slot OCR 57.60 % plates exact, 37.33 % after INT8 | [notebook](notebooks/Md.Rakibul.Hasan/paper18_run.ipynb) · [outputs zip](output/paper18_outputs.zip) · [team table](results/results_table.md) |
| 21 | Advanced deep learning techniques for automated license plate recognition | 🟡 notebook + outputs zip; `weights/paper21/Md.Rakibul.Hasan/` still missing | YOLOv5/8/9/10 plate detectors all mAP50 **98.2-98.8 %**; Tesseract (ben) reads 0.77 % of plates exactly | [notebook](notebooks/Md.Rakibul.Hasan/paper21_run.ipynb) · [outputs zip](output/paper21_outputs.zip) · [team table](results/results_table.md) |
| 30 | DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) | ⬜ | – | – |

### Tashfia Adiba Zaman Nizhum — `Tashfia.Adiba.Zaman.Nizhum`

| # | Paper | Status | Main result (our data) | Files |
|---|---|---|---|---|
| 1 | A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision-Language OCR | 🟡 one notebook for papers 1, 10, 22, 31 and 32 | Precision: **95.80%**; Recall: **94.67%**; mAP@50: **96.50%**; mAP@50-95: **44.46%** | [notebook](notebooks/Tashfia.Adiba.Zaman.Nizhum/papers01_10_22_31_32_run.ipynb) |
| 10 | BLPnet: A new DNN model and Bengali OCR engine for Automatic License Plate Recognition | ⬜ | – | – |
| 22 | Advancing Vehicle Plate Recognition: Multitasking Visual Language Models with VehiclePaliGemma | ⬜ | – | – |
| 31 | Scene Text Recognition with Permuted Autoregressive Sequence Models (PARSeq) | ⬜ | – | – |
| 32 | Optical Text Recognition in Nepali and Bengali: A Transformer-based Approach | ⬜ | – | – |

All numbers side by side: [results/results_table.md](results/results_table.md). Numbers from different dataset
versions or different test sets are not directly comparable.

## ⬆️ How to add your results (short version)

1. Finish the Kaggle run (`FAST = False`) and download the notebook and `paperNN_outputs.zip`.
2. Notebook → `notebooks/<Member>/paperNN_run.ipynb`.
3. README, `results_vs_paper.md`, `paperNN_results.zip`, `best*.pt` (< 25 MB each on the web) → `weights/paperNN/<Member>/`.
4. The full `paperNN_outputs.zip` → team Drive `498R / weights / paperNN /`.
5. Update your row in the paper tracker above.

Full guide, with a README template and the upload steps: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## 📦 Where the big files are

| What | Where |
|---|---|
| Dataset v1.1 (zip) | Team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl), or the Kaggle dataset `498r-bangla-lpr-all-in-one`. The notebooks download it by themselves |
| Full run outputs (`paperNN_outputs.zip`, every `last.pt`) | Team Drive `498R / weights / paperNN /`; some are also in [output/](output/README.md) (Git LFS) |
| Models over 100 MB | Team Drive |

## 🧪 Using a trained model

```bash
pip install -r requirements.txt      # install PyTorch first from pytorch.org
```

Each `weights/paperNN/<Member>/README.md` has a **Quick start** with the exact code for its models. The Bangla
plate decoder (`vocab.py`) is inside each `paperNN_results.zip` (`code/`) and in `src/<Member>/paperNN/kaggle/`.

## 👥 Research team

| Student ID | Team member | University email | Papers |
|---|---|---|---|
| 2211461642 | Md. Yousuf | [md.yousuf01@northsouth.edu](mailto:md.yousuf01@northsouth.edu) | 3, 20, 26, 27, 28 |
| 2211950642 | MD. Rokib Hasan Oli | [rokib.oli@northsouth.edu](mailto:rokib.oli@northsouth.edu) | 2, 6, 8, 19, 23, 24, 33, 34, 35, 36 |
| 2131228042 | Tashfia Adiba Zaman Nizhum | [tashfia.nizum@northsouth.edu](mailto:tashfia.nizum@northsouth.edu) | 1, 10, 22, 31, 32 |
| 2212346042 | Md Rakibul Hasan | [rakibul.hasan30@northsouth.edu](mailto:rakibul.hasan30@northsouth.edu) | 11, 17, 18, 21, 30 |

## 🎯 Research objectives

- **Detect plates** with YOLO-family detectors in photos and video.
- **Read Bangladeshi plate text** (Bangla district names, letters and digits) reliably.
- **Stay robust** at night and in rain, fog, glare and motion blur, for all vehicle types.
- **Run in real time**: measure latency and FPS, including quantised (TensorRT FP16) models.

| Evaluation area | Metrics |
|---|---|
| Detection | Precision, recall, mAP50, mAP50-95, plates found |
| Text recognition | Whole-plate exact match, character accuracy, CER |
| Speed | ms per image, FPS (PyTorch FP32, TensorRT FP16, CPU) |
| Robustness | Results per condition (night, blur, small plates, weather) and per vehicle type |

## 🎓 Academic information

- **Course:** CSE 498R Research
- **Semester:** Fall 2026
- **Faculty:** Dr. Md Mansur ul Mulk (MSLK)
- **Institution:** North South University

---

<div align="center">

*Research project in progress · Fall 2026*

</div>
