# Paper #8: Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN

*Reader: MD. Rokib Hasan Oli · Week 1 · Board group: Bangla / Bangladeshi LPR*
*PDF: `Papers/6_1 Bangla_Bangladeshi LPR (core)/08_2023_Bengali LPR CNN and GFP-GAN.pdf` (6 pages). Page numbers below are the arXiv PDF pages.*

---

## Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN (2023)
```
Problem:   Read the characters of Bangladeshi number plates from blurry, low-resolution images
           (street-camera quality), where earlier Bangla LPR work assumed clean images.
Method:    Enhance the plate image with GFPGAN, then classic OpenCV steps (grey, top-hat/black-hat contrast,
           Otsu threshold, erode/dilate) and contour-based segmentation into single symbols. Each symbol is
           classified by a small CNN (4 large-kernel conv layers, 17 classes); ResNet-50 is tried as an alternative.
           No plate detector: the input is already a plate image.
Data:      Own character dataset: 2,024 crops (32x32 RGB) in 17 classes (digits 0-9; series letters ka, ga, jha,
           la; Dhaka, Metro, Narayanganj), split 1,292 / 429 / 303 (p.3 Table I, p.4). Crops were cut by hand from
           a Kaggle synthetic plate set and the BLRPS set (p.2 footnotes). Public, MIT licence:
           https://github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN (Data.zip)
Results:   Character test set: CNN accuracy = 92.38 %, precision 0.9667, recall 0.9162, F1 0.9476;
           ResNet-50 accuracy = 87.50 %, F1 0.8984 (p.4, Table II). Whole plates: mean SequenceMatcher
           similarity = 88 % on 30 plates; 17/30 >= 0.95 and 24/30 >= 0.90 (p.5). Hardware, FPS and
           training time: not reported.
Limits:    Leaky split: 48 of 303 test crops and 320 of 429 validation crops are byte-identical copies of training
           crops. Every test crop of Dhaka, Metro, Narayanganj and jha is a copy (my audit of Data.zip), so the
           letter/word part of 92.38 % is partly memorisation and 98.8 % val is inflated. Other limits:
           - no detector;
           - only 17 symbols, so most districts and series letters cannot be read;
           - x-only character ordering mixes the two plate lines;
           - no with/without-GFPGAN ablation, so the restoration benefit is unproven;
           - 30-plate test;
           - paper and code disagree on several numbers (see "Paper vs. code" below).
For us:    A ready, labelled 17-class Bangla character set and a baseline (CNN 92.4 % char accuracy, 88 % plate
           similarity) to beat. It also gives us a clear question to test properly: does restoration help, and what does it cost?
           Answer from our v1.1 run (22 Oct 2026): DROP the pipeline, keep the lesson. The symbol CNN is fine (93.1 % of
           4,107 test symbols, 101 tokens), but contour segmentation finds only 52 % of the characters, so it reads
           0.15 % of 651 real plates exactly; GFP-GAN adds 4.8 similarity points for 205.6 ms per plate. A character
           detector (paper #2) reads 88.9 % of the same plates. Restoration stays only as an ablation (x2 upscaling).
Code:      https://github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN
           (official, linked on p.1). The short link in our research plan (…/Bengali-License-Plate-Recognition) is a 404.
BibTeX:    afrin2023bengali  (full entry at the bottom; confirm the key on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** read Bangla plate characters when the image is blurry and low-resolution.
2. **How:** GFPGAN restoration → OpenCV contrast/threshold/contours → per-symbol CNN classifier (17 classes).
3. **How well:** 92.38 % symbol accuracy on their own crops (p.4), 88 % average string similarity on 30 plates (p.5).

**Useful for us?** Partly. It's a weak venue and a small test. But it has working code and a public labelled
character set, and it asks the restoration question we need to answer. Use it as a how-to and as a baseline. Don't cite it as strong evidence.

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Dataset | 2,024 symbol crops, 17 classes, 32×32 RGB; train/val/test = 1,292 / 429 / 303 | p.3 (Table I), p.4 |
| Source of crops | Kaggle `furcifer/bangla-license-plates-synthetic` + GitHub `mmshaifur/BLRPS`; manual cropping | p.2 |
| Class balance | digits 90–140 train images each; letters and city words only 13–30 | p.3 |
| CNN | 4 conv layers (16/32/64/128 filters, kernels 22/16/8/4, "same"), max-pool 4, dense 512 → dropout 0.4 → 128 → 17 | code (CNN.ipynb) |
| ResNet-50 | ImageNet weights, **all layers frozen**, Flatten → Dense(17); 150×150 input | p.2, code |
| Training | lr 1e-4, batch size 1, CNN 40 epochs, ResNet-50 30 epochs; no GPU named | p.4, p.5 |
| Enhancement | GFPGAN v1.3, ×2, Real-ESRGAN background upsampler | p.3, code (GFPGAN.ipynb) |
| Segmentation | resize 333×335, Otsu, erode/dilate, 15 largest contours, sorted by x | p.4, code |
| Plate metric | `difflib.SequenceMatcher` ratio between expected and predicted strings, averaged | p.5 |

**Every result number**
| Result | Value | Page |
|---|---|---|
| CNN test accuracy / precision / recall / F1 | 92.38 % / 0.9667 / 0.9162 / 0.9476 | p.4, Table II |
| ResNet-50 test accuracy / precision / recall / F1 | 87.50 % / 0.9343 / 0.8827 / 0.8984 | p.4, Table II |
| CNN validation accuracy / loss (40 epochs) | 98.80 % / 0.0303 (the same paragraph also says "92 %") | p.4 |
| Plate-level similarity (30 plates) | 88 % average | p.1, p.5 |
| Plates ≥ 0.95 / ≥ 0.90 similarity | 17/30 (≈57 %) / 24/30 (80 %) | p.5 |
| Example plate scores | 1.00, 0.95, 0.86, 0.84 | p.5, Table III |
| Comparison table | Proposed CNN 88 %, ResNet-50 83 %, Kabiraj et al. (CNN + ESRGAN) 84 %, Haque et al. (ESRGAN) 78 %. Different datasets, so this is not a fair comparison. | p.6, Table IV |
| FPS, latency, GPU, training time | not reported | – |

Terms I looked up:
- **GFPGAN:** a GAN that restores *faces*, using a pretrained StyleGAN as a "facial prior" (Wang et al., CVPR 2021). The paper describes it wrongly as a flow-based probabilistic graphical model (p.2) and cites StyleHEAT [12] for it.
- **SequenceMatcher ratio:** 2 × matching characters ÷ total characters of both strings. It gives partial credit, so 0.9 does **not** mean 90 % of plates were correct.

## Pass 3: rebuild it
```
plate image ─► GFPGAN ×2 ─► grey ─► top-hat/black-hat ─► 333×335 ─► Otsu ─► erode/dilate
   (already      (on a plate:    (contrast)                                   │
   cropped)       Real-ESRGAN x2                                              ▼
                  does the work)              15 largest contours ─► sort by x ─► each crop 28×28
                                                                              │
                                                     CNN (17 classes) ◄───────┘
                                                          │
                                       join names ─► "9dhaka94metro773jha7" ─► SequenceMatcher vs. truth
```
**Assumptions:** the plate is already cut out and nearly frontal, and each symbol is one connected blob. The plate holds only the 17 known symbols, and the contour order is a usable reading order.

**What would fail on our Bangladeshi plates:**
- **Two lines:** sorting by x alone interleaves the top and bottom lines. Their "expected" strings follow that mixed order (p.5, Table III), so the 88 % is easier than true reading.
- **Bangla letters:** only 4 series letters and 3 region words. Our plates use 101 tokens (26 series letters, 64 district words, short forms such as চট্ট), e.g. চ and যশোর, which the paper's model cannot output.
- **Night and blur:** Otsu on a dark or glared plate breaks the contours. GFPGAN can sharpen edges but can also invent strokes.
- **Full photos:** there is no detector, so on whole photos a plate detector must come first. Our kit reads the 651 human-cut OCR test crops, so the pipeline gets the plate it expects.
- **Mātrā:** a word like ঢাকা is one connected blob, and a broken মাত্রা splits it, which the paper itself admits (p.5–6).

**How I reproduce it:** two Kaggle notebooks, built by `docs/_kit` like the other papers (steps in [START_HERE.md](../../../src/MD.Rokib.Hasan.Oli/paper08/START_HERE.md)):
1. **Original-paper check** (`kaggle/paper08_original_check.ipynb`, about 1 h): the authors' Data.zip and settings,
   ported to PyTorch, 3 seeds, the leakage audit, a re-test on the clean crops, and the plate test on the synthetic
   plates the authors cut their symbols from (GFP-GAN on / off).
2. **Main notebook on our data** (`kaggle/paper08_kaggle.ipynb`, about 4 h): the same CNN and ResNet-50 trained on
   every token box of the 498R All-in-One recognition split (101 classes), then the paper's plate pipeline on our 651
   OCR test crops with and without GFP-GAN, in x order and two-line order, plus perfect segmentation (human boxes).

The first kits (9-10 Oct, team DIU photos without character labels) are kept in `_old_kit_2026-10-10/`.

## Check yourself (2-minute version)
> They cut Bangla plate symbols into 17 classes, train a small CNN on 32×32 crops and get 92 % symbol accuracy.
> To read a full plate, they sharpen it with GFPGAN, threshold it, cut symbols with contours and run the CNN.
> On 30 plates the output string is 88 % similar to the expected one. But they never test without GFPGAN, there is no
> detector, it knows only 3 districts and 4 letters, and it ignores the two-line layout.

## Paper vs. code (checked 2026-10-09 in the repo, commit `8dc72c7`)
| Topic | Paper | Authors' code / data |
|---|---|---|
| Input size | 32×32×3 (p.3–4) | CNN 28×28, ResNet-50 150×150 |
| Dataset size | "1,292 images" (p.1) | that is only the train split; Data.zip holds 2,024 |
| Class 9 count | 1 train / 45 val (p.3, Table I) | 100 train / 35 val |
| ResNet-50 test accuracy | 87.50 % (p.4) | notebook prints 73.93 % |
| Plate score | average of 30 plates (p.5) | notebook averages 10 hard-coded scores = 87.56 % |
| ResNet-50 learning rate | 1e-4 (p.4) | `optimizer='adam'` = Keras default 1e-3 |
| Plate demo | – | index→label list doesn't match the class order; crops are not scaled by 1/255 before the CNN |
| GFPGAN | "restores" plates (p.3) | face model; with no face on a plate, only its Real-ESRGAN ×2 background step changes the image |
| Image size in the data | "standardized to 32x32" (p.3) | 1,729 of 2,024 crops are 32×32; 295 are other sizes (up to 191×147) |
| Train/test separation | not discussed | 48/303 test and 320/429 val crops are byte-identical to a training crop. All test crops of Dhaka (12), Metro (10), Narayanganj (10) and jha (14) are copies; digits are clean (1 of 229) |

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| Bengali LPR: Unveiling Clarity with CNN and GFP-GAN (Afrin et al.) | 2023 | ICCIT 2023 (IEEE) / arXiv 2312.10701 | Own 17-class Bangla symbol crops, 2,024 (1,292/429/303), public | None (plate image given); OpenCV contours segment the symbols | GFPGAN enhance + 4-layer CNN per symbol (ResNet-50 tried) | Symbol acc 92.38 %, F1 0.9476 (p.4); plate similarity 88 % on 30 plates (p.5); no FPS | Yes (TF/Keras Colab notebooks + data, MIT) | Leaky split (48/303 test, 320/429 val crops copied from train), 17 symbols only, no detector, x-only ordering mixes lines, no GFPGAN ablation, tiny plate test | Public char crops for pre-training; a baseline to beat; we test whether restoration helps and what it costs |

## Code check → Research board
- **Repository:** https://github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN
- **Last update:** 13 Sep 2023, licence MIT, 3 stars, 0 open issues.
- **Contents:** 5 Colab notebooks (TensorFlow 2.12): CNN, ResNet-50, GFPGAN, image processing, SequenceMatcher. Also `Data.zip`.
- **Missing:** no `requirements.txt`, no trained weights, and the README links (installation.md, usage.md …) are dead.
- **Runs?** The notebooks are Colab-only (hard-coded Drive paths) and were written for TF 2.12 / Keras 2, not checked on today's Keras 3. I ported the CNN and ResNet-50 to PyTorch (`docs/_kit/extras/p08_train.py`). It runs on Kaggle T4 and on CPU; a local CPU run gave 90.4 % test accuracy (paper: 92.38 %).

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN
```
```
Official repo, MIT, Sep 2023: 5 Colab notebooks (TF 2.12) + Data.zip (2,024 char crops, 17 classes), no weights. My PyTorch port runs on Kaggle T4: their data (3 seeds, leak audit) + our 101 tokens, GFP-GAN on/off. Paper: CNN 92.38% (p.4), 88% plate similarity (p.5).
```

## BibTeX
```bibtex
@inproceedings{afrin2023bengali,
  title     = {Bengali License Plate Recognition: Unveiling Clarity with {CNN} and {GFP-GAN}},
  author    = {Afrin, Noushin and Hasan, Md Mahamudul and Safin, Mohammed Fazlay Elahi and Amin, Khondakar Rifat
               and Haque, Md Zahidul and Ahmed, Farzad and Shawon, Md Tanvir Rouf},
  booktitle = {2023 26th International Conference on Computer and Information Technology (ICCIT)},
  address   = {Cox's Bazar, Bangladesh},
  year      = {2023},
  note      = {arXiv:2312.10701}
}
```
*The DOI wasn't found online yet. Search the exact title on IEEE Xplore and take the BibTeX from Google Scholar before the report.*

## Earlier local checks (9-10 Oct 2026, my PC, CPU, the authors' Data.zip)
Before the Kaggle runs, the PyTorch port was checked on CPU with the paper's settings (one seed):

| Model, the authors' data | Ours (CPU) | Paper (page) | Authors' own notebook |
|---|---|---|---|
| CNN 28x28, test accuracy: best-val / last epoch | 90.43 % / 91.42 % (14 min) | 92.38 % (p.4) | 93.40 % |
| CNN macro P / R / F1 (best-val) | 0.910 / 0.922 / 0.910 | 0.9667 / 0.9162 / 0.9476 (p.4) | |
| CNN on the clean test crops (copies removed, 255): best-val / last | 88.6 % / 89.8 % | not checked | |
| ResNet-50 frozen 150x150, test accuracy | not run on CPU | 87.50 % (p.4) | 73.93 % |

The Kaggle numbers (3 seeds, the plate test) replace these in "Original-paper check" below.

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper08_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper08/kaggle/paper08_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper08/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170, spandrel; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 4 h for the full run |

| Run | GPU | What |
|---|---|---|
| `cnn28` | 0 | paper CNN, 28x28, 40 epochs, batch 1, Adam 1e-4, on our token crops |
| `resnet50` | 1 | ResNet-50 (ImageNet, frozen) + Dense, 150x150, 30 epochs, batch 1, Adam 1e-3 |
| `plates_cnn28` | 0 (after) | paper plate pipeline with/without GFP-GAN on the 651 OCR test crops + GT boxes |
| `fps_cnn28 / fps_resnet50` | 1 (after) | classifier speed, batch 1, T4 |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper08_outputs.zip` (team Drive `498R / weights / paper08 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #8: ours vs. the paper

| Metric | Paper (page) | cnn28 | resnet50 | plates_cnn28 |
|---|---|---|---|---|
| Classes | 17 (10 digits, 4 letters, 3 region words) (p.3, Table I) | 101 | 101 | 101 |
| Character test accuracy, last epoch (%) | 92.38 (CNN), 87.50 (ResNet-50) (p.4, Table II) | 36.67 | 92.06 | – |
| Character macro precision (%) | 96.67 (CNN), 93.43 (ResNet-50) (p.4, Table II) | 13.69 | 49.03 | – |
| Character macro recall (%) | 91.62 (CNN), 88.27 (ResNet-50) (p.4, Table II) | 7.15 | 49.02 | – |
| Character macro F1 (%) | 94.76 (CNN), 89.84 (ResNet-50) (p.4, Table II) | 8.16 | 46.41 | – |
| Character test accuracy, best-val checkpoint (%) | - | 93.08 | 92.89 | – |
| Best validation accuracy (%) | 98.80 (CNN) (p.4) | 90.20 | 91.71 | – |
| Epochs run | 40 (CNN), 30 (ResNet-50) (p.4-5) | 40 | 25 | – |
| Classifier FPS, batch 1, T4 (model only) | not reported | 760.90 | 171.50 | – |
| Training time (min) | not reported | 151.80 | 198.20 | – |
| Plates: mean similarity, GFP-GAN + x order (%) | 88 (30 plates) (p.5) | – | – | 44.95 |
| Plates with similarity >= 0.95 (%) | 57 (17 of 30) (p.5) | – | – | 0.00 |
| Plates with similarity >= 0.90 (%) | 80 (24 of 30) (p.5) | – | – | 0.00 |
| Plates: mean similarity WITHOUT GFP-GAN (%) | never tested | – | – | 41.86 |
| Plates exact, GFP-GAN + two-line order (%) | not reported | – | – | 0.15 |
| Plates exact, no GFP-GAN + two-line order (%) | - | – | – | 0.00 |
| Plates exact, perfect segmentation (human token boxes) (%) | - | – | – | 59.53 |
| Char accuracy, perfect segmentation (%) | - | – | – | 86.77 |
| GFP-GAN (Real-ESRGAN x2) ms per plate | not reported | – | – | 205.60 |
| Segmentation + CNN ms per plate | not reported | – | – | 6.10 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's 17 classes and 303 test crops (48 are copies of training crops, see the original check) vs our 101 tokens on the grouped split: a harder task, so a lower symbol accuracy is expected. The paper's plate score is the mean difflib SequenceMatcher ratio, which gives partial credit; plate exact is the strict number.

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| plates_cnn28 | 0.15 | 0.21 | 0.00 | 0.00 | – | – |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

<!-- results:end -->

**How to read this table (v1.1 run of 22 Oct 2026, checked 24 Oct; written by hand, outside the markers so re-runs keep it).**
Full analysis: [Paper08_Study_Report.pdf](../../../weights/paper08/MD.Rokib.Hasan.Oli/Paper08_Study_Report.pdf).
- **cnn28 "last epoch" 36.67 % is not our accuracy.** The paper's recipe (batch 1, Adam 1e-4, no normalisation) became
  unstable after epoch 5 on our 47,992 crops (one of our epochs = the paper's whole training); training accuracy fell from
  92 % to 52 %. The kit keeps the best-validation checkpoint (epoch 5): **93.08 %**, and the plate rows use it. The macro
  P/R/F1 of the last-epoch rows belong to the collapsed model too.
- **Macro F1 is low because of rare tokens**: 39 of the 60 test classes have 1-19 crops. On the 21 tokens with >= 20 test
  crops the macro F1 is 89.1 % (CNN) / 89.3 % (ResNet-50). ResNet-50 stopped at epoch 25 of 30 (3.2 h cap), flat since ~12.
- **Plates: the contours are the problem.** Local check (`report_src/seg_check.py`): the paper's Otsu + 15 contours find
  52.4 % of the symbols (IoU >= 0.5), all of them on 8.6 % of plates; binary blobs cost the CNN 10.5 points (92.9 -> 82.4 %).
  Even with perfect human boxes the CNN reads 59.53 % of plates vs 88.87 % for the paper #2 char-YOLO on the same 467
  plates (144 vs 7, McNemar p < 0.0001).
- **GFP-GAN** (on plates = its Real-ESRGAN x2 step): +4.8 similarity points with two-line order (343 plates better, 224
  worse, Wilcoxon p = 4e-11), +1 exact plate, night 43.5 -> 41.2 %, 205.6 ms per plate. Not worth it here.

## Original-paper check (the paper's own data)
<!-- check:start -->
The paper's data / weights are public, so a second notebook checks the printed numbers on the SAME data:
[`kaggle/paper08_original_check.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper08/kaggle/paper08_original_check.ipynb), steps in
[Original_Check_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper08/Original_Check_Guide.md). Runs: `cnn_s42 / s43 / s44`, `resnet50_s42 / s43 / s44`, `cnn32_s42, resnet50norm_s42`, `evalclean, plates_*`, `ours_same_data` (about 1 h).
After the run: `python add_results.py paper08_check_outputs.zip --member "Rokib Hasan Oli"` replaces this table.

| Metric | Paper (page) | Ours, same data |
|---|---|---|
| CNN test accuracy, last epoch (%) | 92.38 (p.4, Table II) | not run yet |
| CNN macro precision (%) | 96.67 (p.4, Table II) | not run yet |
| CNN macro recall (%) | 91.62 (p.4, Table II) | not run yet |
| CNN macro F1 (%) | 94.76 (p.4, Table II) | not run yet |
| CNN best validation accuracy (%) | 98.80 (p.4) | not run yet |
| CNN test accuracy, best-val checkpoint (%) | - (authors' notebook prints 93.40) | not run yet |
| CNN on the clean test crops (copies removed) (%) | not checked in the paper | not run yet |
| CNN digits (%) | - | not run yet |
| CNN series letters (%) | - | not run yet |
| CNN series letters, clean only (%) | - | not run yet |
| CNN region words (%) | - | not run yet |
| CNN at 32x32 (the paper's text), test accuracy (%) | 92.38 (text says 32x32) (p.3-4) | not run yet |
| ResNet-50 test accuracy, last epoch (%) | 87.50 (authors' notebook prints 73.93) (p.4, Table II) | not run yet |
| ResNet-50 macro F1 (%) | 89.84 (p.4, Table II) | not run yet |
| ResNet-50 + ImageNet normalisation, test accuracy (%) | - (our fix) | not run yet |
| Test crops that copy a training crop | not discussed | not run yet |
| Val crops that copy a training crop | not discussed | not run yet |
| Crops that really are 32x32 | "standardized to 32x32" (p.3) | not run yet |
| Classes whose counts match Table I | 17 of 17 (p.3, Table I) | not run yet |
| 30 synthetic plates: mean similarity, GFP-GAN + x order (%) | 88 (p.5) | not run yet |
| 30 plates: share >= 0.95 (%) | 57 (17 of 30) (p.5) | not run yet |
| 30 plates: share >= 0.90 (%) | 80 (24 of 30) (p.5) | not run yet |
| 30 plates WITHOUT GFP-GAN: mean similarity (%) | never tested | not run yet |
| All eligible plates: mean similarity, GFP-GAN + x order (%) | - | not run yet |
| 200 plates with any symbol: mean similarity (%) | - | not run yet |
| GFP-GAN (Real-ESRGAN x2) ms per plate (T4) | not reported | not run yet |
| CNN FPS, batch 1, T4 (model only) | not reported | not run yet |
| Training time per CNN run (min) | not reported | not run yet |
<!-- check:end -->

**Local run of this check (24 Oct 2026, RTX 5060 Ti, one job at a time, the notebook's commands; written by hand outside
the markers; the Kaggle run is still to do and will fill the table above).** Numbers: `report_src/check_local.json`.
- CNN 28x28, last epoch: **88.23 ± 1.48 %** (3 seeds; best-val 90.43 ± 0.93); CNN 32x32 (the paper's text), 1 seed:
  **93.40 %**; ResNet-50, last epoch: **87.46 ± 3.27 %** (paper 87.50; best-val 91.64 ± 0.95); ResNet-50 + ImageNet
  normalisation, 1 seed: 90.43 %. So the PyTorch port reproduces the paper on its own data.
- Clean test (255 crops without a training copy): CNN 86.01 ± 1.76 % (-2.2 points); series letters 87.3 -> 80.2 % on
  the 27 clean letter crops; region words 100 %, but all of their test crops are copies. Validation 98.3 % (75 % copies).
- The batch-1 CNN does not collapse here (best epochs 31, 29, 30 of 40): 52k updates vs 1.9M on our data.
- Not run: the synthetic plate test (needs the Kaggle input furcifer/bangla-license-plates-synthetic).
