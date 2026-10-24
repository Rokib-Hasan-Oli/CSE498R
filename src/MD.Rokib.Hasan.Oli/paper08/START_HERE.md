# Paper #8: start-to-end guide

**Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN**

| What | Where |
|---|---|
| Notes (read these with the paper) | [8_Bengali License Plate Recognition - Unveiling Clarity with CNN and GFP-GAN.md](../../../report/MD.Rokib.Hasan.Oli/paper08/paper08_notes.md) |
| Kaggle notebook | `kaggle/paper08_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | **Trained on v1.0 (before 21 Oct 2026)? Re-run on v1.1:** everything (all of it is on the plate crops). |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2312.10701 · code + data: https://github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN
- Team copy (PDF): `D:\498R\Papers\6_1 Bangla_Bangladeshi LPR (core)\08_2023_Bengali LPR CNN and GFP-GAN.pdf`

## 2. Why this paper matters for 498R
A ready Bangla symbol dataset with code, and the question we must answer for the board: does image restoration (GFP-GAN) before the OCR help, and what does it cost? The paper never tests it without GFP-GAN.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.2-3, Table I: the 17-class symbol data (2,024 crops, split 1,292 / 429 / 303) and where it came from
   - p.3: GFP-GAN, then OpenCV contrast, Otsu and contours to cut the plate into symbols
   - p.4, Table II: CNN 92.38 % vs ResNet-50 87.50 % symbol accuracy (batch 1, Adam, 40 / 30 epochs)
   - p.5: the plate test (30 plates, mean SequenceMatcher similarity 88 %; 17/30 ≥ 0.95, 24/30 ≥ 0.90)
   - Watch out (see "Paper vs. code" in the notes): 48 of 303 test crops copy a training crop; the code uses 28x28, not 32x32; and GFP-GAN is a FACE model, so on a plate only its Real-ESRGAN x2 step does anything
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 4 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper08_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper08_outputs.zip`: everything, with the weights
   - `paper08_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check (the paper's own data):** `kaggle/paper08_original_check.ipynb`, about 1 h. Steps in [Original_Check_Guide.md](Original_Check_Guide.md). Add the Kaggle input it names.

## 5. What to check and learn
- Original check (the authors' Data.zip, 3 seeds): do we get 92.38 % / 87.50 %? How much lower on the clean test crops?
- On our data: the same CNN and ResNet-50 on 101 Bangla tokens (a much harder task than 17 symbols)
- With vs without GFP-GAN on the same 651 plates and the same CNN: similarity, plate exact, ms per plate
- x order (the paper) vs two-line order, and perfect segmentation (human token boxes): where is the error, in cutting the plate or in classifying the symbols?
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper08_results.zip` and `results_vs_paper.md` into `weights/paper08/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper08_outputs.zip` to `498R / weights / paper08 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper08_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- If GFP-GAN does not help (or costs too much), drop restoration from our pipeline and say so with these numbers
- Per-symbol classification vs character detection (#2) vs sequence OCR (#31): use the plate exact rows to choose
- Report the leakage of the paper's split (audit) next to #23 in the method section
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
