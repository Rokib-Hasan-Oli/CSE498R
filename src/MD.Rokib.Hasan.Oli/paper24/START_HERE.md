# Paper #24: start-to-end guide

**Leveraging Model Fusion for Improved License Plate Recognition**

| What | Where |
|---|---|
| Notes (read these with the paper) | [24_Leveraging Model Fusion for Improved License Plate Recognition.md](../../../report/MD.Rokib.Hasan.Oli/paper24/paper24_notes.md) |
| Kaggle notebook | `kaggle/paper24_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | **Trained on v1.0 (before 21 Oct 2026)? Re-run on v1.1:** everything. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2309.04331 · Springer: https://doi.org/10.1007/978-3-031-49249-5_5
- Team copy (PDF): `D:\498R\Papers\6_3 General LPR state of the art\24_2023_Leveraging Model Fusion for Improved LPR.pdf`

## 2. Why this paper matters for 498R
Combining several OCR models (voting) beats any single model. This is how to pick our final plate reader.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.3-6: the fusion rules (HC, MV, MV-BM, per-position MVCP)
   - p.9, Table 3 and p.11, Table 5: gains intra- and cross-dataset; p.12, Table 6: speed vs number of models
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 4 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper24_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper24_outputs.zip`: everything, with the weights
   - `paper24_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check:** not possible for this paper (split lists are public, but most of its 12 datasets need a request).

## 5. What to check and learn
- Best single model vs best fusion, both chosen on VALIDATION and reported on test
- Which pair or triple gives most of the gain for the least time?
- The "oracle" (any model right): the ceiling for fusion
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper24_results.zip` and `results_vs_paper.md` into `weights/paper24/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper24_outputs.zip` to `498R / weights / paper24 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper24_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- Choose the final OCR: a single model or a 2-3 model fusion, by plate exact vs ms per plate
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
