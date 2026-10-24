# Paper #23: start-to-end guide

**Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition**

| What | Where |
|---|---|
| Notes (read these with the paper) | [23_Do We Train on Test Data - The Impact of Near-Duplicates on License Plate Recognition.md](../../../report/MD.Rokib.Hasan.Oli/paper23/paper23_notes.md) |
| Kaggle notebook | `kaggle/paper23_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | **Trained on v1.0 (before 21 Oct 2026)? Re-run on v1.1:** everything (the fair reading test is now KUET's 194 published test crops). |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2304.04653 · IEEE: https://doi.org/10.1109/IJCNN54540.2023.10191584
- Team copy (PDF): `D:\498R\Papers\6_3 General LPR state of the art\23_2023_Do We Train on Test Data Near-Duplicates LPR.pdf`

## 2. Why this paper matters for 498R
It shows that near-duplicate photos in train and test inflate LPR scores. This is why our dataset uses a grouped split; we must prove it on our data.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.2-3: what counts as a near-duplicate and how the fair splits are made
   - p.4-5, Tables I-III: the drop in accuracy on the fair splits (AOLP, CCPD)
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 2.5 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper23_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper23_outputs.zip`: everything, with the weights
   - `paper23_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check:** not possible for this paper (split lists are public, but AOLP needs a request and CCPD is 12 GB+ of Chinese plates).

## 5. What to check and learn
- Leaky (KUET's own split with copies) vs fair (our grouped split): the gap for the detector and the OCR
- Scores on the clean vs duplicated parts of the published test set
- Relative error increase, as in the paper
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper23_results.zip` and `results_vs_paper.md` into `weights/paper23/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper23_outputs.zip` to `498R / weights / paper23 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper23_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- Put the gap in the report's method section: it justifies our split
- Tell teammates to never re-split the dataset randomly
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
