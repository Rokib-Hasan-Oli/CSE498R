# Paper #34: start-to-end guide

**SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition**

| What | Where |
|---|---|
| Notes (read these with the paper) | [34_SVTRv2 - CTC Beats Encoder-Decoder Models in Scene Text Recognition.md](../../../report/MD.Rokib.Hasan.Oli/paper34/paper34_notes.md) |
| Kaggle notebook | `kaggle/paper34_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | New paper (added 25 Oct 2026): there are no v1.0 results; the notebook runs on v1.1. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2411.15858 · ICCV 2025 · code: https://github.com/Topdu/OpenOCR
- Team copy (PDF): `D:\498R\Papers\6_6 Added papers (new5)\34_2025_SVTRv2 CTC Beats Encoder-Decoder Models in STR.pdf`

## 2. Why this paper matters for 498R
Our fastest readers are CTC models (CRNN), our most accurate ones attention models (PARSeq #31). SVTRv2 claims a CTC model can be both, and its multi-size resizing suits two-line Bangla plates.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.1-2: why CTC models fail on irregular text and how SVTRv2 fixes it
   - p.3, Fig. 2 and section 3.1: multi-size resizing (64x64, 48x96, 40x112, 32xW)
   - p.4-5: the feature rearrangement module and the semantic guidance module (training only)
   - p.5-6, Tables 1-2: the ablations (MSR, FRM, SGM)
   - p.7, Table 3: 24 readers on the same data: accuracy, size, FPS
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 3.5 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper34_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper34_outputs.zip`: everything, with the weights
   - `paper34_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check:** not possible for this paper (possible but not built: the released SVTRv2 weights (Google Drive) and the benchmark LMDBs (the #31 check downloads them) are public, but the authors' pickled checkpoints cannot be loaded on our test PC, so no untested notebook was shipped; the main kit uses their model code).

## 5. What to check and learn
- SVTRv2-T/S vs CRNN vs PARSeq (#31) on our plates: plate exact AND ms per plate
- MSR vs fixed 32x128: does the two-line (48x96) group gain?
- English pre-training vs from scratch
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper34_results.zip` and `results_vs_paper.md` into `weights/paper34/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper34_outputs.zip` to `498R / weights / paper34 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper34_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- If SVTRv2 matches PARSeq at CRNN speed, make it the main reader and add it to the #24 fusion (its ocr_preds_*.csv files are ready for that)
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
