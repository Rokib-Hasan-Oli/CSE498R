# Paper #35: start-to-end guide

**HVI: A New Color Space for Low-light Image Enhancement (HVI-CIDNet)**

| What | Where |
|---|---|
| Notes (read these with the paper) | [35_HVI - A New Color Space for Low-light Image Enhancement.md](../../../report/MD.Rokib.Hasan.Oli/paper35/paper35_notes.md) |
| Kaggle notebook | `kaggle/paper35_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | New paper (added 25 Oct 2026): there are no v1.0 results; the notebook runs on v1.1. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2502.20272 · CVPR 2025 · code: https://github.com/Fediory/HVI-CIDNet · weights: https://huggingface.co/Fediory
- Team copy (PDF): `D:\498R\Papers\6_6 Added papers (new5)\35_2025_HVI Color Space for Low-light Image Enhancement.pdf`

## 2. Why this paper matters for 498R
Night plates are our weakest case. HVI-CIDNet is a small (1.88 M) state-of-the-art low-light enhancer with public weights: can it be put in front of our detector and reader?

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.1-3: why sRGB and HSV cause colour / black artifacts in dark images
   - p.3-5: the HVI colour space and CIDNet (two branches, cross-attention)
   - p.5: training settings; p.6, Table 1: LOL results (note the GT-mean trick for LOLv1)
   - p.7, Table 3: the HVI plug-in and GPU time; p.7-8, Table 4: ablations
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 4 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper35_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper35_outputs.zip`: everything, with the weights
   - `paper35_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check (the paper's own data):** `kaggle/paper35_original_check.ipynb`, about 0.5 h. Steps in [Original_Check_Guide.md](Original_Check_Guide.md). It downloads its own data.

## 5. What to check and learn
- Night photos: plates found / mAP50 with and without enhancement; and the whole test with a dark-photo gate
- Night crops: plate exact with and without enhancement
- Is it better than plain CLAHE / gamma, and what does it cost in ms?
- Original check: the released weights on LOL (PSNR / SSIM, with and without GT mean)
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper35_results.zip` and `results_vs_paper.md` into `weights/paper35/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper35_outputs.zip` to `498R / weights / paper35 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper35_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- If it helps at night and the gate keeps day photos safe, add "dark gate -> HVI-CIDNet" to the pipeline; otherwise report that detection / OCR trained on our data does not need it
- Compare with #20 (super-resolution) and #8 (GFP-GAN): all three ask "does restoration help reading?"
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
