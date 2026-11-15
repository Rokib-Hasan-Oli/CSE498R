# Paper #36: start-to-end guide

**Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)**

| What | Where |
|---|---|
| Notes (read these with the paper) | [36_SAHI - Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.md](../../../report/MD.Rokib.Hasan.Oli/paper36/paper36_notes.md) |
| Kaggle notebook | `kaggle/paper36_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | New paper (added 25 Oct 2026): there are no v1.0 results; the notebook runs on v1.1. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2202.06934 · ICIP 2022 · code: https://github.com/obss/sahi (pip sahi) · results: https://github.com/fcakyon/small-object-detection-benchmark
- Team copy (PDF): `D:\498R\Papers\6_6 Added papers (new5)\36_2022_SAHI Slicing Aided Hyper Inference Small Object Detection.pdf`

## 2. Why this paper matters for 498R
845 of our 1,153 test photos are 1280 px, but every detector kit reads them at 640 px, so far plates shrink to a few pixels. SAHI detects on overlapping 640 px slices (and can fine-tune on them): the cheapest way to see far plates bigger. It replaces #15, which asked the same question with a vehicle stage but whose Kaggle data was never found.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.1: the problem (small, far objects in high-resolution photos) and the claimed gains
   - p.2-3, Fig. 2: slicing aided fine-tuning (SF) and slicing aided hyper inference (SAHI), FI and the NMS merge
   - p.3, Tables 1-2: VisDrone and xView results for FCOS, VFNet, TOOD with FI / SAHI / SF / overlap (PO)
   - p.4: training settings, slice sizes (640 VisDrone, 400 xView), 25 % overlap, AP50 with max 500 boxes
   - Watch out: two abstract numbers do not match Table 1 (FCOS SAHI +6.8 vs +5.2 in the table, TOOD +14.5 vs +14.1); the check re-scores the authors' own result files
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 7.5 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper36_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper36_outputs.zip`: everything, with the weights
   - `paper36_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check (the paper's own data):** `kaggle/paper36_original_check.ipynb`, about 0.5 h. Steps in [Original_Check_Guide.md](Original_Check_Guide.md). It downloads its own data.

## 5. What to check and learn
- Original check: the authors' 20 VisDrone result files re-scored with pycocotools vs Table 1 and their eval.json
- Main kit: AP50 for small plates and plates found with FI vs SAHI vs SAHI + FI, and with slicing aided fine-tuning
- Does a 1280 px detector do the same job more cheaply than slicing? (ms per photo)
- Do slices add false alarms on plate-free photos, and does a better box change the e2e plate read?
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper36_results.zip` and `results_vs_paper.md` into `weights/paper36/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper36_outputs.zip` to `498R / weights / paper36 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper36_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- If SAHI (or 1280 px) finds clearly more small / far plates at an acceptable ms, use it for photos (not video)
- Compare with #20 (super-resolution) and #35 (enhancement): which pre-processing really helps far plates?
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
