# Paper #6: start-to-end guide

**YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles**

| What | Where |
|---|---|
| Notes (read these with the paper) | [6_YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles.md](../../../report/MD.Rokib.Hasan.Oli/paper06/paper06_notes.md) |
| Kaggle notebook | `kaggle/paper06_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | **Trained on v1.0 (before 21 Oct 2026)? Re-run on v1.1:** the EasyOCR test on the crops (no training; the detectors will repeat). |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://doi.org/10.22541/au.173748276.69471512/v1 · Authorea preprint, not peer-reviewed (Cloudflare check: open it in a browser)
- Team copy (PDF): `D:\498R\Papers\6_1 Bangla_Bangladeshi LPR (core)\06_2025_YOLOv8-Based LPR for Bangladeshi Vehicles.pdf`

## 2. Why this paper matters for 498R
The simplest full baseline for our plan: a COCO YOLOv8 plate detector plus stock EasyOCR (Bangla). Every other method must beat it, and it shows how far a general OCR is from a plate-specific one.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.6-7: the data (2,600 own photos, boxes drawn in Roboflow) and the split. Note that the augmentation was done BEFORE the split (p.7): copies of a photo can sit in train and test
   - p.8: the pipeline (YOLOv8 box → crop → "preprocessing" → EasyOCR bn); no model size or settings are given
   - p.9, Table 1 and p.11, Fig. 6: P / R / mAP (its "mAP" 94.0 matches mAP@50; mAP@50-95 is near 0.59)
   - p.11-12, Table 2: "character accuracy" 92.7 % and by case (clean / low-res / damaged / weather); never defined
   - p.12, Table 3: 78 FPS (GPU not named); the other rows are copied from other papers, on other data
   - Watch out: the text says TensorFlow, anchors and "object loss", none of which fits YOLOv8 (see the notes)
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 3.5 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper06_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper06_outputs.zip`: everything, with the weights
   - `paper06_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check:** not possible for this paper (no code and no data published (its own 2,600 Roboflow photos are not public; Authorea preprint)).

## 5. What to check and learn
- YOLOv8n vs YOLOv8s on our human boxes: mAP50 / mAP50-95 vs the paper, plates found %, FPS (T4 TensorRT FP16, PyTorch FP32)
- EasyOCR alone on our human crops: char accuracy vs the paper's 92.7 %, and plate exact (the strict number)
- Does the "preprocessing" (4x upscale + grey + CLAHE) help EasyOCR?
- By condition: clear vs small / night / blurry / fog / rain, like its Table 2
- The whole system on e2e photos: how much is lost between "plate found" and "plate read"?
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper06_results.zip` and `results_vs_paper.md` into `weights/paper06/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper06_outputs.zip` to `498R / weights / paper06 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper06_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- Use these rows as the baseline in the results table: #2 (character detection), #24 (fusion) and #31 (PARSeq) must beat EasyOCR clearly
- If YOLOv8n ≈ YOLOv8s on mAP50, take the faster one for the demo detector
- Write in the notes which of the paper's numbers we could not reproduce, and why (its data, its split)
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
