# Paper #2: start-to-end guide

**Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition**

| What | Where |
|---|---|
| Notes (read these with the paper) | [2_Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition.md](../../../report/MD.Rokib.Hasan.Oli/paper02/paper02_notes.md) |
| Kaggle notebook | `kaggle/paper02_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | **Trained on v1.0 (before 21 Oct 2026)? Re-run on v1.1:** the character model (the detector numbers will repeat). |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://doi.org/10.1371/journal.pone.0355858 · data, weights: https://github.com/prosensarker4/BLPD
- Team copy (PDF): `D:\498R\Papers\6_1 Bangla_Bangladeshi LPR (core)\02_2025_Cascaded YOLO Bangladeshi LP Identification and Number Recognition.pdf`

## 2. Why this paper matters for 498R
Our main Bangladeshi baseline. Data, split and weights are public, and its OCR-free character detector (every Bangla word, letter and digit is a YOLO class) is the design our dataset follows.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.11-17: the two datasets (2,140 plate photos, 7,217 character crops) and the 105 class names (some duplicated)
   - p.17-18, p.24: training settings (YOLO12s LPD 100 epochs, YOLO12n LPNR 70 epochs, 640, batch 16, Colab T4)
   - p.21-25, Tables 6-9: plate and character detection (P, R, mAP50, mAP50-95)
   - p.34-36, Tables 10-13: the 150 external photos (95.33 % exact) and latency on GPU / CPU / Jetson
   - p.37-38, Table 15: the EasyOCR-type baseline (15 %), which shows why character detection wins
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 7 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper02_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper02_outputs.zip`: everything, with the weights
   - `paper02_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check (the paper's own data):** `kaggle/paper02_original_check.ipynb`, about 7.5 h. Steps in [Original_Check_Guide.md](Original_Check_Guide.md). Add the Kaggle input it names.

## 5. What to check and learn
- Original check (already done locally): our retraining matches Tables 6, 7, 9; the cascade gets 90.67 % vs 95.33 %
- Main notebook: the same cascade on OUR data and e2e_test (BLPD external photos + NightLPBD night photos)
- Does the vehicle stage help, and what does it cost? Compare "with_vehicle_stage" with "plate_detector_only"
- Remember the audit: 44 % of BLPD's own test photos have a near-duplicate in its train split
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper02_results.zip` and `results_vs_paper.md` into `weights/paper02/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper02_outputs.zip` to `498R / weights / paper02 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper02_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- Use this cascade as the baseline every other method must beat (results table part 2)
- Run #23 to show how much the leakage inflated the paper's numbers
- Use the per-condition table to see where character detection fails (night, small plates)
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
