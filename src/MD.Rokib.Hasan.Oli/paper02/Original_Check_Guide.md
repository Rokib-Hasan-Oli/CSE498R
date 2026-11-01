# Original-paper check for paper #02: step by step

*Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition*

**What is different from the main notebook (`paper02_kaggle.ipynb`):** that one runs the paper's method on OUR data
(498R All-in-One). This one runs on the **paper's own data / split / weights**, so its table is a direct check of the
numbers printed in the paper (same data → comparable).

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper02_original_check.ipynb` | **The only file you upload.** All code is inside. |
| `c02_blpd.py` | yaml for the BLPD split, the authors' weights on val/test, the 150-photo external test |
| `p02_cascade.py` | the vehicle -> plate -> character cascade (shared with the main kit) |

**What it runs:**

| Run | GPU | What |
|---|---|---|
| `official` | GPU 0 | authors' LPD + LPNR on their val/test split, cascade on the 150 external photos |
| `lpd_repro` | GPU 0 (after) | YOLO12s LPD retrained, 100 epochs (paper settings) |
| `lpnr_repro` | GPU 1 | YOLO12n LPNR retrained, 70 epochs, 640 px |
| `e2e_repro` | GPU 0 (after) | cascade with the retrained models on the 150 external photos |

Time: about **7.5 h** for the full run. Quick test (`FAST = True`): about 10–20 min.

## Steps
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper02_original_check.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2**, **Internet On**, Persistence: Files only.
3. **Inputs:**
   - *Recommended:* **Add Input → Datasets →** search `ferozahmedds/bangladeshi-license-plate-detection-and-recognition` and add it.
   - Without it, cell 3 downloads the same files from the authors' GitHub (pinned commit, about 0.6 GB, Internet On). If neither works, cell 3 stops the run with an error (it no longer finishes in a minute with nothing trained).
4. Set `FAST = True` in the first cell → **Run All** (quick test). Every job should end with `[done] ... code=0`.
   In the quick test each training job takes only about 1-3 min (1 epoch on ~3 % of the images): that is expected,
   and its numbers mean nothing.
5. Set `FAST = False` → **Save Version → Save & Run All (Commit)**. This takes about **7.5 h**. If it finishes
   in a few minutes, it failed: look for `----- <run> failed` under the cells and open `logs/<run>.log` in the
   Output tab. The table at the end also lists the runs without results.
6. From the finished version's **Output** tab:
   - `paper02_check_results.zip` (no model files) → team GitHub
     [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper02/original_check/<your-name>/`
   - `paper02_check_outputs.zip` (with weights) → team Drive `498R / weights / paper02 / original_check /`
7. Put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper02_check_results.zip --member "<your name>"`: the table goes into the notes
   (section "Original-paper check") and into the team results table (part 1b, "the papers' own data").

## Common problems

| Problem | Fix |
|---|---|
| no GPU / assertion in setup | Accelerator must be GPU T4 x2 |
| download error | Internet is off, or the source is busy: run the cell again |
| `[time cap] ... stopping` in a log | the safety cap stopped training early; results are still written |
| `STOP: the BLPD data was not found` in cell 3 | add the Kaggle dataset ferozahmedds/bangladeshi-license-plate-detection-and-recognition, or turn Internet on (GitHub download) |
| the whole run ends in a few minutes | a job failed: see `----- <run> failed` under cell 4 and `logs/<run>.log` |
