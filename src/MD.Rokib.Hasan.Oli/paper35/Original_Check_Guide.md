# Original-paper check for paper #35: step by step

*HVI: A New Color Space for Low-light Image Enhancement (HVI-CIDNet)*

**What is different from the main notebook (`paper35_kaggle.ipynb`):** that one runs the paper's method on OUR data
(498R All-in-One). This one runs on the **paper's own data / split / weights**, so its table is a direct check of the
numbers printed in the paper (same data → comparable).

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper35_original_check.ipynb` | **The only file you upload.** All code is inside. |
| `p35_lowlight.py` | HVI-CIDNet code + Hugging Face weights (shared with the main kit) |
| `c35_lol.py` | LOL test pairs from Hugging Face, the authors' PSNR / SSIM, LPIPS, GT-mean on / off |

**What it runs:**

| Run | GPU | What |
|---|---|---|
| `lolv1` | GPU 0 | LOLv1 eval15: LOLv1 weights without / with perceptual loss |
| `lolv2_real` | GPU 1 | LOLv2-Real test: best-PSNR and best-SSIM weights; the generalization weights |

Time: about **0.5 h** for the full run. Quick test (`FAST = True`): about 10–20 min.

## Steps
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper35_original_check.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2**, **Internet On**, Persistence: Files only.
3. **Inputs:**
   - No extra input: the notebook downloads the paper's data itself (Internet must be On).
4. Set `FAST = True` in the first cell → **Run All** (quick test). Every job should end with `[done] ... code=0`.
   In the quick test each training job takes only about 1-3 min (1 epoch on ~3 % of the images): that is expected,
   and its numbers mean nothing.
5. Set `FAST = False` → **Save Version → Save & Run All (Commit)**. This takes about **0.5 h**. If it finishes
   in a few minutes, it failed: look for `----- <run> failed` under the cells and open `logs/<run>.log` in the
   Output tab. The table at the end also lists the runs without results.
6. From the finished version's **Output** tab:
   - `paper35_check_results.zip` (no model files) → team GitHub
     [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper35/original_check/<your-name>/`
   - `paper35_check_outputs.zip` (with weights) → team Drive `498R / weights / paper35 / original_check /`
7. Put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper35_check_results.zip --member "<your name>"`: the table goes into the notes
   (section "Original-paper check") and into the team results table (part 1b, "the papers' own data").

## Common problems

| Problem | Fix |
|---|---|
| no GPU / assertion in setup | Accelerator must be GPU T4 x2 |
| download error | Internet is off, or the source is busy: run the cell again |
| `[time cap] ... stopping` in a log | the safety cap stopped training early; results are still written |
