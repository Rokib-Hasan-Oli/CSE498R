# Original-paper check for paper #08: step by step

*Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN*

**What is different from the main notebook (`paper08_kaggle.ipynb`):** that one runs the paper's method on OUR data
(498R All-in-One). This one runs on the **paper's own data / split / weights**, so its table is a direct check of the
numbers printed in the paper (same data → comparable).

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper08_original_check.ipynb` | **The only file you upload.** All code is inside. |
| `p08_train.py` | the paper's CNN and frozen ResNet-50 (PyTorch port of the authors' Keras notebooks) |
| `p08_plates.py` | the paper's plate pipeline (GFP-GAN = Real-ESRGAN x2, Otsu, contours) shared with the main kit |
| `c08_afrin.py` | Data.zip download + audit, clean re-test, synthetic plate lists and plate test, summary over seeds |

**What it runs:**

| Run | GPU | What |
|---|---|---|
| `cnn_s42 / s43 / s44` | GPU 0 | paper CNN 28x28, 40 epochs, batch 1, Adam 1e-4, 3 seeds |
| `resnet50_s42 / s43 / s44` | GPU 1 | frozen ResNet-50 150x150, 30 epochs, batch 1, Adam 1e-3, 3 seeds |
| `cnn32_s42, resnet50norm_s42` | GPU 0 / 1 (after) | CNN at 32x32 (paper text); ResNet-50 with ImageNet normalisation |
| `evalclean, plates_*` | GPU 0 / 1 (after) | clean-test re-score; plate test on the synthetic plates (if attached) |
| `ours_same_data` | GPU 0 (last) | mean ± std over the seeds, the table vs the paper |

Time: about **1 h** for the full run. Quick test (`FAST = True`): about 10–20 min.

## Steps
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper08_original_check.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2**, **Internet On**, Persistence: Files only.
3. **Inputs:**
   - No input is needed for steps 1-3: the notebook downloads the authors' `Data.zip` itself (Internet On).
   - *Optional, for the plate test (step 4):* **Add Input → Datasets →** search `furcifer/bangla-license-plates-synthetic` ("Bangla License Plates (Synthetic)") and add it.
4. Set `FAST = True` in the first cell → **Run All** (quick test). Every job should end with `[done] ... code=0`.
   In the quick test each training job takes only about 1-3 min (1 epoch on ~3 % of the images): that is expected,
   and its numbers mean nothing.
5. Set `FAST = False` → **Save Version → Save & Run All (Commit)**. This takes about **1 h**. If it finishes
   in a few minutes, it failed: look for `----- <run> failed` under the cells and open `logs/<run>.log` in the
   Output tab. The table at the end also lists the runs without results.
6. From the finished version's **Output** tab:
   - `paper08_check_results.zip` (no model files) → team GitHub
     [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper08/original_check/<your-name>/`
   - `paper08_check_outputs.zip` (with weights) → team Drive `498R / weights / paper08 / original_check /`
7. Put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper08_check_results.zip --member "<your name>"`: the table goes into the notes
   (section "Original-paper check") and into the team results table (part 1b, "the papers' own data").

## Common problems

| Problem | Fix |
|---|---|
| no GPU / assertion in setup | Accelerator must be GPU T4 x2 |
| download error | Internet is off, or the source is busy: run the cell again |
| `[time cap] ... stopping` in a log | the safety cap stopped training early; results are still written |
| `STOP: the authors' Data.zip could not be prepared` | Internet is off, or GitHub is busy: run cell 3 again |
| `plate test skipped` | add the optional Kaggle input furcifer/bangla-license-plates-synthetic |
