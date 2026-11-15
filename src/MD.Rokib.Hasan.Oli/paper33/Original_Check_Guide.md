# Original-paper check for paper #33: step by step

*YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection*

**What is different from the main notebook (`paper33_kaggle.ipynb`):** that one runs the paper's method on OUR data
(498R All-in-One). This one runs on the **paper's own data / split / weights**, so its table is a direct check of the
numbers printed in the paper (same data → comparable).

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper33_original_check.ipynb` | **The only file you upload.** All code is inside. |
| `coco_check.py` | COCO val2017 download, pycocotools mAP, latency loop (shared with #26-#30) |
| `c33_coco.py` | YOLO26 one-to-many + NMS vs NMS-free head: COCO mAP, TensorRT FP16, CPU ONNX |

**What it runs:**

| Run | GPU | What |
|---|---|---|
| `coco_gpu0` | GPU 0 | yolo26n, yolo26s: both heads |
| `coco_gpu1` | GPU 0 (after) | yolo26m (both heads), yolo11n |

Time: about **2 h** for the full run. Quick test (`FAST = True`): about 10–20 min.

## Steps
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper33_original_check.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2**, **Internet On**, Persistence: Files only.
3. **Inputs:**
   - No extra input: the notebook downloads the paper's data itself (Internet must be On).
4. Set `FAST = True` in the first cell → **Run All** (quick test). Every job should end with `[done] ... code=0`.
   In the quick test each training job takes only about 1-3 min (1 epoch on ~3 % of the images): that is expected,
   and its numbers mean nothing.
5. Set `FAST = False` → **Save Version → Save & Run All (Commit)**. This takes about **2 h**. If it finishes
   in a few minutes, it failed: look for `----- <run> failed` under the cells and open `logs/<run>.log` in the
   Output tab. The table at the end also lists the runs without results.
6. From the finished version's **Output** tab:
   - `paper33_check_results.zip` (no model files) → team GitHub
     [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper33/original_check/<your-name>/`
   - `paper33_check_outputs.zip` (with weights) → team Drive `498R / weights / paper33 / original_check /`
7. Put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper33_check_results.zip --member "<your name>"`: the table goes into the notes
   (section "Original-paper check") and into the team results table (part 1b, "the papers' own data").

## Common problems

| Problem | Fix |
|---|---|
| no GPU / assertion in setup | Accelerator must be GPU T4 x2 |
| download error | Internet is off, or the source is busy: run the cell again |
| `[time cap] ... stopping` in a log | the safety cap stopped training early; results are still written |
