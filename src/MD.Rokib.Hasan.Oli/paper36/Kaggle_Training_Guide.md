# Training paper #36 on Kaggle GPU T4 x2: step by step

*Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)*

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper36_kaggle.ipynb` | **The only file you upload.** It contains all the code below and runs everything. |
| `common.py`, `vocab.py`, `metrics.py`, `report.py` | shared helpers: dataset finder, Bangla plate tokens, OCR scores (CER, exact plate), results table |
| `det.py` | trains + tests one detector (Ultralytics), plate counts by size, speed at batch 1 |
| `rec.py` | character-level YOLO OCR: trains, reads every test plate, scores it |
| `seq.py`, `ved.py` | sequence OCR (CRNN / ViTSTR) and transformer OCR (VisionEncoderDecoder) |
| `p36_sahi.py` | SF slice dataset; FI / SAHI / SAHI+FI / SAHI+FI with cut-off slice boxes dropped, with the sahi package; AP50 by size, plates, e2e, per-plate files |

**What it runs (two jobs at the same time, one per T4):**

| Run | GPU | What |
|---|---|---|
| `sf_slices` | GPU CPU | SF training set: all train photos + 640 x 640 slices (20 % overlap) of every photo > 800 px |
| `det_v8s_640` | GPU 0 | YOLOv8s at 640 px, 60 epochs (the usual detector, also used by SAHI) |
| `crnn` | GPU 0 (after) | CRNN plate reader, 60 epochs (reads the e2e photos) |
| `det_v8s_sf` | GPU 1 | YOLOv8s, slicing aided fine-tuning: 8,574 photos + ~11 k slices, 26 epochs (= the steps of 60 on photos) |
| `det_v8s_1280` | GPU 1 (after) | YOLOv8s at 1280 px, 20 epochs, batch 8 (bigger input instead of slicing) |
| `sahi` | GPU 0 (after) | FI / SAHI / SAHI+FI / SAHI+FI cut-off boxes dropped (640 and SF models), FI (1280 model): AP50 by size, plates, ms, e2e, per_plate.csv |

Time: about **7.5 h** for the full run (Save & Run All). Quick test: about 10–15 min.

---

## 0. One-time setup
1. **Kaggle account:** Settings → **verify your phone number** (needed for GPU and internet).
2. **The dataset: nothing to do.** With Internet On, the notebook downloads `BanglaLPR_AllInOne_v1_1.zip` (dataset v1.1)
   from the team Drive folder "All In One 1.1" (https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl) into `/tmp`
   by itself (about 3-6 min; it is not saved as notebook output).
   *Optional, faster and safe from Drive download limits:* one teammate makes a Kaggle dataset from that zip (Create →
   New Dataset → title **498r-bangla-lpr-all-in-one** → Private) and shares it. Then everyone adds it as input and
   the notebook uses it instead of downloading.

## 1. Import the notebook
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper36_kaggle.ipynb`.
2. Right panel → **Session options**: **Accelerator: GPU T4 x2** (not TPU: all code here is PyTorch/CUDA),
   **Internet: On**, Persistence: Files only.
3. *(Optional)* **Add Input → Datasets →** `498r-bangla-lpr-all-in-one`, if the team made it (otherwise the
   dataset is downloaded automatically).

## 2. Quick test first (≈ 10–15 min)
In the first code cell set `FAST = True`, then **Run All** in the interactive session. Check:

| Cell | You should see |
|---|---|
| 0. Setup | two `Tesla T4` lines; pip finishes without red errors |
| 2. Dataset | `dataset: /kaggle/input/...` (attached) or `downloading ... dataset ready: /tmp/498r_data/...`, then the counts |
| 3. Runs | `[done] ... code=0` for every job (a 1-epoch run) |
| last cell | `packed /kaggle/working/paper36_outputs.zip` |

If a job ends with `code=1`, open `/kaggle/working/logs/<run>.log` (right panel → Output → logs) and fix it before
the long run. Common fixes:

| Error | Fix |
|---|---|
| `No GPU` / assertion in setup | Accelerator is not GPU T4 x2 |
| `dataset_manifest.json not found` / `Could not download the dataset` | Internet is off, or Drive's daily download limit was reached: attach the Kaggle dataset instead (step 1.3) |
| `CUDA out of memory` | lower the batch for that run (`--batch` in its job line), or set `G1 = G0` to run jobs one after another |
| pip / download error | Internet is off or your phone is not verified |
| `[time cap] ... stopping` in a log | the safety cap stopped training early (slow session); the results are still
written. Note the epochs actually run (`epochs_run` in results.json) in the settings table |
| `STOP: the slice training set was not made` | read the error printed in cell 3 (disk space: the ~11 k slices need ~1 GB in /tmp) |
| `ModuleNotFoundError: sahi` | the pip line in cell 0 failed: Internet on, run cell 0 again |

## 3. The real run: Save Version → Save & Run All (Commit)
Set `FAST = False` in the first cell → **Save Version → Save & Run All (Commit)**. The run keeps going after you close
the tab, and everything in `/kaggle/working` is kept as the version's **Output**. (An interactive session loses its
files when it stops.)

**How long is right?** The full run takes about **7.5 h**. In the quick test a training job ends after 1-3 min
(1 epoch on ~3 % of the images); that is normal there. In the full run, a training job that ends after a few minutes
has FAILED: look for `----- <run> failed` under the cell and open `logs/<run>.log`.

## 4. Download and store the outputs
From the finished version → **Output** tab:

| File | Where it goes |
|---|---|
| `paper36_results.zip` (results.json, args.yaml, logs, predictions, `results_vs_paper.md`, `code/`; **no model files**) | team GitHub: [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper36/<your-name>/` (Add file → Upload files) |
| small `best.pt` files (YOLO-n/s, CRNN: under 25 MB) | the same GitHub folder (web upload limit 25 MB per file, git push 100 MB) |
| `paper36_outputs.zip` (everything, with all weights) | team Drive: `498R / weights / paper36 / <date> /` |

The notebook cannot write to GitHub or Drive itself: download the files from the Output tab and upload them. Big
weights (transformer OCR, VLM LoRA, YOLO-m/l) stay on Drive, or use Git LFS. **Never put the dataset on GitHub.**

## 5. Settings to write down
They are saved automatically in every `results.json` / `args.yaml` (and in `environment.json`). The short version is in
the notes file, section "Settings I trained with".

## 6. Compare with the paper and with teammates
Put the downloaded zip in `D:\498R\docs\results\` and run:
```
python add_results.py paper36_results.zip --member "<your name>"
```
It adds one row per run to `experiments.csv` and rewrites `results_table.md` (ours vs. the paper, ours vs. teammates,
and a separate part for numbers that are not comparable). Copy the main numbers into the notes file too.

**Reading the numbers:** three ways to see far plates bigger: slices at test time (SAHI), slices at training time (SF) and simply a bigger input (1280 px). Compare small-plate AP50 / found % against ms per photo; if 1280 FI matches SAHI + FI at lower cost, slicing is not worth it for plates. With the paper's merge (NMS IoU 0.5) slice edges leave boxes on parts of large plates; the 'cut-off slice boxes dropped' rows show the merge without that problem (v1.1 run: AP50 91.6 -> 97.5 for the 640 model).


## Checklist for the board / weekly report
- [ ] Notes file filled (Problem · Method · Data · Results with pages · Limits · For us · Code · BibTeX)
- [ ] Literature-table row pasted into the team table
- [ ] Code link + short note on the Research board
- [ ] Quick test passed; full run committed; `paper36_outputs.zip` in the team Drive
- [ ] `add_results.py` run; results table + notes updated with our numbers, FPS and training time
