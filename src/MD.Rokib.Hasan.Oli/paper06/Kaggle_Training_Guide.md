# Training paper #06 on Kaggle GPU T4 x2: step by step

*YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles*

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper06_kaggle.ipynb` | **The only file you upload.** It contains all the code below and runs everything. |
| `common.py`, `vocab.py`, `metrics.py`, `report.py` | shared helpers: dataset finder, Bangla plate tokens, OCR scores (CER, exact plate), results table |
| `det.py` | trains + tests one detector (Ultralytics), plate counts by size, speed at batch 1 |
| `rec.py` | character-level YOLO OCR: trains, reads every test plate, scores it |
| `seq.py`, `ved.py` | sequence OCR (CRNN / ViTSTR) and transformer OCR (VisionEncoderDecoder) |
| `p06_easyocr.py` | EasyOCR (bn) on the OCR test crops, by condition, and the whole system on e2e_test photos |

**What it runs (two jobs at the same time, one per T4):**

| Run | GPU | What |
|---|---|---|
| `det_v8n` | GPU 0 | YOLOv8n (COCO), 120 epochs, 640 px, batch 16, human boxes |
| `det_v8s` | GPU 1 | YOLOv8s (COCO), 120 epochs, 640 px, batch 16, human boxes |
| `easyocr_crops` | GPU 0 (after) | stock EasyOCR bn on the 651 OCR test crops (raw + preprocessed), per condition |
| `e2e_v8n / e2e_v8s` | GPU 1 (after) | whole photos: detector -> best plate -> EasyOCR (204 e2e_test photos) |

Time: about **3.5 h** for the full run (Save & Run All). Quick test: about 10–15 min.

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
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper06_kaggle.ipynb`.
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
| last cell | `packed /kaggle/working/paper06_outputs.zip` |

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
| `easyocr` download error in cell 4 | Internet is off or GitHub is busy: run cell 4 again |

## 3. The real run: Save Version → Save & Run All (Commit)
Set `FAST = False` in the first cell → **Save Version → Save & Run All (Commit)**. The run keeps going after you close
the tab, and everything in `/kaggle/working` is kept as the version's **Output**. (An interactive session loses its
files when it stops.)

**How long is right?** The full run takes about **3.5 h**. In the quick test a training job ends after 1-3 min
(1 epoch on ~3 % of the images); that is normal there. In the full run, a training job that ends after a few minutes
has FAILED: look for `----- <run> failed` under the cell and open `logs/<run>.log`.

## 4. Download and store the outputs
From the finished version → **Output** tab:

| File | Where it goes |
|---|---|
| `paper06_results.zip` (results.json, args.yaml, logs, predictions, `results_vs_paper.md`, `code/`; **no model files**) | team GitHub: [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper06/<your-name>/` (Add file → Upload files) |
| small `best.pt` files (YOLO-n/s, CRNN: under 25 MB) | the same GitHub folder (web upload limit 25 MB per file, git push 100 MB) |
| `paper06_outputs.zip` (everything, with all weights) | team Drive: `498R / weights / paper06 / <date> /` |

The notebook cannot write to GitHub or Drive itself: download the files from the Output tab and upload them. Big
weights (transformer OCR, VLM LoRA, YOLO-m/l) stay on Drive, or use Git LFS. **Never put the dataset on GitHub.**

## 5. Settings to write down
They are saved automatically in every `results.json` / `args.yaml` (and in `environment.json`). The short version is in
the notes file, section "Settings I trained with".

## 6. Compare with the paper and with teammates
Put the downloaded zip in `D:\498R\docs\results\` and run:
```
python add_results.py paper06_results.zip --member "<your name>"
```
It adds one row per run to `experiments.csv` and rewrites `results_table.md` (ours vs. the paper, ours vs. teammates,
and a separate part for numbers that are not comparable). Copy the main numbers into the notes file too.

**Reading the numbers:** compare our mAP50 with the paper's 94.0 % (its "mAP" matches mAP@50 in Fig. 6) and our *plates found %* with its undefined "detection accuracy" 94.8 %. Its 92.7 % character accuracy is undefined and on its own plates: put our EasyOCR-alone char accuracy next to it and say so.


## Checklist for the board / weekly report
- [ ] Notes file filled (Problem · Method · Data · Results with pages · Limits · For us · Code · BibTeX)
- [ ] Literature-table row pasted into the team table
- [ ] Code link + short note on the Research board
- [ ] Quick test passed; full run committed; `paper06_outputs.zip` in the team Drive
- [ ] `add_results.py` run; results table + notes updated with our numbers, FPS and training time
