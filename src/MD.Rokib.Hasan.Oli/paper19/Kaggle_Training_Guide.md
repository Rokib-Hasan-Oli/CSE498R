# Training paper #19 on Kaggle GPU T4 x2: step by step

*LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification*

Files in `kaggle/`:

| File | What it is |
|---|---|
| `paper19_kaggle.ipynb` | **The only file you upload.** It contains all the code below and runs everything. |
| `common.py`, `vocab.py`, `metrics.py`, `report.py` | shared helpers: dataset finder, Bangla plate tokens, OCR scores (CER, exact plate), results table |
| `det.py` | trains + tests one detector (Ultralytics), plate counts by size, speed at batch 1 |
| `rec.py` | character-level YOLO OCR: trains, reads every test plate, scores it |
| `seq.py`, `ved.py` | sequence OCR (CRNN / ViTSTR) and transformer OCR (VisionEncoderDecoder) |
| `p19_legibility.py` | cross-fitted readers, proxy legibility labels, ResNet-50 + EMA loss, the reject-or-read gate |

**What it runs (two jobs at the same time, one per T4):**

| Run | GPU | What |
|---|---|---|
| `readers_crnn` | GPU 0 | CRNN x 3 (half A, half B, all train), 60 epochs: cross-fitted readings + the deployment reader |
| `readers_slot` | GPU 1 | slot CNN x 2 (half A, half B), 100 epochs |
| `legibility_labels` | GPU CPU | 4-class OCR-difficulty labels for every OCR crop |
| `cls_b4_ema / cls_qf_ema` | GPU 0 | ResNet-50, 4 classes / Quality Filter, EMA loss (paper), Adam 1e-5, batch 64 |
| `cls_b4_ce / cls_qf_ce` | GPU 1 | the same with plain cross-entropy (the Table IV / V ablation) |
| `gate` | GPU CPU | reject-or-read: accepted plates exact at 100-50 % coverage, classifier vs OCR confidence |

Time: about **2 h** for the full run (Save & Run All). Quick test: about 10–15 min.

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
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/paper19_kaggle.ipynb`.
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
| last cell | `packed /kaggle/working/paper19_outputs.zip` |

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
| `labels.csv` missing in cell 5 | a reader job failed in cell 3: see `----- readers_... failed` |

## 3. The real run: Save Version → Save & Run All (Commit)
Set `FAST = False` in the first cell → **Save Version → Save & Run All (Commit)**. The run keeps going after you close
the tab, and everything in `/kaggle/working` is kept as the version's **Output**. (An interactive session loses its
files when it stops.)

**How long is right?** The full run takes about **2 h**. In the quick test a training job ends after 1-3 min
(1 epoch on ~3 % of the images); that is normal there. In the full run, a training job that ends after a few minutes
has FAILED: look for `----- <run> failed` under the cell and open `logs/<run>.log`.

## 4. Download and store the outputs
From the finished version → **Output** tab:

| File | Where it goes |
|---|---|
| `paper19_results.zip` (results.json, args.yaml, logs, predictions, `results_vs_paper.md`, `code/`; **no model files**) | team GitHub: [weights/](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) → `weights/paper19/<your-name>/` (Add file → Upload files) |
| small `best.pt` files (YOLO-n/s, CRNN: under 25 MB) | the same GitHub folder (web upload limit 25 MB per file, git push 100 MB) |
| `paper19_outputs.zip` (everything, with all weights) | team Drive: `498R / weights / paper19 / <date> /` |

The notebook cannot write to GitHub or Drive itself: download the files from the Output tab and upload them. Big
weights (transformer OCR, VLM LoRA, YOLO-m/l) stay on Drive, or use Git LFS. **Never put the dataset on GitHub.**

## 5. Settings to write down
They are saved automatically in every `results.json` / `args.yaml` (and in `environment.json`). The short version is in
the notes file, section "Settings I trained with".

## 6. Compare with the paper and with teammates
Put the downloaded zip in `D:\498R\docs\results\` and run:
```
python add_results.py paper19_results.zip --member "<your name>"
```
It adds one row per run to `experiments.csv` and rewrites `results_table.md` (ours vs. the paper, ours vs. teammates,
and a separate part for numbers that are not comparable). Copy the main numbers into the notes file too.

**Reading the numbers:** a legibility gate is worth adding to the pipeline only if, at the same coverage (say 80 % of plates read), the accepted plates are more often right than with the OCR's own confidence. If `gain_at_80%_vs_ocr_confidence_pp` is about 0, the OCR confidence is enough.


## Checklist for the board / weekly report
- [ ] Notes file filled (Problem · Method · Data · Results with pages · Limits · For us · Code · BibTeX)
- [ ] Literature-table row pasted into the team table
- [ ] Code link + short note on the Research board
- [ ] Quick test passed; full run committed; `paper19_outputs.zip` in the team Drive
- [ ] `add_results.py` run; results table + notes updated with our numbers, FPS and training time
