# src/ — code to run each paper again, one folder per member

Each `src/<Member>/paperNN/` folder holds the **Kaggle kit** for one paper:

| File | What it is |
|---|---|
| `START_HERE.md` | Start-to-end guide: what to read, how to run, what to check, what to upload |
| `Kaggle_Training_Guide.md` | Kaggle settings, run time, what each job in the notebook does |
| `Original_Check_Guide.md` | (some papers) how to test the method on the paper's own data |
| `kaggle/paperNN_kaggle.ipynb` | **The notebook to run** on Kaggle (GPU T4 x2, Internet on). It downloads the 498R dataset v1.1 by itself |
| `kaggle/paperNN_original_check.ipynb` | (some papers) the original-paper check notebook |
| `kaggle/pNN_*.py`, `kaggle/cNN_*.py` | Code written for this paper |
| `kaggle/common.py`, `det.py`, `rec.py`, `seq.py`, `ved.py`, `metrics.py`, `vocab.py`, `report.py` | Shared kit modules (dataset download, YOLO training, OCR readers, metrics, Bangla plate vocabulary, results packing) |

## How to run one

1. Open [kaggle.com](https://www.kaggle.com) → **Create → New Notebook → File → Import Notebook** and upload
   `kaggle/paperNN_kaggle.ipynb`.
2. Settings: **Accelerator GPU T4 x2**, **Internet On**.
3. Run once with `FAST = True` (a quick test of every cell), then set `FAST = False` and
   **Save Version → Save & Run All**.
4. When it finishes, download the outputs and put them where [CONTRIBUTING.md](../CONTRIBUTING.md) says.

The `.py` files are copies of the shared kit, so each paper folder is complete on its own.

## What is here now

| Member | Papers |
|---|---|
| [MD.Rokib.Hasan.Oli](MD.Rokib.Hasan.Oli/) | [paper02](MD.Rokib.Hasan.Oli/paper02/), [paper06](MD.Rokib.Hasan.Oli/paper06/), [paper08](MD.Rokib.Hasan.Oli/paper08/), [paper23](MD.Rokib.Hasan.Oli/paper23/), [paper24](MD.Rokib.Hasan.Oli/paper24/) |
| Md.Yousuf · Md.Rakibul.Hasan · Tashfia.Adiba.Zaman.Nizhum | not uploaded yet: add your `kaggle/` folder and guides in the same layout |
