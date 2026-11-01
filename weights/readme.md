# weights/ — trained models and results, one folder per paper and member

Each paper's results package lives in **`weights/paperNN/<Member>/`**. This is the main page for a paper: open its
`README.md` first.

| What goes in `weights/paperNN/<Member>/` | Required | Notes |
|---|---|---|
| `README.md` | yes | Model card: what was trained, main results, quick start, limits, file hashes |
| `results_vs_paper.md` | yes | Our numbers next to the paper's (made by the notebook) |
| `paperNN_results.zip` | yes | All results, logs, curves and code from the run, **without** model files (a few MB) |
| `best_*.pt` (or the model files the README names) | yes | Under **25 MB** for a web upload, under **100 MB** with git. Bigger → team Drive |
| `PaperNN_Study_Report.pdf` | if you made one | Full study report |
| small extra files (`*.json`) | optional | Extra checks that the README quotes |

**Not here:** the full `paperNN_outputs.zip` (40–250 MB, every `last.pt`) goes to the team Drive
`498R / weights / paperNN /`, or to [`output/`](../output/) with Git LFS.

## What is here now

| Paper | Member | Folder | Dataset run |
|---|---|---|---|
| #2 Cascaded YOLO | MD. Rokib Hasan Oli | [`paper02/MD.Rokib.Hasan.Oli`](paper02/MD.Rokib.Hasan.Oli/) | v1.1 |
| #3 Cloud hybrid YOLO + Haar | Md. Yousuf | [`paper03/Md.Yousuf`](paper03/Md.Yousuf/) | v1.0 (still valid: detection only) |
| #6 YOLOv8 + EasyOCR | MD. Rokib Hasan Oli | [`paper06/MD.Rokib.Hasan.Oli`](paper06/MD.Rokib.Hasan.Oli/) | v1.1 |
| #8 CNN + GFP-GAN | MD. Rokib Hasan Oli | [`paper08/MD.Rokib.Hasan.Oli`](paper08/MD.Rokib.Hasan.Oli/) | v1.1 |
| #20 Plate super-resolution | Md. Yousuf | [`paper20/Md.Yousuf`](paper20/Md.Yousuf/) | v1.0 (re-run on v1.1 needed) |
| #23 Near-duplicates | MD. Rokib Hasan Oli | [`paper23/MD.Rokib.Hasan.Oli`](paper23/MD.Rokib.Hasan.Oli/) | v1.1 |
| #24 Model fusion | MD. Rokib Hasan Oli | [`paper24/MD.Rokib.Hasan.Oli`](paper24/MD.Rokib.Hasan.Oli/) | v1.1 |
| #27 YOLOv10 | Md. Yousuf | [`paper27/Md.Yousuf`](paper27/Md.Yousuf/) | v1.0 (character models: re-run on v1.1) |
| #28 YOLOv9 | Md. Yousuf | [`paper28/Md.Yousuf`](paper28/Md.Yousuf/) | v1.0 (still valid: detection only) |

How to add a package: see [CONTRIBUTING.md](../CONTRIBUTING.md#3-where-each-file-goes).
