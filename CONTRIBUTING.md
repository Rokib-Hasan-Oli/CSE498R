# How to add your work to this repo

This guide is for the four team members. Follow it after every Kaggle run so that anyone (teammates, faculty) can
find a paper's results in the same place.

## 1. Your folder name

Use exactly this name everywhere a path says `<Member>`:

| Member | `<Member>` folder name | Papers |
|---|---|---|
| Md. Yousuf | `Md.Yousuf` | 3, 20, 26, 27, 28 |
| MD. Rokib Hasan Oli | `MD.Rokib.Hasan.Oli` | 2, 6, 8, 23, 24 |
| Tashfia Adiba Zaman Nizhum | `Tashfia.Adiba.Zaman.Nizhum` | 1, 10, 22, 31, 32 |
| Md Rakibul Hasan | `Md.Rakibul.Hasan` | 11, 17, 18, 21, 30 |

The paper number always has two digits: `paper03`, `paper11`, `paper24`.

## 2. What a finished Kaggle run gives you

| File | Where you get it | Size |
|---|---|---|
| The finished notebook (`.ipynb`) | Kaggle notebook version → **File → Download notebook** | < 2 MB |
| The run log | Kaggle version → **Logs** → download | < 1 MB |
| `paperNN_outputs.zip` | Kaggle version → **Output** tab | 40–250 MB, all weights |
| Inside the outputs zip: `paperNN_results.zip`, `results_vs_paper.md`, `best.pt` files | unzip it | small |
| `results.zip` (Kaggle's "Download all") | Kaggle Output tab | 150–550 MB: **never upload this one** |

## 3. Where each file goes

| File | Put it in | Rule |
|---|---|---|
| Finished notebook | `notebooks/<Member>/paperNN_run.ipynb` | rename it to this name |
| Run log (optional) | `notebooks/<Member>/paperNN_run.log` | |
| Results README (model card) | `weights/paperNN/<Member>/README.md` | see [section 7](#7-readme-template-for-a-results-folder) |
| `results_vs_paper.md` | `weights/paperNN/<Member>/` | from the outputs zip |
| `paperNN_results.zip` | `weights/paperNN/<Member>/` | results without weights |
| `best*.pt` model files | `weights/paperNN/<Member>/` | < 25 MB per file on the web page, < 100 MB with git; bigger → Drive |
| Study report PDF | `weights/paperNN/<Member>/PaperNN_Study_Report.pdf` | |
| Written report in Markdown | `report/<Member>/` (or `report/<Member>/paperNN/`) | e.g. `Paper03_Report.md` |
| Reading notes, report source | `report/<Member>/paperNN/` | `paperNN_notes.md`, `source/` |
| Kaggle kit (notebook to run, `.py`, guides) | `src/<Member>/paperNN/` | `START_HERE.md`, `kaggle/` |
| `paperNN_outputs.zip` | **team Drive** `498R / weights / paperNN /` | `output/` (Git LFS) only if there is quota: read [output/README.md](output/README.md) |
| Kaggle `results.zip` | nowhere (keep it on your PC or Drive) | too big |

Example: paper #23 of Rokib

```text
notebooks/MD.Rokib.Hasan.Oli/paper23_run.ipynb
weights/paper23/MD.Rokib.Hasan.Oli/README.md, results_vs_paper.md, paper23_results.zip, best_*.pt, Paper23_Study_Report.pdf
report/MD.Rokib.Hasan.Oli/paper23/paper23_notes.md, source/
src/MD.Rokib.Hasan.Oli/paper23/START_HERE.md, kaggle/paper23_kaggle.ipynb, ...
Drive: 498R / weights / paper23 / paper23_outputs.zip
```

## 4. How to upload

### A. On the GitHub web page (easiest)

1. Open the repo, go into the target folder (for example `weights/`).
2. **Add file → Upload files**. You can drag a whole folder: GitHub keeps the folder structure, so dragging a
   folder `paper23` that contains `MD.Rokib.Hasan.Oli/…` creates `weights/paper23/MD.Rokib.Hasan.Oli/…`.
   To make a new folder by hand: **Add file → Create new file**, type `paper23/MD.Rokib.Hasan.Oli/README.md` as the
   name (each `/` makes a folder).
3. Write a commit message (see [section 5](#5-commit-messages)) and choose **Commit directly to the `main` branch**.
4. Limits of the web page: 25 MB per file, 100 files per upload.

### B. With git (for big files, many files, or one commit per file)

```bash
git clone https://github.com/Rokib-Hasan-Oli/CSE498R.git
cd CSE498R
git lfs install                      # once; needed only for output/*.zip
git pull                             # always get teammates' latest work first
# copy your files into the folders from section 3, then commit each file:
git add weights/paper23/MD.Rokib.Hasan.Oli/best_det_fair.pt
git commit -m "paper23: add model best_det_fair.pt"
git push origin main
```

If `git push` is refused because a teammate pushed first: run `git pull --rebase`, then `git push origin main` again.

## 5. Commit messages

`paperNN: add|update|fix|move <what>` for a paper, `repo: <what>` for the whole repo. Examples:

```text
paper23: add model best_det_fair.pt
paper20: update README after the v1.1 re-run
paper03: fix links in README
repo: add CONTRIBUTING guide
```

One file per commit is preferred (that is how this repo was organised), so the history shows each file's story.

## 6. Updating a paper after a re-run (for example v1.0 → v1.1)

1. Replace the files in `weights/paperNN/<Member>/` with the new ones (same names) and replace the notebook in
   `notebooks/<Member>/`.
2. In the results README, update the date, the dataset version (`v1.1`) and the numbers.
3. Keep an old report you still want in `report/<Member>/paperNN/archive/`.
4. Update your paper's row in the paper tracker of the main [README.md](README.md#-paper-tracker) (status and main result).
5. Ask Rokib to refresh [results/results_table.md](results/results_table.md) with your new `paperNN_outputs.zip`.

## 7. README template for a results folder

```markdown
# Paper #NN: <short title> (dataset v1.1)

> **Paper:** <full title>, <authors>, <venue year>, <DOI or arXiv link>
> **Author / maintainer:** <your name>
> **Run:** <date>, Kaggle 2× Tesla T4, 498R BanglaLPR All-in-One v1.1
> **Notebook:** ../../../notebooks/<Member>/paperNN_run.ipynb

## 1. At a glance
| | |
|---|---|
| **Task** | ... |
| **Models** | `best_xxx.pt` — ... |
| **Main result** | ... (paper: ...) |
| **Training** | epochs, minutes, GPU |

## 2. Files
| File | Size | What it is |
|---|---|---|

## 3. Quick start
(a few lines of Python that load the model and run it on one photo)

## 4. Results vs the paper
(copy the key rows of results_vs_paper.md)

## 5. Known limits

## 6. Citation (BibTeX of the paper)
```

## 8. Please do not upload

- **Keys or passwords:** `kaggle.json`, API tokens, `.env` files ([.gitignore](.gitignore) blocks the usual names).
- **Kaggle's `results.zip`** or any file over 100 MB outside `output/`.
- **Fonts copied from Windows** (for example `nirmala.ttf`): their licence does not allow sharing.
- **Publishers' PDFs of the papers:** the repo is public. Link the DOI or arXiv page in your README instead.
- **Files in another member's folder:** ask them first.

## 9. Checklist before you finish

- [ ] Notebook in `notebooks/<Member>/paperNN_run.ipynb`
- [ ] `weights/paperNN/<Member>/` has `README.md`, `results_vs_paper.md`, `paperNN_results.zip` and the model files
- [ ] `paperNN_outputs.zip` uploaded to the team Drive
- [ ] The links in your README open (click them on GitHub)
- [ ] Your row in the paper tracker of [README.md](README.md#-paper-tracker) is updated
