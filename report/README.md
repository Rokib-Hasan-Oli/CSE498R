# report/ — written reports, reading notes and report sources, one folder per member

| Member | Contents |
|---|---|
| [MD.Rokib.Hasan.Oli](MD.Rokib.Hasan.Oli/) | `paperNN/` for 02, 06, 08, 23, 24: `paperNN_notes.md` (reading notes), `source/` (scripts, figures and numbers that build the study report PDF), `archive/` (old v1.0-run report, where there is one) |
| [Md.Yousuf](Md.Yousuf/) | `Paper03_Report.md`, `Paper20_Report.md`, `Paper27_Report.md`, `Paper28_Report.md` (full technical reports) |
| Md.Rakibul.Hasan | none yet |
| Tashfia.Adiba.Zaman.Nizhum | none yet |

The **study report PDFs** sit next to the models, in `weights/paperNN/<Member>/PaperNN_Study_Report.pdf`, so one
folder holds a paper's whole results package.

**Layout for a new paper:**

```text
report/<Member>/paperNN/
├── paperNN_notes.md     reading notes (what the paper does, settings, our results)
├── PaperNN_Report.md    a written report, if you write one in Markdown
├── source/              optional: scripts and figures that build your report
└── archive/             optional: an older report that a re-run replaced
```

Notes about `source/` folders: the scripts read the run's `results.zip` and `paperNN_outputs.zip` from the
author's PC (team Drive copies), and the Bangla font they use (Nirmala UI, `fonts/nirmala.ttf`) is a Windows font
that is **not** in the repo (licence). Copy `C:\Windows\Fonts\Nirmala.ttf` into `source/fonts/` to rebuild a PDF.
