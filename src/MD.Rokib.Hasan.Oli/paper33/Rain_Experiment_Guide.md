# Paper #33 rain experiment on Kaggle GPU T4 x2: step by step

**Question.** In the paper #33 run, YOLO26 models trained with MuSGD found about 20 points more plates in synthetic
rain than models trained with SGD (one seed each). Is that real (does it survive 3 seeds), and which of MuSGD's two
changes causes it: the **Muon term** or the **3x classification-head learning rate**? Full background: section 13 of
`Paper33_Study_Report.pdf`.

**Design.** YOLO26-N, the #33 recipe (640 px, 60 epochs, batch 16, lr0 0.01, human + auto boxes), seeds 1, 2, 3.
Only the optimiser changes:

| Arm | Muon term | 3x head lr | What it is | Part |
|---|---|---|---|---|
| `sgd` | off | off | Ultralytics `optimizer=SGD` (the team default) | 1 |
| `musgd` | on | on | Ultralytics `optimizer=MuSGD` (what `optimizer=auto` picks) | 1 |
| `muon` | on | off | MuSGD with the 3x head groups set back to lr0 | 2 |
| `headlr` | off | on | MuSGD with the Muon step switched off (= Nesterov SGD with a 3x head) | 2 |

The same seed gives both arms of a part the same data order and augmentation, so arms can be compared plate by plate
per seed. Cell 3 runs a self-test that proves it: MuSGD with the Muon step off and no 3x head gives exactly the same
weights as SGD after one step, and switching the Muon step on changes them.

**Files** (folder `kaggle/`): `paper33rain_kaggle.ipynb` (the only file you upload), `p33_rain.py` (the arms, scoring,
statistics; written by the notebook itself), and the usual kit files.

**Budget.** 12 runs x about 2.1-2.8 h = about 29 GPU-hours, so two sessions of about 8 h (`PART = 1`, then
`PART = 2`). Kaggle gives 30 GPU-hours a week: plan the two parts in the same week or one week apart.

---

## 1. Import and set up (as for every kit)
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → upload `kaggle/paper33rain_kaggle.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2**, Internet **On**, Persistence: Files only.
3. *(Optional)* **Add Input → Datasets →** `498r-bangla-lpr-all-in-one` if the team made it (otherwise the notebook
   downloads dataset v1.1 itself).

## 2. Quick test first (about 15 min)
First code cell: `FAST = True`. Cell 3: `PART = 1`. **Run All** in the interactive session. Check:

| Cell | You should see |
|---|---|
| 0. Setup | two `Tesla T4` lines |
| 2. Dataset | `dataset: ...` and the counts |
| 3. Part + self-test | `PART 1: arms musgd and sgd`, the four arms' groups, `SELFTEST OK` |
| 4. Runs | six `[done] rain_..._sN code=0` lines (1 epoch each, numbers mean nothing) |
| 5. Analysis | a table of the two arms and the arm-vs-arm table |
| last cell | `packed /kaggle/working/paper33rain_part1_outputs.zip` |

## 3. Part 1: the real run (about 8 h)
`FAST = False`, `PART = 1` → **Save Version → Save & Run All (Commit)**. When it finishes, the **Output** tab has
`paper33rain_part1_outputs.zip` (all weights), `paper33rain_part1_results.zip` (no weights), `rain_analysis.md`.

## 4. Part 2 (about 8 h)
In the same notebook: `PART = 2`. **Add Input → Notebook Output →** the finished part-1 version of this notebook (so
cell 5 finds part 1's `runs/rain_*/per_plate.csv` under `/kaggle/input/` and prints the full 2 x 2 factorial).
Then **Save Version → Save & Run All**. If you skip the input, part 2 still works and the factorial is made locally:

```
python p33_rain.py analyse --runs <unzipped part 1>/runs <unzipped part 2>/runs --out rain_analysis
```

## 5. What to check before trusting the numbers
- Every job `code=0`; the **Runs** table in `rain_analysis.md` shows `60 / 60` epochs for every run (no
  "(stopped early)").
- Each run's log starts with a `[p33_rain] arm ...` line listing its optimiser groups: `musgd` has lr 0.03 groups and
  `+muon`, `muon` only lr 0.01 groups with `+muon`, `headlr` lr 0.03 groups without `+muon`, `sgd` three plain groups.
- Clean-test plates found should be about 97-98.5 % in every run (as in the #33 run). A run far below that failed.

## 6. How to read the result
- **Does the rain effect replicate?** `rain` row, MuSGD vs SGD: the seed-level 95 % interval of the difference excludes
  0, and the per-seed McNemar tests point the same way in all three seeds.
- **Which part causes it?** The 2 x 2 factorial (rain row): the main effect of the Muon term vs the 3x head lr, with
  95 % intervals; a large interaction means it takes both.
- **Does it cost anything?** The `clean`, `night` and `fog` rows of the same tables.
- **Remember:** the rain is synthetic. A positive result is a hypothesis for real rain photos (part of the NSU test
  set, P5), not yet a claim about rain.

## 7. Store the outputs
| File | Where |
|---|---|
| `paper33rain_part1_results.zip`, `paper33rain_part2_results.zip`, `rain_analysis.md` | GitHub `weights/paper33/MD.Rokib.Hasan.Oli/rain/` |
| `paper33rain_part1_outputs.zip`, `paper33rain_part2_outputs.zip` | team Drive `498R / weights / paper33 / rain /` |

`add_results.py` is optional here (it would add 12 YOLO26-N rows to the team detector table); the notebook writes no
`results_vs_paper.md`, so it never changes the #33 notes.

## Common problems
| Problem | Fix |
|---|---|
| `SELFTEST FAILED` in cell 3 | the Ultralytics version changed: the notebook pins `ultralytics==8.4.170`; check the pip line ran |
| a run ends after a few minutes | read `logs/rain_<arm>_s<seed>.log`; usually the dataset download failed (Internet off) |
| `[time cap] ... stopping` in a log | the session was slow; the run is marked "(stopped early)" in the Runs table: re-run only that run |
| part 2 shows no factorial | part 1's output is not attached as an input: attach it, or run `analyse` locally (step 4) |
