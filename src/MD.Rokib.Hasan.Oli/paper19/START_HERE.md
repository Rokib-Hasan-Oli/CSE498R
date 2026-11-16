# Paper #19: start-to-end guide

**LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification**

| What | Where |
|---|---|
| Notes (read these with the paper) | [19_LPLCv2 - An Expanded Dataset for Fine-Grained License Plate Legibility Classification.md](../../../report/MD.Rokib.Hasan.Oli/paper19/paper19_notes.md) |
| Kaggle notebook | `kaggle/paper19_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | New paper (added 25 Oct 2026): there are no v1.0 results; the notebook runs on v1.1. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2604.08741 · code: https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset (data on e-mail request); IJCNN 2026
- Team copy (PDF): `D:\498R\Papers\6_3 General LPR state of the art\19_2026_LPLCv2 License Plate Legibility Classification.pdf`

## 2. Why this paper matters for 498R
An LPR system should know when a plate cannot be read (and say "unreadable" instead of a wrong number). LPLCv2 trains a classifier for exactly that: perfect / good / poor / illegible plates.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.1-2: why legibility matters (reject, enhance or read)
   - p.3-4, Tables I-III: the dataset (37k photos, 41k plates), the four classes and the four scenarios (Baseline, Legibility / Full Recognition, Quality Filter)
   - p.5: the training (ResNet-50, Adam 1e-5, batch 64, EMA loss) and Table IV (+ revised labels, + images, + EMA)
   - p.6, Tables V-VII: the Quality Filter (96.7 %) and the cross-camera test
   - Watch out (see "Paper vs. code" in the notes): the code gives the class with the MOST errors the LOWEST weight, the opposite of what the text says; the repo config uses batch 16 and early stop 5, not 64 and 20
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 2 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper19_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper19_outputs.zip`: everything, with the weights
   - `paper19_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check:** not possible for this paper (LPLCv2 is given only on e-mail request (subject "2026 LPLCv2 Request Form", from a university address: your action); the code is public. The main kit uses OCR-difficulty labels on our plates instead).

## 5. What to check and learn
- How well a ResNet-50 learns OUR OCR-difficulty labels (4 classes and Quality Filter), EMA vs cross-entropy
- The real question: at 80 % coverage, are the plates the classifier accepts more often read right than the plates the OCR itself is confident about?
- Which conditions it flags (night, blurry)
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper19_results.zip` and `results_vs_paper.md` into `weights/paper19/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper19_outputs.zip` to `498R / weights / paper19 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper19_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- If the gate beats OCR confidence, add it after the plate detector (reject or send to super-resolution #20 / low-light enhancement #35)
- Optional: label 200 test crops by eye (perfect / good / poor / illegible) in ocr/legibility_human.csv to score the classifier against people
- Ask for LPLCv2 by e-mail if the team wants the paper's own data
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
