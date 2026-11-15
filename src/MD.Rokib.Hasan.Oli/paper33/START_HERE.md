# Paper #33: start-to-end guide

**YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection**

| What | Where |
|---|---|
| Notes (read these with the paper) | [33_YOLO26 - Key Architectural Enhancements and Performance Benchmarking.md](../../../report/MD.Rokib.Hasan.Oli/paper33/paper33_notes.md) |
| Kaggle notebook | `kaggle/paper33_kaggle.ipynb` (detailed steps and fixes: [Kaggle_Training_Guide.md](Kaggle_Training_Guide.md)) |
| Dataset | v1.1, team Drive folder [All In One 1.1](https://drive.google.com/drive/folders/11psdzW5Cpr7tXwqzqNxNt3LA-qeDSOQl). **The notebook downloads it by itself** |
| v1.0 results | New paper (added 25 Oct 2026): there are no v1.0 results; the notebook runs on v1.1. |
| Results go to | GitHub [Rokib-Hasan-Oli/CSE498R / weights](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights) and the team Drive |

## 1. Get the paper
- Paper: https://arxiv.org/abs/2509.25164 · code: https://github.com/ultralytics/ultralytics (YOLO26, in ultralytics==8.4.170)
- Team copy (PDF): `D:\498R\Papers\6_6 Added papers (new5)\33_2025_YOLO26 Key Architectural Enhancements and Benchmarking.pdf`

## 2. Why this paper matters for 498R
The newest Ultralytics detector (September 2025): NMS-free head, no DFL, small-target-aware labels, the MuSGD optimiser. It is the default model in our pinned Ultralytics, so we must know if it beats YOLO11 / v8 on plates.

## 3. How to read it (about 2-3 hours, 3 passes)
1. **Pass 1 (10 min):** title, abstract, figures, conclusion. Can you say in 3 lines what problem, what method and
   what result? Compare with "Pass 1" in the notes.
2. **Pass 2 (about 1 h): read these parts closely:**
   - p.1-4: what changed from YOLOv8-v13 (Table 1)
   - p.6-8: DFL removal, NMS-free inference, ProgLoss + STAL, MuSGD
   - p.8, Table 2: COCO accuracy for both heads (NMS / e2e) and speed (CPU ONNX, T4 TensorRT)
   - p.9-11: deployment, quantisation, applications
   - Watch out: this is a review; the numbers are Ultralytics' own, and the abstract's Jetson benchmarks are not in the paper
3. **Pass 3 (about 1 h):** rebuild the method in your head; read "Pass 3: rebuild it" in the notes, which also lists what
   would fail on Bangladeshi plates.

Check yourself: can you explain the method without the paper, name 2 weak points, and say what we take from it?

## 4. Run the code (Kaggle, GPU T4 x2): about 8 h
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → `kaggle/paper33_kaggle.ipynb`.
2. Right panel: **Accelerator GPU T4 x2**, **Internet On**.
   - **The dataset is downloaded automatically** from the team Drive into `/tmp`, in about 3-6 min.
   - Faster, and safe from Drive download limits: **Add Input → Datasets → `498r-bangla-lpr-all-in-one`**, if the
     team created it.
3. **Quick test:** `FAST = True` in the first cell → **Run All** (10-20 min). Every job must end with `code=0`.
4. **Full run:** `FAST = False` → **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. **Output tab** of the finished version:
   - `paper33_outputs.zip`: everything, with the weights
   - `paper33_results.zip`: the same without model files (small)
   - `results_vs_paper.md`: the comparison table

**Original-paper check (the paper's own data):** `kaggle/paper33_original_check.ipynb`, about 2 h. Steps in [Original_Check_Guide.md](Original_Check_Guide.md). It downloads its own data.

## 5. What to check and learn
- YOLO26-N/S vs YOLO11-N on plates: mAP, small-plate recall, latency
- The two heads of one model: how much mAP does NMS-free cost, how many ms does it save (PyTorch, TensorRT, CPU)?
- MuSGD vs SGD, and the P2 head for small plates
- TensorRT INT8: does YOLO26 really keep its accuracy (vs YOLO11-N)?
- Original check: the released weights on COCO, both heads, T4 TensorRT
- **Results by condition** (end of `results_vs_paper.md`): clear, night, blurry, small plates, synthetic rain/fog and
  false alarms. Where does the method fail?
- **Is the comparison fair?** If the paper used other data, read the gap as a direction, not a contest (the table says
  which case it is).

## 6. After the run: save and share
1. **GitHub** ([weights folder](https://github.com/Rokib-Hasan-Oli/CSE498R/tree/main/weights)): open `weights/` → **Add file → Upload files**.
   - Put `paper33_results.zip` and `results_vs_paper.md` into `weights/paper33/<your-name>/`.
   - Add the small `best.pt` files too (under 25 MB for a web upload; under 100 MB with git).
   - Bigger weights (transformer OCR, VLM LoRA) go to the team Drive.
2. **Drive:** upload the full `paper33_outputs.zip` to `498R / weights / paper33 /`.
3. **Team table:** put the zip in `D:\498R\docs\results\` and run
   `python add_results.py paper33_results.zip --member "<your name>"`. This fills the notes ("Results: ours vs.
   the paper") and `results_table.md`.
4. **Notes:**
   - check "Settings I trained with" (`epochs_run` in results.json tells you if the time cap stopped a run)
   - write 2-3 sentences in "For us": keep or drop, and why
5. **Board:** add the code link and short note (`docs/PASTE_board_and_lit_table.md`) and your result in one line.

## 7. What to do next
- Add the rows to the detector comparison (#21, #26-#30) and choose the demo detector
- If NMS-free + INT8 holds up, use it for the CPU / edge demo (#18)
- Then take the next paper in the run order (`docs/FULL_REVIEW.md`, section 5) and discuss your table with the team.
