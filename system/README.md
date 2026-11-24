# BanglaLPR System: V1, V2 and V3 models, web apps and offline Android apps

Real-time Bangla licence plate detection and recognition for the NSU CSE 498R project *"Real-Time Bangla License Plate
Recognition and Detection using Deep Learning"*. Built on 31 October 2026 from the team's own Kaggle-trained models.

- **V1** is built from papers 3, 20, 18, 11, 2, 8 and 10.
- **V2** is built from papers 27, 28, 21, 35, 23, 6, 19 and 34, on top of V1.

> **Honest scope.** No new training was done. V1 and V2 combine weight files the team already trained (paper kits,
> dataset v1.1). Every combination was chosen on the validation split, then scored once on the test split, on this
> PC. See [`docs/V1_V2_MODEL_REPORT.md`](docs/V1_V2_MODEL_REPORT.md).

## In this repository

| Version | Folder | What it is | Android app |
|---|---|---|---|
| V1 | [`V1/`](V1/README.md) | papers 3, 20, 18, 11, 2, 8, 10 (31 Oct 2026) | `BanglaLPR_V1.apk` |
| V2 | [`V2/`](V2/README.md) | V1 + papers 27, 28, 21, 35, 23, 6, 19, 34 (31 Oct 2026) | `BanglaLPR_V2.apk` |
| V3 | [`V3/`](V3/README.md) | the V2 engine and weights unchanged + plate-picture records, CSV/zip export, new app screens (13 Nov 2026), night mode (22 Nov 2026) | `BanglaLPR_V3.apk` |

- **APKs** (147-216 MB each, over GitHub's 100 MB file limit) are on the [Releases page](https://github.com/Rokib-Hasan-Oli/CSE498R/releases/tag/apps-2026-11-23), not in the folders.
- **Not uploaded on purpose:** the web apps' saved data (`webapp/data/`, `V3/records/`: photos and plates read with the camera), the Android build output and Gradle files made on the author's PC, and the ONNX copies inside the Android app (rebuild them with `tools/export_onnx_mobile.py`, see [`android/README.md`](android/README.md)).
- **Paths:** the scripts read the 498R dataset from `D:/498R/dataset` (the author's PC). Elsewhere set `DATASET_ROOT` (or `--data` for `evaluate.py`) to your copy of the dataset v1.1, and use `python` instead of `D:/498R/_dataset_build/venv/Scripts/python.exe`.

## Results (measured on this PC)

| | V1 | V2 | Proposal target |
|---|---|---|---|
| Plate detection mAP@50 (test, 1,153 photos) | 98.96 % | 98.96 % | ≥ 95 % ✓ |
| Character accuracy: 651 test crops / 204 whole photos | 97.97 / 94.74 % | **98.71 / 96.79 %** | ≥ 90 % ✓ |
| Whole plate exact: crops / photos | 90.02 / 89.22 % | **91.71 / 92.65 %** | (stricter) |
| Speed, whole photo, GPU RTX 5060 Ti | 48.8 FPS | 41.7 FPS | ≥ 10 FPS ✓ |
| Speed, whole photo, CPU only (Core Ultra 7 265K) | 17.7 FPS | 20.0 FPS | ≥ 10 FPS ✓ |
| Plates auto-accepted ("verified") / their accuracy | 78.0 % / 96.1 % | 94.9 % / 95.3 % | – |

V2 vs V1 on the same plates: crops 12 fixed / 1 broken (p = 0.003); whole photos 9 / 2 (p = 0.065, not significant,
all of it on night photos). Edge boards (Raspberry Pi, Jetson) are not measured yet.

## Run the web app

```bash
python system/webapp/server.py --port 8498
```

Then open http://127.0.0.1:8498. The first request loads the models (about 10-20 s). Add `--preload` to load both
versions at start.

**On a phone (same Wi-Fi):** start with `--phone` instead. The server listens on the Wi-Fi with https and prints the
address to open, e.g. `https://192.168.0.40:8498`. The phone browser warns "not private" because the certificate is
self-signed and made on this PC (`webapp/data/certs/`, needs the openssl that comes with Git for Windows). Tap
**Advanced → Proceed**, then allow the camera. Phones allow the live camera only on https pages. The PC still does all
the model work. Use this only on a Wi-Fi you trust.

Tabs:

| Tab | What it does |
|---|---|
| Test photo | upload / drop photos, or click a held-out test photo; choose V1, V2 or **Compare**. Shows the annotated photo, every plate's Bangla + English text, its status (verified / likely / review), what each reader saw, timings, and the ~170-byte event message an edge camera would send |
| Video | upload a video, or **Use demo clip** (a synthetic clip made from 8 test photos). Plates are tracked and read until 3 readings agree: one event per vehicle, plus an annotated mp4 |
| Live camera | webcam frames go to the local server; boxes are drawn live; each plate is stored once per 10 s |
| Stored results | everything saved in SQLite: search, filters, crops, CSV export (opens in Excel with Bangla), **retention** (delete older than N days), delete all |
| Watchlist | hot-list demo: matching plates are flagged red everywhere |
| Model report | the measured numbers, the proposal targets with pass/fail, every candidate reader and detector |

**Storage:** `webapp/data/lpr.db` (SQLite) and `webapp/data/media/` (photos, crops, annotated photos/videos).
Nothing leaves the PC. The page loads its fonts from Google Fonts when online and falls back to Nirmala UI offline.

## Use the models from Python

```python
import sys, cv2
sys.path.insert(0, "system")                      # this folder
from lpr.pipeline import LPRSystem
lpr = LPRSystem("v2")                                   # or "v1"
out = lpr.process(cv2.imread("car.jpg"))
for p in out["plates"]:
    print(p["text"], p["text_en"], p["status"], p["box"])   # e.g. ঢাকা মেট্রো-গ ১৭-৪১১৯  Dhaka Metro-Ga 17-4119  verified
print(out["ms"])                                         # detect / read / total ms
```

Video: `from lpr.video import process_video; process_video(lpr, "clip.mp4", "out.mp4")`.

## Folder map

| Path | What |
|---|---|
| **`V1/`** | **standalone V1 model folder**: code, its 8 weight files, `predict.py`, `evaluate.py`, results, **`BanglaLPR_V1_Explained.pdf`** |
| **`V2/`** | **standalone V2 model folder** (V1 upgraded with 8 more papers): same layout, **`BanglaLPR_V2_Explained.pdf`** |
| **`android/`** | **offline Android apps**: `apk/BanglaLPR_V1.apk`, `apk/BanglaLPR_V2.apk` (also inside `V1/` and `V2/`), Java engine + app source, PC parity test, install guide `android/README.md` |
| `webapp/static/mobile.html` | **mobile app** (installable on the phone; open `https://<PC address>:8498/app` with the server started with `--phone`) |
| `lpr/` | the system: `pipeline.py` (detector, retry, crops, status), `readers.py` (char-YOLO, CRNN, slot, SVTRv2-ONNX, character CNN verifiers), `fusion.py` (fallback / vote / verify rules, status), `video.py` (tracking + vote), `store.py` (SQLite), `vocab.py` + `metrics.py` (BRTA grammar and scoring, copied from the team kit) |
| `models/v1/config.json`, `models/v2/config.json` | each version's components, with the reason for every choice |
| `models/weights/` | the team's weights, renamed `pNN_<model>` by paper (345 MB; SVTRv2 exported to ONNX by `tools/export_svtrv2_onnx.py`) |
| `eval/` | the scripts that built and tested V1/V2 (`reader_preds` → `select_fusion` → `detectors` / `detector_combo` → `system_eval`); `eval/results/` holds every number as JSON / CSV |
| `webapp/` | `server.py` (Python standard library only) + `static/` (HTML/CSS/JS) + `demo/` (synthetic clip) |
| `docs/V1_V2_MODEL_REPORT.md` | how V1/V2 were built, what each paper contributes (and what was tested and rejected), all results, limitations |
| `docs/IMPROVEMENT_GUIDE.md` | 10 concrete steps from V2 to V3, ordered by expected value |
| `docs/IMPACT_AND_INNOVATION.md` | how it can solve problems for Bangladesh, companies and industry: data speed, Starlink, solar edge units, law, business models, pilot plan |
| `docs/research/` | the deep-research report (with sources) and its five research notes |

## Requirements

The team venv `D:/498R/_dataset_build/venv` has everything: Python 3.14, torch 2.13 + CUDA 13, ultralytics 8.4.170,
onnxruntime 1.31, opencv, pillow, numpy. On another PC: `pip install -r requirements.txt` (any recent torch build).
Without a GPU everything runs on the CPU (about 18-20 FPS on a fast desktop CPU).
