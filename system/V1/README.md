# BanglaLPR V1 (Bangla cascade)

Standalone model folder. **Read `BanglaLPR_V1_Explained.pdf` first**: it covers how it works, which papers and
trained models are used and how, the results, how to improve it, and what to do next.

Built from the paper studies: Cloud Hybrid (3), Plate SR (20), Embedded LPR (18), Video LPR (11), Cascaded YOLO (2), CNN + GFP-GAN (8), BLPnet (10). No new training: only the team's Kaggle-trained weights in `models/weights/`
(training facts in `results/training_facts.json`), combined and chosen on the validation split, tested once.

| Measured on the team PC (dataset v1.1 test) | |
|---|---|
| Whole plate exact, 204 test photos | 89.22 % |
| Character accuracy, 204 test photos | 94.74 % |
| Whole plate exact / char accuracy, 651 test crops | 90.02 % / 97.97 % |
| Speed, whole photo: GPU RTX 5060 Ti / CPU only | 48.8 / 17.7 FPS |

```bash
pip install -r requirements.txt                 # or use D:/498R/_dataset_build/venv/Scripts/python.exe
python predict.py car.jpg                       # photo(s), a folder, or a video (.mp4); add --cpu without a GPU
python evaluate.py --data D:/498R/dataset       # re-measure the numbers above
```

**Android app (offline, on the phone): `BanglaLPR_V1.apk` on the repo's [Releases page](https://github.com/Rokib-Hasan-Oli/CSE498R/releases/tag/apps-2026-11-23).** Copy it to an Android 7+ phone and
install it; the model runs on the phone with no internet (install guide: `../android/README.md`).

Web app + mobile app for both versions: `python ../webapp/server.py --phone --port 8498`, then open the printed
https address on the PC, or that address + `/app` on a phone on the same Wi-Fi.

| Path | What |
|---|---|
| `models/config.json` | the components of this version and why each was chosen |
| `models/weights/` | the weight files this version uses |
| `lpr/` | the system code (detector, readers, fusion, video, storage) |
| `results/summary.json` | every measured number (test crops, whole photos, CPU, conditions, readers, detectors, selection on val) |
| `results/*.csv` | the prediction for every test crop and photo |
