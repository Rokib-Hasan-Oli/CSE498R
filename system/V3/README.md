# BanglaLPR V3 (V2 engine + plate-image records + new app screens)

V3 reads plates with **exactly the V2 engine and weights** (no new training, so the accuracy is V2's). What is new:

1. **Every plate is saved with its picture.** Each plate gets a picture in `records/plate_images/` and a row in
   `records/plates.csv`. The CSV's **Plate image** column names that picture.
2. **One-click export**: a zip with `results.csv`, `plate_images/`, and `results.html`, a printable page that shows
   every row next to its plate picture.
3. **New, simpler screens** in the web app and in the Android app.

| Measured with the V2 engine on the team PC (dataset v1.1 test, 31 Oct 2026) | |
|---|---|
| Whole plate exact, 204 test photos | 92.65 % |
| Character accuracy, 204 test photos | 96.79 % |
| Whole plate exact / char accuracy, 651 test crops | 91.71 % / 98.71 % |
| Speed, whole photo: GPU RTX 5060 Ti / CPU only | 41.7 / 20.0 FPS |

`results/` holds those V2 measurements (V3 changes nothing in the reading).

## Web app (PC)

Double-click **`start_web_app.bat`**, or run:

```bash
D:/498R/_dataset_build/venv/Scripts/python.exe app/server.py
```

Then open http://127.0.0.1:8499. Options: `--phone` (https on the Wi-Fi so a phone can use its camera), `--cpu`, `--port`.

| Tab | What it does |
|---|---|
| Read a photo | Drop photos, choose them, or paste one with Ctrl+V. Each plate shows big in Bangla and English, with a status, a Copy button and "Saved as record #N". |
| Video | One saved record per vehicle, with its best plate picture. |
| Live camera | Saves a plate once per 10 s. |
| Saved records | Stats, search, status and date filters, card or table view, delete one record. **Export CSV + pictures (.zip)**, **Download CSV**, **Open folder**. |
| Watchlist | Matching plates are shown in red and get `YES` in the CSV's *Watchlist match* column. |
| Help & model | How to use, what each status means, the measured numbers. |

## Command line

```bash
python predict.py car.jpg                 # photos, a folder, or a video; add --cpu without a GPU
```

This writes `predictions/plates.csv`, `predictions/plate_images/`, `predictions/annotated/` and `results.json`.
Running it again with the same `--out` folder adds rows to the same CSV.

## Night mode (added 22 Nov 2026)

A **Night: Auto / On / Off** switch sits in the header of the web app and of the Android app. Auto = on from 18:00 to
06:00 by the device clock. A photo alone does not tell night reliably (many night photos are lit by a flash or
headlights), so the app never switches by itself; it only offers **Read again in night mode** for a clearly dark
photo.

| Night mode does | Measured on the held-out test set (`results/NIGHT_MODE_RESULTS.md`) |
|---|---|
| every reader also reads a CLAHE (contrast-boosted) copy of the plate; all 10 readings vote; the status counts the 5 readings of the plate itself | night plate crops 84.62 → 86.15 % (1 plate, p = 1.0); whole night photos 83.33 → 83.33 %; about 3x the reading time |
| live camera: switches the phone light on (Android app; Chrome on Android in the web app) | not measurable on photos |
| records: a **Night mode** column (on / off) in plates.csv and a tag on each card | |

Honest summary: a small help, not a fix. Real night gains need night training data (Kaggle notebook
`paper35night_kaggle.ipynb`) and light on the plate. Command line: `python predict.py photo.jpg --night`.
Re-run the measurement: `python tools/night_mode_eval.py`.

## Records folder

```
records/
  plates.csv                               one row per plate (UTF-8 with BOM, so Excel shows Bangla)
  plate_images/<date>/<no>_<PLATE>.jpg     the cut-out plate of each row
  photos/<date>/run<no>_<kind>.jpg         the whole photo with plate boxes
  videos/                                  annotated videos
  exports/export_<time>/ + .zip            results.csv + plate_images/ + results.html
  lpr.db                                   search index for the web app (same rows as plates.csv)
```

CSV columns: No, Date, Time, Plate (Bangla), Plate (English), Status, BRTA layout OK, Read confidence, Detector
confidence, Source, File name, **Plate image** (relative to the CSV), Plate image (full path), Vehicle photo,
Watchlist match, Model. Deleting records from the app rewrites the CSV and removes the pictures, so the folder,
the CSV and the app always agree.

## Android app: `BanglaLPR_V3.apk`

Download it from the repo's [Releases page](https://github.com/Rokib-Hasan-Oli/CSE498R/releases/tag/apps-2026-11-23) (216 MB, too big for a normal GitHub file). The `records/` folder is not in the repo: the app makes it on first use.

The app works fully offline on Android 7 or newer (it has no internet permission). It installs next to V1 and V2
(`bd.nsu.banglalpr.v3`). It has four tabs: **Photo**, **Live scan**, **Records** and **Help**.

- Every plate is saved on the phone, as a picture plus a row in `plates.csv`, in
  `Android/data/bd.nsu.banglalpr.v3/files/records/`. A PC can see this folder over USB.
- **Export CSV + pictures** builds a zip (CSV, pictures and `results.html`). On Android 10 and newer, the zip is also
  copied to **Download/BanglaLPR** (open it with the Files app). **Share zip** and **Share CSV** send the files to
  Drive, WhatsApp, Gmail and so on.
- The reading engine is the V2 Java engine. On the PC it gave the same text as the Python system on all 651 test
  crops and all 204 test photos (`../android/desktop-test/parity_v2.txt`). Speed on a real phone has not been
  measured.

Rebuild: `gradle assembleV3Release` in `../android` (see `../android/README.md`). The V3 code is in
`../android/app/src/v3/` (`V3Activity.java` for the screens, `Records.java` for the CSV, pictures and export).

| Path | What |
|---|---|
| `app/` | the V3 web app (`server.py`, `static/`) |
| `lpr/records.py` | plate pictures, CSV, export bundle (used by the web app and predict.py) |
| `lpr/store.py` | SQLite index that keeps the CSV in step |
| `lpr/` (rest) | the V2 engine, unchanged |
| `models/` | V2's config (renamed V3) and weights |
