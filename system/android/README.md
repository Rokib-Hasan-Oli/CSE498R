# BanglaLPR Android apps (V1, V2, V3): run fully on the phone, no internet

Three APKs from one code base. Each carries its own models and runs them **on the phone** with ONNX Runtime. The app
has **no INTERNET permission**, so photos, plates and history never leave the phone.

| APK | Model | Inside |
|---|---|---|
| `BanglaLPR_V1.apk` | V1: Bangla cascade (papers 3, 20, 18, 11, 2, 8, 10) | YOLOv8n-576 detector, char-YOLOv8n + LPNR YOLO12n + slot OCR (fallback), CRNN second opinion, two character CNNs |
| `BanglaLPR_V2.apk` | V2: V1 + papers 27, 28, 21, 35, 23, 6, 19, 34 | V1's detector + YOLOv8s retry (+ CLAHE), two char-YOLOv8n readers always, LPNR + SVTRv2-S/T when they disagree |
| `BanglaLPR_V3.apk` | V3: the V2 engine and models unchanged | plus records (each plate saved as a picture + a row in plates.csv), export zip (CSV + pictures + results.html, copied to Download/BanglaLPR), new screens (`src/v3/`); since 22 Nov 2026 a night mode (Auto / On / Off switch, phone light on Live scan, night reading rule, Night mode CSV column: `V3/results/NIGHT_MODE_RESULTS.md`) |

All can be installed side by side (`bd.nsu.banglalpr.v1`, `.v2`, `.v3`).

> **In the GitHub repo:** the APKs are on the [Releases page](https://github.com/Rokib-Hasan-Oli/CSE498R/releases/tag/apps-2026-11-23) (too big for normal files), and this folder holds the source only. Not uploaded: `apk/`, `onnx/`, `app/build/`, `.gradle/`, `local.properties` (it names the Android SDK folder of the author's PC) and the ONNX model copies in `app/src/v1/assets/` and `app/src/v2/assets/`. To rebuild: run `tools/export_onnx_mobile.py` (writes `android/onnx/`), then copy into `app/src/v1/assets/`: `p02_lpnr_v12n`, `p08_cnn28`, `p10_char_cnn32`, `p11_chars_v8n`, `p18_slot`, `p18_v8n_576`, `p20_crnn_hr` (.onnx), and into `app/src/v2/assets/`: `p02_lpnr_v12n`, `p06_det_v8s`, `p11_chars_v8n`, `p18_v8n_576`, `p27_chars_v8n`, `p34_svtrv2_s`, `p34_svtrv2_t` (.onnx); V3 uses the V2 folder. Create `local.properties` with `sdk.dir=<your Android SDK folder>`.

**Night mode on the phone is slow:** on the PC CPU the night reading takes about 166 ms per plate against about 24 ms
normally (`desktop-test/parity_v2_night.txt`: same text as the PC engine on 99.23 % of test crops and 99.51 % of
photos; the few differences come from the Java CLAHE). On a phone expect about a second or more per photo at night.
The V3 APK built on 22 Nov 2026 has not been run on a phone yet; the 13 Nov build is kept as
`apk/BanglaLPR_V3_2026-11-13_before_night.apk`.

## Install on an Android phone (Android 7.0 or newer, ARM 32/64-bit)

1. Copy `BanglaLPR_V1.apk` and/or `BanglaLPR_V2.apk` to the phone (USB cable, Google Drive, Bluetooth...).
2. Open the file on the phone. Android asks to allow installing from this source (Files / Drive / Chrome): allow it.
3. Install. Android or Play Protect may warn that the app is from an unknown developer. That is because it is signed
   with a development key, not published on the Play Store. Tap **Install anyway**.
4. Open **BanglaLPR V1** / **BanglaLPR V2**. The model loads in a few seconds. Allow the camera when you tap
   **Start scanning**.

With a USB cable and developer mode on, you can also install from the PC:
```bash
D:/498R/_android/sdk/platform-tools/adb.exe install -r BanglaLPR_V2.apk
```

## What the app does

- **Scan:** live back camera. Boxes and Bangla text appear over the video, colour-coded verified / likely / review.
  The time per frame is shown at the top. A new readable plate is saved once per 10 s and the phone vibrates; a
  watchlist plate gives a long vibration and an alert.
- **Photo:** take a photo or pick one from the gallery. You get the annotated photo, every plate (Bangla, English,
  status) and what each reader saw. The result is saved.
- **History:** every saved plate with its crop, a search box, a watchlist (stolen / wanted plates), Export CSV
  (share), and Delete all.
- **About:** which papers and models are inside, and the results measured on the team PC.

## How it was checked (and what was not)

- The reading engine (`app/src/main/java/bd/nsu/banglalpr/core/`) is plain Java. The **same classes** were run on the
  PC with desktop ONNX Runtime on the 498R v1.1 test sets (`desktop-test/`, results in `desktop-test/parity_v*.txt`):

  | | V1 (Java engine) | V2 (Java engine) | Python system |
  |---|---|---|---|
  | 651 test crops, plate exact | 90.02 % | 91.71 % | 90.02 % / 91.71 % |
  | same text as Python, crops | 100 % | 100 % | – |
  | 204 test photos, plate exact | 89.22 % | 92.65 % | 89.22 % / 92.65 % |
  | same text as Python, photos | 99.51 % (203 / 204) | 100 % | – |

- **Not measured:** speed on a phone. The PC CPU runs about 17-18 photos per second with this engine, and a phone will
  be slower (it depends on the chip). The Scan screen shows the real time per frame on your phone.
- **Run on an emulator (1 Nov 2026):** both apps read photos on BlueStacks (Android 9, x86_64). The first V1 build
  crashed there in the character-CNN check: the XNNPACK execution provider ignored the batch size. XNNPACK is now off,
  so the phone uses the same CPU engine that was checked on the PC, and V1 works. Not yet tried on a physical
  phone; any error is shown on screen with the code line.

## Rebuild

```bash
# models (ONNX) from the team's weights
D:/498R/_dataset_build/venv/Scripts/python.exe BanglaLPR_System/tools/export_onnx_mobile.py
# APKs (JDK 17 + Android SDK 34 + Gradle 8.7 in D:/498R/_android)
set JAVA_HOME=D:\498R\_android\jdk17
set GRADLE_USER_HOME=D:\498R\_android\gradle-home
D:\498R\_android\gradle-8.7\bin\gradle.bat assembleV1Release assembleV2Release assembleV3Release
```
The APKs land in `app/build/outputs/apk/v1/release/` and `app/build/outputs/apk/v2/release/`.
