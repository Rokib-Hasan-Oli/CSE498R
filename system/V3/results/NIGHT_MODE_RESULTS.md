# V3 night mode: what was tested and what it does (22 Nov 2026)

The supervisor asked us to show that the app works in low light / at night. V3 now has a **night mode**. This page
lists what was measured on the team PC (RTX 5060 Ti, dataset v1.1). Choices were made on **val**, then **test** was
run. Script: `tools/night_mode_eval.py`; numbers: `results/night_mode.json`.

"Night" here = the dataset's rule: mean grey < 60, or a NightLPBD photo. No new training: same V2 weights.

## 1. Can the app tell night from the photo alone? Not reliably

Many night photos are lit by a flash or headlights and are not dark. The best brightness rule on val (mean grey of
the top third of the photo < 90.6) catches **71.1 %** of val night photos but also flags **15.9 %** of day photos
(test: 64.9 % / 14.2 %).

**So night mode is switched by the clock or the user**: Auto = 18:00-06:00 on the device clock, or On / Off. The app
only *suggests* night mode for a clearly dark photo: background brightness < 63, chosen on val so that at most 5 % of
day photos are flagged (val: 5.0 % day, 49.4 % night caught; test: 4.0 % / 40.5 %).

## 2. What night mode changes in the reading (chosen on val)

| Rule on night crops | Val night (46 crops) | Val all (766) | "Verified" plates right, val night |
|---|---|---|---|
| V2 (`verify`) | 82.61 % | 89.69 % | 94.87 % |
| all 5 readers vote | 82.61 % | 89.56 % | 94.87 % |
| + CLAHE copy only when not verified | 84.78 % | 90.21 % | 94.87 % |
| **all 5 readers on the crop + on a CLAHE copy, all 10 vote (`vote10`)** | **86.96 %** | 90.47 % | 97.37 % |

Rule fixed before looking: best val night score, may not lose more than 0.5 point on all val crops → `vote10`.

**A correction made during the work.** In the first run, the CLAHE readings also counted as opinions for the status.
"Verified" plates then became *less* often right (val night 94.87 → 92.86 %, test night 100 → 91.53 %), because a
reader on a CLAHE copy is not an independent second opinion. The status now counts only the five readings of the
crop itself. The plate text is the same in both runs. Run 1 is kept as `night_mode_run1_status_all_readings.json`.

**Detector:** a lower confidence for the detector did not find more night plates on val (79 of 80 at 0.25, 0.15 and
0.10; extra boxes 3 / 4 / 7), so night mode keeps 0.25.

## 3. Test (run once with the chosen settings)

| | V2 | Night mode | Paired test |
|---|---|---|---|
| Night plate crops (65) | 84.62 % | **86.15 %** | 1 plate fixed, 0 broken, p = 1.0 |
| All plate crops (651) | 91.71 % | 92.32 % | 5 fixed, 1 broken, p = 0.22 |
| Whole night photos (60, NightLPBD) | 83.33 % | **83.33 %** | no change |
| Whole day photos (144) | 96.53 % | 97.92 % | 2 fixed, 0 broken, p = 0.5 |
| "Verified" night crops that are right | 100 % | 100 % | |
| Reading + detection time per photo (GPU) | ~32 ms | ~87 ms | about 3x |

**What this means.**
- **Night mode is a small help, not a fix.** No difference is statistically significant. On whole night photos it
  changes nothing: 6 of the 60 night photos have no plate found at all, and the others that fail are not fixed by a
  contrast-boosted copy.
- **What the night mode really adds for the user:** the switch (Auto by clock), the phone light on the live camera,
  a "looks dark: read again in night mode" button, and a Night mode column in the records.
- **Real night gains need:** night training data (Kaggle kit `paper35night_kaggle.ipynb`, Night_Training_Guide.md),
  light on the plate (phone light, or an 850 nm IR lamp on a fixed camera), and a real night video test.

## 4. Same rules on the phone

The Android V3 app (`android/`, flavor v3) has the same night rule in its Java engine (`core/Pipeline.java`), the
same Auto / On / Off switch, the phone light on Live scan, and the Night mode column in its CSV. Parity of the Java
night rule with this PC engine: see `android/desktop-test/parity_v2_night.txt`.
