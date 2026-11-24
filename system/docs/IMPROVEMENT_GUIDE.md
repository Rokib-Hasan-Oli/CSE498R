# How to improve BanglaLPR (V2 → V3)

Every step below starts from a measured weakness of V2 (`V1_V2_MODEL_REPORT.md`, `eval/results/*.json`). The order is
by expected value for the effort. After any retraining, re-run the selection scripts so that every choice stays on the
validation split:

```
python eval/reader_preds.py      # new reader -> add it to CANDIDATES first
python eval/select_fusion.py     # choose readers + rule on val
python eval/detectors.py         # new detector -> add it to CANDIDATES first
python eval/detector_combo.py
python eval/system_eval.py       # then once on test (GPU), and with --cpu
```

(Use `D:/498R/_dataset_build/venv/Scripts/python.exe`; plain `python` on this PC has a CPU-only torch.)

## Where V2 stands (the facts to improve on)

| Weakness | Evidence |
|---|---|
| Night plates | 84.62 % exact on night crops vs 95.00 % on clear; whole night photos 83.33 % vs 96.53 % day |
| Blurry plates | 82.14 % |
| One wrong symbol in a valid layout | 46 of V2's 54 wrong test plates (25 digit, 13 letter, 8 district) |
| Voting ceiling | oracle of V2's five readers = 93.7 %, so better voting alone can add at most ~2 points |
| No real video | the dataset has stills only; video mode tested on a synthetic clip |
| Edge speed unknown | 41.7 FPS GPU / 20.0 FPS desktop CPU; no Raspberry Pi / Jetson measurement |
| New front plates | BRTA single-line motorcycle front plates (mandatory 15 Nov 2026) are not in the data |

---

## Step 1: Record real video and use the track vote (biggest expected gain)

**Why.** The strongest finding of 2025-2026 plate research is that reading **several frames of the same vehicle**
beats any single-image fix. In the ICPR 2026 low-resolution LPR competition, every top-5 team fused the 5 frames of a
track, and two of the top three used no super-resolution. On UFPR-SR-Plates, fusing 5 frames added about 13.6 points.
(Sources in `research/Bangla LPR impact in Bangladesh.md`, section "Track fusion".) Our own studies agree: pixel
restoration (SR, HVI-CIDNet, SAHI) did not help, but combining readers did. No Bangla multi-frame plate dataset
exists, so building one is also a publishable contribution.

**How.**
1. Film 30-60 min at 3-4 Dhaka spots (day, dusk, night, rain) with a phone or IP camera at 1080p, shutter 1/1000 s if
   possible, camera within 30° of the traffic direction.
2. Run `webapp` → Video tab (or `lpr.video.process_video`) on each clip. It already tracks each plate (ByteTrack) and
   votes over up to 8 readings per track.
3. Label **one plate string per track**, not per frame. That is cheap: correct the `text` of each event in the stored
   results and export the CSV.
4. Measure single-frame accuracy (the first reading of each track) against track-vote accuracy. Then try the
   competition's rules: summing per-frame logits before decoding, quality-weighted votes (weight by crop height and
   reader agreement), and character-position voting.

**Effort:** filming 1-2 days, labelling 1-2 days for 1,000 tracks, no GPU needed for the vote itself.

## Step 2: Grammar-constrained decoding for the sequence readers

**Why.** No reader in our studies was ever right with an invalid BRTA layout, and SVTRv2 / CRNN make many
"almost-plates". Their greedy CTC decode ignores what a plate may contain: one of 64 district words (+ চট্ট / রাজ),
optional মেট্রো, one of 33 series letters, 2 + 4 digits.

**How.** Replace greedy decoding in `lpr/readers.py` (`ctc_greedy`, `SVTRv2Reader.read`) with a constrained beam
search over the CTC probabilities: a small finite-state grammar (district → [মেট্রো] → letter → d d → d d d d) and
beam width 10. Keep the best path that ends in a full plate. Score it on val with `eval/reader_preds.py`, then re-run
the fusion selection. Expected effect: the SVTRv2 / CRNN helpers stop voting for impossible strings. This mainly fixes
plates that now go to "review".

**Effort:** 1 day of code, CPU only.

## Step 3: More night and blurry training data (capture fixes first)

**Why.** Night is the largest gap (−10 points), and image enhancement did not close it (HVI-CIDNet study).

**How.**
1. **At capture time:** an 850 nm IR illuminator and a fixed fast shutter on the camera. This is cheaper and more
   effective than any model change.
2. **Data:** NightLPBD gives only 52 night test crops. Add the night clips from step 1. Pseudo-label them with V2: take
   only video tracks where 3/3 readings agree and the status is *verified*. V2's verified plates are 95.3 % right on
   single frames, and 3-of-3 track agreement is stricter still. A person checks a sample before training.
3. **Retrain** the char-YOLO readers (kit `rec.py`, papers #11/#27 settings: YOLOv8n, 320 px, 100 epochs on a Kaggle
   T4) with night augmentations: gamma 0.4-0.8, motion blur, sensor noise, JPEG quality 30-60.

**Effort:** a Kaggle session of about 1-2 h per reader.

## Step 4: Retrain the final readers on train + val

After all choices are fixed, retrain the chosen readers (#11 char-YOLOv8n, #27 char-YOLOv8n, #2 LPNR) on **train + val**
(766 more crops, +14 %) with the same settings. This is standard practice for a final model. Report it as "V3 (train+val)"
and score it **only** on test, because val is no longer held out.

## Step 5: Calibrated "send to a human" (conformal review threshold)

**Why.** A wrong fine notice costs more than no notice. V2 already sends 5.1 % of plates to review (24 % of those are
right) and auto-accepts 94.9 % (95.3 % right). A field system needs a stated guarantee such as "auto-accepted plates
are ≥ 99 % right".

**How.** Build a score per plate from: how many of the readers that ran agree, the primary's mean box confidence, the
layout validity, crop height, and for video the track agreement. Use half of val to fit a threshold by split conformal
prediction (choose the review rate so that the accepted plates reach 99 % on calibration data), and check it on the
other half and on test. Put the threshold into `lpr/fusion.status`. No plate-specific conformal paper exists yet, so
this is publishable.

**Effort:** 1-2 days, CPU only.

## Step 6: Measure and optimise on edge hardware

**Why.** The proposal says "standard hardware". Our 17.7-20.0 FPS is a desktop CPU. A roadside unit is a Raspberry Pi 5
(+ Hailo NPU) or a Jetson Orin Nano.

**How.**
1. Export the detector: `yolo export model=models/weights/p18_v8n_576.pt format=onnx imgsz=576` (CPU / Hailo
   compiler input) or `format=engine half=True` on a Jetson (TensorRT FP16).
2. Export the char-YOLO readers the same way (imgsz 320 / 640).
3. Keep the sequence readers (SVTRv2, CRNN) in FP16/FP32. The team's Embedded LPR study lost 20 points with dynamic
   INT8 OCR. Use INT8 only with static calibration on our crops, and re-score it with `eval/reader_preds.py`.
4. Time `eval/system_eval.py --cpu` on the board (`LPR_CPU_THREADS=4`). In video mode the reader runs about 3 times
   per vehicle, not per frame, so the detector sets the frame rate.

The research report gives devices, prices and power (Pi 5 + Hailo ≈ 7-8 W, Jetson Orin Nano Super USD 399 after the
July 2026 price rise).

## Step 7: Single-line motorcycle front plates

`vocab.plate_text` orders tokens by kind, so a one-line plate is decoded in the right order. But its `valid` check
requires the words to sit **above** the digits, so every single-line plate is now flagged "layout ✗" and sent to the
helpers or to review. To fix it:

1. Collect and label front plates after 15 Nov 2026 (photos of parked motorcycles are enough to start).
2. Add a single-line layout rule (same tokens, words left of digits) in `lpr/vocab.py`.
3. Fine-tune the char-YOLO readers with these crops.

The first team with a labelled front-plate set owns that dataset.

## Step 8: Synthetic plates for rare districts

The test set is dominated by ঢাকা মেট্রো. Rare district words have few training crops, and 8 of V2's errors are
district words. Render synthetic two-line plates with the BRTA layout and a Bangla font, for all 64 districts × 33
letters, with realistic blur, perspective and lighting (the team's #17 kit has a synthetic-data generator). Add them
to char-YOLO training (≤ 30 % of the mix) and check on val that real accuracy does not drop.

## Step 9: A small vision-language model only for "review" plates

For the 5 % of plates that V2 sends to review, a fine-tuned small VLM (e.g. PaliGemma / Qwen2.5-VL-3B with LoRA, as in
the team's #22 kit) can be called as a last opinion. Its answer must pass the BRTA layout check and agree with at least
one reader. Otherwise the plate stays in review, because generative models can invent plausible digits. This costs
about 100+ ms per plate, but only on 5 % of plates.

## Step 10: Speed-ups that keep accuracy

- Install `onnxruntime-gpu` so that SVTRv2 runs on the GPU (now CPU, 8-14 ms).
- TensorRT FP16 for the detector and the char-YOLO readers on NVIDIA hardware (the team's kit measured the YOLOv8n-576
  detector at 194 FPS with TensorRT FP16 on a T4).
- Read all plates of a frame in one batch (already done in `read_crops`) and skip reading locked tracks in video mode
  (already done).

---

## Checklist for the final report

- [ ] Keep every choice on val; report test once; give McNemar p for every "A beats B" claim.
- [ ] Report plate exact **and** character accuracy (the proposal's metric).
- [ ] Say which numbers come from synthetic data (demo video, synthetic rain/fog).
- [ ] Name papers, not internal numbers, in course reports.
