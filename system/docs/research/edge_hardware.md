# Edge AI hardware for a roadside Bangla LPR unit (detector ~YOLOv8n/YOLO26n at 576-640 px + small char-YOLO / CTC OCR)

Research done 31 Oct 2026. "Accessed 31 Oct 2026" applies to every web page below unless a different date is given.
Ultralytics' current benchmark pages now publish **YOLO26n** (5.3 MB PyTorch file, same nano class as YOLOv8n/YOLO11n) instead of YOLOv8n; YOLOv8n numbers come from Hailo, Coral, Habr, Seeed and PhotonVision pages. FPS values marked "derived" are 1000 / (ms per image) for the network only (pre/post-processing and OCR excluded).

## Comparison table: which edge device fits a roadside Bangladesh LPR unit?

### Takeaway
On published numbers, every accelerator option (Jetson Orin Nano Super, Hailo-8L/8/10H on a Pi 5, RK3588 NPU, Intel Core Ultra with OpenVINO) runs a nano detector at 640 px far faster than the 25-30 FPS a roadside camera delivers. A bare Pi 5 CPU (~15 FPS with NCNN) or an Intel N100 (~4-12 FPS) is marginal. The real choice depends on price after NVIDIA's July 2026 price rise, on Bangladeshi availability (very thin), and on power for a solar pole. A Pi 5 + Hailo HAT draws about 7-8 W for the whole system, a Jetson Orin Nano Super up to 25 W.

### Cited Findings

**Dense comparison table** (detector throughput is for the network only, 640 px, batch 1 unless noted)

| Device | List price USD (date) | BD price seen (BDT, 31 Oct 2026) | Power (source) | Nano-detector speed (source) | INT8 accuracy cost (source) | Software stack |
|---|---|---|---|---|---|---|
| Jetson Orin Nano Super 8GB dev kit | $249 at launch (Dec 2024) → **$399** (NVIDIA FAQ, reported 22 Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found (Techshop BD search "jetson" = 0 products; RoboticsBD page blocked) | 7 / 15 / 25 W modes; 67 TOPS sparse INT8 [NVIDIA blog, 17 Dec 2024](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) | YOLO26n TensorRT FP16 4.57 ms (~219 FPS derived), INT8 3.80 ms (~263 FPS), ONNX 15.76 ms, PyTorch 15.60 ms; JP6.1, Ultralytics 8.4.33 [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) | mAP50-95 0.480 (FP16) → 0.449 (INT8) [same] | JetPack, CUDA, TensorRT, ONNX Runtime-GPU |
| Jetson Orin NX 16GB (module; J4012 carrier in test) | Module $599 → **$999** (1KU+, Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found | Not given in fetched sources (see Gaps) | YOLO26n TRT FP16 4.13 ms (~242 FPS), INT8 3.49 ms (~287 FPS); JP6.0/5.1.3 [Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/) | 0.4789 → 0.4489 [same] | Same as above |
| Jetson Orin NX 8GB module | $399 → **$649** (Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found | Not fetched | Not benchmarked on the Ultralytics page | n/a | Same |
| Jetson Orin Nano 4GB / 8GB modules | $199 → $349; $299 → $399 (Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found | Not fetched | Not benchmarked separately | n/a | Same |
| Jetson AGX Orin dev kit / 64GB module | Dev kit $1,999 → **$3,499**; 64GB module $1,599 → $2,999; 32GB $899 → $1,799 (Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found | Not fetched | YOLO26n TRT FP16 2.62 ms (~382 FPS), INT8 2.30 ms (~435 FPS); JetPack 7.2 [Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/) | 0.4789 → 0.4640 [same] | Same |
| Jetson AGX Thor dev kit | $3,499 → $5,499 (Jul 2026) [CNX](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101) | None found | Not fetched | YOLO26n TRT FP16 1.39 ms, INT8 1.52 ms (INT8 slower) [Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/) | 0.4797 → 0.4273 (−5.2 pts) [same] | Same (overkill for one lane) |
| Raspberry Pi 5 (CPU only) | 8GB RRP $80 (element14 FAQ, date not shown) [element14](https://community.element14.com/products/raspberry-pi/f/raspberry-pi-5-faq/53723/how-much-does-the-raspberry-pi-5-cost) | **4GB ৳30,500; 8GB ৳35,650** ("Official UK Version, authorized reseller"); active cooler ৳960; 27 W PSU ৳2,950; stock not shown [Techshop BD search](https://www.techshopbd.com/search?q=raspberry%20pi%205) | 10.2-10.6 W whole system during CPU-only inference [Jeff Geerling, 2026](https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2) | YOLO26n 640: NCNN 67.03 ms (~15 FPS), MNN 91.87, OpenVINO 104.55, ONNX 125.99, PyTorch 299.09 ms; Ultralytics 8.4.108 [Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) | FP32 only on this page | NCNN (recommended by Ultralytics), ONNX Runtime, OpenVINO CPU, MNN, LiteRT |
| Pi 5 + AI Kit / AI HAT+ 13 TOPS (Hailo-8L) | Not fetched for 13 TOPS (see Gaps) | Not found at Techshop BD | Hailo power limit ~3 W; whole Pi 5 + Hailo system 7.2-7.6 W [Jeff Geerling](https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2) | YOLOv8n 201 FPS (bs 1), 437 FPS (bs 8), measured on an x86 i5-9400 host over PCIe Gen3 x4, DFC v2.20.0 [Hailo Model Zoo H8L](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst); on a real Pi 5 Seeed reports ~136.7 FPS for YOLOv8n at batch 8, 640 px [Seeed wiki](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/) | YOLOv8n mAP 37.0 → 36.4; YOLO11n 39.0 → 37.5 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst) | HailoRT, TAPPAS / rpicam-apps, Hailo Dataflow Compiler |
| Pi 5 + AI HAT+ 26 TOPS (Hailo-8) | **$110** list; India ₹14,999 + GST (Probots); UK £105.60 (Pi Hut) [search summary of Neowin / Probots / Pi Hut](https://www.neowin.net/news/the-new-raspberry-pi-ai-hat-delivers-26-tops-of-performance-for-110/) | Not found | ~as above | YOLOv8n 1036 FPS (bs 1 and 8), x86 host PCIe Gen3 x4 [Hailo Model Zoo H8](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst); YOLOv8m 50-60 FPS on Pi [raspberry.tips](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-llm) | YOLOv8n 37.0 → 36.4 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst) | HailoRT |
| Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS, 8 GB RAM) | $130 in most reports, but the official product page says "$200" (conflict) [raspberrypi.com](https://www.raspberrypi.com/products/ai-hat-plus-2/), [search summary](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-llm) | Not found | Chip ~2.5 W in a comparison table [Jeff Geerling](https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2); ~4.5 W chip [raspberry.tips](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-llm) | YOLOv8m > 60 FPS, YOLOv8l 30-40 FPS (no YOLOv8n figure) [raspberry.tips](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-llm) | Not found | HailoRT (built for on-device LLM/VLM; overkill for LPR) |
| Pi 5 + Sixfab AI HAT+ (DEEPX DX-M1M) | Not fetched | Not found | Not fetched | YOLOv8l 50.1 FPS, "156% ahead" of Pi AI HAT+ (Hailo-8) in Sixfab's own test [Sixfab](https://sixfab.com/?p=202764) | Not found | DEEPX SDK |
| Google Coral USB Edge TPU (+ Pi 4/5) | Not fetched | Not found | Not fetched | YOLOv8n 320 px: 32.2 ms std / 26.7 ms max-freq runtime (~31-37 FPS); 512 px: 73.5 / 60.7 ms (~14-16 FPS); board per row unclear [Ultralytics Coral guide](https://docs.ultralytics.com/guides/coral-edge-tpu-on-raspberry-pi/) | Full-integer quant only; no accuracy numbers given [same] | TFLite + Edge TPU compiler (x86-64 Linux only); "official Coral guide is outdated" [same] |
| Rockchip RK3588 NPU (Orange Pi 5 / Rock 5) | Not fetched | Not found | Not fetched | YOLOv8 (variant and input not stated) RKNN-Toolkit2 INT8 on Orange Pi 5: 17.26 ms (~58 FPS derived), mAP 0.3051 [PhotonVision](https://docs.photonvision.org/en/latest/docs/benchmarks/rknn-model-benchmarks.html); YOLOv8n 53.1 FPS on RK3588, 18.2 FPS on RK3566/68 (Rock 5, ncnn + NPU) [Q-engineering via search](https://github.com/Qengineering/YoloV5-NPU) | INT8 calibrated on only 20 COCO images in the PhotonVision test [PhotonVision](https://docs.photonvision.org/en/latest/docs/benchmarks/rknn-model-benchmarks.html) | RKNN-Toolkit2 / RKNN runtime, rknn_model_zoo |
| Intel Core Ultra 7 258V (Lunar Lake: CPU + Arc iGPU + NPU) | Not fetched | Not found | Not fetched | YOLO26n OpenVINO FP32 3.33 ms (~300 FPS), FP16 3.34, INT8 3.46 ms; PyTorch CPU 27.33 ms; device per row not stated; OpenVINO 2026.2.1 [Ultralytics OpenVINO](https://docs.ultralytics.com/integrations/openvino/) | 0.4764 → 0.4634 (INT8) [same] | OpenVINO (CPU / GPU / NPU) |
| Intel Core Ultra X7 358H (Panther Lake) | Not fetched | Not found | Not fetched | YOLO26n OpenVINO FP32 4.09 ms, INT8 4.24 ms [same](https://docs.ultralytics.com/integrations/openvino/) | 0.4762 → 0.4634 | OpenVINO |
| Intel Core Ultra 7 155H (Meteor Lake, no NPU listed) | Not fetched | Not found | Not fetched | YOLO26n OpenVINO FP32 9.13 ms (~110 FPS), INT8 5.79 ms (~173 FPS); PyTorch 33.19 ms [same](https://docs.ultralytics.com/integrations/openvino/) | 0.4758 → 0.4573 | OpenVINO |
| Intel N100 mini-PC (4 cores) | Not fetched | Not fetched | Not fetched | YOLOv8n 640 PyTorch 249.4 ms (4.0 FPS); NCNN INT8 9.6 FPS; a custom fine-tuned YOLO via ONNX 85.4 ms (11.7 FPS) FP32 / 75.6 ms (11.9 FPS) INT8 (Habr, 19 Jun 2024) [Habr](https://habr.com/en/articles/822917); LattePanda Mu (N100) YOLOv8n "approximately 4 to 7" FPS [LattePanda blog](https://lattepanda.com/blog-323173.html) | Not given | ONNX Runtime, OpenVINO, NCNN |
| Intel i7-12700H laptop | n/a | n/a | n/a | YOLOv8n PyTorch 103.6 ms (9.7 FPS); NCNN INT8 8.1 FPS (Habr, Jun 2024) [Habr](https://habr.com/en/articles/822917) | n/a | ONNX/OpenVINO numbers only in images there |
| Xeon 4th gen (reference) | n/a | n/a | n/a | YOLOv8n OpenVINO ~11.3 ms/img (older Ultralytics table, via search snippet) [Ultralytics OpenVINO](https://docs.ultralytics.com/integrations/openvino) | n/a | OpenVINO |

**Team's own measurements on this project (Bangla plates, Kaggle T4, 26 Oct 2026, from the project memory note `C:\Users\rokib\.claude\projects\D--498R\memory\weekly-reports.md`)** — these are internal, file-backed facts:
- YOLOv8n-576 plate detector: mAP50 99.0; TensorRT INT8 mAP50 98.1; INT8 5.6 ms vs TensorRT FP16 5.2 ms on a T4, so INT8 was not faster. — local note
- Slot OCR: 57.60 % exact plate match (char accuracy 88.58 %); **ONNX dynamic INT8 OCR: 37.33 % exact (−20 points)**. — local note
- End-to-end on 10 plates: 21.2 ms on T4 (47.2 FPS); 74.4 ms on a 4-thread CPU (13.4 FPS). — local note
- YOLO26 (NMS-free) INT8 on T4 TensorRT: −0.04 to −0.52 mAP50 vs FP16, not faster at batch 1; on a 4-thread CPU YOLO26-N end-to-end 15.9 + 0.3 ms vs YOLO11-N 18.5 + 7.7 ms (1.62x faster; network alone 1.16x). — local note

### Inferences
- A roadside camera gives 25-30 FPS (a 33-40 ms budget per frame). Every accelerator row above (Jetson, Hailo, RK3588, Core Ultra) fits detector + OCR in that budget with a wide margin. Pi 5 CPU (~15 FPS detector at 640 px, maybe ~18 FPS at 576 px because the pixel count is (576/640)² = 0.81x) and N100 (4-12 FPS) would need frame skipping plus a tracker. That is workable for slow urban traffic, risky for highway speeds.
- Price per lane after July 2026: a Jetson Orin Nano Super kit ($399 list, before Bangladeshi import markup) now costs roughly as much as a Pi 5 8GB ($80) plus AI HAT+ 26 TOPS ($110) plus a camera. On list prices, the Pi + Hailo route is cheaper and draws ~3x less power. Jetson's advantage is software: Ultralytics → TensorRT export works on the same CUDA/PyTorch stack the team already trains on, and FP16 needs no calibration.
- The Techshop BD Pi 5 8GB price (৳35,650) is far above the $80 RRP. At an **assumed ~122 BDT/USD (exchange rate not verified in this session)** that is about $290, roughly 3.6x RRP. If that markup carries over, a locally bought Hailo HAT or Jetson would cost much more than list. Importing (or asking RoboticsBD / Techshop for a quote) should be priced against this.
- The Hailo model-zoo FPS numbers come from an x86 host with PCIe Gen3 x4. A Pi 5 has a single PCIe lane, so real Pi numbers are lower (Seeed's 136.7 FPS at batch 8 vs the zoo's 437 FPS at batch 8 for the H8L supports this). Even so, they are still far above camera frame rate.
- For OCR, the team's own result (dynamic INT8 cost −20 points of exact-plate accuracy) means the recogniser should stay FP16 (Jetson/TensorRT, OpenVINO FP16) or be quantised only with calibration/QAT and re-checked. On NPUs that only run INT8 (Hailo, RKNN, Coral), this is a real risk for the OCR stage. One option is to run the detector on the NPU and the small OCR on the CPU in FP32/FP16.
- INT8 detector cost is small on Hailo (−0.6 mAP for YOLOv8n) and Intel (−1.3), larger on Jetson TensorRT for YOLO26n (−3.1 on Orin Nano Super, −5.2 on Thor). On the team's own T4 test, INT8 was not faster than FP16 at batch 1. So FP16 is the sensible default on Jetson.

### Gaps
- **No Bangladeshi listing found** for any Jetson, Hailo AI Kit/HAT, Coral, Orange Pi 5/RK3588, Axelera or Sima.ai product. Techshop BD returned 0 products for "jetson"; the RoboticsBD page returned HTTP 402; the Star Tech Pi 5 URL returned 404. Ryans and Star Tech were not reachable for these items. Prices must be got by direct quote.
- Power envelopes for Orin NX (8/16GB) and AGX Orin were not found in the fetched pages. The commonly quoted NVIDIA spec (about 10-25 W for Orin NX, 15-60 W for AGX Orin) was **not verified in this session**.
- No sourced YOLOv8n/YOLO26n FPS or price was found for **Axelera Metis** or **Sima.ai MLSoC**.
- No clean NPU-only YOLO number for Intel Core Ultra: the Ultralytics table does not say whether each OpenVINO row ran on the CPU, iGPU or NPU.
- Prices of the AI Kit / AI HAT+ 13 TOPS, Coral USB, Orange Pi 5 and N100 mini-PCs were not fetched.
- RK3588 numbers come from third parties (PhotonVision, Q-engineering). The official rknn_model_zoo performance page returned 404.

## Raspberry Pi 5 alone and with Hailo-8L / Hailo-8 / Hailo-10H

### Takeaway
A Pi 5 CPU runs a nano YOLO at 640 px in about 67 ms with NCNN (~15 FPS), which is borderline for live roadside video. Adding a Hailo-8L or Hailo-8 HAT moves the detector into the 100-1000 FPS class (published, on an x86 host) at a whole-system power of about 7-8 W.

### Cited Findings
- Pi 5, YOLO26n 640 px FP32: PyTorch 299.09 ms, TorchScript 353.20, ONNX 125.99, OpenVINO 104.55, LiteRT 123.30, MNN 91.87, **NCNN 67.03 ms** (mAP50-95 0.4784), ExecuTorch 144.83 ms. NCNN "delivers the best inference performance on Raspberry Pi devices"; Ultralytics 8.4.108 — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- Pi 5, ONNX: YOLO11n 6.79 FPS → YOLO26n 7.79 FPS (mAP 39.5 → 40.1) — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- The page warns that overclocking "increases heat generation and may lead to thermal throttling" — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- Raspberry Pi 4 for comparison: YOLOv8n PyTorch 2001 ms (0.5 FPS), NCNN INT8 2.0 FPS (Jun 2024) — [Habr](https://habr.com/en/articles/822917)
- Hailo-8L (DFC v2.20.0): YOLOv8n float mAP 37.0, hardware mAP 36.4, 201 FPS bs1, 437 FPS bs8; YOLOv8s 110 / 208 FPS; YOLO11n 157 / 370 FPS; host i5-9400, PCIe Gen3 x4, room temperature — [Hailo Model Zoo, HAILO8L](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)
- Hailo-8: YOLOv8n 1036 FPS (bs1 and bs8); YOLOv8s 491; YOLO11n 185 bs1 / 541 bs8; same host and conditions — [Hailo Model Zoo, HAILO8](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
- Pi 5 + AI Kit (Hailo-8L, 13 TOPS): YOLOv8n ~136.7 FPS at batch 8, 640x640, from a 240 FPS video (vendor test) — [Seeed wiki](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)
- AI HAT+ 26 TOPS list price $110 — [Neowin](https://www.neowin.net/news/the-new-raspberry-pi-ai-hat-delivers-26-tops-of-performance-for-110/); India ₹14,999 + GST — [Probots](https://probots.co.in/official-raspberry-pi-ai-hat-26-tops.html); UK £105.60 incl. VAT — [The Pi Hut](https://thepihut.com/collections/raspberry-pi-hats/products/raspberry-pi-ai-hat-plus)
- AI HAT+ 2 (Hailo-10H, 40 TOPS, 8 GB on-board RAM): Pi 5 system draws 7.2-7.6 W with the Hailo running vs 10.2-10.6 W for CPU-only inference; the Hailo power limit is ~3 W — [Jeff Geerling, 2026](https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2); price "$200" on the official page, $130 in most reports — [Raspberry Pi](https://www.raspberrypi.com/products/ai-hat-plus-2/)
- Hailo-10H YOLOv8m > 60 FPS (~4.5 W chip), YOLOv8l 30-40 FPS; Hailo-8 YOLOv8m 50-60 FPS — [raspberry.tips](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-llm)
- DEEPX DX-M1M HAT: YOLOv8l 50.1 FPS, 156 % ahead of the Hailo-8 AI HAT+ in Sixfab's own test — [Sixfab](https://sixfab.com/?p=202764)
- Techshop BD (31 Oct 2026): Pi 5 4GB ৳30,500, 8GB ৳35,650, official active cooler ৳960, 27 W PSU ৳2,950, aluminium passive case ৳1,750, 5MP night-vision Pi camera ৳4,840. No AI Kit/HAT listed — [Techshop BD](https://www.techshopbd.com/search?q=raspberry%20pi%205)

### Inferences
- For one lane, the Hailo-8L (13 TOPS) is already enough. The 26 TOPS Hailo-8 adds headroom for a character-YOLO second stage on the same chip. The Hailo-10H's extra RAM targets LLMs and is not needed for LPR.
- A Pi 5 + Hailo unit at ~7.5 W (plus camera) is the lowest-power accelerator option with published numbers. That matters most for solar sizing (see the solar section).
- A custom Bangla plate detector and OCR must be compiled with the Hailo Dataflow Compiler. The zoo numbers only cover COCO models; a custom char-YOLO should behave similarly to YOLOv8n, but a CTC/LSTM text recogniser may not map onto the chip (see Gaps).

### Gaps
- No sourced batch-1 YOLOv8n FPS on a real Pi 5 + Hailo-8L/8 (the Seeed figure is batch 8).
- Whether CRNN/LSTM/CTC recognisers compile and keep accuracy on Hailo (or RKNN) was not established in this session.
- No Bangladeshi price for any Hailo product.

## Jetson Orin Nano Super / Orin NX / AGX Orin

### Takeaway
On Jetson, TensorRT FP16 runs a nano YOLO at 640 px in 4.6 ms (Orin Nano Super) down to 2.6 ms (AGX Orin), and INT8 adds little speed but costs 1.5-3 mAP points. NVIDIA's unannounced July 2026 price rise (+33 % to +101 %) puts the cheapest dev kit at $399 list. No Bangladeshi seller listing was found.

### Cited Findings
- New prices in NVIDIA's developer FAQ (reported 22 Jul 2026): Orin Nano Super dev kit $249 → $399; Orin NX 16GB module $599 → $999; Orin NX 8GB $399 → $649; AGX Orin dev kit $1,999 → $3,499; AGX Orin 64GB module $1,599 → $2,999. "There was no specific announcement"; NVIDIA had earlier phased out some modules over high LPDDR4 prices; SparkFun still listed the Nano Super kit at $249 on backorder — [CNX Software](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101)
- Orin Nano Super: 67 TOPS sparse / 33 TOPS dense INT8 (was 40/20), power modes 7/15/25 W (was 7/15 W), memory bandwidth 102 GB/s (was 68); dev kit price $249, down from $499 (17 Dec 2024) — [NVIDIA developer blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)
- YOLO26n at 640 px (ms/img): Orin Nano Super PyTorch 15.60, ONNX 15.76, TRT FP32 7.53, **FP16 4.57**, INT8 3.80; Orin NX 16GB 13.90 / 14.18 / 7.01 / **4.13** / 3.49; AGX Orin 64GB 11.58 / 9.87 / 4.18 / **2.62** / 2.30; AGX Thor 7.39 / 6.58 / 1.90 / 1.39 / 1.52. Power mode not stated (the page recommends `nvpmodel -m 0` + `jetson_clocks`) — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- The Ultralytics guide says INT8 calibration "should use the target device", TensorRT engines are device-specific, and TensorRT 10.3.0 on JetPack 6 blocks INT8 builds with NMS-free or embedded-NMS outputs — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Third-party boxed Orin Nano Super computer (reComputer J3011) listed at $879 — [search summary, rcdrone](https://rcdrone.top/collections/edge-ai-computers)

### Inferences
- The TensorRT 10.3 INT8 restriction on NMS-free outputs matters for YOLO26/YOLOv10-style detectors on JetPack 6. Together with the small INT8 speed gain, this makes FP16 the practical choice.
- The Orin NX gives only ~10 % more nano-detector speed than the Orin Nano Super for ~2.5x the module price. It is only worth it for multi-camera junction poles.
- AGX Orin / Thor are far beyond a one-lane LPR need and above any plausible pole power budget.

### Gaps
- No BDT price or Bangladeshi stock for any Jetson (Techshop BD: 0 results; RoboticsBD page blocked).
- Watts for Orin NX / AGX Orin were not in the fetched sources.
- No measured whole-board wall power for a YOLO + OCR workload on the Orin Nano Super.

## Intel Core Ultra / N100 / i5 CPUs, Coral, RK3588, Axelera, Sima.ai

### Takeaway
Recent Intel Core Ultra laptops run a nano YOLO through OpenVINO in 3-9 ms. An Intel N100 manages roughly 4-12 FPS depending on runtime. Coral is limited to small inputs (320-512 px) with dated tooling. RK3588 reaches ~50-58 FPS with INT8 RKNN. No sourced figures were found for Axelera or Sima.ai.

### Cited Findings
- YOLO26n OpenVINO (OpenVINO 2026.2.1): Core Ultra 7 258V FP32 3.33 ms / INT8 3.46 ms; Core Ultra X7 358H 4.09 / 4.24; Core Ultra 7 155H 9.13 / 5.79; PyTorch CPU 27-33 ms. YOLO26 on the NPU is only supported on Core Ultra 2xxV / 3xx series and newer — [Ultralytics OpenVINO integration](https://docs.ultralytics.com/integrations/openvino/)
- Intel N100 (Habr, 19 Jun 2024, Ultralytics 8.2.28): YOLOv8n PyTorch 249.4 ms (4.0 FPS); NCNN INT8 9.6 FPS; custom YOLO via ONNX 85.4 ms (11.7 FPS) FP32, 75.6 ms (11.9 FPS) INT8. The article says OpenVINO INT8 and ONNX allow real-time yolov8n on the N100, but those numbers are only in images — [Habr](https://habr.com/en/articles/822917)
- LattePanda Mu (N100): YOLOv8n detection "approximately between 4 to 7" FPS — [LattePanda blog](https://lattepanda.com/blog-323173.html)
- Coral USB Edge TPU (Pi 4B 2GB / Pi 5 8GB): YOLOv8n 320 px 32.2 ms (std) / 26.7 ms (max-freq); 512 px 73.5 / 60.7 ms; YOLOv8s 512 px 149.6 / 125.3 ms. Export only on x86-64 Linux; the model must end in `_edgetpu.tflite`; "the official Coral guide is outdated" — [Ultralytics Coral guide](https://docs.ultralytics.com/guides/coral-edge-tpu-on-raspberry-pi/)
- RK3588 (Orange Pi 5, RKNN-Toolkit2 INT8, 20 calibration images): YOLOv5 15.03 ms (mAP 0.2243), YOLOv8 17.26 ms (0.3051), YOLO11 22.99 ms (0.3251); variant and input not stated — [PhotonVision RKNN benchmarks](https://docs.photonvision.org/en/latest/docs/benchmarks/rknn-model-benchmarks.html)
- RK3588 YOLOv8n 53.1 FPS, RK3566/68 18.2 FPS (Rock 5, Ubuntu 22.04, ncnn + NPU, INT8) — [Q-engineering, via search](https://github.com/Qengineering/YoloV5-NPU)
- Orange Pi 5 Max RK3588 ResNet18 244 FPS (4.09 ms) — [TinyComputers](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html)

### Inferences
- A used or refurbished Core Ultra laptop or mini-PC is the easiest "no new toolchain" option for a pilot (OpenVINO export from Ultralytics, FP16, no calibration). Its power draw is higher and less predictable than a Pi + Hailo, though.
- The N100 is roughly comparable to a Pi 5 CPU for this job, with better x86 software support. On its own it is not a strong pick for a 24/7 live roadside unit.
- Coral's 512-px cap is below the team's 576-640 px detector size, so it is a poor fit.

### Gaps
- No sourced Axelera Metis or Sima.ai YOLOv8n FPS, price or Bangladeshi availability.
- No N100 or Core Ultra mini-PC prices from Bangladeshi retailers (not searched due to tool budget).
- No measured Intel NPU-only FPS for a nano YOLO.

## INT8 quantisation pitfalls for small detectors and OCR

### Takeaway
Detectors lose little under calibrated INT8 (−0.6 to −3 mAP points; −5 on Thor). Text recognisers are the risk: the team's own dynamic-INT8 OCR lost 20 points of exact-plate accuracy, and PaddleOCR says calibrated quantisation-aware training is needed for "essentially no loss".

### Cited Findings
- Hailo INT8: YOLOv8n mAP 37.0 → 36.4; YOLO11n 39.0 → 37.5 — [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)
- TensorRT INT8 YOLO26n: Orin Nano Super 0.480 → 0.449; AGX Orin 0.4789 → 0.4640; Thor 0.4797 → 0.4273 — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- OpenVINO INT8 YOLO26n: 0.4764 → 0.4634 (258V) — [Ultralytics OpenVINO](https://docs.ultralytics.com/integrations/openvino/)
- PaddleOCR / PaddleSlim: quantisation converts FP32 to INT8 "with essentially no loss of accuracy", online quantisation training beats offline, and quantised parameters only speed up on a runtime that supports INT8 (TensorRT recommended) — [PaddleOCR quantization docs](https://www.paddleocr.ai/v2.10.0/ppocr/model_compress/quantization.html)
- OpenCV Zoo warns its INT8 PP-OCRv3 text-detection model "may produce unstable results due to some loss of accuracy" and ships a block-quantised (block_size=64) variant — [OpenCV Zoo on Hugging Face](https://huggingface.co/opencv/text_detection_ppocr)
- Team result: ONNX **dynamic** INT8 slot-OCR 37.33 % vs 57.60 % exact (−20 pts); YOLOv8n-576 TRT INT8 mAP50 98.1 vs 99.0 and not faster than FP16 on T4 (26 Oct 2026) — local note

### Inferences
- Dynamic INT8 (weights only, no calibration) is the worst case for small CTC/attention recognisers. Static INT8 with calibration on real Bangla plate crops, or QAT, should be tried before ruling INT8 out on NPUs. Otherwise keep the OCR in FP16 or FP32 on a CPU/GPU.
- Calibration set size matters: PhotonVision used only 20 COCO images for RKNN. Calibrating on in-domain night, rain and day plate crops is likely to matter more for Bangla plates.

### Gaps
- No published before/after INT8 accuracy for CRNN or char-YOLO plate OCR on Hailo, RKNN or Coral was found.

## Power budget and solar sizing for a 24/7 pole in Bangladesh

### Takeaway
Bangladesh gets about 3 (December) to 5 (April) kWh/m²/day, with a monsoon dip from May to September. A ~15 W Pi 5 + Hailo + camera unit needs roughly a 150-200 W panel and a ~100-120 Ah 12 V LiFePO4 battery for 3 days of autonomy. A Jetson-based unit needs about twice that. The sizing figures are calculated here, not taken from a source.

### Cited Findings
- Measured Bangladesh irradiation (older 1993 study): April highest ~5 kWh/m²/day, December lowest ~3 kWh/m²/day, monsoon effect visible May-September — [JMERD (BUET)](https://jmerd.me.buet.ac.bd/archive/volume_16/JMERD_paper_vol_16_page_41-46.pdf)
- Modelled PV yield (NASA POWER-based): Naogaon spring 5.98 kWh/kW/day, summer/autumn ~4.5-4.7, winter ~4.0; Kishorganj spring 5.78, summer 4.39, winter 3.92 — [profilesolar Kishorganj](https://profilesolar.com/locations/Bangladesh/Kishorganj/)
- Monsoon June-September "can dramatically reduce solar irradiance through persistent cloud cover" — [profilesolar](https://profilesolar.com/locations/Bangladesh/Kishorganj/)
- Device power: Pi 5 + Hailo whole system 7.2-7.6 W, Pi 5 CPU-only inference 10.2-10.6 W — [Jeff Geerling](https://jeffgeerling.com/blog/2026/raspberry-pi-ai-hat-2); Orin Nano Super max mode 25 W — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)
- Hailo zoo figures were measured at "room temperature"; Ultralytics warns about thermal throttling under overclocking or high-frequency Coral runtimes — [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst), [Ultralytics Coral guide](https://docs.ultralytics.com/guides/coral-edge-tpu-on-raspberry-pi/)

### Inferences (my own calculation, assumptions stated)
- **Assumptions (not sourced):** ANPR camera with IR ~6-10 W; 4G router ~3-5 W; PV derate 0.7 (heat, dust, MPPT and wiring losses); design month 3.5 sun-hours (between the December ~3 and the monsoon dip); 3 days of autonomy for monsoon cloud; LiFePO4 at 80 % depth of discharge.
- **Pi 5 + Hailo unit** ≈ 7.5 + 8 + 4 ≈ 20 W → 480 Wh/day → panel 480 / (3.5 × 0.7) ≈ 196 W (about a 200 W panel); battery 480 × 3 / 0.8 = 1.8 kWh ≈ 12.8 V × 140 Ah LiFePO4. With a 15 W average (IR LEDs only at night, detector duty-cycled) this falls to about 150 W and 100-120 Ah.
- **Jetson Orin Nano Super unit** at 15 W mode ≈ 15 + 8 + 4 ≈ 27 W → 650 Wh/day → ~265 W panel, ~2.4 kWh battery. At 25 W mode ≈ 37 W → ~360 W panel, ~3.3 kWh battery.
- Heat: a sealed IP66 box in direct Bangladeshi sun will run well above ambient. Passive heatsinks bonded to the enclosure wall, a sun shield, and a fanless design (no fans to clog in humidity) suit a Pi + Hailo better than a 25 W Jetson. Monsoon humidity argues for IP66 with a breather vent / desiccant and conformal-coated boards.

### Gaps
- No sourced Bangladesh-specific roadside enclosure temperatures or IP66 enclosure prices from BD sellers.
- Camera and router power figures above are assumptions; no ANPR camera datasheet wattage was fetched.
- Solar panel and LiFePO4 battery prices in BDT were not searched.

## Camera requirements for ALPR and ANPR camera prices

### Takeaway
Industry guides agree on a fast shutter (about 1/1000-1/2000 s), a shallow angle (≤15° ideal, ≤30° max), plate tilt within ±5°, about 120-140 px across the plate, and IR illumination at night. A dedicated Hikvision ANPR camera is listed in Bangladesh (Ryans) without a price; the same family sells for ₹78,000 in India. Ordinary Hikvision CCTV cameras in Bangladesh cost ৳1,300-6,450.

### Cited Findings
- Hikvision LPR installation guide: camera view within 30° of the ground and the travel path, plate tilt within ±5°, reduce the shutter to 1/1000 or 1/2000 depending on speed and light, dim the IR LEDs at night if the plate is overexposed — [Hikvision LPR installation guide](https://cdn.connectec.uk/Files/Products/CCTV/Camera/Hikvision_Trade/installation_guide_of_hikvision_lpr_camera.pdf)
- Trassir checklist: plate width 120-140 px horizontally; angles ≤15° (≤30° max) — [Trassir camera setup checklist](https://storage.trassir.com/files/Camera%20setup%20checklist%20for%20AutoTRASSIR.pdf)
- Sighthound: target 75 pixels per foot across the plate area; "30fps with 1/100 Shutter speed is a good starting point", tuned to plate speed — [Sighthound ALPR camera guidelines](https://support.sighthound.com/kb/article/223-alpr-camera-guidelines). This conflicts with the much faster 1/1000-1/2000 s advised by Hikvision and IPCamTalk users.
- UK CCTV blog: 15-30° from perpendicular to cut headlight glare; 0° (straight on) reflects headlights and overexposes the plate — [cctvblog](https://cctvblog.mataroa.blog/blog/shutter-speed-settings-for-number-plate-capture/)
- IPCamTalk users suggest shutters like 1/2000 s; one Plate Recognizer user reports accuracy falls as plate angle grows — [IPCamTalk "New to LPR?"](https://ipcamtalk.com/threads/new-to-lpr-considerations-before-you-begin.52233/post-502628)
- Hanwha publishes an LPR quick reference guide (not read in detail) — [Hanwha LPR Quick Reference v9](https://customvideosecurity.com/datasheets/Hanwha%20LPR%20Quick%20Reference%20Guide%20v9.pdf)
- Prices: Ryans (Bangladesh) lists Hikvision iDS-TCM403-BI-UHK 4MP IR ANPR bullet (no price in snippet) — [Ryans IP cameras](https://www.ryans.com/category/security-camera-panel-ip-camera?page=8); IndiaMART lists Hikvision iDS-TCM403-BI at ₹78,000 per piece — [IndiaMART](https://m.indiamart.com/proddetail/ip-camera-27632006330.html); Hikvision CC cameras in Bangladesh ৳1,300-4,960 — [Tech Land BD](https://www.techlandbd.com/hikvision-cc-camera); 2MP turret ৳1,450 — [Star Tech](https://www.startech.com.bd/hikvision-ds-2ce76d0t-itpf-fixed-turret-camera); 6MP ColorVu bullet ৳6,450, 2MP ColorVu IP ৳4,273 — [BDStall](https://www.bdstall.com/brand/hikvision/)

### Inferences
- Bangla plates are two-line with small Bangla glyphs (district name plus series letter on the top line). The 120-140 px plate-width guidance is for Latin plates; Bangla plates likely need the upper end or more. The team's 576-640 px detector input on a 1080p or 4MP frame should be checked against that.
- A cheap non-ANPR 4MP IP camera (৳4,000-6,500) with manual shutter control, plus a separate 850 nm IR illuminator, may be enough for a pilot. Dedicated ANPR cameras cost many times more (₹78,000 in India for the iDS-TCM403-BI). Night performance is the main risk: the team's slot OCR fell to 27.7 % at night (local note).

### Gaps
- Plate Recognizer's own camera guide URL returned 404, so its official numbers are not included.
- No BDT price was found for any dedicated ANPR camera (Hikvision, Dahua, Axis).
- Dahua and Axis ANPR spec sheets were not fetched.
