# Newest research directions (2024-2026) for turning a working Bangla LPR system into a publishable solution

Context assumed (from the brief, not re-verified here): team has YOLOv8n/v9s/v12s plate detectors (~98-99 % mAP50), character-YOLO readers (~87-88 % plate exact), SVTRv2 and CRNN readers (73-82 %), a 4-reader vote (~91 % plate exact), and found HVI-CIDNet low-light enhancement, SAHI slicing, LPLCv2 legibility gating and single-image super-resolution give little or no gain on their data. Today = 31 Oct 2026. Every paper below was found and opened (abstract or full text) during this search unless marked "snippet only". "T4 difficulty" ratings are my own judgement (Inferences), not from the sources.

## 1. Multi-frame / video plate recognition (temporal fusion, tracking + voting, multi-frame SR)

### Takeaway
This is the strongest, best-documented 2025-2026 direction: the ICPR 2026 LRLPR competition (99 valid teams) showed the common factor among the top entries was exploiting the 5-frame track (logit summing, quality-weighted fusion, character voting, temporal Transformer), and two of the top three used **no super-resolution at all** — which matches the team's finding that single-image SR does not help. Gains from fusing frames are large on hard low-resolution data (+12 to +14 pts on UFPR-SR-Plates; 14 % → 86 % on dash-cam sequences in MF-LPR²), but no Bangla multi-frame plate dataset exists, so building one is itself a contribution.

### Cited Findings
- **ICPR 2026 Competition on Low-Resolution License Plate Recognition** (Laroca, Nascimento, ... Menotti; arXiv 2604.22506, v2 Oct 2026; presented at ICPR 2026, Lyon). Dataset LRLPR-26: 20,000 training tracks and 3,000 test tracks, each training track = 5 LR + 5 HR images of the same plate. 269 teams from 41 countries registered, 99 submitted valid blind-test entries. Winner 82.13 % recognition rate; four teams > 80 %; mean of all participants 61.3 %; top 10 within 3.13 pts — [arXiv 2604.22506](https://arxiv.org/abs/2604.22506); [HTML v2](https://arxiv.org/html/2604.22506v2)
- ICPR 2026 top-5 methods (from the paper): 1st DLmath (Korea Univ., 82.13 %) — HATFIR/MambaIRv2 SR jointly trained with GP-LPR + Transformer OCR in a teacher-student (EMA) setup, **logits of all 5 LR frames summed before decoding**, 3-model logit ensemble, added public RodoSol-ALPR/UFPR-ALPR/OpenALPR-BR data. 2nd AIO_JiangnamCoffee (81.73 %) — **no SR**; STN + SE-ResNet + Transformer + CTC, CNN attention module estimating per-frame quality for weighted 5-frame fusion, layout (Brazilian/Mercosur) constraints enforced, HR frames synthetically degraded to double training data. 3rd OpenOCR (Fudan, 80.17 %) — **no SR**; SVTRv2-AR, 4 models × 5 frames = 20 predictions fused by **character-level majority vote**, pre-training with diffusion-synthesised TextSSR data. 4th CAP2 (80.10 %) — extends MF-LPR² restoration, keeps 5 restored candidates, test-time augmentation, Optuna-weighted position-wise ensemble. 5th UIT-MeoBeo (79.83 %) — PE-Core-L encoder + 2-layer temporal Transformer, dual heads for text + layout, fixed 7-char decoding with position-wise letter/digit masks — [arXiv 2604.22506 HTML](https://arxiv.org/html/2604.22506v2)
- ICPR 2026 paper conclusions: no single dominant design (with and without SR both win); "how well they exploited the five LR frames" was the consistent factor; layout constraints in decoding helped; confidence calibration ("Confidence Gap") varied widely between teams with similar accuracy (2.38 % vs 14.86 %); the paper does **not** quantify single-frame vs multi-frame gain — [arXiv 2604.22506 HTML](https://arxiv.org/html/2604.22506v2)
- Korea University press release: 1st place 82 %, "537 AI researchers" took part, only 4 teams ≥ 80 % (self-reported, consistent with the paper) — [Korea Univ. TLO news](https://tlo.korea.ac.kr/en/news/177)
- **Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark** (Nascimento, Lima, Ribeiro, Schwartz, Laroca, Menotti; arXiv 2505.06393, 2025; Journal of the Brazilian Computer Society). UFPR-SR-Plates = 10,000 tracks, 100,000 real paired LR/HR plate images (5 LR far + 5 HR near per vehicle), Brazilian + Mercosur layouts, public. Abstract (latest version): LR 1.7 % → 31.1 % with LCDNet + MVCP → 44.7 % when OCR outputs of five super-resolved images are fused (+13.6 pts over single SR image) — [arXiv abs](https://arxiv.org/abs/2505.06393)
- UFPR-SR-Plates v1 full-text table (numbers differ slightly from the abstract version): LR 2.2 %, HR 85.2 % (single frame, GP_LPR OCR); five SR models (SR3, Real-ESRGAN, LPSRGAN, PLNET, LCDNet); LCDNet single frame 29.9 % → 5-frame fusion 35.7 % (Highest Confidence) / 40.7 % (sequence Majority Vote) / 42.3 % (Majority Vote by Character Position, MVCP). Character-position voting beat sequence voting and highest-confidence for every SR model — [arXiv HTML v1](https://arxiv.org/html/2505.06393v1). NOTE: v1 table (42.3 %) vs abstract (44.7 %) disagree; use the latest version when citing.
- **MF-LPR²: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow** (Na, Oh, Cho, Kim, Cho, Choi, Kim; CVIU vol. 256, 2025; arXiv 2508.14797). Dash-cam footage; optical flow alignment with filtering/refinement of wrong flow; new RLPR dataset of 200 LQ sequences + pseudo-GT. Recognition 86.44 % vs best single-frame LPR 14.04 % vs best previous multi-frame LPR 82.55 % (11 baselines); argues generative priors add artefacts — [arXiv 2508.14797](https://arxiv.org/abs/2508.14797)
- **HAT Super-Resolution and a PARSeq+CLIP4STR Voting Ensemble for Extreme In-the-Wild License Plate Recognition** (K. & K. Sivarama Krishnan; arXiv 2607.08896, Jul 2026) — entry to the **ICIP 2026 Grand Challenge on Extreme In-the-Wild LP Super-Resolution (XLPSR)**; scoring +2 correct / −1 wrong / 0 abstain; confidence-weighted character voting that abstains on uncertain positions; 9.73 wECR on public validation leaderboard; 1.7 s per sequence on RTX 3090 — [papers.cool summary](https://papers.cool/arxiv/2607.08896); [arXiv PDF](https://arxiv.org/pdf/2607.08896)
- XLPSR was held at IEEE ICIP 2026 (Tampere, Finland) (LinkedIn post, low-authority) — [LinkedIn](https://de.linkedin.com/in/cem-eteke)
- **Leveraging Model Fusion for Improved License Plate Recognition** (Laroca et al.; CIARP 2023, arXiv 2309.04331 — 2023, older than the target window but directly relevant to the team's 4-reader vote): fusing up to 12 OCR models raised mean intra-dataset recognition from 92.4 % (best single) to 97.6 %; cross-dataset from 87.6 % to > 90 %; **sequence-level majority vote beat character-level vote and highest-confidence**; recommends 4-6 models as the speed/accuracy sweet spot — [arXiv 2309.04331](https://arxiv.org/pdf/2309.04331)

### Inferences
- **Direction 1A — "Bangla multi-frame plate benchmark" (high novelty, T4: easy-medium).** Record road video in Dhaka, run the existing YOLO detector + a tracker, store 5-10 crops per track, and report single-frame vs track-level voting (HC / MV / MVCP) for every reader the team already has. UFPR-SR-Plates and LRLPR-26 show this kind of dataset + benchmark is publishable on its own (JBCS, ICPR competition). Main cost is data collection and labelling one string per track, not GPU.
- **Direction 1B — temporal fusion inside the reader (T4: medium).** The ICPR winners' cheapest trick (sum per-frame logits before CTC/attention decoding, 1st place) or quality-weighted attention over frames (2nd place) can be bolted onto the team's CRNN/SVTRv2 and trained on a T4; the 300M-parameter temporal Transformer of 5th place is heavier.
- The team's "SR/enhancement does not help" result is consistent with 2nd/3rd place at ICPR 2026 (no SR) and with MF-LPR²'s warning about generative artefacts; framing their negative SR result next to these papers strengthens a paper.
- Their 4-reader vote (~91 %) is the single-frame analogue of OpenOCR's 20-prediction character vote; extending vote-over-readers to vote-over-readers×frames is a natural, low-cost extension. Laroca 2023 found sequence-level vote > character-level vote across models, while UFPR-SR-Plates found character-position vote > sequence vote across frames — testing which holds for Bangla two-line plates is a concrete experiment.

### Gaps
- The ICPR 2026 paper does not report an ablation of single-frame vs multi-frame for the top teams, so the exact fusion gain on LRLPR-26 is unknown.
- No official XLPSR leaderboard or rules page was found; only one participant's paper.
- Gains on *already-readable, mid-resolution* plates (the team's regime, ~91 % single-frame) were not reported by any source; the big gains above are on low-resolution/degraded data.

## 2. Vision-language models for plates, and generative/synthetic plate data

### Takeaway
2025-2026 evidence shows (a) a small open VLM fine-tuned on only 600 synthetic plates can reach 87.6 % plate accuracy and edge out GPT-4o (86 %), (b) VLMs work best as a **confidence-gated fallback** for hard plates rather than the primary reader, and (c) synthetic data (templates, character permutation, GAN, diffusion) reliably helps, especially with small real training sets. For Bangla, PaddleOCR-VL-1.5 (0.9B) added Bengali in 2026 and a Bangla plate paper already uses a ViT + BanglaBERT encoder-decoder, but no Bangla plate VLM benchmark exists.

### Cited Findings
- **Multitasking vision language models for vehicle plate recognition with VehiclePaliGemma** (AlDahoul, Tan, Tera, Karim, Lim, Mishra, Zaki; Scientific Reports 15:26189, 2025). Fine-tuned PaliGemma on **600 synthetic Malaysian plates**, tested on 258 hard real plates: plate accuracy 69 % (base) → 87.6 %; character accuracy 90.92 % → 97.66 %. Other models on the same 258 plates: GPT-4o 86 %, GPT-4o-mini 85.7 %, Gemini 1.5 Flash 77.5 %, Claude 3.5 Sonnet 72.1 %, Gemini 1.5 Pro 71.7 %, Llama 3.2 11B 67.83 %, LLaVA-NeXT-34b 58.9 %, moondream2 39.5 %, KerasOCR 40.53 %. Speed: fine-tuned PaliGemma 0.135 s/plate locally (A100-80GB, ~7 FPS), moondream2 0.09 s, GPT-4o API 1.6 s. Full fine-tune (not LoRA), 5 epochs, batch 16, lr 2e-5. Multitask set (140 web images): VehicleGPT 97.16 %, VehiclePaliGemma 94.32 % plate-level; make/model/colour only qualitative — [PMC12274586](https://pmc.ncbi.nlm.nih.gov/articles/PMC12274586/)
- **BLPR: Robust License Plate Recognition under Viewpoint and Illumination Variations via Confidence-Driven VLM Fallback** (Auza Banegas et al.; arXiv 2604.09927, Apr/Jul 2026). Bolivian plates; geometric rectification + illumination correction; calls **Gemma3 4B only in ambiguous / low-confidence cases**; detector pretrained on Blender synthetic data; 89.6 % character-level accuracy; releases first public Bolivian plate dataset. Threshold policy, fraction routed, and latency not given in abstract — [arXiv 2604.09927](https://arxiv.org/abs/2604.09927)
- **Evaluating Vision-Language Models as a Zero-Shot Learning Alternative to YOLO and OCR for Nigerian License Plate Recognition** (Tijjani et al.; arXiv 2607.02025, Jul 2026): 88 real images, five VLMs (Gemini 2.0 Flash Exp, Qwen2.5-VL-7B-Instruct, GPT-4o, Claude 4 Sonnet, Llama 3.2 Vision 90B); Gemini and Qwen2.5-VL-7B "significantly outperform" the others on CER (numbers not in abstract) — [arXiv 2607.02025](https://arxiv.org/abs/2607.02025)
- **Layout-Independent License Plate Recognition via Integrated Vision and Language Models** (Shabaninia, Asadi-zeydabadi, Nezamabadi-pour; arXiv 2510.10533, Oct 2025): transformer vision model + iterative language modelling that learns plate formatting rules and does post-OCR refinement without layout classification; evaluated on IR-LPR, UFPR-ALPR, AOLP; no numbers in abstract — [arXiv 2510.10533](https://arxiv.org/abs/2510.10533)
- **PaddleOCR-VL** (0.9B VLM, arXiv 2510.14528, 2025) and **PaddleOCR-VL-1.5** (arXiv 2601.21957, Jan 2026): v1.5 adds Tibetan and **Bengali** recognition (109 → 111 languages); PaddleOCR docs ship a Bengali quick-start fine-tuning set for PaddleOCR-VL-0.9B; no Bengali-specific accuracy found — [PaddleOCR-VL-1.5 arXiv](https://arxiv.org/pdf/2601.21957); [PaddleOCR-VL SFT docs](https://ernie.readthedocs.io/en/develop/paddleocr_vl_sft.html); [PaddleOCR-VL arXiv](https://arxiv.org/pdf/2510.14528)
- Florence-2 is MIT-licensed and LoRA fine-tuning is documented (Roboflow); a public Florence-2 plate model card has no accuracy numbers (unverified) — [Roboflow Florence-2](https://inference-models.roboflow.com/models/florence2/); [HF model card](https://huggingface.co/jamyasir/florence-finetuned-license-plate-detection)
- **Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion: A Comprehensive Evaluation** (Laroca, Estevam, Moreira, Minetto, Menotti; IET Intelligent Transport Systems 19(1) e70086, 2025; arXiv 2601.07671): 16 OCR models × 12 public datasets; template-based generation, character permutation and GAN synthesis; combining them is "synergistic"; synthetic data "substantially boosts" intra- and cross-dataset accuracy and gives strong results with small fractions of real training data; reports end-to-end results above SOTA and commercial systems (no numbers in abstract) — [arXiv 2601.07671](https://arxiv.org/abs/2601.07671)
- **License Plate Images Generation with Diffusion Models** (Shpir, Shvai, Nakib; FAIA vol. 392 / ECAI 2024, arXiv 2501.03374): DDPM trained on Ukrainian plates; adding pseudo-labelled synthetic data gives **+3 % LPR accuracy** over baseline; 10,000 synthetic LP images released on Zenodo — [arXiv 2501.03374](https://arxiv.org/abs/2501.03374)
- **TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition** (Ye, Du, Tao, Chen; ICCV 2025, arXiv 2412.01137): region-centric text generation with position-glyph enhancement; TextSSR-F dataset (3.55 M instances per HF abstract; 0.4 M in arXiv v1 — sources disagree); STR models trained on it beat other synthetic sets; used by ICPR 2026 3rd place for pre-training — [ICCV 2025 open access](https://openaccess.thecvf.com/content/ICCV2025/html/Ye_TextSSR_Diffusion-based_Data_Synthesis_for_Scene_Text_Recognition_ICCV_2025_paper.html); [HF papers](https://huggingface.co/papers/2412.01137)
- **CharDiff-LP: A Diffusion Model with Character-Level Guidance for License Plate Image Restoration** (arXiv 2510.17330, Oct/Dec 2025): character-region-masked attention guidance; vs fine-tuned TCDM: CER 16.1 % → 15.1 %, plate-level LPR 63.7 % → 69.7 % — [arXiv 2510.17330](https://arxiv.org/abs/2510.17330v2)
- ICPR 2026 2nd and 5th place built extra training data by synthetically degrading HR frames (blur, noise, JPEG, down/up-scaling) — [arXiv 2604.22506 HTML](https://arxiv.org/html/2604.22506v2)

### Inferences
- **Direction 2A — confidence-gated VLM fallback for Bangla (T4: easy).** The team's 4-reader vote gives a free disagreement signal; route only disagreeing plates (~9 % of plates) to a small VLM (Gemma3-4B / Qwen2.5-VL-3B / PaliGemma, 4-bit, fits a 16 GB T4 by my estimate) and report accuracy-vs-latency. BLPR (2026) did this for Bolivia; nobody has done it for Bangla script.
- **Direction 2B — fine-tune a small VLM on Bangla plates (T4: medium).** VehiclePaliGemma did a full fine-tune on an A100-80GB; on a T4 this needs LoRA/QLoRA or a smaller model (Florence-2 base/large, PaddleOCR-VL 0.9B, which now lists Bengali). A head-to-head "char-YOLO vs SVTRv2 vs fine-tuned VLM on Bangla two-line plates, accuracy + ms/plate" is publishable because no such Bangla benchmark exists.
- **Direction 2C — synthetic Bangla plates for rare district names (T4: easy for font rendering, medium-hard for diffusion).** Template + character-permutation rendering with Bangla fonts (the Laroca 2025 recipe) is CPU-cheap and directly targets rare districts/letters; diffusion-based (TextSSR-style) Bangla glyph synthesis is riskier because diffusion text models are mostly Latin/Chinese-trained (inference — not verified for Bengali). Report gains on rare-class slices, not just overall accuracy.
- VLM hallucination risk is real for plates (see Section 4: SR3 hallucinating digits); any VLM route should be paired with grammar checks (Section 3).

### Gaps
- No published head-to-head of fine-tuned Qwen2.5-VL vs fine-tuned PaliGemma vs Florence-2 on the same plate dataset was found.
- No VLM plate-reading accuracy for Bangla/Bengali script was found; Bengali per-language numbers for PaddleOCR-VL-1.5 were not found.
- Nigerian VLM paper's CER/latency numbers were not visible in the abstract.
- No 2024-2026 paper on diffusion synthesis of Bangla (or any Indic) plate text was found.

## 3. Layout-independent / two-line recognition, LP-specific language models / grammar-constrained decoding, uncertainty & conformal "send to human" decisions

### Takeaway
Grammar/layout-constrained decoding is now standard among top systems (several ICPR 2026 top-5 teams enforced plate layout masks), and 2025 work integrates a language-model refinement step into the recognizer. For review routing, the 2026 field signals are (i) ICPR 2026's separate "Confidence Gap" metric and (ii) ICIP 2026 XLPSR's +2/−1/0 abstention scoring; split conformal prediction has been applied to document OCR/KIE (2025-2026) but **not yet to license plates** — an open, cheap, publishable niche.

### Cited Findings
- ICPR 2026: 2nd place enforced Brazilian/Mercosur layout constraints; 5th place used fixed 7-character output with position-wise letter/digit masks plus a layout-classification head, choosing between layout hypotheses by beam score and layout prior; organisers list "layout-aware decoding" and "confidence estimation" as future directions; Confidence Gap differed from 2.38 % to 14.86 % among teams with ~80 % accuracy — [arXiv 2604.22506 HTML](https://arxiv.org/html/2604.22506v2)
- Layout-independent VLM + iterative language modelling (arXiv 2510.10533, 2025) learns plate formatting rules from data without heuristics — [arXiv 2510.10533](https://arxiv.org/abs/2510.10533)
- **GP-LPR — Irregular License Plate Recognition via Global Information Integration** (Liu et al., MMM 2024, LNCS 14555): global perception module against attention drift; +6.7 % on irregular plates; code at github.com/MMM2024/GP_LPR; used as the OCR in UFPR-SR-Plates and by the ICPR 2026 winner — [Springer](https://link.springer.com/10.1007/978-3-031-53308-2_24); [arXiv 2505.06393](https://arxiv.org/pdf/2505.06393)
- Multi-line plates confuse transformer OCR through misaligned sequences (Indian trucks, 2022 — older) — [arXiv 2211.13194](https://arxiv.org/pdf/2211.13194)
- XLPSR entry: confidence-weighted character voting that **abstains** on uncertain positions, exploiting +2/−1/0 scoring — [papers.cool 2607.08896](https://papers.cool/arxiv/2607.08896)
- Split conformal prediction as a post-hoc layer on multimodal transformers for key-information extraction, using prediction-set size to auto-accept confident outputs and route uncertain ones to humans; calibrated sets expose layout-dependent failure modes (IJDAR 2026) — [Springer s10032-026-00572-y](https://link.springer.com/10.1007/s10032-026-00572-y)
- TSR-OCR-UQ (arXiv 2507.02009, 2025): conformal scoring per table cell on top of OCR, asks whether OCR confidence alone suffices to flag errors; code lamps-lab/TSR-OCR-UQ — [arXiv 2507.02009](https://arxiv.org/html/2507.02009v1)
- Caution: conformal prediction sets may be in tension with human-AI decision-making needs (Hullman, blog discussion, 2024) — [Statistical Modeling blog](https://statmodeling.stat.columbia.edu/2024/03/15/conformal-prediction-and-people/)

### Inferences
- **Direction 3A — Bangla plate grammar decoder (T4: easy).** Bangladeshi plates have a fixed structure (city/district word + vehicle-class letter on the top line; two-digit series + four-digit number on the bottom line — team should confirm against their labels). A constrained beam search / lexicon over district names + class letters applied to CRNN/SVTRv2 logits, and as a validity filter on the 4-reader vote, is cheap and likely to lift the weaker readers (73-82 %) the most.
- **Direction 3B — conformal "send-to-human" for plates (T4: trivial compute, high novelty).** Use a held-out calibration split; non-conformity score = 1 − vote agreement or 1 − min char probability; report the review rate needed to guarantee e.g. ≥ 99 % accuracy on auto-accepted plates, and compare with plain thresholding and the ICPR "Confidence Gap" metric. No plate-specific conformal paper was found, so this is a gap a student team can fill with existing models only.
- Selective prediction with XLPSR-style asymmetric scoring (+2/−1/0) is a ready-made evaluation protocol for Bangladesh toll/enforcement use cases where a wrong read costs more than no read.

### Gaps
- No 2024-2026 paper specifically on two-line/stacked plate recognition with numbers was found (2510.10533 does not say it handles two-row plates).
- No conformal-prediction or calibrated-uncertainty paper specific to license plates was found.
- No study measuring human reviewer performance with conformal sets for OCR was found.

## 4. Domain adaptation / test-time adaptation (night, rain, fog), event cameras, cross-country models

### Takeaway
Plate-specific test-time adaptation is essentially unexplored: TTA papers from 2024-2026 target object detection in bad weather or generic/industrial text recognition, not plates. Weather robustness in plate work is mostly augmentation, restoration or weather-aware detectors. Event-camera text recognition benchmarks appeared in 2025 (EventSTR) but **no event-camera plate dataset was found**. Cross-country work favours large synthetic + real multi-dataset training.

### Cited Findings
- **Billet Number Recognition Based on Test-Time Adaptation** (arXiv 2502.09026, Feb 2025): integrates TTA with prior knowledge, DB detector + **SVTR** recognizer, handles train/test distribution shift (industrial stamped numbers — closest TTA-for-text-codes analogue to plates) — [arXiv 2502.09026](https://arxiv.org/html/2502.09026v1)
- Hybrid test-time adaptation (HTTA) for out-of-distribution scene text in driving scenes (Xian et al., PRCV 2024) — [BUAA research page](https://research.buaa.edu.cn/en/publications/exploring-out-of-distribution-scene-text-recognition-for-driving-/)
- "Stratified Domain Adaptation" progressive self-training for STR (Le et al., WACV 2025; listing only) — [ML Anthology](https://mlanthology.org/authors/l/le-kha-nhat/)
- CD-Buffer: test-time adaptation for adverse-weather **object detection** (arXiv 2603.26092, Mar 2026) updates source models online from test streams only — [arXiv 2603.26092](https://arxiv.org/html/2603.26092v1)
- Weather-adaptive CNN on YOLOv10 for plate detection in bad weather (Sensors, Dec 2024) — [PMC11644901](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644901/)
- **Fog-Resilient Bangla Car Plate Recognition Using Dark Channel Prior and YOLO** (Nasim et al., WACV 2024 Workshops): dehazing → YOLOv8 → OCR on Bangla plates — [CVF open access](https://openaccess.thecvf.com/content/WACV2024W/WVLL/html/Nasim_Fog-Resilient_Bangla_Car_Plate_Recognition_Using_Dark_Channel_Prior_and_WACVW_2024_paper.html)
- "Foggy License Plates Worldwide" Mendeley dataset: 4,420 images incl. Bangladesh, Thailand, Saudi Arabia (synthetic fog on secondary data) — [Mendeley rgpddwxrx5](https://data.mendeley.com/datasets/rgpddwxrx5)
- **Mapping License Plate Recoverability Under Extreme Viewing Angles for Opportunistic Urban Sensing** (Adamenko et al.; arXiv 2604.23814, Apr 2026; MDPI AI 7(7):237): ~90-92 % of a 0-89° angle grid recoverable at ≥ 90 % plate OCR accuracy; restoration adds +15 to +38 exact-match pts at extreme angles on CCPD and RodoSol; Diffusion-SR3 **hallucinates plausible but wrong digits** at extreme angles — [arXiv 2604.23814](https://arxiv.org/pdf/2604.23814); [emergentmind](https://www.emergentmind.com/papers/2604.23814)
- **EventSTR: A Benchmark Dataset and Baselines for Event Stream based Scene Text Recognition** (Wang et al., arXiv 2502.09020, Feb 2025): 9,928 HD (1280×720) event samples, Chinese + English, targeting low light and motion blur; SimC-ESTR baseline (encoder + Q-former + memory + LLM); follow-up ESTR-CoT (arXiv 2507.02200) adds chain-of-thought with 16,222 reasoning pairs — [arXiv 2502.09020](https://arxiv.org/pdf/2502.09020); [Prophesee summary](https://www.prophesee.ai/2025/03/13/eventstr-a-benchmark-dataset-and-baselines-for-event-stream-based-scene-text-recognition/)
- Multinational synthetic + real fusion across 12 datasets improves cross-dataset LPR (IET ITS 2025) — [arXiv 2601.07671](https://arxiv.org/abs/2601.07671)
- **Global License Plate Dataset** (Agrawal; arXiv 2405.10949, Mar 2024): > 5 M images, 74 countries, characters + masks + corners + vehicle make/colour/model; no metrics in abstract; dataset link and whether Bangladesh is included not stated; CC BY-NC-ND — [arXiv 2405.10949](https://arxiv.org/abs/2405.10949)

### Inferences
- **Direction 4A — TTA for Bangla night/rain plates (T4: easy-medium).** Since HVI-CIDNet enhancement did not help, adapting the *reader* rather than the *image* is the logical next test: entropy-minimisation / BN-statistics TTA on CRNN/SVTRv2 using unlabeled night crops, plus pseudo-label self-training on their own night video. A "enhancement vs TTA vs synthetic-weather augmentation" comparison on Bangla night plates would be novel (no plate TTA paper found).
- **Direction 4B — cross-country transfer (T4: medium).** Pre-train readers on large multinational/synthetic data (Laroca 2025 recipe, GLPD if accessible) then fine-tune on Bangla; measure data-efficiency curves (10/25/50/100 % Bangla data).
- Event cameras: low priority for a Kaggle-only team — no hardware, no plate dataset; would require a frame-to-event simulator (not verified here).

### Gaps
- No TTA / unsupervised domain adaptation paper specific to license plates (2024-2026) was found.
- No event-camera license plate dataset or paper was found.
- Whether GLPD is downloadable and contains Bangladeshi plates is unknown.

## 5. Vehicle re-identification + plate fusion, make/model/colour recognition for cloned/fake plates

### Takeaway
2024-2026 academic resources now exist for joint vehicle attributes + plates (UFPR-VCR colour dataset 2024; UFPR-VeSV 2026 with make/model/type/colour **and** plate text), and VLMs can answer plate + attribute queries in one model, but peer-reviewed work explicitly detecting cloned plates via plate-vs-vehicle mismatch is thin (older journal paper, UK police report, patents, vendor claims). This is a real gap for Bangladesh, but evaluation is hard without registry (BRTA) data.

### Cited Findings
- **Toward Enhancing Vehicle Color Recognition in Adverse Conditions: A Dataset and Benchmark** (Lima, Laroca et al.; SIBGRAPI 2024, arXiv 2408.11589): UFPR-VCR, 10,039 images incl. night, occlusion, front/rear views; four models benchmarked; many errors at night — [arXiv 2408.11589](https://arxiv.org/pdf/2408.11589); [SBC](https://sol.sbc.org.br/index.php/sibgrapi/article/view/31701)
- **Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition** (Lima, Nascimento, Santos, Nascimento Jr, Laroca, Menotti; arXiv 2604.05271, Apr 2026; JBCS): UFPR-VeSV = 24,945 images of 16,297 vehicles, 13 colours, 26 makes, 136 models, 14 types; police surveillance incl. night IR; plate text + corners provided; annotations validated using plate info; explores joint FGVC + ALPR (cloned-plate detection not mentioned in abstract) — [arXiv 2604.05271](https://arxiv.org/abs/2604.05271)
- VehiclePaliGemma/VehicleGPT answer attribute prompts ("red cars", make) in one VLM; plate-level 94.32-97.16 % on 140 web images; attribute accuracy only qualitative (2025) — [PMC12274586](https://pmc.ncbi.nlm.nih.gov/articles/PMC12274586/)
- Counterfeit plate detection by reading plate + make/model/year/colour with deep learning and flagging stolen/random plates (Ağgül & Erdemir, Balkan J. Electrical & Computer Eng., 2022 — older; accuracy claim > 98.7 % not verified) — [IZU open access](https://openaccess.izu.edu.tr/entities/publication/3cdeb6ac-02c0-4310-8a7f-3db73536a1cf)
- UK N8PRP report: ML-classified vehicle characteristics compared with registration records as a "methodologically justifiable" false-plate detector (2021) — [N8PRP ANPR report](https://www.n8prp.org.uk/wp-content/uploads/sites/315/2021/10/ANPR-final.pdf)
- Limits: clones are often chosen to match the real car's make/model/colour (consumer source) — [carVertical](https://www.carvertical.com/gb/blog/what-is-number-plate-cloning); vendor systems claim registry make/model/colour mismatch alerts (vendor claim) — [scOS](https://scos.co.uk/features/dvla-vehicle-recognition/)
- Vehicle-Rear dataset pairs whole-vehicle features with HR plate patches for re-ID (2019, older) — [arXiv 1911.05541](https://arxiv.org/pdf/1911.05541.pdf)

### Inferences
- **Direction 5A — plate + vehicle-attribute consistency for Bangladesh (T4: medium).** Train a light colour/type classifier (Bangladeshi CNG, bus, truck, car, bike) on the team's own vehicle crops; demonstrate "plate says private car, vehicle is a truck/CNG" mismatch detection on simulated swaps. Publishable as a dataset + baseline; real cloned-plate evaluation needs BRTA registry access.
- **Direction 5B — appearance re-ID as fallback when plate is unreadable (T4: medium-hard).** Needs multi-camera data; most suitable if the team collects multi-camera video for Direction 1A anyway.

### Gaps
- No 2024-2026 peer-reviewed paper benchmarking plate + appearance fusion for unreadable plates was found.
- No Bangladeshi vehicle make/model/type dataset was found in this search.
- UFPR-VeSV benchmark accuracies were not visible in the abstract.

## 6. Privacy-preserving / federated learning, on-device continual learning, edge-cloud split inference

### Takeaway
Federated LPR exists mainly as preprints, student theses (including a BRAC University one) and conference papers; edge-cloud split inference and continual learning for plates are documented but mostly pre-2024. These directions are cheap to simulate on one T4 but are less likely to give high-impact results than Sections 1-3; their best use is as a system/deployment section of a paper.

### Cited Findings
- **PlateSegFL: a privacy-preserving license plate detection system using federated segmentation learning** (BRAC University; U-Net + FL; ~95 % F1 self-reported) — [BRAC DSpace](https://dspace.bracu.ac.bd/items/acac5760-93de-4377-8c18-437f1550d66f)
- **Asynchronous Federated Learning for Real-Time Multiple Licence Plate Recognition Through Semantic Communication** (CityU HK): cameras cannot share data for privacy; asynchronous FL across network cameras — [CityU Scholars](https://scholars.cityu.edu.hk/en/publications/asynchronous-federated-learning-for-real-time-multiple-licence-pl/)
- License plate recognition using federated learning (Manipal; only model updates sent to server) — [Manipal](https://researcher.manipal.edu/en/publications/license-plate-recognition-using-federated-learning/)
- FL for connected/automated vehicles survey lists plate detection/recognition among use cases (arXiv 2308.10407, 2023) — [arXiv 2308.10407](https://arxiv.org/pdf/2308.10407)
- Synthetic plate generation is motivated by GDPR-type limits on sharing real plate data (2024) — [arXiv 2501.03374](https://arxiv.org/abs/2501.03374)
- Auto-Split (2021): edge-cloud split of YOLOv3 for a deployed parking LPR system — [arXiv 2108.13041](https://arxiv.org/pdf/2108.13041)
- Continual learning (GDumb-style) for plate detection on edge devices approaches full-retraining results under limited resources (VISAPP 2023) — [SciTePress](https://scitepress.org/PublishedPapers/2023/116563)
- Bangladesh BRTA plate detector with a dynamic retraining pipeline for low-confidence cases (ICCIT 2024; arXiv 2609.06507) — [arXiv 2609.06507](https://arxiv.org/abs/2609.06507)
- Jetson Nano plate segmentation + recognition at ~2.2 FPS at 640×640 (Vietnam, Mar 2025) — [VJOL](https://vjol.info.vn/mostc/en/article/view/110491)

### Inferences
- **Direction 6A — federated Bangla LPR simulation (T4: easy).** Partition data by camera/location/day-night into clients, run FedAvg/FedProx (e.g. Flower) on one T4, report accuracy vs centralised and vs communication rounds. Novel for Bangla but incremental methodologically.
- **Direction 6B — confidence-triggered on-device continual learning (T4: easy-medium).** Combine the uncertainty router (Section 3) with replay-buffer fine-tuning on reviewed plates; measure forgetting on old districts. Fits a "human-in-the-loop Bangla LPR" paper.
- Edge-cloud split: the team's FP16/ONNX readers on device + VLM fallback in cloud (Section 2A) is a natural split-inference story.

### Gaps
- No 2024-2026 peer-reviewed benchmark of federated LPR with accuracy vs centralised numbers was found.
- No 2024-2026 plate-specific split-inference paper was found.

## 7. Bangla / Indic-specific work (Bangla OCR, Bangladeshi plate datasets 2024-2026, Indian/Nepali/Pakistani transfer)

### Takeaway
Bangla plate research 2024-2026 is still mostly single-image YOLO + OCR (EasyOCR, VisionEncoderDecoder with BanglaBERT, OCR-free 102-class YOLO); none uses multi-frame fusion, VLM fallback, grammar-constrained decoding or calibrated review routing — so Sections 1-3 are open for Bangla. New public data is small (e.g. 2,947-image Mendeley set, Nov 2024). Nepali (Devanagari, embossed plates) work is the closest Indic analogue; no 2024-2026 Pakistani plate paper was found.

### Cited Findings
- **A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision-Language OCR** (Hasin, Nishat, Islam, Al Hasan, Newaz; arXiv 2603.10267, Mar/May 2026; IEEE ICAD 2026): YOLOv8 two-stage adaptive training, localisation accuracy 97.83 %, IoU 91.3 %; VisionEncoderDecoder OCR, best **ViT + BanglaBERT: CER 0.1323, WER 0.1068**; "consistent" on an external dataset (no numbers); data/code release not confirmed — [arXiv 2603.10267](https://arxiv.org/abs/2603.10267)
- **A Cloud-Based Hybrid Model for Real-Time Detection of BRTA-Approved Licence Plates Using YOLO Tiny and Haar Cascade** (Suvra, Sanam; ICCIT 2024, arXiv 2609.06507): detection only, claims a publicly accessible BRTA-plate image + video dataset (no link on abs page) — [arXiv 2609.06507](https://arxiv.org/abs/2609.06507)
- Vehicle License Plate Detection Dataset V1 (Mendeley, 20 Nov 2024): 2,947 images, 5 classes, Dhaka/Ashulia/Narsingdi, Oct-Nov 2024 — [Mendeley nntnnzffw4](https://data.mendeley.com/datasets/nntnnzffw4)
- Older public Bangla data: Bangla LPDB-A (2021; 1,928 vehicle images, 2,662 plate crops) — [Zenodo 4718238](https://zenodo.org/record/4718238); Khulna Univ. Bangladeshi LPR character set (2022; 3,480/363/218 train/val/test) — [Dataset Ninja](https://datasetninja.com/bangladeshi-license-plate-recognition-character)
- OCR-free YOLOv8 Bangla plate reader with 102 character/numeral/district classes, > 3,000 images (2026 listing, venue unclear) — [VU publication](https://vu.edu.bd/publications/3151/license-plate-recognition-system-using-deep-learning-approach)
- IDAA 2025 (Atlantis Press): two-line Bangladeshi plates, YOLOv8x best detector, EasyOCR with custom preprocessing — [Atlantis IDAA-25](https://www.atlantis-press.com/proceedings/idaa-25/126024943)
- Fog-resilient Bangla plates (WACVW 2024) — [CVF](https://openaccess.thecvf.com/content/WACV2024W/WVLL/html/Nasim_Fog-Resilient_Bangla_Car_Plate_Recognition_Using_Dark_Channel_Prior_and_WACVW_2024_paper.html)
- **BornoDrishti: Leveraging Vision Encoders and Domain-Adaptive Learning for Bangla OCR on Diverse Documents** (Islam et al., EACL 2026 Industry): first unified ViT OCR for printed + handwritten Bangla, claims SOTA on printed (numbers not seen) — [ACL Anthology preview](https://preview.aclanthology.org/credits/2026.eacl-industry.20)
- PaddleOCR-VL-1.5 adds Bengali (Jan 2026) — [arXiv 2601.21957](https://arxiv.org/pdf/2601.21957); `bangla-ocr` PyPI package v1.0.3 (Nov 2025) — [pepy](https://pepy.tech/project/bangla-ocr)
- Nepali: **Character Recognition of Nepali Number Plate** (arXiv 2606.28946, 2026): YOLO plate/char detection + CNN over 34 Devanagari characters, up to 93 %, extra training on embossed plates — [arXiv 2606.28946](https://arxiv.org/pdf/2606.28946); embossed Nepali plates, 5,247 images, YOLOv8 + PaddleOCR bilingual, ~91.2 % (Research Square preprint, unreviewed) — [Research Square](https://www.researchsquare.com/article/rs-9212377/v1); YOLOv8 + CNN on Raspberry Pi 4B, 0.995 mAP50, 99.17 % char accuracy on small data (572 plates) — [NepJOL](https://www.nepjol.info/index.php/jhce/article/view/100214)
- Indian: IEEE DataPort datasets targeting hand-painted fonts and multi-line square plates for merging multi-line text, and VisionPlate ANPR crops for TrOCR training (dates unconfirmed) — [IEEE DataPort](https://ieee-dataport.org/documents/indian-license-plate-recognition-dataset-yolo-ocr); [VisionPlate](https://ieee-dataport.org/documents/visionplateanpr-indian-vehicle-classification-license-plate-detection-and-ocr-dataset)
- Transfer learning across Indian-language scene text gave +23 % word recognition on MLT-17 Bangla (2022, older) — [arXiv 2201.03180](https://arxiv.org/pdf/2201.03180)
- Pakistani: only 2021-2022 works found (e.g. ~6,000-image Pakistani LP dataset, Sensors 2021) — [PMC8622297](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8622297/)

### Inferences
- The team's numbers (~91 % plate exact via vote, char-YOLO 87-88 %) are already stronger than the most recent Bangla VED paper's reported CER/WER (0.1323 / 0.1068), so a Bangla paper should lead with a *new capability* (multi-frame track fusion, calibrated review routing, VLM fallback, grammar decoding) plus a released benchmark, not with yet another YOLO + OCR pipeline.
- **Shortlist ranked by (novelty × feasibility on a free T4):**
  1. Bangla multi-frame track benchmark + track-level voting/logit fusion (Section 1) — high novelty, easy-medium.
  2. Conformal / calibrated "send-to-human" routing on top of the 4-reader vote (Section 3B) — high novelty, trivial compute.
  3. Bangla plate grammar-constrained decoding (Section 3A) — medium novelty, easy; combine with 2.
  4. Confidence-gated small-VLM fallback, plus LoRA fine-tune of a ≤ 4B VLM / PaddleOCR-VL on Bangla plates (Section 2A/2B) — medium-high novelty, easy (fallback) to medium (fine-tune).
  5. Font-rendered synthetic plates for rare districts (Section 2C) — medium novelty, easy.
  6. Reader TTA for night/rain vs the failed enhancement approach (Section 4A) — medium novelty, easy-medium.
  7. Plate-vehicle attribute consistency (Section 5A), federated simulation (Section 6A) — lower priority.
- Nepali embossed-plate work and Indian multi-line datasets are the closest transfer sources for Indic two-line layouts; Devanagari-to-Bangla transfer (shared Brahmic structure) is plausible but unverified for plates.

### Gaps
- No Bangladeshi plate *video / multi-frame* dataset was found (2024-2026).
- BornoDrishti and PaddleOCR-VL-1.5 Bengali accuracy numbers were not found.
- Download availability of the BRTA dataset (arXiv 2609.06507) and the 2603.10267 data was not confirmed.
- No 2024-2026 Pakistani (Urdu) plate recognition paper was found.
