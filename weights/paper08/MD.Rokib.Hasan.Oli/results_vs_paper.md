# Paper #8: ours vs. the paper

| Metric | Paper (page) | cnn28 | resnet50 | plates_cnn28 |
|---|---|---|---|---|
| Classes | 17 (10 digits, 4 letters, 3 region words) (p.3, Table I) | 101 | 101 | 101 |
| Character test accuracy, last epoch (%) | 92.38 (CNN), 87.50 (ResNet-50) (p.4, Table II) | 36.67 | 92.06 | – |
| Character macro precision (%) | 96.67 (CNN), 93.43 (ResNet-50) (p.4, Table II) | 13.69 | 49.03 | – |
| Character macro recall (%) | 91.62 (CNN), 88.27 (ResNet-50) (p.4, Table II) | 7.15 | 49.02 | – |
| Character macro F1 (%) | 94.76 (CNN), 89.84 (ResNet-50) (p.4, Table II) | 8.16 | 46.41 | – |
| Character test accuracy, best-val checkpoint (%) | - | 93.08 | 92.89 | – |
| Best validation accuracy (%) | 98.80 (CNN) (p.4) | 90.20 | 91.71 | – |
| Epochs run | 40 (CNN), 30 (ResNet-50) (p.4-5) | 40 | 25 | – |
| Classifier FPS, batch 1, T4 (model only) | not reported | 760.90 | 171.50 | – |
| Training time (min) | not reported | 151.80 | 198.20 | – |
| Plates: mean similarity, GFP-GAN + x order (%) | 88 (30 plates) (p.5) | – | – | 44.95 |
| Plates with similarity >= 0.95 (%) | 57 (17 of 30) (p.5) | – | – | 0.00 |
| Plates with similarity >= 0.90 (%) | 80 (24 of 30) (p.5) | – | – | 0.00 |
| Plates: mean similarity WITHOUT GFP-GAN (%) | never tested | – | – | 41.86 |
| Plates exact, GFP-GAN + two-line order (%) | not reported | – | – | 0.15 |
| Plates exact, no GFP-GAN + two-line order (%) | - | – | – | 0.00 |
| Plates exact, perfect segmentation (human token boxes) (%) | - | – | – | 59.53 |
| Char accuracy, perfect segmentation (%) | - | – | – | 86.77 |
| GFP-GAN (Real-ESRGAN x2) ms per plate | not reported | – | – | 205.60 |
| Segmentation + CNN ms per plate | not reported | – | – | 6.10 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's 17 classes and 303 test crops (48 are copies of training crops, see the original check) vs our 101 tokens on the grouped split: a harder task, so a lower symbol accuracy is expected. The paper's plate score is the mean difflib SequenceMatcher ratio, which gives partial credit; plate exact is the strict number.

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| plates_cnn28 | 0.15 | 0.21 | 0.00 | 0.00 | – | – |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
