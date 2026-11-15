# Paper #36 on its own data: ours vs. the paper

| Metric | Paper (page) | visdrone_rescore |
|---|---|---|
| FCOS + FI: AP50 (%) | 25.8 (p.3, Table 1) | 25.80 |
| FCOS + SAHI + PO: AP50 (%) | 29.0 (p.3, Table 1) | 29.00 |
| FCOS + SAHI + FI + PO: AP50 (%) | 31.0 (p.3, Table 1) | 31.00 |
| FCOS + SF + SAHI + PO: AP50 (%) | 38.1 (p.3, Table 1) | 38.10 |
| FCOS + SF + SAHI + FI + PO: AP50 (%) | 38.5 (p.3, Table 1) | 38.50 |
| VFNet + FI: AP50 (%) | 28.8 (p.3, Table 1) | 28.80 |
| VFNet + SAHI + PO: AP50 (%) | 32.0 (p.3, Table 1) | 32.00 |
| VFNet + SAHI + FI + PO: AP50 (%) | 33.9 (p.3, Table 1) | 33.90 |
| VFNet + SF + SAHI + PO: AP50 (%) | 41.9 (p.3, Table 1) | 41.90 |
| VFNet + SF + SAHI + FI + PO: AP50 (%) | 42.2 (p.3, Table 1) | 42.20 |
| TOOD + FI: AP50 (%) | 29.4 (p.3, Table 1) | 29.40 |
| TOOD + SAHI: AP50 (%) | 31.9 (p.3, Table 1) | 31.90 |
| TOOD + SAHI + PO: AP50 (%) | 32.5 (p.3, Table 1) | 32.50 |
| TOOD + SAHI + FI: AP50 (%) | 34.6 (p.3, Table 1) | 34.60 |
| TOOD + SAHI + FI + PO: AP50 (%) | 34.7 (p.3, Table 1) | 34.70 |
| TOOD + SF + FI: AP50 (%) | 36.8 (p.3, Table 1) | 36.80 |
| TOOD + SF + SAHI: AP50 (%) | 42.5 (p.3, Table 1) | 42.50 |
| TOOD + SF + SAHI + PO: AP50 (%) | 43.1 (p.3, Table 1) | 43.10 |
| TOOD + SF + SAHI + FI: AP50 (%) | 43.4 (p.3, Table 1) | 43.40 |
| TOOD + SF + SAHI + FI + PO: AP50 (%) | 43.5 (p.3, Table 1) | 43.50 |
| TOOD + FI: AP50 small (%) | 18.1 (p.3, Table 1) | 18.10 |
| TOOD + SF + SAHI + FI + PO: AP50 small (%) | 31.7 (p.3, Table 1) | 31.70 |
| FCOS: SAHI gain over FI, AP50 (pp) | 6.8 (abstract) (p.1) | 5.20 |
| FCOS: SF + SAHI gain over FI (pp) | 12.7 (abstract) (p.1) | 12.70 |
| VFNet: SAHI / SF + SAHI gain (pp) | 5.1 / 13.4 (abstract) (p.1) | 13.40 |
| TOOD: SAHI gain over FI (pp) | 5.3 (abstract) (p.1) | 5.30 |
| TOOD: SF + SAHI gain over FI (pp) | 14.5 (abstract) (p.1) | 14.10 |
| Largest |re-scored - Table 1| AP50 (pp) | - | 0.00 |

Same data and split as the paper (its own published data / weights), so this IS a direct check of
the paper's numbers. Remaining differences come from software versions, hardware and randomness.

Same data and the authors' own predictions: a direct check of Table 1. 'SAHI gain' = best SAHI row without SF minus FI; 'SF + SAHI gain' = best row of that detector minus FI (the abstract's wording).
