Test plates: 651, val plates: 766. "vs chars_v8n": the val-best single model; fixed / broken = test plates the row gets right / wrong where chars_v8n does not, p = exact McNemar. ms: summed per plate; fallback = average (helpers only run on some plates).

| models | rule | val exact % | test exact % | test CER % | fixed / broken vs chars_v8n (p) | ms/plate |
|---|---|---|---|---|---|---|
| chars_v8n | single | 82.38 | 86.79 | 2.51 |  | 9.36 |
| lpnr_v12n | single | 80.55 | 87.71 | 2.15 |  | 23.55 |
| crnn | single | 62.27 | 73.43 | 5.53 |  | 6.47 |
| svtrv2_t_msr | single | 73.76 | 81.26 | 2.49 |  | 7.62 |
| svtrv2_s_msr | single | 75.33 | 81.87 | 3.07 |  | 15.34 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr | MV-BM+format | 89.3 | 90.94 | 1.55 | 28 / 1 (p 0.0) | 55.87 |
| chars_v8n + lpnr_v12n + crnn + svtrv2_t_msr + svtrv2_s_msr | MV-BM+format | 89.3 | 90.78 | 1.98 | 28 / 2 (p 0.0) | 62.34 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr | MV-BM+format | 88.64 | 90.78 | 1.57 | 27 / 1 (p 0.0) | 40.53 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr | MV-HC | 88.64 | 90.94 | 1.49 | 29 / 2 (p 0.0) | 55.87 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr | fallback | 88.38 | 90.32 | 1.71 | 23 / 0 (p 0.0) | 12.65 (helpers on 7.07 %) |
| chars_v8n + lpnr_v12n + crnn + svtrv2_t_msr + svtrv2_s_msr | fallback | 88.38 | 90.02 | 2.16 | 21 / 0 (p 0.0) | 13.11 (helpers on 7.07 %) |
| chars_v8n + lpnr_v12n + crnn + svtrv2_t_msr | MV-BM+format | 88.38 | 90.32 | 2.19 | 25 / 2 (p 0.0) | 47.0 |
| chars_v8n + lpnr_v12n + svtrv2_s_msr | MVCP-HC | 88.38 | 91.24 | 1.45 | 30 / 1 (p 0.0) | 48.25 |
| chars_v8n + lpnr_v12n + svtrv2_s_msr | MV-BM+format | 88.38 | 90.94 | 1.85 | 27 / 0 (p 0.0) | 48.25 |
| chars_v8n + lpnr_v12n + crnn + svtrv2_t_msr + svtrv2_s_msr | MV-HC | 88.38 | 90.48 | 2.14 | 29 / 5 (p 0.0) | 62.34 |
| chars_v8n + lpnr_v12n + crnn + svtrv2_t_msr + svtrv2_s_msr | MV-BM | 88.38 | 90.17 | 1.93 | 24 / 2 (p 0.0) | 62.34 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr | fallback | 88.25 | 90.17 | 1.61 | 22 / 0 (p 0.0) | 11.56 (helpers on 7.07 %) |
| chars_v8n + lpnr_v12n + svtrv2_s_msr | fallback | 88.12 | 90.17 | 1.89 | 22 / 0 (p 0.0) | 12.11 (helpers on 7.07 %) |
| chars_v8n + lpnr_v12n + crnn + svtrv2_s_msr | MV-BM+format | 88.12 | 90.32 | 2.19 | 23 / 0 (p 0.0) | 54.72 |
| chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr | MVCP-HC | 87.99 | 90.48 | 1.3 | 29 / 5 (p 0.0) | 55.87 |

- Best single on val: chars_v8n, test 86.79 %.
- Chosen on val (all rules): chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr, MV-BM+format: val 89.3 %, test 90.94 % (+4.15 pp vs chars_v8n; fixed 28, broken 1, exact McNemar p = 0.0), 55.87 ms per plate.
- Best of the paper's rules: chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr, MV-HC: val 88.64 %, test 90.94 % (+4.15 pp vs chars_v8n; fixed 29, broken 2, exact McNemar p = 0.0), 55.87 ms per plate.
- Best fallback: chars_v8n + lpnr_v12n + svtrv2_t_msr + svtrv2_s_msr, fallback: val 88.38 %, test 90.32 % (+3.53 pp vs chars_v8n; fixed 23, broken 0, exact McNemar p = 0.0), 12.65 ms per plate (helpers run on 7.07 % of the test plates).
- Oracle (any model right): test 93.7 %.
