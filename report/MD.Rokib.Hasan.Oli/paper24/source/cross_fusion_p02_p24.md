Test plates: 651, val plates: 766. "vs chars_v8n": the val-best single model; fixed / broken = test plates the row gets right / wrong where chars_v8n does not, p = exact McNemar. ms: summed per plate; fallback = average (helpers only run on some plates).

| models | rule | val exact % | test exact % | test CER % | fixed / broken vs chars_v8n (p) | ms/plate |
|---|---|---|---|---|---|---|
| lpnr_v12n | single | 80.55 | 87.71 | 2.15 |  | 23.55 |
| chars_v8n | single | 82.38 | 86.79 | 2.51 |  | 9.36 |
| crnn | single | 62.27 | 73.43 | 5.53 |  | 6.47 |
| lpnr_v12n + chars_v8n | fallback | 86.55 | 88.79 | 2.3 | 13 / 0 (p 0.0002) | 11.02 (helpers on 7.07 %) |
| lpnr_v12n + chars_v8n | MV-BM+format | 86.55 | 88.79 | 2.3 | 13 / 0 (p 0.0002) | 32.91 |
| lpnr_v12n + chars_v8n + crnn | MV-BM+format | 86.03 | 89.25 | 2.54 | 16 / 0 (p 0.0) | 39.38 |
| lpnr_v12n + chars_v8n + crnn | fallback | 85.77 | 88.79 | 2.57 | 13 / 0 (p 0.0002) | 11.48 (helpers on 7.07 %) |
| lpnr_v12n + chars_v8n | HC | 85.38 | 89.71 | 1.92 | 22 / 3 (p 0.0002) | 32.91 |
| lpnr_v12n + chars_v8n | MV-HC | 85.38 | 89.71 | 1.92 | 22 / 3 (p 0.0002) | 32.91 |
| lpnr_v12n + chars_v8n | MVCP-HC | 85.38 | 89.71 | 1.92 | 22 / 3 (p 0.0002) | 32.91 |
| chars_v8n + crnn | fallback | 84.99 | 88.63 | 2.69 | 12 / 0 (p 0.0005) | 9.82 (helpers on 7.07 %) |
| chars_v8n + crnn | MV-BM+format | 84.99 | 88.63 | 2.69 | 12 / 0 (p 0.0005) | 15.83 |
| lpnr_v12n + crnn | fallback | 84.33 | 89.55 | 2.68 | 30 / 12 (p 0.0079) | 24.07 (helpers on 7.99 %) |
| lpnr_v12n + crnn | MV-BM+format | 84.33 | 89.55 | 2.68 | 30 / 12 (p 0.0079) | 30.02 |
| lpnr_v12n + chars_v8n + crnn | MV-BM | 84.2 | 88.17 | 2.43 | 9 / 0 (p 0.0039) | 39.38 |
| lpnr_v12n + chars_v8n + crnn | MVCP-HC | 84.2 | 88.17 | 2.68 | 15 / 6 (p 0.0784) | 39.38 |
| lpnr_v12n + chars_v8n + crnn | MV-HC | 83.55 | 88.33 | 2.99 | 18 / 8 (p 0.0755) | 39.38 |
| chars_v8n + crnn | MV-BM | 82.38 | 86.79 | 2.51 | 0 / 0 (p 1.0) | 15.83 |

- Best single on val: chars_v8n, test 86.79 %.
- Chosen on val (all rules): lpnr_v12n + chars_v8n, fallback: val 86.55 %, test 88.79 % (+2.00 pp vs chars_v8n; fixed 13, broken 0, exact McNemar p = 0.0002), 11.02 ms per plate (helpers run on 7.07 % of the test plates).
- Best of the paper's rules: lpnr_v12n + chars_v8n, HC: val 85.38 %, test 89.71 % (+2.92 pp vs chars_v8n; fixed 22, broken 3, exact McNemar p = 0.0002), 32.91 ms per plate.
- Best fallback: lpnr_v12n + chars_v8n, fallback: val 86.55 %, test 88.79 % (+2.00 pp vs chars_v8n; fixed 13, broken 0, exact McNemar p = 0.0002), 11.02 ms per plate (helpers run on 7.07 % of the test plates).
- Oracle (any model right): test 91.71 %.
