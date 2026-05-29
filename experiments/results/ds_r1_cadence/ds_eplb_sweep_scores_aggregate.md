# DS-R1 Sweep Expected Composite Scores

Scores use `DS-EPLB@1024:layers=all` as the per-case baseline and the current formula: `100 * par_ratio + 25 * (1 - transmit_ratio)`.

| variant | expected composite score | min case score | max case score |
|---|---:|---:|---:|
| Default | 86.931010 | 68.344463 | 102.933030 |
| DS-EPLB@256:layers=all | 55.324881 | 36.604371 | 87.241727 |
| DS-EPLB@512:layers=all | 82.959039 | 75.937626 | 99.596796 |
| DS-EPLB@1024:layers=all | 100.000000 | 100.000000 | 100.000000 |
| DS-EPLB@2048:layers=all | 101.918806 | 82.591121 | 111.245670 |
| DS-EPLB@4096:layers=all | 90.876154 | 73.681987 | 105.977773 |
