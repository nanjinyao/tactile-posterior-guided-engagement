# M8 hardware-easy-v2 robustness audit

Captured traces are counted as mechanics-routing outcomes, not as direction predictions.

| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|
| base_p1v3 / strict | 180 / 180 | 1.30 / 4.05 deg | 0.989 | 0.994 / 2.63 | 0.994 / 0.994413407821229 | 0 |
| base_p1v3 / easy_v2 | 152 / 180 | 14.68 / 54.60 deg | 0.224 | 0.559 / 5.86 | 0.099 / 0.13333333333333333 | 0 |
| adapted_easy_v2 / strict | 180 / 180 | 1.07 / 3.57 deg | 0.989 | 0.989 / 2.59 | 1.000 / 0.9888888888888889 | 0 |
| adapted_easy_v2 / easy_v2 | 152 / 180 | 1.93 / 5.07 deg | 0.980 | 1.000 / 2.65 | 0.974 / 0.9864864864864865 | 0 |

Promotion gate: **PASS**.

Boundary: Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from direction accuracy. This is not real-hardware fine-tuning or full-loop validation.
