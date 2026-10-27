# M8 hardware-easy-v2 robustness audit

Captured traces are counted as mechanics-routing outcomes, not as direction predictions.

| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|
| base_p1v3 / strict | 167 / 180 | 12.15 / 53.87 deg | 0.311 | 0.850 / 6.33 | 0.204 / 0.3235294117647059 | 0 |
| base_p1v3 / easy_v2 | 170 / 180 | 14.39 / 55.53 deg | 0.206 | 0.735 / 5.54 | 0.176 / 0.3 | 0 |
| adapted_easy_v2 / strict | 167 / 180 | 1.85 / 4.79 deg | 0.958 | 0.982 / 2.44 | 0.904 / 1.0 | 0 |
| adapted_easy_v2 / easy_v2 | 170 / 180 | 1.78 / 3.45 deg | 1.000 | 1.000 / 2.19 | 0.982 / 1.0 | 0 |

Promotion gate: **PASS**.

Boundary: Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from direction accuracy. This is not real-hardware fine-tuning or full-loop validation.
