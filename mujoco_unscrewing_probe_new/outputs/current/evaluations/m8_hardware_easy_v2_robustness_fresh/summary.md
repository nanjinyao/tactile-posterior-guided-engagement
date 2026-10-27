# M8 hardware-easy-v2 robustness audit

Captured traces are counted as mechanics-routing outcomes, not as direction predictions.

| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|
| base_p1v3 / strict | 160 / 180 | 9.91 / 38.28 deg | 0.312 | 0.756 / 5.24 | 0.250 / 0.325 | 0 |
| base_p1v3 / easy_v2 | 158 / 180 | 13.39 / 39.39 deg | 0.222 | 0.557 / 5.70 | 0.177 / 0.07142857142857142 | 1 |
| adapted_easy_v2 / strict | 160 / 180 | 3.93 / 18.83 deg | 0.775 | 0.963 / 4.36 | 0.575 / 0.9239130434782609 | 0 |
| adapted_easy_v2 / easy_v2 | 158 / 180 | 1.80 / 7.52 deg | 0.899 | 0.962 / 3.08 | 0.854 / 0.9629629629629629 | 0 |

Promotion gate: **FAIL**.

Boundary: Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from direction accuracy. This is not real-hardware fine-tuning or full-loop validation.
