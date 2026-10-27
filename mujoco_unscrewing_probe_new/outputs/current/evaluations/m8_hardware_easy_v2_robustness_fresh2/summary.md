# M8 hardware-easy-v2 robustness audit

Captured traces are counted as mechanics-routing outcomes, not as direction predictions.

| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|
| base_p1v3 / strict | 162 / 180 | 11.66 / 27.64 deg | 0.315 | 0.716 / 5.90 | 0.160 / 0.34615384615384615 | 0 |
| base_p1v3 / easy_v2 | 160 / 180 | 16.61 / 106.15 deg | 0.250 | 0.637 / 6.58 | 0.150 / 0.625 | 1 |
| adapted_easy_v2 / strict | 162 / 180 | 1.69 / 5.23 deg | 0.926 | 1.000 / 3.43 | 0.537 / 0.9885057471264368 | 0 |
| adapted_easy_v2 / easy_v2 | 160 / 180 | 2.28 / 5.19 deg | 0.950 | 0.988 / 2.73 | 0.544 / 0.9770114942528736 | 0 |

Promotion gate: **FAIL**.

Boundary: Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from direction accuracy. This is not real-hardware fine-tuning or full-loop validation.
