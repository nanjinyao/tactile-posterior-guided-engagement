# M8 hardware-easy-v2 robustness audit

Captured traces are counted as mechanics-routing outcomes, not as direction predictions.

| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|
| base_p1v3 / strict | 165 / 180 | 13.03 / 56.83 deg | 0.230 | 0.539 / 6.45 | 0.236 / 0.28205128205128205 | 2 |
| base_p1v3 / easy_v2 | 169 / 180 | 15.08 / 33.63 deg | 0.183 | 0.716 / 5.59 | 0.112 / 0.21052631578947367 | 0 |
| adapted_easy_v2 / strict | 165 / 180 | 2.56 / 6.51 deg | 0.921 | 0.988 / 3.38 | 0.782 / 0.9922480620155039 | 0 |
| adapted_easy_v2 / easy_v2 | 169 / 180 | 2.82 / 7.13 deg | 0.911 | 0.988 / 2.78 | 0.893 / 0.9139072847682119 | 0 |

Promotion gate: **FAIL**.

Boundary: Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from direction accuracy. This is not real-hardware fine-tuning or full-loop validation.
