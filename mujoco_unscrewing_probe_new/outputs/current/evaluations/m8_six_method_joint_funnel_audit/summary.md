# M8 Six-Method Joint-Funnel Paired Audit

All six methods use the same 36 unseen Sobol poses and pre-hardware wrench prior. Efficiency medians are conditioned on offline-validated capture.

| method | success | first attempt | capture time | actual/contact path | wear | max F p90 | max tau p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 26/36 | 26/36 | 17.51 s | 7.08/6.87 mm | 11.78 mJ | 10.54 N | 36.84 mNm |
| proposed_posterior | 34/36 | 34/36 | 12.47 s | 14.71/14.61 mm | 29.87 mJ | 9.48 N | 22.33 mNm |
| baseline_a | 30/36 | 30/36 | 36.06 s | 77.11/77.07 mm | 135.71 mJ | 9.44 N | 23.03 mNm |
| baseline_b | 28/36 | 25/36 | 7.49 s | 5.36/5.36 mm | 13.27 mJ | 7.54 N | 21.34 mNm |
| baseline_c_entropy | 34/36 | 34/36 | 18.68 s | 12.44/2.76 mm | 1.02 mJ | 8.70 N | 20.55 mNm |
| baseline_d_random | 19/36 | 19/36 | 10.19 s | 20.91/20.82 mm | 32.90 mJ | 8.97 N | 21.86 mNm |

## Proposed+ mechanism diagnostics

- global_fallback_count: 18
- entropy_probe_count_median: 2.0
- entropy_probe_count_p90: 4.0
- positive_value_probe_total: 96
- local_spiral_count: 23
- posterior_credible_radius_median_mm: 1.178955551636233

## Boundary

Frozen M8 MuJoCo audit inside the declared visual prior. The contact nuisance and F/T bias/noise/coupling/delay distributions are pre-hardware engineering priors, not measured iiwa statistics. Proposed+ and Proposed share a byte-identical conical fingerprint and frozen P1v3 output for each pose. No method receives the true center online; MuJoCo geometry is used only for offline success scoring. Baseline-B reset travel remains excluded by the frozen source audit. This can support a simulation mechanism claim, not a hardware robustness or cross-size claim.
