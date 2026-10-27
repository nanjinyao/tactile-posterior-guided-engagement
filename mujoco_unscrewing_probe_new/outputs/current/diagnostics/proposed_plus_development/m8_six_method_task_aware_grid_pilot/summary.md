# M8 Six-Method Joint-Funnel Paired Audit

All six methods use the same 36 unseen Sobol poses and pre-hardware wrench prior. Efficiency medians are conditioned on offline-validated capture.

| method | success | first attempt | capture time | actual/contact path | wear | max F p90 | max tau p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 10/12 | 10/12 | 9.90 s | 6.24/4.78 mm | 12.45 mJ | 10.15 N | 42.87 mNm |
| proposed_posterior | 11/12 | 11/12 | 12.36 s | 14.34/14.28 mm | 28.32 mJ | 9.16 N | 22.13 mNm |
| baseline_a | 9/12 | 9/12 | 36.47 s | 77.93/77.93 mm | 134.94 mJ | 10.08 N | 25.05 mNm |
| baseline_b | 9/12 | 7/12 | 7.79 s | 6.05/6.05 mm | 13.06 mJ | 7.96 N | 20.45 mNm |
| baseline_c_entropy | 11/12 | 11/12 | 17.43 s | 7.08/2.57 mm | 0.94 mJ | 8.73 N | 18.84 mNm |
| baseline_d_random | 6/12 | 6/12 | 5.53 s | 11.40/11.33 mm | 21.12 mJ | 8.07 N | 23.03 mNm |

## Proposed+ mechanism diagnostics

- global_fallback_count: 4
- entropy_probe_count_median: 3.0
- entropy_probe_count_p90: 6.0
- positive_value_probe_total: 49
- local_spiral_count: 0
- posterior_credible_radius_median_mm: 6.184166494486953

## Boundary

Frozen M8 MuJoCo audit inside the declared visual prior. The contact nuisance and F/T bias/noise/coupling/delay distributions are pre-hardware engineering priors, not measured iiwa statistics. Proposed+ and Proposed share a byte-identical conical fingerprint and frozen P1v3 output for each pose. No method receives the true center online; MuJoCo geometry is used only for offline success scoring. Baseline-B reset travel remains excluded by the frozen source audit. This can support a simulation mechanism claim, not a hardware robustness or cross-size claim.
