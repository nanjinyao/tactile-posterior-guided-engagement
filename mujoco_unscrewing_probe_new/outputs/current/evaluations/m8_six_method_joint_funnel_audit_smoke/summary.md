# M8 Six-Method Joint-Funnel Paired Audit

All six methods use the same 36 unseen Sobol poses and pre-hardware wrench prior. Efficiency medians are conditioned on offline-validated capture.

| method | success | first attempt | capture time | actual/contact path | wear | max F p90 | max tau p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 1/1 | 1/1 | 10.23 s | 6.33/4.09 mm | 7.96 mJ | 4.66 N | 13.61 mNm |
| proposed_posterior | 1/1 | 1/1 | 12.43 s | 15.08/14.95 mm | 24.35 mJ | 5.83 N | 15.92 mNm |
| baseline_a | 1/1 | 1/1 | 46.74 s | 102.20/102.19 mm | 134.94 mJ | 8.59 N | 18.56 mNm |
| baseline_b | 1/1 | 1/1 | 7.85 s | 6.25/6.25 mm | 12.54 mJ | 5.74 N | 17.61 mNm |
| baseline_c_entropy | 0/1 | 0/1 | n/a s | n/a/n/a mm | n/a mJ | 3.10 N | 6.44 mNm |
| baseline_d_random | 0/1 | 0/1 | n/a s | n/a/n/a mm | n/a mJ | 5.92 N | 15.23 mNm |

## Proposed+ mechanism diagnostics

- global_fallback_count: 0
- entropy_probe_count_median: 2.0
- entropy_probe_count_p90: 2.0
- positive_value_probe_total: 2
- local_spiral_count: 0
- posterior_credible_radius_median_mm: 1.333624349329103

## Boundary

Frozen M8 MuJoCo audit inside the declared visual prior. The contact nuisance and F/T bias/noise/coupling/delay distributions are pre-hardware engineering priors, not measured iiwa statistics. Proposed+ and Proposed share a byte-identical conical fingerprint and frozen P1v3 output for each pose. No method receives the true center online; MuJoCo geometry is used only for offline success scoring. Baseline-B reset travel remains excluded by the frozen source audit. This can support a simulation mechanism claim, not a hardware robustness or cross-size claim.
