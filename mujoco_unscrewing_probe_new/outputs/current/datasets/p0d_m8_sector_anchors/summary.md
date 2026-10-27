# M8 P0d Paired Sector Anchor Audit

Updated: 2026-07-10T01:59:36.504589+00:00

- complete paired orbits: 54 / 54
- rotated comparisons: 270
- raw selected-channel mean / p95 NRMSE: 0.4116 / 0.5222
- normalized primary-shape mean / p05 correlation: 0.9299 / 0.8336

| state | comparisons | mean shape correlation | p05 shape correlation |
|---|---:|---:|---:|
| double_overhang | 90 | 0.9433 | 0.8934 |
| outer_partial | 90 | 0.9288 | 0.8311 |
| supported_search | 90 | 0.9178 | 0.7913 |

## Boundary

Paired bounded-nuisance MuJoCo sector residual audit. It tests whether C6 copies are a safe training shortcut; it is not a trained posterior, hardware symmetry proof, or control result.
