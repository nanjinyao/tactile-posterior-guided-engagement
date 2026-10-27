# M8 P0a C6 Symmetry Audit

Updated: 2026-07-10T01:27:14.886189+00:00

## Result

- complete base groups: 12 / 12
- rotated comparisons: 60
- raw selected-channel mean NRMSE: 0.3822
- raw selected-channel p95 NRMSE: 0.4708
- primary normalized-shape mean correlation: 0.9446
- primary normalized-shape p05 correlation: 0.8829

Primary shape channels are `Fz`, `Tx`, `Ty`, and `tip_z`, circularly smoothed and standardized per trial.

## State Breakdown

| state | comparisons | mean shape correlation | p05 shape correlation |
|---|---:|---:|---:|
| double_overhang | 20 | 0.9564 | 0.9456 |
| outer_partial | 20 | 0.9414 | 0.8828 |
| supported_search | 20 | 0.9361 | 0.8802 |

## Interpretation Boundary

The nominal MuJoCo data supports provisional C6 augmentation for the normalized direction/shape branch. Raw lateral wrench amplitudes are not sufficiently invariant to use rotated copies as independent evidence for radial distance or hardware transfer.

P0a has no nuisance repeats. Compare these residuals with the P0b nuisance baseline before enabling C6 augmentation in the final training pipeline.
