# S4 validation-only head hyperparameter selection

This sweep was completed before any S4 development or untouched full-loop
rollout.  It used the pre-frozen `validation` C6 orbits only for selection;
`calibration` remained restricted to temperature/APS fitting and no full-loop
task was inspected.

Selection is lexicographic over both sizes:

1. sum of validation circular-control errors above 90 degrees;
2. maximum validation worst-state circular-control p90;
3. one minus the minimum validation state accuracy;
4. mean validation circular-control p90.

| candidate | M10 >90 / worst-state p90 / state acc. | M12 >90 / worst-state p90 / state acc. | selected |
|---|---:|---:|---:|
| initial: 80 epochs, L2-SP 0.002, sigma 7.5 deg | 0 / 12.50 deg / 0.900 | 1 / 30.42 deg / 0.900 | no |
| 160 epochs, L2-SP 0.015, sigma 7.5 deg | 0 / 35.60 deg / 0.900 | 0 / 28.13 deg / 0.900 | no |
| 160 epochs, L2-SP 0.010, sigma 5.0 deg | 0 / 27.84 deg / 0.867 | 0 / 35.02 deg / 0.900 | no |
| **160 epochs, L2-SP 0.010, sigma 6.0 deg** | **0 / 34.18 deg / 0.867** | **0 / 31.92 deg / 0.900** | **yes** |
| 160 epochs, L2-SP 0.010, sigma 7.0 deg | 0 / 36.18 deg / 0.867 | 0 / 28.92 deg / 0.900 | no |
| 160 epochs, L2-SP 0.010, sigma 7.5 deg | 0 / 36.32 deg / 0.900 | 0 / 27.41 deg / 0.900 | no |

All other candidate directories remain under this folder.  The selected model
uses learning rate 0.001 and state-loss weight 1.0.  These are head diagnostics,
not full-loop evidence, and the selection does not support a real-robot claim.
