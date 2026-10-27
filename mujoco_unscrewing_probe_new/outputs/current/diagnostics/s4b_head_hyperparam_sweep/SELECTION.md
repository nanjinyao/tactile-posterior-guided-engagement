# S4b validation-only head hyperparameter selection

This sweep was completed after the S4b train/validation/calibration task
contract was frozen and sampled, and before any S4b full-loop rollout was
inspected.  Only the frozen `validation` C6 orbits were used to rank
candidates.  The `calibration` role remained restricted to temperature/APS
fitting; neither the old S3/S4 development trajectories nor a new full-loop
task was an input to model selection.

The predeclared lexicographic objective is minimized across both sizes:

1. sum of validation circular-control errors above 90 degrees;
2. maximum validation worst-state circular-control p90;
3. one minus the minimum validation state accuracy;
4. mean validation circular-control p90.

All candidates updated only `state_head` and `direction_head`; the encoder and
shared trunk remained frozen.  The complete candidate reports are retained in
the sibling directories.  The decisive candidates were:

| candidate | M10 >90 / worst-state p90 / state acc. | M12 >90 / worst-state p90 / state acc. | selected |
|---|---:|---:|---:|---:|
| sigma 7.0 deg, L2-SP 0.010 | 1 / 16.31 deg / 1.000 | 0 / 35.71 deg / 0.833 | no |
| sigma 10.0 deg, L2-SP 0.010 | 1 / 17.58 deg / 1.000 | 0 / 25.41 deg / 0.833 | no |
| sigma 12.5 deg, L2-SP 0.010 | 1 / 18.11 deg / 1.000 | 0 / 23.99 deg / 0.861 | no |
| **sigma 15.0 deg, L2-SP 0.010** | **1 / 19.63 deg / 1.000** | **0 / 23.72 deg / 0.861** | **yes** |
| sigma 20.0 deg, L2-SP 0.010 | 1 / 17.84 deg / 1.000 | 0 / 24.04 deg / 0.861 | no |

The selected run uses 160 epochs, learning rate 0.001, state-loss weight 1.0,
and circular soft-label sigma 15 degrees.  One M10 validation outlier remains
above 90 degrees; this is retained as a limitation rather than removed or
relabelled.  These are held-out head diagnostics, not full-loop or real-robot
evidence.
