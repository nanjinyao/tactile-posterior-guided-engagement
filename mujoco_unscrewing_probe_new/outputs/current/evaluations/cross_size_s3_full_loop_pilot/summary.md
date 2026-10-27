# Fresh cross-size S3 untouched full-loop pilot

All five methods share the same paired poses, 130 s zero-extension budget,
sensor-only capture, tip-RCC/R3 post-capture funnel, and strict CAD-bottom
terminal verifier. The evaluation set was generated only after model and gate freeze.
Pre-registered gate: oracle >= 0.90; fine-tuned >= 0.80;
fine-tuned hard stops <= 0; >90 deg control errors <= 0.

| size | method | strict CAD-bottom | capture | hard stop | bottom time | path | control error med/p90 | R3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M10 | M8 zero-shot | 10/12 | 11/12 | 1 | 22.81 s | 33.94 mm | 4.89/47.30 deg | 2 |
| M10 | size calibration only | 10/12 | 11/12 | 1 | 39.56 s | 72.04 mm | 5.73/31.34 deg | 2 |
| M10 | direction-head fine-tuned | 10/12 | 11/12 | 1 | 30.42 s | 48.44 mm | 8.37/83.75 deg | 2 |
| M10 | pure outward | 9/12 | 10/12 | 1 | 34.20 s | 69.60 mm | n/a/n/a deg | 6 |
| M10 | privileged straight-to-center | 9/12 | 11/12 | 3 | 4.06 s | 4.74 mm | n/a/n/a deg | 2 |

- M10 adapted gate: **FAIL**
- Evidence interpretation: `oracle_execution_feasibility_gate_failed`

| M12 | M8 zero-shot | 9/12 | 10/12 | 1 | 30.49 s | 48.68 mm | 2.69/32.83 deg | 3 |
| M12 | size calibration only | 8/12 | 10/12 | 3 | 34.87 s | 58.03 mm | 5.06/18.98 deg | 3 |
| M12 | direction-head fine-tuned | 9/12 | 10/12 | 3 | 43.72 s | 69.84 mm | 4.92/36.83 deg | 5 |
| M12 | pure outward | 7/12 | 8/12 | 4 | 47.19 s | 95.89 mm | n/a/n/a deg | 3 |
| M12 | privileged straight-to-center | 12/12 | 12/12 | 0 | 5.28 s | 5.91 mm | n/a/n/a deg | 1 |

- M12 adapted gate: **FAIL**
- Evidence interpretation: `direction_head_finetuning_not_sufficient_to_pass_this_pilot_gate`

## Claim boundary

Fresh paired MuJoCo S3 pilot only. A fine-tuned pass shows that a small size-specific direction-head update can recover this simulated full-loop gate under the stated contract. Fine-tuning is called necessary only when both zero-shot and calibration-only fail the same frozen gate. This cannot establish that real-robot data fine-tuning is necessary: no real F/T sample, manufacturing tolerance, passivity, motor coupling, threading, or extraction is represented. The privileged straight-to-center arm is an execution gate, not a guaranteed performance upper bound: its direct path can itself hard-stop.
