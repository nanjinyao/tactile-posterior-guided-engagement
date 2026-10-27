# M8 paired recoverable physical-funnel ablation

All three arms use the same 12 task realizations, ordinary outward search, sensor-only capture handoff, nuisance parameters, 60 s active-time budget, CAD-conditioned insertion, and terminal verifier. Only the post-capture funnel/retry contract changes.

| arm | bottom | hard stop | safe abort | recoveries | terminal attempts | median success time [s] |
|---|---:|---:|---:|---:|---:|---:|
| no_funnel | 2/12 | 2 | 0 | 0 | 10 | 5.457 |
| rcc_shield_r0 | 0/12 | 2 | 10 | 0 | 0 | - |
| rcc_shield_r3 | 9/12 | 2 | 1 | 11 | 9 | 5.275 |

On this fixed 12-case set, R3 verified 9/12 bottoms versus 2/12 without the funnel and 0/12 with R0. Against no funnel, R3 uniquely recovered 7 paired cases and regressed 0 (exact two-sided paired sign p=0.015625). It still produced 2 hard stops and 1 safe abort, so the closed loop is improved but not guaranteed.

## Mechanism and boundary

R0 keeps tip-frame RCC and the safety observer but permits no recovery. R3 permits at most three unload/retract/upright/retouch cycles. Every recovery rebases XY from the measured tip pose; the controller is never given the true socket center. Simulator resets are forbidden and counted explicitly.

This is a paired MuJoCo post-capture stress ablation, not a proof that a funnel must succeed. A bottom is counted only after the same CAD-depth/force/low-wrench dwell verifier. R3 improvement, if present, is causal only for this fixed moderate-stress set and current controller; it is not hardware evidence, a passivity proof, or a guarantee outside the tested envelope.
