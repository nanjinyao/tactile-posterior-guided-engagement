# Real multi-size conical-direction runtime

`infer_conical_direction.py` is a one-shot, phase-boundary CPU inference
process. It never connects to ROS, Bota, KST, or the robot. MATLAB records one
completed cone, invokes this process once after DirectServo has stopped, reads
the posterior, and remains the sole online command owner.

## Active bundle and authority

The active bundle is
`bundles/real_multisize_direction_v1`. It was selected from exactly three
clean-only leave-one-size-out candidates:

1. real-only Fourier/circular;
2. frozen simulation encoder plus real input/output adapters;
3. the same adapter with only `shared.0` unfrozen and anchored by L2SP.

The real-only `REAL_FOURIER_LINEAR_V1` candidate won the fixed selection rule.
The bundle contains three final seeds trained on all 48 clean real samples.
The 12 consumed holdout samples are report-only; quarantine and sentinel data
do not fit weights or normalization.

Its explicit boundary is:

- qualification: `DEVELOPMENT_CV_CONTROL`;
- deployment authority: `CAPPED_SHORTCUT`;
- hardware-control size: M12 only;
- M10/M8 under the base promotion contract: inference and training domains
  only;
- model authority: direction/state posterior plus CAD-bounded shortcut,
  corridor and residual-target proposals;
- no model authority over physical capture, insertion completion, terminal
  success, hard protection, or online weight updates.

Low confidence is not a validity veto. A complete, fresh, finite cone still
produces a bounded M12 action. Incomplete, stale, nonfinite or contract-mismatched
input fails explicitly; the controller does not silently fall back to a
classical direction heuristic.

One separate operator-requested exception is deliberately narrower than a
bundle promotion:
`configs/m10_fixed_demo_pilot_authorization_v1.json` hash-binds the exact
`run_m10_proposed_plus_entropy_trial` entrypoint, one M10 offset near the
twice-completed A003p75 real sample, and at most three separately invoked
attempts.  Only that invocation may add M10 to the effective hardware-routing
sizes.  MATLAB independently checks the same record, keeps capture/insertion/
terminal authority outside the model, and enforces a fresh-centre
socket-head-support fail-stop (physical radius 5.792458 mm; commanded radius
5.192458 mm after a 0.600 mm coupled-motion allowance).  The fresh centre is
never a posterior input, target-ranking input, or steering input.  This
overlay is fixed-demo authority only; it does not modify or widen the M12-only
base promotion contract.

## Input and output contract

The model input is `7 x 360`:

- six WORLD-aligned relative wrench channels;
- commanded tip-Z relative to the cone pivot;
- a five-value context containing size one-hot, force-target ratio and
  normalized bit radius.

The output contains a 48-bin direction posterior, the three contact-state
probabilities (`supported_search`, `double_overhang`,
`outer_partial_support`), TOP1 and circular-mean directions, entropy,
concentration, APS and continuous sample quality. The actual command direction
is TOP1; circular mean is diagnostic because a multimodal posterior can make
the mean point between physical modes.

The printed-CAD `-15 deg` peg-shape rotation is already baked into geometry.
The installation maps the selected B/C normal to `+X_F` once, giving peg/socket
relative yaw `0 deg mod 60 deg`. Runtime therefore applies no additional
`-15/+15/+45 deg` compensation. The current fixture yaw is an operator-taught
commissioning assumption, not independent as-built yaw metrology.

## Offline replay

The checked-in runtime environment is intentionally separate from
`.venv-offline-learning`. A no-motion replay is:

```bash
.venv/bin/python infer_conical_direction.py \
  --trace-csv ../data/M12_event_trigger_insert_live_20260724_023025_306/conical_directservo_trace.csv \
  --output-json /tmp/m12_preflight.json \
  --model-root bundles/real_multisize_direction_v1 \
  --manifest bundles/real_multisize_direction_v1/deployment_manifest.json \
  --size m12 \
  --target-down-force-n 3.5
```

The pinned input SHA-256 is
`e2de7a701031175ffdd0a656f96f2489d28b06cb8f033487227cb5717d4b5aaa`.
The current verified replay returns finite normalized probabilities,
`hardware_control_allowed=true`, TOP1 `191.25 deg`, and completes well below
the 3 s preflight limit on the lab PC.

The former
`hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving`
bundle remains historical evidence only. Its `SHADOW_ONLY` result and
simulation transport gate are not the active controller contract.

No MATLAB, ROS, sensor, KST, or robot connection is needed for these replay
checks.
