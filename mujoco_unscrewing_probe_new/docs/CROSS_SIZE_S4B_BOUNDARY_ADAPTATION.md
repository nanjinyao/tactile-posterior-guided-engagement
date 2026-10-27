# Cross-size S4b outer-boundary adaptation contract

S4b is a development-data revision, not confirmatory evidence. Development v2
exposed an aggregate observability gap near the full-loop visual boundary. S4b
uses that diagnosis only to add a generic geometry-defined coverage ring:

$$
r_5 = r_{\mathrm{vis}}
= \max(\texttt{source\_geometry.visual\_offset\_sweep\_m}).
$$

The original five S4 rings are retained. The new sixth ring is labelled
`double_overhang`, is still inside the inner-overlap domain, and is instantiated
as complete C6 orbits at role-specific phase anchors. Train, validation and
calibration therefore contain 72, 36 and 36 scans per size (12, 6 and 6 complete
orbits), with no shared orbit or physical $(x,y)$ point across roles.

The generator does not load or copy a development task pose, nuisance draw,
rollout trace, or raw fingerprint. New Sobol and sensor-noise seeds are frozen in
the S4b contract. Development-v2 tasks are permanently ineligible for a later
S4b result; a new untouched task contract is required after the model,
controller and gates are frozen.

Contract-only generation (no dynamic sampling):

```bash
python3 -m src.sample_cross_size_s4b_adaptation
```

Later opt-in sampling, after reviewing the contracts:

```bash
python3 -m src.sample_cross_size_s4b_adaptation --sample --workers 24
```

The intended, non-overwriting model command is:

```bash
python3 -m src.finetune_cross_size_s4_state_direction_heads \
  --data outputs/current/datasets/cross_size_s4b_adaptation \
  --output outputs/current/models/cross_size_s4b_state_direction_heads
```

The existing `outputs/current/datasets/cross_size_s4_adaptation` and
`outputs/current/models/cross_size_s4_state_direction_heads` trees are not S4b
outputs and must not be overwritten.
