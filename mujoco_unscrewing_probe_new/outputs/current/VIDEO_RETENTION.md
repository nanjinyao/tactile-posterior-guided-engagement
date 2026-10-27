# Video Retention

`VIDEO_RETENTION.json` is the machine-readable source of truth. Five video
files representing four selected mechanisms are current:

1. retry-x3 muted-fingerprint/short-shortcut 2x2 (`sobol_016`);
2. retry-x3 clear-fingerprint/long-shortcut 2x2 (`sobol_000`);
3. improved four-channel versus Jin-style three-channel direct-x3 (`sobol_005`);
4. actual M8 entropy-grid top-view replay (`sobol_003`) in Chinese and English,
   both rendered from the same audited replay.

The two 2x2 matrices and the estimator-interface video come from the current
five-method retry-x3/shared-R3 evidence pass. The entropy video is an exact
selected strict replay with measured MuJoCo tip trace and explicitly offline
capture-region overlays. All are mechanism examples; success/time ranking
comes only from
`evaluations/m8_five_method_retry3_r3_closed_loop/`.

The registry also points to the frozen Sobol-016 Proposed+ physics audit and
the ten-arm bumpless-transition development regression. The former prevents a
selected successful video from being mislabeled as entropy convergence; the
latter is mechanism evidence (7 strict bottoms, 3 named safe aborts, 0 hard
stops), not a distributional success-rate estimate.

The improved-versus-Jin result is mixed, not a superiority result. The fixed
physical-funnel stress ablation shows an R3 completion improvement but not a
guarantee. Cross-size S4b development-v5 passes the M10 and M12 fine-tuned
gates, but the first permanently test-only untouched-v1 readout fails M10 on
one primary hard stop and passes only M12. Zero/calibration controls also pass,
so two-size sufficiency, fine-tuning necessity and hardware transfer are not
established.

Eleven MP4s are retained but downgraded in the JSON registry: the old one-ray
strict matrices, superseded strict entropy matrix, old Proposed+ and planar
entropy explainers, capture-era matrices, guard/extreme-recovery clips, and the
historical six-method/race composites. They remain reproducible diagnostics
and are not current paper-facing evidence.

Older stage, component and development videos were archived outside the active
project on 2026-07-12:

```text
/home/njy/njy_ws/archive/group_meeting_legacy/
mujoco_unscrewing_probe_new_historical_videos_2026-07-12.tar.zst
```

Archive SHA-256:
`009d413425186744edee68acc68b8e13b9558e9c1d9d2d2135c188830fcc2b01`.
The sibling checksum, `zstd -t`, and a pre-removal `tar --compare` verified 43
members. Restoring an old component restores provenance only; it does not
promote that component to current evidence.
