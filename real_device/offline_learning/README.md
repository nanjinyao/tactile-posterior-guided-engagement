# Real multisize offline learning

This directory is offline-only.  It imports no ROS, KST, or Bota command
interfaces.

Create the isolated CPU environment once from `real_device`:

```bash
./offline_learning/bootstrap_offline_learning.sh
```

Then rebuild, calibrate, train, export, report, and run the focused contract
tests with one command:

```bash
./offline_learning/run_pipeline.sh
```

The builder reads only the three explicit fresh-run allowlist entries.  The
trainer opens only `dataset/train_clean.npz`; the consumed holdout,
quarantine, and sentinel arrays are retained for diagnostics but never opened
by model selection or fitting.

Selection is clean-only leave-one-size-out.  C6 augmentation stays inside the
training fold and retains its physical origin group.  The exact candidates
are: (1) real-only Fourier/circular, (2) frozen simulation encoder plus real
input/output adapter, and (3) candidate 2 with only `shared.0` unfrozen and
anchored to the pretrained simulation value by L2-SP.  The old tiny CNN is
not a selection candidate.  Both simulation-assisted candidates read only
`clean_candidate` rows from the real-calibrated replay, exclude the held-out
size, and use a fixed 0.8 real / 0.2 simulation loss.  Real data alone
determines normalization.

The exported bundle is
`model_runtime/bundles/real_multisize_direction_v1`.  Its authority is limited
to high-level posteriors and capped CAD shortcut/residual-corridor selection;
it has no capture, insertion, terminal-verification, Pure-Outward, or online
learning authority.  Initial hardware control is M12-only; M10 and M8 supply
training evidence but remain inference-only.

The fixed M10 demonstration uses a separate, exact-path/hash runtime pilot
overlay.  It does not alter this exporter, retrain the bundle, or widen the
M12-only base promotion contract.  Its scope and independent MATLAB safety
envelope are documented in
`configs/m10_fixed_demo_pilot_authorization_v1.json`.

The CAD clocking contract is
`v1.5-integrated-flange-tools-yaw-datum`.  Its historical `-15 deg` shape
correction is already represented by the selected `+45 deg` peg-flat normal
and the operator-taught `0 deg (mod 60 deg)` fixture alignment.  It must not
be applied again in simulation, augmentation, runtime inference, or control.
That fixture alignment is an operator-taught commissioning assumption, not an
as-built yaw measurement.
