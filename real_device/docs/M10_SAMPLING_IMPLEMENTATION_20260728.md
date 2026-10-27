# M10 26-point sampling implementation — 2026-07-28

The M10 acquisition lane reuses the M12 completion-first sampling state
machine while keeping M10 plan, campaign, checkpoint, data prefix, dashboard
and printed-peg identity separate.

- Printed geometry: head D18.975, mouth AF10.54, wall AF8.54, tip AF6.4,
  shaft AF8.0, bore D0.5.
- Execution order:
  `[4,14,10,5,6,25,24,12,20,16,11,22,19,26,23,9,18,3,15,8,21,2,17,7,13,1]`.
- Closed route: 97.999 mm; sentinel separations: 20 and 12.
- Clean-state coverage: supported/double/outer-partial = 6/6/4.
- Current M10 gravity is handled by the fresh start and per-point air
  baselines; raw Bota six-axis data remains unchanged.
- The top view retains `(u,v)=(-x_F,-y_F)` and uses actual M10 dimensions.
- No UI, additional KST query, subscriber or added file write occurs inside
  the DirectServo loop.

Lightweight no-motion checks passed:

- M10 focused static tests: 8/8.
- M12 sampling regression: 40/40.
- Conical guard regression: 13/13.
- Real baseline regression: 31/31.
- Deterministic M10 and M12 plan generators: current.

No ROS, Bota or KST session was started and no robot command was sent during
these checks. MATLAB/Octave CLI was unavailable on this host, so `checkcode`
was not run.

Key SHA-256 values at implementation handoff:

- shared executor:
  `9d12f704f29c4f95a8847b48284de16c92e7b8f12a4b442e84d61fa359ad65e4`
- M10 plan:
  `9b09f0d9cd7ff5db454bae986e7c2e74e724d19549ca1a927952cbcd55ef3859`
- M10 installation record:
  `d38f4e332f2db51943e8e5fece10e73ac9b5601778388d6fd9c80b73472e6254`
- current operator plan PNG:
  `451f36d88f60120c95106346addc653c59dbec4779fe4ff0bb7c502104ab0c44`
