# Cross-size S4b full-loop audit

## Decision

The base-anchored S4b arm closes the frozen gate on both sizes in
development-v5, but the first untouched-v1 test closes only M12. M10 reaches
the required bottom rate but violates the zero-hard-stop clause. Therefore:

- development closure: yes, for M10 and M12;
- independent two-size closure: no;
- fine-tuning sufficiency across sizes: not established;
- fine-tuning necessity: false under the current paired controls;
- hardware-data necessity: untested.

Safe abort, timeout, search exhaustion and terminal rejection are failures for
strict-bottom scoring even when they are reasonable closed-loop outcomes.

## Frozen gate

For size $s$,

$$
G_s=(r_{\rm bottom}\ge0.80)\land(N_{\rm hard}=0)
\land(N_{>90^\circ,\,\rm executed}=0)
\land(r_{\rm oracle}\ge0.90).
$$

The direction clause applies only when a non-zero posterior-directed shortcut
is actually dispatched. Latent errors on an abstained posterior remain
diagnostic. All arms share a 130 s active budget, zero extension, zero simulator
reset, size-conditioned admittance/R3 and the original 20 N / 60 mN m raw hard
limits.

## Results

| phase / arm | M10 bottom / hard / gate | M12 bottom / hard / gate |
|---|---:|---:|
| development-v5 fine-tuned | 12/12 / 0 / pass | 10/12 / 0 / pass |
| untouched-v1 zero-shot | 12/12 / 0 / pass | 10/12 / 0 / pass |
| untouched-v1 calibration-only | 12/12 / 0 / pass | 11/12 / 0 / pass |
| untouched-v1 fine-tuned | 10/12 / 1 / **fail** | 10/12 / 0 / pass |
| untouched-v1 privileged center | 12/12 / 0 / pass | 12/12 / 0 / pass |

Untouched necessity is false because the legal weight-identical
calibration-only control passes on both sizes; zero-shot also passes M10.

## Untouched failure mechanisms

### M10 pose 006: primary raw-torque stop

- control error: 5.972 degrees; accepted shortcut: 2.0 mm;
- nominal corridor nearest-center distance: 0.103896 mm;
- measured nearest-center distance: 0.604552 mm;
- command-tracking error maximum: 1.121045 mm;
- corridor soft-escapes and global fallback begins at 11.247 s;
- contact-loaded `fallback_reset` triggers `raw_physical_wrench_limit:Ty` at
  13.855 s;
- raw torque norm reaches 63.726050 mN m; the frozen per-axis threshold implies
  only the strict statement $T_y>60$ mN m, not an exact $T_y$ value;
- `result=hard_stop`, `terminal_attempts=0`, and
  `post_terminal_event_display_new_hard_stop=false`.

This is a primary active-rollout safety failure. Nominal geometric corridor
coverage did not imply measured contact-path coverage.

### M10 pose 010: lost cue after release

The 2.0 mm shortcut has 4.108 degrees control error. A sensor-progress candidate
appears, but offline capture is false and low-force requalification remains
0/100. The controller exits with a named safe abort at 10.2385 s, without a
hard stop or terminal attempt.

### M12 pose 005: recovery does not guarantee recapture

One complete `tilt_limit` recovery executes ownership entry, unload, retouch
and resume with raw peaks of 6.994 N / 28.720 mN m. Requalification still
remains 0/100, so the controller safely aborts before terminal insertion. This
trial demonstrates why recovery changes reachable state and failure
consequence, but cannot guarantee success.

### M12 pose 010: correct abstention, later false cue

Fine-tuned concentration is 0.307915, below the validation-selected 0.60 floor.
The adapter correctly sets shortcut length to zero. A 109.663 s global-outward
search later produces a remote chamfer cue at radius 14.300432 mm; low-force
requalification remains 0/100 and rejects it. This failure must not be
attributed to an executed wrong direction.

## Provenance

Development-v5:

- task contract:
  `4ce62d410bc802ebe45d840dafeffed9837694360321adfc3f833bcb4b5d5b7f`;
- execution bundle:
  `b10befd13b3ecc17f87fa22591135253a74e4c9532fa03286e0e4de3901308f3`;
- run ID: `2cb77b51a0fe44d6b6d1ac63bb256527`.

Untouched-v1:

- task contract:
  `ad64b03372653fbdf7fd838d6a09bbd352346435be78315a6f9fcb61111dd724`;
- execution bundle:
  `a32d7ab6a72840254190ada55a712e475e0808bc1ad180e9d05d252ef7ca84dd`;
- run ID: `2401daf1df1448319946fb2404fc3ec6`.

All 24 trials in each phase use one run ID, contain no worker error, preserve
learned first-probe hashes, and match worker-init, worker-completion and parent
bundle hashes. Untouched-v1 was executed once and is permanently forbidden for
training, calibration, controller/gate selection or tuned confirmation.

## Claim boundary

These are MuJoCo geometric-engagement results. Strict success includes offline
true-center/contact-tag scoring unavailable to the deployed non-oracle
controller. Straight-to-center is a privileged execution diagnostic, not a
deployable policy or guaranteed upper bound. No result here proves real-robot
robustness, hardware-data fine-tuning necessity, motor coupling or unscrewing.
