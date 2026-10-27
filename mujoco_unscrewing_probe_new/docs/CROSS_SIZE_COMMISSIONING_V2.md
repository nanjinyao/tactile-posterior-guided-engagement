# Cross-size commissioning v2

## Physical scaling contract

Let

\[
s=\frac{r_{\mathrm{bit}}}{r_{\mathrm{bit,M8}}}.
\]

The M8/M10/M12 MuJoCo meshes all assign the bit a mass of 0.5 kg.  With fixed
mass and translational stiffness \(K_t\propto 1/s\), the nominal response time
is therefore

\[
T_t\propto\sqrt{M/K_t}\propto\sqrt{s},
\]

not \(s\).  v2 uses \(\sqrt{s}\) for capture, handoff, recovery, and descent
rate scheduling.  Translation uses \(K_t\propto 1/s\) and
\(D_t\propto1/\sqrt{s}\).  Since a similar angular error acts through a lever
arm proportional to \(s\), rotation uses \(K_r\propto1/s^2\) and
\(D_r\propto1/s\).

The raw hard-wrench vector is copied unchanged from M8:

\[
[20,20,20\ \mathrm{N},\ 60,60,60\ \mathrm{mNm}].
\]

No terminal-verifier threshold is size-relaxed.

## Terminal-set state transition

Development telemetry showed that a rejected verifier dwell was incorrectly
treated as lost capture.  Retraction and re-touch generated a second impact.
v2 separates these events:

1. Before CAD-depth entry, a true funnel-guard event still invokes the ordinary
   unload/retract/re-touch recovery.
2. On first CAD-depth entry, v2 rebases nominal XYZ to the measured tip.  This
   releases the lateral command error between the old capture point and the
   position already corrected by the chamfer/socket.
3. The axial admittance integrator is disabled only for the terminal dwell,
   holding the measured z pose without a command jump.  Raw wrench monitoring
   remains active at every physics step.
4. The unchanged terminal verifier must still accumulate either the force dwell
   or 100 consecutive stable low-wrench depth samples.
5. A rejected dwell inside the terminal set is retried in place.  It is not a
   license to declare success.

All three new switches default to `False` in `FunnelGuardConfig` and are enabled
explicitly only by the larger-size commissioning-v2 interface.

## Development evidence and boundary

The pre-fix event telemetry on M10 `002/006/007` showed:

- `006`: all 12,000 samples per attempt remained inside the depth band, but
  torque interruptions limited the low-wrench streak to 75--79/100.
- `007`: roughly 38 depth-band entries/exits per attempt; the oracle hold point
  was about 0.24 mm from center while physical guidance had already moved the
  tip to about 0.12 mm.
- `002`: 0.08--0.09 mm old-command/measured-tip mismatch; 100-sample low-wrench
  windows occurred but their depth span exceeded 0.02 mm.

In the frozen final targeted development run, the oracle completed all three
poses with zero hard stops and zero terminal rejections.  The S4b fine-tuned
arm completed `002` and `007`; `006` remained a pre-capture search exhaustion,
which the terminal transition cannot fix.

These are reused, known-failed development poses.  They establish a mechanism
diagnosis and a commissioning candidate, not a confirmatory success rate.

The later frozen development-v5 full-loop run completed that pending check:
the privileged arm reached 12/12 on M10 and 11/12 on M12, while the S4b
fine-tuned arm reached 12/12 and 10/12 respectively with zero hard stops. Both
development gates passed. The first separately frozen untouched-v1 run then
found 12/12 and 12/12 privileged execution, but the fine-tuned arm reached
10/12 with one primary M10 hard stop and 10/12 with zero M12 hard stops. Thus
M12 passes the untouched gate and M10 fails its zero-hard-stop clause.

This is the intended separation between commissioning and confirmation:
development closure does not erase an untouched safety failure. Untouched-v1
is now permanently test-only and cannot be reused to tune the commissioning
numbers above.
