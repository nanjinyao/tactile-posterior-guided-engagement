# Hardware tooling v1.5: CAD-to-MuJoCo alignment contract

The printable hardware source is
`../real_device/model/parametric_tooling/design_parameters.json`; generated
STEP/STL/3MF files and topology checks are under its `generated/` directory.
`configs/real_tooling_v1.json` is the simulation-side contract.

## Installed force/torque sensor identity

The active profile records the installed sensor as Bota Systems SensONE Gen A,
product `BFT-SENS-CAT2-B4`, serial `SN001349`. The body remains the CAT2-B4
EtherCAT product, but the 2026-07-21 PC path is the direct Bota USB Binary CDC ACM
device `0483:5740 / 3885355D3532`; KUKA separately owns the dock Ethernet. The
matching active driver configuration is
`/home/ubuntu20/robot_ws/src/iiwa_lab_bringup/config/bota_usb_SN001349.json`.
Historical EtherCAT captures remain provenance rather than the active topology.

This establishes identity and an observation data path. The official SensONE
Gen A drawing additionally fixes the wrench origin at the center of the
tool-side mounting plane, 38.00 mm from the robot-side mounting plane, with
`+Z*` pointing robot-side to tool-side. The side center hole is an IMU location,
not the wrench origin. The measuring-range variant, calibration certificate,
and installed measurement-origin-to-pink-tool metrology remain unknown. The
contract therefore keeps
`qualification: observation_logging_only`; it does not yet support
hardware-aligned force control or absolute force/torque-limit claims.

## Integrated flange-tools

The first physical print showed that a separate receiver, supported cartridge
shoulder, and U-clip added print and assembly uncertainty without being needed
for three manual size changes. Revision v1.5 exports three complete
one-piece tools:

- `integrated_flange_peg_m8`;
- `integrated_flange_peg_m10`;
- `integrated_flange_peg_m12`.

Each current pink part preserves the historical flange CAD mating face and
mounting pattern, then joins it through a short solid root to the size-specific
peg. The
complete flange-tool is manually unbolted and replaced. There is no receiver,
cartridge fit, retaining clip, or M4 tool-change hardware. This removes those
interfaces as sources of compliance and pose error; it does not remove the need
to measure mounting repeatability, printed stiffness, mass, COM, inertia, and
the sensor measurement-origin-to-tip transform.

All three tools retain the same pink-CAD-mating-plane-to-tip datum of 42 mm and the same
13 mm lower contact length, measured from the physical tip and including the
tip chamfer. The lower peg AF remains 6/8/10 mm for M8/M10/M12.

### Peg-to-flange yaw datum

The first installed-axis inspection identified the photographed A/C reference
as the opposed local tangent planes on the flange's central R36 outer
cylindrical surface. Their normals are `45/225 deg` in canonical flange XY.
The photographed peg face was the legacy `60 deg` face, so all three complete
pegs are clocked by

\[
\psi_{\mathrm{shape}}=45^\circ-60^\circ=-15^\circ
\]

about pink-CAD `+Z`. The tip chamfer, uniform contact shaft, and AF14
transition rotate together. The flange, TCP axis, socket cavities, and fixture
markings do not rotate. Exported STEP sections measure the selected peg-flat
normal at `45.000 deg`, parallel to the flange tangent datum. CAD tolerance is
`0.05 deg`; the as-printed target is `1 deg`.

Per-size commissioning also requires a measured pink-CAD mating-plane-to-tip
distance of `42.00 +/- 0.25 mm` and radial tip-axis runout of at most
`0.20 mm`. These are provisional conservative FDM acceptance gates, not
population-derived manufacturing tolerances. The runout limit is below the
smallest `0.25 mm` single-side deep-wall clearance. A tool outside the height
band cannot use the current fixed 42 mm TCP term without an updated calibration
and requalification. Runout here means maximum radial tip-center eccentricity
from the fitted flange axis; divide a peak-to-peak dial-indicator TIR by two.

This angle is a part-construction parameter, not a calibrated transform.
Physical teaching must align the flange `+45 deg` tangent-datum normal with
fixture `+X`, after which the actual peg/socket relative yaw is measured. The
runtime must apply one measured canonical-to-world rotation consistently to
sensor `Fxy/Txy`, conical phase, learned direction posterior, and XY motion.
Only 60-degree multiples are exact hex symmetries; a residual 15-degree error
cannot be dismissed as augmentation.

The hardware-easy MuJoCo models intentionally keep peg and socket aligned at
relative yaw zero. The `-15 deg` CAD clocking must not be added again to the
MuJoCo bit, because flange-datum teaching removes it at assembly level. Existing
models can first be reused only when the measured residual peg/socket yaw lies
inside the current `[-2,+2] deg` sampling support and observations are expressed
in the training canonical frame. Otherwise all three sizes require sampling
around the measured yaw, fine-tuning, and a fresh audit.

## Strict and hardware-easy-v2 peg tips

The strict control remains available in the contract: the source tip face is
`0.98 * peg AF`, with 0.35/0.45/0.55 mm axial transitions. The selected physical
tools instead use a visible 45-degree FDM chamfer:

| peg profile/size | full AF | radial setback and axial length | tip-face AF |
|---|---:|---:|---:|
| simulation exact M8 | 6.00 mm | 0.06 / 0.35 mm | 5.88 mm |
| simulation exact M10 | 8.00 mm | 0.08 / 0.45 mm | 7.84 mm |
| simulation exact M12 | 10.00 mm | 0.10 / 0.55 mm | 9.80 mm |
| hardware easy v2 M8 | 6.00 mm | 0.60 / 0.60 mm | 4.80 mm |
| hardware easy v2 M10 | 8.00 mm | 0.80 / 0.80 mm | 6.40 mm |
| hardware easy v2 M12 | 10.00 mm | 1.00 / 1.00 mm | 8.00 mm |

Equal radial setback and axial length make the selected chamfers 45 degrees.
The larger tip chamfer makes initial edge contact less abrupt and gives a wider
geometric transition into the socket funnel. It is an engineering capture aid,
not a guarantee: friction, tilt, FDM burrs, multi-point jamming, structural
deflection, and controller behavior can still prevent insertion.

## Socket profiles and the selected hardware-easy-v2 fixture

The three socket mouths share one Z datum, lie at X = -45/0/+45 mm, and use a
flat-normal +X yaw convention. The 0.5 mm center bores are cut through the final
fused fixture; at FDM scale they are drill pilots and must be opened and
deburred after printing.

| socket profile/size | wall AF | 45-degree width/depth | mouth AF | outer head diameter |
|---|---:|---:|---:|---:|
| simulation exact M8 | 6.20 mm | 0.15 mm | 6.50 mm | 13.00 mm |
| simulation exact M10 | 8.24 mm | 0.18 mm | 8.60 mm | 16.50 mm |
| simulation exact M12 | 10.30 mm | 0.22 mm | 10.74 mm | 18.50 mm |
| hardware easy v2 M8 | 6.50 mm | 0.80 mm | 8.10 mm | 14.95 mm |
| hardware easy v2 M10 | 8.54 mm | 1.00 mm | 10.54 mm | 18.975 mm |
| hardware easy v2 M12 | 10.60 mm | 1.20 mm | 13.00 mm | 21.275 mm |

The selected wall clearances are 0.25/0.27/0.30 mm per flat side relative to
the 6/8/10 mm pegs. The larger outer heads keep useful material around the
larger funnels and make the geometry visually balanced. Enlarged, engraved
labels are fixture markings only and do not enter the contact model.

The physical interpretation is deliberately limited. The wider mouth and the
matched peg/socket chamfers enlarge the geometric region in which an edge can
generate a restoring contact normal. The wider straight wall also lowers the
risk that FDM dimensional error produces an unintended press fit. Conversely,
it increases translational and rotational free play after entry; yaw wall
contact and torque evidence can therefore occur later and at a different
threshold than in the strict source geometry. A 45-degree nominal surface does
not override Coulomb friction or guarantee inward sliding.

For aligned yaw and motion along a hex-flat normal, a useful geometric
diagnostic (not a guaranteed capture radius) is

\[
c_{\mathrm{cap}}
=
\frac{S_{\mathrm{wall}}-P}{2}
+w_{\mathrm{socket}}+w_{\mathrm{peg}}.
\]

It gives 1.65/2.07/2.50 mm for M8/M10/M12. At the centered deep-wall position,
the corresponding ideal geometric yaw free play is approximately
plus-or-minus 9.75/7.59/6.63 degrees. These values expose the intended trade:
capture is easier, while wall-torque onset is later and the tactile yaw cue may
be weaker. The 42 mm pink-CAD-plane lever arm also remains mechanically relevant;
for example, a 5 N lateral force corresponds to 0.21 N m about that plane before
using the actual Bota measurement origin.

Every compared hardware method may use this same fixture for a fair within-
hardware comparison. It must not be reported as a strict nominal-geometry
sim-to-real result. Geometry-matched simulation must instantiate both the
`hardware_easy_v2` socket and peg-tip profiles from
`configs/real_tooling_v1.json` in a separate scene; frozen source JSON/XML files
must not be overwritten.

An easy fixture can saturate every method if the entire initial-error
distribution lies inside its enlarged guiding region. Hardware comparisons
must therefore preregister a shared offset/yaw distribution and finite time
budget that include cases outside that region. Direct capture, search-assisted
capture, timeout, guarded abort, and terminal success remain distinct outcomes.

Printer compensation is independent of the intended geometry. The selected
wall and mouth AF values must be corrected only after measuring the internal-
hole AF error from the size-specific coupons. The unvalidated `fdm_trial` file
is not the selected physical geometry.

## Exterior-rib boundary

The physical M8/M10/M12 head exteriors carry 18/24/30 shallow vertical ribs
below a common 2 mm smooth top band. The current MuJoCo
`screw_head_side_visual` is collision-disabled, so these ribs do not change the
current simulated contact. Current simulation also cannot certify trajectories
that fall down the physical head exterior. If exterior-side collision is
enabled later, outer-boundary and global-fallback experiments must be rerun.

## What is deliberately not guessed

The official Bota wrench origin is now defined, but the installed transform
from that origin to the current pink CAD mating plane is not. FDM mass also
depends on material, walls, infill, and as-built geometry. The following
therefore remain null:

- full sensor measurement-origin to pink-CAD-mating-plane transform;
- full sensor measurement-origin to tip vector;
- each complete integrated flange-tool mass, COM, and inertia;
- fixture-to-robot/world transform.

Until these fields are measured, this profile is not admitted for hardware-
aligned wrench simulation. CAD volume is traceability data only; it is not a
substitute for measured mass.

`src/hardware_tooling_profile.py` enforces this boundary. It verifies the CAD
parameter and STEP hashes, one-piece architecture, 42 mm datum, peg-to-flange
yaw contract, both peg-tip profiles, both socket profiles, and the fixed socket
layout. It lists missing metrology and per-size clocking requalification and
raises `IncompleteHardwareProfileError` before an incomplete profile can
provide a calibrated wrench lever arm.

## Runtime change

`MuJoCoAdmittanceRollout` accepts an explicit `wrench_reference_to_tip_m`.
Supplying it makes the contact moment use the calibrated reference point.
Omitting it retains the frozen legacy simulation value
`0.18 * bit_length_m`; that fallback is only for reproducing previous evidence
and must not be used for hardware-aligned runs.

For force `F`, contact point `p_c`, and sensor reference `p_s`, the reported
moment must use

\[
\tau_s = \tau_c + (p_c-p_s)\times F.
\]

The 42 mm pink-CAD-plane datum is not automatically the Bota measurement-origin
lever arm. The complete measured vector, including the white flange, must be
passed to every controller, renderer, and data sampler that uses torque.

## Required commissioning sequence

1. Print and measure the three hardware-easy-v2 socket coupons; select AF
   compensation, then print the fixture and open/deburr its center drill pilots.
2. Print the three integrated flange-tools sensor-face-down, inspect the tip and
   root, and verify flange seating, bolt access, common tip height, runout,
   `<=1 deg` selected-flat/tangent-datum parallelism, and repeatability after
   repeated whole-tool swaps.
3. Obtain the exact Bota range/calibration; use the official measurement-frame
   definition and measure `T_sensor_to_pink`, tip touch-off, each complete tool's mass, COM, and
   inertia.
4. Fill the null fields and verify that the loader returns the calibrated
   sensor-to-tip lever arm.
5. Teach the flange tangent datum against fixture `+X`, measure each
   peg/socket relative yaw, and run a low-load fingerprint/capture
   requalification. Old v1.4 poses/releases remain historical and must fail
   closed for v1.5.
6. Build a separate hardware-specific MuJoCo scene using the selected socket
   and peg-tip geometry. Do not overwrite the frozen source scenes or add the
   flange construction angle as a second peg/socket yaw.
7. Re-identify yaw wall-contact onset and torque observability, then retune
   force/torque and terminal Fz/dwell gates. Re-run fingerprint calibration,
   entropy-grid thresholds, and all geometry-matched full-loop comparisons
   before reusing any legacy success-rate claim.
