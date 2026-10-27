# Parametric real-device tooling

Active design revision: `v1.5-integrated-flange-tools-yaw-datum`  
Frozen on: 2026-07-24 after the first installed-axis inspection.

## 1. Active hardware architecture

The quick-change stack is retired.  Each size is now one printable solid:

`current pink CAD flange -> solid D44/D36 load path -> AF14 reinforcement -> M8/M10/M12 peg`

The complete flange-tool is manually unbolted and exchanged.  There is no
separate receiver, cartridge fit, shoulder support, or retaining clip.  This is
slower than a quick-change mechanism, but it removes the fit hysteresis and the
unsupported cartridge shoulder that failed in the first physical print.

The original flange outline and four mounting holes are reconstructed from
`../fast_mount_ATI_ft_1.STEP`.  The generator checks its volume against the
source STEP and verifies that the complete exported tools retain the clean
flange.  The currently installed force/torque sensor identity and its still
known official measurement-origin definition and still-pending installed
measurement-origin-to-tool metrology are recorded separately in
`mujoco_unscrewing_probe_new/configs/real_tooling_v1.json`; the source filename
must not be treated as a sensor-frame calibration.

## 2. Files to print

Final physical set:

- `generated/integrated_flange_peg_m8.step`
- `generated/integrated_flange_peg_m10.step`
- `generated/integrated_flange_peg_m12.step`
- `generated/socket_base_hardware_easy_v2.step`

Recommended calibration coupons before the final base:

- `generated/socket_fit_coupon_m8.step`
- `generated/socket_fit_coupon_m10.step`
- `generated/socket_fit_coupon_m12.step`

Research controls, not the default hardware print:

- `generated/socket_base_nominal.step` reproduces the strict MuJoCo socket.
- `generated/socket_base_fdm_trial_p0p10mm_per_side.step` is only a labeled
  printer-compensation trial; it is not a measured correction.

The retired v1.3 quick-change source and every old generated artifact are kept
under `archive/v1_3_quick_change_2026-07-21/`.  They are intentionally absent
from the active `generated/` directory.

## 3. One-piece flange tools

All three tools use the same coordinate convention and common datum:

- current pink CAD mating plane: `Z = 0 mm`;
- current pink flange tool-side plane: `Z = -8 mm`;
- physical peg tip: `Z = -42 mm`;
- lower working length from physical tip to AF14 reinforcement: `13 mm`;
- socket selected hex-flat normal: `0 deg` (`+X`);
- integrated peg construction rotation from the legacy phase: `-15 deg`
  about canonical `+Z`;
- integrated peg selected flat normal in the flange CAD frame: `+45 deg`.

The solid root is `D44 -> D36 boss -> D16.2 collar -> AF14`.  When the
sensor-side flange face is placed on the print bed and the peg points upward,
every section after the flange becomes smaller with height.  The tool therefore
does not require the shoulder support that produced the first-print spaghetti.

### Hardware-easy-v2 peg tips

The source simulation used a `0.98` tip scale, which means only
`0.06/0.08/0.10 mm` radial setback per flat and was nearly invisible after FDM.
The physical tools now use a true 45-degree chamfer while retaining 80% of the
shaft AF at the tip face:

| Size | Shaft AF | 45-degree peg chamfer (radial = axial) | Tip-face AF | Full-AF straight length |
|---|---:|---:|---:|---:|
| M8 | 6.00 mm | 0.60 mm | 4.80 mm | 12.40 mm |
| M10 | 8.00 mm | 0.80 mm | 6.40 mm | 12.20 mm |
| M12 | 10.00 mm | 1.00 mm | 8.00 mm | 12.00 mm |

The tip remains at `Z=-42 mm`; enlarging the chamfer does not alter the common
TCP length.  The strict source tip remains recorded in the parameters and the
hardware-aligned MuJoCo profile must use the new physical tip.

### v1.5 peg-to-flange yaw datum

The installed inspection identified the useful flange reference as the two
opposed local tangent planes on the central R36 outer cylindrical surface
(the photographed A/C regions), not the straight sides of the four mounting
arms. Their outward normals are `45/225 deg` in canonical flange XY. The
legacy selected peg flat was at `60 deg`, so the smallest clocking correction
is

$$
\Delta\psi_{\mathrm{peg/flange}}=45^\circ-60^\circ=-15^\circ.
$$

The generator applies this angle to the complete peg: tip chamfer, uniform
contact shaft, and AF14 transition. It does not rotate the flange, socket
cavities, fixture markings, or TCP axis. STEP re-import checks all three peg
sections and reports a selected flat normal of `45.000 deg`; CAD residual to
the flange tangent datum is zero within the `0.05 deg` numerical tolerance.
The physical acceptance target is `<=1 deg`, measured after printing.
The same commissioning record must measure the pink-CAD mating plane to tip as
`42.00 +/- 0.25 mm` and radial tip-axis runout as `<=0.20 mm`. These are
provisional conservative FDM gates: 0.20 mm is below the smallest 0.25 mm
single-side deep-wall clearance. Outside the height band, the current fixed
42 mm TCP term is invalid and must be repaired or recalibrated before motion.

This is a visible mechanical teaching datum, not a sensor-to-tool or
fixture-to-world calibration. For the current socket yaw `0 deg`, teach the
installed flange so the **outward normal of the B/C local tangent plane** (not
the tangent line itself) maps to fixture `+X_F`. Because that same normal and
the selected peg-flat normal are both `+45 deg` in flange-CAD coordinates,
this one installation mapping puts the selected peg-flat normal at fixture
`0 deg`, coincident with the socket selected-flat normal at fixture `0 deg`.
The peg's `-15 deg` construction correction is already embedded in the printed
CAD: do not apply a second `-15 deg`, `+15 deg`, or `+45 deg` compensation in
commissioning, simulation, training, or runtime transforms. Then measure the
actual peg/socket relative yaw. Old v1.4 work poses and releases are not valid
authorization for the reprinted v1.5 tools.

The yaw preview is explicitly an underside view: the viewer is at the peg tip
looking toward the flange along `+Z`. It keeps flange/fixture `+X` to screen
right, so `+Y` necessarily points screen-down, `+Z` points into the page, and
positive right-handed yaw about `+Z` appears clockwise. A screen-up `+Y` plot
would instead be the opposite (flange-side) view and must not be used to judge
the physical clocking.

The current physical tool is the **pink** one-piece flange tool. Some generator
inputs and generated-manifest keys still contain the historical word `blue`;
those are CAD-local legacy names for this delivered solid, not the distinct
blue reference connector that was ruler-observed at 10–11 mm. The active
hardware contract uses the pink CAD mating plane and excludes that separate
blue observation from the pink TCP equation.

## 4. Hardware-easy-v2 socket base

The final physical base deliberately differs from the strict source geometry:

| Size | Peg AF | Deep-wall AF | Per-side deep gap | 45-degree socket chamfer | Mouth AF | Outer head diameter |
|---|---:|---:|---:|---:|---:|---:|
| M8 | 6.00 | 6.50 | 0.25 | 0.80 | 8.10 | 14.95 mm |
| M10 | 8.00 | 8.54 | 0.27 | 1.00 | 10.54 | 18.975 mm |
| M12 | 10.00 | 10.60 | 0.30 | 1.20 | 13.00 | 21.275 mm |

The outer heads and support necks are scaled by `1.15x`, leaving approximately
`2.80/3.40/3.13 mm` of minimum top-rim material at the hex corners.  The three
center bores remain `0.5 mm` true through-holes.  The M8/M10/M12 markings are
now bold 7 mm text engraved 0.8 mm deep; `+X` is 6.5 mm and the arrow is also
engraved.  All markings lie on the recessed plate and do not touch the active
socket contact surfaces.

## 5. Physical interpretation and claim boundary

For yaw-aligned hexes and motion along a flat normal, a useful first-contact
estimate is

`c_guide = deep-wall gap + socket-chamfer width + peg-chamfer width`.

It gives `1.65/2.07/2.50 mm` for M8/M10/M12, about
`5.32x/5.45x/5.32x` the corresponding source-geometry estimate.  This number
means only that the tip has reached the combined guiding region.  It is not a
guaranteed capture radius: yaw error, tilt, corner contact, FDM burrs, friction,
and insufficient compliance can still jam the pair.

The larger deep-wall gaps also permit approximately
`+/-9.75/+/-7.59/+/-6.63 deg` of ideal centered hex yaw before wall contact.
This helps insertion but delays torque contact and can weaken the tactile
direction cue.  Hardware operation therefore requires known-yaw prealignment,
low-speed/low-force Z motion, XY and tilt compliance, moment limits, and a
terminal check using depth plus low lateral wrench and, where torque transfer
is required, a bounded seating twist/torque gate.

The one-piece tool is more rigid and repeatable than the retired printed
quick-change interface, but the pink-CAD-plane-to-tip lever arm is still 42 mm.
For example, 5 N lateral load still produces about 0.21 N m about that plane.
Actual sensor-origin-to-tip transform, printed mass, center of mass, inertia,
static deflection, return-to-zero, and safe wrench thresholds remain hardware
measurements, not CAD facts.

All compared algorithms may use the same hardware-easy-v2 fixture for a fair
hardware comparison.  It must be disclosed as an engineering capture bias,
not strict nominal sim-to-real geometry and not evidence that an estimator
became more accurate.  A hardware-matched simulation rerun and terminal-gate
retuning are required before reusing previous success-rate claims.

The easier mechanics can also saturate every method if all initial offsets lie
inside the enlarged guiding region.  The preregistered hardware offset/yaw
distribution and finite time budget must therefore include trials outside that
region, while keeping exactly the same distribution and budget for every
method.  Report direct capture, search-assisted capture, timeout, guarded abort,
and terminal success separately.

## 6. Regenerate and verify

From `real_device/`:

```bash
.venv-cad/bin/python model/parametric_tooling/generate_models.py
python3 model/parametric_tooling/render_previews.py
python3 model/parametric_tooling/verify_print_meshes.py
```

`generated/manifest.json` is the machine-readable geometry and SHA-256 record.
It includes STEP re-import sections for all peg tips and socket profiles,
through-hole checks, common-datum checks, the combined-funnel calculation, and
the explicit peg/flange yaw contract and non-guarantee boundary.
`generated/mesh_validation.json` records watertight/manifold/self-intersection
checks for every active STL and 3MF (nine parts in each format, 18 meshes in
total). The tapered hexes are exported as explicit planar faces; this avoids
the self-intersecting tessellation that a generic B-spline loft produced for
the rotated M8/M10 tools.

`generated/preview_peg_flange_yaw_alignment.png` is the handed tip-side
orthographic datum drawing used for the physical teaching check. Its left
panel verifies the CAD-local `-15 deg -> selected flat +45 deg -> B/C normal
+45 deg` chain. Its right panel shows the single installation mapping
`B/C tangent NORMAL -> +X_F`, after which the peg selected-flat normal and the
socket `0 deg` normal coincide; no second yaw compensation is permitted.

Detailed print order and acceptance checks are in
`PRINTING_STEP_CHECKLIST_CN.md`.
