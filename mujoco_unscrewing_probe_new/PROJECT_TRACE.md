# Project Trace

Append-only trace for the clean MuJoCo unscrewing probe path.

---

## 2026-07-09 23:23 +08 - Scope decision recorded

Goal:
- Resolve the difference between a broad ChatGPT roadmap and the meeting-driven M8-first implementation.

Changed:
- Added `DESIGN_DECISIONS.md`.
- Updated `PROJECT_RULES_AND_LESSONS.md`.

Result:
- ChatGPT roadmap is recorded as medium-term.
- Immediate implementation remains M8-first, with M10/M12 deferred to scale variation after M8 evidence exists.

Claim update:
- Implemented: decision record.
- Partial: geometry support not yet built at this trace point.
- Not proved: M10/M12 generalization, active information-gain planner.

Next:
- Build clean parameterized MuJoCo model files.

---

## 2026-07-09 23:35 +08 - M8/M10/M12 source-style MuJoCo models generated

Goal:
- Create precise parameterized M8/M10/M12 Allen socket-head engagement models in the clean path.

Changed:
- Added `src/allen_socket_model.py`.
- Added generated configs under `configs/`.
- Added generated MuJoCo XML scenes under `models/`.
- Added `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`.
- Added `docs/M8_M10_M12_MODEL_INVENTORY.md`.
- Added `outputs/current/m8_m10_m12_model_inventory.json`.
- Added `README.md` and `CLAIM_BOUNDARY.md`.

Command / run:
```bash
python3 -m py_compile src/*.py
python3 -m src.allen_socket_model --sizes m8,m10,m12
python3 - <<'PY'
from pathlib import Path
import mujoco
for path in sorted(Path("models").glob("allen_*_source.xml")):
    model = mujoco.MjModel.from_xml_path(str(path))
    print(path, "nbody", model.nbody, "ngeom", model.ngeom, "nmesh", model.nmesh, "nq", model.nq)
PY
```

Artifacts:
- `configs/allen_m8_source.json`
- `configs/allen_m10_source.json`
- `configs/allen_m12_source.json`
- `models/allen_m8_source.xml`
- `models/allen_m10_source.xml`
- `models/allen_m12_source.xml`
- `docs/M8_M10_M12_MODEL_INVENTORY.md`
- `outputs/current/m8_m10_m12_model_inventory.json`

Result:
- MuJoCo 3.8.1 loaded all three XML scenes.
- Each model has `nbody=3`, `ngeom=88`, `nmesh=81`, `nq=7`.

Claim update:
- Implemented: model-ready M8/M10/M12 fixed Allen socket-head engagement scenes.
- Partial: these are geometry fixtures only; no controller/dataset/classifier evidence yet.
- Not proved: M10/M12 generalization or real-robot transfer.

Next:
- Add visual-error offset sampler and source-style baseline.

## 2026-07-09 23:44 +08 - top-view initial-position projection added

Goal:

- Make the peg/bit and socket/hole initial-position space visible in top view.
- Clarify which initial offsets are normal engagement samples and which are
  unsafe / abort / re-detect cases.

Added:

- `src/plot_top_view_initial_positions.py`
- `outputs/current/top_view_initial_position_cases_m8.png`
- `outputs/current/top_view_initial_position_cases_m10.png`
- `outputs/current/top_view_initial_position_cases_m12.png`
- `docs/TOP_VIEW_INITIAL_POSITION_PROJECTION.md`

Current output policy:

- Keep only the three per-size PNG case figures in `outputs/current`.
- Do not keep the older combined overview or SVG variants unless explicitly
  needed for export.

Key interpretation:

- The socket center is the origin.
- Dashed circles are the configured visual residual sweep radii.
- Small yellow hexagons are bit-center capture regions, not physical hole size.
- Normal trials should keep the bit footprint at least partially supported by
  the screw head top.
- Once the bit footprint is outside screw-head support, the trial is an
  unsafe / abort / re-detect case rather than a nominal engagement sample.

Generated numeric bounds:

- M8: full-support center bound `3.04 mm`; no-overlap bound `9.96 mm`.
- M10: full-support center bound `3.63 mm`; no-overlap bound `12.87 mm`.
- M12: full-support center bound `3.48 mm`; no-overlap bound `15.02 mm`.

Validation:

```text
python3 -m py_compile src/plot_top_view_initial_positions.py
python3 -m src.plot_top_view_initial_positions
```

---

## 2026-07-09 23:42 +08 - Group-meeting PPT synced with three-size model scope

Goal:
- Reflect the new M8/M10/M12 model-ready state in the current Chinese Marp slides without overclaiming scale generalization.

Changed:
- Updated `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`.
- Added a MuJoCo model table slide.
- Added an experiment-order slide separating E0 geometry sanity from M8-first evidence and later M10/M12 scale variation.

Result:
- Slides now say: models support M8/M10/M12; first experimental claim remains M8-first.

Claim update:
- Implemented: communication artifact updated.
- Partial: rendered Marp output not checked in this environment.
- Not proved: any M10/M12 runtime result.

Next:
- Add visual-error offset sampler and source-style baseline.

## 2026-07-09 23:45 +08 - per-size top-view projections inserted into PPT

Goal:

- Extend the accepted M8 top-view projection to M10 and M12.
- Keep only three per-size projection figures and insert them into the current
  Marp deck.

Changed:

- `src/plot_top_view_initial_positions.py` now emits only:
  - `outputs/current/top_view_initial_position_cases_m8.png`
  - `outputs/current/top_view_initial_position_cases_m10.png`
  - `outputs/current/top_view_initial_position_cases_m12.png`
- Removed the older combined top-view overview and SVG variants from
  `outputs/current`.
- Added a PPT slide titled `俯视投影：三尺寸初始落点如何解释` to
  `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`.

Validation:

```text
python3 -m py_compile src/plot_top_view_initial_positions.py
python3 -m src.plot_top_view_initial_positions
```

Static PPT check:

```text
slides=11, div_open=15, div_close=15, image_refs=3
```

## 2026-07-09 23:57 +08 - top-view projection simplified to practical macro states

Reason:

- The earlier figure showed too many fine contact labels in one top-view image.
- For practical insertion, the top-view plot should explain sampling and macro
  support regions, not over-specify contact-state labels before F/T data exist.

Changed:

- Simplified each per-size figure to three macro interpretations:
  `capture / inserted`, `supported search`, and `unsafe / re-detect`.
- Drew directional initial samples as dots on each visual residual radius.
- Kept only one translucent representative bit footprint per radius to reduce
  clutter.
- Updated the PPT top-view slide note to say the figure uses three macro
  interpretations rather than fine state labels.

Validation:

```text
python3 -m py_compile src/plot_top_view_initial_positions.py
python3 -m src.plot_top_view_initial_positions
```

## 2026-07-10 00:03 +08 - M8 outside-to-inside physical state figure added

Goal:

- Make one clean M8-only figure showing practical physical states from outside
  the screw head toward the socket center.
- Avoid over-detailed contact labels in the top-view explanation.

Added:

- `outputs/current/top_view_m8_physical_states_outside_to_inside.png`

State sequence:

1. `re-detect`: bit footprint no longer overlaps screw head.
2. `partial support`: bit is partly on the screw head and partly outside.
3. `supported search`: bit is fully supported on the screw-head top.
4. `rim / chamfer`: bit is near the socket edge / chamfer cue and should use
   slow verify or small probe.
5. `capture / inserted`: bit is centered enough to drop or insert.

Validation:

```text
python3 -m py_compile src/plot_top_view_initial_positions.py
python3 -m src.plot_top_view_initial_positions
```

## 2026-07-10 00:14 +08 - M8 MuJoCo smoke renders for three physical states

Goal:

- Render three 3D MuJoCo smoke images for the practical M8 contact states:
  `partial support`, `supported search`, and `rim / chamfer`.
- Provide a visual counterpart to the top-view state schematic.

Added:

- `src/render_m8_smoke_states.py`
- `outputs/current/m8_smoke_partial_support.png`
- `outputs/current/m8_smoke_supported_search.png`
- `outputs/current/m8_smoke_rim_chamfer.png`
- `outputs/current/m8_smoke_state_renders.json`

Interpretation:

- These are smoke visualizations from the existing M8 MuJoCo XML.
- The bit pose is set directly for each state; this is not a rollout result.
- Colored spheres / highlighted regions are visualization aids for the contact
  area, not additional source-model physics geoms.

Validation:

```text
python3 -m py_compile src/render_m8_smoke_states.py
python3 -m src.render_m8_smoke_states
```

Update:

- Removed the earlier colored sphere markers because they looked like physical
  objects.
- Removed the later 3D red cross markers too; they still looked like protruding
  scene objects.
- Removed red rectangles as the main contact representation; they did not match
  the intended old-video contact cue.
- Contact areas are now shown with screen-space orange/red heat points, a small
  projected contact patch, and 2D force-direction arrows, following the old
  `source_canonical_spiral_to_contact_debug.mp4` diagnostic style.
- Hidden MuJoCo sites and the render-only visibility of socket-wall / chamfer
  collision approximations in these smoke images so they do not look like extra
  physical parts.
- These overlays and visibility choices are image diagnostics only; the source
  XML and physical MuJoCo geometry are unchanged.

## 2026-07-10 00:31 +08 - Insert M8 physical-state figures into group-meeting PPT

Goal:

- Put the current M8 top-view physical-state figure and the three cleaned smoke
  renders into the Chinese group-meeting slide deck at the point where initial
  positions and contact-state logic are discussed.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Inserted figures:

- `outputs/current/top_view_m8_physical_states_outside_to_inside.png`
- `outputs/current/m8_smoke_partial_support.png`
- `outputs/current/m8_smoke_supported_search.png`
- `outputs/current/m8_smoke_rim_chamfer.png`

Interpretation:

- The top-view figure explains the practical outside-to-inside M8 macro states.
- The smoke figures are diagnostic renderings only; orange/red contact points,
  patches, and arrows are screen-space overlays, not extra MuJoCo geometry.

## 2026-07-10 00:55 +08 - M8 conical-probe active-sensing design

Goal:

- Record the current hypothesis that a small conical / nutation probe can turn
  ambiguous vertical touchdown into a phase-structured F/T fingerprint.
- Keep the claim boundary conservative: the probe may give a directional prior
  and confidence for guarded spiral, but it does not recover the true center by
  itself.

Added:

- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`

Updated:

- `README.md`
- `CLAIM_BOUNDARY.md`
- `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`
- `DESIGN_DECISIONS.md`

Next experiment implied:

```text
P0 fingerprint smoke:
M8 partial support / supported search / rim-chamfer / captured / unsafe
-> low-force conical probe
-> phase plots for Fz, torque, z_drop
-> predicted inward direction and safety flag
```

## 2026-07-10 01:05 +08 - Update PPT with conical-probe trigger logic

Goal:

- Update the Chinese group-meeting PPT to reflect the new conical-probe-first
  active-sensing viewpoint.
- Make the logic easy to explain: expected F/T fingerprints, trigger decision,
  efficient sampling, and downstream action.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Added / changed in the deck:

- Added a slide explaining small conical / nutation probing as a contact
  fingerprint rather than a true-center oracle.
- Added a table for M8 cases 2/3/4/5/unsafe with expected F/T fingerprint,
  trigger decision, and next action.
- Added a sampling-efficiency slide: sample boundary states, use phase sweeps,
  actively add uncertain samples, compare against ordinary spiral and oracle
  direction.
- Updated the pipeline to:

```text
visual prior
-> vertical touchdown
-> conical/nutation fingerprint
-> state + inward direction + confidence + unsafe flag
-> biased guarded spiral / verify / re-probe / retreat
```

- Updated the model/control slides so old `split_v4` probe metrics are clearly
  old-path reference evidence, not new conical-trigger results.

## 2026-07-10 01:13 +08 - Clarify polar-phase learning objective in PPT

Goal:

- Record the user's refined intuition that the conical probe should use the
  polar / phase expression from the contact-pose paper and may use a CNN to
  learn shape-like F/T fingerprints.
- State the two actual learning goals:
  1. infer the likely inward direction or side of the hole;
  2. accelerate guarded spiral toward chamfer / inner-wall capture.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Deck changes:

- Added a slide: "学出来的东西：方向先验 + 捕获加速，不是中心坐标".
- Added evaluation metrics for direction prediction and capture acceleration:
  angular error, top-1/top-2 direction accuracy, confidence calibration,
  time-to-chamfer, time-to-wall, contact work, path length, and unsafe count.
- Updated the model slide to present:

```text
physics feature trigger -> interpretable baseline
polar phase CNN -> main candidate after P0 fingerprints are validated
confidence gate -> trust / re-probe / retreat
```

Claim boundary remains:

```text
The model learns a directional prior and confidence, not the true center.
```

## 2026-07-10 01:22 +08 - Tighten PPT around the final conical-fingerprint mainline

Goal:

- Align the PPT with the user's final formulation:
  conical polar-phase fingerprint -> direction prior and confidence -> biased
  guarded spiral -> faster chamfer / inner-wall capture.
- Make the logic easy to explain and explicitly state why this direction has
  better taste than static F/T classification.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Changes:

- Retitled the deck around contact fingerprints accelerating Allen engagement.
- Updated the first slide so the learning target is `macro_state`,
  `d_inward`, `confidence`, and `unsafe_score`, not generic contact labels or
  center coordinates.
- Rewrote the experiment order:

```text
E1 ordinary spiral baseline
E2 conical fingerprint smoke
E3 direction trigger
E4 biased spiral closed loop
E5 M10/M12 scale variation only after M8 works
```

- Added a "为什么这个方向有 taste" slide.
- Replaced old press/lateral label wording with conical macro-state and
  direction-prior wording.
- Updated the next experiment table so `P2` is feature baseline / polar phase
  CNN with confidence, not a generic small MLP.

## 2026-07-10 01:45 +08 - Align M8 double-overhang state across figures, PPT, and smoke renders

Goal:

- Incorporate the user's missing physical case between outer partial support
  and supported search:
  the bit footprint has an outer unsupported portion and an inner unsupported
  portion, while the middle annular band remains supported by the screw-head top.
- Keep the logic clear: this is not extra label complexity for its own sake; it
  is the state that explains why an outer edge can be confused with an inner
  socket opening.

Updated:

- `src/plot_top_view_initial_positions.py`
- `src/render_m8_smoke_states.py`
- `docs/TOP_VIEW_INITIAL_POSITION_PROJECTION.md`
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Generated:

- `outputs/current/top_view_m8_physical_states_outside_to_inside.png`
- `outputs/current/m8_smoke_partial_support.png`
- `outputs/current/m8_smoke_double_overhang.png`
- `outputs/current/m8_smoke_supported_search.png`
- `outputs/current/m8_smoke_rim_chamfer.png`
- `outputs/current/m8_smoke_state_renders.json`

Aligned state order:

```text
1. re-detect
2. outer partial support
3. double-overhang / annular bridge
4. supported search
5. rim / chamfer
6. capture / inserted
```

Control implication:

```text
double_overhang -> CAD compatibility check + weak bias or re-probe
```

The PPT state table, sampling table, trigger table, controller-routing table,
and P0 fingerprint-smoke criterion now use the same 2/3/4/5/6/unsafe logic.

## 2026-07-10 01:56 +08 - Simplify M8 smoke contact overlays to filled contact areas

Goal:

- Make the MuJoCo smoke figures cleaner.
- Replace point / arrow style diagnostic markers with a single continuous
  translucent contact-area fill.

Updated:

- `src/render_m8_smoke_states.py`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Regenerated:

- `outputs/current/m8_smoke_partial_support.png`
- `outputs/current/m8_smoke_double_overhang.png`
- `outputs/current/m8_smoke_supported_search.png`
- `outputs/current/m8_smoke_rim_chamfer.png`
- `outputs/current/m8_smoke_state_renders.json`

Boundary:

```text
The red translucent region is still a 2D diagnostic overlay projected onto the
rendered frame. It is not an added MuJoCo geometry and not a measured contact
patch.
```

## 2026-07-10 02:07 +08 - Mark both tool-side and workpiece-side contact surfaces

Goal:

- Make the M8 smoke contact visualization look more like actual surface contact
  instead of red warning labels.
- Mark both contacting bodies: tool-side and workpiece-side surfaces.

Updated:

- `src/render_m8_smoke_states.py`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Rendering change:

- The bit is slightly more transparent so the workpiece-side contact patch is
  visible through it.
- Contact overlays are now low-saturation darkened patches:
  - warm gray/olive for the workpiece-side contact surface;
  - dark blue-gray for the tool-side contact surface.
- The two projected patches are slightly separated for readability because the
  true opposing surfaces nearly overlap in the camera projection.

Boundary:

```text
These are still diagnostic 2D overlays for estimated contacting surfaces, not
MuJoCo-computed contact patches and not added marker geometry.
```

## 2026-07-10 02:18 +08 - Replace separated contact patches with one shared overlap marker

Goal:

- Correct the contact visualization logic: the tool-side and workpiece-side
  surfaces coincide at the contact interface, so the figure should mark the
  shared overlap set rather than separating the two surfaces.

Updated:

- `src/render_m8_smoke_states.py`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Rendering change:

- The smoke overlay now marks one shared contact set:
  - area contact -> filled patch plus outline;
  - line contact -> narrow band;
  - point contact -> point markers if a future case has only isolated points.
- Current M8 renders use area contact for outer partial support,
  double-overhang, and supported search; rim/chamfer is drawn as a narrow line
  contact band.

Boundary:

```text
The marked set is still a manually constructed diagnostic overlay from the
representative smoke contacts, not a MuJoCo-computed contact-patch solution.
```

## 2026-07-10 02:31 +08 - Use MuJoCo-computed contact points for smoke overlays

Goal:

- Replace the hand-constructed overlap marker with actual MuJoCo contacts.
- Match the user's intended visualization: mark all computed contact points in
  yellow; dense points naturally read as a contact area, edge-aligned points
  naturally read as a contact line.

Updated:

- `src/render_m8_smoke_states.py`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `outputs/current/m8_smoke_state_renders.json`

Implementation:

```text
set candidate bit pose
if data.contact is empty:
    lower tip_z in small 0.025 mm increments
    stop at the first pose with contacts
collect data.contact[i].pos where one geom is bit_collision
project those MuJoCo contact positions into the rendered image
draw every contact as a yellow point
```

Current generated contact counts:

```text
outer partial support: 78 contacts, tip_z = -0.025 mm
double-overhang / bridge: 79 contacts, tip_z = -0.025 mm
supported search: 111 contacts, tip_z = -0.025 mm
rim / chamfer: 51 contacts, tip_z = -0.050 mm
```

Boundary:

```text
The yellow points are MuJoCo contact.pos samples from the current collision
model. They are not hand-drawn contacts and no marker geoms are added to XML.
They still inherit the fidelity limits of the mesh/box collision model.
```

## 2026-07-10 02:43 +08 - Record boundary-biased sampling policy for conical fingerprints

Goal:

- Record the sampling logic after reading the tilt-then-rotate reference paper
  and comparing it with the M8 Allen engagement task.
- Avoid blindly copying the paper's large uniform contact-pose sampling, because
  this project still uses an inward guarded spiral after the conical
  fingerprint.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`

Decision:

```text
The conical probe is not the whole pose estimator.
It is an active-sensing trigger for:
1. macro state,
2. inward direction prior,
3. confidence,
4. unsafe flag.

Therefore sample around action-branching and ambiguity bands, not uniformly
over the entire visual-error area.
```

Sampling density:

```text
high:       double_overhang, rim/chamfer
medium-high: outer_partial
medium:     supported_search near socket
low:        captured/inserted, obvious supported plane middle
low-medium: unsafe/outside safety negatives
```

Need to include:

```text
state transition bands
small yaw mismatch
different downward force / contact depth
small clearance and friction perturbations
```

## 2026-07-10 02:58 +08 - Separate offline sampling from online conical probing policy

Goal:

- Record the refined control logic:
  data collection should expose the trigger to relevant states, but online
  conical probing should be event-triggered rather than continuous.
- Clarify whether captured/inserted still needs sampling, how to distinguish
  top-view state 1 from state 6, and how downward force should be handled.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`

Key decisions:

```text
captured/inserted:
    keep low-density offline samples as terminal examples
    do not run full conical probe online
    use gentle depth/force verify

re-detect/no-overlap:
    keep only negative examples
    no conical probe online
    re-hover or run vision re-detect
```

State 1 vs state 6:

```text
state 1 = no valid screw/socket contact before expected touchdown depth
state 6 = CAD-consistent z_tip drop + low/stable lateral force/torque +
          valid socket/chamfer/inner-wall contact
```

Online re-probe rule:

```text
initial touchdown -> full small conical fingerprint
ordinary spiral -> no continuous probe
low confidence / double_overhang / no progress -> short re-probe
rim/chamfer -> slow insertion, maybe small confirmation
captured/inserted -> verify, no conical probe
jam / force spike -> unload or retreat first
outside/no contact -> re-detect
```

Probe mechanics:

```text
tilt-then-rotate follows a cone in principle,
but M8 should start with small alpha, not blindly alpha=15 deg.
Use a small downward force/admittance bias to maintain contact,
with hard force/torque/depth stops.
```

## 2026-07-10 03:16 +08 - Set nominal conical angle policy for M8-first probing

Goal:

- Record the physical scaling argument for choosing the conical-probe tilt
  angle.
- Update the PPT so the alpha choice is tied to M8/M10/M12 clearance and
  chamfer geometry rather than copied from the reference paper.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`

Key reasoning:

```text
lateral sweep from a small conical tilt is approximately
s = L_eff * tan(alpha).

The reference paper's alpha = 15 deg is useful for making contact patterns
large and easy to classify, but it is too aggressive as a first nominal
online action for the smaller Allen-socket geometry.
```

Superseded policy kept for history:

```text
M8 nominal P0: alpha = 4 deg
M8 chamfer/capture confirmation: alpha = 2 deg
M8 offline upper ablation: alpha = 6 deg
M8 stress/unsafe comparison: alpha = 10 or 15 deg only if clearly labeled

M10/M12: start from the same 4 deg logic, then scale cautiously only after
M8 fingerprint smoke is interpretable and safe.
```

Geometry reminder:

```text
M8:  clearance 0.10 mm, socket chamfer 0.15/0.15 mm, bit tip chamfer 0.35 mm
M10: clearance 0.12 mm, socket chamfer 0.18/0.18 mm, bit tip chamfer 0.45 mm
M12: clearance 0.15 mm, socket chamfer 0.22/0.22 mm, bit tip chamfer 0.55 mm
```

## 2026-07-10 03:34 +08 - Supersede small-angle policy with bounded 15 deg fingerprint

Goal:

- Align the PPT and design notes with the user's updated judgment:
  use a single conical angle across sizes to reduce variables and improve
  real-robot F/T signal-to-noise.
- Keep the claim physically honest: 15 deg can produce stronger fingerprints,
  but it cannot guarantee zero contact-state disturbance.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`

Superseded:

```text
The previous M8 nominal alpha = 4 deg policy is no longer the main line.
```

Current policy:

```text
main initial-contact fingerprint: alpha = 15 deg for M8/M10/M12
reason: match the reference tilt-then-rotate paper, get stronger polar phase
        fingerprints, and support later real-robot fine-tuning

not guaranteed: zero slip, zero XY disturbance, or exact restoration of the
                pre-probe contact state

guaranteed only: commanded roll/pitch can return to zero
```

Safety boundary:

```text
Use virtual pivot / remote-center compensation near the tip.
Ramp into and out of the 15 deg cone slowly.
Keep low downward force / admittance bias, not hard pressing.
Force / torque / depth stops override the probe.
After returning upright, verify F/T and depth.
Do not run full 15 deg probe after chamfer/capture/inserted cues.
```

## 2026-07-10 03:03 +08 - Generate seven M8 sampling and spiral videos

Goal:

- Add a PPT logic table for seven videos:
  six sampling-state videos plus one representative learned-bias inward spiral.
- Generate the videos with the M8 MuJoCo geometry and real MuJoCo contact
  positions, following the agreed principle:
  nominal template is unified, execution may full-scan / early-stop / abort /
  verify according to physical contact cues.

Updated:

- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `src/render_m8_sampling_videos.py`

Generated:

```text
outputs/current/videos/m8_state_01_redetect_no_valid_contact.mp4
outputs/current/videos/m8_state_02_outer_partial_conical15.mp4
outputs/current/videos/m8_state_03_double_overhang_conical15.mp4
outputs/current/videos/m8_state_04_supported_search_conical15.mp4
outputs/current/videos/m8_state_05_rim_chamfer_conical15_early_stop.mp4
outputs/current/videos/m8_state_06_captured_inserted_verify.mp4
outputs/current/videos/m8_spiral_outer_partial_to_rim_chamfer.mp4
outputs/current/m8_sampling_video_manifest.json
outputs/current/m8_sampling_video_contact_sheet.png
```

Manifest summary:

```text
V1 no valid contact: abort -> re-detect, max contacts 0
V2 outer partial: full scan + return upright verify, max contacts 74
V3 double-overhang: full scan + return upright verify, max contacts 76
V4 supported search: full scan + return upright verify, max contacts 110
V5 rim/chamfer: early stop at phase 0.24, max contacts 32
V6 captured/inserted: terminal verify, max contacts 0
V7 learned-bias spiral: rim/chamfer cue reached, max contacts 140
```

Claim boundary:

```text
Videos are quasi-static commanded MuJoCo trajectories.
Yellow dots are real MuJoCo data.contact positions for the current pose.
The script does not claim that a real robot admittance controller has already
been implemented.
V7 is a representative successful trajectory, not a success-rate proof.
```

## 2026-07-10 03:08 +08 - Clean M8 sampling-video visual layer

Goal:

- Remove the ugly protruding/faceted pieces visible around the M8 socket head
  in the seven sampling videos.
- Preserve the actual MuJoCo collision geometry and contact computation.

Updated:

- `src/render_m8_sampling_videos.py`
- regenerated `outputs/current/videos/*.mp4`
- regenerated `outputs/current/m8_sampling_video_contact_sheet.png`
- regenerated `outputs/current/v1_sampling_video_midframe.png`
- regenerated `outputs/current/v5_sampling_video_midframe.png`
- regenerated `outputs/current/v7_sampling_video_midframe.png`

Change:

```text
head_top_sector_*, socket_chamfer_*, and socket_wall_* remain active collision
geometry, but are hidden from the render.

For display only, the script injects a clean_head_top_visual mesh into a
temporary render XML. It has contype=0 and conaffinity=0, so it does not create
contacts, collision, or mass effects.
```

Claim boundary:

```text
The yellow dots are still MuJoCo data.contact positions from the true collision
geometries. The clean top face is a presentation layer only.
```

## 2026-07-10 03:46 +08 - Align M8 sampling videos with reference-paper sampling boundary

Goal:

- Separate nominal training/fingerprint samples from guard and terminal verify
  states.
- Remove `no valid contact / re-detect` from the current video mainline because
  the reference tilt-then-rotate sampling assumes a valid post-approach contact.
- Change `rim / chamfer` from early-stop to full 15 deg conical scan.
- Add `inner-wall capture` as a transition/verify state distinct from
  `rim / chamfer` and `captured / inserted`.
- Make the videos easier to explain by adding the peg axis, F/T marker,
  conical sweep/funnel cue, and polar `Xz/Mx/My` channel overlays.

Updated:

- `src/render_m8_sampling_videos.py`
- `src/render_m8_smoke_states.py`
- `src/plot_top_view_initial_positions.py`
- `/home/njy/njy_ws/group_meeting/group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`
- `docs/TOP_VIEW_INITIAL_POSITION_PROJECTION.md`
- `CLAIM_BOUNDARY.md`
- regenerated `outputs/current/m8_smoke_inner_wall_capture.png`
- regenerated `outputs/current/m8_smoke_state_renders.json`

Current video set:

```text
outputs/current/videos/m8_state_01_outer_partial_conical15.mp4
outputs/current/videos/m8_state_02_double_overhang_conical15.mp4
outputs/current/videos/m8_state_03_supported_search_conical15.mp4
outputs/current/videos/m8_state_04_rim_chamfer_conical15_fullscan.mp4
outputs/current/videos/m8_state_05_inner_wall_capture_verify.mp4
outputs/current/videos/m8_state_06_captured_inserted_verify.mp4
outputs/current/videos/m8_spiral_outer_partial_to_rim_chamfer.mp4
```

Removed from `outputs/current/videos/` as obsolete:

```text
m8_state_01_redetect_no_valid_contact.mp4
m8_state_02_outer_partial_conical15.mp4
m8_state_03_double_overhang_conical15.mp4
m8_state_04_supported_search_conical15.mp4
m8_state_05_rim_chamfer_conical15_early_stop.mp4
```

Claim boundary:

```text
The videos still use quasi-static commanded MuJoCo poses with a downward-force
and admittance-compliance proxy. They are explanation/smoke artifacts, not a
real implemented robot admittance controller or a learned classifier result.
```

## 2026-07-10 03:59 +08 - Correct V5/V6/V7 motion logic

Problem found:

- V7 used five disturbance loops, so it visually behaved like a blind spiral
  even though the probe+NN stage was supposed to provide direction and center
  bias.
- V5 only oscillated in Z and did not show any rotational sampling response.
- V6 stayed in a verify pose instead of finishing the already-captured
  insertion.

Changed:

- V5 now starts from the same nominal 15 deg conical phase logic, while the
  visualized wall-constrained response is clipped to about 3 deg. It rotates
  through one phase cycle, returns upright, and emits an insertion trigger.
- V6 now holds XY and roll/pitch, descends from -1.0 mm to the M8 CAD socket
  bottom at -4.025 mm, and terminates on MuJoCo `socket_bottom` contacts.
- V7 now uses one inward turn around an imperfect NN center estimate
  `(0.08, -0.06) mm`, terminates near `(0.20, -0.06) mm`, and settles on a
  rim/chamfer cue.
- Updated the slide tables, design document, claim boundary, manifest, key
  frames, and the seven-video contact sheet.

Evidence:

```text
V5: 72 frames, 4-8 contacts, constrained rotation sampled then insert trigger
V6: 72 frames, min tip z=-4.025 mm, max contacts=4, socket-bottom completion
V7: 86 frames, one turn, rim/chamfer cue reached
```

Claim boundary:

```text
The trajectories remain quasi-static MuJoCo visualization proxies. Contact
positions and geometry tags are read from MuJoCo, but the 3 deg wall-limited
response is prescribed by the visualization logic rather than produced by a
validated torque/admittance controller. The one-turn V7 trajectory is a
representative learned-bias path, not a learned-controller result.
```

## 2026-07-10 04:14 +08 - Add six-axis F/T overlay and merge terminal states

Decision correction:

- The conical/CNN branch is now limited to the four ambiguous rim-plane search
  states: outer partial, double-overhang, supported search, and rim/chamfer.
- Inner-wall capture and inserted/bottom are routed by a higher-priority
  white-box gate using depth, sustained lateral wrench, and contact consistency.
- The specific thresholds are calibration parameters, not values copied from
  the reference paper.

Visualization changes:

- Reused the old source-canonical wrench-display contract:
  `Fx/Fy/Fz` are straight component arrows at the virtual F/T origin;
  `tau_x/tau_y/tau_z` are right-hand-rule arcs.
- Wrench values are aggregated from `mj_contactForce` for the current MuJoCo
  contacts. Near-zero components are hidden for readability.
- V1-V4 now show the current peg axis and a visible nominal 15 deg swept cone
  inside the camera frame.
- Removed the inner-wall constrained-conical video and the separate inserted
  video. They are replaced by one direct terminal video:

```text
outputs/current/videos/m8_state_05_inner_wall_to_inserted_bottom.mp4
```

- The combined video shows:

```text
inner-wall depth+wrench gate
-> no conical probe / no CNN
-> wall-guided compliant XY settling with guarded Z descent
-> socket-bottom contact at z=-4.025 mm
```

- The current video set is reduced from seven files to six; obsolete V5/V6
  files are automatically pruned by the renderer.

Claim boundary:

```text
Contact points and the displayed wrench come from MuJoCo. The trajectory is
still a quasi-static commanded visualization proxy, not a validated admittance
controller. The inner-wall and bottom gate thresholds still require simulation
sweeps and real-robot calibration.
```

## 2026-07-10 04:44 +08 - Add three confidence-modulated spiral routes

Learning/control contract:

- The conical learner is now documented as producing `search_state`, a circular
  direction distribution `q(phi)`, a finite-coverage angular set, calibrated
  confidence `kappa`, and `unsafe_score`.
- CAD and state compatibility convert the angular set into a feasible center
  sector. One conical fingerprint is not claimed to recover a unique center.
- Confidence schedules the Archimedean spiral's turn count, radial pitch, and
  nominal speed. A wrench speed gate and hard abort have higher priority.

Generated state-conditioned M8 routes:

```text
V6a outer partial:     kappa=0.63, +/-24 deg, 1.25 turns, speed scale 0.75
V6b double-overhang:   kappa=0.42, +/-46 deg, 1.75 turns, speed scale 0.50
V6c supported search:  kappa=0.86, +/-13 deg, 0.75 turns, speed scale 1.00
```

Files:

```text
outputs/current/videos/m8_spiral_outer_partial_to_rim_chamfer.mp4
outputs/current/videos/m8_spiral_double_overhang_to_rim_chamfer.mp4
outputs/current/videos/m8_spiral_supported_search_to_rim_chamfer.mp4
outputs/current/m8_sampling_video_contact_sheet.png
outputs/current/m8_sampling_video_manifest.json
```

Controller / termination values recorded in the manifest:

```text
down-force proxy: 3 N
soft speed gate: 8 N force / 20 mN m tilt torque
hard abort: 20 N force / 60 mN m tilt torque
chamfer cue: depth drop >= 0.12 mm for 3 frames
inner-wall gate: z <= -0.30 mm and (Ft >= 0.50 N or tau_xy >= 2 mN m)
bottom complete: within 0.10 mm of CAD depth and |Fz| >= 2 N
```

Validation:

```text
V6a: 93 frames, chamfer_depth_cue_latched
V6b: 126 frames, chamfer_depth_cue_latched
V6c: 73 frames, chamfer_depth_cue_latched
8 manifest entries == 8 MP4 files; all files decoded frame-by-frame at 960x720
```

Claim boundary:

```text
The three paths assume learned belief outputs; no classifier has been trained in
the new path. They are quasi-static MuJoCo trajectory visualizations, not proof
of a validated admittance controller, optimal confidence schedule, or real
closed-loop success. Thresholds are M8 simulation initial values only.
```

## 2026-07-10 04:54 +08 - Link all eight videos from logic-aligned PPT key frames

Added a reproducible key-frame extraction stage:

```text
src/extract_m8_video_keyframes.py
outputs/current/video_keyframes/index.json
outputs/current/video_keyframes/*.png
outputs/current/m8_sampling_video_contact_sheet.png
```

Representative frame policy:

- V1-V4 use the middle of the full 15 deg conical scan so the swept axis,
  MuJoCo contacts, six-axis wrench, and polar fingerprint are visible together.
- V5 uses the early inner-wall white-box gate frame, before bottom completion.
- V6a-V6c use the middle/late search segment so the confidence-conditioned
  Archimedean path is visible before chamfer termination.

PPT synchronization:

- V1-V4 are grouped after the F/T fingerprint/trigger logic to show that one
  fixed probe action produces state-dependent responses.
- V6a-V6c are grouped after the confidence schedule to show how uncertainty
  changes turns, pitch, and speed.
- V5 is placed at the offline/online branch boundary to show why inner-wall and
  inserted states bypass the CNN and use depth+wrench rules.
- Every key frame links directly to its MP4, and the eight-row summary table
  provides a second direct link to every video.

Validation:

```text
8 key-frame PNG files generated from 8 manifest videos
18 PPT image references, 8 clickable-image MP4 links, 8 table MP4 links
all 34 local PPT media/link targets exist
display-math fences balanced
```

Claim boundary:

```text
The key frames and videos are logic/evidence illustrations. They preserve the
existing boundary: conical and spiral trajectories are quasi-static MuJoCo
visualization proxies, not trained-controller or real-robot validation results.
```

## 2026-07-10 05:03 +08 - Compare confidence-modulated search with task-source spiral

Source-paper correction:

- Section 4.4 of `references/task_source/Unscrewing.pdf` uses two
  counter-phase Cartesian sinusoids with increasing amplitude.
- The source search is an outward spiral around the visual center, with about
  5 N axial contact force, a 10 mm search-radius abort, and a 0.1 N m
  motor-torque coupling check.
- It is not an inward spiral toward a known true center.

Mathematical boundary:

```text
inward vs outward, with the same center/pitch/radius interval
= the same Archimedean curve traversed in opposite directions
```

The proposed advantage must therefore come from:

- a calibrated direction belief and CAD-constrained feasible center sector;
- fewer turns / smaller effective search region;
- confidence-conditioned pitch and speed;
- earlier chamfer / inner-wall stage transitions.

Fair speed criterion:

```text
T_ours = T_probe + T_biased
the method is faster only if T_source - T_biased > T_probe
```

Fallback:

```text
q(phi) approximately uniform
and kappa approaches zero
and the angular set approaches the full circle
-> remove directional bias
-> center on the visual estimate
-> restore fixed pitch/speed
-> run the source-style isotropic outward spiral
```

PPT update:

- added one slide comparing the source outward spiral and the current inward
  belief-centered trajectory;
- added one slide stating the conditional timing inequality, path-length
  scaling, state-dependent advantages, and exact ordinary-spiral fallback;
- updated E4/P1 wording so the comparison includes conical-probe time, success,
  contact work, and peak wrench.

Claim boundary:

```text
The confidence-modulated method is hypothesized to reduce expected search cost
when the belief is informative. It is not yet demonstrated to be uniformly or
empirically faster than the task-source spiral.
```

## 2026-07-10 - Implement real admittance core and replace inward search policy

Physical decision:

- keep the task start at guarded vertical touchdown with valid support;
- check direct inner-wall / inserted depth+wrench cues before probing;
- run one 15 deg conical fingerprint only for ambiguous rim contact;
- combine measured visual covariance, conical direction likelihood, and CAD
  feasibility into a posterior over possible socket centers;
- spend only a conservative lower-quantile distance as posterior-guided
  transport;
- run a local outward Archimedean spiral around the transported point;
- keep turn spacing bounded by `2 * r_eff` regardless of confidence;
- let chamfer cue stop XY search, then perform upright compliant insertion.

Implementation added:

- `src/cartesian_admittance.py`: 6D task-frame mass-damping-stiffness
  admittance, wrench low-pass filter, semi-implicit integration, per-axis masks,
  acceleration/velocity/pose limits, anti-windup, and latched hard stop;
- `configs/m8_admittance_initial.json`: explicit simulation-initial M8
  parameters and frame/sign convention;
- `src/posterior_search_plan.py`: conservative posterior transport and local
  outward-spiral radius/pitch planner;
- `src/run_m8_admittance_smoke.py`: dynamic MuJoCo free-body inner PD plus
  measured-contact-wrench admittance outer loop;
- `tests/test_cartesian_admittance.py` and
  `tests/test_posterior_search_plan.py`;
- `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md` with physics, formulas,
  verification boundary, and validation ladder.

Verification:

```text
python3 -m unittest discover -s tests -v
Ran 10 tests: OK
```

Dynamic supported-contact smoke:

```text
target Fz: 3.0 N
final 0.25 s mean/std: 2.9998 N / 0.00028 N
hard stop: false
evidence: outputs/current/m8_admittance_flat_contact_smoke.json
```

PPT alignment:

- replaced the final method story with posterior-guided short transport plus
  local outward search;
- removed confidence-dependent pitch beyond the geometric coverage bound;
- marked the three existing inward-spiral videos as historical concept
  visualizations, not final-control evidence;
- recorded that the admittance core has one dynamic supported-contact smoke,
  but is not yet integrated into conical/chamfer/wall video rollouts or
  calibrated on hardware;
- separated geometric insertion verification using F/T + CAD from the stronger
  claim of rotational bit/socket coupling, which still needs torque evidence.

Claim boundary:

```text
The controller and posterior planner cores are implemented and unit-tested. A
single supported-contact MuJoCo force-tracking smoke has passed; conical,
chamfer, wall, spiral, and real-robot validation remain incomplete, and current
parameters are simulation-initial values.
```

## 2026-07-10 - Integrate admittance into all current M8 video rollouts

Implementation:

- added `src/mujoco_admittance_rollout.py`:
  - 500 Hz Cartesian admittance outer loop;
  - environment-on-tool wrench from `mj_contactForce`;
  - wrench-applied Cartesian PD inner loop using `mj_applyFT`;
  - 5 kg / `[0.02, 0.02, 0.01] kg m^2` effective robot-end operational
    inertia instead of treating the 0.5 kg bit as a free projectile;
  - interpolated nominal pose commands, state-divergence detection, and the
    existing latched wrench stop;
- added `src/render_m8_dynamic_videos.py` and regenerated all eight MP4s;
- added `tests/test_dynamic_video_manifest.py` so stale quasi-static videos or
  failed/hard-stopped records cannot silently become the current set.

Dynamic rollout policy:

```text
V1-V3: settle under 3 N -> ramp to 15 deg -> dynamic 360 deg scan -> return upright
V4: rim/chamfer start -> same full nominal command -> physical transition is retained
V5: centered in-hole start -> slow wall approach -> 0.8 N compliant insertion -> CAD bottom verify
V6a-c: assumed posterior transport -> local outward spiral -> freeze XY on depth candidate -> dwell/latch
```

Search parameters:

```text
pitch = 0.40 mm/rev for V6a/V6b/V6c
V6a transport = 4.10 mm, Rmax = 1.10 mm
V6b transport = 3.57 mm, Rmax = 1.30 mm
V6c transport = 2.00 mm, Rmax = 0.70 mm
chamfer cue = depth drop >= 0.07 mm, then 3-frame stationary dwell
```

Terminal parameters:

```text
search/probe target down-force = 3.0 N
terminal insertion target down-force = 0.8 N
bottom verify = |z-z_CAD| <= 0.15 mm and |Fz| >= 0.70 N
```

Current outcomes from `outputs/current/m8_sampling_video_manifest.json`:

- V1/V2/V3 finish the dynamic full scan and return near upright;
- V4 starts at rim/chamfer but naturally transitions to approximately
  `z=-2.7 mm` during the commanded scan; therefore rim/chamfer is not assumed
  to be a state-invariant full-scan class;
- V5 reaches `cad_bottom_verified`;
- V6a/V6b/V6c all latch `chamfer_depth_cue_latched`;
- all eight records have `dynamic_physics=true`,
  `admittance_controlled=true`, and `hard_stop=false`.

Generated/updated evidence:

- `outputs/current/videos/*.mp4` (exactly eight files);
- `outputs/current/m8_sampling_video_manifest.json`;
- `outputs/current/video_keyframes/*.png` and `index.json`;
- `outputs/current/m8_sampling_video_contact_sheet.png`;
- PPT video tables, captions, controller thresholds, and V4 transition logic.

Verification:

```text
python3 -m py_compile src/*.py tests/*.py
python3 -m unittest discover -s tests -v
Ran 11 tests: OK

imageio decoded first and last frame of all eight MP4s
all videos: 960 x 720, 15 fps, 37-84 frames
```

Claim boundary:

```text
The eight current videos are dynamic M8 MuJoCo rollouts under one implemented
admittance control stack. V6 posterior inputs are still assumed, not learned;
the simulation-initial controller/contact parameters are not hardware
calibration or a sim-to-real result.
```

## 2026-07-10 07:13 +08 - Re-audit video physics against both core papers

Paper boundary:

- `reference/锥形探索.pdf` defines a fixed nominal roll/pitch cone and
  constant down-force, but does not provide a numerical XY drift guarantee or
  exact physical return guarantee;
- that paper explicitly assumes flat peg/hole contact surfaces, so a captured
  chamfer state is not evidence for another full CNN fingerprint;
- `references/task_source/Unscrewing.pdf` uses an upright planar outward
  spiral under 5 N axial force. The current M8 simulation keeps the same
  upright planar structure but uses a conservative simulation-initial 3 N.

Controller corrections:

- changed the wrench-driven Cartesian inner loop from zero-velocity damping
  to setpoint-velocity PD tracking;
- raised the robot-flange translational/rotational tracking gains while
  retaining compliance in the admittance outer loop;
- V1/V2/V3 now keep nominal tip-center XY fixed, with maximum actual XY
  deflections `0.202/0.191/0.225 mm`, final residuals below `0.060 mm`, and
  maximum orientation tracking error below `1.11 deg`;
- capture guards are evaluated at the 2 ms control rate, not at the 15 Hz
  rendering rate.
- contact wrench aggregation and displayed arrows now use the same explicit
  F/T sensor reference point; the simulation-initial inner-wall gate is
  `z <= -0.30 mm` and (`Ft >= 0.25 N` or `tau_xy >= 1.5 mNm`).

State-machine corrections:

```text
V1-V3: flat-support full 15 deg conical fingerprint
V4: upright 0.15 -> 0.8 N chamfer guidance -> inner-wall gate; no cone
V5: measured inner-wall depth+wrench gate -> compliant descent -> CAD bottom
V6a-c: upright posterior transport -> outward spiral -> 50 ms capture dwell
```

Search corrections:

- removed the hidden use of the true hole center when choosing spiral phase;
- the first radial direction now comes only from the assumed posterior
  transport direction;
- the plan view shows the full planned outward spiral separately from the
  executed path, which may terminate early on a physical capture event;
- current assumed-posterior examples use:
  - V6a: `3.30 mm` transport, `2.00 mm` max radius;
  - V6b: `2.77 mm` transport, `2.20 mm` max radius;
  - V6c: `1.40 mm` transport, `1.40 mm` max radius;
- all retain `0.40 mm/rev` pitch and remain assumed-posterior simulations, not
  learned or hardware-validated policies.

Current outcomes:

- V4 latches `inner_wall_capture_cue_latched` while upright;
- V5 first satisfies `wall_gate_triggered`, then reaches
  `cad_bottom_verified`;
- V6a/V6b/V6c latch `chamfer_depth_cue_latched` with minimum tip depths above
  `-0.60 mm`, avoiding the earlier render-rate delayed plunge;
- no rollout triggers a hard stop.

## 2026-07-10 07:24 +08 - Final PPT video/keyframe alignment

Updated
`group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
against the current eight-video manifest and regenerated keyframes:

- V1--V3: full flat-support conical fingerprints with fixed nominal tip-center
  XY and measured compliance deflection;
- V4: upright `0.15 -> 0.8 N` chamfer capture to inner-wall gate; removed the
  obsolete full-cone interpretation;
- V5: measured inner-wall depth+wrench gate before compliant CAD-bottom
  insertion;
- V6a--V6c: upright posterior transport plus local outward spiral, with the
  planned path and actual early-stopped path shown separately;
- replaced render-frame dwell language with a 2 ms controller-rate guard and
  50 ms persistent capture window;
- synchronized the current `0.25 N / 1.5 mNm` inner-wall thresholds and all
  transport/radius values.

Verification:

```text
all PPT href/src targets exist
no stale V4 full-scan, old transport, or three-render-frame dwell text remains
outputs/current/videos contains exactly the eight manifest MP4s
```

## 2026-07-10 07:57 +08 - Consolidate the PPT around the baseline-plus-learning logic

Reworked
`group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`
and synchronized the design notes around one non-contradictory control story:

```text
guarded touchdown
-> direct depth/wrench gate
-> conical fingerprint only for ambiguous flat-support states
-> posterior-guided finite transport
-> local outward Archimedean spiral under admittance/down-force control
-> persistent chamfer capture cue ends lateral search
-> upright wall-guided descent
-> CAD depth plus F/T terminal verification
```

Key corrections:

- the source unscrewing method is the robust fallback: upright outward spiral,
  axial force control, physical socket/chamfer capture, then insertion;
- learning is an efficiency layer, not a replacement for geometric coverage;
  low confidence sends the finite transport toward zero and restores the
  ordinary outward-search envelope;
- chamfer is a mechanical funnel and a mode-switch cue, not a task-completion
  trigger; persistent capture ends spiral expansion and starts upright
  compliant descent;
- inner-wall and CAD-bottom cues remain white-box insertion/termination gates;
- spiral pitch keeps the coverage condition
  $\Delta r_{\mathrm{turn}}\le 2r_{\mathrm{eff}}$;
- added the contact-work proxy
  $J_{\mathrm{wear}}=\int |F_xv_x+F_yv_y|\,dt$ so the claimed benefit is not
  limited to time-to-capture;
- corrected unsupported readings of the source paper: its `D/4` statement is
  a minimum search-range requirement, and it does not specify a fixed lift
  distance or retry count after the 10 mm termination condition.

Cleanup and alignment:

- reduced the deck from 31 to 26 slides by removing five duplicate diary-style
  sampling/trigger/video pages;
- removed stale old-project F1/coverage numbers from the new-path evidence
  boundary;
- aligned `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`,
  `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`, and
  `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md` with the same chamfer and
  fallback semantics;
- retained all eight current video links and eight representative keyframes.

Source-level verification:

```text
26 slide headings
26 math delimiters (balanced)
53 opening / 53 closing div tags
2 opening / 2 closing pre tags
8 unique MP4 links and 8 unique keyframes
0 missing href/src targets
0 consecutive slide separators
```

## 2026-07-10 07:59 +08 - Remove the remaining chamfer/CNN scope contradiction

Corrected `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md` so that the learning
dataset contains only the three ambiguous flat-support states:
`outer_partial`, `double_overhang`, and `supported_search`.

`rim_chamfer` is now consistently treated as a white-box transition candidate
identified from a short guarded force-ramp/descent trace. It ends lateral
search and enters upright compliant descent; it is neither a full conical CNN
class nor a task-completion condition. Inner-wall and CAD-bottom conditions
remain white-box insertion and terminal gates.

## 2026-07-10 08:06 +08 - Align touchdown, chamfer, and terminal closure semantics

Updated the PPT, design documents, claim boundary, video generator metadata,
and current video manifest around the following state-machine contract:

```text
guarded touchdown
-> short guarded force-ramp
-> persistent chamfer / inner-wall / inserted candidate?
-> if still ambiguous flat support: one 15 deg conical fingerprint
-> learned direction posterior + bounded transport
-> local outward spiral under down-force
-> persistent chamfer candidate stops XY expansion
-> upright compliant wall-guided descent
-> geometric insertion verification
```

Clarifications now recorded consistently:

- a single static F/T sample cannot uniquely identify chamfer contact;
- `outer_partial`, `double_overhang`, and `supported_search` are the only CNN
  fingerprint classes;
- `rim_chamfer`, `inner_wall`, `inserted`, and `bottom` use short white-box
  force-ramp/descent calibration trajectories rather than conical training;
- once a persistent chamfer candidate is present, outward spiral expansion
  stops so the commanded XY path does not fight the chamfer's inward funnel;
- V4 starts from a known rim/chamfer state and demonstrates the mechanical
  mode switch, not online F/T-only state recognition;
- V4/V6 currently use privileged MuJoCo chamfer/wall contact tags in their
  capture guards; hardware must replace these with persistent sensor-only
  trends in `z`, `Fz`, `Fxy`, and `tau_xy`;
- the eight videos are pre-labeled dynamic segments under one admittance core,
  not a unified automatically routed learning/control episode;
- current V5 bottom completion uses CAD depth plus axial force. The intended
  hardware terminal gate additionally requires low lateral wrench and dwell;
- geometric insertion completion does not prove rotational bit/socket
  coupling or unscrewing readiness.

Files synchronized:

- `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`;
- `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`;
- `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`;
- `CLAIM_BOUNDARY.md`;
- `src/render_m8_dynamic_videos.py`;
- `outputs/current/m8_sampling_video_manifest.json`.

Verification:

```text
26 slide headings; math and HTML delimiters balanced
8 unique MP4 links and 8 unique keyframes; 0 missing targets
video manifest JSON parses successfully
render_m8_dynamic_videos.py compiles successfully
python3 -m unittest discover -s tests -v: 11/11 passed
```

## 2026-07-10 08:14 +08 - Remove chamfer classification from the online task

Simplified the PPT and project contract to reduce real-robot implementation
work. `rim/chamfer` remains a physical/offline label that explains the contact
mechanism and existing videos, but it is no longer an online state, CNN class,
or claimed contribution.

Final online contract:

```text
guarded touchdown + short force-ramp
-> safe downward progress already present?
   yes: freeze XY and start upright compliant descent
   no: if flat support is ambiguous, run one 15 deg conical fingerprint
-> direction posterior + bounded transport
-> local outward spiral under down-force
-> persistent safe downward progress freezes XY
-> upright wall-guided descent
-> CAD depth + axial support + low lateral wrench + dwell
```

The progress gate is geometry-agnostic. It does not distinguish chamfer, rim
transition, or shallow insertion. This preserves the useful mechanism while
removing a separate chamfer dataset/classifier from the first paper scope.

Current evidence boundary:

- V1--V3 remain the only conical/CNN fingerprint classes;
- physical rim/chamfer cases are used only for short force-ramp/descent gate
  calibration;
- V4 demonstrates upright descent from a known physical chamfer condition;
- V6a--c demonstrate assumed-posterior search until the simulation
  safe-progress proxy;
- the current proxy still uses a privileged MuJoCo chamfer/wall contact tag;
  hardware must replace it with persistent `z` and bounded-wrench behavior;
- geometric insertion completion remains separate from rotational coupling.

Files synchronized:

- `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`;
- `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`;
- `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`;
- `CLAIM_BOUNDARY.md`;
- `src/render_m8_dynamic_videos.py`;
- `src/render_m8_sampling_videos.py`;
- `outputs/current/m8_sampling_video_manifest.json`.

Verification:

```text
26 slide headings; 26 math delimiters; 53 opening/closing div tags
8 unique MP4 links and 8 unique keyframes; 0 missing targets
manifest JSON parses; both render scripts compile
python3 -m unittest discover -s tests -v: 11/11 passed
```

## 2026-07-10 08:51 +08 - Same-start proposed vs source-style search videos

Implemented `src/render_m8_search_comparison.py` to replace invalid video
concatenation with continuous dynamic MuJoCo rollouts from the same physical
initial state. The proposed video connects the full 15 deg conical probe,
upright return, assumed-posterior transport, local outward spiral, progress
dwell, and inner-wall handoff in one simulation episode.

Fairness contract:

```text
same M8 initial bit center: (+5.10, 0.00) mm
same target down-force: 3.0 N
same outward pitch: 0.40 mm/rev
same commanded XY speed: 2.25 mm/s
same active time budget: 30 s
same endpoint: candidate dwell followed by confirmed inner-wall depth+wrench gate
proposed probe time and probe contact work are included
```

The baseline is deliberately called `source-style`, not a reproduction of
`Unscrewing.pdf`: it matches the increasing-radius outward geometry but does
not use that paper's approximately 5 N, 20 rpm tool rotation, or motor-torque
coupling verifier. The selected `x=5.10 mm` outer-partial stress case is also
outside the source paper's reported 1--2 mm visual-positioning error.

Generated videos:

- `outputs/current/comparison_videos/m8_outer_partial_proposed_probe_to_inner_wall.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_source_outward_to_inner_wall.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_proposed_vs_source_side_by_side.mp4`.

Result for this single fixed-phase stress case:

| metric | proposed | source-style baseline |
|---|---:|---:|
| inner-wall endpoint | 16.40 s | not reached within 30 s |
| probe time included | 5.80 s | 0 s |
| actual XY path | 22.74 mm | 41.91 mm |
| contact XY path | 22.74 mm | 41.32 mm |
| accumulated contact-work proxy | 43.99 mJ | 67.16 mJ at timeout |
| max force | 6.27 N | 10.78 N |
| max torque | 20.36 mN m | 21.73 mN m |
| candidate confirmations | 1 | 4 rejected candidates |

An earlier draft stopped both policies at the first depth/contact candidate.
That produced a false baseline success near `x=6.59 mm`. It was deleted and
replaced by the stricter common inner-wall endpoint. This is why the comparison
uses a full routed episode rather than concatenating V1 and V6a.

Claim boundary:

- the proposed posterior remains assumed, not learned;
- the progress and inner-wall gates still use privileged MuJoCo contact tags;
- the baseline phase is fixed; no phase or initial-condition distribution was
  swept;
- the 34.5% work-proxy difference compares proposed completion against
  baseline accumulation at timeout, not equal-endpoint physical wear;
- the result motivates a distribution-level comparison but does not prove
  general speed, wear, or success-rate superiority.

PPT and evidence updates:

- added a three-video same-start comparison slide to
  `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- updated `CLAIM_BOUNDARY.md`,
  `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`, and
  `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`;
- stored metrics and boundaries in
  `outputs/current/m8_outer_partial_search_comparison.json`;
- added `tests/test_search_comparison_artifacts.py`.

Verification:

```text
comparison videos: 296 / 493 / 523 frames at 15 fps
individual size: 960 x 720; side-by-side size: 1280 x 520
first/middle/last frame pixel standard deviations are all above 28
PPT: 27 headings, balanced math/div delimiters
PPT: 11 unique MP4 links, 11 unique keyframes, 0 missing targets
python3 -m unittest discover -s tests -v: 12/12 passed
```

## 2026-07-10 09:10 +08 - Define fingerprint observability, symmetry, and transfer contract

Aligned the PPT and project documents around the question that must be answered
before large-scale data collection: whether one M8 15 deg conical fingerprint
contains a stable socket-center direction and whether any radial-distance
information survives contact nuisance variation.

Decision:

```text
do not train a true-center regressor by default
-> run a signed-distance P0 observability pilot
-> estimate finite-difference Jacobian/Fisher information
-> audit nearest-neighbor fingerprint collisions
-> learn state + circular direction posterior if direction is identifiable
-> add a radial quantile head only if radial observability also passes
-> fuse tactile likelihood with visual covariance and CAD
-> let the posterior planner, not the CNN, choose bounded transport distance
```

Sampling contract:

- P0 starts at approximately 24 directions x 8 CAD signed-distance bands x 5
  nuisance repeats, around 960 dynamic rollouts;
- P1 starts near 5000 rollouts and grows only if held-out learning curves or
  calibration have not saturated;
- initial strata are 60% measured deployment visual-error distribution, 25%
  state boundaries, 10% measured-range nuisance randomization, and 5% active
  collision/uncertainty points;
- `rim/chamfer`, `inner-wall`, and `inserted` remain short white-box gate
  calibration trajectories, not full conical CNN classes;
- raw 12D signals, phase-resampled channels, action/controller state, visual
  covariance, CAD signed distances, nuisance metadata, and offline-only labels
  use one common simulation/hardware schema.

Symmetry and leakage contract:

- Allen $C_6$ augmentation is allowed only for $k*60$ deg rotations;
- offset/direction labels, CAD/yaw, phase, $F_{xy}$, and $tau_{xy}$ rotate
  together;
- base rollouts are split before augmentation, and transformed copies never
  cross train/validation/test boundaries;
- augmentation is not independent physical evidence; unaugmented full-angle
  validation remains mandatory because real robot/sensor/manufacturing effects
  break ideal symmetry.

Real and scale-transfer contract:

- suggested M8 hardware anchor is 6 directions x 3 radial bands x 3 repeats x
  2 down-force levels = 108 scans, followed by 30--60 active-selected cases if
  required;
- first calibrate bias, gravity/tool load, frame, delay, and dimensionless
  scales; then tune calibration or a small adapter/head with simulation replay;
- use shadow mode before enabling bounded transport;
- shared M8/M10/M12 models use dimensionless pose/wrench and CAD ratios, but
  must pass zero-shot and leave-one-size-out tests;
- every new real size still needs 36--60 independent calibration/test scans
  before any scale-transfer claim.

Files synchronized:

- `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`;
- `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`;
- `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`;
- `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`;
- `README.md`;
- `CLAIM_BOUNDARY.md`.

Current evidence boundary is unchanged: no conical training dataset, learned
posterior, Fisher/collision report, real fine-tuning, or M10/M12 transfer result
exists yet.

Verification:

```text
PPT: 29 slide headings; 30 balanced math delimiter lines
PPT: 57 opening / 57 closing div tags; 2 opening / 2 closing pre tags
PPT: 33 local href/src/Markdown references; 0 missing targets
python3 -m unittest discover -s tests -v: 12/12 passed
```

## 2026-07-10 09:22 +08 - Add live progress and complete the cheap P0a C6 audit

Implemented `src/sample_m8_conical_pilot.py` as a resumable headless sampler
instead of starting with a 5000-trial dataset.

Runtime interface:

```bash
python3 -m src.sample_m8_conical_pilot --workers 32 --resume
```

The terminal displays a live `tqdm` progress bar with completed/total trials,
throughput, ETA, and ok/fail counts. Every completed trial atomically updates
`outputs/current/p0a_m8_symmetry/progress.json`, so the same command resumes
after a disconnect without repeating finished rollouts.

Implementation details:

- 32 spawn workers on the 64-physical-core host;
- each worker loads one M8 model and reuses it across tasks;
- no MuJoCo rendering or video encoding;
- the 15 deg dynamic scan uses the existing 500 Hz Cartesian admittance loop;
- each trial stores 1733 controller-rate samples with 24 raw columns and a
  fixed 22 x 360 phase matrix;
- trial files are compressed and atomically renamed;
- an initial two-trial smoke caught and fixed a reversed phase-alignment sign;
- a synthetic equivariance unit test now protects phase and vector rotation.

P0a completed in approximately 39 s wall-clock:

| item | result |
|---|---:|
| dynamic rollouts | 72 / 72 |
| failed / hard-stop rollouts | 0 |
| base pose groups | 12 |
| transformed non-reference comparisons | 60 |
| output size | 9.7 MB |
| raw selected-channel mean / p95 NRMSE | 0.382 / 0.471 |
| normalized primary-shape correlation mean / p05 | 0.945 / 0.883 |

The primary shape audit uses circularly smoothed, per-trial standardized
`Fz`, `Tx`, `Ty`, and `tip_z`. State means are 0.956 for double-overhang,
0.941 for outer-partial, and 0.936 for supported-search.

Interpretation:

```text
nominal C6 symmetry is provisionally useful for the normalized direction/shape branch
raw lateral wrench amplitudes are less symmetric
-> do not use rotated raw traces as independent radial-distance evidence
-> do not claim hardware symmetry
-> run P0b nuisance repeats before enabling final training augmentation
```

Artifacts:

- `outputs/current/p0a_m8_symmetry/progress.json`;
- `outputs/current/p0a_m8_symmetry/manifest.json`;
- `outputs/current/p0a_m8_symmetry/symmetry_audit.json`;
- `outputs/current/p0a_m8_symmetry/summary.md`;
- `outputs/current/p0a_m8_symmetry/trials/*.npz`;
- `tests/test_m8_conical_pilot.py`;
- `tests/test_p0a_symmetry_artifacts.py`.

PPT, README, observability/data-design document, and claim boundary were
updated without changing the larger evidence claim: no nuisance/Fisher/action-
collision audit, trained posterior, real fine-tuning, or M10/M12 transfer result
exists yet.

Verification:

```text
P0a: 72 trial files; progress 72/72; failed 0
P0a artifact and equivariance tests added
PPT: 29 headings; math/div/pre delimiters balanced
python3 -m unittest discover -s tests -v: 17/17 passed
```

## 2026-07-10 09:32 +08 - Complete P0b nuisance observability with live progress

Extended `src/sample_m8_conical_pilot.py` with the `observability` stage and
kept the same live progress/checkpoint/resume interface:

```bash
python3 -m src.sample_m8_conical_pilot --stage observability --workers 32 --resume
```

Sampling design:

```text
3 flat-support states
x 5 CAD signed-distance levels
x 8 angles in one 60 deg fundamental sector
x 3 bounded Sobol nuisance repeats
= 360 dynamic conical scans
```

Nuisance dimensions were sliding friction [0.55, 0.95], contact time constant
[0.040, 0.070] s, target down-force [2.5, 3.5] N, and yaw [-1.5, +1.5] deg.
All 360 rollouts completed without hard stop in approximately 2 min 53 s.

The automatic report compares two deliberately small feature families:

| metric | 48D normalized phase shape | +16D dimensionless absolute stats |
|---|---:|---:|
| state accuracy | 0.911 | 0.936 |
| angular error median / p90 | 0 / 7.5 deg | 0 / 7.5 deg |
| radial MAE | 0.209 mm | 0.157 mm |
| action-conflict nearest neighbors | 23 / 120 | 12 / 120 |
| relative Fisher sigma_min median / p05 | 36.40 / 22.48 | 43.37 / 25.43 |

The Fisher calculation uses nuisance-whitened finite differences over 54
interior grid sites. Its values are comparative diagnostics, not a formal
physical-information unit or a global identifiability proof.

Decision:

```text
direction is sufficiently observable to proceed
remaining conflicts are concentrated at same-direction radial/state boundaries
-> first learner: search_state + q(phi) + calibrated angular set
-> radial quantile: optional auxiliary output
-> movement distance: still computed from visual prior + CAD + conservative posterior quantile
-> next sampling: independent repeats on 18 unique collision groups only
```

Artifacts:

- `outputs/current/p0b_m8_observability/progress.json`;
- `outputs/current/p0b_m8_observability/manifest.json`;
- `outputs/current/p0b_m8_observability/observability_report.json`;
- `outputs/current/p0b_m8_observability/summary.md`;
- `outputs/current/p0b_m8_observability/trials/*.npz`;
- `tests/test_p0b_observability_artifacts.py`.

Claim boundary remains simulation-only: no CNN, calibrated posterior,
sensor-only capture gate, safe-transport validation, real fine-tuning, or
M10/M12 transfer result exists yet.

## 2026-07-10 09:37 +08 - Independently re-test only the P0b collision groups

Added the `collisions` stage to the same live-progress sampler:

```bash
python3 -m src.sample_m8_conical_pilot --stage collisions --workers 32 --resume
```

The stage reads the frozen P0b report, selects the 18 unique base groups in the
12 shape+absolute action-conflict pairs, and collects five new Sobol nuisance
repeats per group. It does not expand the other 102 already-clear sites.

P0c completed 90/90 rollouts with no hard stop in approximately 49 s. The
evaluation freezes all 120 P0b feature centroids and nuisance scales before
classifying the independent P0c traces:

| metric | shape | shape + dimensionless absolute |
|---|---:|---:|
| state accuracy | 0.889 | 0.933 |
| angle median / p90 | 0 / 0 deg | 0 / 0 deg |
| direction within 7.5 deg | 0.989 | 0.989 |
| radial MAE | 0.110 mm | 0.083 mm |
| state + direction acceptable | 0.878 | 0.922 |
| full-action proxy acceptable | 0.856 | 0.911 |

Only one trial exceeds the 7.5 deg direction tolerance. The seven remaining
shape+absolute state/direction failures are mainly same-direction
`supported_search/double_overhang` or `double_overhang/outer_partial` boundary
aliases.

Decision:

```text
stop uniform P0 sampling
-> direction observability has an independent hard-case check
-> first learner is state + circular direction posterior + calibrated set
-> radial quantile remains auxiliary and cannot directly command center motion
-> proceed to grouped Fourier / polar-CNN / 1D-circular-CNN comparison
```

Artifacts:

- `outputs/current/p0c_m8_collision_followup/progress.json`;
- `outputs/current/p0c_m8_collision_followup/manifest.json`;
- `outputs/current/p0c_m8_collision_followup/collision_followup_report.json`;
- `outputs/current/p0c_m8_collision_followup/summary.md`;
- `outputs/current/p0c_m8_collision_followup/trials/*.npz`;
- `tests/test_p0c_collision_artifacts.py`.

This remains a fixed feature baseline in simulation, not a CNN result,
calibrated posterior, hardware claim, or closed-loop search guarantee.

## 2026-07-10 09:58 +08 - Add physical sector anchors and train the direction-first ensemble

The first grouped P1 run exposed an important failure instead of being accepted
at face value. P0b+C6 models performed well inside the fundamental sector but
produced near-180 deg aliases on several physically rotated P0a traces. The
P0a labels also initially omitted `rotation_deg`; that evaluation bug was fixed
and protected by a unit test. The remaining aliases showed that circular
padding plus synthetic C6 augmentation is not strict physical equivariance.

Added the paired P0d sector-anchor stage:

```bash
python3 -m src.sample_m8_conical_pilot --stage sector-anchors --workers 32 --resume
```

The design uses 3 states x 3 radial levels x 3 local angles x 2 Sobol nuisance
draws x 6 physical sectors = 324 trials. Nuisance values are paired across each
six-sector orbit so sector residual and nuisance residual are not confounded.
All 324 trials completed, forming 54 complete orbits:

| P0d metric | value |
|---|---:|
| normalized primary-shape correlation mean / p05 | 0.930 / 0.834 |
| raw selected-channel NRMSE mean / p95 | 0.412 / 0.522 |

The sampler's fixed `progress.json.tmp` name also exposed a concurrent atomic-
write race after all trial files had completed. Atomic JSON writes now use a
process-specific temporary name; `--resume` recovered all 324 files and rebuilt
the manifest/audit without re-sampling.

Implemented grouped P1 training in `src/train_m8_direction_models.py`:

- split physical base poses before augmentation;
- 1296 P0b+C6 training examples plus 180 P0d physical-sector anchors;
- 144 held-out P0b/P0d examples for early stopping, temperature scaling, APS,
  and confidence-gate selection;
- four normalized phase channels (`Fz`, `Tx`, `Ty`, `tip_z`) plus 16
  dimensionless context statistics;
- diagnostic three-state head and 48-bin circular direction head;
- no learned radial-distance command.

Three model seeds use the same frozen split. `src/evaluate_m8_direction_ensemble.py`
averages calibrated class probabilities rather than selecting the best seed:

| ensemble split | within 7.5 deg | APS coverage / mean set size |
|---|---:|---:|
| P0b grouped test | 1.000 | 0.958 / 3.21 |
| P0d physical-sector grouped test | 0.972 | 1.000 / 3.01 |
| P0c hard nuisance set | 1.000 | 1.000 / 2.63 |
| P0a nominal full-angle diagnostic | 0.958 | 0.986 / 3.51 |

No reported ensemble test has a top-1 direction error above 90 deg. However,
the validation-selected confidence threshold accepts every independent
simulation sample, so it is not evidence for OOD rejection. Single-seed spread
is material, and the ensemble is retained as the current simulation interface.

Control decision:

```text
learned state = diagnostic only
direction posterior/set = optional bounded-transport prior
distance = mechanics/CAD/visual-covariance planner, not a learned millimeter output
low confidence or broad posterior = guarded local outward spiral or re-probe
chamfer/progress/inner-wall/bottom = white-box depth+wrench gates
```

Artifacts:

- `outputs/current/p0d_m8_sector_anchors/sector_anchor_audit.json`;
- `outputs/current/p1_m8_direction_models/ensemble_metrics.json`;
- `outputs/current/p1_m8_direction_models/ensemble_interface_contract.json`;
- `outputs/current/p1_m8_direction_models/seeds/seed_*/`;
- `tests/test_p0d_p1_artifacts.py`.

Full verification is 29/29 tests. This remains grouped simulation evidence, not
a real-robot posterior, OOD guarantee, M10/M12 transfer result, or closed-loop
speed/wear benefit.

## 2026-07-10 10:32 +08 - Connect the frozen ensemble to a 54-rollout P2 search pilot

Implemented `src/m8_direction_runtime.py` as the deployment adapter for the
three fixed-split CNN seeds. It owns only:

```text
phase resampling to 360 bins
-> Fz / Tx / Ty / tip-z preprocessing
-> 16D dimensionless context
-> per-seed temperature scaling
-> mean class probabilities + APS direction set
```

The adapter does not receive the true center. A regression test reproduces the
held-out P0d direction result through the runtime path.

Extended the dynamic search runner with `learned` and `oracle` modes. Learned
execution now performs:

```text
real 15 deg conical MuJoCo scan
-> frozen ensemble q(phi)
-> synthetic visual Gaussian samples reweighted by q(phi)
-> existing coverage-bounded posterior planner
-> finite transport + local outward spiral
-> white-box capture handoff
```

The ordinary outward baseline reads no direction. The oracle uses the true
center only as a lower bound. All true-center values remain offline labels for
learned evaluation.

Added `src/run_m8_learned_search_benchmark.py` with `tqdm`, per-trial JSON,
atomic `progress.json`, and resume. It selects 18 poses from the frozen P0d
spatial test split: six held-out pose groups and sectors 0/2/4. Each pose runs
learned, ordinary outward, and oracle, for 54 dynamic rollouts.

Fairness contract:

- 3 N down-force;
- 2.25 mm/s XY speed;
- 0.40 mm/rev spiral pitch;
- 30 s active budget, explicitly frozen inside P2;
- 2 mm isotropic synthetic visual-prior sigma;
- full learned probe time and work included;
- nominal contact dynamics in this first pilot.

The shared video script was externally changed during the run from a 30 s to a
110 s budget. P2 now passes its own explicit 30 s argument, so its report no
longer depends on that mutable video constant. Only the two temporary outer
learned reruns affected by the external change were overwritten under the
frozen 30 s contract.

P2 result:

| method | success | penalized time median | contact path median | wear proxy median |
|---|---:|---:|---:|---:|
| learned | 7/18 | 30.00 s | 30.51 mm | 55.27 mJ |
| ordinary outward | 2/18 | 30.00 s | 27.35 mm | 42.60 mJ |
| true-center oracle | 18/18 | 2.09 s | 4.71 mm | 7.85 mJ |

Every learned top-1 direction is within 7.5 deg, but overall learned success is
only 38.9%. The decisive split is whether the synthetic visual prior contains
the true center:

| prior stratum | learned | baseline | oracle |
|---|---:|---:|---:|
| inside 4.90 mm visual 95% radius | 7/9 | 2/9 | 9/9 |
| outside visual 95% radius | 0/9 | 0/9 | 9/9 |

Inside the visual support, learned penalized-time median is 15.51 s versus the
baseline 30 s timeout. Outside the support, a correct direction cannot supply
the missing radial distance. This is the first downstream evidence for the
direction-first interface and simultaneously a strong boundary against adding
an ungrounded learned distance command.

The endpoint was corrected to accept both wall-guided capture and centered
low-wrench below-rim insertion. Requiring lateral wall force had falsely marked
two exactly centered oracle trajectories as failures.

Artifacts:

- `outputs/current/p2_m8_learned_search_benchmark/progress.json`;
- `outputs/current/p2_m8_learned_search_benchmark/report.json`;
- `outputs/current/p2_m8_learned_search_benchmark/summary.md`;
- `outputs/current/p2_m8_learned_search_benchmark/trials/*.json`;
- `tests/test_m8_direction_runtime.py`;
- `tests/test_m8_learned_search_benchmark.py`;
- `tests/test_p2_learned_search_artifacts.py`.

Claim boundary: capture still uses privileged MuJoCo tags, visual covariance is
synthetic, only one pose has both learned and baseline success, and nominal
contact dynamics are used. P2 therefore does not prove general speed, wear,
hardware, OOD, or sim-to-real performance.

## 2026-07-10 10:39 +08 - Re-run P2 after the continuous-baseline correction

The 10:32 P2 entry is retained as historical trace, but its `baseline 2/18`
result was generated before the ordinary outward implementation received the
continuous 3 N force-control/search correction. To keep code and evidence
aligned, all 54 trials were regenerated with `--no-resume` under the same P2
contract: 30 s active budget, 3 N down-force, 2.25 mm/s XY speed, and
0.40 mm/rev pitch.

Current-code result:

| method | success | penalized time median | contact path median | wear proxy median |
|---|---:|---:|---:|---:|
| learned | 7/18 | 30.00 s | 30.51 mm | 55.27 mJ |
| continuous ordinary outward | 6/18 | 30.00 s | 56.08 mm | 91.25 mJ |
| true-center oracle | 18/18 | 2.09 s | 4.71 mm | 7.85 mJ |

Within the synthetic visual-prior 95% radius, learned is 7/9 with a 15.51 s
penalized-time median and baseline is 6/9 with a 17.11 s median. Outside that
radius both are 0/9. Five poses have both learned and baseline success;
learned-only and baseline-only counts are two and one respectively.

This correction materially narrows the claim. The current pilot shows at most
a small conditional benefit when the distance prior already covers the center.
The one-trial success difference is not evidence of general speed, success, or
wear superiority. It still demonstrates the more important boundary: a correct
direction posterior cannot recover radial distance that is absent from the
visual/CAD prior.

## 2026-07-10 10:42 +08 - Complete and accelerate the same-start comparison

Goal:

- represent the source-style baseline as an uninterrupted outward spiral under
  stable down-force until physical capture;
- let both policies reach the same endpoint while keeping the proposed route
  visibly faster and shorter;
- keep synchronized 1x playback until proposed completes, then accelerate only
  the remaining baseline segment with an explicit speed label.

Changed:

- `src/render_m8_search_comparison.py`;
- `tests/test_search_comparison_artifacts.py`;
- `CLAIM_BOUNDARY.md`;
- `docs/MUJOCO_MODELING_AND_EXPERIMENT_DESIGN.md`;
- `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`.

Command / run:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python3 -m src.render_m8_search_comparison
```

Artifacts:

- `outputs/current/comparison_videos/m8_outer_partial_proposed_vs_source_side_by_side.mp4`;
- `outputs/current/m8_outer_partial_search_comparison.json`;
- `outputs/current/comparison_keyframes/m8_outer_partial_comparison_keyframe.png`.

Result:

| metric | proposed | continuous source-style baseline |
|---|---:|---:|
| same privileged capture evaluator | reached | reached |
| active time | 15.91 s | 50.59 s |
| actual XY path | 21.70 mm | 109.86 mm |
| work proxy | 43.25 mJ | 182.74 mJ |
| hard stop | no | no |
| capture event radius | 0.190 mm | 0.118 mm |

The common initial condition is now `(4.00, 0.00) mm`, still an M8
outer-partial case. Retaining `(5.10, 0.00) mm` caused the ordinary spiral to
hit the outside wall geometry and cross the torque hard-stop before returning
to the hole; that case cannot truthfully show both policies completing under
the shared safety gates. The endpoint now also requires the bit center to be
inside the M8 top capture radius. This is a privileged simulation evaluator,
not information available to the online policy.

The side-by-side video is 23.13 s at 15 fps. It stays synchronized at 1x until
the proposed task finishes, then freezes the proposed result and displays the
baseline at 8x. Both panels retain physical simulation task clocks. Rendering
uses MuJoCo EGL on RTX A5000 GPU 0; contact dynamics remain CPU-solved.

Claim update:

- Implemented: one fixed-phase equal-endpoint visualization in which both
  policies complete and proposed uses less time and XY path.
- Partial: capture and safe-progress evaluation remain privileged MuJoCo/CAD
  diagnostics; proposed posterior remains assumed.
- Not proved: distribution-level superiority, learned closed-loop benefit,
  hardware timing/path/wear, or exact source-paper reproduction.

## 2026-07-10 10:44 +08 - Final P2/PPT evidence alignment

Synchronized the active claim documents and the Chinese Marp deck after both
current-code reruns completed. Two experiments now remain explicitly separate:

- P2 distribution pilot: 18 held-out poses x 3 methods, fixed 30 s budget;
  learned 7/18, continuous ordinary outward 6/18, oracle 18/18. Within the
  synthetic visual 95% radius the result is 7/9 versus 6/9 with 15.51 versus
  17.11 s penalized-time medians; outside it both are 0/9.
- Same-start communication video: one x=4.00 mm fixed-phase case, common 70 s
  budget and privileged equal endpoint; proposed 15.91 s versus continuous
  baseline 50.59 s. The proposed posterior remains assumed rather than learned.

Updated:

- `README.md`;
- `CLAIM_BOUNDARY.md`;
- `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`;
- `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- `tests/test_p2_learned_search_artifacts.py`.

Verification:

```text
python3 -m py_compile src/*.py: passed
python3 -m unittest discover -s tests -v: 33/33 passed
PPT: 32 H1 slides, div 63/63, figure 10/10
PPT: 32 unique local media references, 0 missing
PPT: 0 stale P2 or same-start result strings
```

The active claim is deliberately narrow: current P2 shows a one-trial
conditional success difference, not general superiority. The stronger
single-video separation is an assumed-posterior visualization and cannot be
used as learned distribution evidence.

## 2026-07-10 11:29 +08 - P2b in-domain coverage benchmark

Replaced the arbitrary 30 s success cutoff with a geometry-derived coverage
contract while preserving the original P2 artifacts. The declared simulation
acceptance set is the 2 mm isotropic visual prior's 4.895 mm radius. With the
0.25 mm effective capture radius, the ordinary spiral centerline covers to
4.645 mm. The generated 0.40 mm/rev path is 169.65 mm, requiring 75.40 s at
2.25 mm/s. A 175.45 s program guard is used only to stop pathological
verification churn; capture, radius exhaustion, and hard stop remain the
physical termination modes.

Implemented:

- truncated visual-prior support in `center_posterior_from_direction`;
- configurable posterior coverage in the dynamic learned rollout;
- `src/run_m8_p2b_coverage_benchmark.py` with live progress, resume, selective
  rerun, per-trial JSON, radius strata, and equal-endpoint paired metrics;
- five P0d radii inside the bound x six physical sectors x three methods =
  90/90 dynamic rollouts;
- corrected oracle routing: exact-center transport enters guarded descent
  without requiring a preceding chamfer/wall cue.

Final result:

| method | success | successful-time median | all-trial path median | all-trial work median |
|---|---:|---:|---:|---:|
| learned | 21/30 | 15.08 s | 31.29 mm | 59.77 mJ |
| continuous outward | 24/30 | 20.48 s | 50.37 mm | 86.02 mJ |
| corrected true-center oracle | 30/30 | 1.23 s | 2.68 mm | 4.31 mJ |

The overall medians mix endpoints. On the 21 poses where learned and baseline
both reach the same gate, learned has a -3.66 s paired median time difference,
a -19.09 mm paired median contact-path difference, shorter path on 21/21, and
lower work proxy on 20/21. Peak force/torque are not lower.

Radius audit:

| radius | learned | baseline | oracle |
|---|---:|---:|---:|
| 1.518 mm | 6/6 | 6/6 | 6/6 |
| 2.171 mm | 6/6 | 6/6 | 6/6 |
| 2.823 mm | 5/6 | 6/6 | 6/6 |
| 3.404 mm | 4/6 | 6/6 | 6/6 |
| 4.875 mm | 0/6 | 0/6 | 6/6 |

At 4.875 mm, baseline has six hard stops; learned has five safe radius
exhaustions and one program-guard exhaustion. All learned direction errors are
within 7.5 deg with median/p90 0 deg. This is therefore a radial action
boundary, not a direction-learning failure. With the current isotropic prior,
radius and angle are independent, so angular reweighting cannot create metric
distance information. The next narrow P2c experiment should fuse the support
state posterior with CAD-derived radial annuli on a separate split.

Claim boundary: nominal MuJoCo dynamics only; capture uses privileged contact
tags; no friction/stiffness/yaw nuisance, hardware, peak-wrench reduction, or
M10/M12 transfer claim.

## 2026-07-10 11:58 +08 - Trace-driven nominal FSM stabilization

Compressed the next iteration around one falsifiable nominal failure instead
of launching another large training run. The CAD full-support radius is

```text
R_nominal = R_head - R_bit = 6.5 - 6/sqrt(3) = 3.036 mm.
```

Added an opt-in physics-rate CSV trace to the dynamic search rollout and replayed
`supported_search_r04_s03` for learned and baseline. Each row records FSM stage,
nominal/actual pose, raw and filtered 6D wrench, candidate decisions, hard-stop
state, and offline MuJoCo contact tags. The initial learned trace reproduced:

```text
direction error: 0 deg; confidence: 0.980
candidate 1: tip z -0.096 -> -0.988 mm during frozen dwell
cue did not remain continuously true for 50 ms
FSM resumed planar spiral below the -0.30 mm CAD depth band
three attempts -> 74.10 mNm hard stop
```

Implemented the minimal white-box fix: when measured tip depth enters the CAD
below-rim band without crossing hard wrench limits, latch a sensor-depth commit,
forbid any planar-search resume, and enter guarded descent. No network,
direction posterior, or force threshold was retuned.

Same-point result after the fix:

| metric | before | after |
|---|---:|---:|
| result | hard stop | centered-insertion gate |
| candidate attempts | 3 | 1 |
| max torque | 74.10 mNm | 19.89 mNm |
| contact path | 44.79 mm | 44.20 mm |
| work proxy | 87.72 mJ | 79.70 mJ |

Then ran the three nominal radii x six physical sectors x learned/baseline =
36/36 development-set regression with live progress:

| metric | learned | baseline |
|---|---:|---:|
| capture / hard stop | 18/18 / 0 | 18/18 / 0 |
| capture-time median | 12.93 s | 17.11 s |
| contact-path median | 12.18 mm | 35.19 mm |
| work-proxy median | 28.03 mJ | 60.87 mJ |
| force / torque p90 | 8.87 N / 24.50 mNm | 7.92 N / 21.75 mNm |

Paired: learned is faster on 10/18, shorter-path on 18/18, and lower-work on
17/18. Peak force/torque remain slightly higher. This is a reused development
set after an FSM change, not independent evidence. Capture still terminates on
privileged MuJoCo-tag gates; only the new depth-commit latch is sensor/CAD based.

Artifacts:

- `outputs/current/p2b_nominal_failure_diagnostic/learned_trace.csv`;
- `outputs/current/p2b_nominal_failure_diagnostic/baseline_trace.csv`;
- `outputs/current/p2b_nominal_failure_diagnostic/summary.json`;
- `outputs/current/p2b_nominal_fsm_acceptance/report.json`;
- `src/run_m8_nominal_failure_diagnostic.py`;
- `src/run_m8_nominal_fsm_acceptance.py`.

Next gate: unseen local angles with applied friction/stiffness/yaw/down-force
nuisance, zero nominal hard stops, and a fully sensor-only progress/terminal
gate before any P2c transition claim.

## 2026-07-10 16:30 +08 - Align sampling plan with the conical-probe paper

Goal:
- decide what to borrow from `reference/锥形探索.pdf` before collecting the
  next CNN dataset, without treating its 5000-trial recipe as a requirement.

Paper facts checked:
- fixed $15^\circ$ tilt-then-rotate scan through $2\pi$ with constant
  down-force and admittance control;
- 12D pose+wrench, $N=2000$, channel normalization, 20-sample moving average,
  and polar representation;
- $X_z,M_x,M_y$ selected for a small CNN;
- 5000 uniformly sampled simulation poses over $x,y\in[-20,20]$ mm and
  yaw in $[-3^\circ,3^\circ]$, random 80/20 split, and 5% admittance-parameter
  noise;
- output is discrete error direction only; the paper explicitly does not
  recover error magnitude.

Decision:
- retain the fixed probe, phase semantics, simulation self-supervision,
  informative-channel ablation, and common wrench reference frame;
- do not copy the uniform square distribution, fixed 2000-step/time-rate
  coupling, mandatory polar rasterization, random sample split, or the claim
  that normalization guarantees scale/controller invariance;
- freeze the next independent M8 audit at 3 nominal plus 2 transition radial
  bands x 12 physical directions x 3 new Sobol nuisance repeats = 180 scans;
- keep these 180 scans out of training and add new training data only where
  angular calibration, APS set size, sector residual, or ensemble agreement
  fails;
- keep `outer_partial` as stress/OOD because its current CAD onset at 6.714 mm
  lies outside the declared 4.895 mm visual 95% domain;
- collect rim/chamfer, inner-wall, and inserted traces separately for white-box
  force-ramp/depth gate calibration, not direction-CNN training.

Changed:
- `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`;
- `../group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md`;
- `PROJECT_TRACE.md`.

Claim boundary:
- ready for deployment-weighted independent simulation sampling and a small
  hardware safety/calibration pilot;
- not yet real-posterior calibration, sensor-only terminal validation, or
  hardware scale-transfer evidence.

## 2026-07-10 16:55 +08 - Run the frozen 180-scan P1v2 direction audit

Goal:
- test the pre-existing P1 ensemble on unseen continuous angles and new
  nuisance combinations without fitting, recalibration, or threshold changes.

Implementation:
- added `build_independent_direction_audit_tasks` with five in-domain radial
  strata, 12 physical directions, and three paired Sobol nuisance repeats;
- added `src/sample_m8_independent_direction_audit.py` with 56-worker C-MuJoCo
  execution, live `tqdm`, atomic progress, resume, and a `training_use=forbidden`
  contract;
- added `src/evaluate_m8_independent_direction_audit.py` to evaluate the frozen
  three-seed ensemble once and write per-trial predictions;
- added task-design and artifact-boundary tests.

Run:
```bash
python3 -m src.sample_m8_independent_direction_audit --workers 56 --resume
python3 -m src.evaluate_m8_independent_direction_audit
```

Sampling result:
- 180/180 dynamic scans, zero failure/hard stop;
- approximately 68 s wall-clock on 56 CPU workers;
- radii 1.366, 2.277, 2.945, 3.486, and 4.550 mm;
- 108 supported-search and 72 double-overhang scans;
- 25 MB active evidence package.

Frozen ensemble result:

| metric | result |
|---|---:|
| angle median / p90 | 3.75 / 11.25 deg |
| within 7.5 deg | 0.761 |
| APS coverage / mean size | 0.972 / 3.68 bins |
| >90 deg error | 6/180 |
| catastrophic allowed by current transport gate | 5/6 |
| minimum catastrophic APS size | 9 bins |
| confidence-gate accept rate | 0.994 |
| diagnostic state accuracy | 0.983 |

Interpretation:
- the phase fingerprint retains useful set-valued direction information;
- the current maximum-softmax gate is overconfident on narrow unseen
  nuisance/contact regimes and is not ready for hardware transport;
- the $15^\circ$ local-angle stratum is materially harder than $45^\circ$,
  exposing the hard 48-bin label grid rather than a need for center-distance
  regression;
- P1v2 is now a viewed test set and will never be used for fitting,
  temperature scaling, gate selection, or targeted sample optimization.

Next:
- targeted new training data around innermost nominal and outermost transition
  regimes, with independent nuisance neighborhoods and continuous angle jitter;
- circular soft labels/density loss;
- APS width/circular concentration/ensemble-disagreement transport gate,
  calibrated away from P1v2;
- one new untouched audit after the model and gate are frozen.

## 2026-07-11 13:06 +0800 - Add live proposed polar fingerprint to comparison

Goal:

- make the proposed probe's information output visible in the same-start
  comparison instead of showing only the mechanical trajectory and wrench
  arrows.

Changed:

- `src/render_m8_search_comparison.py` now records render-rate measured
  `(theta, Xz, Mx, My)` samples during the 360 deg conical scan;
- `src/render_m8_sampling_videos.py` supports the same three-channel polar
  panel in a narrower upper-right layout;
- `tests/test_search_comparison_artifacts.py` checks the overlay contract.

Command / run:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python3 -m src.render_m8_search_comparison
python3 -m unittest discover -s tests -v
```

Artifacts:

- `outputs/current/comparison_videos/m8_outer_partial_proposed_vs_source_side_by_side.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_proposed_probe_to_inner_wall.mp4`;
- `outputs/current/comparison_keyframes/m8_outer_partial_comparison_keyframe.png`;
- `outputs/current/m8_outer_partial_search_comparison.json`.

Result:

- the proposed upper-right panel grows dynamically during the conical scan and
  then holds the completed `Xz/Mx/My` fingerprint through posterior transport,
  local search, completion, and the baseline 8x segment;
- the baseline speed badge remains separate and visible;
- completion metrics are unchanged: proposed 15.91 s / 21.70 mm, baseline
  50.59 s / 109.86 mm, both without hard stop;
- 47/47 tests pass.

Claim boundary:

- the polar plot visualizes measured simulation channels already used by the
  probe pipeline; it is not an additional learned result or hardware evidence.

## 2026-07-11 13:18 +08 - Repair the narrow direction interface and freeze P1v5

Goal:
- address the P1v2 continuous-angle alias without adding a radial-distance
  learner or broadening the learned control authority;
- perform one new final audit after model, gate, and runtime direction choice
  are frozen.

Implementation:
- added a 288-scan P1v3 targeted sampler over the inner nominal and outer
  transition regimes, with off-grid local angles and independent nuisance
  repeats;
- split P1v3 by base pose into 240 training and 48 calibration scans;
- trained three circular 1D-CNN seeds on GPUs 0--2 using 7.5 deg circular soft
  targets;
- changed the control direction from hard-bin top-1 to posterior circular mean;
- replaced the old max-softmax gate with a structured gate: APS set size at
  most 3 bins and ensemble top-1 span at most 7.5 deg;
- kept P1v2 out of fitting, calibration, and gate selection;
- used P1v4 only as a development audit for the circular-mean interface;
- generated P1v5 from new radii, 7/41 deg local angles, and a new Sobol seed
  after the interface freeze.

Runs:
```bash
python3 -m src.sample_m8_targeted_direction_training --workers 56
# Three parallel train_m8_direction_models runs on GPUs 0, 1, and 2.
python3 -m src.evaluate_m8_direction_ensemble \
  --output outputs/current/p1v3_m8_direction_models \
  --seeds 20260720 20260721 20260722 --split-seed 20260720 \
  --targeted-data outputs/current/p1v3_m8_targeted_direction_training
python3 -m src.sample_m8_final_untouched_direction_audit --workers 56
python3 -m src.evaluate_m8_independent_direction_audit \
  --input outputs/current/p1v5_m8_final_untouched_direction_audit \
  --model-root outputs/current/p1v3_m8_direction_models
```

P1v5 result:

| metric | result |
|---|---:|
| dynamic scans / hard stops | 180 / 0 |
| circular-mean median / p90 | 1.30 / 4.05 deg |
| within 7.5 deg | 0.989 |
| APS coverage / mean size | 0.994 / 2.63 bins |
| structured-gate accept / accepted accuracy | 179/180 / 0.994 |
| error above 90 deg | 0/180 |

Interpretation:
- the declared M8 simulation direction interface is now sufficiently stable
  to stop adding simulation direction data and prepare a hardware shadow-mode
  audit;
- the only 38.1 deg outlier is rejected by APS size, while one accepted 8.24
  deg error remains;
- max-class ECE is 0.304, so softmax is not a calibrated success probability;
- direction learning still does not recover radial distance, own force safety,
  validate sensor-only terminal events, or establish closed-loop superiority.

Next:
- identify the real F/T bias, frame, delay, force ramp, and admittance range;
- collect role-separated M8 hardware anchors and an untouched real test set;
- run the frozen P1v3 model in shadow mode before enabling any short transport;
- keep the guarded outward spiral as the fallback and do not tune on P1v5.

## 2026-07-11 13:26 +08 - Render every M8 conical sampling position on MuJoCo geometry

Goal:
- show the exact spatial coverage behind the direction learner without
  confusing nuisance repeats with new physical positions;
- make train/calibration, development audits, and final untouched evidence
  visually separable in the group-meeting deck.

Implementation:
- added `src/render_m8_sampling_position_map.py`;
- loaded the completed records from P0a/P0b/P0c/P0d and P1v2/P1v3/P1v4/P1v5
  manifests;
- rendered the actual `models/allen_m8_source.xml` screw/socket from a top-down
  MuJoCo camera and projected every commanded initial bit-center XY onto it;
- deduplicated only spatially identical XY markers while retaining all rollout
  counts in `m8_all_conical_sampling_positions.json`;
- marked the CAD 3.036 mm full-support radius, declared 4.895 mm visual 95%
  radius, dataset role, and P1v5 no-reuse boundary;
- added an artifact test and a dedicated PPT logic slide.

Artifacts:
- `outputs/current/m8_all_conical_sampling_positions_mujoco.png`;
- `outputs/current/m8_all_conical_sampling_positions.json`.

Result:
- 1674 completed dynamic scans;
- 651 unique commanded XY positions;
- eight role-separated source datasets;
- outer P0 observability/stress samples remain visible but are explicitly not
  presented as deployment probability or nominal-domain evidence.
- Python compilation and all 51 unit/artifact tests pass; all 33 local PPT
  image/video references resolve.

Claim boundary:
- the map proves where simulation data were collected, not uniform deployment
  coverage, radial identifiability, real-robot validity, or closed-loop benefit;
- rim/chamfer, inner-wall, and inserted white-box traces remain outside the
  complete-cone direction-CNN dataset.

## 2026-07-11 13:38 +08 - Compare frozen P1v3 output with the assumed-posterior video

Question:
- whether the completed sampler/training stack produces the same shortcut
  behavior as `m8_outer_partial_proposed_probe_to_inner_wall.mp4`.

Audit:
- confirmed that the existing MP4 and
  `m8_outer_partial_search_comparison.json` still use
  `direction_source=assumed_profile`;
- ran a new no-render MuJoCo episode from the same `[4.0, 0.0]` mm initial XY,
  with the frozen P1v3 three-seed ensemble, 3 N target force, 2 mm visual sigma,
  4.895 mm visual bound, and the same planner/controller path;
- saved the exact result in
  `outputs/current/p1v3_same_start_learned_outer_partial.json`.

Result:

| item | assumed-posterior video | frozen P1v3 learned rollout |
|---|---:|---:|
| center direction | 180 deg assumed | 176.25 deg, 3.75 deg error |
| posterior gate | not applicable | allowed; confidence 0.530; APS size 2 |
| bounded transport | 2.200 mm | 0.724 mm |
| local spiral max radius | 2.000 mm | 3.434 mm |
| capture time | 15.91 s | 42.06 s |
| actual XY path | 21.70 mm | 79.54 mm |
| hard stop | no | no |
| terminal mode | inner-wall guided | centered free insertion |

Interpretation:
- the learned output is qualitatively consistent with the assumed-posterior
  story: it identifies the correct center side, passes the structured gate,
  executes a bounded shortcut, then reaches insertion through outward search;
- it is not quantitatively equivalent: the assumed profile injects a 2.2 mm
  radial move, while the learned interface has only directional information
  and therefore keeps the transport conservative;
- the existing MP4 remains an optimistic mechanism illustration, not a video
  of the frozen P1v3 model;
- this single same-start run still uses privileged MuJoCo contact tags for the
  capture verifier and is not independent P1v5, hardware, or superiority
  evidence.

## 2026-07-11 14:31 +08 - Render genuine P1v3 full closure and freeze next experiments

Goal:
- replace the hand-assumed posterior only in a new artifact, while preserving
  every existing comparison video;
- continuously execute probe -> learned posterior -> bounded transport ->
  outward coverage -> compliant insertion -> CAD/Fz bottom verification;
- freeze the deployment-random and M10/M12 zero-shot tasks before generating
  more data.

Implementation:
- extended `run_method(..., continue_to_bottom=True)` with a continuous
  insertion handoff and CAD-depth/Fz dwell verifier;
- the first handoff smoke exposed a real nominal-Z discontinuity in the shallow
  assumed-posterior inner-wall case, producing a hard stop;
- fixed the handoff by anchoring to the current measured tip pose, ramping down
  force from 0.15 to 0.8 N over 1 s, then commanding bounded upright descent;
- added learned overlay fields for circular-mean direction, hard-bin top
  direction, APS size, concentration, seed span, structured gate, transport,
  and remaining spiral radius;
- added `src/render_m8_frozen_learned_full_comparison.py` and a synchronized
  1x three-panel composer;
- added `docs/NEXT_RANDOM_AND_SCALE_EXPERIMENTS.md` and matching PPT logic.

Frozen P1v3 output at `[4,0]` mm:
- top direction 176.25 deg; control circular mean 179.59 deg;
- true direction 180 deg; hard-bin error 3.75 deg;
- APS `[176.25,183.75]` deg, concentration 0.998, seed span 7.5 deg;
- structured gate allows transport;
- visual/CAD posterior planner chooses 0.724 mm transport and 3.434 mm local
  outward coverage.

Full-closure result:

| method | capture | CAD bottom | XY path | work proxy | hard stop |
|---|---:|---:|---:|---:|---:|
| hand-assumed posterior | 15.91 s | 20.78 s | 22.18 mm | 43.36 mJ | 0 |
| frozen P1v3 learned | 42.06 s | 44.67 s | 79.77 mm | 140.16 mJ | 0 |
| source-style outward | 50.59 s | 54.39 s | 110.24 mm | 182.75 mJ | 0 |

Artifacts:
- `outputs/current/comparison_videos/m8_outer_partial_p1v3_learned_full_to_bottom.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_assumed_posterior_full_to_bottom.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_source_outward_full_to_bottom.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_assumed_vs_learned_vs_source_full.mp4`;
- `outputs/current/m8_outer_partial_full_closed_loop_comparison.json`;
- four matching keyframes under `outputs/current/comparison_keyframes/`.

Preserved originals:
- `m8_outer_partial_proposed_probe_to_inner_wall.mp4`;
- `m8_outer_partial_source_outward_to_inner_wall.mp4`;
- `m8_outer_partial_proposed_vs_source_side_by_side.mp4`.

Next frozen order:
1. measure real visual residual covariance;
2. run 120 visual-prior random M8 triplets plus a separate 30-point boundary
   challenge without model/gate/planner tuning;
3. run M8 hardware shadow mode and sensor-only capture/terminal calibration;
4. run 60 zero-shot direction scans per M10/M12;
5. admit 36--60 calibration scans per size only after a documented zero-shot
   failure.

Claim boundary:
- this is the first video driven by the frozen P1v3 direction output and the
  current visual/CAD posterior planner, but it is still one synthetic-prior
  MuJoCo start with privileged capture tags;
- the assumed branch remains a hand-radial shortcut upper illustration, not a
  learned result;
- no random-deployment, sensor-only, hardware, or cross-size claim follows.

Verification:
- all 53 unit/artifact tests pass;
- all 41 local PPT image/video references resolve;
- the three original comparison MP4 modification times remain earlier than
  the new full-closure artifacts;
- the synchronized triptych keyframe was visually checked for nonblank MuJoCo
  panels, readable learned posterior fields, and non-overlapping headings.

## 2026-07-11 14:11 +08 - Retire the predecessor outside the active tree

Goal:
- keep `group_meeting/` focused on `mujoco_unscrewing_probe_new` without
  discarding the predecessor's code, raw data, and historical evidence.

Changed:
- compressed the complete predecessor directory into
  `/home/njy/njy_ws/archive/group_meeting_legacy/mujoco_unscrewing_probe_legacy_2026-07-11.tar.zst`;
- wrote the adjacent `.sha256` checksum file;
- updated `PROJECT_INDEX.md`, `PROJECT_RULES_AND_LESSONS.md`, and
  `CLEANUP_MANIFEST.md` to identify the new path as active and the old path as
  archived;
- removed the uncompressed predecessor from the active `group_meeting` tree
  only after archive verification.

Verification:
- `zstd -t` passed;
- archive/source entry counts both equal `2910`;
- `tar --compare` reported no differences;
- SHA-256 is
  `e097bf5faf3fcbe25cddbf2082ec03098c493c48f8bad53e900abcb9571d3dd2`.

Claim boundary:
- no active new-path source, evidence, model, or test artifact was removed;
- historical paths inside the append-only top-level `project_trace.md` remain
  historical records and were not rewritten.

## 2026-07-11 14:31 +0800 - Add Jin 2021-inspired direct-and-retry baseline

Goal:

- add a credible comparison for `reference/锥形探索.pdf` that exposes the
  direction-only action contract and the paper's documented full-probe recovery
  cost;
- contrast direct motion under direction error with the proposed bounded local
  spiral without claiming an exact paper reproduction.

Paper audit:

- Section III-A uses a full tilt-then-rotate scan under constant downward force;
- Section III-C outputs a discrete error direction from polar `Xz/Mx/My`
  channels and guides the peg along that direction with admittance;
- the paper explicitly states that error magnitude is unavailable;
- Section III-D attributes failures to wrong direction or insufficient
  admittance compensation, then shifts initialization and repeats the full
  probe; more than three attempts is counted as failure;
- Table I/II contain nonzero multi-attempt and failure counts.

Changed:

- added `src/render_m8_paper_direct_retry_comparison.py`;
- added `docs/JIN2021_DIRECT_RETRY_BASELINE.md`;
- added `tests/test_paper_direct_retry_comparison.py`;
- updated `src/render_m8_search_comparison.py`, `README.md`,
  `CLAIM_BOUNDARY.md`, and comparison artifact tests.

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python3 -m src.render_m8_paper_direct_retry_comparison
python3 -m unittest discover -s tests -v
```

Artifacts:

- `outputs/current/videos/baselines/m8_outer_partial_paper_direct_retry_to_inner_wall.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_proposed_vs_paper_direct_retry_side_by_side.mp4`;
- `outputs/current/m8_paper_direct_retry_comparison.json`;
- `outputs/current/comparison_keyframes/m8_outer_partial_proposed_vs_paper_direct_retry_keyframe.png`.

Selected stress sequence:

| attempt | reset XY | scripted direction error | result | active time end |
|---:|---:|---:|---|---:|
| 1 | `(4.00, 0.00)` mm | `+22.5 deg` | no capture | 8.63 s |
| 2 | `(4.00, 0.25)` mm | `-22.5 deg` | no capture | 19.27 s |
| 3 | `(4.00, -0.25)` mm | `0 deg` | capture | 28.85 s |

Comparison result:

| method | successful attempt | capture time | actual XY path | hard stop |
|---|---:|---:|---:|---|
| proposed assumed posterior + local spiral | 1 | 15.91 s | 21.70 mm | no |
| Jin 2021-inspired direct + retry | 3 | 28.85 s | 26.56 mm | no |

Claim boundary:

- the direction errors are scripted class-boundary stress values selected to
  visualize recovery; they are not outputs from the paper CNN or local P1v3;
- this is a selected single M8 case, not a random draw or failure-rate estimate;
- the original square/pentagonal geometry, CNN, training data, and controller
  gains are not reproduced;
- the local spiral has a coverage argument only inside its declared posterior
  region with valid pitch/capture assumptions, not a global or hardware
  guarantee;
- 53/53 tests pass.

## 2026-07-11 15:36 +08 - Add P2c F/T-state and approximate-CAD radial fusion

Goal:

- use the contour information already encoded in the conical F/T phase trace to
  make the posterior transport distance more geometrically grounded;
- preserve the old direction-only rollout and keep outward coverage as the
  residual-error guarantee.

Changed:

- added `CadRadialGeometry`, soft support-state likelihoods, and optional CAD
  conditioning to `src/render_m8_search_comparison.py`;
- the frozen learner now supplies both `q(phi)` and `p(search_state|F/T)`;
- added posterior radial quantiles, CAD boundaries, state probabilities, and an
  explicit posterior-model identifier to rollout JSON;
- added unit/artifact tests for soft boundary behavior, radial-information gain,
  output provenance, preserved videos, and the negative peak-force result;
- updated `README.md`, `CLAIM_BOUNDARY.md`,
  `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`,
  `docs/REAL_ADMITTANCE_AND_POSTERIOR_SEARCH.md`, and the Chinese Marp deck.

Artifacts:

- `outputs/current/comparison_videos/m8_outer_partial_p1v3_cad_fused_full_to_bottom.mp4`;
- `outputs/current/comparison_videos/m8_outer_partial_assumed_vs_cad_fused_vs_source_full.mp4`;
- `outputs/current/comparison_keyframes/m8_outer_partial_p1v3_cad_fused_full_keyframe.png`;
- `outputs/current/comparison_keyframes/m8_outer_partial_cad_fused_triptych_keyframe.png`;
- `outputs/current/m8_outer_partial_cad_fused_full_closed_loop_comparison.json`.

Same-start result:

| quantity | direction-only | P2c soft CAD | source outward |
|---|---:|---:|---:|
| transport / local radius | 0.724 / 3.434 mm | 1.500 / 2.945 mm | none / 4.200 mm |
| CAD-bottom time | 44.67 s | 31.62 s | 54.39 s |
| XY path | 79.77 mm | 45.89 mm | 110.24 mm |
| work proxy | 140.16 mJ | 83.81 mJ | 182.75 mJ |
| max force | 7.54 N | 22.43 N | 6.12 N |
| hard stop | no | no | no |

Claim boundary:

- CAD is approximate and enters as a 0.35 mm soft transition with a 0.02 floor;
- no true center, MuJoCo contact point, or learned distance target enters the
  posterior;
- capture still uses privileged MuJoCo tags and vision remains synthetic;
- efficiency improves in this selected case, but peak force worsens; distribution
  and transition-stress validation are required before hardware transport.

## 2026-07-11 15:36 +0800 - Add default-off post-progress micro-yaw assist

Goal:

- add a bounded engineering option for residual hex clocking after a persistent
  downward-progress/depth-commit cue, without changing the default policy;
- keep the conical fingerprint, posterior transport, and local spiral free of
  yaw oscillation.

Changed:

- added `src/capture_yaw_assist.py` with a default-off configuration and a
  smoothly ramped `+/-0.5 deg`, `1.5 Hz` command;
- connected it only to guarded descent/insertion in
  `src/render_m8_search_comparison.py`;
- added a yaw-only soft abort at `|tau_z| >= 10 mN m`; the 60 mN m six-axis
  controller hard stop and guarded down-force behavior remain independent;
- added CLI parameters headed by the explicit `--capture-yaw-assist` switch;
- connected the same default-off switch to the full-closure comparison
  generator, where the post-capture insertion stage is actually present;
- added `tests/test_capture_yaw_assist.py` and updated the design, controller,
  README, and claim-boundary documentation.

Smoke results:

- default off: same outer-partial result, 15.9075 s inner-wall confirmation,
  no yaw command, no yaw abort;
- enabled with the inner-wall endpoint: no unnecessary yaw because the endpoint
  is already satisfied before guarded insertion;
- enabled with `continue_to_bottom=True`: yaw was used, the 10 mN m soft limit
  disabled it, and the original descent still reached the CAD-bottom verifier
  at 20.7850 s without changing the hard-stop contract.

Claim boundary:

- this implements the mechanism and safety routing only;
- it does not show a capture-rate, clocking, wear, or hardware-robustness gain;
- the existing MP4 and JSON evidence was not regenerated because the feature is
  disabled by default;
- 61/61 tests pass. The artifact inventory test was synchronized with the two
  already-present CAD-fused full-closure videos and no media file was changed.

## 2026-07-11 15:48 +08 - P2c verification and slide synchronization

- full dynamic render completed in four stages and preserved the historical
  direction-only learned and triptych videos;
- visually inspected the new learned and triptych keyframes; posterior fields,
  F/T state, transport, local radius, and comparison metrics are readable;
- `python3 -m unittest discover -s tests -v`: 62/62 pass;
- deck static audit: 43 slides, 41 local media links, zero missing links, 77/77
  paired `div` tags, and 36 display-math markers;
- Marp CLI compilation was not run because this host has `node` but no
  `npm`/`npx`/`marp` executable.

## 2026-07-11 20:38 +0800 - Move micro-yaw assist before chamfer capture

Correction:

- the earlier post-progress-only placement could not improve the probability of
  initial chamfer capture because it started after the capture cue;
- the default-off assist now starts at the first proposed/learned local-spiral
  step, after posterior transport, and is superimposed on the planned XY path;
- yaw continues through the 50 ms progress dwell and guarded insertion, while
  the conical fingerprint, posterior transport, and source-style baseline stay
  yaw-free;
- the existing 10 mN m yaw-only soft abort and 60 mN m controller hard stop are
  unchanged.

Same-case trace smoke:

- the first nonzero yaw command is in stage group `spiral`, before capture;
- inner-wall confirmation remains successful at 15.9100 s with no hard stop;
- maximum measured `|tau_z|` is 6.078 mN m, so the yaw soft abort does not fire;
- this verifies routing only and is not a capture-probability improvement
  result; 62/62 tests pass.

## 2026-07-11 20:51 +08 - Replace biased direct stress with a fair active baseline

Audit:

- the old Jin-style comparison injected `+22.5/-22.5/0 deg` errors rather than
  consuming the local learner output;
- it also used the true initial offset magnitude to choose direct travel;
- it is therefore a selected recovery stress diagnostic, not a fair active
  baseline and not evidence that proposed is superior.

Changed:

- added a `direct` method to `src/render_m8_search_comparison.py` that uses the
  exact same conical F/T trace and frozen P1v3 circular-mean direction as the
  proposed method;
- direct receives no true center distance and no CAD radial likelihood; it
  follows a ray up to the declared visual bound and stops on the common capture
  gate;
- added `src/render_m8_fair_three_method_comparison.py` and the clean
  `outputs/current/active_comparison/` entry point;
- moved the old scripted stress video/result/keyframes under
  `outputs/current/diagnostics/paper_direct_scripted_stress/`;
- added `outputs/current/README.md` and synchronized README, claim boundary,
  baseline design, experiment plan, cleanup manifest, tests, and Chinese deck.

Fair same-start result:

| method | control evidence | CAD bottom | XY path | work | max force |
|---|---|---:|---:|---:|---:|
| proposed | shared P1v3 179.59 deg + F/T state + vision/CAD | 31.62 s | 45.89 mm | 83.81 mJ | 22.43 N |
| Baseline A outward | no probe/posterior | 54.39 s | 110.24 mm | 182.75 mJ | 6.12 N |
| Baseline B direct | same probe + same P1v3 179.59 deg | 11.22 s | 6.25 mm | 15.93 mJ | 4.72 N |

Boundary:

- Baseline B wins this selected case; proposed superiority is not supported;
- retry is part of the paper-style failure policy but is not exercised because
  attempt one succeeds;
- proposed residual coverage must be tested on frozen angular-error, multimodal,
  and boundary distributions before any robustness advantage is claimed.

Verification:

- visually inspected the fair direct and three-method keyframes; headers and
  metrics do not overlap;
- `python3 -m unittest discover -s tests -v`: 63/63 pass;
- deck static audit: 44 slides, 47 local media references, zero missing, 80/80
  paired `div` tags, and 36 display-math markers;
- Marp CLI remains unavailable because this host has no `npm`/`npx`/`marp`.

## 2026-07-11 21:12 +08 - Physically organize `outputs/current`

Goal:

- remove loose top-level images, JSON files, and P0/P1/P2 directories without
  deleting evidence or leaving generators pointed at obsolete paths.

New root structure:

```text
outputs/current/
  README.md
  active_comparison/
  datasets/
  models/
  evaluations/
  media/
  reports/
  diagnostics/
```

Changed:

- added the idempotent migration tool `src/organize_current_outputs.py`;
- moved P0/P1 sampling under `datasets/`, frozen models under `models/`, audits
  and benchmarks under `evaluations/`;
- split media into `media/videos/{stages,baselines,comparisons}` and
  `media/keyframes/{stages,comparisons}`, with loose figures under
  `media/figures/`;
- moved compact top-level JSON outputs under `reports/` and retained failure
  traces/stress illustrations under `diagnostics/`;
- rewrote source defaults, tests, generated JSON paths, documentation, and the
  Chinese deck; active-comparison links now target the new media tree.

Verification:

- migration is idempotent: second run reports zero moves and zero rewrites;
- root contains only the eight entries listed above;
- 2060 regular files plus 3 readable active-method symlinks remain;
- all 100 `outputs/current/...` references found in 267 JSON files resolve;
- 63/63 tests pass and all source files compile;
- deck audit remains 44 slides / 47 local media references / zero missing /
  80 paired `div` tags.

## 2026-07-11 22:18 +08 - Force audit, posterior distance release, and frozen fair-policy test

Questions:

- why the selected Proposed rollout reported 22.43 N under a 3 N target;
- whether the fixed 1.50 mm transport cap was unnecessarily conservative;
- whether a fair Baseline B naturally fails under the same learned direction,
  rather than through injected errors.

Controller findings and changes:

- replayed the selected rollout with a 0.5 ms trace; the 22.43 N / 141.02 mN m
  value was a single search-to-insertion handoff sample, while probe/transport/
  spiral maxima were 4.35/5.23/7.07 N;
- replaced instantaneous force/admittance handoffs with continuous force
  ownership; the identical full rollout remained successful and fell to
  7.07 N / 20.89 mN m;
- added raw-wrench safety observation at every physics step, independent of the
  2 ms admittance update;
- removed the redundant fixed-XY descent confirmation after a persistent 50 ms
  safe-progress cue; all policies now use that cue only to stop planar search;
- retained upright compliant insertion and CAD/depth-wrench verification as the
  later completion stages.

Distance policy:

- single-case caps 1.50 / 2.00 / posterior-limited selected transports of
  1.500 / 2.000 / 2.257 mm;
- posterior-limited reduced capture time 26.72 -> 15.91 s, contact path
  45.56 -> 21.82 mm, and work proxy 83.76 -> 43.21 mJ without increasing peak
  force;
- the frozen policy uses projected Q10 transport, Q99 residual local coverage,
  and a full visual-support outward fallback only after local exhaustion.

Fair benchmark design:

- 36 new Sobol positions inside the declared visual 95% radius;
- paired friction 0.55--0.95, contact time constant 0.04--0.07 s, and down-force
  2.5--3.5 N;
- Proposed and Baseline B share byte-identical first fingerprints and the same
  frozen P1v3 direction output;
- Baseline B receives no true distance/CAD radial posterior and retries at most
  three times with frozen +/-0.5 mm perpendicular offsets.

Frozen result:

| policy | capture | hard stop | median time | median contact path | median work |
|---|---:|---:|---:|---:|---:|
| Proposed posterior/local/global | 34/36 | 2 | 16.00 s | 21.72 mm | 45.72 mJ |
| Baseline A full outward | 31/36 | 5 | 44.32 s | 82.85 mm | 151.37 mJ |
| Baseline B shared direction/direct retry | 34/36 | 0 | 7.46 s | 5.52 mm | 13.50 mJ |

Interpretation:

- the strict evaluator additionally requires entry into the CAD socket capture
  region, preventing screw-head outer-edge chamfer drop from counting as capture;
- Proposed used global fallback once; Proposed and B each had two exclusive
  successes, so McNemar p=1.0 shows no net success-rate difference;
- B was faster, shorter, and lower-work on all 32 common successes;
- versus A, Proposed was faster in 80.0%, shorter in 96.7%, and lower-work in
  93.3% of 30 common successes, but had no peak-force advantage;
- the supported claim is a posterior shortcut with geometric fallback, not
  blanket superiority in time, wear, force, or success rate.

Artifacts:

- `docs/M8_FORCE_AND_TRANSPORT_AUDIT.md`;
- `outputs/current/evaluations/m8_transport_distance_ablation/`;
- `outputs/current/evaluations/m8_fair_policy_final_audit/`;
- viewed/pre-fix runs are retained under `outputs/current/diagnostics/` and were
  not reused as final evidence.

Final synchronization:

- corrected the capture evaluator so screw-head outer-edge chamfer drop cannot
  count as socket capture; it now also requires entry into the CAD capture region;
- regenerated the learned/source full-closure videos, active three-method video,
  keyframes, and manifests with the corrected evaluator;
- selected-case Proposed / A / B CAD-bottom times are 17.27 / 51.93 / 9.05 s;
- `python3 -m unittest discover -s tests -v`: 67/67 pass;
- all Python sources compile;
- deck audit: 45 slides, 47 local media references, zero missing, 82/82 paired
  `div` tags, and 36 display-math markers;
- 426 JSON files contain 100 `outputs/current/...` path references, all resolving.

## 2026-07-12 00:07 +08 - Shared observation-noise audit and honest Baseline-B failure video

Question:

- add realistic sim-to-real perturbations before collecting expensive hardware
  scans;
- expose a Baseline-B failure only if it emerges from a frozen paired test,
  rather than by injecting a hand-selected direction error.

Implementation:

- added `src/wrench_observation.py`, a deterministic stateful six-axis
  observation chain with residual episode bias, random-walk drift, colored
  noise, gain/cross-axis error, and delay;
- inserted the observed wrench before admittance, fingerprint preprocessing,
  frozen P1v3 inference, progress gates, and safety observation;
- retained the underlying MuJoCo wrench as physical simulation state and used
  simulation truth only for offline evaluation;
- extended `src/run_m8_fair_policy_benchmark.py` with a shared-noise mode while
  preserving the clean default;
- Proposed and Baseline B use the same initial pose, contact nuisance, sensor
  seed, and byte-identical first noisy fingerprint.

Pre-hardware 36-pose audit:

| method | capture | hard stop | median success time | median contact path | median work |
|---|---:|---:|---:|---:|---:|
| Proposed posterior + residual coverage | 30/36 | 5 | 20.28 s | 32.60 mm | 64.93 mJ |
| Baseline A full outward | 26/36 | 10 | 39.12 s | 84.27 mm | 137.22 mJ |
| Baseline B shared direction + direct/retry | 25/36 | 0 | 7.33 s | 6.00 mm | 14.10 mJ |

Paired interpretation:

- Proposed/B: 21 both-success, 9 Proposed-only, 4 B-only, 2 both-failed;
- McNemar exact two-sided `p=0.267`, so no significant success-rate claim;
- B remains faster, shorter, and lower-work on every common success;
- negative direct-ray geometric margin occurs in 13/36 trials; B attempt-one
  success is 15.4% there versus 91.3% for non-negative margin;
- the supported mechanism claim is that low-confidence geometric residual
  coverage changes the failure set, not that noise makes Proposed universally
  superior.

Audit-selected video:

- selection was frozen as the lexicographically first trial where Proposed
  succeeds and the full Baseline-B retry policy fails;
- `sobol_002` is selected: circular-mean error 18.98 deg, APS size 8, so
  Proposed sets `d_move=0` and succeeds through local outward coverage;
- Baseline B attempt one exhausts its direct ray, and the audit records failure
  after all three retries; Baseline A hard-stops in this case;
- the video contains no scripted angle perturbation.

Artifacts:

- `outputs/current/evaluations/m8_pre_hardware_noise_audit/`;
- `outputs/current/active_comparison/robustness_noise/`;
- Chinese deck now separates the 36-pose statistics from the conditional
  failure illustration and records the observation-model claim boundary.

Boundary:

- observation magnitudes are pre-hardware engineering priors, not measured
  sensor statistics;
- hardware calibration must replace them with measured bias/gravity residual,
  covariance/Allan behavior, frame error, and end-to-end delay;
- the clean comparison and clean frozen audit remain preserved and active.

Verification:

- `python3 -m unittest discover -s tests -v`: 71/71 pass;
- all Python source and tests compile;
- four new MP4 files decode at 15 fps; individual videos are 960 x 720 and
  the synchronized comparison is 1440 x 416;
- visually inspected a mid-rollout frame and the final comparison keyframe;
  headers, polar panels, paths, and failure summary do not overlap incoherently;
- deck audit: 46 slides, 49 local media references, zero missing, 88/88 paired
  `div` tags, and 38 display-math markers;
- all JSON files parse; 129 `outputs/current/...` references resolve with zero
  missing paths;
- `outputs/current/` retains the eight-entry clean root structure.

## 2026-07-12 00:40 +08 - Graded-transport diagnosis and complete Baseline-B retry video

User-visible issue:

- in `sobol_002`, Proposed appeared not to translate before spiraling;
- the noisy comparison showed only Baseline-B attempt one even though the
  frozen audit evaluated three shifted full re-probes.

Transport diagnosis:

- the observation-shifted P1v3 output has APS size 8 and 30 deg seed span, so
  the frozen binary gate sets `transport_allowed=false` and `d_move=0`;
- offline geometry is $r=3.055$ mm and circular-mean error 18.98 deg;
- along that direction, $d^*=r cos(error)=2.889$ mm would leave 0.994 mm
  residual, while 0.5/1.0 mm moves would leave 2.587/2.134 mm;
- therefore the zero/nonzero gate is safe but plausibly too conservative;
- no policy result was silently changed. The proposed next ablation is graded
  transport from the projected joint-posterior quantile, with zero reserved for
  reverse-mass, severe multimodality, or OOD.

Baseline-B visualization correction:

- rendered three separate full 15 deg probe + direct-ray attempts using the
  exact audit starts, sensor seeds, and frozen P1v3 outputs;
- inserted two dynamic reset clips: force released, vertical retract, 0.5 mm
  shifted re-initialization, guarded retouchdown, and contact settle;
- recomposed `03_baseline_b_full_three_attempt_retry.mp4` and the synchronized
  three-method comparison;
- the two reset clips add 7.47 s of rendered motion; the frozen audit still
  excludes this time/path, which favors B;
- attempts remain independent rollout episodes. At each cut the solver state is
  reconstructed at the same final XY and released nominal surface height before
  the reset motion; continuous contact-solver state across cuts is not claimed.

Documentation:

- added the graded-gate derivation and frozen ablation plan to the Chinese deck
  and `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`;
- updated the Baseline-B paper boundary, active comparison README, manifest,
  and artifact tests to require all three attempts and both reset clips.

Verification:

- 71/71 tests pass and all Python files compile;
- visually inspected both reset clips, an attempt-two conical frame, the full
  Baseline-B summary, and the synchronized comparison keyframe;
- reset banner text fits its panel and the three-attempt labels remain visible;
- active MP4 files decode at 15 fps: Proposed 611 frames, A 463, full B 553,
  synchronized comparison 656;
- deck audit: 47 slides, 49 local media references, zero missing, 90/90 paired
  `div` tags, and 42 display-math markers;
- all JSON files parse and 137 `outputs/current/...` references resolve;
- `outputs/current/` retains the clean eight-entry root structure.

## 2026-07-12 01:30 +08 - Continuous posterior transport and regenerated noisy comparison

Planner change:

- replaced the legacy `APS > 3 -> d_move=0` execution gate with continuous,
  risk-bounded transport in `src/posterior_search_plan.py`;
- the planner forms projected center samples $s_i=(c_i-p)^T u$, continuously
  selects a lower quantile between Q10 and Q25 from circular concentration and
  ensemble seed span, and discounts that quality by posterior reverse mass;
- transport is still capped by a quality-scaled posterior radius and optional
  hardware step limit; reverse mass at or above 0.20 still forces zero move;
- the 99% residual posterior is recomputed after transport, so the local
  outward spiral remains a coverage fallback rather than being removed.

Selected-case result (`sobol_002`):

- the same noisy fingerprint gives direction quality 0.898, reverse mass
  0.001, selected quantile 0.235, and `d_move=0.991 mm`;
- residual local radius decreases from the legacy 3.160 mm to 2.599 mm;
- Proposed reaches the inner-wall gate at 20.84 s and CAD bottom at 22.23 s;
  XY path is 32.73 mm, max force 7.43 N, and max torque 18.47 mNm;
- Baseline A reaches the socket neighborhood at XY (-0.202, -0.295) mm and
  z=-0.977 mm, but terminates in chamfer/wall contact at 14.35 N and
  69.31 mNm. It therefore fails the safe capture gate and trips the common
  60 mNm wrench safety limit; this is not a radius/program stop or failure to
  geometrically reach the hole.

Regenerated audit:

- reran all 36 visual-95% Sobol poses with the same contact and pre-hardware
  observation nuisance design;
- Proposed/A/B captures are 32/26/25 with hard stops 3/10/0;
- Proposed/B paired counts are 23 both, 9 Proposed-only, 2 B-only, and 2 both
  failed (`McNemar p=0.065`);
- Proposed median successful time/path/work are 12.30 s / 17.72 mm / 34.02 mJ;
  B remains faster, shorter, and lower-work on all 23 common successes;
- the result therefore supports a robust hybrid/fallback claim, not universal
  dominance or hardware-level statistical evidence.

Artifacts and synchronization:

- regenerated `outputs/current/active_comparison/robustness_noise/` including
  Proposed, Baseline A, all three Baseline-B attempts, physical reset clips,
  synchronized comparison, keyframe, and per-method CSV traces;
- regenerated the clean CAD-fused full-closure and active three-method videos
  with the same continuous planner. In the high-quality `[4,0]` mm case,
  transport/local radius are now 2.880/1.800 mm and CAD-bottom time/path/work
  are 13.63 s / 13.97 mm / 29.42 mJ;
- updated the Chinese deck, README, claim boundary, dataset/observability note,
  force/transport audit, Baseline-B note, manifest, and artifact tests;
- preserved the old clean videos and historical binary-gate trace entries.

Verification:

- `python3 -m unittest discover -s tests -v`: 72/72 pass;
- all Python source and tests compile;
- the regenerated comparison keyframe was visually inspected and the torque
  row, three method columns, and failure summary remain readable;
- the observation distribution is still an engineering prior and capture
  evaluation still uses privileged MuJoCo tags, so neither is hardware proof.

## 2026-07-12 03:10 +08 - Sensor-only capture handoff and physical fallback closure

Independent physical diagnosis:

- the previous Baseline-A failure at `sobol_002` was not caused by insufficient
  geometric coverage. A safe downward-progress point already existed near
  `r=0.317 mm`, `F=2.16 N`, and `tau_xy=3.86 mNm`, but the controller waited for
  a privileged `r<=0.25 mm`/contact-tag condition and later wedged at 69 mNm;
- relaxing the 20 N / 60 mNm hard stop would hide the mechanism and reduce
  hardware safety. The correction is an earlier sensor-trend handoff plus
  unloading/compliance, not a looser failure definition;
- Proposed cannot be assumed to dominate Baseline A merely because it includes
  an outward fallback. Its shortcut changes contact history, so the fallback
  must physically unload, reset, retouch, and receive an independent complete
  coverage budget.

Controller changes:

- search-to-insertion handoff now uses only tip-z history and observed wrench:
  at least 0.07 mm absolute depth drop, another 0.04 mm over 50 ms, and raw
  wrench below 8 N / 20 mNm;
- on a candidate, XY is frozen, target down-force ramps to 2 N over 150 ms, and
  planar impedance changes from 12 kN/m / 420 N s/m to 4 kN/m / 280 N s/m;
  another 0.04 mm descent confirms capture, otherwise parameters are restored;
- a 12 N / 40 mNm pre-hard escape unloads and resets search. The original
  immediate 20 N / 60 mNm hard stop remains unchanged;
- Proposed global fallback now performs unload, lift, return to visual anchor,
  guarded retouchdown, admittance reset, and a newly budgeted complete
  Baseline-A outward search.

Updated 36-pose pre-hardware audit:

- Proposed / Baseline A / Baseline B capture is 36/36, 35/36, and 29/36 with
  hard-stop counts 0/1/0;
- successful median time is 14.06 / 42.84 / 7.51 s; median contact path is
  17.72 / 84.09 / 6.02 mm and median work proxy is 34.01 / 136.87 / 14.19 mJ;
- Proposed/B paired counts are 29 both, 7 Proposed-only, 0 B-only, 0 both-fail
  (`McNemar p=0.015625`). Proposed/A counts are 35 both and one Proposed-only;
- Proposed used four physical-reset/full-coverage fallbacks. This explains the
  finite simulation success-set inclusion but is not a mathematical or
  hardware guarantee. Baseline B remains the fastest direct shortcut when its
  ray captures.

Video and documentation synchronization:

- the new audit-selected `sobol_002` comparison shows Proposed and A succeeding;
  B misses twice, physically resets twice, and succeeds on attempt three;
- regenerated the clean full-closure and fair three-method active videos with
  the same sensor/depth-trend handoff. The clean case reaches CAD bottom in
  13.73 / 52.05 / 9.17 s for Proposed/A/B;
- preserved the old noisy failure comparison as a legacy pre-handoff artifact;
- synchronized the Chinese deck, README, claim boundary, controller note,
  force/transport audit, dataset note, Baseline-B note, manifests, and tests.

Verification:

- `python3 -m unittest discover -s tests`: 72/72 pass;
- Python source compiles; refreshed comparison keyframes were visually checked;
- deck audit: 48 slides, 49 local media references, zero missing, 91/91 paired
  `div` tags, and 44 display-math markers;
- all 471 JSON artifacts under `outputs/current/` parse; the clean, noisy-retry,
  and full-closure comparison MP4s decode at 15 fps with 846/621/846 frames;
- search handoff no longer reads true center or MuJoCo contact tags. CAD-bottom
  remains a simulation-stage endpoint, and all noise values still require
  hardware calibration.

## 2026-07-12 04:04 +08 - iiwa tip-RCC funnel guard and CAD-depth defense

Physical decision:

- kept the conical fingerprint, frozen CNN, posterior transport, outward
  spiral, and their baselines unchanged;
- introduced the new mechanism only after the sensor-only downward-progress
  handoff, because the 15 degree probe is geometrically incompatible with a
  0.10 mm-clearance, 4 mm-deep M8 socket;
- treated the transfer from `reference/漏斗效应.pdf` as an execution principle
  (compliance plus environmental constraints), not evidence that its inclined
  corner funnel or no-chamfer experiments prove the M8 capture basin;
- assigned the iiwa native Cartesian impedance to the low-level motion, the
  external F/T sensor to observation/safety, and a calibrated peg-tip frame to
  the virtual-RCC compliance center. Hardware transforms are not yet known.

Implementation:

- added `src/funnel_guard.py` with a 150 ms contact-energy/progress observer,
  sustained power and 0.75 degree tilt guards, a CAD-depth envelope, and one
  bounded ordinary-stall unload/retract/upright/retouch recovery;
- extended `src/mujoco_admittance_rollout.py` with tip-frame translational and
  rotational impedance so search behavior remains unchanged and the proxy RCC
  is activated only post-capture;
- updated `src/render_m8_search_comparison.py` so entry into the CAD terminal
  band freezes further axial command for F/T dwell. CAD overshoot now unloads
  and safely aborts; it is not mislabeled as a recoverable stall;
- added a bounded post-capture TCP-command bias to the ablation interface so
  the guard is tested under calibration/execution error without true-center
  access;
- added `src/run_m8_funnel_guard_ablation.py`, a 12-task/24-rollout paired
  audit, and `src/render_m8_funnel_guard_ablation.py`, whose video case is
  selected by a fixed rule from that audit rather than hand-picked.

Paired MuJoCo result:

| metric | legacy insertion | funnel guard |
|---|---:|---:|
| CAD bottom verified | 5/12 | 4/12 |
| hard stop | 1/12 | 0/12 |
| CAD-depth safe abort | 0/12 | 6/12 |
| peak force p90 | 17.31 N | 11.15 N |
| peak torque p90 | 24.32 mN m | 25.61 mN m |
| maximum CAD-depth overrun | 1.00 mm | 0.49 mm |

Interpretation:

- the guard currently acts as a defense layer: it removes one hard stop and
  limits the worst depth overrun, but also rejects one boundary completion;
- the result does not support higher completion rate, lower torque, hardware
  passivity, wear reduction, or M10/M12 parameter transfer;
- all six interventions in this batch are CAD-depth safe aborts. The
  energy-without-progress observer is unit-tested, but the complete physical
  stall-recovery sequence was not dynamically exercised, so no recovery-benefit
  claim is allowed;
- structural transfer is plausible, while force, stiffness, energy, dwell,
  and tilt thresholds must be calibrated per hardware tool/size.

Artifacts:

- `docs/IIWA_FUNNEL_GUARD.md`;
- `outputs/current/evaluations/m8_funnel_guard_ablation/report.json`;
- `outputs/current/active_comparison/funnel_guard/comparison_post_capture_funnel_guard.mp4`;
- `outputs/current/active_comparison/funnel_guard/comparison_post_capture_funnel_guard_keyframe.png`.

Synchronization and verification:

- regenerated the clean assumed/learned/outward full-closure videos; all three
  use the same post-capture guard and still reach CAD bottom at 17.41/13.73/
  52.05 s respectively;
- regenerated the active shared-direction direct baseline and clean triptych;
  Baseline B still wins the selected case at 9.17 s, so the guard does not
  change the search-comparison conclusion;
- regenerated the audit-selected shared-noise/retry comparison; Proposed,
  Baseline A, and all three Baseline-B attempts use the same post-capture guard;
- `python3 -m unittest discover -s tests -q`: 80/80 pass;
- all 485 JSON files under `outputs/current/` parse; the 50-slide deck has 51
  local media references with zero missing and 94/94 paired `div` tags;
- the new guard, clean three-method, and noisy-retry comparison videos decode
  as 207/846/623 nonblank frames. `ffprobe` is not installed, so decoding was
  verified through the same ImageIO/FFmpeg backend used to write the videos.

## 2026-07-12 05:17 +08 - Full closed-loop scientific audit and four-method freeze

Goal:

- audit the complete task from physical assumptions through sensing, sampling,
  learning, posterior planning, compliant control, safety, and terminal logic;
- add a scientifically fair random-search baseline without weakening the two
  existing baselines;
- align the active evidence, slide deck, claim boundary, and minimum-cost real
  data plan.

Scientific decisions:

- froze the mechanism as two funnels: a perception funnel from one conical
  fingerprint to a bounded direction shortcut, and an execution funnel from
  residual outward coverage to guarded upright insertion;
- retained the state head only as a calibrated soft state-CAD radial
  likelihood with a nonzero floor. It cannot be a hard state gate, radial
  distance head, capture trigger, or terminal verifier;
- removed true center and MuJoCo contact tags from online inner-wall/inserted
  decisions. Geometry and tags are now episode-end offline scorers only;
- defined Baseline C as seeded area-uniform random waypoints in the same visual
  disk, with Baseline-A speed, force, path budget, sensor handoff, and
  post-capture guard. It is a generic unstructured baseline, not a reproduction
  of either paper;
- kept Baseline B strong: Proposed/B share a byte-identical first fingerprint
  and frozen direction output; B receives no state-CAD radius or spiral
  fallback, retries up to three times, and still excludes reset travel/time;
- clarified that the MuJoCo tip-RCC/impedance layer is an iiwa-compatible
  Cartesian proxy, not a simulated or connected KUKA iiwa full-arm controller;
- clarified that CAD-bottom verification ends geometric engagement only, not
  motor coupling, reverse unscrewing, or screw extraction.

Implementation:

- added random-waypoint planning and `method="random"` rollout support to
  `src/render_m8_search_comparison.py`;
- extended `src/run_m8_fair_policy_benchmark.py` to Proposed/A/B/C, online
  sensor-only gates, offline geometry scoring, and success-conditioned
  efficiency summaries;
- added `src/render_m8_four_method_comparison.py` and generated the same-start
  four-panel full-engagement video;
- added P1v5 state NLL/Brier/ECE reporting and
  `src/audit_m8_state_cad_fusion.py`;
- added `src/build_closed_loop_audit_manifest.py` and a machine-readable
  cross-module evidence snapshot;
- added random-baseline and full-audit regression tests.

Frozen 36-pose result:

| method | eventual / first attempt | successful time median | successful path median | hard stop |
|---|---:|---:|---:|---:|
| Proposed | 34/36 / 34 | 12.47 s | 14.61 mm | 0 |
| Baseline A outward | 30/36 / 30 | 36.06 s | 77.07 mm | 1 |
| Baseline B direct/retry | 28/36 / 25 | 7.49 s | 5.36 mm | 0 |
| Baseline C random | 19/36 / 19 | 10.19 s | 20.82 mm | 1 |

- Proposed/B: 27 both, 7 Proposed-only, 1 B-only, 1 both-failed,
  McNemar `p=0.0703125`;
- Proposed/A common successes: Proposed faster on 82.8%, shorter and lower
  work on 86.2%;
- Proposed/C: random can be fast on lucky chords, but eventual success is
  19/36 and its all-trial path median is 86.38 mm;
- Proposed uses four physical-reset/full-outward fallbacks;
- Proposed/B first fingerprint match rate is 1.0;
- 13 direct rays have negative geometric capture margin; B first-attempt
  success is 0.154 there versus 1.000 at nonnegative margin.

State-CAD action-interface result:

- P1v5 state accuracy/NLL/Brier/ECE is
  `0.9889/0.01887/0.01052/0.01260`;
- direction+visual versus soft state-CAD posterior coverage is
  `0.9944/1.0000`;
- median post-transport true residual is `1.399/0.805 mm`;
- reverse-move rate is zero for both. This is offline simulation action
  evidence, not a learned-distance or dynamic-insertion result.

Real-data plan:

- H0: measured visual covariance plus F/T frame, gravity, bias, covariance and
  delay;
- H1: 12 low-force six-sector commissioning scans;
- H2: 24 adaptation scans (18 structured anchors + 6 repeats);
- H3: 12 permanently untouched random scans;
- active-add at most 12 uncertainty/action-boundary scans only after failure;
- M10/M12 start with 18 structured zero-shot + 12 untouched random per size;
  calibration is admitted only after a documented zero-shot failure.

Artifacts:

- `docs/FULL_CLOSED_LOOP_SCIENTIFIC_AUDIT.md`;
- `outputs/current/reports/closed_loop_scientific_audit.json`;
- `outputs/current/evaluations/m8_four_method_sensor_only_audit/`;
- `outputs/current/evaluations/m8_state_cad_fusion_audit/`;
- `outputs/current/active_comparison/four_methods/comparison_four_methods_2x2.mp4`;
- `outputs/current/active_comparison/four_methods/comparison_four_methods_2x2_keyframe.png`;
- `tests/test_m8_closed_loop_scientific_audit.py`;
- `tests/test_random_search_baseline.py`.

Documentation synchronization:

- compacted `README.md`, `CLAIM_BOUNDARY.md`, and
  `outputs/current/README.md` around the current interface while retaining old
  evidence in this append-only trace;
- updated `DESIGN_DECISIONS.md`, `CLEANUP_MANIFEST.md`, the sampling/model note,
  and the Chinese Marp deck;
- moved the stale root `MUJOCO_LOG.TXT` into
  `outputs/current/diagnostics/legacy_runtime/` without deleting its two
  historical unstable-QACC warnings;
- the deck now contains the four-baseline ladder, 2x2 video, frozen 36-point
  result, safety ablation, exact claim boundary, and 48-scan initial hardware
  plan.

Verification:

- `python3 -m unittest discover -s tests -v`: 87/87 pass;
- all Python source/tests compile;
- all 539 JSON files under `outputs/current/` parse;
- deck audit: 50 slides, 47 local media references, zero missing, 99/99 paired
  `div` tags, and 48 display-math markers;
- all four individual videos and the 2x2 MP4 decode at 15 fps; the four-panel
  keyframe and frames 60/180/360/600 were visually inspected;
- the active 2x2 case reaches CAD bottom for all four methods and is explicitly
  marked explanatory rather than success-rate evidence.

Current claim:

- M8 simulation supports a calibrated circular direction shortcut, a soft
  vision/CAD action posterior, residual geometric coverage, and a separate
  post-capture defense layer;
- it does not support true-center prediction, universal superiority, measured
  wear, passivity, M10/M12 transfer, real iiwa robustness, or autonomous
  unscrewing.

## 2026-07-12 Baseline C Maximum-Entropy Grid Alignment

Decision:

- replaced active Baseline C seeded random waypoints with a Chen-inspired
  visual/CAD center-grid belief and maximum expected entropy contact selection;
- retained random waypoints only as an appendix action-selection ablation and
  moved the old active 2x2 assets to
  `outputs/current/diagnostics/four_methods_seeded_random_legacy/`;
- did not copy the paper's full-contact-footprint claim: this checkout only has
  wrist 6D F/T, so C uses a CAD support-centroid likelihood for signed
  $[\tau_x,\tau_y]/\|F\|$ categories;
- C has no conical learner and no spiral fallback. Each information action is
  unload -> lift -> shift -> guarded low-force touchdown. If no probe captures,
  the final action is the same guarded primitive at the posterior mean.

Physical audit and correction:

- a magnitude-only cue was rejected because inner-socket and outer-head edges
  can produce similar $\|\tau_{xy}\|/\|F\|$ while carrying opposite signs;
- the 12-point nominal M8 cue audit found all four inner/outer pairs
  magnitude-ambiguous, all four signed categories distinct, CAD category 11/12
  matched directly, and zero hard stops;
- early probe implementations exposed loaded-contact transients and an unsafe
  loaded posterior ray. They are retained under
  `outputs/current/diagnostics/m8_four_method_entropy_interrupted_unsafe_probe/`
  and `outputs/current/diagnostics/m8_four_method_entropy_interrupted_high_force_commit/`;
- the frozen primitive now unloads first, uses slow vertical approach, freezes
  nominal Z at first contact, ramps only to 0.8 N, and hands off on sensor-only
  below-rim progress. Retests of the previously unsafe cases produced no hard
  stop; a wrong posterior becomes a bounded top-support failure rather than a
  force spike.

Frozen paired evaluation:

| method | capture / 36 | no-retry | success time median | successful total/contact XY | hard stop |
|---|---:|---:|---:|---:|---:|
| Proposed | 34 | 34 | 12.47 s | 14.71 / 14.61 mm | 0 |
| Baseline A outward | 30 | 30 | 36.06 s | 77.11 / 77.07 mm | 1 |
| Baseline B direct/retry | 28 | 25 | 7.49 s | 5.36 / 5.36 mm | 0 |
| Baseline C entropy-grid | 22 | 22 | 11.14 s | 5.35 / 1.36 mm | 0 |

- P/A/B trial dictionaries were reused byte-for-byte from the frozen
  sensor-only audit; C used exactly the same 36 task/noise seeds;
- P/C paired counts are both-success 21, P-only 13, C-only 1, both-fail 1,
  exact McNemar $p=0.001831$;
- among the 21 common successes C is faster in 71.4%, has lower total path in
  85.7%, and lower contact path/work in 100%, but its median 7.5 active contacts
  and 22/36 coverage expose its real tradeoff;
- the defensible claim is therefore fewer active probes and higher coverage for
  one learned fingerprint plus residual/full coverage, not universal speed or
  path superiority.

Video alignment:

- selected `sobol_002` by a frozen rule: first lexicographic trial where
  Proposed succeeds and Baseline B succeeds only after retry;
- Proposed and B share the identical first noisy 15-degree fingerprint; B's
  unload/lift/shift/retouchdown/full re-probe sequence is visible for all three
  attempts;
- all four methods share the initial task/nuisance, sensor-only capture handoff,
  hard limits and post-capture guard;
- C captures during its fourth entropy-selected probe in this conditional
  example and is allowed to look faster than Proposed. The video explains
  mechanism only and is not a success-rate estimate.

Current artifacts:

- `docs/M8_ENTROPY_GRID_BASELINE_C.md`;
- `outputs/current/evaluations/m8_entropy_contact_cue_calibration/`;
- `outputs/current/evaluations/m8_four_method_entropy_audit/`;
- `outputs/current/active_comparison/four_methods/comparison_four_methods_2x2.mp4`;
- `outputs/current/active_comparison/four_methods/comparison_four_methods_2x2_keyframe.png`;
- `tests/test_entropy_grid_baseline.py`;
- `tests/test_m8_closed_loop_scientific_audit.py`.

Claim boundary:

- C is an adaptation of grid belief and entropy selection, not an exact
  reproduction of the paper's contact-footprint sensor;
- the signed-F/T likelihood is nominal MuJoCo calibration, not iiwa hardware
  calibration;
- the old random search remains traceable but is no longer an active 2x2 method;
- selected videos cannot replace distribution-level paired statistics.

Terminal/video boundary:

- the 2x2 is scored and labeled at sensor-only inner-wall/capture handoff;
- C captures on probe four in `sobol_002` but does not satisfy CAD-bottom dwell,
  so this selected video is not evidence that all four methods complete the
  terminal insertion verifier.

Verification:

- `python3 -m unittest discover -s tests -v`: 93/93 pass;
- all Python source/tests compile;
- all 612 JSON files under `outputs/current/` parse;
- deck audit: 51 slides, 48 unique local references, zero missing, 100/100
  paired `div` tags, and 50 display-math markers;
- Proposed/A/B/C source videos and the 2x2 decode at 15 fps with nonblank
  first/middle/last frames; the final keyframe and an in-probe C frame were
  visually inspected.

## 2026-07-12 Terminal False-Negative Correction

The preceding statement that entropy-grid C had not completed terminal
insertion was too strict. C had already reached the CAD depth band with only
`0.000425 mm` error, low wrench and no hard stop. The old verifier required
`|Fz| >= 0.70 N` for every contact mode; because C entered as
`centered_free_insertion` with little preload, freezing the axial command on
first band entry prevented it from building that force. Proposed/A arrived
with chamfer/wall preload and passed the same force-only gate. This was a
mode-dependent terminal false negative, not a C search failure.

Correction:

- added a method-independent `TerminalCompletionVerifier`;
- retained `bottom_force_dwell`: CAD depth tolerance 0.15 mm and axial force at
  least 0.70 N for 50 ms;
- added `cad_depth_low_wrench_dwell`: the same depth band, lateral force at most
  0.25 N, tilt torque at most 1.5 mN m, and depth span at most 0.02 mm for
  50 ms;
- neither channel reads the true center or MuJoCo contact tags;
- high lateral load or unstable depth cannot pass the low-wrench channel.

The same frozen `sobol_002` rerun keeps all four C probe actions unchanged. C
captures on probe four and now returns `cad_bottom_verified` at 7.77 s through
`cad_depth_low_wrench_dwell`, with `0.000457 mm` depth error and no hard stop.
The 2x2 keyframe therefore labels all four methods `complete`. This is selected
simulation evidence of geometric insertion, not a distribution-level success
rate or a real bit/socket coupling claim.

Post-correction verification:

- `python3 -m unittest discover -s tests -v`: 97/97 pass;
- all 612 current JSON files parse and all Python source/tests compile;
- deck audit remains 51 slides, 48 local references, zero missing, 100/100
  paired `div` tags and 50 display-math markers;
- the regenerated C and 2x2 videos decode at 15 fps with nonblank
  first/middle/last frames; the final keyframe was visually inspected.

## 2026-07-12 Proposed+ Entropy-Order Correction

Finding:

- the first six-method Proposed+ implementation ran a short entropy-refinement
  prefix before returning to the original anchor and executing the old
  transport + spiral tail;
- in the video the two sub-millimeter probes were nearly invisible, so the
  method appeared to be conical -> straight line -> spiral;
- this did not match the intended claim that the joint posterior should first
  make a shortcut and then use entropy-grid contacts as the primary residual
  search.

Correction:

- reordered Proposed+ to conical fingerprint -> 0.990 mm bounded shortcut ->
  piecewise-straight low-force entropy nodes with quasi-static endpoint dwell;
- node motion keeps contact at at most 0.8 N and uses minimum-jerk timing; only
  endpoint samples update the signed-F/T likelihood;
- one non-contracting spatial update is tolerated, but two consecutive
  non-contracting updates stop the grid and reject its posterior;
- on rejection, the controller returns under low force to the shortcut anchor,
  restores nominal search force/impedance, and only then enables local outward
  coverage. The spiral is recorded as
  `value_of_information_exhausted_before_capture`;
- fixed the audit wording so Baseline-C/Proposed+ probe ceiling 64 is documented
  as a watchdog rather than a nominal sample budget.

Selected-case physical result (`sobol_002`):

| field | value |
|---|---:|
| entropy nodes | 3 |
| capture / CAD bottom | 27.09 / 28.44 s |
| total / contact XY | 36.78 / 36.52 mm |
| max force / torque | 5.89 N / 18.68 mN m |
| local / global fallback | yes / no |
| terminal result | `cad_bottom_verified` |

The grid observations reduced Shannon entropy but made $r_{95}$ expand on the
last two updates, so using the frozen conical plan was the physically safer
choice. This is evidence for routing and fallback, not Proposed+ superiority.
In the same selected video Proposed completes at 22.28 s, C at 11.56 s, and the
frozen random route at 5.68 s by luck.

Artifacts:

- `docs/PROPOSED_PLUS_ENTROPY_POLICY.md`;
- `outputs/current/active_comparison/six_methods/comparison_six_methods_3x2.mp4`;
- `outputs/current/active_comparison/six_methods/comparison_six_methods_3x2_keyframe.png`;
- `outputs/current/active_comparison/three_method_race/comparison_proposed_plus_proposed_entropy_race_1x3.mp4`;
- `outputs/current/active_comparison/three_method_race/comparison_proposed_plus_proposed_entropy_race_keyframe.png`.

Verification:

- selected no-render MuJoCo rollout completed without hard stop;
- the Proposed+ single video was visually sampled at 8/11/14/17/20/23 s and
  shows entropy nodes before the explicitly labeled fallback spiral;
- all six selected methods reach the shared CAD-bottom verifier;
- the 1x3 display accelerates unfinished panels by 4x only after the first
  physical completion and does not change simulation metrics.
- final automated verification: 101/101 unit/artifact tests pass; all 701 JSON
  files under `outputs/current/` parse; all source/tests compile;
- deck audit: 53 slides, 51 unique local media links, zero missing, 103/103
  paired `div` tags and 54 balanced display-math markers;
- the 3x2 and 1x3 MP4 files decode at 15 fps with nonblank first/middle/last
  frames.

## 2026-07-12 Proposed+ Task-Aware Belief-Grid Correction

Independent mechanism audit:

- visual information is an initial center density, not a complete solution;
- the fixed 15-degree conical fingerprint contributes a direction/state
  likelihood, not a metric center command;
- approximate CAD defines support/capture geometry and the observation model;
- pure maximum entropy can select a diagnostic edge while making little task
  progress, so it is insufficient as the Proposed+ action objective.

Implemented correction:

- changed each action score to capture probability along the whole guarded
  segment plus expected information gain, minus travel and contact cost;
- made absence of capture a soft negative swept-area observation;
- temperature-downweighted surprising F/T classes so a mismatched simulated
  likelihood cannot dominate geometric evidence;
- retained minimum-jerk low-force node-to-node motion and quasi-static endpoint
  labels;
- disabled Proposed+ local spiral; full visual-support outward coverage remains
  a separately reported last resort;
- kept capture handoff, tip virtual RCC, energy/health shield, hard limits, and
  CAD-bottom verifier shared with all comparison methods.

Selected `sobol_002` result after correction:

| field | value |
|---|---:|
| bounded shortcut | 0.990 mm |
| task-grid contacts | 3 |
| capture / CAD bottom | 9.828 / 11.696 s |
| total / contact XY | 5.334 / 4.795 mm |
| wear proxy | 11.308 mJ |
| max force / torque | 5.873 N / 15.390 mN m |
| local / global fallback | no / no |
| terminal result | `cad_bottom_verified` |

Development-pilot boundary:

- the first 12 frozen Sobol tasks were used while correcting the policy;
- they gave 11/12 completion, median 4.5 task-grid contacts, zero local spirals,
  and two full-support global fallbacks;
- these 12 tasks cannot be reused as independent evidence. The remaining 24
  frozen tasks are reserved as the first untouched holdout for this policy.

Active artifacts:

- `docs/PROPOSED_PLUS_ENTROPY_POLICY.md`;
- `outputs/current/active_comparison/six_methods/comparison_six_methods_3x2.mp4`;
- `outputs/current/active_comparison/three_method_race/comparison_proposed_plus_proposed_entropy_race_1x3.mp4`;
- `outputs/current/active_comparison/funnel_guard/comparison_post_capture_funnel_guard.mp4`.

## 2026-07-12 Proposed+ Fresh Validation And Capture-Gate Closure

Trace-driven findings:

- the untouched 24-pose split first scored 21/24 and exposed two global-fallback
  handoff hard stops plus two online outer-wall false capture cues;
- one reset trace showed actual peg position already near the socket while the
  nominal controller still pulled toward the visual anchor, growing lateral
  force/torque to the hard limit;
- the false cues moved -2.12/-1.50 mm along the frozen conical top-1 direction,
  while strict true captures in the audit were all at least +1.44 mm.

Implemented closure:

- low-force continuous return replaces lift/retouch for Proposed+ global
  fallback; capture preempts force ramp, return, and spiral;
- force, planar/rotational impedance, and nominal Z now transfer without a
  command jump;
- a CAD-scaled one-sided direction-consistency gate rejects reverse-side
  depth/F/T candidates;
- no hard threshold was relaxed and no online true center/contact tag was added;
- unit tests cover forward/local acceptance, reverse-side rejection, and the
  0.25 mm CAD tolerance.

Same-task regression:

- 35/36 strict capture, 0 hard stops;
- one `search_radius_exhausted` replaces the previous outer-wall false success;
- this is regression evidence only because those tasks were used for repair.

Fresh frozen seed-20260723 validation:

| method | strict capture | time | total/contact path | wear | hard stop |
|---|---:|---:|---:|---:|---:|
| Proposed+ | 21/24 | 18.35 s | 10.58/9.58 mm | 12.11 mJ | 0 |
| Proposed | 23/24 | 12.58 s | 14.69/14.66 mm | 29.64 mJ | 0 |
| A outward | 23/24 | 33.28 s | 70.71/70.71 mm | 124.80 mJ | 0 |
| B direct/retry | 22/24 | 7.93 s | 6.18/6.18 mm | 14.26 mJ | 0 |
| C entropy-grid | 23/24 | 12.14 s | 6.82/2.18 mm | 0.62 mJ | 0 |

On Proposed+/Proposed common successes, Proposed+ changes median total/contact
path and wear by -2.38/-2.33 mm and -7.15 mJ, and has lower peak force in 85.7%
of pairs, but is +2.41 s slower. It therefore remains experimental and does not
replace Proposed. Frozen full-depth continuation of the three Proposed+
near-boundary misses reaches CAD bottom in 0/3 cases: two safe aborts and one
bottom not verified.

Active evidence:

- `outputs/current/evaluations/m8_fresh_24_pabc_direction_gate/`;
- `outputs/current/evaluations/m8_fresh_24_task_aware_validation/`;
- `outputs/current/evaluations/m8_fresh_near_boundary_bottom_check/`;
- `outputs/current/evaluations/m8_six_method_task_aware_grid_direction_gate_regression/`.

## 2026-07-12 Final Proposed+ Artifact Verification

- regenerated the 3x2 mechanism matrix and the Proposed+/Proposed/C race after
  the task-aware grid and direction-consistency fixes;
- confirmed the physical-funnel comparison remains a separate post-capture
  safety ablation and does not enter search-policy ranking;
- moved two development pilots and one interrupted rollout from active
  evaluations to `outputs/current/diagnostics/proposed_plus_development/`;
- 110/110 unit and artifact tests pass;
- all 891 JSON files under `outputs/current/` parse and all `src/`/`tests/`
  Python sources compile;
- deck audit: 54 slides, 52 local media references (51 unique), zero missing,
  104/104 paired `div` tags, 3/3 paired `pre` tags, and 54 balanced display-math
  markers;
- visually inspected the final 3x2, 1x3, and funnel-guard keyframes. The videos
  retain explicit conditional-example and claim-boundary labels.

## 2026-07-12 Output Cleanup, Closed-Loop Audit, And Cross-Size Direction Test

Taste/architecture decision:

- kept Proposed as the paper-facing method because its load-bearing mechanism
  is one informative conical fingerprint plus coverage-preserving search;
- retained Proposed+ as an experimental task-aware action-selection ablation,
  since fresh M8 evidence does not justify replacing the simpler method;
- preserved the hierarchy `physical probe -> narrow learner -> geometry-aware
  planner -> independent safe controller`.

Closed-loop implementation correction:

- found that `MuJoCoAdmittanceRollout` hard-coded the dynamic body name
  `allen_bit_m8`, which made M10/M12 execution fail before physics started;
- replaced the size-specific lookup with automatic dynamic-body discovery from
  the shared `bit_tip` site;
- added a three-size regression test for this interface.

Frozen cross-size direction audit:

- generated 72 new scans for each M8/M10/M12 geometry using 3 support states,
  2 CAD-normalized radial bands, 2 off-grid angles, 6 physical sectors, and
  paired Sobol contact nuisance;
- the M8 P1v3 ensemble, preprocessing, 15 deg action, APS, and thresholds were
  frozen; no M10/M12 sample was used for fitting or calibration;
- all 216 scans completed with zero hard stops;
- M8/M10/M12 top-1 within-15-deg rates are 1.000/0.986/0.972 and APS coverage
  is 1.000/0.986/0.972, with no greater-than-90-deg errors;
- M12 has 15.3% context beyond the M8 5-sigma envelope and p90 raw force/torque
  7.82 N/30.5 mN m, so only the normalized direction interface passes. Full
  scale transfer remains unproved.

Output cleanup:

- retained exactly three active MP4 files: the 3x2 search matrix, the 1x3
  Proposed+/Proposed/C race, and the funnel safety ablation;
- archived 43 historical/component videos (123.5 MiB source) to
  `/home/njy/njy_ws/archive/group_meeting_legacy/`
  `mujoco_unscrewing_probe_new_historical_videos_2026-07-12.tar.zst`;
- archive SHA-256 is
  `009d413425186744edee68acc68b8e13b9558e9c1d9d2d2135c188830fcc2b01`;
- verified with `zstd -t`, `sha256sum -c`, a 43-member listing, and
  `tar --compare` before source removal;
- moved old 2x2/noise metadata to
  `outputs/current/diagnostics/historical_comparisons/`.

Communication cleanup:

- archived the previous 54-slide deck externally and rebuilt the active deck
  as a 24-slide mechanism-first narrative;
- only the three retained videos remain clickable; old dynamic stages are
  represented by static keyframes;
- updated README, claim boundary, machine-readable audit, tests, and video
  retention manifests together.

Diagnostic raw-trace cleanup:

- identified 21 historical diagnostic CSV traces totaling 753,051,805 bytes;
- archived them to
  `/home/njy/njy_ws/archive/group_meeting_legacy/`
  `mujoco_unscrewing_probe_new_diagnostic_traces_2026-07-12.tar.zst`;
- archive SHA-256 is
  `9a433b3cba644106b9da8576adad7d7ceb166ebbb826c59017b9080473d8a183`;
- verified with `zstd -t`, `sha256sum -c`, a 21-member listing, and
  `tar --compare` before removing the active CSV files;
- retained compact JSON reports, manifests and keyframes as readable indexes.

Final verification:

- active `outputs/current` contains exactly 3 MP4 files and no video symlinks;
- first/middle/last frames of all 3 MP4 files decode at 15 fps, are nonblank,
  and differ over time;
- active output size is 318 MiB, reduced from approximately 1.1 GiB before raw
  diagnostic cleanup;
- 113/113 unit and artifact tests pass;
- all 897 JSON files under `outputs/current` parse and all `src/`/`tests/`
  Python files compile;
- compact deck audit: 25 slides, 14 unique local media links, zero missing,
  17/17 paired `div` tags, 2/2 paired `pre` tags, and 16 balanced display-math
  markers;
- both external archive checksums pass after active-tree cleanup.

## 2026-07-12 M10/M12 S2 Full-Loop Decision And Three-Funnel Paper Deck

Cross-size S2 implementation:

- generalized the dynamic runtime to explicit M10/M12 CAD and MuJoCo model
  paths while preserving the M8 defaults;
- added size-conditioned capture radius, visual support, transport cap,
  finite-width spiral pitch and CAD-conditioned low-force terminal descent;
- ran 30 unseen Sobol poses per M10/M12 size with a frozen M8 P1v3 learner and
  shared Proposed/outward/oracle nuisance;
- preserved all 60 trials as `training_use=forbidden` evaluation evidence in
  `outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop/`.

S2 result:

| size | Proposed / outward / oracle bottom | Proposed hard stop | executed control direction median/p90 | executed direction >90 deg |
|---|---:|---:|---:|---:|
| M10 | 23/30 / 21/30 / 27/30 | 3 | 2.58/21.78 deg | 1 |
| M12 | 21/30 / 19/30 / 30/30 | 4 | 3.58/24.76 deg | 1 |

- both sizes fail the preregistered full-loop transfer gate;
- on paired common successes, Proposed is 15.88/17.86 s faster and
  46.16/50.54 mm shorter than outward for M10/M12;
- each size has two failed trials with executed control-direction error above
  45 deg, but more failures arise from search/terminal or accurate-direction
  capture-release/reset/spiral hard stops;
- corrected the report to gate and summarize the circular-mean direction that
  the planner actually executes, while retaining top-1 bin error as a
  compatibility diagnostic;
- decision: M10/M12 need fresh 12--24-scan calibration plus independent
  controller/terminal commissioning, not immediate full backbone retraining.

Paper communication:

- rebuilt the active Chinese Marp deck around three successive contractions:
  fingerprint funnel, search funnel and physical funnel;
- added paper-generated figures for the mechanism overview, information flow,
  related-work gap, real phase/posterior example, M8 untouched audit, bounded
  search, direct-ray margin, paired benchmark, funnel guard, cross-size S1/S2
  and minimum-cost sim-to-real plan;
- kept Proposed as the paper-facing method and Proposed+ as an explicit
  action-selection ablation;
- documented the S2 failure decision in
  `docs/CROSS_SIZE_S2_DECISION.md` and synchronized README, claim boundary,
  scale plan and full scientific audit.

Final verification:

- 116/116 unit and artifact tests pass;
- all 963 active JSON files parse and all `src/`/`tests/` Python files compile;
- the active deck contains 48 slides, 28 unique local media targets, zero
  missing references, balanced HTML blocks and balanced display math markers;
- Marp generated both a 48-page PDF and a 48-slide PPTX; the PPTX archive
  integrity check passes;
- visually inspected all 48 rendered pages and corrected title contrast,
  section-divider contrast, malformed LaTeX spacing commands and the stale S2
  limitation sentence.

## 2026-07-15 Paper-Facing Four-Method Simplification

- Promoted Proposed / outward A / shared-fingerprint direct-retry B /
  seed-fixed random C as the sole paper-facing baseline ladder.
- Generated
  `outputs/current/active_comparison/four_methods/comparison_four_methods_2x2.mp4`
  and its keyframe/manifest from the same-start, same-nuisance, shared-safety
  six-method source panels; Random is relabeled from historical D to active C.
- Repointed the paper benchmark figure and claim to the frozen 36-pose
  `m8_four_method_sensor_only_audit`: Proposed/A/B/C = 34/30/28/19 captures.
- Moved the former six-method matrix and three-method race to
  `outputs/current/diagnostics/historical_comparisons/`; entropy-grid and
  Proposed+ remain traceable deferred research and were removed from the deck.
- Synchronized the Chinese deck, README, claim boundary, full audit, scale
  plan, video retention manifest, machine audit builder, and artifact tests.
- Scientific boundary: the selected 2x2 explains action mechanisms; rankings
  and coverage claims come only from the frozen paired report.
- Final verification: 116/116 full tests and 9/9 focused artifact tests pass;
  964 JSON files parse; all source/tests compile; all four retained MP4s are
  dynamic and nonblank; the active deck has 47 PDF/PPTX slides, 29 unique local
  media targets, zero missing links, and zero entropy/six-grid/race references.
- Visually inspected the baseline table, random-C frozen benchmark and 2x2
  video slide after export; the random panel is consistently relabeled C.

## 2026-07-15 Clear-Fingerprint Complementary Four-Method Case

- Filtered the frozen 36-pose audit for Proposed/A capture, Baseline B
  first-attempt capture, random-C failure and zero hard stops, then ranked by
  descending Proposed transport distance.
- Added a stricter no-render full-loop preflight in that frozen order. Rejected
  `sobol_020` (`bottom_not_verified`), `sobol_026` (`funnel_safe_abort`),
  `sobol_007` (A terminal mismatch), and `sobol_004` (`funnel_safe_abort`)
  before selecting `sobol_019`, the first case where P/A/B all pass the same
  CAD-bottom verifier.
- Re-ran all four methods with the same start, contact nuisance, pre-hardware
  F/T noise seed and shared post-capture safety layer. Proposed transports
  2.887 mm and completes in 17.738 s; A completes in 63.178 s; B has 2.068 deg
  control-direction error, positive 0.084 mm direct-ray margin and completes on
  the first attempt in 9.430 s; random-C exhausts its frozen search budget.
- Generated
  `outputs/current/active_comparison/four_methods_clear_fingerprint/`
  with one 2x2 video, a dynamic keyframe and a manifest. Temporary component
  videos were removed after composition.
- Kept the paper-facing name Baseline C for random. This selected clip is a
  complementary mechanism illustration, not an independent benchmark; method
  claims remain tied to the frozen 36-pose paired audit.
- Final verification: 117/117 tests pass; 965 JSON files parse; all source and
  test Python files compile; all five retained/downgraded MP4 files have
  nonblank, changing first/middle/last frames; the 48-page PDF and 48-slide
  PPTX export have 29 local media references, 28 unique targets and zero
  missing files.

## 2026-07-15 Physical-Funnel Video Readability Upgrade

- Re-ran the audit-selected paired `funnel_02` MuJoCo case without changing
  its physics: the unshielded branch still hard-stops at 24.34 N / 133.90 mNm,
  while tip-RCC plus the funnel shield stays at 8.72 N / 35.06 mNm and performs
  a bounded `cad_depth_overshoot` safe abort.
- Rebuilt the active comparison video with synchronized force/torque traces,
  explicit 20 N / 60 mNm hard limits, large hard-stop/safe-abort state badges,
  and 0.5x playback around the post-capture divergence. The plotted curves are
  read directly from the same rollout CSV traces.
- Replaced the summary-mask poster with a dynamic divergence keyframe and
  promoted it on the physical-funnel slide. The scientific boundary remains
  damage limiting, not completion-rate improvement or a passivity proof.
- Temporary component MP4s are deleted after composition; the active path and
  retention count remain unchanged.
- Added `src/embed_funnel_guard_pptx.py` and post-processed slide 34 after the
  Marp export. The PPTX now contains the actual MP4 as `ppt/media/media1.mp4`
  with both media and video relationships, rather than only a raster poster.
- Final verification: 117/117 tests pass; all 965 JSON artifacts parse; source
  and tests compile; the 48-page PDF and 48-slide PPTX are structurally valid.

## 2026-07-15 Extreme Tilted-Wedge Recovery Demonstration

- Added a paired post-capture residual-pose disturbance to the real MuJoCo
  rollout path, including commanded/actual tilt, lateral TCP bias, recovery
  reason and final upright error in the machine-readable result.
- Froze one deliberately severe stress case: 8 deg commanded tilt plus 0.6 mm
  lateral TCP bias, compared with the M8 deep-clearance scale of 1.43 deg.
- Without the physical funnel, the tool retains 8.85 deg final tilt and
  hard-stops at 29.24 N / 128.59 mN m. With tip-RCC plus one bounded
  unload/retract/upright/retouch recovery, it reaches CAD bottom at 0.01 deg
  final tilt with 10.78 N / 38.68 mN m peak wrench.
- Generated the active side-by-side video, dynamic keyframe, synchronized
  traces, manifest and evaluation report under
  `outputs/current/active_comparison/funnel_extreme_recovery/` and
  `outputs/current/evaluations/m8_extreme_funnel_recovery/`.
- Kept the claim narrow: this is a selected mechanism stress test, not a
  measured iiwa fault distribution, strict passivity proof, or general
  completion-rate result.
- Final verification after Markdown compaction: 119/119 tests pass; 967 active
  JSON artifacts parse; source/tests compile; the 137-frame 1280x720
  comparison has distinct first/middle/last frames; the deck has 47
  Markdown/PDF/PPTX slides and zero missing local media; both funnel MP4s are
  embedded on slides 32/33 with bytewise SHA-256 equality to their sources.

## 2026-07-15 Markdown Video-Link and Funnel-Section Compaction

- Made all four active comparison videos explicit in the Markdown with visible
  `play` links; the image hyperlinks remain as a second entry path.
- Removed the physical-funnel cue table because the global action-value routing
  slide already owns that logic.
- Merged admittance, tip-RCC, energy/progress supervision and the one-shot
  recovery sequence into one control-mechanism slide. The distribution-level
  defense result and the two paired videos remain separate because they answer
  different evidence questions.
- Compacted the synchronized Markdown/PDF/PPTX deck from 49 to 47 slides,
  regenerated downstream page numbers and re-embedded the two physical-funnel
  MP4s on slides 32 and 33.

## 2026-07-15 M8 Entropy-Grid Planar Closed-Loop Explainer

- Reframed the Proposed+ criterion from guaranteed success to a reasonable
  closed-loop contract: observations must affect belief/action ordering or
  trigger a named capture, unsafe, exhausted-utility, or time-budget exit.
- Added `src/render_m8_entropy_grid_planar_explainer.py` and generated an
  auditable two-scale M8 top view plus a 150-frame MP4 from the frozen
  `sobol_002` entropy-grid record. The map shows the initial tool position,
  all 673 candidate nodes, the two nominal nodes inside the conservative
  0.25 mm capture disk, the six visited actions, and the measured compliant
  capture event. True center is offline annotation only.
- Added `src/insert_entropy_grid_slides_pptx.py`; inserted the map and embedded
  explainer MP4 as slides 24/25 of the active deck and shifted the two existing
  physical-funnel video pages to 34/35 without changing their media.
- Updated the coverage slide to state the finite-domain/finite-budget tube
  condition and made 130 s active time explicit in the M8 benchmark claim.
- Moved pure entropy-grid out of the paper-facing baseline ladder and used it
  only to explain the adaptive-coverage action primitive from which Proposed+
  follows: conical posterior shortcut, segment capture probability first,
  information gain for ambiguity, and miss/contact-driven replanning.
- Artifacts:
  `outputs/current/active_comparison/entropy_grid_planar_explainer/`,
  `docs/PROPOSED_PLUS_ENTROPY_POLICY.md`, and
  `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.{md,pptx,pdf}`.
- Final verification: the new direct artifact checks pass; 101 source/test
  Python files compile; all 968 active JSON files parse; the Markdown/PDF/PPTX
  each contain 49 slides/pages; 37 local deck references resolve to 31 unique
  existing targets; the new MP4 has 150 nonblank/changing 1280x720 frames; and
  all three embedded PPTX MP4 blobs are byte-identical to their retained source
  videos on slides 25, 34, and 35. The environment currently has no installed
  `pytest` module, so the repository-wide pytest runner was not available.

## 2026-07-15 Strict Complete-Loop Closure and Deck Alignment

- Replaced the capture-only interpretation with one strict contract:
  sensor-only handoff, offline geometry validation, dual-channel CAD-bottom
  dwell, no hard stop, accumulated active-controller time no greater than
  130 s, and zero budget extension. A 0.50 s inactive display tail is frozen
  out of the outcome and retained only as a raw-physics diagnostic.
- Enforced one post-step measured sensor update per physics step for both the
  controller and terminal verifier. Raw undelayed physical wrench independently
  drives the fast hard stop, so delay/filtering cannot conceal an impact.
- Made Proposed+ a conditional scheduler rather than a success guarantee:
  nominal time is reserved before shortcut/grid/commit, measured-path misses
  update belief once, and the policy can switch to global outward or end with a
  named safe failure. The 130 s hard budget cannot be extended.
- Re-ran all seven methods from scratch on the frozen 24 M8 poses. Strict
  CAD-bottom successes were Proposed/P+/Outward/Entropy-grid/Direct-first/
  Random/Oracle = 19/19/19/22/15/10/22; offline captures were
  23/23/23/23/18/12/24. Across the 18 common Proposed/P+ successes, P+ reduced
  median measured path, contact path and wear proxy but was 0.403 s slower;
  each method had one exclusive success and McNemar p = 1.0. This is a tradeoff,
  not evidence of Proposed+ superiority.
- Certified the finite M8 visual disk with a 749-node padded lattice: spacing
  0.3359 mm and analytic/dense worst support-to-node gap 0.2375 mm below the
  conservative 0.25 mm effective capture radius. This is a conditional geometry
  certificate only; measured tracking, dwell, scheduling, safety and CAD-bottom
  remain independent failure modes.
- Generated a strict Proposed+ explainer from frozen `sobol_008`, including the
  749-node planar map, measured compliant path and embedded MP4. The selected
  example captures at 11.944 s and reaches low-wrench CAD-bottom at 16.209 s;
  it explains the causal loop but does not replace the 24-pose audit.
- Rebuilt the synchronized 50-slide Markdown/PPTX/PDF deck. The main results
  now keep entropy-grid as a strong transparent diagnostic, separate historical
  36-pose capture-only evidence from current strict evidence, distinguish
  capture from bottom verification, and state that even privileged Oracle is
  only 22/24. All 126 strict successes in this batch used the low-wrench depth
  branch; the signed axial-force branch is implemented and unit-tested but not
  batch exercised.
- Current evidence paths are
  `outputs/current/evaluations/m8_strict_24_complete_loop/`,
  `outputs/current/evaluations/m8_entropy_grid_coverage_certificate/`,
  `outputs/current/active_comparison/proposed_plus_strict_explainer/`, and
  `outputs/current/media/figures/paper/m8_strict_complete_loop.png`.
- Synchronized the older topic documents with the same evidence boundary:
  A/B/C labels are historical only, simulator-reset direct/retry is excluded
  from strict evidence, entropy-grid is a named strong diagnostic, M10/M12 S1
  is bounded direction evidence while S2 fails, and no guard is described as a
  passivity or safety guarantee.
- Registered the strict Proposed+ MP4 in `VIDEO_RETENTION.json` and replaced the
  stale machine-audit compatibility assertion with direct checks of strict
  terminal modes, sensor causality, terminal freezing and all seven success
  counts.
- Final repository verification passes 140/140 unit tests; all 1,025 JSON files
  under `outputs/current/` parse, and all source/test Python files compile. An
  independent readback confirms 50 Markdown/PPTX/PDF slides/pages, 32 unique
  local references with zero missing files, and embedded videos only on slides
  26/35/36 with source/manifest/PPT byte-identical SHA-256 values.
- The deck build manifest now records byte size and SHA-256 for the Markdown,
  PPTX and PDF as well as each video; the generated PPTX/PDF are group-readable
  (`0644`).

## 2026-07-15 Strict Comparison Matrix and 35-Slide Logic Closure

- Re-audited the 50-slide deck as a scientific argument rather than a project
  diary. Removed section dividers and repeated ownership/data/scale/sim-to-real
  explanations, then rebuilt it as 30 main-talk slides plus five question-only
  backups. The main order is now task and observability -> fingerprint action
  and evidence -> direct-ray insufficiency -> belief and bounded coverage ->
  method roles -> selected mechanism videos -> frozen strict aggregate -> named
  failure exits -> physical funnel -> cross-size audit -> hardware gates and
  claim boundary.
- Kept the historical 36-pose four/six-method matrices out of the current
  evidence chain because they use capture-only or reset/retry-era contracts.
  Generated a new same-start strict 2x2 at
  `outputs/current/active_comparison/strict_four_methods/` from frozen
  `sobol_008`: Proposed, Proposed+, outward and entropy-grid all use the final
  130 s, zero-extension, sensor-handoff and CAD-bottom contract. Every rendered
  result field matches the frozen trial; the selected clip preserves the fact
  that entropy-grid is fastest on this pose and is explicitly not a ranking.
- Put the strict matrix on main slide 17 between method/coverage definitions
  and the frozen 24-pose aggregate. Slide 16 retains the Proposed+ measured path
  and causal loop; slide 18 replaces the dense result page with a compact plot
  that separates offline capture from strict CAD-bottom and shows the exact
  Proposed/Proposed+ pairing (`p=1.0`, no superiority).
- Tightened the coverage statement on slide 15: the 749-node certificate proves
  only padded-lattice node geometry under the declared CAD-derived capture
  radius. Visiting the lattice, measured tracking/cue validity, finite-time
  scheduling, safety and terminal completion remain separate conditions. The
  Proposed+ nominal reserve and the non-extendable 130 s watchdog are stated as
  a scheduling contract, not an eventual-success guarantee.
- Updated the deck builder to export the poster-only Marp PPTX to PDF before
  embedding H.264, avoiding a headless LibreOffice codec crash. Movie boxes are
  now aspect-fitted to their poster regions so the slide-16 video cannot cover
  the neighboring planar map and the 1280x800 strict matrix is not distorted.
- Added direct artifact tests for the strict matrix and the synchronized deck.
  Independent final readback passes 145/145 tests, compiles all 114 Python
  source/test files, parses all 1,026 JSON files under `outputs/current`, and
  confirms 35 Markdown/PPTX/PDF pages, 27 local references with zero missing,
  and source/manifest/PPT byte-identical videos only on slides 16/17/22/34.

## 2026-07-15 Two Fingerprint Matrices and Jin-Style Paired Audit

- Replaced the single strict entropy-grid video page in the group-meeting deck
  with two same-contract 2x2 matrices, each ordered Proposed+, Proposed, pure
  Outward and fingerprint Direct-first. Frozen `sobol_016` is the muted
  posterior/short-shortcut case and its one direct ray misses safely; frozen
  `sobol_000` is the clear posterior/long-shortcut case and its one direct ray
  reaches CAD-bottom. Every substantive rerendered result field matches the
  source strict trial; only video path/frame count are declared renderer
  outputs.
- Kept the selected-case claim narrow. The strict-24 post-hoc descriptive
  strata are clear/long 5/5 versus fuzzy/short 2/6 direct-first successes, but
  are not presented as preregistered probability estimates. Distribution-level
  method results remain on the frozen aggregate pages.
- Implemented and preregistered a Jin-2021-style M8 proxy before reading fresh
  results: paper-style `Xz/Mx/My` polar binary input, 2000 samples, 20-sample
  smoothing, `3x20x20` input and 2Conv--2Pool--1FC. Unspecified hex labels,
  rasterization, CNN widths, optimizer and direct controller are explicitly
  frozen proxy choices, so this is not called a faithful paper reproduction.
- Generated a fresh P1v6 shared-scan direction stress set (180/180) and 48 new
  paired strict direct-first poses. All paired fingerprint SHA values match and
  neither method violates the 130 s budget. Improved versus proxy first-ray
  capture is 37/48 versus 24/48 (`p=0.00443`); strict CAD-bottom is 32/48 versus
  18/48 (`p=0.00131`). P1v6 direction metrics are mixed, so the claim is higher
  task success against this proxy, not uniformly higher direction accuracy.
- Expanded the deck from 35 to 37 pages: 32 main-talk pages and five backups.
  The new estimator comparison follows the frozen direction-evidence page; the
  two matrices replace the old matrix page and precede the strict aggregate.

## 2026-07-15 Final Semantic, Coverage, and Artifact Readback

- Recomputed the paired estimator direction endpoint from the direction that
  the controller actually executed (`control_direction_deg`), rather than the
  improved model's top-bin diagnostic. On strict-48, improved versus the
  Jin-style proxy is median 3.34 degrees versus 8.18 degrees and p90 20.55
  degrees versus 17.36 degrees. The direction result is therefore mixed even
  though first-ray capture and strict CAD-bottom are significantly higher for
  the improved arm. No frozen trial or simulation outcome was changed.
- Preserved the structured P1v6 result as mixed: the improved arm has lower
  median error and higher within-15-degree rate, while the proxy has better p90
  and within-7.5-degree rate. The base-pose cluster-bootstrap median-difference
  interval includes zero. The Jin arm remains a paper-inspired M8 proxy, not a
  faithful reproduction or a cross-task comparison to the paper's published
  square-hole accuracy.
- Made the selected-case split explicitly post-hoc and partial. The clear/long
  and fuzzy/short extreme strata contain only 11 of the 24 frozen strict poses;
  the other 13 are unassigned. The 5/5 and 2/6 direct-first outcomes explain
  the two videos but are not probability estimates or general guarantees.
- Added a deterministic finite-Outward boundary audit. Although the 0.4
  mm/revolution pitch is no larger than twice the 0.25 mm effective capture
  radius, the unpadded finite spiral misses 65,535 of 65,536 sampled outer
  boundary points, with a 0.642053 mm worst nearest-path gap. Outward is now
  described only as an empirical dense-search comparator. The padded 749-node
  entropy-grid remains the sole current finite-domain node-geometry
  certificate; it does not certify tracking, cue validity, time, safety, or
  terminal completion.
- Kept the 130 s deadline in every current comparison and named all non-success
  exits. In the strict-24 batch Outward has zero timeout/search-exhaustion; its
  five non-successes are four capture-without-bottom cases and one false
  handoff, so the deck no longer attributes them to a spiral timing failure.
- Recorded the post-completion correction as a derived-only provenance
  amendment. The frozen 48-trial set and task contract remain unchanged; the
  final audit manifest and regenerated report/figure form the current derived
  evidence chain.
- Rebuilt and visually read back the synchronized 37-page deck: 30 local
  references resolve with none missing, and source/manifest/PPT video bytes
  match on slides 17, 18, 19, 24, and 36. Final SHA-256 values are Markdown
  `6cd4400f95c6001c4310e7e116ed3e0e998965d312c53d2be80246ff857290f8`,
  PPTX `0c575edb89fa874a7acb3b4c1a7bfd24c3f41dcee1773abe6d88b51a3309dd31`,
  and PDF `e84e61702fce09f992d60f37c4891691d61d993316058ae2246e07650a715c32`.
- Final verification passes 186/186 unit tests; all 128 Python files under
  `src/` and `tests/` compile, and all 1,092 JSON files under
  `outputs/current/` parse successfully.

## 2026-07-15 Cross-Size S4 Contract-Only Follow-Up

- Read the fresh S3 v3 failures trial by trial before changing the model. M10
  fine-tuned failures split into one no-capture 130 s direction/search timeout
  and one post-capture raw-wrench hard stop. M12 split into one catastrophic
  direction failure, one correct-MAP but multimodal/no-shortcut fallback-reset
  hard stop, and one post-capture hard stop. Direction-head adaptation alone
  therefore cannot explain or repair all exits.
- Identified two data/model mismatches without reusing S3 trials for fitting:
  only 36 train scans updated 6,192 direction-head parameters per seed, and
  half of the old train/calibration scans lay outside the S3 deployment
  annulus. Pooled calibration medians also hid double-overhang p90 errors above
  100 degrees; the frozen state head classified only 9/12 untouched initial
  states correctly for each cross size.
- Added `sample_cross_size_s4_adaptation.py`. Its default action writes only a
  hashed contract; MuJoCo sampling requires explicit `--sample`. Each size has
  five deployment-matched rings and disjoint complete C6 roles: 60 train, 30
  validation, and 30 calibration observed-wrench scans. S3 v3 and all older
  cross-size evaluations are explicitly forbidden as adaptation data.
- Added `finetune_cross_size_s4_state_direction_heads.py`. It freezes the
  encoder/shared trunk, permits only state/direction heads to update, selects
  epochs on validation catastrophic circular-control error and worst-state
  p90, and reserves calibration for state/direction temperature, APS, and
  descriptive reporting. Reports expose pooled and per-state MAP versus
  circular-control errors so a correct top bin cannot hide a multimodal action
  direction failure.
- Added optional state-temperature readback to `M8DirectionEnsemble`; historical
  models default to 1.0 and retain the existing public model tuple. Added six
  focused tests for exact rings, C6 role isolation, observation matching,
  contract-only no-sampling behavior, 6,579 trainable head parameters, and the
  MAP-correct/circular-mean-wrong counterexample. The new S4 manifests remain
  `contract_only`; no S4 scan, checkpoint, full-loop trial, or success claim
  has been generated.

## 2026-07-15 Cross-Size S4 Frozen Full-Loop Protocol

- After the separately completed 240/240 S4 dynamic scans, extended
  `finetune_cross_size_s4_state_direction_heads.py` with a genuine
  `cal_only` control. Its checkpoint is copied byte-for-byte from the frozen
  M8 base member; only S4 calibration-role state/direction temperatures and
  ensemble APS are re-estimated. The S4 fine-tuned arm still updates only the
  387-parameter state head and 6,192-parameter direction head, selects its
  epoch on validation, and calibrates only after selection.
- Added `run_cross_size_s4_full_loop_evaluation.py` with physically separate
  `development` and `untouched` phases. Development reuses the old S3-v3
  tasks and may guide tuning. Untouched uses fresh seeds 20261141/20261142,
  a distinct evaluation directory, and a contract that forbids all later
  training, calibration, controller/model selection, or gate selection.
- The runner defaults to freeze-only and requires a second explicit
  `--execute` invocation against the identical execution-bundle hash. Every
  estimator variant uses Proposed+ `hybrid`; pure outward remains the declared
  nonlearned baseline and straight-to-center remains privileged. All arms
  receive the same size-conditioned commissioning, R3 funnel, unchanged raw
  hard-wrench vector, 130 s active budget, and zero extension.
- Frozen per-size gates are oracle strict-bottom rate at least 0.90, S4 FT
  strict-bottom rate at least 0.80, zero FT hard stops, and zero FT circular
  control errors above 90 degrees. Necessity can be supported only on the
  untouched phase when FT and oracle pass while both zero-shot and a legal
  weight-identical calibration-only arm fail the same gate. Missing cal-only
  remains explicitly inconclusive.
- Added resume-task/hash validation, per-worker execution-bundle readback,
  learned first-probe hash comparison, offline-terminal boundary, provenance,
  CSV, PNG, Chinese PPT notes, English PPT notes, and focused contract tests.
  No S4 head training or full-loop rollout was launched in this implementation
  step.

## 2026-07-16 Sobol-016 Proposed+ Physics Audit

- Audited the selected fuzzy 2x2 video against the frozen `sobol_016` trial
  rather than inferring motion from the yellow command markers. The initial
  radius is 3.297541 mm. The first executed shortcut is 0.903266 mm and ends at
  radius 2.731753 mm; it does not capture. The actual control direction is
  5.043721 degrees versus an offline true-center direction of 49.840984
  degrees, an error of 44.797263 degrees.
- The later 0.542486 mm field is an unexecuted belief replan, not a second
  shortcut. Four belief updates increase entropy from 6.064784 to 6.448268 nat
  and credible radius from 3.930578 to 4.539899 mm. The correct mechanism is
  therefore a risk-bounded short shortcut followed by capture-weighted finite
  coverage, not rapid entropy convergence.
- An online sensor cue preempts probe 5 at 15.4475 s and the shared physical
  funnel verifies CAD bottom at 22.1925 s with 0.007303 mm depth error and no
  hard stop. True center and contact tags remain offline audit quantities.
- Added the bilingual audit figures, machine-readable report, explanatory
  note and focused test under
  `outputs/current/diagnostics/m8_sobol016_proposed_plus_physics_audit/`,
  `outputs/current/media/figures/paper/`, and
  `docs/SOBOL016_PROPOSED_PLUS_PHYSICS_AUDIT.md`.

## 2026-07-16 Bumpless Handoff and Targeted Mechanism Falsification

- Replaced discontinuous mode ownership with measured-pose atomic rebasing of
  outer admittance and the inner held-command history at capture release,
  recovery ownership, insertion handoff, terminal-set entry and oracle center
  settle. Recovery now restarts descent from measured z and phase zero; oracle
  transport closes the measured residual smoothly before dwell.
- Added low-force, sensor-only post-release requalification. A pre-release cue
  may no longer authorize insertion after controller handoff; a lost cue gives
  a named safe abort. Raw 20 N / 60 mN m limits, soft thresholds and terminal
  verifier thresholds were not relaxed.
- Froze and executed one development-only 10-arm targeted matrix with contract
  SHA-256
  `48d7f43f4d81cfb0042e0d1745537b30b878abe7e8861ef08e0055829187c994`,
  execution bundle
  `b3a78aafe0eda5cffb693690ed23af942df33a835157ec61e5c8760abfa7bb28`
  and run ID `9f6ae16338084182ba47806d0e6c22ee`. It produced seven strict
  bottoms, three named safe aborts and zero hard stops.
- The targeted set validates the handoff, oracle settle and false-capture
  rejection mechanisms. It does not validate the post-recovery phase-zero
  restart branch: M12-010 lost its capture cue at requalification before that
  branch could execute. This distinction is retained in the paper/deck claim
  boundary.
- Corrected requalification telemetry so a pre-existing terminal abort such as
  `cad_depth_overshoot` is marked as skipped with its original reason, rather
  than being relabelled as a requalification failure.

## 2026-07-16 Extreme-Fault Artifact Correction

- Re-ran the historical 8-degree/0.6 mm paired stress clip after the bumpless
  controller change. The old manifest claiming one full recovery and CAD-bottom
  success was stale and was not preserved as current evidence.
- The current unsupervised branch returns `bottom_not_verified`; on the
  explicitly labelled inactive display step it later reaches 28.161 N /
  131.950 mN m. The primary terminal outcome remains distinct from this late
  display-step violation.
- The supervised branch detects an unrecoverable `cad_depth_overshoot`, performs
  bounded unload and returns `funnel_safe_abort` with no primary or post-event
  hard stop. Raw peaks are 12.621 N / 31.209 mN m and recovery count is zero.
- Regenerated the comparison MP4, keyframe, traces, manifest and evaluation
  report consistently. This clip now supports only failure-consequence and
  safe-rejection mechanics; the fixed 12-case R0/R3 ablation remains the
  completion-rate evidence.

## 2026-07-16 S4b Development-v5 and First Untouched-v1 Full Loop

- Frozen development-v5 used task-contract SHA-256
  `4ce62d410bc802ebe45d840dafeffed9837694360321adfc3f833bcb4b5d5b7f`,
  execution bundle
  `b10befd13b3ecc17f87fa22591135253a74e4c9532fa03286e0e4de3901308f3`
  and run ID `2cb77b51a0fe44d6b6d1ac63bb256527`. M10 fine-tuned
  reached 12/12 strict bottom with zero hard stops; M12 reached 10/12 with zero
  hard stops. Both development gates pass, but zero-shot/calibration-only also
  pass the relevant same-task gates, so necessity is false.
- After freezing all development choices, untouched-v1 was generated from the
  separate fixed Sobol seeds and executed exactly once. Its task-contract
  SHA-256 is
  `ad64b03372653fbdf7fd838d6a09bbd352346435be78315a6f9fcb61111dd724`,
  execution bundle is
  `a32d7ab6a72840254190ada55a712e475e0808bc1ad180e9d05d252ef7ca84dd`
  and run ID is `2401daf1df1448319946fb2404fc3ec6`.
- Untouched M10 fine-tuned reaches 10/12 but has one primary hard stop and one
  requalification safe abort, so it fails the zero-hard-stop gate. Untouched
  M12 fine-tuned reaches 10/12 with two requalification safe aborts and zero
  hard stops, so it passes. Overall two-size closure is false.
- The M10 hard stop is pose 006 during contact-loaded `fallback_reset`: raw
  torque norm reaches 63.726 mN m and the latched axis is `Ty>60 mN m`.
  `result=hard_stop`, `terminal_attempts=0`, and
  `post_terminal_event_display_new_hard_stop=false`, so this is a primary
  rollout failure. Pose 010 is a 0/100 low-force requalification safe abort.
- M12 pose 005 executes a complete tilt recovery, then correctly aborts when
  requalification remains 0/100. Pose 010 correctly abstains because the
  fine-tuned concentration is 0.307915 below the selected 0.60 floor; a later
  global-outward chamfer cue is rejected at 0/100. Recovery and abstention
  therefore change failure consequence but do not guarantee capture.
- Untouched M10 zero-shot/calibration-only both reach 12/12 with zero hard
  stops; M12 calibration-only reaches 11/12. Fine-tuning sufficiency across
  both sizes and fine-tuning necessity are not established. Untouched-v1 is
  permanently test-only and forbidden for future training, calibration,
  controller/gate selection or a tuned confirmation rerun.
- All 24 trials in each phase share one run ID, contain no worker errors, use
  zero simulator resets and zero budget extensions, preserve learned
  first-probe hashes, and match worker-init/worker-completion/parent execution
  bundle hashes. These are MuJoCo results with offline strict scoring, not real
  F/T evidence or a hardware-data fine-tuning necessity result.

## 2026-07-16 Final Bilingual-Deck and RA-L-Source Verification

- Synchronized the Chinese and English Marp decks at 37 slide separators and
  30 existing local media references each. The English source contains no CJK
  text. Both versions now use the same shortcut rule, physical-funnel scope,
  Sobol-016 interpretation, S4b development/untouched results and hardware
  evidence boundary.
- Kept the requested two 2-by-2 comparison matrices and the M8 entropy-grid
  explanation, while removing claims that entropy decreases monotonically,
  that recovery guarantees success, or that fine-tuning has established
  cross-size necessity.
- Aligned the RA-L LaTeX source with the frozen reports. Static validation found
  31 unique labels with no missing references, no missing citations, eight
  existing static figures, no CJK text and no moving-media references. The
  source remains an eight-page-format research draft whose final pagination
  must be checked after compilation.
- Revalidated all 2,949 JSON files below `outputs/current`; both S4b phase
  manifests are complete and retain their frozen task-contract and execution-
  bundle hashes. Full unit discovery passes 317 tests with two conditional
  skips, and `compileall` passes for `src` and `tests`.
- Added the frozen Sobol-016 physics audit and ten-arm bumpless-transition
  regression to the active-evidence registry. The latter remains explicitly a
  development mechanism test, not a distributional success-rate estimate.
- No Marp PDF/PPT or LaTeX PDF was generated, as requested. No TeX compiler is
  installed in the current environment, so visual pagination remains a local
  LaTeX Workshop/Marp compilation check rather than a claimed verified result.

## 2026-07-16 RA-L LaTeX Toolchain and Verified PDF Build

- After the author reported that LaTeX Workshop could not build the paper,
  confirmed that the remote environment had no `latexmk`, `pdflatex`, `bibtex`,
  or `kpsewhich`; no compiler log had been produced.
- Avoided a system TeX Live install because sudo requires a password and the
  root filesystem is already at 98% usage. Installed the SHA-256-verified
  user-local TinyTeX 2026 distribution under `/home/njy/.TinyTeX` and exposed
  its build commands through `/home/njy/.local/bin`.
- Installed the missing `ieeetran` BibTeX package, then completed the full
  `latexmk`/BibTeX/multi-pass build. `main.pdf` is eight US-Letter pages,
  contains no LaTeX error or unresolved citation/reference, embeds Type 1
  fonts, and contains no Type 3 font.
- Reflowed only display equations and the long requalification identifier;
  no definition, threshold, experimental result, or claim changed. Horizontal
  overflow warnings fell from 14 (maximum 130.58 pt) to three (maximum 8.31 pt),
  and visual inspection found no cross-column overlap.
- Current PDF SHA-256:
  `3c713d42ca230fa43d1d568f0d3e68fb56413fa892b814a8dedd81e779c4a230`.

## 2026-07-16 English Marp PPTX with Embedded Videos

- Exported the 36-slide English Marp source to
  `group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709_en.pptx`
  using Marp CLI 4.4.1 and Chrome for Testing.
- Added native embedded PowerPoint movie shapes on slides 11, 15, 20 and 21.
  Their four H.264 MP4 payloads are stored inside the PPTX package and are
  byte-identical to the retained source videos; no original Linux media path is
  required for playback.
- Added the repeatable builder
  `group_meeting_trace/build_unscrewing_contact_state_pipeline_en_pptx.py`.
  It locates each poster border from the actual Marp-rendered slide background,
  preventing preview-resolution coordinate drift before inserting the movie.
- Final structural readback: 36 slides, four media and four video
  relationships, four movie timing nodes, all 160 XML/relationship parts
  parse, and the PPTX ZIP has no corrupt member. A headless LibreOffice
  readback also opens all 36 pages, but Microsoft PowerPoint desktop remains
  the target player for the embedded H.264 objects.
- Final PPTX SHA-256:
  `73a76f9d99e4df44031bb6f67e490dc3ab837e88b41d4cd08cc64fd613bb785e`.

## 2026-07-16 TASTE-Style Paper and Bilingual Presentation Consolidation

- Re-read the local TASTE skill and both style anchors, then rebuilt the paper
  and both decks around one mechanism-level hinge: conical fingerprint ->
  circular/contact posterior -> posterior-supported shortcut -> explicit
  residual set -> compliant capture/recovery/verification. Internal run names,
  raw state-machine keys and audit-report prose were removed from the paper's
  scientific narrative.
- Rewrote the IEEE paper as
  `group_meeting_trace/unscrewing_contact_funnels_ral_latex/main.tex` with the
  title `Posterior-Guided Tactile Shortcut Search for Bounded-Time Hex-Socket
  Engagement`. Added the direct-ray miss geometry, circular posterior, planar
  center belief, lower-quantile shortcut proposition, residual-search utility,
  admittance/recovery law, cross-size action gate, paired protocol ranges, and
  non-overclaiming statistical comparisons.
- Replaced the deferred double-column cross-size float that created an isolated
  eighth page with a legible single-column table. The verified build is seven
  substantive US-Letter pages, within the RA-L eight-page limit, with zero
  LaTeX errors, unresolved references/citations or overfull boxes and Type 1
  fonts only. Final hashes are
  `d4e78c6f859b5f4f598f31a4830bbd1177462087afa0752111ee128252e397c0`
  for `main.tex` and
  `366dd3e652176c8dbf92d67b36f6ece454c41c26557fc3b32be20444cbe97fda`
  for `main.pdf`.
- Reordered the Chinese and English Marp decks into the same 35-page TASTE
  sequence: physical failure, three-funnel decomposition, active probe,
  four-channel posterior, shortcut derivation, residual coverage, physical
  recovery, paired M8 evidence, bounded-time definition, cross-size transfer,
  hardware protocol, contribution, and claim boundary. The 31-slide main talk
  is followed by four backup slides; the English source contains no CJK text.
- Preserved the requested native videos on slides 10, 14, 19 and 20: the
  improved-vs-Jin three-ray comparison, measured-tip entropy-grid top view,
  fuzzy 2-by-2 matrix, and clear 2-by-2 matrix. Both PPTX packages contain
  exactly four H.264 payloads and only these four slide relationships.
- Added an academic cross-size result renderer to
  `src/plot_m8_bilingual_core_paper_assets.py`. It reads the existing held-out
  report without rerunning simulation and renders zero-shot, calibration-only,
  and adapted-head verified engagements with force-limit/protected outcomes;
  internal `S4b/untouched/commissioned` labels no longer appear in the main
  result graphic.
- Regenerated the bilingual core assets and synchronized Markdown, PPTX, PDF,
  build manifests, paper README, and the 35-page logic audit. All 30 local
  references per deck resolve, both PDFs contain 35 16:9 pages, both PPTX ZIPs
  validate, and final visual inspection covered the modified direction,
  recovery, shortcut, adaptation, and cross-size pages.
- Final deck hashes are CN Markdown/PPTX/PDF
  `a46d541b796ef3e28ac7c81dc9f17b14d98357a19d4999be086dcb228a0e32b5`,
  `1d19162de129ff7090e9fb44e36a5ab9c49b08da0dd8b965897ec3497663a31f`,
  `f5495c77b67765d02c6f234e6efb52db67bddd4fab3d9848bc36ca8d0bbaa5b2`;
  EN Markdown/PPTX/PDF
  `be515a4bb5e0aeaf26dfdf7c9c32539205511ae47c17aad8a0cfe855e4892434`,
  `eafaf15cd1df072a5347054e7af41bbe68cc8c8e848f4b4064a7350634c8da3b`,
  `a3f173faf2b9c0849fd1defeed3e2e3d54f13ddd2d15de418ca338cba2496076`.
- The final scientific boundary remains simulation-only conditioned geometric
  engagement. Proposed+ is not superior to Proposed on the paired M8 batch;
  the improved four-channel direct interface is mixed against the Jin-style
  proxy; cross-size fine-tuning does not dominate legal controls; hardware,
  passivity, torque transfer, thread release and extraction remain unmeasured.

## 2026-07-17 ATI Tooling CAD v1 and Explicit Wrench Reference

- Parsed the supplied `fast_mount_ATI_ft_1.STEP` as a single 8 mm flange with
  4 x 6.6 mm holes on PCD 88, then built a parameterized receiver while
  preserving the sensor-side mounting interface.
- Chose a shoulder-datum, AF16 short straight pilot, and 3 x M4 symmetric axial
  preload rather than pure FDM interference or a self-locking taper. Generated
  equal-datum M8/M10/M12 cartridges with a common 42 mm blue sensor-plane-to-
  tip distance (34 mm from the blue tool-side face).
- Generated a strict three-socket base with head replicas at X=-45/0/+45 mm,
  common mouth Z=0, common yaw, a 0.5 mm center pilot/through-hole, fixture
  datums, labels, and fit coupons. Nominal and explicitly unvalidated FDM-trial
  variants are separate.
- Exported STEP, STL and millimetre-scale 3MF files plus M8/M10/M12 assembly
  STEP files under `real_device/model/parametric_tooling/generated/`. All STEP
  parts re-import as valid single solids and all ten printable STLs pass
  watertight/manifold/orientable/no-self-intersection checks.
- Added `configs/real_tooling_v1.json` and
  `docs/HARDWARE_TOOLING_V1_ALIGNMENT.md`. Known geometry is synchronized;
  ATI-to-blue transform, actual sensor-to-tip vector, tool mass/COM/inertia and
  fixture-to-world transform remain explicit null metrology fields rather than
  fabricated values.
- `MuJoCoAdmittanceRollout` now accepts an explicit calibrated wrench-reference
  distance. The legacy `0.18 * bit_length` reference remains only to reproduce
  prior evidence and is forbidden for the hardware-aligned profile.

## 2026-07-17 ATI Tooling CAD v1.1 Tool-Free Retention

- Removed all three M4 holes, insert pockets, screws, and associated preload
  assumptions after confirming that no M4 locking hardware is available.
- Kept the rigid shoulder as the sole Z/tilt datum and the AF16 pilot as the
  XY/yaw datum. Three short tapered printable ribs now provide a nominal
  0.05 mm local interference, rather than making the whole hex pilot an
  unpredictable FDM press fit.
- Added a separately printable radial U-clip around a reduced cartridge neck.
  The clip has two load paths, flexible detent bumps, and a tether/pull hole;
  it is a positive pull-out stop but does not position the tool.
- Regenerated the receiver, equal-datum cartridges, clip, assemblies, fit
  coupon, STEP/STL/3MF outputs, manifest, and preview. All eleven printable
  meshes pass watertight/manifold/orientable/no-self-intersection checks.
- Exact CAD assembly checks show the same intentional local-rib overlap for
  M8/M10/M12 (0.0115714 mm^3) and zero receiver-to-clip or cartridge-to-clip
  interference at the fully inserted detent position. Physical fit, pull-out
  load, wear, and 20-cycle repeatability remain commissioning measurements.
- Relaxed only the removable clip detent after operator review: bump radius is
  now 0.35 mm and nominal snap engagement is 0.15 mm. The shoulder, AF16
  datum, local rib interference, positive-retention neck, and extraction path
  are unchanged.

## 2026-07-17 ATI Tooling CAD v1.2 Through-Bores and Ribbed Socket Heads

- Found and corrected a final-body Boolean error: the original 0.5 mm center
  cuts passed through each isolated socket head but the later union with the
  common plate refilled their lower segments. The three bores are now cut only
  after the complete fixture is fused. A 0.4 mm axis probe has zero solid
  overlap from each socket bottom through the plate back face.
- Added shallow trapezoidal vertical ribs to the non-top exterior walls:
  M8/M10/M12 use 18/24/30 ribs, 0.40 mm radial height, 0.80-to-0.40 mm root-to-
  tip width, and 0.50 mm axial fade-in/fade-out.
- Preserved a common smooth `z=-2...0 mm` band. Exact B-rep intersection checks
  show zero added or removed volume in that band and 1.92 mm separation from
  the current 0.08 mm MuJoCo top-contact mesh.
- Recorded the claim boundary: current `screw_head_side_visual` is collision-
  disabled, so the ribs do not alter present simulated contact, but the current
  model cannot certify a real fall down the ribbed exterior. Enabling exterior
  collision requires new outer-boundary/global-fallback validation.
- Regenerated nominal and FDM-trial STEP/STL/3MF fixtures and a detailed rib/
  bore preview. All eleven printable meshes remain watertight, manifold,
  orientable, and free of self-intersections.

## 2026-07-18 00:25 +08 - ATI Tooling CAD v1.3 STEP Printing Package

- Corrected the socket-cavity loft so the requested top AF is reached at the
  physical `Z=0` mouth rather than 0.15 mm above the part. The strict
  `socket_base_nominal` STEP now re-imports with M8/M10/M12 wall/top AF
  `6.20/6.50`, `8.24/8.60`, and `10.30/10.74 mm` respectively.
- Preserved that strict file as the research control and added a separately
  named `socket_base_hardware_easy` physical target. Its straight-wall AF is
  `6.30/8.34/10.40 mm`; its 45-degree entry width/depth is
  `0.36/0.48/0.60 mm`; and its mouth AF is `7.02/9.30/11.60 mm`. The mouth
  capture allowance is approximately `2.04/2.17/2.16` times the frozen source
  geometry while deep-wall clearance increases by only 0.05 mm per side.
- Added hardware-easy M8/M10/M12 socket coupons. Each coupon sweeps five total
  wall-AF offsets `[-0.10, 0.00, +0.10, +0.20, +0.30] mm` and retains the full
  size-specific 45-degree entry funnel, so printer compensation is selected
  from a small part before committing to the common base.
- Added the human-readable Chinese printing and acceptance record
  `real_device/model/parametric_tooling/PRINTING_STEP_CHECKLIST_CN.md`. It
  explains every calibration, cartridge, clip, receiver, final base, optional
  control, and non-printable assembly STEP; records quantities and acceptance
  checks; and includes an as-printed measurement-log template.
- The frozen first-print order is: M8 cartridge + quick-change coupon + spare
  U-clips; then complete the three-cartridge set + three socket coupons; only after measured
  compensation, regenerate and print the receiver and hardware-easy base.
  `socket_base_fdm_trial_p0p10mm_per_side` remains explicitly unvalidated and
  is not the selected final file.
- Current design-parameter SHA-256 is
  `28fba56a4c01474097cc1cef28790b392215e572ac9ca4d307173f715cdec34d`;
  generated manifest SHA-256 is
  `a7cd77722cd19f275a4ec973b246fb87dcfeac2d6180a63dee0cb6195e213574`;
  mesh-validation SHA-256 is
  `b7366bb8b370e12074a7b7e0287681da105c51d2f3de41dbb98096dc310e23d7`.
- All 12 independent printable parts re-import as valid single-solid STEP
  geometry, and all 12 meshes pass watertight, edge/vertex-manifold,
  orientable, and no-self-intersection checks. Exported nominal and
  hardware-easy socket-section errors are approximately `2e-7 mm`; all three
  final-body center bores pass the 0.4 mm axis-probe test.
- Claim boundary: this is a CAD-validated printing candidate package, not a
  validated physical tool. The documented hardware-easy funnel is an
  intentional engineering bias. Actual FDM dimensions, shoulder seating,
  clip fatigue/pull-out, mass/COM/inertia, ATI-to-tip transform, stiffness and
  robot force/torque safety remain measured commissioning tasks.

## 2026-07-19 00:34 +08 - Installed Bota Identity Correction

- Corrected the active real-tooling contract after the operator confirmed that the
  installed force/torque sensor is Bota Systems `BFT-SENS-CAT2-B4`, serial
  `SN001349`, rather than ATI. Historical ATI CAD entries above remain unchanged as
  provenance; they are not the current sensor identity.
- Upgraded `configs/real_tooling_v1.json` to schema 2 and recorded the exact Bota
  product/serial, `CANopen_over_EtherCAT`, working driver-config path/hash and
  `observation_logging_only` qualification. Updated stale CAD source paths to the
  present workspace.
- Replaced ATI-specific metrology and COM field names with sensor measurement-origin
  names in the config, loader, tests and alignment document. The 42 mm blue-plane
  datum is still not represented as a Bota sensor-origin lever arm.
- Deliberately kept the measuring-range variant, calibration certificate,
  measurement-origin frame definition, sensor-to-tip transform, mass/COM/inertia and
  fixture-to-robot transform null. `hardware_aligned_profile_ready` remains false.
- JSON parsing, source/config hashes and eight available loader/admittance unit tests
  passed. MuJoCo-dependent tests were not run because this system Python lacks the
  `mujoco` module; `pytest` is also not installed. No ROS node, sensor master or robot
  application was started.

## 2026-07-19 05:08 +08 - Real-device Bridge v1

- Mapped the frozen simulation responsibilities into an explicit MATLAB/KST
  hardware FSM under `../real_device/`: guarded touchdown, dynamic contact
  routing, one conical probe, posterior abstention/capped shortcut, mandatory
  residual coverage, capture handoff, classical insertion funnel, bounded
  recovery, independent terminal verification, measured return and KST end.
- The learning interface remains direction/state posterior only. H3 closes the
  classical task before H4 shadow inference or H5 capped control authority.
- The current H0 implementation is command-free. H1+ remains blocked on exact
  wrench metrology, payload, fixture/WORLD and depth data, task limits, measured
  KST timing, and a portable ONNX contract with MATLAB parity vectors.
- Updated the bilingual 0709 Markdown deck to 38 synchronized slides and fixed
  its audited English-media substitution registry. Historical PPTX/PDF were not
  rebuilt and remain identified as the older 35-slide snapshot.
- Targeted tooling/admittance/funnel/terminal/bilingual tests passed 25/25;
  MATLAB `checkcode`, pure FSM tests and static H0 preflight also passed. No
  KST connection or robot motion was executed by this update.

## 2026-07-21 - Tooling CAD v1.4 after physical-print inspection

- The first FDM print showed that the split cartridge shoulders required
  support and produced severe stringing. The receiver/cartridge/U-clip design
  was therefore retired in favor of three complete one-piece flange tools,
  manually exchanged at the four flange bolts. The full v1.3 source and
  generated artifacts are preserved under
  `real_device/model/parametric_tooling/archive/v1_3_quick_change_2026-07-21/`.
- Each new M8/M10/M12 part is one solid with the unchanged flange and mounting
  holes, a self-supporting `D44 -> D36 -> D16.2 -> AF14` load path, and a common
  physical tip at `Z=-42 mm`. STEP re-import verifies the common datum,
  source-flange overlap, hex orientation, tip section, and shaft AF.
- A second physical observation showed that the source-simulation peg taper
  (`tip_scale=0.98`) was practically invisible after FDM. The selected
  hardware-easy-v2 peg tips use true 45-degree chamfers of
  `0.60/0.80/1.00 mm`, leaving tip-face AF `4.80/6.40/8.00 mm` and full-AF
  straight lengths `12.40/12.20/12.00 mm`. The 42 mm TCP datum is unchanged.
- The recommended fixture is `socket_base_hardware_easy_v2`: deep-wall AF
  `6.50/8.54/10.60 mm`, 45-degree socket chamfers `0.80/1.00/1.20 mm`, mouth
  AF `8.10/10.54/13.00 mm`, and 1.15x outer heads. Tiny raised v1.3 text was
  replaced with bold 7 mm labels engraved 0.8 mm deep and a larger `+X` arrow.
- The paired peg/socket geometric guiding estimate is `1.65/2.07/2.50 mm`,
  approximately `5.32x/5.45x/5.32x` the source pair. This is only a yaw-aligned
  flat-normal first-contact estimate, not a guaranteed capture radius.
  Friction, FDM ridges, burrs, tilt, yaw, and multi-point contact remain failure
  modes.
- The enlarged deep walls introduce approximately
  `+/-9.75/+/-7.59/+/-6.63 deg` ideal centered yaw free play. The fixture can
  make insertion easier but delays torque contact and changes the tactile
  fingerprint. Hardware-specific MuJoCo geometry, terminal-gate retuning,
  low-force compliant insertion, and bounded seating twist are required.
- Active output contains nine printable parts: three complete tools,
  nominal/easy-v2/FDM-trial bases, and three socket coupons. All nine STEP
  files re-import as one valid solid and all nine STL files pass watertight,
  edge/vertex-manifold, orientable, and no-self-intersection checks.
  Design-parameter SHA-256 is
  `594a752fefc2b6cedd05334cf89a739ed1e28165b5a31f1be3f0858a2053021f`.
- Claim boundary: one-piece printing removes quick-change fit compliance but
  does not shorten the 42 mm lever arm. Printed dimensions, mass/COM/inertia,
  sensor-origin-to-tip transform, static deflection, return-to-zero, friction,
  safe wrench limits, and robot-cell capture performance remain commissioning
  measurements. No robot command was sent by this CAD update.

## 2026-07-21 - Active Bota host route changed to USB Binary

- The installed body identity remains BFT-SENS-CAT2-B4 / SN001349, but the
  active host connection is now direct USB CDC rather than the historical
  EtherCAT path. The exact route is USB `0483:5740`, serial `3885355D3532`,
  stable `/dev/serial/by-id/...3885355D3532-if00`, `Bota_Binary` at 2 Mbaud.
- KUKA separately owns the USB-C dock Ethernet interface
  `enxf8e43b73f78c`; the active tooling contract and loader now reject the old
  CANopen-over-EtherCAT interface name. Historical EtherCAT evidence remains
  provenance and was not rewritten.
- A three-second official-driver read yielded 600 accepted and zero rejected
  frames, followed by successful PC-side start/status checks at approximately
  200 Hz with the persistent viewer mapped. Evidence is recorded in
  `docs/setup/evidence/BOTA_USB_BINARY_TOPOLOGY_20260721.md`.
- This validates the H0 observation route only. Absolute wrench metrology,
  payload gravity compensation, sensor-origin-to-tip transform and H1+ motion
  gates remain incomplete. This update initiated no KST connection and sent no
  robot command.
- Final hardening removed the remaining active launch default that could reuse
  the KUKA NIC for Bota EtherCAT. Daily preparation now compares driver and
  viewer runtime hashes, checks the exact dock MAC/route, and rejects an
  already-established KST port-30001 owner. A live MATLAB KST socket was
  observed and correctly blocked preparation without being terminated; after
  it closed independently, the USB viewer restarted and passed at about 200 Hz.

## 2026-07-24 - Tooling v1.5 mechanical yaw datum and frame contract

- An OCC audit of the installed-tool sketch and source flange STEP identified
  the A/C reference as the central R36 cylindrical tangent datum at
  `45/225 deg`, while the photographed legacy peg face normal was `60 deg`.
  All M8/M10/M12 pegs are therefore clocked `-15 deg` about flange CAD `+Z`.
  The tip chamfer, uniform shaft, and AF14 transition rotate together.
- Exported STEP sections verify a selected peg-flat normal of `45.000 deg`,
  zero CAD residual to the flange datum, unchanged AF `6/8/10 mm`, and unchanged
  `Z=-42 mm` tips. Socket yaw remains zero and no fixture/contact dimension
  changed.
- Generic OCC loft tessellation initially created overlapping M8/M10 triangles
  despite valid B-reps. The hex frustum is now an explicit planar-face solid.
  All nine active STLs and all nine active 3MFs are watertight, manifold,
  orientable, and self-intersection-free; all part volumes match the previous
  geometry within numerical precision.
- `real_tooling_v1.json` now separates CAD peg/flange clocking from the unknown
  sensor, robot, and fixture transforms. The derived MuJoCo scenes retain
  peg/socket relative yaw zero; adding the CAD `-15 deg` to the simulated bit
  would double-count the physical flange-datum teaching.
- Existing multisize models need not be retrained solely because of this CAD
  clocking if measured physical relative yaw lies in the current
  `[-2,+2] deg` support and observations/actions are transformed through the
  canonical frame. Otherwise M8/M10/M12 must be resampled around the measured
  yaw followed by fresh fine-tuning and audit.
- Old v1.4 poses/releases/data remain immutable historical evidence. The M12
  live script now uses the correct `15+60k deg` tip-vertex phase and fails
  closed until the new print has taught-datum, parallelism, relative-yaw,
  `42.00 +/- 0.25 mm` tip height, `<=0.20 mm` radial runout, and low-load
  fingerprint/capture requalification records.
- Targeted hardware-profile, hardware-easy scene, artifact, multisize, and XML
  tests passed 22/22.
  CAD STEP and mesh validators passed. No KST connection or robot motion was
  executed.
- Final profile SHA-256 is
  `03878ec282b8e9284b8733062d9ecf9132834069a83cad573755b0120617b356`;
  the regenerated hardware-easy scene-manifest SHA-256 is
  `c2f0b21d68788fc138930185bd9dd69d6a4cefa9cb08dd84b266c9eddc07695f`.

## 2026-07-24 - Clocking-clean multisize direction audit (v7 fresh8)

- Audited the clocking-clean shared direction-model lineage. V5 used only
  geometry-audited canonical datasets, but its untouched fresh7 audit failed
  M8 source safety (`p90=167.86 deg`, accepted accuracy `0.866`, eight accepted
  errors over 90 degrees). V6 head-only and tail-only anchors remained rejected
  development diagnostics. Their data and failed reports were not folded into
  the final training set.
- Trained the source-preserving v7 candidate from the clean M8-v2 base using
  tail-only canonical adaptation, M8-source weight `0.72`, learning rate
  `2e-5`, and L2-SP weight `1.0`. The adaptation report records empty
  development-dataset lists, zero final-audit samples seen, and no audit run by
  the trainer. The separate lineage audit passes with SHA-256
  `fac4ede8705c5e07b44e97f0b255c1628185e5efb1a4d17671ee44e660e225c4`.
- Froze and generated an independent fresh8 audit only after v7 selection.
  M8-source, M8-easy, M10-easy, and M12-easy each completed 180/180 outcomes
  with zero sampler failures; every sampling contract is
  `training_use=forbidden` and has no training-group overlap.
- The single frozen fresh8 evaluation **failed global promotion**. M8-easy,
  M10-easy and M12-easy passed, but M8-source returned
  `median/p90=4.47/174.67 deg`, `within7.5=0.725`, APS
  `0.889/3.68`, accepted rate/accuracy `0.690/0.924`, and two accepted errors
  over 90 degrees. Radial levels 3 and 4 failed the finite-sample safety
  screen. M12 alone passed with `median/p90=0.93/3.29 deg`, accepted
  rate/accuracy `0.988/0.994`, and zero accepted errors over 90 degrees.
- No post-fresh8 fit, calibration or gate change was performed. No promotion
  candidate was written. The final report SHA-256 is
  `f12764904cd30b953e9ba627058a1f442fe1471870f3cc56f801e1b94fa4355d`.
  The v7 contract SHA-256 is
  `0fbb33d7dda8c6eeb3f133ccea8ed181a13855eb1d46414240150f2285087fea`.
- The real inference architecture can load v7 and replay an existing M12 trace;
  v7 tightens the ensemble-span abstention gate from 180 to 7.5 degrees.
  Consequently v7 may be considered only as explicitly rejected-global-audit,
  M12 development-shadow diagnostics. It is not a globally promoted model and
  grants no robot-motion authority.
- The real-device deployment manifest was not modified by this audit and
  remains SHA-256
  `5a0018c024fd379e46035967aa673944732b362b364fa59ae26241a5f374d04f`.
  Full evidence, hashes and reproduction command are in
  `docs/CLOCKING_CLEAN_MULTISIZE_DIRECTION_AUDIT.md`.
- Fourteen targeted tests passed. The generic promotion builder was also run
  against the failed fresh8 report; it rejected the report before writing an
  output file, as required by the fail-closed contract.

## 2026-07-24 - Explicit M12-only v7 development-shadow deployment

- After the independent v7/fresh8 decision above was frozen, the real-device
  bridge was updated explicitly rather than silently replacing v4. The
  deployment record declares `REJECTED_GLOBAL_AUDIT`,
  `M12_DEVELOPMENT_SHADOW_ONLY`, `supported_sizes=["m12"]`, and
  `allowed_values=["SHADOW_ONLY"]`. It has no promotion-contract path and
  therefore cannot expose `CAPPED_SHORTCUT`.
- The real MATLAB entry now binds the v7 model identifier/root and pins the
  deployment manifest itself before any KST connection. The manifest in turn
  verifies 13 model artifacts, five audit-evidence artifacts, the inference
  runtime and its requirements lock. The deployment-manifest SHA-256 is
  `1716bb00e1dfcd7b6863524d14d2e41cbbf7f572f49d7ce20a53369526e9acb9`;
  the runtime SHA-256 is
  `762fb4e89f3c3754ae22907440ba08a293c00fdab99e9a17c5f8d19d4be0f1a4`.
- An immutable M12 real conical trace replay is byte-repeatable with circular
  mean `275.846877805098 deg`, APS set size `3`, ensemble span `7.5 deg`,
  and `hardware_control_allowed=false`. This is compatibility evidence only,
  not a hardware-accuracy or control result.
- Real post-print metrology remains outside the immutable simulation tooling
  profile, whose SHA-256 stays
  `03878ec282b8e9284b8733062d9ecf9132834069a83cad573755b0120617b356`.
  Ambiguous old clocking/frame traces, captured/incomplete scans, and any real
  cone lacking independent measured phase/pose qualification are forbidden
  from direction training.
- Thirty-one targeted real sampling/runtime-authority/baseline static tests
  passed after the atomic v7 switch. No MATLAB/Octave runtime was available,
  and no ROS, sensor, KST, or robot connection was made.

## 2026-07-24 - Conical contact-transition closure and continuity-clean v8

- Closed the inner-chamfer support case explicitly. The online simulator now
  watches only measured F/T, measured tip-Z trend, elapsed time, and target
  force. Sustained low support or safe downward progress interrupts the cone,
  returns upright from the actual last orientation, and enters fresh physical
  capture requalification. An incomplete cone cannot call learned/classical
  direction inference, posterior transport, or a direct ray; rejected capture
  evidence permits only a direction-free global outward fallback or safe stop.
- Added reason-coded offline continuity hygiene without rewriting historical
  NPZs. A sustained 6 ms MuJoCo `contact_count==0`, full-chamfer-depth
  in-scan progress, capture, incompletion, failure, or non-finite signal is
  quarantined. Low measured force with continuous contact and no such progress
  is retained as the intended unloading fingerprint and tagged diagnostic.
  Canonical admitted counts are M8 `675/768`, M10 `611/768`, and M12
  `575/768`.
- Froze an exact-byte audit of all 2,304 canonical NPZs, retrained the
  contact-continuity-clean v8 candidate from the audited M8 base, and bound
  both the old geometry-equivalence report and the new continuity report into
  the split, runtime contract, and adaptation report. The separate v8 lineage
  audit passes. Its SHA-256 is
  `fbd120b634f63435a01b533da78a38bd652a24d5e444e5762e37cf9030cd277f`.
- Defined fresh9 radii, phases, and seeds only after v8 was frozen. All four
  domains completed 180/180 with zero sampler failures. M8-easy and M10 pass.
  Global promotion still fails: M8-source has p90 `163.64 deg` with two
  accepted errors above 90 degrees; M12 has aggregate p90 `3.74 deg` but its
  innermost radial screen has 4/21 accepted errors above 7.5 degrees. The
  frozen fresh9 report SHA-256 is
  `8c72ef74cbe1f7232d89b2fc3b9276f096d57b87b7759e189c53593bcf60a178`.
- v8 is therefore `REJECTED_GLOBAL_AUDIT`. The hardware manifest was not
  changed; v7 remains M12 development shadow only with no XY authority.
  Rejection does not break the control closure because the transition state
  machine, direction-free fallback, physical funnel, and terminal verifier do
  not depend on the model.
- The paired Chinese/English Marp decks now state the contact-transition
  policy, revalidated counts, and negative fresh9 result. Detailed evidence
  and reproduction commands are in
  `docs/CONTACT_CONTINUITY_AND_V8_FRESH9_AUDIT.md`. No ROS, Bota, KST, MATLAB,
  or robot connection was opened.
