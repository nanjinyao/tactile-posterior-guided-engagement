# Learning-Based Tactile Posterior-Guided Search for Robotic Hex-Socket Engagement

Research code and evidence accompanying the manuscript in
[`overleaf_unscrewing/main.tex`](overleaf_unscrewing/main.tex).
A conical tactile observation guides a bounded shortcut and residual search;
compliant control, capture checks, recovery, and terminal verification govern
physical execution. The demonstrated endpoint is **geometric engagement**,
not screw turning, thread release, or extraction.

**Release status: prepared for review, not cleared for public publication.**
See [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) for missing hardware dependencies,
source-version provenance, license/privacy decisions, and validation failures.
No results or manuscript claims were changed during repository preparation.

## Start here

1. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact workflows and limits.
2. Use [the figure/table map](REPRODUCIBILITY.md#paper-to-artifact-map) to locate
   the evidence behind each manuscript result.
3. Run the dependency-free stored-evidence check from this directory:

   ```bash
   python tools/verify_paper_evidence.py
   ```

This checks existing records, selected counts, hashes, and figure availability.
It does not rerun an experiment.

## Repository layout

| Path | Purpose |
|---|---|
| `mujoco_unscrewing_probe_new/src/` | Simulation, inference, search/control, evaluation and rendering modules |
| `mujoco_unscrewing_probe_new/configs/`, `models/` | Size-specific geometry, contact/tooling configuration and MuJoCo models |
| `mujoco_unscrewing_probe_new/outputs/current/` | Frozen datasets, checkpoints, evaluation reports, individual trials, diagnostics and media |
| `mujoco_unscrewing_probe_new/tests/` | Controller, model, contract and evidence checks |
| `real_device/` | MATLAB hardware entry points and shared controller |
| `real_device/configs/`, `model/` | Installation/commissioning records and CAD/tooling |
| `real_device/data/` | Raw named hardware runs, acquisition campaigns, manifests and event logs |
| `real_device/offline_learning/`, `offline_analysis/` | Real-data construction/training code and retained outputs |
| `real_device/model_runtime/` | Offline Python inference interface and exported model bundles |
| `overleaf_unscrewing/` | Current LaTeX manuscript, ten included figures, bibliography and figure builders; existing nested Git repository |
| `group_meeting_trace/`, `outputs/` | Meeting drafts and presentations, not the canonical paper result registry |
| `references/`, `Philosophical taste style rhythm and subtle finesse/` | Third-party reading material and separate reference projects; redistribution review required |
| `archive/`, `tmp/`, project trace files | Historical provenance and intermediate assets; retained, not automatically designated for release |
| `tools/` | Stored-evidence verification, local audit and existing trace utilities |

## Environment and installation

The original frozen simulation's complete software lock was not found.
`requirements.txt` lists direct imports found in the active code, with no
invented version pins. A new environment is useful for inspection but is not
evidence of numerical equivalence to the original runs.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Additional tools: FFmpeg for video keyframes; a LaTeX installation with
`pdflatex`, `bibtex`, `ieeeconf.cls` (included), and `IEEEtran.bst` for the paper;
MATLAB for the hybrid Fig. 3 builder and hardware code. CAD regeneration uses
the separate tooling dependencies documented under `real_device/model/`.
The simulation's S4 evaluation uses Unix `fcntl`, so that workflow needs Linux.

The audit used Windows, CPython 3.11.16 (AST scan: 3.12.14), and a fresh isolated
environment. Static and selected controller tests ran, but MuJoCo 3.13.0 and
PyTorch 2.14.0 failed DLL initialization here. This is **not** a tested full
simulation environment. The pre-existing hardware offline-learning pins target
Python 3.8; preserve their separate bootstrap and dependency files.

## Simulation and experiments

Run simulation modules from `mujoco_unscrewing_probe_new/`, so `src` imports
resolve. Existing frozen results can be inspected without running MuJoCo.
Once the simulation environment is qualified, the five-method runner accepts:

```bash
cd mujoco_unscrewing_probe_new
python -m src.run_m8_five_method_retry3_audit --help
python -m src.run_m8_five_method_retry3_audit \
  --poses 24 --workers 1 --no-resume --output outputs/reproduction/m8_five_method
```

The second command is a full experiment, not a smoke check; it was not run in
this audit. It uses the frozen source tasks and model bundles and writes new
`manifest.json`, `task_contract.json`, `trials/`, `report.json`, and `summary.md`.
For headless rendering on a qualified Linux EGL installation, set `MUJOCO_GL=egl`.
Do not replace `outputs/current/` with a new run or treat an unqualified software
environment as a reproduction of the published numerical results.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the recovery ablation, cross-size
protocol, training provenance, figure generation, and manuscript build.

## Hardware workflow

Hardware requires a KUKA LBR iiwa with the compatible Sunrise/DirectServo setup,
the maintained KST MATLAB client, Bota force/torque sensor and ROS interface,
MATLAB with the ROS functionality used by the controller, and the registered
printed hex tools/fixture. The maintained KST client is currently expected
outside this folder and was not found in the audit environment.

Start with [`real_device/README.md`](real_device/README.md),
[`real_device/CLAIM_BOUNDARY.md`](real_device/CLAIM_BOUNDARY.md), and
[`real_device/model_runtime/README.md`](real_device/model_runtime/README.md).
Commission installation, TCP/fixture frames, sensor baseline/freshness and
protection before acquiring data or enabling motion. Several independent
metrology fields remain pending. Existing run-specific authorizations are
historical records, not qualification of another installation.

Simulation uses a four-channel circular CNN ensemble. The selected hardware
bundle documents a seven-channel real Fourier model and TOP1 routing. Do not
silently equate these interfaces; this distinction needs clarification in the
manuscript. Hardware examples are named feasibility/mechanism runs, not a
randomized success-rate benchmark. Forces/torques are baseline-relative at the
Bota sensor origin.

## Figures, videos and limitations

All ten `main.tex` figure inputs are present. Data-derived plots read the frozen
M8 trials and named hardware CSV/event records. Figs. 2–3 include supplied or
hybrid raster components, so complete source-level regeneration is not yet
established. Selected videos illustrate individual mechanisms; the 24-pose
evaluation estimates method outcomes. Consult `outputs/current/VIDEO_RETENTION.json`
inside the simulation project before selecting a video.

The audit found four existing static-test failures, missing KST files, a hardware
source hash mismatch with the named paper run, and stale lower-level documentation.
The included manuscript PDF is 14 pages. Licensing, third-party redistribution,
privacy/anonymity, nested Git handling, and large-file distribution remain author
decisions. See the audit report for exact scope and unresolved items.

## Citation and license

Publication metadata has not been supplied. Replace every placeholder before
using this entry; no authors, venue, year, or DOI are inferred:

```bibtex
@misc{hex_socket_engagement_placeholder,
  title = {Learning-Based Tactile Posterior-Guided Search for Robotic Hex-Socket Engagement},
  author = {AUTHOR METADATA PENDING},
  year = {YEAR PENDING},
  note = {Publication and DOI metadata pending}
}
```

No project-wide license has been selected. This preparation does not grant a
license to project code, data, photos, videos, or third-party material. Retain
existing third-party notices and resolve redistribution scope before publishing.
