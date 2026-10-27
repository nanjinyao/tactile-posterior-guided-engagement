# Reproducibility and evidence provenance

Audit date: 2026-09-17. Canonical manuscript: `overleaf_unscrewing/main.tex`.
Figure/table numbering below follows its included PDF/auxiliary labels (14 pages,
10 figures, 4 tables). Older decks and README titles are historical context.

## What has and has not been reproduced

The audit verified availability of the ten figure inputs, frozen M8 counts and
selected paired statistics, stored manifest hashes, cross-size artifact tests,
and selected control/static tests. It did not rerun the experiments, retrain a
model, connect to hardware, or establish numerical equivalence on a fresh
machine. See [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) for failures and limits.

Run `python tools/verify_paper_evidence.py` from the repository root to check
stored evidence without installing scientific packages. It prints JSON and
does not modify the evidence. `python tools/audit_publication.py` performs a
broader inventory, text scan, Python/JSON parse and duplicate check; reports
go into ignored `.publication-audit/` and may contain sensitive path names.

## Paper-to-artifact map

Path prefixes in this table are exact aliases:

- **S** = `mujoco_unscrewing_probe_new/`
- **C** = `mujoco_unscrewing_probe_new/outputs/current/`
- **P** = `overleaf_unscrewing/`
- **R** = `real_device/`

| Paper result | Exact source, config/data and output | Reproduction status |
|---|---|---|
| Fig. 1, hardware coordinate frames | `P/tools/build_fig1_frames.py`; input `P/figures/fig1_hardware_coordinate_frames_source.png`; output `P/figures/fig1_hardware_coordinate_frames.png` | Source photo and annotation builder present; photo/installation metrology and release permission require author review |
| Fig. 2, control framework | `P/figures/fig2_control_framework.png`; `P/tools/plot_publication_figures.py:plot_framework` is a legacy layout deliberately not called by `main()` | Included raster exists; current supplied/composite figure is not established as reproducible from that legacy function |
| Fig. 3, action/observation/routing | `P/tools/build_fig3_hybrid.m`; inputs `P/figures/fig3_real_probe_keyframes.png`, `tactile_fingerprint_and_direction.png`, `fig1_hardware_coordinate_frames.png`; output `fig3_action_observation_routing.png` in same directory | Hybrid MATLAB composition exists. The frozen tactile raster can be traced to `plot_tactile_observation()` and `C/evaluations/p1v5_m8_final_untouched_direction_audit/trials/double_overhang_audit3_r04_l01_s04_n01.npz` plus `predictions.csv`. Real keyframe extraction times/source are not fully encoded in this builder; routing illustration is schematic |
| Fig. 4, posterior shortcut and residual geometry | `P/tools/plot_publication_figures.py:plot_shortcut_geometry`; `C/evaluations/m8_five_method_retry3_r3_closed_loop/trials/sobol_*.json`; inset files `P/figures/fig4_m8_surface_contact_inset_rgba.png` and `fig4_m8_mujoco_interface.png`; output `P/figures/posterior_shortcut_geometry.pdf` | Data/plotter/insets present; retain selected-case IDs from the plotter rather than choosing new cases |
| Fig. 5 and Table I, Direct-4C versus Direct-3C | `S/src/run_m8_five_method_retry3_audit.py`; `C/evaluations/m8_five_method_retry3_r3_closed_loop/{task_contract.json,report.json,trials/sobol_*.json}`; `P/tools/plot_publication_figures.py:plot_representation_ablation`; output `P/figures/m8_four_channel_representation_ablation.png` | Artifact tests pass. 17/24 versus 16/24 terminal; first-ray 18/24 versus 16/24; any-ray 19/24 each. This is a Jin-style proxy, not a faithful reimplementation of Jin et al. |
| Fig. 6, four-policy Sobol 016 keyframes | `S/src/render_m8_retry3_selected_media.py`; `C/active_comparison/m8_retry3_selected_media/sobol_016/comparison_fuzzy_retry3_2x2_manifest.json` and MP4s in that directory; `P/tools/plot_publication_figures.py:plot_policy_keyframes`; output `P/figures/m8_four_policy_keyframe_matrix.png` | Selected mechanism example; not aggregate statistics. FFmpeg required |
| Fig. 7 and Table II, M8 success/time/path/wear | Same frozen 24 paired trials and `report.json` as Table I; `P/tools/plot_publication_figures.py:plot_budgeted_profiles`; output `P/figures/m8_budgeted_engagement_profiles.png` | Counts verified from individual trials: PGS-O 20, PGS-G 19, Outward 19, Direct-4C 17 (each /24). Successful-subset medians are not paired treatment effects |
| Paired efficiency paragraph | `C/evaluations/m8_five_method_retry3_r3_closed_loop/trials/sobol_*.json`; common-success restriction on `result == cad_bottom_verified` | 19 common PGS-O/Outward successes; all 19 path differences negative; median difference -41.141685 mm rechecked. Other reported tests require their stated success subset/statistical definition; no standalone manuscript-table/statistics exporter found |
| Recovery ablation, 0/12 versus 9/12 | `S/src/run_m8_recoverable_funnel_r3_ablation.py`; `C/evaluations/m8_recoverable_funnel_r3_ablation/{task_contract.json,report.json,report.md,summary.csv,paired_outcomes.csv,trials.csv,trials/recoverable_*.json}` | Three arms: no-funnel 2/12, shield/R0 0/12, shield/R3 9/12. The causal recovery comparison is R3 versus R0, simulation only |
| Table III, M10/M12 transfer | `S/src/run_cross_size_s4_full_loop_evaluation.py`; `C/diagnostics/cross_size_s4b_full_loop_development_v5/` and `C/evaluations/cross_size_s4b_full_loop_untouched_v1/`: each has `report.json`, `task_contract.json`, `provenance.json`, `manifest.json`, size-specific trial records; adaptation `C/datasets/cross_size_s4b_adaptation/`, models `C/models/cross_size_s4b_state_direction_heads/` | Existing cross-size artifact tests pass. Untouched M10 fine-tuning gate fails; preserve mixed transfer result and test-only status |
| Fig. 8, named M12 PGS-G feasibility | `R/data/M12_proposed_plus_entropy_grid_20260729_215410_474/{robot_state.csv,world_star_relative_wrench.csv,conical_directservo_trace.csv,events.csv,summary.json}`; `P/tools/plot_publication_figures.py:plot_hardware_trace`; output `P/figures/hardware_m12_named_feasibility.png` | Single-run evidence. `task_completed=true` coexists with `success=false`; interpret event/endpoint semantics, not a generic success Boolean |
| Fig. 9, same-start hardware pair | `R/data/M12_proposed_plus_entropy_grid_20260730_030400_362/` and `R/data/M12_baseline_improved_direct_x3_20260730_032603_301/`; same state/wrench/events/summary files; `plot_hardware_same_start_pair`; output `P/figures/hardware_m12_same_start_mechanism_pair.png` | Two named extended-budget mechanism runs, not randomized population evidence |
| Fig. 10 and Table IV, four hardware policies | `R/data/M12_proposed_plus_entropy_grid_20260729_215410_474/`, `M12_proposed_local_spiral_20260729_215905_875/`, `M12_baseline_improved_direct_x3_20260729_220513_919/`, `M12_baseline_pure_outward_spiral_20260729_225552_920/`; `plot_hardware_four_policy_matrix`; output `P/figures/hardware_m12_four_policy_evidence_matrix.png` | Same registered task, named runs. Outward terminal after extension is not success within 130 s |
| Table IV, 78 conical acquisitions | `R/data/M8_conical_fingerprints_26pt_20260728_161249_458/`, `M10_conical_fingerprints_26pt_20260728_152808_877/`, `M12_conical_fingerprints_26pt_20260728_175502_318/`; `sampling_campaign.json`, `sampling_attempts.csv`, `sampling_attempts/`; allowlist in `R/offline_learning/build_real_multisize_dataset.py` | 16 clean + 4 consumed holdout + 4 quarantine + 2 sentinel per size. This is acquisition/QC evidence, not 78 task completions |
| Table IV, two M10 pilots | `R/data/M10_proposed_plus_entropy_grid_20260730_044101_906/` and `M10_proposed_plus_entropy_grid_20260730_045327_162/`; `summary.json`, state/wrench/events CSVs; entry `R/run_m10_proposed_plus_entropy_trial.m`, pilot config `R/configs/m10_fixed_demo_pilot_authorization_v1.json` | Both stored `task_completed` flags true; fixed-site feasibility, not unseen-position transfer |

## Algorithms and parameter provenance

| Manuscript component | Implementation and provenance |
|---|---|
| Four-channel, 360-phase, 48-bin simulation ensemble | `S/src/train_m8_direction_models.py`, `m8_direction_runtime.py`; `C/models/p1v3_m8_direction_models/ensemble_interface_contract.json`, `ensemble_metrics.json`, `seeds/seed_20260720`, `seed_20260721`, `seed_20260722` (checkpoint, normalization, metrics) |
| Three-channel proxy/direct comparison | `S/src/train_m8_jin_style_direction_proxy.py`, `m8_jin_style_direction.py`; `C/models/m8_jin_2021_style_direction_proxy/runtime_contract.json` and `seeds/` |
| Posterior/CAD sampling and task execution | `S/src/render_m8_search_comparison.py`; exact frozen task seeds/nuisance/limits in the five-method `task_contract.json`; source `C/evaluations/m8_fresh_24_pabc_direction_gate/manifest.json` |
| Quality-dependent shortcut and residual radius | `S/src/posterior_search_plan.py:plan_posterior_guided_search`; caller overrides and recorded trial values take precedence over generic defaults |
| Information grid and measured miss update | `S/src/entropy_grid_baseline.py`, `render_m8_search_comparison.py`; grid diagnostics and selected entropy manifests under `C/active_comparison/` |
| Admittance, wrench observation, recovery, dwell | `S/src/cartesian_admittance.py`, `mujoco_admittance_rollout.py`, `wrench_observation.py`, `funnel_guard.py`, `terminal_completion_verifier.py`; frozen contract includes R3 settings |
| Geometry and hardware-conditioned models | `S/configs/allen_{m8,m10,m12}_source.json`, `allen_{m8,m10,m12}_hardware_easy_v2.json`; `S/models/`; `R/model/parametric_tooling/design_parameters.json` and generated CAD manifest |
| S4b training/calibration/selection | `S/src/sample_cross_size_s4b_adaptation.py`, `finetune_cross_size_s4_state_direction_heads.py`, `cross_size_s4b_base_anchored_adapter.py`; `S/docs/CROSS_SIZE_S4B_FULL_LOOP_AUDIT.md`; dataset/model manifests and execution bundles |
| Hardware model and labels | `R/offline_learning/common.py`, `build_real_multisize_dataset.py`, `train_real_multisize_direction.py`; `R/offline_analysis/real_multisize_learning_v1/{dataset,training}/`; exported `R/model_runtime/bundles/real_multisize_direction_v1/` |
| Physical hardware execution | `R/execute_m12_contact_engagement_policy.m`, four `run_m12_full_loop_*.m` wrappers, M10 pilot wrapper, `R/configs/` and each run's manifest/summary |

No standalone algorithm environment is used in the current LaTeX source;
the method is specified by equations and prose. The posterior-mass proposition
is a mathematical claim, not established by passing software tests. This audit
maps its implementation but does not independently prove it.

Absolute paths in original contracts are provenance and remain unchanged.
The simulation loader now resolves archived checkpoint locations within the
same named local bundle, preserving seed and filename. Other archived path
strings are not automatically asserted to be portable. Hardware source hashes
must be linked to exact historical code before claiming exact reproduction.

## Installation and tests

See root `requirements.txt` for inferred direct dependencies and
`docs/AUDIT_ENVIRONMENT.txt` for the actual diagnostic environment. The latter
is not a validated experiment lock. In particular, the original simulation's
MuJoCo/Python/driver versions remain unconfirmed. Do not confuse the copied
Linux virtual environments with installable, portable dependencies.

From `mujoco_unscrewing_probe_new/`, lightweight commands used in the audit are:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_posterior_search_plan.py tests/test_cartesian_admittance.py \
  tests/test_terminal_completion_verifier.py tests/test_checkpoint_relocation.py
python -m pytest -q -p no:cacheprovider \
  tests/test_m8_five_method_retry3_artifacts.py \
  tests/test_cross_size_s4b_full_loop_artifacts.py
```

These tests do not rerun the full benchmark. The broad existing suite includes
artifact and rendering requirements; a green subset is not a green full suite.

## Simulation execution and ablations

The root README gives the five-method execution command with a separate output.
Its 24 poses and 130 s budget come from the stored contract, not a newly chosen
test distribution. Training is separate from evaluating the supplied models.
Review the relevant dataset/model manifests before invoking any trainer;
original environment and complete source-version provenance remain incomplete.

The recovery runner has **no output-directory option**. Its exact existing CLI,
run from the simulation project, is:

```bash
python -m src.run_m8_recoverable_funnel_r3_ablation --workers 1 --tasks 12 --no-resume
```

Run this only in a separate disposable copy of the repository; it writes to
the canonical recovery-evidence directory. It was not run during this audit.

The S4b full-loop runner defaults to contract freezing, not physics execution:

```bash
python -m src.run_cross_size_s4_full_loop_evaluation \
  --phase development --output outputs/reproduction/s4b_development
```

`--execute` is a separate opt-in after its freeze/provenance/lock requirements
are satisfied. This module imports Unix `fcntl`. Use retained untouched-v1
records for assessment; do not retrain, select gates, or tune/rerun against that
consumed test set. See `S/docs/CROSS_SIZE_S4B_FULL_LOOP_AUDIT.md` for the protocol.

## Hardware acquisition, learning and inference

Sampling wrappers are `run_m{8,10,12}_sampling_conical_fingerprints_26pt.m`, with
separate resume wrappers. They share `execute_m12_contact_engagement_policy.m`.
Full-loop wrappers are `run_m12_full_loop_proposed_local_spiral.m`,
`run_m12_full_loop_proposed_plus_entropy_grid.m`,
`run_m12_full_loop_baseline_improved_direct_x3.m`, and
`run_m12_full_loop_baseline_pure_outward_spiral.m`.
These are motion-capable code, not audit commands. Their installation-specific
commissioning and hardware dependencies are prerequisites.

The hardware offline code has existing explicit Python-3.8-compatible pins:
`R/offline_learning/requirements.in`, `requirements-lock.txt`, and the separate
`R/model_runtime/requirements-lock.txt`. On a correctly prepared Linux copy:

```bash
cd real_device
bash offline_learning/bootstrap_offline_learning.sh
bash offline_learning/run_pipeline.sh
```

The pipeline constructs data, calibrates/trains/evaluates, exports bundles, and
runs focused tests. **Use a disposable copy:** defaults write to retained
dataset/model destinations. The bootstrap expects a compatible `python3`; it
does not choose Python 3.8 automatically. It installs the CPU PyTorch wheel from
its separate index. None of these training commands were run during this audit.

Use the no-motion replay command in `R/model_runtime/README.md` after rebuilding
that runtime environment. It reads the explicitly named historical cone,
checks a deployment manifest, and writes a posterior JSON. Historical
`hardware_control_allowed` output is not fresh physical authorization.

## Figures, videos and paper build

The data figure builder now separates input assets from the output directory:

```bash
# From repository root, with ffmpeg installed on PATH (or FFMPEG set to its path):
python overleaf_unscrewing/tools/plot_publication_figures.py \
  --output-dir .publication-audit/figures
```

It writes PNG/PDF versions of seven data-derived figures. It intentionally does
not rebuild supplied/hybrid Figs. 2–3 or Fig. 1. The audit verified `--help` and
attempted this command: it reached saving the first plot, then failed loading
fontTools' native PDF dependency. Complete figure regeneration is unverified.
No shipped figure was replaced. The existing Fig. 1 Python and Fig. 3 MATLAB
builders write into `P/figures/`; use a separate copy to inspect those outputs.

Current selected videos are indexed in `C/VIDEO_RETENTION.json`. Sobol 016 and
000 illustrate muted/clear cues, Sobol 005 the proxy interface comparison.
`C/active_comparison/strict_entropy_grid_topview_sobol_003/` contains the measured
entropy replay; its `_en` counterpart reuses the same physics trace. Those
videos are supporting examples, not additional independent experiments.

To build the manuscript from included figure assets, run in an isolated copy
of `overleaf_unscrewing/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`IEEEtran.bst` must be supplied by the TeX installation. The existing PDF was
read, its first page rendered/inspected, and all figure paths checked. This audit
did not validate a fresh TeX build or visually review all 14 pages. The older
seven-page validation in `P/README.md` refers to an earlier draft.
