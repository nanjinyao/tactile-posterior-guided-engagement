from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import RunResult, run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
    _method_summary,
    _paired_counts,
    _paired_efficiency,
    _success,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/current/evaluations/m8_six_method_joint_funnel_audit"
ENTROPY_SOURCE = ROOT / "outputs/current/evaluations/m8_four_method_entropy_audit"
RANDOM_SOURCE = ROOT / "outputs/current/evaluations/m8_four_method_sensor_only_audit"
METHODS = (
    "proposed_plus_hybrid",
    "proposed_posterior",
    "baseline_a",
    "baseline_b",
    "baseline_c_entropy",
    "baseline_d_random",
)


_RUNTIME: M8DirectionEnsemble | None = None
_OUTPUT: Path | None = None
_ENTROPY_SOURCE: Path | None = None
_RANDOM_SOURCE: Path | None = None
_REUSE_C_FROM: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _worker_init(
    output: str,
    entropy_source: str,
    random_source: str,
    reuse_c_from: str | None,
) -> None:
    global _RUNTIME, _OUTPUT, _ENTROPY_SOURCE, _RANDOM_SOURCE, _REUSE_C_FROM
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble(MODEL_ROOT)
    _OUTPUT = Path(output)
    _ENTROPY_SOURCE = Path(entropy_source)
    _RANDOM_SOURCE = Path(random_source)
    _REUSE_C_FROM = None if reuse_c_from is None else Path(reuse_c_from)


def _load_trial(root: Path, pose_id: str) -> dict:
    return json.loads(
        (root / "trials" / f"{pose_id}.json").read_text(encoding="utf-8")
    )


def _run_hybrid(task: PoseTask) -> RunResult:
    assert _RUNTIME is not None
    return run_method(
        "hybrid",
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=_RUNTIME,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        max_transport_m=None,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=130.0,
        continue_to_bottom=False,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed),
            delay_s=task.sensor_delay_s,
        ),
    )


def _run_entropy(task: PoseTask) -> RunResult:
    return run_method(
        "entropy_grid",
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=None,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        max_transport_m=None,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=130.0,
        continue_to_bottom=False,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed),
            delay_s=task.sensor_delay_s,
        ),
    )


def _run_pose(task: PoseTask) -> dict:
    assert _OUTPUT is not None
    assert _ENTROPY_SOURCE is not None and _RANDOM_SOURCE is not None
    started = time.perf_counter()
    entropy_record = _load_trial(_ENTROPY_SOURCE, task.pose_id)
    random_record = _load_trial(_RANDOM_SOURCE, task.pose_id)
    expected_task = asdict(task)
    if entropy_record["task"] != expected_task or random_record["task"] != expected_task:
        raise ValueError(f"frozen task mismatch for {task.pose_id}")

    shared_legacy_methods = ("proposed_posterior", "baseline_a", "baseline_b")
    legacy_identical = all(
        entropy_record["methods"][method] == random_record["methods"][method]
        for method in shared_legacy_methods
    )
    if not legacy_identical:
        raise ValueError(f"legacy method mismatch for {task.pose_id}")

    hybrid = _run_hybrid(task)
    entropy = (
        _run_entropy(task)
        if _REUSE_C_FROM is None
        else RunResult(
            **_load_trial(_REUSE_C_FROM, task.pose_id)["methods"][
                "baseline_c_entropy"
            ]
        )
    )
    proposed = entropy_record["methods"]["proposed_posterior"]
    fingerprint_match = hybrid.fingerprint_sha256 == proposed["fingerprint_sha256"]
    if not fingerprint_match:
        raise ValueError(f"conical fingerprint mismatch for {task.pose_id}")

    record = {
        "task": expected_task,
        "methods": {
            "proposed_plus_hybrid": asdict(hybrid),
            "proposed_posterior": proposed,
            "baseline_a": entropy_record["methods"]["baseline_a"],
            "baseline_b": entropy_record["methods"]["baseline_b"],
            "baseline_c_entropy": asdict(entropy),
            "baseline_d_random": random_record["methods"]["baseline_c_random"],
        },
        "fairness": {
            "same_initial_pose_nuisance_and_sensor_prior": True,
            "proposed_plus_and_proposed_fingerprint_match": fingerprint_match,
            "frozen_pab_reused_byte_for_byte": True,
            "baseline_c_rerun_with_cad_scaled_grid": True,
            "baseline_c_reused_for_hybrid_only_pilot": _REUSE_C_FROM is not None,
            "frozen_random_reused_byte_for_byte": True,
            "legacy_common_methods_identical_across_sources": legacy_identical,
            "online_true_center_or_contact_tags": False,
        },
        "diagnostics": entropy_record["diagnostics"],
        "worker_elapsed_s": time.perf_counter() - started,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _quantile(values: list[dict], key: str, q: float) -> float | None:
    data = [float(value[key]) for value in values if value.get(key) is not None]
    return None if not data else float(np.quantile(data, q))


def _summary(records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    hybrid_values = [record["methods"]["proposed_plus_hybrid"] for record in records]
    hybrid_summary = methods["proposed_plus_hybrid"]
    hybrid_summary.update(
        {
            "entropy_probe_count_median": _quantile(
                hybrid_values, "entropy_grid_probe_count", 0.5
            ),
            "entropy_probe_count_p90": _quantile(
                hybrid_values, "entropy_grid_probe_count", 0.9
            ),
            "positive_value_probe_total": int(
                sum(
                    sum(
                        bool(gate.get("executed")) and gate.get("probe") is not None
                        for gate in value.get("hybrid_probe_gate_records", [])
                    )
                    for value in hybrid_values
                )
            ),
            "local_spiral_count": int(
                sum(bool(value.get("entropy_grid_used_spiral")) for value in hybrid_values)
            ),
            "global_fallback_count": int(
                sum(bool(value.get("global_outward_fallback_used")) for value in hybrid_values)
            ),
            "posterior_credible_radius_median_mm": _quantile(
                hybrid_values, "entropy_grid_credible_radius_mm", 0.5
            ),
        }
    )

    comparisons = {}
    for right in METHODS[1:]:
        key = f"proposed_plus_hybrid_vs_{right}"
        comparisons[key] = {
            "paired_outcomes": _paired_counts(
                records, "proposed_plus_hybrid", right
            ),
            "paired_efficiency": _paired_efficiency(
                records, "proposed_plus_hybrid", right
            ),
        }

    return {
        "stage": "M8_six_method_joint_funnel_paired_audit",
        "design": {
            "poses": len(records),
            "task_source": str(
                (ENTROPY_SOURCE / "manifest.json").relative_to(ROOT)
            ),
            "random_source": str(
                (RANDOM_SOURCE / "manifest.json").relative_to(ROOT)
            ),
            "random_position_design": (
                "area-uniform Sobol inside the declared visual prior; unseen by P1v3"
            ),
            "same_pre_hardware_wrench_prior": True,
            "proposed_plus_new_rollouts_only": True,
            "pab_and_random_records_frozen": True,
            "proposed_plus_and_c_new_rollouts": not bool(
                records[0]["fairness"]["baseline_c_reused_for_hybrid_only_pilot"]
            ),
            "same_conical_fingerprint_for_proposed_plus_and_proposed": True,
            "online_capture_uses_privileged_mujoco_tags": False,
            "offline_success_scoring_uses_mujoco_geometry": True,
            "proposed_plus": (
                "same P1v3 conical fingerprint and visual/CAD posterior as Proposed; posterior is projected "
                "onto the CAD grid; each guarded segment maximizes capture probability plus information gain "
                "minus contact and travel cost; a no-capture segment is a soft negative observation; F/T "
                "likelihood is temperature-downweighted under predictive surprise; no local spiral is used, "
                "a CAD-scaled one-sided conical-direction gate rejects reverse-side capture candidates, "
                "while full visual-support outward coverage remains a final safety fallback"
            ),
            "baseline_d": (
                "seeded area-uniform random route with the same path budget; no conical learner, CAD "
                "likelihood, entropy objective, true center, or spiral"
            ),
        },
        "methods": methods,
        "paired": comparisons,
        "fingerprint_match_rate": float(
            np.mean(
                [
                    record["fairness"][
                        "proposed_plus_and_proposed_fingerprint_match"
                    ]
                    for record in records
                ]
            )
        ),
        "claim_boundary": (
            "Frozen M8 MuJoCo audit inside the declared visual prior. The contact nuisance and F/T "
            "bias/noise/coupling/delay distributions are pre-hardware engineering priors, not measured iiwa "
            "statistics. Proposed+ and Proposed share a byte-identical conical fingerprint and frozen P1v3 "
            "output for each pose. No method receives the true center online; MuJoCo geometry is used only "
            "for offline success scoring. Baseline-B reset travel remains excluded by the frozen source audit. "
            "This can support a simulation mechanism claim, not a hardware robustness or cross-size claim."
        ),
        "updated_utc": _utc_now(),
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_report(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 Six-Method Joint-Funnel Paired Audit",
        "",
        "All six methods use the same 36 unseen Sobol poses and pre-hardware wrench prior. "
        "Efficiency medians are conditioned on offline-validated capture.",
        "",
        "| method | success | first attempt | capture time | actual/contact path | wear | max F p90 | max tau p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        lines.append(
            f"| {method} | {item['successes']}/{item['trials']} | "
            f"{item['first_attempt_successes']}/{item['trials']} | "
            f"{_fmt(item['time_success_median_s'])} s | "
            f"{_fmt(item['actual_path_median_mm'])}/{_fmt(item['contact_path_median_mm'])} mm | "
            f"{_fmt(item['wear_proxy_median_mj'])} mJ | "
            f"{_fmt(item['max_force_p90_n'])} N | {_fmt(item['max_torque_p90_mnm'])} mNm |"
        )
    lines.extend(
        [
            "",
            "## Proposed+ task-grid diagnostics",
            "",
            *[
                f"- {key}: {value}"
                for key, value in report["methods"]["proposed_plus_hybrid"].items()
                if key
                in {
                    "entropy_probe_count_median",
                    "entropy_probe_count_p90",
                    "positive_value_probe_total",
                    "local_spiral_count",
                    "global_fallback_count",
                    "posterior_credible_radius_median_mm",
                }
            ],
            "",
            "## Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _tasks_from_source(source: Path, poses: int) -> list[PoseTask]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tasks = [PoseTask(**item) for item in manifest["tasks"]]
    if poses > len(tasks):
        raise ValueError(f"requested {poses} poses, source has only {len(tasks)}")
    return tasks[:poses]


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    entropy_source = Path(args.entropy_source).resolve()
    random_source = Path(args.random_source).resolve()
    (output / "trials").mkdir(parents=True, exist_ok=True)
    tasks = _tasks_from_source(entropy_source, args.poses)

    existing: list[dict] = []
    pending: list[PoseTask] = []
    for task in tasks:
        path = output / "trials" / f"{task.pose_id}.json"
        if args.resume and path.exists():
            existing.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)
    records = list(existing)
    started = time.perf_counter()

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        done_new = len(records) - len(existing)
        rate = done_new / elapsed
        _atomic_json(
            output / "progress.json",
            {
                "total_poses": len(tasks),
                "completed_poses": len(records),
                "rate_poses_s": rate,
                "eta_s": None if rate <= 0.0 else (len(tasks) - len(records)) / rate,
                "updated_utc": _utc_now(),
            },
        )

    checkpoint()
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(
                str(output),
                str(entropy_source),
                str(random_source),
                None
                if args.reuse_c_from is None
                else str(Path(args.reuse_c_from).resolve()),
            ),
        ) as pool:
            futures = {pool.submit(_run_pose, task): task for task in pending}
            with tqdm(
                total=len(tasks),
                initial=len(existing),
                desc="M8 six-method audit",
                unit="pose",
                dynamic_ncols=True,
            ) as bar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "task": asdict(task),
                            "error": repr(exc),
                            "created_utc": _utc_now(),
                        }
                        _atomic_json(
                            output / "trials" / f"{task.pose_id}.json", record
                        )
                    records.append(record)
                    checkpoint()
                    bar.update(1)
                    valid = [item for item in records if "error" not in item]
                    bar.set_postfix(
                        hybrid=sum(
                            _success(item["methods"]["proposed_plus_hybrid"])
                            for item in valid
                        ),
                        refresh=False,
                    )

    errors = [record for record in records if "error" in record]
    if errors:
        print(json.dumps(errors, indent=2))
        return 1

    records.sort(key=lambda item: item["task"]["pose_id"])
    report = _summary(records)
    _write_report(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "tasks": [asdict(task) for task in tasks],
            "completed": len(records),
            "entropy_source": str(entropy_source.relative_to(ROOT)),
            "random_source": str(random_source.relative_to(ROOT)),
            "updated_utc": _utc_now(),
        },
    )
    checkpoint()
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired M8 Proposed+/Proposed/A/B/C/D audit."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--entropy-source", default=str(ENTROPY_SOURCE))
    parser.add_argument("--random-source", default=str(RANDOM_SOURCE))
    parser.add_argument(
        "--reuse-c-from",
        default=None,
        help="reuse CAD-scaled Baseline-C trial records for a hybrid-only pilot",
    )
    parser.add_argument("--poses", type=int, default=36)
    parser.add_argument(
        "--workers", type=int, default=min(36, max(1, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
