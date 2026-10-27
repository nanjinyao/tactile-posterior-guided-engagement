from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
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
DEFAULT_SOURCE = ROOT / "outputs/current/evaluations/m8_fresh_24_pabc_direction_gate"
DEFAULT_OUTPUT = ROOT / "outputs/current/evaluations/m8_fresh_24_task_aware_validation"
METHODS = (
    "proposed_plus_hybrid",
    "proposed_posterior",
    "baseline_a",
    "baseline_b",
    "baseline_c_entropy",
)
ACTIVE_TIME_BUDGET_S = 130.0

_RUNTIME: M8DirectionEnsemble | None = None
_SOURCE: Path | None = None
_OUTPUT: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _worker_init(source: str, output: str) -> None:
    global _RUNTIME, _SOURCE, _OUTPUT
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble(MODEL_ROOT)
    _SOURCE = Path(source)
    _OUTPUT = Path(output)


def _load_source_trial(task: PoseTask) -> dict:
    assert _SOURCE is not None
    record = json.loads(
        (_SOURCE / "trials" / f"{task.pose_id}.json").read_text(encoding="utf-8")
    )
    if record["task"] != asdict(task):
        raise ValueError(f"source task mismatch for {task.pose_id}")
    return record


def _run_pose(task: PoseTask) -> dict:
    assert _RUNTIME is not None and _OUTPUT is not None
    source = _load_source_trial(task)
    hybrid = run_method(
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
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
        continue_to_bottom=False,
        conditioned_on_ambiguous_top_contact=True,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed), delay_s=task.sensor_delay_s
        ),
    )
    proposed = source["methods"]["proposed_posterior"]
    fingerprint_match = hybrid.fingerprint_sha256 == proposed["fingerprint_sha256"]
    if not fingerprint_match:
        raise ValueError(f"fingerprint mismatch for {task.pose_id}")
    if hybrid.total_active_time_s > ACTIVE_TIME_BUDGET_S + 1.0e-9:
        raise RuntimeError(f"hard active-time budget violated for {task.pose_id}")
    if hybrid.fallback_budget_extension_s != 0.0:
        raise RuntimeError(f"fallback extended the hard budget for {task.pose_id}")
    record = {
        "task": asdict(task),
        "methods": {
            "proposed_plus_hybrid": asdict(hybrid),
            **{method: source["methods"][method] for method in METHODS[1:]},
        },
        "fairness": {
            "fresh_seed_frozen_before_rollout": True,
            "same_initial_pose_nuisance_and_sensor_seed": True,
            "proposed_plus_and_proposed_fingerprint_match": fingerprint_match,
            "pabc_reused_byte_for_byte": True,
            "online_true_center_or_contact_tags": False,
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "proposed_plus_hard_budget_enforced": True,
            "conditioned_on_ambiguous_top_contact": True,
            "preprobe_route": hybrid.preprobe_route,
        },
        "updated_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _summary(records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    comparisons = {}
    for right in METHODS[1:]:
        comparisons[f"proposed_plus_hybrid_vs_{right}"] = {
            "paired_outcomes": _paired_counts(
                records, "proposed_plus_hybrid", right
            ),
            "paired_efficiency": _paired_efficiency(
                records, "proposed_plus_hybrid", right
            ),
        }
    hybrid = [record["methods"]["proposed_plus_hybrid"] for record in records]
    methods["proposed_plus_hybrid"].update(
        {
            "task_grid_contacts_median": float(
                np.median([value["entropy_grid_probe_count"] for value in hybrid])
            ),
            "local_spiral_count": int(
                sum(bool(value["entropy_grid_used_spiral"]) for value in hybrid)
            ),
            "global_fallback_count": int(
                sum(bool(value["global_outward_fallback_used"]) for value in hybrid)
            ),
            "hard_budget_compliance_count": int(
                sum(
                    float(value["total_active_time_s"])
                    <= ACTIVE_TIME_BUDGET_S + 1.0e-9
                    and float(value.get("fallback_budget_extension_s", 0.0)) == 0.0
                    for value in hybrid
                )
            ),
            "budget_exhaustion_count": int(
                sum(value.get("budget_exhaustion_stage") is not None for value in hybrid)
            ),
        }
    )
    return {
        "stage": "M8_task_aware_fresh_24_validation",
        "design": {
            "poses": len(records),
            "source_seed": 20260723,
            "source": str(DEFAULT_SOURCE.relative_to(ROOT)),
            "frozen_before_rollout": True,
            "same_fingerprint_for_proposed_plus_and_proposed": True,
            "online_true_center_or_contact_tags": False,
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "proposed_plus_budget_is_hard_and_non_extensible": True,
            "conditioned_on_ambiguous_top_contact": True,
            "upstream_online_ambiguity_router_evaluated": False,
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
            "Fresh simulation-only Sobol validation frozen after policy and gate "
            "design. It uses a synthetic visual prior and pre-hardware wrench "
            "nuisance, so it does not establish iiwa, real wear, M10/M12, or "
            "universal superiority claims."
        ),
        "updated_utc": _utc_now(),
    }


def _write_summary(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 Fresh 24-Pose Task-Aware Validation",
        "",
        "| method | success | capture time | total/contact path | wear | hard stop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = report["methods"][method]
        lines.append(
            f"| {method} | {value['successes']}/{value['trials']} | "
            f"{value['time_success_median_s']:.2f} s | "
            f"{value['actual_path_median_mm']:.2f}/{value['contact_path_median_mm']:.2f} mm | "
            f"{value['wear_proxy_median_mj']:.2f} mJ | {value['hard_stops']} |"
        )
    lines.extend(["", "## Boundary", "", report["claim_boundary"], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.joinpath("trials").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tasks = [PoseTask(**item) for item in manifest["tasks"]]
    if args.poses > len(tasks):
        raise ValueError(f"requested {args.poses} poses, source has {len(tasks)}")
    tasks = tasks[: args.poses]
    existing: list[dict] = []
    pending: list[PoseTask] = []
    for task in tasks:
        path = output / "trials" / f"{task.pose_id}.json"
        if args.resume and path.exists():
            existing.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)
    records = list(existing)
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(str(source), str(output)),
        ) as pool:
            futures = {pool.submit(_run_pose, task): task for task in pending}
            with tqdm(
                total=len(tasks),
                initial=len(existing),
                desc="M8 fresh task-aware validation",
                unit="pose",
                dynamic_ncols=True,
            ) as bar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {"task": asdict(task), "error": repr(exc)}
                        _atomic_json(
                            output / "trials" / f"{task.pose_id}.json", record
                        )
                    records.append(record)
                    bar.update(1)
                    valid = [record for record in records if "error" not in record]
                    bar.set_postfix(
                        hybrid=sum(
                            _success(record["methods"]["proposed_plus_hybrid"])
                            for record in valid
                        ),
                        refresh=False,
                    )
    errors = [record for record in records if "error" in record]
    if errors:
        print(json.dumps(errors, indent=2))
        return 1
    records.sort(key=lambda record: record["task"]["pose_id"])
    report = _summary(records)
    _write_summary(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "tasks": [asdict(task) for task in tasks],
            "completed": len(records),
            "source": str(source.relative_to(ROOT)),
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen Proposed+ on a fresh P/A/B/C Sobol source."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poses", type=int, default=24)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
