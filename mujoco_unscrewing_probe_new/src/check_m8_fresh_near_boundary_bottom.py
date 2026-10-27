from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/current/evaluations/m8_fresh_24_task_aware_validation"
OUTPUT = ROOT / "outputs/current/evaluations/m8_fresh_near_boundary_bottom_check"
POSE_IDS = ("sobol_002", "sobol_005", "sobol_020")


def _run(pose_id: str) -> dict:
    source = json.loads(
        (SOURCE / "trials" / f"{pose_id}.json").read_text(encoding="utf-8")
    )
    task = PoseTask(**source["task"])
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    result = run_method(
        "hybrid",
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=runtime,
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
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed), delay_s=task.sensor_delay_s
        ),
    )
    return {
        "pose_id": pose_id,
        "strict_search_result": source["methods"]["proposed_plus_hybrid"],
        "continued_to_bottom": asdict(result),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        futures = {pool.submit(_run, pose_id): pose_id for pose_id in POSE_IDS}
        for future in as_completed(futures):
            records.append(future.result())
            print(f"[{len(records)}/{len(POSE_IDS)}] {futures[future]}", flush=True)
    records.sort(key=lambda record: record["pose_id"])
    report = {
        "design": {
            "selection": "all strict r_cap_top near-boundary misses from the fresh 24-pose validation",
            "parameters_changed_after_validation": False,
            "continue_to_cad_bottom": True,
            "online_true_center_or_contact_tags": False,
        },
        "records": records,
        "cad_bottom_verified": sum(
            record["continued_to_bottom"]["result"] == "cad_bottom_verified"
            for record in records
        ),
        "claim_boundary": (
            "Post-hoc completion diagnostic only. It tests whether strict top-capture "
            "near misses can be recovered by the already frozen post-capture funnel; "
            "it does not change the fresh 21/24 search score."
        ),
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cad_bottom_verified": report["cad_bottom_verified"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
