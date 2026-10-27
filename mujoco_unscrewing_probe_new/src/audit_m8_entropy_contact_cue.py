from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.entropy_grid_baseline import (
    EntropyGridBelief,
    EntropyGridConfig,
    torque_category_from_wrench,
)
from src.render_m8_dynamic_videos import SIM_TIMESTEP_S, setup
from src.render_m8_search_comparison import NOMINAL_TIP_Z_M, ROOT, comparison_case


DEFAULT_OUT = ROOT / "outputs/current/evaluations/m8_entropy_contact_cue_calibration"


@dataclass(frozen=True)
class CueTask:
    radius_mm: float
    angle_deg: float


def _run(task: CueTask) -> dict:
    spec, model, data, rollout, renderer, _, _ = setup(
        comparison_case("entropy_grid"), render=False
    )
    angle = math.radians(task.angle_deg)
    xy = 0.001 * task.radius_mm * np.array([math.cos(angle), math.sin(angle)])
    tip = np.array([xy[0], xy[1], NOMINAL_TIP_Z_M], dtype=float)
    rollout.set_planar_impedance(1000.0, 100.0)
    rollout.set_target_down_force(3.0)
    rollout.set_initial_tip_pose(tip)
    snapshot = None
    for _ in range(max(1, int(round(0.60 / SIM_TIMESTEP_S)))):
        snapshot = rollout.step(tip, np.zeros(3))
        if snapshot.command.hard_stop or snapshot.state_invalid:
            break
    assert snapshot is not None
    config = EntropyGridConfig()
    observed_category, lever = torque_category_from_wrench(
        snapshot.measured_wrench,
        low_lever_threshold_m=config.low_lever_threshold_m,
        direction_bins=config.torque_direction_bins,
    )
    belief = EntropyGridBelief(
        np.zeros(2),
        visual_sigma_m=0.002,
        visual_bound_m=0.004895,
        support_inner_radius_m=0.5 * float(spec["derived"]["socket_top_af_m"]),
        support_outer_radius_m=0.5 * float(spec["screw_head_diameter_m"]),
        probe_radius_m=float(spec["key_af_m"]) / math.sqrt(3.0),
        config=config,
    )
    true_index = int(np.argmin(np.linalg.norm(belief.candidate_xy_m, axis=1)))
    predicted_probability = belief.category_probabilities(xy)[true_index]
    return {
        **asdict(task),
        "tip_z_mm": 1000.0 * float(snapshot.tip_position[2]),
        "force_n": snapshot.measured_wrench[:3].tolist(),
        "torque_mnm": (1000.0 * snapshot.measured_wrench[3:]).tolist(),
        "normalized_torque_lever_xy_mm": (1000.0 * lever).tolist(),
        "lever_magnitude_mm": 1000.0 * float(np.linalg.norm(lever)),
        "observed_category": observed_category,
        "cad_model_category": int(np.argmax(predicted_probability)),
        "cad_model_probability": predicted_probability.tolist(),
        "hard_stop": bool(snapshot.command.hard_stop or snapshot.state_invalid),
    }


def run(output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        CueTask(radius, angle)
        for angle in (0.0, 90.0, 180.0, 270.0)
        for radius in (2.0, 4.5, 6.0)
    ]
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), mp_context=context
    ) as pool:
        futures = {pool.submit(_run, task): task for task in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="M8 signed F/T cue", unit="pose"
        ):
            records.append(future.result())
    records.sort(key=lambda item: (item["angle_deg"], item["radius_mm"]))
    pairs = []
    for angle in (0.0, 90.0, 180.0, 270.0):
        inner = next(item for item in records if item["angle_deg"] == angle and item["radius_mm"] == 2.0)
        outer = next(item for item in records if item["angle_deg"] == angle and item["radius_mm"] == 6.0)
        pairs.append(
            {
                "angle_deg": angle,
                "both_magnitudes_informative": min(
                    inner["lever_magnitude_mm"], outer["lever_magnitude_mm"]
                ) >= 0.25,
                "signed_categories_distinct": inner["observed_category"]
                != outer["observed_category"],
                "inner_category": inner["observed_category"],
                "outer_category": outer["observed_category"],
            }
        )
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "task": "offline M8 wrist-F/T cue calibration for entropy-grid Baseline C",
        "records": records,
        "inner_outer_pairs": pairs,
        "all_inner_outer_magnitudes_ambiguous": all(
            item["both_magnitudes_informative"] for item in pairs
        ),
        "all_signed_categories_distinct": all(
            item["signed_categories_distinct"] for item in pairs
        ),
        "hard_stops": sum(item["hard_stop"] for item in records),
        "claim_boundary": (
            "Nominal MuJoCo calibration only. It demonstrates why signed torque is needed "
            "instead of magnitude-only edge detection; it is not an iiwa/F-T likelihood calibration."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(Path(args.output).resolve(), args.workers)


if __name__ == "__main__":
    main()
