from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path

from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import (
    BASELINE_MAX_RADIUS_M,
    INITIAL_XY_M,
    run_method,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/current/evaluations/m8_transport_distance_ablation"
MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
CAPS_M = {"cap_1p5_mm": 0.0015, "cap_2p0_mm": 0.0020, "posterior_limited": None}


def _run(case: tuple[str, float | None]) -> tuple[str, dict]:
    name, cap_m = case
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    result = run_method(
        "learned",
        initial_xy_m=INITIAL_XY_M.copy(),
        direction_runtime=runtime,
        visual_sigma_m=0.0020,
        visual_bound_m=0.004895,
        posterior_coverage=0.95,
        posterior_seed=20260710,
        max_transport_m=cap_m,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        active_time_budget_s=90.0,
        continue_to_bottom=True,
    )
    return name, asdict(result)


def _write_report(output: Path, results: dict[str, dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "same_initial_xy_mm": [4.0, 0.0],
            "same_fingerprint_and_frozen_model": True,
            "same_visual_prior_and_soft_cad_likelihood": True,
            "same_search_speed_force_pitch": True,
            "varied_parameter": "absolute posterior-guided transport cap",
            "caps_mm": {
                key: None if value is None else 1000.0 * value
                for key, value in CAPS_M.items()
            },
        },
        "results": results,
        "claim_boundary": (
            "Single deterministic M8 case used to choose a safe transport-cap policy. "
            "It is an ablation, not a distributional or hardware performance claim."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# M8 Posterior Transport Distance Ablation",
        "",
        "| policy | transport | local radius | bottom | time | XY path | wear proxy | raw / filtered peak force | peak torque |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CAPS_M:
        value = results[name]
        lines.append(
            f"| {name} | {value['transport_distance_mm']:.3f} mm | "
            f"{value['spiral_max_radius_mm']:.3f} mm | {value['bottom_reached']} | "
            f"{value['total_active_time_s']:.2f} s | {value['contact_xy_path_mm']:.2f} mm | "
            f"{value['wear_proxy_mj']:.2f} mJ | {value['max_force_n']:.2f} / "
            f"{value['max_filtered_force_n']:.2f} N | {value['max_torque_mnm']:.2f} mNm |"
        )
    lines.extend(
        [
            "",
            "The posterior-limited row removes only the fixed absolute cap; the lower-quantile "
            "transport rule and the residual 95% spiral coverage bound remain active.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    cases = list(CAPS_M.items())
    results: dict[str, dict] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)), mp_context=context) as pool:
        futures = {pool.submit(_run, case): case[0] for case in cases}
        with tqdm(total=len(cases), desc="M8 transport cap", unit="rollout", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                bar.update(1)
                bar.set_postfix(policy=name, force=f"{result['max_force_n']:.2f}N", refresh=False)
    ordered = {name: results[name] for name in CAPS_M}
    _write_report(output, ordered)
    print(json.dumps(ordered, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate the M8 posterior transport cap.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=min(3, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
