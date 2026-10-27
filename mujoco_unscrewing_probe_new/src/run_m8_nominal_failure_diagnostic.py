from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
from src.run_m8_p2b_coverage_benchmark import (
    ACTIVE_GUARD_BUDGET_S,
    BASELINE_MAX_RADIUS_M,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs"
    / "current"
    / "p2b_m8_in_domain_coverage"
    / "trials"
    / "supported_search_r04_s03_learned.json"
)
OUT = ROOT / "outputs" / "current" / "diagnostics" / "p2b_nominal_failure_diagnostic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", choices=("learned", "baseline"))
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    task = source["task"]
    runtime = M8DirectionEnsemble()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    results = dict(existing.get("results", {}))
    methods = tuple(args.method or ("learned", "baseline"))
    with tqdm(total=len(methods), desc="nominal failure trace", unit="policy") as bar:
        for method in methods:
            result = run_method(
                method,
                initial_xy_m=np.array([task["x_m"], task["y_m"]]),
                direction_runtime=runtime if method == "learned" else None,
                visual_sigma_m=VISUAL_SIGMA_M,
                visual_bound_m=VISUAL_BOUND_M,
                posterior_coverage=POSTERIOR_COVERAGE,
                posterior_seed=int(task["posterior_seed"]),
                baseline_phase_rad=float(task["baseline_phase_rad"]),
                baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
                active_time_budget_s=ACTIVE_GUARD_BUDGET_S,
                trace_output_path=OUT / f"{method}_trace.csv",
            )
            results[method] = asdict(result)
            bar.update(1)
            bar.set_postfix(method=method, result=result.result, refresh=False)
    payload = {
        "source_task": task,
        "results": results,
        "claim_boundary": (
            "Single nominal M8 failure diagnostic. Contact tags are logged only as offline truth; "
            "the current online gates still contain privileged tags and are not hardware-ready."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
