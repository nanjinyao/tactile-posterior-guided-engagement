from __future__ import annotations

import argparse

from src.sample_m8_independent_direction_audit import run as run_sampler
from src.sample_m8_conical_pilot import (
    DEFAULT_P1V5_AUDIT_OUT,
    build_final_untouched_direction_audit_tasks,
)


def run(args: argparse.Namespace) -> int:
    args.task_builder = build_final_untouched_direction_audit_tasks
    args.stage_name = "P1v5_M8_final_untouched_direction_audit"
    args.training_use = "forbidden"
    args.progress_description = "P1v5 M8 final audit"
    args.claim_boundary = (
        "Final untouched simulation audit generated after P1v3 model, structured gate, and circular-mean interface freeze. "
        "Forbidden for fitting, calibration, threshold selection, or interface changes."
    )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the final untouched M8 audit.")
    parser.add_argument("--output", default=str(DEFAULT_P1V5_AUDIT_OUT))
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
