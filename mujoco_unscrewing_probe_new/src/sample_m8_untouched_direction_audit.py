from __future__ import annotations

import argparse

from src.sample_m8_independent_direction_audit import run as run_sampler
from src.sample_m8_conical_pilot import (
    DEFAULT_P1V4_AUDIT_OUT,
    build_untouched_direction_audit_tasks,
)


def run(args: argparse.Namespace) -> int:
    args.task_builder = build_untouched_direction_audit_tasks
    args.stage_name = "P1v4_M8_untouched_direction_audit"
    args.training_use = "forbidden"
    args.progress_description = "P1v4 M8 untouched audit"
    args.claim_boundary = (
        "Untouched simulation audit generated after P1v3 model and gate freeze. "
        "Forbidden for fitting, calibration, or threshold selection."
    )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the P1v4 untouched M8 audit.")
    parser.add_argument("--output", default=str(DEFAULT_P1V4_AUDIT_OUT))
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
