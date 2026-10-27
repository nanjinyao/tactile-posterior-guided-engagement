from __future__ import annotations

import argparse

from src.sample_m8_independent_direction_audit import run as run_sampler
from src.sample_m8_conical_pilot import (
    DEFAULT_P1V6_JIN_COMPARISON_OUT,
    build_jin_comparison_direction_audit_tasks,
)


def run(args: argparse.Namespace) -> int:
    args.task_builder = build_jin_comparison_direction_audit_tasks
    args.stage_name = "P1v6_M8_Jin_style_vs_improved_direction_audit"
    args.training_use = "forbidden_for_both"
    args.progress_description = "P1v6 M8 Jin-style comparison audit"
    args.claim_boundary = (
        "Fresh paired simulation direction audit frozen after both the Jin-2021-style M8 proxy and "
        "improved estimator were fixed. The trials are forbidden for fitting, calibration, threshold "
        "selection, architecture changes, or model selection for either estimator."
    )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the frozen paired P1v6 M8 Jin-style-vs-improved direction audit."
    )
    parser.add_argument("--output", default=str(DEFAULT_P1V6_JIN_COMPARISON_OUT))
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins != 360:
        parser.error("the frozen P1v6 interface requires exactly 360 phase bins")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
