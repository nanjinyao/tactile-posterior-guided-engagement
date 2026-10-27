from __future__ import annotations

import argparse

from src.sample_m8_independent_direction_audit import run as run_sampler
from src.sample_m8_conical_pilot import (
    DEFAULT_P1V3_TRAIN_OUT,
    build_targeted_direction_training_tasks,
)


def run(args: argparse.Namespace) -> int:
    args.task_builder = build_targeted_direction_training_tasks
    args.stage_name = "P1v3_M8_targeted_direction_training"
    args.training_use = "allowed_after_grouped_split"
    args.progress_description = "P1v3 M8 targeted train"
    args.claim_boundary = (
        "Targeted simulation training/calibration data with grouped spatial splits. "
        "Not an independent test, hardware posterior, or closed-loop benefit result."
    )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the 288-trial M8 targeted training set.")
    parser.add_argument("--output", default=str(DEFAULT_P1V3_TRAIN_OUT))
    parser.add_argument("--spec")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins != 360:
        parser.error("the P1 interface requires exactly 360 phase bins")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
