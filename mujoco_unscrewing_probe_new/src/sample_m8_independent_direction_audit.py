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
import hashlib

from tqdm import tqdm

from src.sample_m8_conical_pilot import (
    DEFAULT_P1V2_OUT,
    PHASE_COLUMNS,
    RAW_COLUMNS,
    MODEL_PATH,
    SPEC_PATH,
    _sample_task,
    _worker_init,
    build_independent_direction_audit_tasks,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    spec_path = Path(getattr(args, "spec", None) or SPEC_PATH).resolve()
    model_path = Path(getattr(args, "model", None) or MODEL_PATH).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_builder = getattr(args, "task_builder", build_independent_direction_audit_tasks)
    stage = getattr(args, "stage_name", "P1v2_M8_frozen_independent_direction_audit")
    training_use = getattr(args, "training_use", "forbidden")
    claim_boundary = getattr(
        args,
        "claim_boundary",
        (
            "Frozen unseen-angle and unseen-nuisance simulation audit. "
            "Never used for model fitting, calibration, gate selection, or active-sampling decisions before evaluation."
        ),
    )
    tasks = task_builder(spec)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    output = Path(args.output or DEFAULT_P1V2_OUT).resolve()
    trials = output / "trials"
    trials.mkdir(parents=True, exist_ok=True)

    existing, pending = [], []
    for task in tasks:
        path = trials / f"{task.trial_id}.npz"
        if args.resume and path.exists():
            existing.append({"trial_id": task.trial_id, "status": "existing", "path": str(path)})
        else:
            pending.append(task)

    _atomic_json(
        output / "sampling_contract.json",
        {
            "stage": stage,
            "total": len(tasks),
            "training_use": training_use,
            "phase_bins": args.phase_bins,
            "geometry": {
                "profile": spec.get("geometry_profile", "frozen_source"),
                "spec_path": str(spec_path),
                "spec_sha256": _sha256(spec_path),
                "model_path": str(model_path),
                "model_sha256": _sha256(model_path),
                "frozen_source_config_sha256": spec.get(
                    "frozen_source_config_sha256"
                ),
                "frozen_source_model_sha256": spec.get(
                    "frozen_source_model_sha256"
                ),
            },
            "tasks": [asdict(task) for task in tasks],
            "claim_boundary": claim_boundary,
            "updated_utc": _utc_now(),
        },
    )

    results = list(existing)
    ok = len(existing)
    failed = 0
    started = time.perf_counter()

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        completed_new = len(results) - len(existing)
        rate = completed_new / elapsed
        remaining = max(len(tasks) - len(results), 0)
        _atomic_json(
            output / "progress.json",
            {
                "stage": stage,
                "total": len(tasks),
                "completed": len(results),
                "succeeded_or_existing": ok,
                "failed": failed,
                "workers": min(args.workers, max(len(pending), 1)),
                "rate_trials_s": rate,
                "eta_s": remaining / rate if rate > 0.0 else None,
                "output_dir": str(output),
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
                args.phase_bins,
                str(model_path),
                str(spec_path),
            ),
        ) as pool:
            futures = {pool.submit(_sample_task, task): task for task in pending}
            bar = tqdm(
                total=len(tasks),
                initial=len(existing),
                desc=getattr(args, "progress_description", "P1v2 M8 frozen audit"),
                unit="trial",
                dynamic_ncols=True,
            )
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"trial_id": task.trial_id, "status": "failed", "error": repr(exc)}
                results.append(result)
                if result["status"] in {"ok", "existing", "skipped"}:
                    ok += 1
                else:
                    failed += 1
                bar.update(1)
                bar.set_postfix(ok=ok, fail=failed, refresh=False)
                checkpoint()
            bar.close()

    result_by_id = {result["trial_id"]: result for result in results}
    _atomic_json(
        output / "manifest.json",
        {
            "stage": stage,
            "tasks": len(tasks),
            "records": [
                {
                    **asdict(task),
                    "status": result_by_id.get(task.trial_id, {}).get("status", "missing"),
                }
                for task in tasks
            ],
            "raw_columns": RAW_COLUMNS,
            "phase_columns": PHASE_COLUMNS,
            "geometry_profile": spec.get("geometry_profile", "frozen_source"),
            "spec_sha256": _sha256(spec_path),
            "model_sha256": _sha256(model_path),
            "claim_boundary": claim_boundary,
            "updated_utc": _utc_now(),
        },
    )
    checkpoint()
    print(
        json.dumps(
            {"output_dir": str(output), "tasks": len(tasks), "ok_or_existing": ok, "failed": failed},
            indent=2,
        )
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the frozen 180-trial M8 direction audit.")
    parser.add_argument("--output")
    parser.add_argument("--spec")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=min(56, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins != 360:
        parser.error("the frozen P1 interface requires exactly 360 phase bins")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
