from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs/current"
DECK = ROOT.parent / "group_meeting_trace/unscrewing_contact_state_pipeline_slides_0709.md"

DATASET_DIRS = (
    "p0a_m8_symmetry",
    "p0b_m8_observability",
    "p0c_m8_collision_followup",
    "p0d_m8_sector_anchors",
    "p1v3_m8_targeted_direction_training",
)
MODEL_DIRS = ("p1_m8_direction_models", "p1v3_m8_direction_models")
EVALUATION_DIRS = (
    "p1v2_m8_independent_direction_audit",
    "p1v4_m8_untouched_direction_audit",
    "p1v5_m8_final_untouched_direction_audit",
    "p2_m8_learned_search_benchmark",
    "p2b_m8_in_domain_coverage",
    "p2b_m8_in_domain_coverage_smoke",
    "p2b_nominal_fsm_acceptance",
)
DIAGNOSTIC_DIRS = ("p2b_nominal_failure_diagnostic",)

TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".yaml", ".yml"}


def _code_path_replacements() -> list[tuple[str, str]]:
    groups = {
        "datasets": DATASET_DIRS,
        "models": MODEL_DIRS,
        "evaluations": EVALUATION_DIRS,
        "diagnostics": DIAGNOSTIC_DIRS,
    }
    replacements: list[tuple[str, str]] = []
    for group, names in groups.items():
        for name in names:
            replacements.append(
                (
                    f'ROOT / "outputs" / "current" / "{name}"',
                    f'ROOT / "outputs" / "current" / "{group}" / "{name}"',
                )
            )
    return replacements


def _move(source: Path, target: Path, changes: list[tuple[str, str]]) -> None:
    if not source.exists():
        return
    if target.exists():
        raise FileExistsError(f"refusing to merge existing target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    changes.append((str(source.relative_to(ROOT)), str(target.relative_to(ROOT))))


def _rewrite_text(path: Path, replacements: list[tuple[str, str]]) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    updated = original
    for old, new in replacements:
        updated = updated.replace(old, new)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def organize() -> dict:
    before_files = sum(1 for path in CURRENT.rglob("*") if path.is_file())
    before_bytes = sum(path.stat().st_size for path in CURRENT.rglob("*") if path.is_file())
    changes: list[tuple[str, str]] = []

    for name in DATASET_DIRS:
        _move(CURRENT / name, CURRENT / "datasets" / name, changes)
    for name in MODEL_DIRS:
        _move(CURRENT / name, CURRENT / "models" / name, changes)
    for name in EVALUATION_DIRS:
        _move(CURRENT / name, CURRENT / "evaluations" / name, changes)
    for name in DIAGNOSTIC_DIRS:
        _move(CURRENT / name, CURRENT / "diagnostics" / name, changes)

    _move(CURRENT / "videos/baselines", CURRENT / "media/videos/baselines", changes)
    if (CURRENT / "videos").exists():
        _move(CURRENT / "videos", CURRENT / "media/videos/stages", changes)
    _move(CURRENT / "comparison_videos", CURRENT / "media/videos/comparisons", changes)
    _move(CURRENT / "video_keyframes", CURRENT / "media/keyframes/stages", changes)
    _move(CURRENT / "comparison_keyframes", CURRENT / "media/keyframes/comparisons", changes)

    for path in sorted(CURRENT.glob("*.png")):
        _move(path, CURRENT / "media/figures" / path.name, changes)
    for path in sorted(CURRENT.glob("*.json")):
        if path.name == "VIDEO_RETENTION.json":
            continue
        _move(path, CURRENT / "reports" / path.name, changes)

    replacements = sorted(
        [
            (old, new)
            for old, new in changes
            if old.startswith("outputs/current/") and new.startswith("outputs/current/")
        ]
        + _code_path_replacements(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    rewrite_roots = [ROOT / "src", ROOT / "tests", ROOT / "docs"]
    rewrite_files = [ROOT / "README.md", ROOT / "CLAIM_BOUNDARY.md", ROOT / "CLEANUP_MANIFEST.md", DECK]
    for directory in rewrite_roots:
        rewrite_files.extend(
            path for path in directory.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    rewrite_files.extend(
        path for path in CURRENT.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES
    )
    rewritten = 0
    for path in dict.fromkeys(rewrite_files):
        if path == Path(__file__).resolve():
            continue
        if path.exists() and _rewrite_text(path, replacements):
            rewritten += 1

    after_files = sum(1 for path in CURRENT.rglob("*") if path.is_file())
    after_bytes = sum(path.stat().st_size for path in CURRENT.rglob("*") if path.is_file())
    root_entries = sorted(path.name for path in CURRENT.iterdir())
    expected = sorted(
        [
            "README.md",
            "VIDEO_RETENTION.json",
            "VIDEO_RETENTION.md",
            "active_comparison",
            "datasets",
            "diagnostics",
            "evaluations",
            "media",
            "models",
            "reports",
        ]
    )
    if root_entries != expected:
        raise RuntimeError(f"unexpected current-output root entries: {root_entries}")
    if before_files != after_files:
        raise RuntimeError(
            f"file preservation failed: files {before_files}->{after_files}"
        )
    return {
        "files": after_files,
        "bytes_before_text_rewrite": before_bytes,
        "bytes": after_bytes,
        "moves": len(changes),
        "rewritten_text_files": rewritten,
        "root_entries": root_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    args = parser.parse_args()
    if not args.apply:
        print("Pass --apply to organize outputs/current; no files changed.")
        return
    print(organize())


if __name__ == "__main__":
    main()
