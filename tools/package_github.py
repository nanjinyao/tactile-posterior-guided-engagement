"""Build a curated GitHub snapshot without changing research artifacts.

Usage: python tools/package_github.py --output D:/path/outside/workspace
Standard library only. Refuses an existing output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "nanjinyao/tactile-posterior-guided-engagement"
TAG = "research-snapshot-2026-10-27"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "node_modules", "venv", "tmp", "archive", "reference", "skills"}
ROOT_FILES = {"README.md", "REPRODUCIBILITY.md", "PUBLICATION_AUDIT.md", "requirements.txt"}
ROOT_TREES = {"mujoco_unscrewing_probe_new", "real_device", "overleaf_unscrewing", "docs", "tools"}
DECKS = {"Learning_Based_Tactile_Posterior_Guided_Search_Compact_39_Slides.pptx",
         "Learning_Based_Tactile_Posterior_Guided_Search_Group_Review.pptx"}
AUX = {".aux", ".blg", ".fls", ".bbl", ".log", ".out", ".toc", ".nav", ".snm", ".vrb", ".lof", ".lot", ".guard"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bsk-[A-Za-z0-9]{24,}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if len(rel.parts) == 1:
        return rel.name in ROOT_FILES
    if rel.parts[0] == "outputs":
        return len(rel.parts) == 2 and rel.name in DECKS
    if rel.parts[0] not in ROOT_TREES:
        return False
    if any(p in SKIP_DIRS or p.startswith(".") or p.startswith("venv") for p in rel.parts[:-1]):
        return False
    if rel.name.startswith((".", "~$")) or "sync-conflict" in rel.name:
        return False
    if path.suffix.lower() in AUX or ".bak" in rel.name or rel.name.endswith(".synctex.gz"):
        return False
    if rel.parts[0] == "docs" and rel.name == "SNAPSHOT_MANIFEST.json":
        return False
    if rel.parts[0] == "real_device" and path.suffix.lower() == ".pptx":
        return False
    return True


def group_for(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    if path.suffix.lower() not in {".npz", ".pt", ".mp4"} and path.stat().st_size < 40 * 1024**2:
        return None
    if path.suffix.lower() == ".mp4":
        return "videos"
    for group in ("datasets", "evaluations", "diagnostics", "models"):
        if group in rel.parts:
            return group
    return "hardware"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents or out.exists():
        ap.error("Output must be a new directory outside the source workspace")
    snapshot = out / "repository"
    assets = out / "assets"
    snapshot.mkdir(parents=True)
    assets.mkdir()
    records, zips = [], {}
    credential_hits = []
    try:
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            if Path(base) == ROOT:
                dirs[:] = [d for d in dirs if d in ROOT_TREES or d == "outputs"]
            for name in sorted(files):
                src = Path(base) / name
                if not selected(src):
                    continue
                if src.is_symlink():
                    raise ValueError(f"Review symlink before packaging: {src.relative_to(ROOT)}")
                rel = src.relative_to(ROOT).as_posix()
                size = src.stat().st_size
                if src.suffix.lower() in {".py", ".m", ".sh", ".json", ".yaml", ".yml", ".txt", ".md", ".xml", ".tex", ".csv"} and size < 20_000_000:
                    if SECRET.search(src.read_bytes()):
                        credential_hits.append(rel)
                group = group_for(src)
                before = digest(src)
                if group:
                    if group not in zips:
                        filename = f"artifacts-{group}.zip"
                        zips[group] = zipfile.ZipFile(assets / filename, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True)
                        print(f"Packaging {group}", flush=True)
                    zips[group].write(src, arcname=rel)
                else:
                    dst = snapshot / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    if digest(dst) != before:
                        raise ValueError(f"Copy hash mismatch: {rel}")
                if src.stat().st_size != size or digest(src) != before:
                    raise ValueError(f"Source changed while packaging: {rel}")
                records.append({"path": rel, "bytes": size, "sha256": before, "storage": group or "git"})
    finally:
        for z in zips.values():
            z.close()
    if credential_hits:
        raise ValueError("Credential-pattern candidates require review: " + ", ".join(credential_hits))
    archives = []
    for group in sorted(zips):
        path = assets / f"artifacts-{group}.zip"
        archives.append({"group": group, "asset": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"Archive CRC failure: {bad}")
            for info in z.infolist():
                h = hashlib.sha256()
                with z.open(info) as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(block)
                expected = next(r for r in records if r["path"] == info.filename)
                if h.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Archived file hash mismatch: {info.filename}")
        print(f"Verified {path.name}: {path.stat().st_size:,} bytes", flush=True)
    manifest = {"schema": 1, "snapshot_date": "2026-10-27", "repository": REPOSITORY, "release_tag": TAG,
                "source_root": "group_meeting", "archives": archives, "files": records,
                "manifest_excludes_itself": True,
                "excluded": ["local environments, caches, editor/agent/sync metadata", "nested Git histories",
                             "third-party reference papers and independent reference projects", "temporary/archived/conflict/backup files",
                             "duplicate 1.32 GB experimental slides", "historical top-level traces and unselected meeting drafts"]}
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (snapshot / "docs/SNAPSHOT_MANIFEST.json").write_text(text, encoding="utf-8")
    (ROOT / "docs/SNAPSHOT_MANIFEST.json").write_text(text, encoding="utf-8")
    (snapshot / ".gitignore").write_text("**/.venv*/\n**/venv/\n**/__pycache__/\n**/.pytest_cache/\n**/.vscode/\n**/.idea/\n/.artifact-downloads/\n/.publication-audit/\n*.py[cod]\n~$*\n*.aux\n*.blg\n*.fls\n*.fdb_latexmk\n*.synctex.gz\n*.log\n# Restored Release artifacts are recorded in docs/SNAPSHOT_MANIFEST.json.\n*.npz\n*.pt\n*.mp4\n", encoding="utf-8")
    with (snapshot / ".gitignore").open("a", encoding="utf-8") as f:
        for record in records:
            if record["storage"] != "git" and Path(record["path"]).suffix not in {".npz", ".pt", ".mp4"}:
                f.write("/" + record["path"] + "\n")
    (assets / "SHA256SUMS.txt").write_text("".join(f"{a['sha256']}  {a['asset']}\n" for a in archives), encoding="utf-8")
    print(json.dumps({"snapshot": str(snapshot), "files": len(records), "bytes": sum(r["bytes"] for r in records),
                      "storage_counts": dict(Counter(r["storage"] for r in records)), "archives": archives}, indent=2), flush=True)


if __name__ == "__main__":
    main()
