"""Restore private GitHub Release artifacts and verify the snapshot (Python 3.10+)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def destination(name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe artifact path: {name}")
    path = (ROOT / Path(*rel.parts)).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"Artifact escapes repository: {name}")
    return path


def verify(manifest: dict) -> int:
    missing, mismatched = [], []
    checked = 0
    for record in manifest["files"]:
        path = destination(record["path"])
        if not path.is_file():
            missing.append(record["path"])
        elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            mismatched.append(record["path"])
        else:
            checked += 1
    print(json.dumps({"verified_files": checked, "missing_count": len(missing), "mismatched_count": len(mismatched),
                      "missing_examples": missing[:15], "mismatched": mismatched}, ensure_ascii=False, indent=2))
    return 1 if missing or mismatched else 0


def restore(archive: dict, records: list[dict], local_assets: Path | None, manifest: dict) -> None:
    cache = ROOT / ".artifact-downloads"
    cache.mkdir(exist_ok=True)
    if local_assets:
        source = local_assets / archive["asset"]
    else:
        source = cache / archive["asset"]
        if not source.is_file() or digest(source) != archive["sha256"]:
            # gh uses the existing private-repository credentials without printing tokens.
            subprocess.run(["gh", "release", "download", manifest["release_tag"], "--repo", manifest["repository"],
                            "--pattern", archive["asset"], "--dir", str(cache), "--clobber"], check=True)
    if source.stat().st_size != archive["bytes"] or digest(source) != archive["sha256"]:
        raise ValueError(f"Archive hash/size mismatch: {source.name}")
    expected = {r["path"]: r for r in records}
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f"Archive membership mismatch: {source.name}")
        # Validate every member and all conflicts before writing repository files.
        for info in z.infolist():
            path = destination(info.filename)
            record = expected[info.filename]
            if info.file_size != record["bytes"]:
                raise ValueError(f"Member size mismatch: {info.filename}")
            if path.exists() and (not path.is_file() or digest(path) != record["sha256"]):
                raise ValueError(f"Refusing to overwrite different local content: {info.filename}")
            h = hashlib.sha256()
            with z.open(info) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != record["sha256"]:
                raise ValueError(f"Member hash mismatch: {info.filename}")
        for info in z.infolist():
            path = destination(info.filename)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                    temp = Path(f.name)
                    with z.open(info) as src:
                        shutil.copyfileobj(src, f)
                temp.replace(path)
            finally:
                if temp and temp.exists():
                    temp.unlink()
    print(f"Restored {archive['group']}: {len(expected)} verified files", flush=True)


def main() -> int:
    manifest = json.loads((ROOT / "docs/SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    groups = [a["group"] for a in manifest["archives"]]
    ap = argparse.ArgumentParser(description=__doc__)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--group", nargs="+", choices=groups)
    choice.add_argument("--verify", action="store_true")
    ap.add_argument("--local-assets", type=Path, help="Use already downloaded archives from this directory")
    args = ap.parse_args()
    if args.verify:
        return verify(manifest)
    wanted = groups if args.all else args.group
    for archive in manifest["archives"]:
        if archive["group"] in wanted:
            records = [r for r in manifest["files"] if r["storage"] == archive["group"]]
            restore(archive, records, args.local_assets, manifest)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
