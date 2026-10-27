"""Read-only workspace audit; reports stay in the ignored .publication-audit/.

Run with Python 3.10+. No project imports, model loading, or robot connections.
Findings are candidates for review, not a certification that secrets are absent.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".publication-audit"
LOCAL = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
         ".venv", ".venv-cad", ".venv-offline-learning", "node_modules"}
TEXT = {".py", ".m", ".sh", ".ps1", ".json", ".xml", ".yaml", ".yml",
        ".md", ".tex", ".bib", ".txt", ".csv", ".cpp", ".h", ".hpp",
        ".c", ".js", ".mjs", ".toml", ".ini", ".cfg", ".cls"}
PATTERNS = {
    "absolute_path": re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|media)/)[^\s\"'<>]*"),
    "private_network": re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "credential_candidate": re.compile(r"(?i)(?:api[_-]?key|password|passwd|secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*[\"'][^\"']{6,}[\"']|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9]{24,}"),
    "credential_url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    counts = collections.defaultdict(lambda: [0, 0])
    imports = collections.defaultdict(set)
    findings, errors, large, symlinks = [], [], [], []
    source_hashes, duplicate_sizes = {}, collections.defaultdict(list)
    scanned = collections.Counter()
    with (OUT / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "local_environment_or_git"])
        for directory, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(d for d in dirs if d != ".publication-audit")
            for name in sorted(files):
                path = Path(directory) / name
                rel = path.relative_to(ROOT)
                local = bool(set(rel.parts) & LOCAL) or any(p.startswith(".venv") for p in rel.parts)
                try:
                    size = path.stat().st_size
                    writer.writerow([rel.as_posix(), size, local])
                    counts[rel.parts[0]][0] += 1
                    counts[rel.parts[0]][1] += size
                    if path.is_symlink():
                        symlinks.append(rel.as_posix())
                    if local:
                        continue
                    if size >= 100_000_000:
                        large.append({"path": rel.as_posix(), "bytes": size})
                    if size:
                        duplicate_sizes[size].append(path)
                    if path.suffix.lower() not in TEXT or size > 20_000_000:
                        continue
                    content = path.read_text(encoding="utf-8-sig")
                    scanned[path.suffix.lower()] += 1
                    if path.suffix.lower() in {".py", ".m", ".sh", ".tex", ".json", ".xml", ".bib"}:
                        source_hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                    for category, pattern in PATTERNS.items():
                        for match in pattern.finditer(content):
                            findings.append({"path": rel.as_posix(), "line": content.count("\n", 0, match.start()) + 1, "category": category})
                    if path.suffix == ".py":
                        tree = ast.parse(content, filename=str(rel))
                        for node in ast.walk(tree):
                            module = node.module if isinstance(node, ast.ImportFrom) else None
                            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([module] if module else [])
                            for module in names:
                                imports[module.split(".")[0]].add(rel.as_posix())
                    if path.suffix == ".json":
                        json.loads(content)
                except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
                    errors.append({"path": rel.as_posix(), "error": type(exc).__name__, "line": getattr(exc, "lineno", None)})
    duplicates = []
    for size, paths in duplicate_sizes.items():
        if len(paths) < 2 or size < 1024:
            continue
        hashes = collections.defaultdict(list)
        for path in paths:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            hashes[h.hexdigest()].append(path.relative_to(ROOT).as_posix())
        duplicates.extend({"bytes_each": size, "sha256": h, "paths": ps} for h, ps in hashes.items() if len(ps) > 1)
    summary = {"python": sys.version.split()[0], "top_level": dict(counts), "text_files_scanned": dict(scanned),
               "finding_counts": dict(collections.Counter(x["category"] for x in findings)),
               "large_files": large, "parse_or_decode_errors": errors, "symlinks": symlinks,
               "duplicate_groups": len(duplicates), "scope": "Inventory all files; scan UTF-8 text <=20 MB outside environments/Git; AST parse Python and parse JSON. Binary media not privacy-certified."}
    for filename, data in [("summary.json", summary), ("findings.json", findings), ("imports.json", {k: sorted(v) for k, v in imports.items()}),
                           ("source_hashes.json", source_hashes), ("duplicates.json", duplicates)]:
        (OUT / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": sum(x[0] for x in counts.values()), "bytes": sum(x[1] for x in counts.values()), "scanned_text": sum(scanned.values()), "finding_counts": summary["finding_counts"], "errors": len(errors), "large_files": len(large), "duplicate_groups": len(duplicates)}))


if __name__ == "__main__":
    main()
