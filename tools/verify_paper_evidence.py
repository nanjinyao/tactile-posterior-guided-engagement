"""Verify frozen paper evidence without importing the simulator or robot stack.

Python 3.10+; standard library only. Prints JSON, writes no project artifacts.
This checks stored results, not the scientific validity of a fresh experiment.
"""
from pathlib import Path
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "mujoco_unscrewing_probe_new"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    paper = ROOT / "overleaf_unscrewing"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", (paper / "main.tex").read_text(encoding="utf-8"))
    for figure in figures:
        require((paper / figure).is_file(), f"Missing figure: {figure}")
    folder = SIM / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
    manifest = read(folder / "manifest.json")
    for filename, key in [("report.json", "report_sha256"), ("summary.md", "summary_sha256"), ("task_contract.json", "task_contract_sha256")]:
        require(hashlib.sha256((folder / filename).read_bytes()).hexdigest() == manifest[key], f"Hash mismatch: {filename}")
    contract = read(folder / "task_contract.json")
    for name, item in contract["model_contracts"].items():
        require(hashlib.sha256((SIM / item["path"]).read_bytes()).hexdigest() == item["sha256"], f"Model contract mismatch: {name}")
    require(hashlib.sha256((SIM / contract["source_manifest"]).read_bytes()).hexdigest() == contract["source_manifest_sha256"], "Source task manifest mismatch")
    records = [read(p) for p in sorted((folder / "trials").glob("sobol_*.json"))]
    require(len(records) == 24, "Expected 24 paired starts")
    expected = {"proposed": 20, "proposed_plus": 19, "outward": 19, "improved_direct_retry3": 17, "jin_direct_retry3": 16}
    report = read(folder / "report.json")
    counts = {}
    for method, count in expected.items():
        counts[method] = sum(r["methods"][method]["result"] == "cad_bottom_verified" for r in records)
        require(counts[method] == count == report["methods"][method]["strict_bottom_successes"], f"Count mismatch: {method}")
    paired = []
    for record in records:
        a, b = record["methods"]["proposed"], record["methods"]["outward"]
        if a["result"] == b["result"] == "cad_bottom_verified":
            paired.append(a["actual_xy_path_mm"] - b["actual_xy_path_mm"])
    require(len(paired) == 19 and sum(x < 0 for x in paired) == 19, "Paired path claim mismatch")
    require(round(statistics.median(paired), 2) == -41.14, "Paired path median mismatch")
    recovery = read(SIM / "outputs/current/evaluations/m8_recoverable_funnel_r3_ablation/report.json")
    require(recovery["methods"]["rcc_shield_r0"]["bottom_verified"] == 0, "R0 mismatch")
    require(recovery["methods"]["rcc_shield_r3"]["bottom_verified"] == 9, "R3 mismatch")
    runs = ["M8_conical_fingerprints_26pt_20260728_161249_458", "M10_conical_fingerprints_26pt_20260728_152808_877", "M12_conical_fingerprints_26pt_20260728_175502_318"]
    for run in runs:
        require((ROOT / "real_device/data" / run / "sampling_campaign.json").is_file(), f"Missing acquisition campaign: {run}")
    print(json.dumps({"status": "PASS", "included_figures": len(figures), "m8_verified": counts, "common_successes": len(paired), "median_paired_path_difference_mm": statistics.median(paired), "recovery_r0_r3": [0, 9], "acquisition_campaigns_located": runs, "scope": "Stored evidence and selected numerical claims only; no new simulation, inference or hardware validation."}, indent=2))


if __name__ == "__main__":
    main()
