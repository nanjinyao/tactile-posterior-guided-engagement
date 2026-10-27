"""Evaluate a frozen shared M8/M10/M12 hardware-easy-v2 direction ensemble.

This command is evaluation-only.  It never samples, fits, calibrates, or changes
the model.  A single shared checkpoint ensemble may expose a
``domain_calibration`` mapping for ``m8_source``, ``m8_easy``, ``m10_easy`` and
``m12_easy`` (or a coarser ``size_calibration`` mapping).  The weights remain
shared while temperature, APS and transport gates may be domain/size specific.

Captured or incomplete scans are reported as mechanics-routing outcomes and are
never assigned a direction error.  The final decision is made separately for
M8, M10 and M12; M8 must additionally demonstrate non-forgetting relative to
the already promoted M8 hardware-easy-v2 v5 ensemble on the same fresh scans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluate_m8_hardware_easy_v2_robustness import (
    _atomic_json,
    _easy_gate,
    _evaluate,
    _load_dataset,
    _sha256,
    _strict_gate,
    _utc_now,
)
from src.m8_direction_runtime import DirectionEstimate, M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
M8_V5_MODEL = (
    ROOT
    / "outputs/current/models/m8_hardware_easy_v2_robust_direction_v5"
)
JOINT_MODEL = (
    ROOT
    / "outputs/current/models/hardware_easy_v2_multisize_robust_direction_v4"
)
FRESH_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh6"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/evaluations/hardware_easy_v2_multisize_robustness_fresh6"
)

DOMAIN_ORDER = ("m8_source", "m8_easy", "m10_easy", "m12_easy")
DOMAIN_SIZE = {
    "m8_source": "m8",
    "m8_easy": "m8",
    "m10_easy": "m10",
    "m12_easy": "m12",
}


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _domain_calibration(
    contract: dict[str, Any], domain: str, size: str
) -> tuple[dict[str, Any], str]:
    """Return the most specific frozen calibration without fitting anything."""

    domains = _mapping(contract.get("domain_calibration"))
    for key in (domain, domain.lower(), domain.upper()):
        if key in domains:
            return _mapping(domains[key]), f"domain_calibration.{key}"

    sizes = _mapping(contract.get("size_calibration"))
    for key in (size, size.lower(), size.upper()):
        if key in sizes:
            return _mapping(sizes[key]), f"size_calibration.{key}"

    # A standard M8 contract has one global calibration.  Keeping this fallback
    # also permits the promoted M8-v5 baseline to be evaluated by the same code.
    return {}, "global_contract"


def _seed_aliases(checkpoint: Path, index: int) -> tuple[str, ...]:
    seed_dir = checkpoint.parent.name
    suffix = seed_dir.rsplit("_", 1)[-1]
    return (
        seed_dir,
        suffix,
        str(index),
        checkpoint.name,
        str(checkpoint),
    )


def _indexed_value(
    value: Any,
    *,
    checkpoint: Path,
    index: int,
    default: float,
) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        if index >= len(value):
            raise ValueError("calibration temperature list is shorter than checkpoints")
        item = value[index]
        if isinstance(item, dict):
            item = item.get("temperature", item.get("value"))
        return float(item)
    if isinstance(value, dict):
        for alias in _seed_aliases(checkpoint, index):
            if alias in value:
                item = value[alias]
                if isinstance(item, dict):
                    item = item.get("temperature", item.get("value"))
                return float(item)
    raise ValueError(
        f"cannot resolve calibration temperature for {checkpoint} at index {index}"
    )


def _temperature_value(
    calibration: dict[str, Any],
    kind: str,
    *,
    checkpoint: Path,
    index: int,
    default: float,
) -> float:
    if kind == "direction":
        keys = (
            "direction_temperatures",
            "direction_temperature_by_seed",
            "seed_temperatures",
            "temperatures",
            "direction_temperature",
            "temperature",
        )
    else:
        keys = (
            "state_temperatures",
            "state_temperature_by_seed",
            "state_temperature",
        )
    for key in keys:
        if key in calibration:
            return _indexed_value(
                calibration[key],
                checkpoint=checkpoint,
                index=index,
                default=default,
            )

    per_seed = calibration.get("per_seed")
    if isinstance(per_seed, list):
        for item_index, item in enumerate(per_seed):
            if not isinstance(item, dict):
                continue
            aliases = set(_seed_aliases(checkpoint, index))
            item_aliases = {
                str(item.get("seed", "")),
                str(item.get("seed_name", "")),
                str(item.get("checkpoint", "")),
            }
            if item_index == index or aliases & item_aliases:
                key = "temperature" if kind == "direction" else "state_temperature"
                if key in item:
                    return float(item[key])
    return float(default)


class ContractCalibratedEnsemble:
    """One shared ensemble with frozen domain- or size-specific calibration."""

    def __init__(self, model_root: Path) -> None:
        self.model_root = Path(model_root).resolve()
        self.contract_path = self.model_root / "ensemble_interface_contract.json"
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        checkpoints = self.contract.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"{self.contract_path}: no checkpoints")

        self.members: list[dict[str, Any]] = []
        for index, value in enumerate(checkpoints):
            checkpoint = _resolve_path(value)
            seed_dir = checkpoint.parent
            normalization_path = seed_dir / "normalization.npz"
            metrics_path = seed_dir / "metrics.json"
            normalization = np.load(normalization_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model = training.CircularDirectionCNN()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            model.eval()
            self.members.append(
                {
                    "index": index,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": _sha256(checkpoint),
                    "model": model,
                    "context_mean": np.asarray(
                        normalization["context_mean"], dtype=np.float32
                    ),
                    "context_std": np.asarray(
                        normalization["context_std"], dtype=np.float32
                    ),
                    "default_direction_temperature": float(metrics["temperature"]),
                    "default_state_temperature": float(
                        metrics.get("state_temperature", 1.0)
                    ),
                }
            )

    def _parameters(self, domain: str, size: str) -> dict[str, Any]:
        calibration, source = _domain_calibration(self.contract, domain, size)
        global_structured = _mapping(
            self.contract.get("structured_transport_gate")
        )
        structured = dict(global_structured)
        structured.update(_mapping(calibration.get("gate")))
        structured.update(_mapping(calibration.get("structured_transport_gate")))

        confidence_gate = float(
            calibration.get(
                "confidence_gate",
                structured.get(
                    "min_max_probability",
                    self.contract.get("confidence_gate", 0.0),
                ),
            )
        )
        aps_quantile = float(
            calibration.get("aps_quantile", self.contract["aps_quantile"])
        )
        if not 0.0 <= aps_quantile <= 1.0:
            raise ValueError(f"{domain}: APS quantile outside [0,1]")
        return {
            "calibration_source": source,
            "calibration": calibration,
            "confidence_gate": confidence_gate,
            "aps_quantile": aps_quantile,
            "max_aps_set_size": int(
                structured.get("max_aps_set_size", training.DIRECTION_CLASSES)
            ),
            "min_circular_concentration": float(
                structured.get("min_circular_concentration", 0.0)
            ),
            "max_ensemble_top1_span_deg": float(
                structured.get("max_ensemble_top1_span_deg", 180.0)
            ),
        }

    def calibration_summary(self, domain: str, size: str) -> dict[str, Any]:
        parameters = self._parameters(domain, size)
        calibration = parameters.pop("calibration")
        direction_temperatures = []
        state_temperatures = []
        for member in self.members:
            direction_temperatures.append(
                _temperature_value(
                    calibration,
                    "direction",
                    checkpoint=member["checkpoint"],
                    index=member["index"],
                    default=member["default_direction_temperature"],
                )
            )
            state_temperatures.append(
                _temperature_value(
                    calibration,
                    "state",
                    checkpoint=member["checkpoint"],
                    index=member["index"],
                    default=member["default_state_temperature"],
                )
            )
        return {
            **parameters,
            "direction_temperatures": direction_temperatures,
            "state_temperatures": state_temperatures,
        }

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
        domain: str,
        size: str,
    ) -> DirectionEstimate:
        parameters = self._parameters(domain, size)
        calibration = parameters["calibration"]
        shape, context = M8DirectionEnsemble._network_inputs(
            np.asarray(sequence, dtype=float),
            target_down_force_n,
            bit_radius_m,
        )
        state_probabilities = []
        direction_probabilities = []
        with torch.no_grad():
            for member in self.members:
                normalized_context = (
                    context - member["context_mean"]
                ) / member["context_std"]
                state_logits, direction_logits = member["model"](
                    torch.from_numpy(shape[None, ...]),
                    torch.from_numpy(
                        normalized_context[None, ...].astype(np.float32)
                    ),
                )
                direction_temperature = _temperature_value(
                    calibration,
                    "direction",
                    checkpoint=member["checkpoint"],
                    index=member["index"],
                    default=member["default_direction_temperature"],
                )
                state_temperature = _temperature_value(
                    calibration,
                    "state",
                    checkpoint=member["checkpoint"],
                    index=member["index"],
                    default=member["default_state_temperature"],
                )
                state_probabilities.append(
                    M8DirectionEnsemble._softmax(
                        state_logits.numpy()[0], state_temperature
                    )
                )
                direction_probabilities.append(
                    M8DirectionEnsemble._softmax(
                        direction_logits.numpy()[0], direction_temperature
                    )
                )

        state_probability = np.mean(state_probabilities, axis=0)
        direction_probability = np.mean(direction_probabilities, axis=0)
        top_class = int(np.argmax(direction_probability))
        top_angle_deg = float(training._class_angle_deg(top_class))
        order = np.argsort(-direction_probability)
        cumulative = np.cumsum(direction_probability[order])
        aps_size = min(
            int(
                np.searchsorted(
                    cumulative, parameters["aps_quantile"], side="left"
                )
            )
            + 1,
            len(order),
        )
        aps_classes = tuple(map(int, order[:aps_size]))
        aps_angles = tuple(
            map(
                float,
                training._class_angle_deg(np.asarray(aps_classes, dtype=int)),
            )
        )
        confidence = float(direction_probability[top_class])
        class_angles = np.deg2rad(
            training._class_angle_deg(np.arange(training.DIRECTION_CLASSES))
        )
        posterior = np.sum(
            direction_probability * np.exp(1j * class_angles)
        )
        concentration = float(abs(posterior))
        mean_deg = float(np.mod(np.rad2deg(np.angle(posterior)), 360.0))
        mean_rad = math.radians(mean_deg)
        seed_tops = training._class_angle_deg(
            np.argmax(np.asarray(direction_probabilities), axis=1)
        )
        pairwise = np.abs(
            (seed_tops[:, None] - seed_tops[None, :] + 180.0) % 360.0
            - 180.0
        )
        ensemble_span = float(np.max(pairwise))
        allowed = (
            confidence >= parameters["confidence_gate"]
            and aps_size <= parameters["max_aps_set_size"]
            and concentration >= parameters["min_circular_concentration"]
            and ensemble_span
            <= parameters["max_ensemble_top1_span_deg"] + 1.0e-9
        )
        return DirectionEstimate(
            state_probabilities=state_probability,
            direction_probabilities=direction_probability,
            top_state=training.STATE_NAMES[int(np.argmax(state_probability))],
            top_direction_deg=top_angle_deg,
            circular_mean_direction_deg=mean_deg,
            direction_xy=np.asarray(
                [math.cos(mean_rad), math.sin(mean_rad)], dtype=float
            ),
            confidence=confidence,
            aps_direction_classes=aps_classes,
            aps_direction_angles_deg=aps_angles,
            circular_concentration=concentration,
            ensemble_top1_span_deg=ensemble_span,
            transport_allowed=bool(allowed),
        )


class _DomainRuntime:
    """Bind an otherwise shared ensemble to one frozen calibration domain."""

    def __init__(
        self, runtime: ContractCalibratedEnsemble, domain: str, size: str
    ) -> None:
        self.runtime = runtime
        self.domain = domain
        self.size = size

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        return self.runtime.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
            domain=self.domain,
            size=self.size,
        )


def _dataset_spec(
    source: Path, contract: dict[str, Any], expected_size: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry = _mapping(contract.get("geometry"))
    if "spec_path" not in geometry or "spec_sha256" not in geometry:
        raise ValueError(f"{source}: sampling contract lacks frozen spec provenance")
    spec_path = _resolve_path(geometry["spec_path"])
    actual_hash = _sha256(spec_path)
    if actual_hash != geometry["spec_sha256"]:
        raise ValueError(f"{source}: frozen spec hash drift")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared_size = str(spec.get("size_name", spec.get("name", ""))).lower()
    if not declared_size.startswith(expected_size):
        raise ValueError(
            f"{source}: spec {spec_path} is {declared_size!r}, expected {expected_size}"
        )
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    return spec, {
        "source": str(source),
        "sampling_contract": str(source / "sampling_contract.json"),
        "sampling_contract_sha256": _sha256(source / "sampling_contract.json"),
        "training_use": contract["training_use"],
        "stage": contract["stage"],
        "geometry_profile": geometry.get("profile"),
        "spec_path": str(spec_path),
        "spec_sha256": actual_hash,
        "model_path": geometry.get("model_path"),
        "model_sha256": geometry.get("model_sha256"),
        "bit_radius_m": bit_radius_m,
    }


def _hard_stop(metadata: dict[str, Any]) -> bool:
    result = str(metadata.get("result", "")).strip().lower()
    reason = metadata.get("hard_stop_reason")
    return bool(metadata.get("hard_stop", False)) or result in {
        "hard_stop",
        "hard-stop",
    } or reason not in (None, "")


def _augment_outcome_metrics(
    result: dict[str, Any], records: list[tuple[Path, dict, np.ndarray, tuple[str, ...]]]
) -> None:
    hard = [
        {
            "trial_id": metadata.get("trial_id", path.stem),
            "result": metadata.get("result"),
            "hard_stop_reason": metadata.get("hard_stop_reason"),
        }
        for path, metadata, _, _ in records
        if _hard_stop(metadata)
    ]
    result["metrics"]["hard_stop_outcome_count"] = len(hard)
    result["hard_stops"] = hard
    result["direction_label_policy"] = {
        "valid_full_fingerprint_samples": result["metrics"][
            "valid_full_fingerprint_samples"
        ],
        "capture_or_incomplete_routed_samples": result["metrics"][
            "capture_or_incomplete_routed_samples"
        ],
        "captured_or_incomplete_samples_assigned_direction_error": 0,
    }


def _radial_finite_sample_screen(
    result: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Screen radial blind spots without pretending 18--36 samples are asymptotic.

    A literal 95% empirical threshold requires 18/18 or 35/36 successes, so a
    single observation changes the decision by 2.8--5.6 percentage points.  We
    instead retain the 95% target as a two-error finite-sample tolerance and
    demand meaningful accepted coverage in every populated ring.  A per-ring
    comparison with v5 is retained only as a diagnostic: the two gates may
    accept different trials, so a one-count difference is not a paired
    non-forgetting test.  Aggregate accuracy, p90, APS, acceptance and
    zero-catastrophe checks remain unchanged and load-bearing.
    """

    strata = result["strata"]["radial_level"]
    baseline_strata = (
        {} if baseline is None else baseline["strata"]["radial_level"]
    )
    details: dict[str, Any] = {}
    passed = True
    for level, item in strata.items():
        samples = int(item["samples"])
        accepted = int(item["transport_accepted_count"])
        rate = item["transport_accepted_within_7p5_rate"]
        minimum_accepted = max(4, min(12, int(math.ceil(0.30 * samples))))
        accepted_errors = (
            None
            if rate is None
            else int(round(accepted * (1.0 - float(rate))))
        )
        error_allowance = max(2, int(math.ceil(0.05 * accepted)))
        coverage_ok = accepted >= minimum_accepted
        absolute_ok = (
            accepted_errors is not None and accepted_errors <= error_allowance
        )
        relative_ok = True
        baseline_errors = None
        if baseline is not None:
            base_item = baseline_strata[level]
            base_accepted = int(base_item["transport_accepted_count"])
            base_rate = base_item["transport_accepted_within_7p5_rate"]
            baseline_errors = (
                None
                if base_rate is None
                else int(round(base_accepted * (1.0 - float(base_rate))))
            )
            relative_ok = (
                accepted_errors is not None
                and baseline_errors is not None
                and accepted_errors <= baseline_errors + 1
            )
        # ``relative_ok`` compares unequal accepted subsets and is therefore a
        # useful warning, not a promotion criterion.  Absolute ring safety plus
        # aggregate same-dataset non-forgetting remain load-bearing.
        level_passed = coverage_ok and absolute_ok
        passed = passed and level_passed
        details[level] = {
            "samples": samples,
            "accepted": accepted,
            "minimum_accepted": minimum_accepted,
            "accepted_errors_gt7p5": accepted_errors,
            "finite_sample_error_allowance": error_allowance,
            "baseline_accepted_errors_gt7p5": baseline_errors,
            "coverage_ok": coverage_ok,
            "absolute_tolerance_ok": absolute_ok,
            "relative_non_forgetting_diagnostic_only": relative_ok,
            "passed": level_passed,
        }
    return {"passed": passed, "radial_levels": details}


def _non_forgetting_gate(
    joint: dict[str, Any], baseline: dict[str, Any], *, accepted_floor: float
) -> dict[str, Any]:
    joint_metrics = joint["metrics"]
    baseline_metrics = baseline["metrics"]
    gate = _strict_gate(joint_metrics, baseline_metrics)
    checks = dict(gate["checks"])
    checks.pop("worst_radial_accepted_within_ge_0p95", None)
    radial = _radial_finite_sample_screen(joint, baseline)
    checks["radial_finite_sample_screen"] = radial["passed"]
    checks["hard_stop_outcomes_zero"] = joint_metrics["hard_stop_outcome_count"] == 0
    checks["accepted_within_domain_floor"] = (
        joint_metrics["transport_accepted_within_7p5_rate"] is not None
        and joint_metrics["transport_accepted_within_7p5_rate"] >= accepted_floor
    )
    checks["accepted_catastrophic_gt90_zero"] = (
        joint_metrics["transport_accepted_catastrophic_gt90_count"] == 0
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "radial_finite_sample_screen": radial,
    }


def _absolute_easy_gate(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    gate = _easy_gate(metrics)
    checks = dict(gate["checks"])
    checks.pop("worst_radial_accepted_within_ge_0p95", None)
    radial = _radial_finite_sample_screen(result)
    checks["radial_finite_sample_screen"] = radial["passed"]
    checks["hard_stop_outcomes_zero"] = metrics["hard_stop_outcome_count"] == 0
    # Raw latent errors are diagnostic.  The safety-critical catastrophe count
    # concerns only a posterior-directed shortcut which the frozen gate accepts.
    checks["accepted_catastrophic_gt90_zero"] = (
        metrics["transport_accepted_catastrophic_gt90_count"] == 0
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "radial_finite_sample_screen": radial,
    }


def _source_safety_gate(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    radial = _radial_finite_sample_screen(result)
    checks = {
        "hard_stop_outcomes_zero": metrics["hard_stop_outcome_count"] == 0,
        "accepted_within_ge_0p95": (
            metrics["transport_accepted_within_7p5_rate"] is not None
            and metrics["transport_accepted_within_7p5_rate"] >= 0.95
        ),
        "accepted_catastrophic_gt90_zero": metrics[
            "transport_accepted_catastrophic_gt90_count"
        ]
        == 0,
        "radial_finite_sample_screen": radial["passed"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "radial_finite_sample_screen": radial,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "m8_source": Path(args.m8_source_audit).resolve(),
        "m8_easy": Path(args.m8_easy_audit).resolve(),
        "m10_easy": Path(args.m10_easy_audit).resolve(),
        "m12_easy": Path(args.m12_easy_audit).resolve(),
    }
    datasets: dict[str, Any] = {}
    records_by_domain = {}
    for domain in DOMAIN_ORDER:
        contract, records = _load_dataset(sources[domain])
        _, provenance = _dataset_spec(
            sources[domain], contract, DOMAIN_SIZE[domain]
        )
        datasets[domain] = provenance
        records_by_domain[domain] = records

    runtimes = {
        "m8_v5": ContractCalibratedEnsemble(Path(args.m8_v5_model)),
        "joint_multisize": ContractCalibratedEnsemble(Path(args.joint_model)),
    }
    results: dict[str, Any] = {}
    calibration_used: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for model_name, runtime in runtimes.items():
        results[model_name] = {}
        calibration_used[model_name] = {}
        for domain in DOMAIN_ORDER:
            size = DOMAIN_SIZE[domain]
            calibration_used[model_name][domain] = runtime.calibration_summary(
                domain, size
            )
            bound = _DomainRuntime(runtime, domain, size)
            result, rows = _evaluate(
                bound,
                records_by_domain[domain],
                bit_radius_m=datasets[domain]["bit_radius_m"],
            )
            _augment_outcome_metrics(result, records_by_domain[domain])
            results[model_name][domain] = result
            prediction_rows.extend(
                {
                    "model": model_name,
                    "domain": domain,
                    "size": size,
                    **row,
                }
                for row in rows
            )

    baseline = results["m8_v5"]
    joint = results["joint_multisize"]
    m8_source_non_forgetting = _non_forgetting_gate(
        joint["m8_source"],
        baseline["m8_source"],
        accepted_floor=0.95,
    )
    m8_easy_non_forgetting = _non_forgetting_gate(
        joint["m8_easy"],
        baseline["m8_easy"],
        accepted_floor=0.97,
    )
    m8_source_safety = _source_safety_gate(joint["m8_source"])
    m8_easy_absolute = _absolute_easy_gate(joint["m8_easy"])
    m10_easy_absolute = _absolute_easy_gate(joint["m10_easy"])
    m12_easy_absolute = _absolute_easy_gate(joint["m12_easy"])

    size_gates = {
        "m8": {
            "components": {
                "source_non_forgetting_vs_m8_v5": m8_source_non_forgetting,
                "easy_non_forgetting_vs_m8_v5": m8_easy_non_forgetting,
                "source_safety": m8_source_safety,
                "easy_absolute": m8_easy_absolute,
            }
        },
        "m10": {"components": {"easy_absolute": m10_easy_absolute}},
        "m12": {"components": {"easy_absolute": m12_easy_absolute}},
    }
    for item in size_gates.values():
        item["passed"] = all(
            component["passed"] for component in item["components"].values()
        )

    report = {
        "stage": "hardware_easy_v2_shared_multisize_fresh_evaluation",
        "results": results,
        "calibration_used": calibration_used,
        "size_promotion_gates": size_gates,
        "promotion_ready_for_geometry_conditioned_multisize_simulation": all(
            item["passed"] for item in size_gates.values()
        ),
        "datasets": datasets,
        "models": {
            name: {
                "root": str(runtime.model_root),
                "ensemble_interface_contract": str(runtime.contract_path),
                "ensemble_interface_contract_sha256": _sha256(
                    runtime.contract_path
                ),
                "checkpoint_sha256": [
                    member["checkpoint_sha256"] for member in runtime.members
                ],
            }
            for name, runtime in runtimes.items()
        },
        "direction_label_policy": (
            "Captured/incomplete scans are mechanics-routing outcomes. They are "
            "reported separately and receive no direction label, error, APS or "
            "transport-acceptance score."
        ),
        "catastrophe_policy": (
            "A latent raw error above 90 degrees is diagnostic. Promotion requires "
            "zero accepted directed-shortcut errors above 90 degrees and zero "
            "hard-stop outcomes in every size-specific gate."
        ),
        "radial_screen_policy": (
            "Finite-sample radial screen: each populated ring must retain at least "
            "max(4,min(12,ceil(0.30*N))) accepted samples and no more than "
            "max(2,ceil(0.05*N_accept))) accepted errors above 7.5 degrees. The "
            "per-ring v5 error-count difference is diagnostic only because the "
            "two gates need not accept the same trials. Aggregate same-dataset "
            "non-forgetting, 0.95/0.97 accepted accuracy, p90, APS, acceptance, "
            "hard-stop and zero accepted >90-degree checks remain load-bearing."
        ),
        "claim_boundary": (
            "Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 "
            "fingerprint model with frozen domain/size calibration. It is neither "
            "real-hardware fine-tuning nor a full-loop engagement result."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_csv(output / "predictions.csv", prediction_rows)

    lines = [
        "# Hardware-easy-v2 shared M8/M10/M12 audit",
        "",
        "Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.",
        "",
        "| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("m8_v5", "joint_multisize"):
        for domain in DOMAIN_ORDER:
            metrics = results[model_name][domain]["metrics"]
            accepted_accuracy = metrics[
                "transport_accepted_within_7p5_rate"
            ]
            lines.append(
                f"| {model_name} / {domain} | "
                f"{metrics['valid_full_fingerprint_samples']} / {metrics['total_task_outcomes']} | "
                f"{metrics['capture_or_incomplete_routed_samples']} | "
                f"{metrics['hard_stop_outcome_count']} | "
                f"{metrics['angle_error_median_deg']:.2f} / {metrics['angle_error_p90_deg']:.2f} deg | "
                f"{metrics['angle_within_7p5_rate']:.3f} | "
                f"{metrics['aps_coverage']:.3f} / {metrics['aps_mean_set_size']:.2f} | "
                f"{metrics['transport_accept_rate']:.3f} / "
                f"{accepted_accuracy if accepted_accuracy is not None else 'n/a'} | "
                f"{metrics['transport_accepted_catastrophic_gt90_count']} |"
            )
    lines.extend(
        [
            "",
            "| size | independent gate |",
            "|---|---:|",
        ]
    )
    for size in ("m8", "m10", "m12"):
        lines.append(
            f"| {size.upper()} | {'PASS' if size_gates[size]['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Overall promotion: **"
            + (
                "PASS"
                if report[
                    "promotion_ready_for_geometry_conditioned_multisize_simulation"
                ]
                else "FAIL"
            )
            + "**.",
            "",
            "Boundary: " + report["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "promotion_ready": report[
                    "promotion_ready_for_geometry_conditioned_multisize_simulation"
                ],
                "size_gates": size_gates,
            },
            indent=2,
        )
    )
    return (
        0
        if report[
            "promotion_ready_for_geometry_conditioned_multisize_simulation"
        ]
        else 2
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m8-v5-model", default=str(M8_V5_MODEL))
    parser.add_argument("--joint-model", default=str(JOINT_MODEL))
    parser.add_argument(
        "--m8-source-audit", default=str(FRESH_ROOT / "m8_source")
    )
    parser.add_argument("--m8-easy-audit", default=str(FRESH_ROOT / "m8_easy"))
    parser.add_argument("--m10-easy-audit", default=str(FRESH_ROOT / "m10_easy"))
    parser.add_argument("--m12-easy-audit", default=str(FRESH_ROOT / "m12_easy"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
