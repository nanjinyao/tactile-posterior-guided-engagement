from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.finetune_cross_size_s4_state_direction_heads import (
    _head_logits,
    _hidden,
    _load_contract,
    _load_role_examples,
    _softmax,
    probability_metrics,
)
from src.m8_direction_runtime import DirectionEstimate, M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "outputs/current/datasets/cross_size_s4b_adaptation"
DEFAULT_BASE_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
DEFAULT_FINE_TUNED_ROOT = (
    ROOT / "outputs/current/models/cross_size_s4b_state_direction_heads"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/current/models/cross_size_s4b_base_anchored_adapter"
)
SIZES = ("m10", "m12")
SELECTION_ROLES = ("train", "validation")

# Fixed before candidate evaluation.  Alpha is the deployable fine-tuned
# residual weight in p_A = p_B + alpha (p_F - p_B).  Neither zero (a disguised
# base model) nor one (unanchored fine-tuning) is in the candidate set.
RESIDUAL_ALPHA_GRID = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.88,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.98,
)
MIN_FINE_TUNED_CONCENTRATION_GRID = (
    0.00,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)

# Two circular means separated by more than 90 degrees lie in opposite open
# half-planes.  Such a pair does not support a directed shortcut, irrespective
# of labels.  This threshold is a geometric design constant, not a fitted one.
MAX_BASE_FT_DISAGREEMENT_DEG = 90.0
MIN_OVERALL_GATE_COVERAGE = 0.80
# The smaller state strata contain only 12 validation samples.  A 0.70 floor
# preserves at least 9/12 validation examples and, for the 48-sample M12 train
# double-overhang stratum, 35/48 examples.  It rules out abstain-most solutions
# without pretending that every uncertain boundary fingerprint is actionable.
MIN_PER_STATE_GATE_COVERAGE = 0.70
EPSILON = 1.0e-12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _circular_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0)


def _circular_statistics(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=float)
    class_angles = np.deg2rad(
        training._class_angle_deg(np.arange(training.DIRECTION_CLASSES))
    )
    vector = np.sum(probability * np.exp(1j * class_angles), axis=-1)
    return np.mod(np.rad2deg(np.angle(vector)), 360.0), np.abs(vector)


def residual_probability_mix(
    base_probabilities: np.ndarray,
    fine_tuned_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Base-anchored affine residual shrinkage on a probability simplex.

    Because both inputs are normalized non-negative posteriors and
    0 < alpha < 1, this is simultaneously a convex posterior mixture and the
    residual form p_B + alpha (p_F - p_B).  Its first circular moment therefore
    obeys z_A = z_B + alpha (z_F - z_B) exactly.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("residual alpha must be strictly between zero and one")
    base = np.asarray(base_probabilities, dtype=float)
    fine_tuned = np.asarray(fine_tuned_probabilities, dtype=float)
    if base.shape != fine_tuned.shape:
        raise ValueError("base/fine-tuned probability shapes differ")
    if base.ndim < 1:
        raise ValueError("probability arrays must have at least one dimension")
    if np.any(base < -EPSILON) or np.any(fine_tuned < -EPSILON):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(np.sum(base, axis=-1), 1.0, atol=1.0e-7):
        raise ValueError("base probabilities are not normalized")
    if not np.allclose(np.sum(fine_tuned, axis=-1), 1.0, atol=1.0e-7):
        raise ValueError("fine-tuned probabilities are not normalized")
    mixed = base + float(alpha) * (fine_tuned - base)
    mixed = np.clip(mixed, 0.0, None)
    return mixed / np.sum(mixed, axis=-1, keepdims=True)


@dataclass(frozen=True)
class AdapterTelemetry:
    residual_alpha: float
    minimum_fine_tuned_concentration: float
    maximum_base_fine_tuned_disagreement_deg: float
    base_direction_deg: float
    fine_tuned_direction_deg: float
    adapted_direction_deg: float
    base_fine_tuned_disagreement_deg: float
    base_concentration: float
    fine_tuned_concentration: float
    adapted_raw_concentration: float
    conservative_concentration: float
    conservative_span_deg: float
    state_total_variation: float
    aps_union_size: int
    risk_gate_passed: bool
    risk_gate_reasons: tuple[str, ...]


def combine_direction_estimates(
    base: DirectionEstimate,
    fine_tuned: DirectionEstimate,
    *,
    alpha: float,
    minimum_fine_tuned_concentration: float,
    maximum_disagreement_deg: float = MAX_BASE_FT_DISAGREEMENT_DEG,
) -> tuple[DirectionEstimate, AdapterTelemetry]:
    """Combine same-fingerprint estimates without creating extra confidence.

    Control direction and state use the non-zero residual mixture.  Quantities
    that control shortcut length are conservative envelopes: concentration and
    confidence cannot exceed either frozen parent, APS is a union, and span is
    at least the parent disagreement.  A failed risk gate sets direction
    quality to zero through concentration=0 and span=180 degrees; it requests
    the existing finite-budget search rather than a directed shortcut.
    """

    state_probability = residual_probability_mix(
        base.state_probabilities, fine_tuned.state_probabilities, alpha
    )
    direction_probability = residual_probability_mix(
        base.direction_probabilities, fine_tuned.direction_probabilities, alpha
    )
    adapted_direction, adapted_concentration = _circular_statistics(
        direction_probability[None, :]
    )
    adapted_direction_deg = float(adapted_direction[0])
    adapted_raw_concentration = float(adapted_concentration[0])
    disagreement = float(
        _circular_error_deg(
            np.asarray([base.circular_mean_direction_deg]),
            np.asarray([fine_tuned.circular_mean_direction_deg]),
        )[0]
    )
    reasons = []
    if fine_tuned.circular_concentration + EPSILON < minimum_fine_tuned_concentration:
        reasons.append("fine_tuned_concentration_below_train_validation_selected_floor")
    if disagreement > maximum_disagreement_deg + EPSILON:
        reasons.append("base_fine_tuned_means_in_opposite_half_planes")
    gate_passed = not reasons

    conservative_concentration = min(
        float(base.circular_concentration),
        float(fine_tuned.circular_concentration),
        adapted_raw_concentration,
    )
    conservative_span = max(
        float(base.ensemble_top1_span_deg),
        float(fine_tuned.ensemble_top1_span_deg),
        disagreement,
    )
    if not gate_passed:
        conservative_concentration = 0.0
        conservative_span = 180.0

    union_classes = set(map(int, base.aps_direction_classes)) | set(
        map(int, fine_tuned.aps_direction_classes)
    )
    ordered_union = tuple(
        sorted(union_classes, key=lambda index: (-direction_probability[index], index))
    )
    union_angles = tuple(
        map(
            float,
            training._class_angle_deg(np.asarray(ordered_union, dtype=int)),
        )
    )
    top_class = int(np.argmax(direction_probability))
    top_angle = float(training._class_angle_deg(top_class))
    direction_rad = math.radians(adapted_direction_deg)
    state_tv = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(base.state_probabilities)
                - np.asarray(fine_tuned.state_probabilities)
            )
        )
    )
    estimate = DirectionEstimate(
        state_probabilities=state_probability,
        direction_probabilities=direction_probability,
        top_state=training.STATE_NAMES[int(np.argmax(state_probability))],
        top_direction_deg=top_angle,
        circular_mean_direction_deg=adapted_direction_deg,
        direction_xy=np.asarray(
            [math.cos(direction_rad), math.sin(direction_rad)], dtype=float
        ),
        confidence=min(
            float(base.confidence),
            float(fine_tuned.confidence),
            float(direction_probability[top_class]),
        ),
        aps_direction_classes=ordered_union,
        aps_direction_angles_deg=union_angles,
        circular_concentration=conservative_concentration,
        ensemble_top1_span_deg=conservative_span,
        transport_allowed=bool(
            gate_passed
            and fine_tuned.transport_allowed
            and len(ordered_union)
            <= max(
                len(base.aps_direction_classes),
                len(fine_tuned.aps_direction_classes),
            )
        ),
    )
    telemetry = AdapterTelemetry(
        residual_alpha=float(alpha),
        minimum_fine_tuned_concentration=float(
            minimum_fine_tuned_concentration
        ),
        maximum_base_fine_tuned_disagreement_deg=float(maximum_disagreement_deg),
        base_direction_deg=float(base.circular_mean_direction_deg),
        fine_tuned_direction_deg=float(fine_tuned.circular_mean_direction_deg),
        adapted_direction_deg=adapted_direction_deg,
        base_fine_tuned_disagreement_deg=disagreement,
        base_concentration=float(base.circular_concentration),
        fine_tuned_concentration=float(fine_tuned.circular_concentration),
        adapted_raw_concentration=adapted_raw_concentration,
        conservative_concentration=conservative_concentration,
        conservative_span_deg=conservative_span,
        state_total_variation=state_tv,
        aps_union_size=len(ordered_union),
        risk_gate_passed=gate_passed,
        risk_gate_reasons=tuple(reasons),
    )
    return estimate, telemetry


class BaseAnchoredS4bDirectionAdapter:
    """Deployable same-fingerprint base + S4b FT residual adapter.

    This class is intentionally independent of the full-loop runner.  Merely
    constructing or evaluating it cannot launch a simulation or read a
    full-loop task file.
    """

    def __init__(
        self,
        size: str,
        contract_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        if size not in SIZES:
            raise ValueError(f"unsupported size {size!r}")
        self.size = size
        self.contract_root = Path(contract_root).resolve()
        contract_path = self.contract_root / "adapter_contract.json"
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if self.contract.get("status") != "train_validation_selected_frozen":
            raise ValueError("adapter contract is not frozen")
        selected = self.contract["selected_parameters"]
        self.alpha = float(selected["residual_alpha"])
        self.minimum_fine_tuned_concentration = float(
            selected["minimum_fine_tuned_concentration"]
        )
        self.maximum_disagreement_deg = float(
            selected["maximum_base_fine_tuned_disagreement_deg"]
        )
        roots = self.contract["model_roots"]
        self.base = M8DirectionEnsemble(ROOT / roots["base"])
        self.fine_tuned = M8DirectionEnsemble(
            ROOT / roots["fine_tuned_by_size"][size]
        )
        self.last_telemetry: AdapterTelemetry | None = None

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        base = self.base.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
        fine_tuned = self.fine_tuned.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
        estimate, telemetry = combine_direction_estimates(
            base,
            fine_tuned,
            alpha=self.alpha,
            minimum_fine_tuned_concentration=(
                self.minimum_fine_tuned_concentration
            ),
            maximum_disagreement_deg=self.maximum_disagreement_deg,
        )
        self.last_telemetry = telemetry
        return estimate

    def infer_scan(
        self,
        phase_rad: np.ndarray,
        wrench: np.ndarray,
        tip_z_m: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        wrench_values = np.asarray(wrench, dtype=float)
        if wrench_values.ndim != 2 or wrench_values.shape[1] != 6:
            raise ValueError("wrench must have shape (N, 6)")
        values = np.column_stack(
            (wrench_values[:, 2], wrench_values[:, 3], wrench_values[:, 4], tip_z_m)
        )
        sequence = M8DirectionEnsemble._phase_resample(phase_rad, values)
        return self.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )


def _ensemble_probabilities(model_root: Path, examples: dict) -> dict:
    contract_path = model_root / "ensemble_interface_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    state_members = []
    direction_members = []
    checkpoint_records = []
    for checkpoint_value in contract["checkpoints"]:
        checkpoint = _checkpoint_path(checkpoint_value)
        normalization_path = checkpoint.parent / "normalization.npz"
        metrics_path = checkpoint.parent / "metrics.json"
        normalization = np.load(normalization_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = training.CircularDirectionCNN()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        hidden = _hidden(
            model,
            examples,
            np.asarray(normalization["context_mean"], dtype=np.float32),
            np.asarray(normalization["context_std"], dtype=np.float32),
        )
        state_logits, direction_logits = _head_logits(model, hidden)
        state_members.append(
            _softmax(state_logits, float(metrics.get("state_temperature", 1.0)))
        )
        direction_members.append(
            _softmax(direction_logits, float(metrics["temperature"]))
        )
        checkpoint_records.append(
            {
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "normalization": _relative(normalization_path),
                "normalization_sha256": _sha256(normalization_path),
                "metrics": _relative(metrics_path),
                "metrics_sha256": _sha256(metrics_path),
            }
        )
    return {
        "state": np.mean(state_members, axis=0),
        "direction": np.mean(direction_members, axis=0),
        "direction_members": np.stack(direction_members),
        "contract": contract,
        "contract_path": contract_path,
        "checkpoint_records": checkpoint_records,
    }


def _gate_mask(
    base_direction: np.ndarray,
    fine_tuned_direction: np.ndarray,
    minimum_fine_tuned_concentration: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base_mean, _ = _circular_statistics(base_direction)
    fine_tuned_mean, fine_tuned_concentration = _circular_statistics(
        fine_tuned_direction
    )
    disagreement = _circular_error_deg(base_mean, fine_tuned_mean)
    concentration_ok = (
        fine_tuned_concentration + EPSILON
        >= float(minimum_fine_tuned_concentration)
    )
    disagreement_ok = (
        disagreement <= MAX_BASE_FT_DISAGREEMENT_DEG + EPSILON
    )
    return concentration_ok & disagreement_ok, {
        "base_direction_deg": base_mean,
        "fine_tuned_direction_deg": fine_tuned_mean,
        "fine_tuned_concentration": fine_tuned_concentration,
        "base_fine_tuned_disagreement_deg": disagreement,
        "concentration_ok": concentration_ok,
        "disagreement_ok": disagreement_ok,
    }


def _gate_metrics(
    state_probability: np.ndarray,
    direction_probability: np.ndarray,
    examples: dict,
    gate: np.ndarray,
    gate_inputs: dict[str, np.ndarray],
) -> dict:
    raw = probability_metrics(state_probability, direction_probability, examples)
    mean_direction, concentration = _circular_statistics(direction_probability)
    error = _circular_error_deg(mean_direction, examples["angle_deg"])
    accepted = np.asarray(gate, dtype=bool)
    accepted_errors = error[accepted]
    states = np.asarray(examples["state"], dtype=int)
    per_state_coverage = {}
    for state_index, state_name in enumerate(training.STATE_NAMES):
        mask = states == state_index
        if np.any(mask):
            per_state_coverage[state_name] = float(np.mean(accepted[mask]))
    outliers = []
    for index in np.flatnonzero(error > 90.0):
        outliers.append(
            {
                "trial_id": examples["trial_ids"][int(index)],
                "state": training.STATE_NAMES[int(states[index])],
                "true_direction_deg": float(examples["angle_deg"][index]),
                "adapted_direction_deg": float(mean_direction[index]),
                "adapted_error_deg": float(error[index]),
                "accepted_for_directed_shortcut": bool(accepted[index]),
                "fine_tuned_concentration": float(
                    gate_inputs["fine_tuned_concentration"][index]
                ),
                "base_fine_tuned_disagreement_deg": float(
                    gate_inputs["base_fine_tuned_disagreement_deg"][index]
                ),
            }
        )
    return {
        "samples": int(len(error)),
        "raw": raw,
        "raw_control_gt90_count": int(np.sum(error > 90.0)),
        "raw_control_error_median_deg": float(np.median(error)),
        "raw_control_error_p90_deg": float(np.quantile(error, 0.90)),
        "raw_circular_concentration_median": float(np.median(concentration)),
        "accepted_samples": int(np.sum(accepted)),
        "abstained_samples": int(np.sum(~accepted)),
        "gate_coverage": float(np.mean(accepted)),
        "per_state_gate_coverage": per_state_coverage,
        "accepted_control_gt90_count": int(
            np.sum((error > 90.0) & accepted)
        ),
        "accepted_control_error_median_deg": (
            float(np.median(accepted_errors)) if len(accepted_errors) else None
        ),
        "accepted_control_error_p90_deg": (
            float(np.quantile(accepted_errors, 0.90))
            if len(accepted_errors)
            else None
        ),
        "fine_tuned_concentration_min": float(
            np.min(gate_inputs["fine_tuned_concentration"])
        ),
        "base_fine_tuned_disagreement_max_deg": float(
            np.max(gate_inputs["base_fine_tuned_disagreement_deg"])
        ),
        "control_gt90_outliers": outliers,
    }


def _candidate_metrics(
    cached: dict,
    alpha: float,
    minimum_concentration: float,
) -> dict:
    result = {}
    for size, size_roles in cached.items():
        result[size] = {}
        for role, item in size_roles.items():
            state = residual_probability_mix(
                item["base"]["state"], item["fine_tuned"]["state"], alpha
            )
            direction = residual_probability_mix(
                item["base"]["direction"],
                item["fine_tuned"]["direction"],
                alpha,
            )
            gate, gate_inputs = _gate_mask(
                item["base"]["direction"],
                item["fine_tuned"]["direction"],
                minimum_concentration,
            )
            result[size][role] = _gate_metrics(
                state, direction, item["examples"], gate, gate_inputs
            )
    return result


def _frozen_baseline_metrics(cached: dict) -> dict:
    result = {}
    for size, size_roles in cached.items():
        result[size] = {}
        for role, item in size_roles.items():
            result[size][role] = {}
            for variant in ("base", "fine_tuned"):
                probabilities = item[variant]
                result[size][role][variant] = probability_metrics(
                    probabilities["state"],
                    probabilities["direction"],
                    item["examples"],
                    member_direction_probabilities=probabilities[
                        "direction_members"
                    ],
                )
    return result


def _candidate_rejection_reasons(
    metrics: dict,
    frozen: dict,
) -> list[str]:
    reasons = []
    for size in SIZES:
        for role in SELECTION_ROLES:
            candidate = metrics[size][role]
            fine_tuned = frozen[size][role]["fine_tuned"]
            if candidate["accepted_control_gt90_count"] != 0:
                reasons.append(f"{size}/{role}:accepted_control_gt90_nonzero")
            if (
                candidate["raw_control_gt90_count"]
                > fine_tuned["control_direction_error_gt90_count"]
            ):
                reasons.append(f"{size}/{role}:raw_control_gt90_worse_than_ft")
            if (
                candidate["raw"]["state_accuracy"] + EPSILON
                < fine_tuned["state_accuracy"]
            ):
                reasons.append(f"{size}/{role}:state_accuracy_worse_than_ft")
            if candidate["gate_coverage"] + EPSILON < MIN_OVERALL_GATE_COVERAGE:
                reasons.append(f"{size}/{role}:overall_gate_coverage_below_floor")
            if (
                min(candidate["per_state_gate_coverage"].values()) + EPSILON
                < MIN_PER_STATE_GATE_COVERAGE
            ):
                reasons.append(f"{size}/{role}:per_state_gate_coverage_below_floor")
    return reasons


def _load_selection_cache(
    data_root: Path,
    base_root: Path,
    fine_tuned_root: Path,
    sizes: Iterable[str],
) -> tuple[dict, dict]:
    cache = {}
    provenance = {
        "dataset_contracts": {},
        "base_model_contract": {
            "path": _relative(base_root / "ensemble_interface_contract.json"),
            "sha256": _sha256(base_root / "ensemble_interface_contract.json"),
        },
        "fine_tuned_model_contracts": {},
        "execution_files": {},
    }
    for size in sizes:
        contract = _load_contract(data_root, size)
        contract_path = data_root / size / "dataset_contract.json"
        manifest_path = data_root / size / "dataset_manifest.json"
        provenance["dataset_contracts"][size] = {
            "path": _relative(contract_path),
            "sha256": _sha256(contract_path),
            "manifest": _relative(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        }
        cache[size] = {}
        provenance["execution_files"][size] = {}
        for role in SELECTION_ROLES:
            examples = _load_role_examples(data_root, size, contract, role)
            base = _ensemble_probabilities(base_root, examples)
            fine_tuned = _ensemble_probabilities(
                fine_tuned_root / size / "fine_tuned", examples
            )
            cache[size][role] = {
                "examples": examples,
                "base": base,
                "fine_tuned": fine_tuned,
            }
            # These records are identical across roles.  Keep one frozen list
            # per size so the contract is readable rather than duplicative.
            provenance["execution_files"][size] = {
                "base": base["checkpoint_records"],
                "fine_tuned": fine_tuned["checkpoint_records"],
            }
        ft_contract = (
            fine_tuned_root / size / "fine_tuned/ensemble_interface_contract.json"
        )
        provenance["fine_tuned_model_contracts"][size] = {
            "path": _relative(ft_contract),
            "sha256": _sha256(ft_contract),
        }
    return cache, provenance


def _write_summary(output: Path, report: dict) -> None:
    selected = report["selected_parameters"]
    lines = [
        "# S4b base-anchored residual adapter: train + head-held-out-validation selection",
        "",
        (
            f"Selected shared parameters: alpha={selected['residual_alpha']:.2f}, "
            f"minimum FT concentration={selected['minimum_fine_tuned_concentration']:.2f}, "
            f"maximum base-FT mean disagreement="
            f"{selected['maximum_base_fine_tuned_disagreement_deg']:.0f} deg."
        ),
        "",
        "Selection uses development train plus validation held out from FT gradients.",
        "The adapter never reads calibration examples or any full-loop pose labels.",
        "Its raw posterior can still be wrong; the gate only decides whether a",
        "directed shortcut is admissible. A rejected shortcut falls back to the",
        "existing finite-budget search. Full-loop benefit is not established here.",
        "",
        "| size | role | raw state acc | raw control med/p90 | raw >90 | gate coverage | accepted p90 | accepted >90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        for role in SELECTION_ROLES:
            metrics = report["selected_metrics"][size][role]
            lines.append(
                f"| {size.upper()} | {role} | "
                f"{metrics['raw']['state_accuracy']:.3f} | "
                f"{metrics['raw_control_error_median_deg']:.2f}/"
                f"{metrics['raw_control_error_p90_deg']:.2f} deg | "
                f"{metrics['raw_control_gt90_count']} | "
                f"{metrics['gate_coverage']:.3f} | "
                f"{metrics['accepted_control_error_p90_deg']:.2f} deg | "
                f"{metrics['accepted_control_gt90_count']} |"
            )
    lines.extend(
        [
            "",
            "M10 validation retains one raw >90 deg posterior outlier. It is not",
            "magically corrected because base and FT agree on the wrong half-plane;",
            "the deployable FT-concentration guard abstains on that sample, so the",
            "accepted directed-shortcut outlier count is zero.",
            "",
        ]
    )
    content = "\n".join(lines)
    (output / "summary.md").write_text(content, encoding="utf-8")
    # Human-facing selection handoff beside the machine-readable contract.
    (output / "SELECTION.md").write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    base_root = Path(args.base_root).resolve()
    fine_tuned_root = Path(args.fine_tuned_root).resolve()
    output = Path(args.output).resolve()
    sizes = tuple(args.sizes)
    if sizes != SIZES:
        raise ValueError(
            "shared adapter selection requires the frozen joint M10/M12 cohort"
        )
    output.mkdir(parents=True, exist_ok=True)
    cache, provenance = _load_selection_cache(
        data_root, base_root, fine_tuned_root, sizes
    )
    source_path = Path(__file__).resolve()
    provenance["adapter_source"] = {
        "path": _relative(source_path),
        "sha256": _sha256(source_path),
    }
    frozen = _frozen_baseline_metrics(cache)
    candidates = []
    selected = None
    selected_metrics = None
    for alpha in RESIDUAL_ALPHA_GRID:
        for minimum_concentration in MIN_FINE_TUNED_CONCENTRATION_GRID:
            metrics = _candidate_metrics(
                cache, alpha, minimum_concentration
            )
            reasons = _candidate_rejection_reasons(metrics, frozen)
            record = {
                "residual_alpha": alpha,
                "minimum_fine_tuned_concentration": minimum_concentration,
                "maximum_base_fine_tuned_disagreement_deg": (
                    MAX_BASE_FT_DISAGREEMENT_DEG
                ),
                "eligible": not reasons,
                "rejection_reasons": reasons,
            }
            candidates.append(record)
            # Grids are ascending.  The first feasible pair is therefore the
            # maximum base shrinkage, then the least abstention threshold.
            if selected is None and not reasons:
                selected = record
                selected_metrics = metrics
    if selected is None or selected_metrics is None:
        raise RuntimeError("no base-anchored adapter candidate met the frozen rule")

    selected_parameters = {
        "residual_alpha": selected["residual_alpha"],
        "base_anchor_weight": 1.0 - selected["residual_alpha"],
        "minimum_fine_tuned_concentration": selected[
            "minimum_fine_tuned_concentration"
        ],
        "maximum_base_fine_tuned_disagreement_deg": (
            MAX_BASE_FT_DISAGREEMENT_DEG
        ),
    }
    selection_rule = {
        "candidate_order": (
            "ascending residual alpha, then ascending FT concentration floor"
        ),
        "objective": (
            "first feasible candidate = strongest base anchoring and then "
            "maximum shortcut coverage"
        ),
        "hard_constraints_for_every_size_and_train_validation_role": [
            "accepted control-direction error >90 deg count equals zero",
            "raw control-direction >90 deg count no worse than frozen FT",
            "state accuracy no worse than frozen FT",
            f"overall gate coverage >= {MIN_OVERALL_GATE_COVERAGE:.2f}",
            f"each observed state gate coverage >= {MIN_PER_STATE_GATE_COVERAGE:.2f}",
        ],
        "geometric_constant": (
            "base-FT circular means separated by more than 90 deg occupy "
            "opposite open half-planes, so directed transport abstains"
        ),
    }
    report = {
        "stage": "cross_size_S4b_base_anchored_residual_adapter_selection",
        "status": "complete_train_and_head_heldout_validation",
        "selected_parameters": selected_parameters,
        "selection_rule": selection_rule,
        "candidate_grids": {
            "residual_alpha": list(RESIDUAL_ALPHA_GRID),
            "minimum_fine_tuned_concentration": list(
                MIN_FINE_TUNED_CONCENTRATION_GRID
            ),
            "maximum_base_fine_tuned_disagreement_deg": [
                MAX_BASE_FT_DISAGREEMENT_DEG
            ],
        },
        "candidate_eligibility": candidates,
        "frozen_baseline_metrics": frozen,
        "selected_metrics": selected_metrics,
        "data_access_contract": {
            "example_roles_read": list(SELECTION_ROLES),
            "train_role_semantics": (
                "development constraint and adapter selection"
            ),
            "validation_role_semantics": (
                "held out from FT gradient updates, but used for adapter selection; "
                "not an untouched test set"
            ),
            "calibration_examples_read": 0,
            "calibration_labels_read": 0,
            "full_loop_rollouts_read": 0,
            "full_loop_pose_labels_read": 0,
            "untouched_rollouts_read": 0,
            "note": (
                "Frozen base/FT seed metrics include their pre-existing "
                "temperatures; the adapter selector never opens an S4b "
                "calibration trial or uses its labels."
            ),
        },
        "architecture": {
            "state_and_direction_posterior": (
                "p_A = p_B + alpha (p_F - p_B)"
            ),
            "circular_vector_identity": (
                "z_A = z_B + alpha (z_F - z_B)"
            ),
            "shortcut_confidence_envelope": (
                "rho_shortcut = min(rho_B, rho_F, abs(z_A)); "
                "confidence = min(conf_B, conf_F, conf_A)"
            ),
            "shortcut_span_envelope": (
                "Delta_shortcut = max(Delta_B, Delta_F, angle(z_B,z_F))"
            ),
            "aps_policy": "union of frozen base and FT APS class sets",
            "failed_gate_behavior": (
                "set shortcut concentration to zero and span to 180 deg; "
                "retain the finite-budget search rather than directed transport"
            ),
        },
        "provenance": provenance,
        "claim_boundary": (
            "S4b train plus head-training-held-out validation diagnostics only; "
            "validation is the adapter-selection set, not a new test set. The "
            "M10 raw >90-degree posterior outlier remains but is abstained. No "
            "full-loop improvement or untouched pass is claimed."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "selection_report.json", report)
    _write_summary(output, report)

    adapter_contract = {
        "stage": "cross_size_S4b_base_anchored_residual_adapter",
        "status": "train_validation_selected_frozen",
        "sizes": list(SIZES),
        "selected_parameters": selected_parameters,
        "selection_rule": selection_rule,
        "model_roots": {
            "base": _relative(base_root),
            "fine_tuned_by_size": {
                size: _relative(fine_tuned_root / size / "fine_tuned")
                for size in SIZES
            },
        },
        "probability_contract": {
            "same_fingerprint_required": True,
            "posterior_formula": "p_A = p_B + alpha (p_F - p_B)",
            "alpha_strictly_between_zero_and_one": True,
            "fine_tuned_contribution_nonzero": True,
            "base_anchor_contribution_nonzero": True,
            "aps_is_parent_union": True,
            "shortcut_confidence_is_parent_lower_envelope": True,
            "shortcut_span_is_parent_upper_envelope": True,
        },
        "data_access_contract": report["data_access_contract"],
        "provenance": provenance,
        "selection_report": _relative(output / "selection_report.json"),
        "selection_report_sha256": _sha256(output / "selection_report.json"),
        "selection_readme": _relative(output / "SELECTION.md"),
        "selection_readme_sha256": _sha256(output / "SELECTION.md"),
        # Selection is deliberately unable to launch a rollout.  A separate
        # full-loop runner may consume this frozen adapter only after it has
        # frozen the adapter and both parent ensembles into its own execution
        # bundle.  Keeping these two permissions separate avoids interpreting
        # model selection as evaluation authorization.
        "full_loop_execution_authorized": False,
        "selector_may_launch_full_loop": False,
        "eligible_for_separately_frozen_full_loop_evaluation": True,
        "claim_boundary": report["claim_boundary"],
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "adapter_contract.json", adapter_contract)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete_train_and_head_heldout_validation",
            "adapter_contract_sha256": _sha256(
                output / "adapter_contract.json"
            ),
            "selection_report_sha256": _sha256(
                output / "selection_report.json"
            ),
            "summary_sha256": _sha256(output / "summary.md"),
            "selection_readme_sha256": _sha256(output / "SELECTION.md"),
            "calibration_examples_read": 0,
            "full_loop_rollouts_read": 0,
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a same-fingerprint base-anchored S4b posterior adapter "
            "using only the frozen train/validation roles."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--fine-tuned-root", default=str(DEFAULT_FINE_TUNED_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
