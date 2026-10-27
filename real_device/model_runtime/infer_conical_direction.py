#!/usr/bin/env python3
"""Frozen phase-boundary inference for promoted real conical models.

This process never opens ROS, KST, a robot socket, or a sensor device.  It
accepts one completed conical fingerprint, verifies the frozen model bundle,
runs the CPU ensemble once, and writes a provenance-rich JSON high-level
proposal.  The promoted default may select only a CAD-bounded direction,
shortcut, residual target, or corridor; MATLAB retains all physical guards,
capture, insertion, and terminal authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REAL_DEVICE_ROOT = HERE.parent
WORKSPACE_ROOT = REAL_DEVICE_ROOT.parents[1]
SIM_ROOT = WORKSPACE_ROOT / "group_meeting" / "mujoco_unscrewing_probe_new"
DEFAULT_MODEL_ROOT = HERE / "bundles" / "real_multisize_direction_v1"
DEFAULT_MANIFEST = DEFAULT_MODEL_ROOT / "deployment_manifest.json"

DIRECTION_BIN_DEG = 7.5
DIRECTION_OFFSET_DEG = 3.75
DIRECTION_CLASSES = 48
STATE_NAMES = (
    "supported_search",
    "double_overhang",
    "outer_partial_support",
)
SIZE_ORDER = ("m8", "m10", "m12")
SIZE_BIT_RADIUS_M = {
    "m8": 0.006 / math.sqrt(3.0),
    "m10": 0.008 / math.sqrt(3.0),
    "m12": 0.010 / math.sqrt(3.0),
}
M12_BIT_RADIUS_M = SIZE_BIT_RADIUS_M["m12"]
TARGET_FORCE_REFERENCE_N = 3.5
M10_FIXED_DEMO_PILOT_PATH = (
    REAL_DEVICE_ROOT / "configs" / "m10_fixed_demo_pilot_authorization_v1.json"
).resolve()
M10_FIXED_DEMO_PILOT_SHA256 = (
    "8f09bdc86949ee448d29795ca7100347be4b865083f93dfa7761eae9834dc825"
)
M10_FIXED_DEMO_PILOT_ID = "M10_FIXED_PROPOSED_PLUS_DEMO_20260730_V1"


class CircularDirectionCNN(nn.Module):
    """Inference-only copy of the frozen training architecture.

    Keeping this tiny definition local prevents the deployment lane from
    importing the training stack (MuJoCo, scikit-learn, joblib, tqdm).
    State-dict and artifact hashes still bind the actual learned parameters.
    """

    def __init__(self, context_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 24, 9, padding=4, padding_mode="circular"),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, 7, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(45),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 45 + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.state_head = nn.Linear(128, len(STATE_NAMES))
        self.direction_head = nn.Linear(128, DIRECTION_CLASSES)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sequence).flatten(1)
        hidden = self.shared(torch.cat((encoded, context), dim=1))
        return self.state_head(hidden), self.direction_head(hidden)


class RealTinyCircularCNN(nn.Module):
    """Exact deployment copy of REAL_TINY_CIRCULAR_CNN_V1."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(7, 16, 9, padding=4, padding_mode="circular"),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(16, 24, 7, padding=3, padding_mode="circular"),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(24, 24, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(30),
        )
        self.shared = nn.Sequential(
            nn.Linear(24 * 30 + 5, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.state_head = nn.Linear(48, len(STATE_NAMES))
        self.direction_head = nn.Linear(48, DIRECTION_CLASSES)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sequence).flatten(1)
        hidden = self.shared(torch.cat((encoded, context), dim=1))
        return self.state_head(hidden), self.direction_head(hidden)


class RealFourierLinear(nn.Module):
    """Deployment copy of REAL_FOURIER_LINEAR_V1."""

    def __init__(self, harmonics: int = 6) -> None:
        super().__init__()
        self.harmonics = int(harmonics)
        feature_dim = 7 * 2 * self.harmonics + 5
        self.state_head = nn.Linear(feature_dim, len(STATE_NAMES))
        self.direction_head = nn.Linear(feature_dim, DIRECTION_CLASSES)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(sequence, dim=2) / sequence.shape[2]
        selected = spectrum[:, :, 1 : self.harmonics + 1]
        fourier = torch.cat((selected.real, selected.imag), dim=2).flatten(1)
        features = torch.cat((fourier, context), dim=1)
        return self.state_head(features), self.direction_head(features)


def class_angle_deg(class_index: np.ndarray | int) -> np.ndarray:
    return (
        DIRECTION_OFFSET_DEG
        + DIRECTION_BIN_DEG * np.asarray(class_index)
    ) % 360.0


def smoothed_standardized(values: np.ndarray, window: int = 7) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if window <= 1:
        smoothed = vector.copy()
    else:
        half = window // 2
        extended = np.concatenate((vector[-half:], vector, vector[:half]))
        smoothed = np.convolve(
            extended, np.ones(window, dtype=float) / window, mode="valid"
        )
    return (smoothed - np.mean(smoothed)) / max(
        float(np.std(smoothed)), 1.0e-9
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def circular_distance_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((first - second + 180.0) % 360.0 - 180.0)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled -= np.max(scaled)
    probability = np.exp(scaled)
    return probability / np.sum(probability)


def phase_resample(
    phase_rad: np.ndarray, values: np.ndarray, bins: int = 360
) -> np.ndarray:
    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * math.pi)
    samples = np.asarray(values, dtype=float)
    if phase.ndim != 1 or samples.ndim != 2 or samples.shape[0] != len(phase):
        raise ValueError("phase must be (N,) and values must be (N,C)")
    finite = np.isfinite(phase) & np.all(np.isfinite(samples), axis=1)
    phase = phase[finite]
    samples = samples[finite]
    if len(phase) < 32:
        raise ValueError("fewer than 32 finite conical samples")
    order = np.argsort(phase)
    phase = phase[order]
    samples = samples[order]
    phase, unique = np.unique(phase, return_index=True)
    samples = samples[unique]
    if len(phase) < 32:
        raise ValueError("fewer than 32 unique conical phases")
    phase_extended = np.concatenate(
        (phase[-1:] - 2.0 * math.pi, phase, phase[:1] + 2.0 * math.pi)
    )
    grid = np.linspace(0.0, 2.0 * math.pi, bins, endpoint=False)
    channels: list[np.ndarray] = []
    for index in range(samples.shape[1]):
        values_extended = np.concatenate(
            (samples[-1:, index], samples[:, index], samples[:1, index])
        )
        channels.append(np.interp(grid, phase_extended, values_extended))
    return np.vstack(channels)


def network_inputs(
    sequence: np.ndarray,
    requested_size: str,
    target_down_force_n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact raw 7x360 sequence and five-value training context."""

    if sequence.shape != (7, 360):
        raise ValueError("real model sequence must have shape (7,360)")
    if requested_size not in SIZE_ORDER:
        raise ValueError(f"unsupported size {requested_size!r}")
    raw = np.asarray(sequence, dtype=np.float32).copy()
    # The training builder stores commanded Z relative to phase-bin zero.
    raw[6] -= raw[6, 0]
    context = np.zeros(5, dtype=np.float32)
    context[SIZE_ORDER.index(requested_size)] = 1.0
    context[3] = (
        float(target_down_force_n) / TARGET_FORCE_REFERENCE_N
    )
    context[4] = SIZE_BIT_RADIUS_M[requested_size] / M12_BIT_RADIUS_M
    return raw, context


def _verify_hardware_authority(
    manifest: dict[str, Any],
    manifest_path: Path,
    model_root: Path,
) -> bool:
    authority = manifest.get("hardware_authority", {})
    default = str(authority.get("default", ""))
    allowed = tuple(str(value) for value in authority.get("allowed_values", ()))
    if default not in allowed or default not in ("SHADOW_ONLY", "CAPPED_SHORTCUT"):
        raise ValueError("unsupported hardware authority")

    configured_path = authority.get("promotion_contract_path")
    configured_hash = authority.get("promotion_contract_sha256")
    if default == "SHADOW_ONLY":
        if configured_path is not None or configured_hash is not None:
            raise ValueError(
                "SHADOW_ONLY must not carry a latent promotion contract"
            )
        return False

    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("CAPPED_SHORTCUT promotion contract path is absent")
    if (
        not isinstance(configured_hash, str)
        or len(configured_hash) != 64
        or any(character not in "0123456789abcdef" for character in configured_hash)
    ):
        raise ValueError("CAPPED_SHORTCUT promotion contract hash is invalid")
    promotion_path = Path(configured_path)
    if not promotion_path.is_absolute():
        promotion_path = (manifest_path.parent / promotion_path).resolve()
    if not promotion_path.is_file():
        raise FileNotFoundError(
            f"promotion contract is absent: {promotion_path}"
        )
    if sha256_file(promotion_path) != configured_hash:
        raise ValueError("promotion contract hash mismatch")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if (
        promotion.get("schema_version") != 1
        or promotion.get("status") != "QUALIFIED"
        or promotion.get("model_id") != model_root.name
        or promotion.get("deployment_manifest_model_id") != manifest.get("model_id")
        or not bool(
            promotion.get("authorization", {}).get(
                "capped_shortcut_hardware_control", False
            )
        )
    ):
        raise ValueError(
            "promotion contract identity, status, or authorization is invalid"
        )
    return True


def load_and_verify_manifest(
    manifest_path: Path, model_root: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported deployment manifest schema")
    if manifest.get("model_id") != model_root.name:
        raise ValueError("deployment manifest model_id does not match model root")
    if "deployment_authority" in manifest:
        authority = str(manifest.get("deployment_authority", ""))
        allowed = tuple(
            str(value) for value in manifest.get("allowed_execution_modes", ())
        )
        if authority != "CAPPED_SHORTCUT" or authority not in allowed:
            raise ValueError(
                "promoted real bundle lacks CAPPED_SHORTCUT authority"
            )
        contract_path = model_root / str(
            manifest["ensemble_interface_contract"]
        )
        if (
            not contract_path.is_file()
            or sha256_file(contract_path)
            != manifest["ensemble_interface_contract_sha256"]
        ):
            raise ValueError("ensemble interface contract hash mismatch")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            contract.get("schema_version") != 1
            or contract.get("model_id") != manifest.get("model_id")
            or contract.get("architecture_type")
            != manifest.get("architecture_type")
        ):
            raise ValueError(
                "ensemble interface identity or architecture mismatch"
            )
        if tuple(
            str(value)
            for value in contract.get("output", {}).get("state_names", ())
        ) != STATE_NAMES:
            raise ValueError("ensemble interface state order/name mismatch")
        input_contract = dict(contract.get("input", {}))
        expected_channels = (
            "fx_world_star_n",
            "fy_world_star_n",
            "fz_world_star_n",
            "tx_world_star_nm",
            "ty_world_star_nm",
            "tz_world_star_nm",
            "cmd_z_relative_mm",
        )
        expected_context = (
            "size_onehot_m8",
            "size_onehot_m10",
            "size_onehot_m12",
            "target_down_force_over_3p5N",
            "bit_radius_over_m12_radius",
        )
        output_contract = dict(contract.get("output", {}))
        expected_returns = (
            "state_probabilities",
            "direction_probabilities",
            "top_direction_deg",
            "circular_mean_direction_deg",
            "direction_world_xy",
            "max_probability",
            "circular_concentration",
            "aps_direction_classes",
            "aps_direction_angles_deg",
            "aps_set_size",
            "sample_quality",
        )
        if (
            tuple(input_contract.get("shape", ())) != (7, 360)
            or int(input_contract.get("phase_bins", -1)) != 360
            or tuple(input_contract.get("channels", ())) != expected_channels
            or tuple(input_contract.get("context_shape", ())) != (5,)
            or tuple(input_contract.get("context_order", ())) != expected_context
            or int(output_contract.get("direction_classes", -1)) != 48
            or tuple(output_contract.get("returns", ())) != expected_returns
            or abs(
                float(output_contract.get("direction_bin_offset_deg", math.nan))
                - DIRECTION_OFFSET_DEG
            )
            > 1.0e-12
            or abs(
                float(output_contract.get("direction_bin_width_deg", math.nan))
                - DIRECTION_BIN_DEG
            )
            > 1.0e-12
        ):
            raise ValueError(
                "ensemble interface input/context/direction contract mismatch"
            )
        supported_sizes = tuple(
            str(value) for value in contract.get("supported_sizes", ())
        )
        if not supported_sizes or any(
            value not in SIZE_ORDER for value in supported_sizes
        ):
            raise ValueError("ensemble interface supported_sizes is invalid")
        manifest_hardware_sizes = tuple(
            str(value)
            for value in manifest.get("hardware_control_sizes", ())
        )
        interface_hardware_sizes = tuple(
            str(value)
            for value in contract.get("policy_boundary", {}).get(
                "hardware_control_sizes", ()
            )
        )
        hardware_sizes = manifest_hardware_sizes
        if not hardware_sizes or any(
            value not in supported_sizes for value in hardware_sizes
        ) or interface_hardware_sizes != hardware_sizes:
            raise ValueError("hardware_control_sizes is absent or invalid")
        seed_models = list(manifest.get("seed_models", ()))
        contract_seeds = list(contract.get("seeds", ()))
        if (
            int(manifest.get("seed_count", -1)) != 3
            or len(seed_models) != 3
            or len(contract_seeds) != 3
        ):
            raise ValueError("promoted real bundle must bind exactly three seeds")
        for record in seed_models:
            for path_key, hash_key in (
                ("model_path", "model_sha256"),
                ("normalization_path", "normalization_sha256"),
                ("metrics_path", "metrics_sha256"),
                ("model_config_path", "model_config_sha256"),
            ):
                artifact = model_root / str(record[path_key])
                if (
                    not artifact.is_file()
                    or sha256_file(artifact) != record[hash_key]
                ):
                    raise ValueError(
                        f"real bundle artifact hash mismatch: {record[path_key]}"
                    )
        manifest_seed_identity = [
            (
                int(record["seed"]),
                str(record["model_path"]),
                str(record["model_sha256"]),
                str(record["normalization_path"]),
                str(record["normalization_sha256"]),
                str(record["metrics_path"]),
                str(record["metrics_sha256"]),
                str(record["model_config_path"]),
                str(record["model_config_sha256"]),
            )
            for record in seed_models
        ]
        contract_seed_identity = [
            (
                int(record["seed"]),
                str(record["model_path"]),
                str(record["model_sha256"]),
                str(record["normalization_path"]),
                str(record["normalization_sha256"]),
                str(record["metrics_path"]),
                str(record["metrics_sha256"]),
                str(record["model_config_path"]),
                str(record["model_config_sha256"]),
            )
            for record in contract_seeds
        ]
        if manifest_seed_identity != contract_seed_identity:
            raise ValueError("manifest/interface three-seed identity mismatch")
        promotion_path = model_root / str(manifest["promotion_contract"])
        if (
            not promotion_path.is_file()
            or sha256_file(promotion_path)
            != manifest["promotion_contract_sha256"]
        ):
            raise ValueError("promotion contract hash mismatch")
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        if (
            promotion.get("schema_version") != 1
            or promotion.get("status") != "QUALIFIED"
            or promotion.get("qualification") != "DEVELOPMENT_CV_CONTROL"
            or promotion.get("model_id") != manifest.get("model_id")
            or not bool(
                promotion.get("authorization", {}).get(
                    "capped_shortcut_hardware_control", False
                )
            )
        ):
            raise ValueError(
                "promotion identity, qualification, or authorization is invalid"
            )
        promotion_hardware_sizes = tuple(
            str(value)
            for value in promotion.get("authorization", {}).get(
                "hardware_control_sizes", ()
            )
        )
        if (
            hardware_sizes != ("m12",)
            or promotion_hardware_sizes != hardware_sizes
        ):
            raise ValueError(
                "manifest/interface/promotion hardware scope must be M12 only"
            )
        manifest["_promotion_contract_verified"] = True
        manifest["_evidence_artifacts_verified"] = 0
        manifest["_runtime_artifacts_verified"] = 0
        manifest["_interface_contract_path"] = str(contract_path)
        manifest["_supported_sizes"] = list(supported_sizes)
        manifest["_hardware_control_sizes"] = list(hardware_sizes)
        return manifest

    for item in manifest["artifacts"]:
        artifact = model_root / item["path"]
        if not artifact.is_file():
            raise FileNotFoundError(f"frozen model artifact is absent: {artifact}")
        actual = sha256_file(artifact)
        if actual != item["sha256"]:
            raise ValueError(
                f"frozen model artifact hash mismatch: {item['path']} "
                f"{actual} != {item['sha256']}"
            )
    evidence_count = 0
    for item in manifest.get("evidence_artifacts", []):
        evidence = Path(item["path"])
        if not evidence.is_absolute():
            evidence = (manifest_path.parent / evidence).resolve()
        if not evidence.is_file():
            raise FileNotFoundError(
                f"frozen audit evidence is absent: {evidence}"
            )
        actual = sha256_file(evidence)
        if actual != item["sha256"]:
            raise ValueError(
                f"frozen audit evidence hash mismatch: {item['path']} "
                f"{actual} != {item['sha256']}"
            )
        evidence_count += 1
    runtime_count = 0
    for item in manifest.get("runtime_artifacts", []):
        runtime_artifact = Path(item["path"])
        if not runtime_artifact.is_absolute():
            runtime_artifact = (manifest_path.parent / runtime_artifact).resolve()
        if not runtime_artifact.is_file():
            raise FileNotFoundError(
                f"frozen inference runtime artifact is absent: {runtime_artifact}"
            )
        actual = sha256_file(runtime_artifact)
        if actual != item["sha256"]:
            raise ValueError(
                f"frozen inference runtime artifact hash mismatch: "
                f"{item['path']} {actual} != {item['sha256']}"
            )
        runtime_count += 1
    manifest["_promotion_contract_verified"] = _verify_hardware_authority(
        manifest, manifest_path, model_root
    )
    audit = manifest.get("audit_decision", {})
    if audit.get("global_status") == "REJECTED_GLOBAL_AUDIT":
        authority = manifest.get("hardware_authority", {})
        if (
            authority.get("default") != "SHADOW_ONLY"
            or authority.get("allowed_values") != ["SHADOW_ONLY"]
            or authority.get("promotion_contract_path") is not None
            or authority.get("promotion_contract_sha256") is not None
            or audit.get("diagnostic_scope")
            != "M12_DEVELOPMENT_SHADOW_ONLY"
        ):
            raise ValueError(
                "rejected global audit must remain M12 development shadow only"
            )
    manifest["_evidence_artifacts_verified"] = evidence_count
    manifest["_runtime_artifacts_verified"] = runtime_count
    return manifest


def verify_requested_size(manifest: dict[str, Any], requested_size: str) -> None:
    supported = tuple(
        str(value)
        for value in manifest.get(
            "_supported_sizes", manifest.get("supported_sizes", ())
        )
    )
    if requested_size not in supported:
        raise ValueError(
            f"size {requested_size!r} is outside this deployment's "
            f"diagnostic scope {supported}"
        )


def verify_fixed_demo_pilot_authorization(
    configured_path: Path | None,
    manifest: dict[str, Any],
    requested_size: str,
) -> dict[str, Any]:
    """Verify the narrow M10 pilot without widening general bundle promotion.

    The frozen bundle remains generally promoted for M12 only. This optional
    record permits one M10 fixed-offset launcher to consume the already
    supported M10 inference domain. MATLAB separately pins this record's hash
    and enforces the exact private trial offset before any robot connection.
    """

    if configured_path is None:
        return {
            "verified": False,
            "path": "",
            "sha256": "",
            "authorization_id": "",
            "scope": "",
        }
    path = configured_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixed-demo pilot authorization is absent: {path}")
    if (
        path != M10_FIXED_DEMO_PILOT_PATH
        or sha256_file(path) != M10_FIXED_DEMO_PILOT_SHA256
    ):
        raise ValueError(
            "M10 fixed-demo pilot path or immutable SHA-256 trust anchor changed"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    authorization = dict(value.get("authorization", {}))
    fixed_demo = dict(value.get("fixed_demo", {}))
    source = dict(value.get("real_source_evidence", {}))
    geometry = dict(value.get("geometry_contract", {}))
    control = dict(value.get("control_boundary", {}))
    expected_offset = np.asarray(
        (4.590151046897575, 0.3008543944586581), dtype=float
    )
    actual_offset = np.asarray(fixed_demo.get("offset_fixture_mm", ()), dtype=float)
    source_trace = Path(str(source.get("clean_conical_trace_csv", ""))).resolve()
    source_trace_sha256 = str(source.get("clean_conical_trace_sha256", ""))
    physical_support_radius_mm = float(
        fixed_demo.get(
            "maximum_physical_tip_center_radius_from_fresh_center_mm",
            float("nan"),
        )
    )
    support_tracking_allowance_mm = float(
        fixed_demo.get("endpoint_tracking_allowance_mm", float("nan"))
    )
    commanded_support_radius_mm = float(
        fixed_demo.get(
            "maximum_commanded_tip_center_radius_from_fresh_center_mm",
            float("nan"),
        )
    )
    derived_physical_support_radius_mm = (
        0.5 * float(geometry.get("socket_head_outer_diameter_mm", float("nan")))
        - float(geometry.get("peg_tip_face_af_mm", float("nan")))
        / math.sqrt(3.0)
    )
    retained_guards = tuple(
        str(item) for item in control.get("independent_guards_retained", ())
    )
    if (
        value.get("schema_version") != 1
        or value.get("authorization_id") != M10_FIXED_DEMO_PILOT_ID
        or value.get("status")
        != "EXPERIMENTAL_FIXED_DEMO_PILOT_AUTHORIZED_BY_OPERATOR_REQUEST"
        or value.get("model_id") != manifest.get("model_id")
        or value.get("size") != "m10"
        or requested_size != "m10"
        or value.get("execution_mode")
        != "FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY_M10"
        or value.get("public_entrypoint")
        != "run_m10_proposed_plus_entropy_trial"
        or not bool(authorization.get("fixed_demo_only", False))
        or not bool(
            authorization.get("capped_shortcut_hardware_control", False)
        )
        or not bool(authorization.get("residual_entropy_target_selection", False))
        or bool(authorization.get("capture_control", True))
        or bool(authorization.get("insertion_control", True))
        or bool(authorization.get("terminal_verification", True))
        or bool(authorization.get("online_weight_update", True))
        or int(authorization.get("maximum_launcher_attempts", -1)) != 3
        or actual_offset.shape != (2,)
        or not np.all(np.isfinite(actual_offset))
        or not np.allclose(actual_offset, expected_offset, atol=1.0e-12, rtol=0.0)
        or fixed_demo.get("source_sample_id")
        != "M10-CLEAN_CANDIDATE-SUPPORTED-A003p75"
        or not bool(
            fixed_demo.get(
                "fresh_operator_taught_m10_center_required_each_attempt", False
            )
        )
        or not bool(
            fixed_demo.get("historical_absolute_world_center_forbidden", False)
        )
        or not bool(
            fixed_demo.get("fresh_center_support_envelope_required", False)
        )
        or fixed_demo.get("support_envelope_role")
        != (
            "SAFETY_FAIL_STOP_ONLY_NEVER_POSTERIOR_INPUT_"
            "NEVER_TARGET_RANKING_NEVER_STEERING"
        )
        or not math.isclose(
            physical_support_radius_mm,
            derived_physical_support_radius_mm,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            physical_support_radius_mm,
            5.792458277186394,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            support_tracking_allowance_mm,
            0.6,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            commanded_support_radius_mm,
            physical_support_radius_mm - support_tracking_allowance_mm,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            commanded_support_radius_mm,
            5.192458277186394,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not bool(source.get("clean_cone_completed", False))
        or not bool(source.get("clean_safe_retract_completed", False))
        or float(source.get("clean_phase_coverage_fraction", 0.0)) < 0.98
        or not source_trace.is_file()
        or sha256_file(source_trace) != source_trace_sha256
        or not any(
            "fresh-center M10 head-support command envelope" in item
            for item in retained_guards
        )
        or "not general M10 full-loop promotion"
        not in str(control.get("claim_limit", ""))
    ):
        raise ValueError(
            "M10 fixed-demo pilot identity, evidence, scope, or no-authority "
            "boundaries are invalid"
        )
    return {
        "verified": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "authorization_id": str(value["authorization_id"]),
        "scope": str(control["claim_limit"]),
        "support_envelope_role": str(fixed_demo["support_envelope_role"]),
        "maximum_physical_tip_center_radius_mm": physical_support_radius_mm,
        "endpoint_tracking_allowance_mm": support_tracking_allowance_mm,
        "maximum_commanded_tip_center_radius_mm": commanded_support_radius_mm,
    }


def relocated_checkpoint(model_root: Path, configured: str) -> Path:
    configured_path = Path(configured)
    seed_name = configured_path.parent.name
    candidate = model_root / "seeds" / seed_name / configured_path.name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"relocated checkpoint for {configured_path} is absent: {candidate}"
        )
    return candidate


def legacy_network_inputs(
    sequence: np.ndarray,
    target_down_force_n: float,
    bit_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the immutable four-channel simulation runtime input."""

    if sequence.shape != (4, 360):
        raise ValueError("legacy sequence must have shape (4,360)")
    shape = np.vstack(
        [smoothed_standardized(channel) for channel in sequence]
    ).astype(np.float32)
    force_scale = max(float(target_down_force_n), 1.0e-6)
    torque_scale = force_scale * float(bit_radius_m)
    dimensionless = (
        sequence[0] / force_scale,
        sequence[1] / torque_scale,
        sequence[2] / torque_scale,
        sequence[3] / float(bit_radius_m),
    )
    context: list[float] = []
    for values in dimensionless:
        context.extend(
            (
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
            )
        )
    return shape, np.asarray(context, dtype=np.float32)


def run_legacy_ensemble(
    model_root: Path,
    sequence: np.ndarray,
    target_down_force_n: float,
    bit_radius_m: float,
) -> dict[str, Any]:
    contract_path = model_root / "ensemble_interface_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shape, context = legacy_network_inputs(
        sequence, target_down_force_n, bit_radius_m
    )
    seed_state: list[np.ndarray] = []
    seed_direction: list[np.ndarray] = []
    seed_top_angles: list[float] = []
    seed_records: list[dict[str, Any]] = []
    for configured in contract["checkpoints"]:
        checkpoint = relocated_checkpoint(model_root, configured)
        seed_dir = checkpoint.parent
        normalization = np.load(seed_dir / "normalization.npz")
        metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        model = CircularDirectionCNN()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        normalized_context = (
            context - np.asarray(normalization["context_mean"], dtype=np.float32)
        ) / np.asarray(normalization["context_std"], dtype=np.float32)
        with torch.no_grad():
            state_logits, direction_logits = model(
                torch.from_numpy(shape[None, ...]),
                torch.from_numpy(normalized_context[None, ...].astype(np.float32)),
            )
        state_probability = softmax(
            state_logits.numpy()[0], float(metrics.get("state_temperature", 1.0))
        )
        direction_probability = softmax(
            direction_logits.numpy()[0], float(metrics["temperature"])
        )
        seed_state.append(state_probability)
        seed_direction.append(direction_probability)
        top_class = int(np.argmax(direction_probability))
        top_angle = float(class_angle_deg(top_class))
        seed_top_angles.append(top_angle)
        seed_records.append(
            {
                "seed": seed_dir.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "top_direction_deg": top_angle,
                "max_probability": float(direction_probability[top_class]),
            }
        )

    state_probability = np.mean(np.asarray(seed_state), axis=0)
    direction_probability = np.mean(np.asarray(seed_direction), axis=0)
    top_class = int(np.argmax(direction_probability))
    class_angles_deg = np.asarray(
        class_angle_deg(np.arange(DIRECTION_CLASSES)), dtype=float
    )
    class_angles_rad = np.deg2rad(class_angles_deg)
    posterior_vector = np.sum(
        direction_probability * np.exp(1j * class_angles_rad)
    )
    concentration = float(abs(posterior_vector))
    mean_deg = float(np.mod(np.rad2deg(np.angle(posterior_vector)), 360.0))
    order = np.argsort(-direction_probability)
    cumulative = np.cumsum(direction_probability[order])
    aps_size = min(
        int(
            np.searchsorted(
                cumulative, float(contract["aps_quantile"]), side="left"
            )
        )
        + 1,
        len(order),
    )
    aps_classes = order[:aps_size].astype(int)
    seed_top = np.asarray(seed_top_angles, dtype=float)
    seed_span = float(
        np.max(circular_distance_deg(seed_top[:, None], seed_top[None, :]))
    )
    structured = contract.get("structured_transport_gate", {})
    max_probability = float(direction_probability[top_class])
    simulation_gate = bool(
        max_probability >= float(contract.get("confidence_gate", 0.0))
        and aps_size
        <= int(structured.get("max_aps_set_size", DIRECTION_CLASSES))
        and concentration
        >= float(structured.get("min_circular_concentration", 0.0))
        and seed_span
        <= float(structured.get("max_ensemble_top1_span_deg", 180.0)) + 1.0e-9
    )
    mean_rad = math.radians(mean_deg)
    return {
        "state_names": list(STATE_NAMES),
        "state_probabilities": state_probability.tolist(),
        "top_state": STATE_NAMES[int(np.argmax(state_probability))],
        "direction_class_angles_deg": class_angles_deg.tolist(),
        "direction_probabilities": direction_probability.tolist(),
        "top_direction_class": top_class,
        "top_direction_deg": float(class_angles_deg[top_class]),
        "circular_mean_direction_deg": mean_deg,
        "direction_task_xy": [math.cos(mean_rad), math.sin(mean_rad)],
        "max_probability": max_probability,
        "circular_concentration": concentration,
        "aps_direction_classes": aps_classes.tolist(),
        "aps_direction_angles_deg": class_angles_deg[aps_classes].tolist(),
        "aps_set_size": int(aps_size),
        "ensemble_top1_span_deg": seed_span,
        "simulation_transport_gate_passed": simulation_gate,
        "seed_predictions": seed_records,
        "model_contract_sha256": sha256_file(contract_path),
    }


def _instantiate_real_model(
    architecture_type: str, constructor: dict[str, Any]
) -> nn.Module:
    if architecture_type == "REAL_FOURIER_LINEAR_V1":
        return RealFourierLinear(harmonics=int(constructor.get("harmonics", 6)))
    if architecture_type == "REAL_TINY_CIRCULAR_CNN_V1":
        return RealTinyCircularCNN()
    raise ValueError(f"unsupported real architecture {architecture_type!r}")


def _real_seed_records(
    model_root: Path, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    records = list(contract.get("seeds", ()))
    if len(records) != 3:
        raise ValueError("real bundle must contain exactly three seed records")
    for record in records:
        for path_key, hash_key in (
            ("model_path", "model_sha256"),
            ("normalization_path", "normalization_sha256"),
            ("metrics_path", "metrics_sha256"),
            ("model_config_path", "model_config_sha256"),
        ):
            artifact = model_root / str(record[path_key])
            if not artifact.is_file() or sha256_file(artifact) != record[hash_key]:
                raise ValueError(
                    f"real seed artifact hash mismatch: {record[path_key]}"
                )
    return records


def _size_from_bit_radius(bit_radius_m: float) -> str:
    requested = float(bit_radius_m)
    return min(
        SIZE_ORDER,
        key=lambda name: abs(SIZE_BIT_RADIUS_M[name] - requested),
    )


def run_ensemble(
    model_root: Path,
    sequence: np.ndarray,
    requested_size: str | None = None,
    target_down_force_n: float = TARGET_FORCE_REFERENCE_N,
    *,
    bit_radius_m: float | None = None,
) -> dict[str, Any]:
    """Run the promoted 7x360 real ensemble with no training-stack import.

    ``requested_size`` is the public v2 interface.  ``bit_radius_m`` remains a
    keyword-only compatibility bridge for the immutable simulation-shadow
    evaluator; it does not alter the promoted real-data geometry contract.
    """

    model_root = Path(model_root).resolve()
    contract_path = model_root / "ensemble_interface_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    architecture_type = str(contract.get("architecture_type", ""))

    # Preserve read-only replay of the old simulation-shadow bundle.  It is a
    # separate four-channel path and can never acquire v2 hardware authority.
    if not architecture_type.startswith("REAL_"):
        if bit_radius_m is None:
            if requested_size not in SIZE_ORDER:
                raise ValueError("legacy inference requires a size or bit radius")
            bit_radius_m = SIZE_BIT_RADIUS_M[str(requested_size)]
        return run_legacy_ensemble(
            model_root,
            np.asarray(sequence, dtype=float),
            float(target_down_force_n),
            float(bit_radius_m),
        )

    if requested_size is None:
        if bit_radius_m is None:
            raise ValueError("promoted real inference requires requested_size")
        requested_size = _size_from_bit_radius(float(bit_radius_m))
    requested_size = str(requested_size).lower()
    supported_sizes = tuple(
        str(value) for value in contract.get("supported_sizes", ())
    )
    if requested_size not in supported_sizes:
        raise ValueError(
            f"size {requested_size!r} is outside real contract {supported_sizes}"
        )

    raw_sequence, raw_context = network_inputs(
        np.asarray(sequence, dtype=float),
        requested_size,
        float(target_down_force_n),
    )
    state_by_seed: list[np.ndarray] = []
    direction_by_seed: list[np.ndarray] = []
    seed_top_angles: list[float] = []
    seed_records: list[dict[str, Any]] = []
    expected_architecture = architecture_type
    for record in _real_seed_records(model_root, contract):
        model_path = model_root / str(record["model_path"])
        normalizer_path = model_root / str(record["normalization_path"])
        metrics_path = model_root / str(record["metrics_path"])
        config_path = model_root / str(record["model_config_path"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        seed_architecture = str(config.get("architecture_type", ""))
        if seed_architecture != expected_architecture:
            raise ValueError("seed architecture differs from ensemble contract")
        normalizer = np.load(normalizer_path)
        sequence_mean = np.asarray(
            normalizer["sequence_mean"], dtype=np.float32
        )
        sequence_std = np.asarray(
            normalizer["sequence_std"], dtype=np.float32
        )
        context_mean = np.asarray(
            normalizer["context_mean"], dtype=np.float32
        )
        context_std = np.asarray(
            normalizer["context_std"], dtype=np.float32
        )
        if (
            sequence_mean.shape != (7,)
            or sequence_std.shape != (7,)
            or context_mean.shape != (5,)
            or context_std.shape != (5,)
            or np.any(sequence_std <= 0.0)
            or np.any(context_std <= 0.0)
        ):
            raise ValueError("real normalizer shape/range contract is invalid")
        normalized_sequence = (
            raw_sequence - sequence_mean[:, None]
        ) / sequence_std[:, None]
        normalized_context = (
            raw_context - context_mean
        ) / context_std
        model = _instantiate_real_model(
            seed_architecture, dict(config.get("constructor", {}))
        )
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            state_logits, direction_logits = model(
                torch.from_numpy(normalized_sequence[None, ...].astype(np.float32)),
                torch.from_numpy(normalized_context[None, ...].astype(np.float32)),
            )
        state_probability = softmax(
            state_logits.numpy()[0],
            float(metrics.get("state_temperature", 1.0)),
        )
        direction_probability = softmax(
            direction_logits.numpy()[0],
            float(metrics.get("direction_temperature", metrics.get("temperature", 1.0))),
        )
        if (
            not np.all(np.isfinite(state_probability))
            or not np.all(np.isfinite(direction_probability))
        ):
            raise ValueError("real seed returned non-finite probabilities")
        state_by_seed.append(state_probability)
        direction_by_seed.append(direction_probability)
        top_index = int(np.argmax(direction_probability))
        top_angle = float(class_angle_deg(top_index))
        seed_top_angles.append(top_angle)
        seed_records.append(
            {
                "seed": int(record["seed"]),
                "model_sha256": str(record["model_sha256"]),
                "normalization_sha256": str(record["normalization_sha256"]),
                "top_direction_deg": top_angle,
                "max_probability": float(direction_probability[top_index]),
            }
        )

    state_probability = np.mean(np.stack(state_by_seed), axis=0)
    direction_probability = np.mean(np.stack(direction_by_seed), axis=0)
    state_probability /= np.sum(state_probability)
    direction_probability /= np.sum(direction_probability)
    if (
        abs(float(np.sum(state_probability)) - 1.0) > 1.0e-8
        or abs(float(np.sum(direction_probability)) - 1.0) > 1.0e-8
    ):
        raise ValueError("ensemble probability normalization failed")

    angles_deg = np.asarray(
        class_angle_deg(np.arange(DIRECTION_CLASSES)), dtype=float
    )
    angles_rad = np.deg2rad(angles_deg)
    top_index = int(np.argmax(direction_probability))
    top_deg = float(angles_deg[top_index])
    posterior_vector = np.sum(
        direction_probability * np.exp(1j * angles_rad)
    )
    concentration = float(abs(posterior_vector))
    mean_deg = float(np.mod(np.rad2deg(np.angle(posterior_vector)), 360.0))
    entropy = float(
        -np.sum(
            direction_probability
            * np.log(np.clip(direction_probability, 1.0e-12, 1.0))
        )
    )
    calibration = dict(contract.get("calibration", {}))
    aps_quantile_value = float(calibration.get("aps_quantile", 0.90))
    order = np.argsort(-direction_probability)
    cumulative = np.cumsum(direction_probability[order])
    aps_size = min(
        int(np.searchsorted(cumulative, aps_quantile_value, side="left")) + 1,
        DIRECTION_CLASSES,
    )
    aps_classes = order[:aps_size].astype(int)
    seed_top = np.asarray(seed_top_angles, dtype=float)
    seed_span = float(
        np.max(circular_distance_deg(seed_top[:, None], seed_top[None, :]))
    )
    maximum_probability = float(direction_probability[top_index])
    sample_quality = float(
        np.clip(
            0.45 * maximum_probability
            + 0.35 * concentration
            + 0.20 * (1.0 - (aps_size - 1) / 47.0),
            0.0,
            1.0,
        )
    )
    top_rad = math.radians(top_deg)
    mean_rad = math.radians(mean_deg)
    return {
        "architecture_type": architecture_type,
        "state_names": list(STATE_NAMES),
        "state_probabilities": state_probability.tolist(),
        "top_state": STATE_NAMES[int(np.argmax(state_probability))],
        "direction_class_angles_deg": angles_deg.tolist(),
        "direction_probabilities": direction_probability.tolist(),
        "top_direction_class": top_index,
        "top_direction_deg": top_deg,
        "circular_mean_direction_deg": mean_deg,
        "direction_task_xy": [math.cos(top_rad), math.sin(top_rad)],
        "circular_mean_direction_task_xy": [
            math.cos(mean_rad),
            math.sin(mean_rad),
        ],
        "command_direction_source": "TOP1_POSTERIOR_MODE",
        "max_probability": maximum_probability,
        "direction_entropy_nats": entropy,
        "circular_concentration": concentration,
        "aps_direction_classes": aps_classes.tolist(),
        "aps_direction_angles_deg": angles_deg[aps_classes].tolist(),
        "aps_set_size": int(aps_size),
        "aps_quantile": aps_quantile_value,
        "ensemble_top1_span_deg": seed_span,
        "sample_quality": sample_quality,
        # Compatibility field consumed by existing MATLAB evidence records.
        # It is diagnostic true for a finite verified ensemble, never a weak
        # posterior veto and never a simulation-derived authority gate.
        "simulation_transport_gate_passed": True,
        "posterior_quality_is_validity_veto": False,
        "weak_complete_fingerprint_remains_actionable": True,
        "seed_predictions": seed_records,
        "model_contract_sha256": sha256_file(contract_path),
    }


def rotation_z(angle: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_y(angle: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )


def rotation_x(angle: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )


def kuka_abc_to_rotation(abc: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(abc, dtype=float)
    return rotation_z(float(a)) @ rotation_y(float(b)) @ rotation_x(float(c))


def trace_csv_to_input(path: Path) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if row.get("stage") == "SCAN"
        and row.get("phase_rad") not in {"", "NaN"}
    ]
    if len(rows) < 32:
        raise ValueError(f"trace has only {len(rows)} SCAN samples")
    phase = np.asarray([float(row["phase_rad"]) for row in rows])
    source_stamp = np.asarray(
        [float(row["source_stamp_s"]) for row in rows], dtype=float
    )
    wrench_age = np.asarray(
        [float(row["wrench_age_s"]) for row in rows], dtype=float
    )
    if (
        not np.all(np.isfinite(source_stamp))
        or np.any(np.diff(source_stamp) <= 0.0)
    ):
        raise ValueError("trace source timestamps are stale or non-monotonic")
    if (
        not np.all(np.isfinite(wrench_age))
        or np.any(wrench_age < 0.0)
        or float(np.max(wrench_age)) > 0.10
    ):
        raise ValueError("trace wrench age is invalid or exceeds 0.10 s")
    sorted_phase = np.sort(np.mod(phase, 2.0 * math.pi))
    cyclic_gaps = np.diff(
        np.concatenate(
            (sorted_phase, sorted_phase[:1] + 2.0 * math.pi)
        )
    )
    maximum_phase_gap_rad = float(np.max(cyclic_gaps))
    if maximum_phase_gap_rad > math.radians(2.0) + 1.0e-12:
        raise ValueError("trace cyclic phase gap exceeds 2 degrees")
    force_world = np.asarray(
        [
            [
                float(row["fx_world_star_n"]),
                float(row["fy_world_star_n"]),
                float(row["fz_world_star_n"]),
            ]
            for row in rows
        ]
    )
    torque_world = np.asarray(
        [
            [
                float(row["tx_world_star_nm"]),
                float(row["ty_world_star_nm"]),
                float(row["tz_world_star_nm"]),
            ]
            for row in rows
        ]
    )
    command_z = np.asarray([float(row["cmd_z_mm"]) for row in rows])
    actual_z = np.asarray([float(row["actual_z_mm"]) for row in rows])
    pivot_z = float(rows[0]["cmd_z_mm"])
    upright_rows = [row for row in all_rows if row.get("stage") == "SETTLE"]
    if not upright_rows:
        raise ValueError("trace has no upright SETTLE sample for task-frame anchor")
    pivot_abc = np.asarray(
        [
            float(upright_rows[0]["cmd_a_rad"]),
            float(upright_rows[0]["cmd_b_rad"]),
            float(upright_rows[0]["cmd_c_rad"]),
        ]
    )
    world_from_task = kuka_abc_to_rotation(pivot_abc)
    torque_task = (world_from_task.T @ torque_world.T).T
    normal_sign = 1.0 if float(np.median(force_world[:, 2])) >= 0.0 else -1.0
    return {
        "schema_version": 2,
        "source_kind": "REAL_TRACE_CSV_REPLAY_WORLD_FIXTURE_ALIGNED",
        "source_path": str(path.resolve()),
        "phase_rad": phase.tolist(),
        "model_fx_n": force_world[:, 0].tolist(),
        "model_fy_n": force_world[:, 1].tolist(),
        "model_fz_n": (normal_sign * force_world[:, 2]).tolist(),
        "model_tx_nm": torque_world[:, 0].tolist(),
        "model_ty_nm": torque_world[:, 1].tolist(),
        "model_tz_nm": torque_world[:, 2].tolist(),
        "model_cmd_z_relative_mm": (command_z - pivot_z).tolist(),
        # The old immutable simulation-shadow replay is kept alongside the
        # real-v2 channels.  New inference never consumes these fields.
        "legacy_model_fz_n": (normal_sign * force_world[:, 2]).tolist(),
        "legacy_model_tx_nm": torque_task[:, 0].tolist(),
        "legacy_model_ty_nm": torque_task[:, 1].tolist(),
        "legacy_model_tip_z_m": (
            0.0003 + (actual_z - pivot_z) / 1000.0
        ).tolist(),
        # Recorded columns are already WORLD*.  The commissioned fixture axes
        # are WORLD aligned, so no TCP-ABC or CAD clocking correction belongs
        # here.  In particular, do not reapply the CAD -15/+45 bookkeeping.
        "world_from_task_xy": np.eye(3)[:, :2].tolist(),
        "phase_coverage_fraction": float((np.max(phase) - np.min(phase)) / (2 * math.pi)),
        "maximum_wrench_age_s": float(np.max(wrench_age)),
        "maximum_phase_gap_rad": maximum_phase_gap_rad,
        "source_timestamps_strictly_fresh": True,
        "normal_sign_world_z": normal_sign,
        "frame_contract": (
            "phase and six WORLD* wrench channels in WORLD-aligned fixture "
            "axes; Fz contact-positive; commanded Z relative to cone pivot "
            "in mm; no additional CAD clocking compensation"
        ),
    }


def validate_input(value: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    schema_version = int(value.get("schema_version", -1))
    if schema_version not in (1, 2):
        raise ValueError("unsupported fingerprint input schema")
    phase = np.asarray(value["phase_rad"], dtype=float)
    if schema_version == 2:
        channel_names = (
            "model_fx_n",
            "model_fy_n",
            "model_fz_n",
            "model_tx_nm",
            "model_ty_nm",
            "model_tz_nm",
            "model_cmd_z_relative_mm",
        )
    else:
        channel_names = (
            "model_fz_n",
            "model_tx_nm",
            "model_ty_nm",
            "model_tip_z_m",
        )
    columns = np.column_stack(
        tuple(np.asarray(value[name], dtype=float) for name in channel_names)
    )
    if columns.shape != (len(phase), len(channel_names)):
        raise ValueError("fingerprint channel lengths differ")
    if not np.all(np.isfinite(columns)) or not np.all(np.isfinite(phase)):
        raise ValueError("fingerprint contains non-finite values")
    coverage = float(value["phase_coverage_fraction"])
    if coverage < 0.98:
        raise ValueError(f"phase coverage {coverage:.6f} is below 0.98")
    world_from_task_xy = np.asarray(value["world_from_task_xy"], dtype=float)
    if world_from_task_xy.shape != (3, 2):
        raise ValueError("world_from_task_xy must have shape (3,2)")
    if not np.all(np.isfinite(world_from_task_xy)):
        raise ValueError("world_from_task_xy contains non-finite values")
    gram = world_from_task_xy.T @ world_from_task_xy
    if not np.allclose(gram, np.eye(2), atol=1.0e-5):
        raise ValueError("world_from_task_xy columns are not orthonormal")
    return phase, columns


def legacy_columns_from_input(
    value: dict[str, Any], validated_columns: np.ndarray
) -> np.ndarray:
    """Return four immutable shadow channels from either input schema."""

    if int(value.get("schema_version", -1)) == 1:
        return validated_columns
    names = (
        "legacy_model_fz_n",
        "legacy_model_tx_nm",
        "legacy_model_ty_nm",
        "legacy_model_tip_z_m",
    )
    if all(name in value for name in names):
        return np.column_stack(
            tuple(np.asarray(value[name], dtype=float) for name in names)
        )
    # A MATLAB schema-v2 live input carries no legacy tip datum.  This branch
    # is only for diagnostic compatibility and cannot gain control authority.
    return np.column_stack(
        (
            validated_columns[:, 2],
            validated_columns[:, 3],
            validated_columns[:, 4],
            0.0003 + validated_columns[:, 6] / 1000.0,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path)
    source.add_argument("--trace-csv", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--size", choices=SIZE_ORDER, default="m12")
    parser.add_argument("--pilot-authorization", type=Path)
    parser.add_argument(
        "--target-down-force-n",
        type=float,
        default=TARGET_FORCE_REFERENCE_N,
    )
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_and_verify_manifest(manifest_path, model_root)
    verify_requested_size(manifest, args.size)
    fixed_demo_pilot = verify_fixed_demo_pilot_authorization(
        args.pilot_authorization, manifest, args.size
    )
    if args.input_json is not None:
        input_path = args.input_json.resolve()
        input_value = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        input_path = args.trace_csv.resolve()
        input_value = trace_csv_to_input(input_path)
    phase, columns = validate_input(input_value)
    real_bundle = "deployment_authority" in manifest
    if real_bundle:
        if int(input_value.get("schema_version", -1)) != 2:
            raise ValueError(
                "promoted real bundle requires MATLAB/trace input schema 2"
            )
        sequence = phase_resample(phase, columns, bins=360)
        estimate = run_ensemble(
            model_root,
            sequence,
            requested_size=args.size,
            target_down_force_n=args.target_down_force_n,
        )
        bit_radius_m = SIZE_BIT_RADIUS_M[args.size]
    else:
        legacy_columns = legacy_columns_from_input(input_value, columns)
        sequence = phase_resample(phase, legacy_columns, bins=360)
        bit_radius_m = float(manifest["size_bit_radius_m"][args.size])
        estimate = run_ensemble(
            model_root,
            sequence,
            requested_size=args.size,
            target_down_force_n=args.target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
    world_from_task_xy = np.asarray(input_value["world_from_task_xy"], dtype=float)
    direction_world = world_from_task_xy @ np.asarray(
        estimate["direction_task_xy"], dtype=float
    )
    direction_world_xy = direction_world[:2]
    direction_world_xy /= max(float(np.linalg.norm(direction_world_xy)), 1.0e-12)
    mean_task_xy = np.asarray(
        estimate.get(
            "circular_mean_direction_task_xy",
            estimate["direction_task_xy"],
        ),
        dtype=float,
    )
    mean_world = world_from_task_xy @ mean_task_xy
    mean_world_xy = mean_world[:2]
    mean_world_xy /= max(float(np.linalg.norm(mean_world_xy)), 1.0e-12)

    if real_bundle:
        authority = str(manifest["deployment_authority"])
        promoted_hardware_sizes = tuple(manifest["_hardware_control_sizes"])
        pilot_authorized = bool(fixed_demo_pilot["verified"])
        hardware_sizes = promoted_hardware_sizes
        if pilot_authorized and args.size not in hardware_sizes:
            hardware_sizes = (*hardware_sizes, args.size)
        hardware_control_allowed = bool(
            authority == "CAPPED_SHORTCUT"
            and manifest["_promotion_contract_verified"]
            and (
                args.size in promoted_hardware_sizes
                or (args.size == "m10" and pilot_authorized)
            )
            and np.all(np.isfinite(sequence))
            and float(input_value["phase_coverage_fraction"]) >= 0.98
        )
    else:
        authority = str(manifest["hardware_authority"]["default"])
        hardware_sizes = ()
        hardware_control_allowed = False
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "OK",
        "model_id": model_root.name,
        "model_root": str(model_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "source_kind": input_value.get("source_kind", "MATLAB_PHASE_BOUNDARY"),
        "frame_contract": input_value.get("frame_contract", ""),
        "size": args.size,
        "bit_radius_m": bit_radius_m,
        "target_down_force_n": args.target_down_force_n,
        "input_samples": int(len(phase)),
        "phase_coverage_fraction": float(input_value["phase_coverage_fraction"]),
        "preprocessing": {
            "phase_bins": 360,
            "channels": (
                [
                    "Fx_N",
                    "Fy_N",
                    "Fz_N",
                    "Tx_Nm",
                    "Ty_Nm",
                    "Tz_Nm",
                    "cmd_z_relative_mm",
                ]
                if real_bundle
                else ["Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m"]
            ),
            "z_channel": (
                "commanded_z_relative_to_cone_pivot_mm"
                if real_bundle
                else "simulation_nominal_0p0003m_plus_real_relative_world_z"
            ),
            "wrench_origin": "Bota_measurement_origin_no_tip_moment_shift",
            "cad_clocking_compensation_applied": False,
        },
        "deployment_authority": authority,
        "hardware_control_sizes": list(hardware_sizes),
        "hardware_control_allowed": hardware_control_allowed,
        "fixed_demo_pilot_authorization_verified": bool(
            fixed_demo_pilot["verified"]
        ),
        "fixed_demo_pilot_authorization_id": str(
            fixed_demo_pilot["authorization_id"]
        ),
        "fixed_demo_pilot_authorization_path": str(fixed_demo_pilot["path"]),
        "fixed_demo_pilot_authorization_sha256": str(
            fixed_demo_pilot["sha256"]
        ),
        "fixed_demo_pilot_scope": str(fixed_demo_pilot["scope"]),
        "control_boundary": (
            "Model owns only high-level direction/state posterior and CAD-bounded "
            "routing when hardware_control_allowed. MATLAB/KST remains the sole "
            "command owner; hard guards, capture, insertion, and terminal "
            "verification remain deterministic and external."
        ),
        **estimate,
        "direction_world_xy": direction_world_xy.tolist(),
        "top_direction_world_xy": direction_world_xy.tolist(),
        "circular_mean_direction_world_xy": mean_world_xy.tolist(),
    }
    if fixed_demo_pilot["verified"]:
        output.update(
            {
                "fixed_demo_support_envelope_role": str(
                    fixed_demo_pilot["support_envelope_role"]
                ),
                "fixed_demo_maximum_physical_tip_center_radius_mm": float(
                    fixed_demo_pilot[
                        "maximum_physical_tip_center_radius_mm"
                    ]
                ),
                "fixed_demo_endpoint_tracking_allowance_mm": float(
                    fixed_demo_pilot["endpoint_tracking_allowance_mm"]
                ),
                "fixed_demo_maximum_commanded_tip_center_radius_mm": float(
                    fixed_demo_pilot[
                        "maximum_commanded_tip_center_radius_mm"
                    ]
                ),
            }
        )
    if real_bundle and not hardware_control_allowed:
        output["hardware_control_denial_reason"] = (
            "requested_size_outside_M12_control_scope"
            if args.size not in hardware_sizes
            else "bundle_authority_or_complete_input_contract_failed"
        )
    write_json(args.output_json.resolve(), output)
    console_summary = {
        "status": output["status"],
        "model_id": output["model_id"],
        "direction_task_deg": output["circular_mean_direction_deg"],
        "top_direction_task_deg": output["top_direction_deg"],
        "direction_world_xy": output["direction_world_xy"],
        "aps_set_size": output["aps_set_size"],
        "hardware_control_allowed": output["hardware_control_allowed"],
    }
    if real_bundle:
        console_summary["real_input_contract_complete"] = bool(
            np.all(np.isfinite(sequence))
            and float(input_value["phase_coverage_fraction"]) >= 0.98
        )
    else:
        console_summary["simulation_gate"] = output[
            "simulation_transport_gate_passed"
        ]
    print(json.dumps(console_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
