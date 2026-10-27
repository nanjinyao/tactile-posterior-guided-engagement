from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn

from src.m8_direction_runtime import M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / "outputs/current/datasets/cross_size_s3_adaptation_pilot"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/models/cross_size_s3_direction_head_pilot"
)
BASE_MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SIZES = ("m10", "m12")
FROZEN_PREFIXES = ("encoder.", "shared.", "state_head.")
DIRECTION_PREFIX = "direction_head."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if not key.startswith(prefixes):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_contract(data_root: Path, size: str) -> dict:
    contract_path = data_root / size / "dataset_contract.json"
    manifest_path = data_root / size / "dataset_manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"{size}: adaptation dataset is incomplete")
    if manifest["dataset_contract_sha256"] != _sha256(contract_path):
        raise ValueError(f"{size}: dataset contract hash drift")
    if contract["orbit_intersection"]:
        raise ValueError(f"{size}: C6 orbit leakage")
    return contract


def _load_role_examples(data_root: Path, size: str, contract: dict, role: str) -> dict:
    spec = json.loads(
        (ROOT / "configs" / f"allen_{size}_source.json").read_text(
            encoding="utf-8"
        )
    )
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    task_by_id = {task["trial_id"]: task for task in contract["tasks"][role]}
    sequences = []
    contexts = []
    labels = []
    angles = []
    trial_ids = []
    orbit_ids = []
    for trial_id, task in sorted(task_by_id.items()):
        path = data_root / size / "trials" / f"{trial_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            phase_sequence = np.asarray(data["phase_sequence"], dtype=float)
            columns = tuple(map(str, data["phase_columns"].tolist()))
        if metadata.get("status") != "ok":
            raise ValueError(f"{trial_id}: non-ok adaptation scan")
        raw = np.vstack(
            [
                phase_sequence[columns.index(name)]
                for name in training.INPUT_CHANNELS
            ]
        )
        shape, context = M8DirectionEnsemble._network_inputs(
            raw,
            float(metadata["target_down_force_n"]),
            bit_radius_m,
        )
        angle = training._direction_angle_deg(metadata, 0)
        sequences.append(shape)
        contexts.append(context)
        labels.append(training._direction_class(angle))
        angles.append(angle)
        trial_ids.append(trial_id)
        orbit_ids.append(str(task["symmetry_orbit_id"]))
    return {
        "sequence": np.stack(sequences).astype(np.float32),
        "context": np.stack(contexts).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "angle_deg": np.asarray(angles, dtype=np.float32),
        "trial_ids": trial_ids,
        "orbit_ids": orbit_ids,
        "bit_radius_m": bit_radius_m,
    }


def _hidden_and_logits(
    model: training.CircularDirectionCNN,
    examples: dict,
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    sequence = torch.from_numpy(examples["sequence"])
    normalized_context = (
        examples["context"] - context_mean[None, :]
    ) / context_std[None, :]
    context = torch.from_numpy(normalized_context.astype(np.float32))
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(sequence).flatten(1)
        hidden = model.shared(torch.cat((encoded, context), dim=1))
        state_logits = model.state_head(hidden).cpu().numpy()
        direction_logits = model.direction_head(hidden).cpu().numpy()
    return hidden.detach(), state_logits, direction_logits


def _fit_direction_head(
    model: training.CircularDirectionCNN,
    train: dict,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2sp: float,
    soft_sigma_deg: float,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.direction_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.direction_head.train()
    hidden, _, _ = _hidden_and_logits(model, train, context_mean, context_std)
    target = torch.from_numpy(
        training._circular_soft_targets(train["angle_deg"], soft_sigma_deg)
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.direction_head.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.direction_head.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model.direction_head(hidden)
        direction_loss = -torch.mean(
            torch.sum(target * nn.functional.log_softmax(logits, dim=1), dim=1)
        )
        penalty = sum(
            torch.sum((parameter - initial[name]) ** 2)
            for name, parameter in model.direction_head.named_parameters()
        )
        loss = direction_loss + float(l2sp) * penalty
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                predicted = training._class_angle_deg(
                    torch.argmax(model.direction_head(hidden), dim=1).cpu().numpy()
                )
            errors = training._circular_error_deg(predicted, train["angle_deg"])
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "direction_loss": float(direction_loss.detach()),
                    "l2sp_penalty": float(penalty.detach()),
                    "train_error_median_deg": float(np.median(errors)),
                    "train_within_15_rate": float(np.mean(errors <= 15.0)),
                }
            )
    model.eval()
    return history


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _aps_quantile_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
) -> float:
    scores = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == int(target))[0])
        scores.append(float(cumulative[rank]))
    level = min(
        math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores),
        1.0,
    )
    return float(np.quantile(scores, level, method="higher"))


def _probability_metrics(
    probabilities: np.ndarray,
    examples: dict,
    aps_quantile: float,
) -> dict:
    predicted_class = np.argmax(probabilities, axis=1)
    predicted_deg = training._class_angle_deg(predicted_class)
    errors = training._circular_error_deg(predicted_deg, examples["angle_deg"])
    covered = []
    sizes = []
    for probability, target in zip(probabilities, examples["label"]):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        size = min(
            int(np.searchsorted(cumulative, aps_quantile, side="left")) + 1,
            len(order),
        )
        covered.append(int(target) in set(map(int, order[:size])))
        sizes.append(size)
    return {
        "samples": len(errors),
        "top1_error_median_deg": float(np.median(errors)),
        "top1_error_p90_deg": float(np.quantile(errors, 0.90)),
        "top1_within_15_rate": float(np.mean(errors <= 15.0 + 1.0e-9)),
        "catastrophic_gt90_count": int(np.sum(errors > 90.0)),
        "aps_coverage": float(np.mean(covered)),
        "aps_mean_set_size": float(np.mean(sizes)),
    }


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _build_variant(
    size: str,
    variant: str,
    examples: dict[str, dict],
    output: Path,
    base_contract: dict,
    args: argparse.Namespace,
) -> dict:
    variant_root = output / size / variant
    seed_records = []
    cal_probabilities = []
    train_probabilities = []
    for checkpoint_value in base_contract["checkpoints"]:
        base_checkpoint = _checkpoint_path(checkpoint_value)
        seed_name = base_checkpoint.parent.name
        seed = int(seed_name.rsplit("_", 1)[1])
        base_normalization = np.load(base_checkpoint.parent / "normalization.npz")
        context_mean = np.asarray(base_normalization["context_mean"], dtype=np.float32)
        context_std = np.asarray(base_normalization["context_std"], dtype=np.float32)
        base_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        model = training.CircularDirectionCNN()
        model.load_state_dict(base_state)
        history: list[dict] = []
        if variant == "fine_tuned":
            history = _fit_direction_head(
                model,
                examples["train"],
                context_mean,
                context_std,
                seed=seed,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2sp=args.l2sp,
                soft_sigma_deg=args.soft_sigma_deg,
            )
        elif variant != "cal_only":
            raise ValueError(variant)
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        seed_dir = variant_root / "seeds" / seed_name
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "circular_cnn_best.pt"
        if variant == "cal_only":
            shutil.copy2(base_checkpoint, checkpoint)
        else:
            torch.save(model.state_dict(), checkpoint)
        shutil.copy2(base_checkpoint.parent / "normalization.npz", seed_dir / "normalization.npz")

        trained_state = model.state_dict()
        frozen_equal = all(
            torch.equal(base_state[key], trained_state[key])
            for key in base_state
            if key.startswith(FROZEN_PREFIXES)
        )
        direction_equal = all(
            torch.equal(base_state[key], trained_state[key])
            for key in base_state
            if key.startswith(DIRECTION_PREFIX)
        )
        _, _, cal_logits = _hidden_and_logits(
            model, examples["calibration"], context_mean, context_std
        )
        _, _, train_logits = _hidden_and_logits(
            model, examples["train"], context_mean, context_std
        )
        temperature = training._fit_temperature(
            cal_logits,
            examples["calibration"]["label"],
            torch.device("cpu"),
        )
        cal_probabilities.append(_softmax(cal_logits, temperature))
        train_probabilities.append(_softmax(train_logits, temperature))
        metrics = {
            "stage": "cross_size_S3_direction_head_adaptation",
            "size": size,
            "variant": variant,
            "seed": seed,
            "temperature": temperature,
            "base_checkpoint": _relative(base_checkpoint),
            "base_checkpoint_sha256": _sha256(base_checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "encoder_shared_state_head_frozen_exact": frozen_equal,
            "direction_head_equal_to_base": direction_equal,
            "frozen_state_hash_before": _state_hash(base_state, FROZEN_PREFIXES),
            "frozen_state_hash_after": _state_hash(trained_state, FROZEN_PREFIXES),
            "direction_state_hash_before": _state_hash(base_state, (DIRECTION_PREFIX,)),
            "direction_state_hash_after": _state_hash(trained_state, (DIRECTION_PREFIX,)),
            "trainable_parameter_names": [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "weight_update_data_role": (
                "train" if variant == "fine_tuned" else "none"
            ),
            "calibration_data_role": "calibration",
            "history": history,
            "updated_utc": _utc_now(),
        }
        _atomic_json(seed_dir / "metrics.json", metrics)
        _atomic_json(
            seed_dir / "interface_contract.json",
            {
                "input": {
                    "phase_bins": 360,
                    "sequence_channels": list(training.INPUT_CHANNELS),
                    "dimensionless_context_features": 16,
                    "normalization_source": _relative(
                        base_checkpoint.parent / "normalization.npz"
                    ),
                },
                "adaptation": {
                    "variant": variant,
                    "only_direction_head_trainable": variant == "fine_tuned",
                    "encoder_shared_state_head_frozen": True,
                    "temperature_fit_on": "fresh size-specific calibration C6 orbits",
                },
                "updated_utc": _utc_now(),
            },
        )
        seed_records.append(metrics)

    cal_mean_probability = np.mean(cal_probabilities, axis=0)
    train_mean_probability = np.mean(train_probabilities, axis=0)
    aps_quantile = _aps_quantile_from_probabilities(
        cal_mean_probability,
        examples["calibration"]["label"],
    )
    checkpoints = [
        variant_root / "seeds" / Path(value).parent.name / "circular_cnn_best.pt"
        for value in base_contract["checkpoints"]
    ]
    ensemble_contract = {
        "checkpoints": [_relative(path) for path in checkpoints],
        "combiner": "mean size-calibrated class probabilities",
        "confidence_gate": float(base_contract["confidence_gate"]),
        "aps_quantile": aps_quantile,
        "direction_bins": training.DIRECTION_CLASSES,
        "direction_bin_width_deg": training.DIRECTION_BIN_DEG,
        "structured_transport_gate": base_contract["structured_transport_gate"],
        "state_role": base_contract["state_role"],
        "direction_role": base_contract["direction_role"],
        "fallback": base_contract["fallback"],
        "adaptation_contract": {
            "size": size,
            "variant": variant,
            "encoder_shared_state_head_frozen": True,
            "weight_update_role": "train" if variant == "fine_tuned" else "none",
            "temperature_and_APS_role": "calibration",
            "test_samples_seen": 0,
        },
        "claim_boundary": (
            "Simulation-only small-data cross-size adaptation pilot. Calibration-set "
            "metrics are descriptive, not untouched generalization evidence."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(variant_root / "ensemble_interface_contract.json", ensemble_contract)
    variant_report = {
        "size": size,
        "variant": variant,
        "seeds": seed_records,
        "all_frozen_components_exact": all(
            record["encoder_shared_state_head_frozen_exact"] for record in seed_records
        ),
        "all_direction_heads_changed": all(
            not record["direction_head_equal_to_base"] for record in seed_records
        ),
        "all_cal_only_checkpoints_equal_base": all(
            record["checkpoint_sha256"] == record["base_checkpoint_sha256"]
            for record in seed_records
        ),
        "aps_quantile": aps_quantile,
        "train_descriptive_metrics": _probability_metrics(
            train_mean_probability, examples["train"], aps_quantile
        ),
        "calibration_descriptive_metrics": _probability_metrics(
            cal_mean_probability, examples["calibration"], aps_quantile
        ),
        "ensemble_contract_sha256": _sha256(
            variant_root / "ensemble_interface_contract.json"
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(variant_root / "adaptation_report.json", variant_report)
    return variant_report


def _write_summary(output: Path, report: dict) -> None:
    lines = [
        "# Cross-size S3 direction-head adaptation pilot",
        "",
        "All weights except `direction_head` are bitwise frozen from the three M8 seeds.",
        "Train and calibration are disjoint complete C6 orbit sets; no old S1/S2",
        "evaluation artifact is loaded. Calibration-only changes temperature/APS only.",
        "",
        "| size | variant | train med/p90 | calibration med/p90 | calibration <=15 | APS coverage/size |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        for variant in ("cal_only", "fine_tuned"):
            item = report["sizes"][size][variant]
            train = item["train_descriptive_metrics"]
            cal = item["calibration_descriptive_metrics"]
            lines.append(
                f"| {size.upper()} | {variant} | "
                f"{train['top1_error_median_deg']:.2f}/{train['top1_error_p90_deg']:.2f} deg | "
                f"{cal['top1_error_median_deg']:.2f}/{cal['top1_error_p90_deg']:.2f} deg | "
                f"{cal['top1_within_15_rate']:.3f} | "
                f"{cal['aps_coverage']:.3f}/{cal['aps_mean_set_size']:.2f} |"
            )
    lines.extend(
        [
            "",
            "These are fitting/calibration diagnostics only. The untouched full-loop pilot",
            "is the sole S3 task-level evaluation.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_contract_path = BASE_MODEL_ROOT / "ensemble_interface_contract.json"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    size_reports = {}
    for size in args.sizes:
        contract = _load_contract(data_root, size)
        train_orbits = set(contract["train_orbit_ids"])
        calibration_orbits = set(contract["calibration_orbit_ids"])
        if train_orbits & calibration_orbits:
            raise ValueError(f"{size}: C6 leakage detected before adaptation")
        examples = {
            role: _load_role_examples(data_root, size, contract, role)
            for role in ("train", "calibration")
        }
        size_reports[size] = {
            variant: _build_variant(
                size,
                variant,
                examples,
                output,
                base_contract,
                args,
            )
            for variant in ("cal_only", "fine_tuned")
        }
        _atomic_json(
            output / size / "model_manifest.json",
            {
                "status": "complete",
                "size": size,
                "base_M8_ensemble_contract": _relative(base_contract_path),
                "base_M8_ensemble_contract_sha256": _sha256(base_contract_path),
                "adaptation_dataset_contract": _relative(
                    data_root / size / "dataset_contract.json"
                ),
                "adaptation_dataset_contract_sha256": _sha256(
                    data_root / size / "dataset_contract.json"
                ),
                "train_calibration_C6_intersection": [],
                "old_S1_S2_training_samples_used": 0,
                "variants": {
                    variant: {
                        "root": _relative(output / size / variant),
                        "report_sha256": _sha256(
                            output / size / variant / "adaptation_report.json"
                        ),
                    }
                    for variant in ("cal_only", "fine_tuned")
                },
                "updated_utc": _utc_now(),
            },
        )
    report = {
        "stage": "cross_size_S3_direction_head_adaptation_pilot",
        "sizes": size_reports,
        "base_model_root": _relative(BASE_MODEL_ROOT),
        "base_model_contract_sha256": _sha256(base_contract_path),
        "frozen_modules": ["encoder", "shared", "state_head"],
        "trainable_module": "direction_head",
        "cal_only_weight_updates": False,
        "fine_tuned_weight_data_role": "fresh train C6 orbits only",
        "temperature_APS_data_role": "fresh disjoint calibration C6 orbits only",
        "untouched_full_loop_samples_seen": 0,
        "claim_boundary": (
            "This stage proves provenance and implements the requested small-head "
            "adaptation. It does not itself prove cross-size full-loop transfer."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "sizes": list(args.sizes),
            "report_sha256": _sha256(output / "report.json"),
            "summary_sha256": _sha256(output / "summary.md"),
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze M8 backbone/state and adapt only the direction head."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--l2sp", type=float, default=5.0e-4)
    parser.add_argument("--soft-sigma-deg", type=float, default=7.5)
    args = parser.parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0.0 or args.l2sp < 0.0:
        parser.error("invalid adaptation hyperparameter")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
