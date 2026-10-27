from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.m8_jin_style_direction import (
    DEFAULT_MODEL_ROOT,
    JinStylePolarCNN,
    circular_error_deg,
    direction_centers_deg,
    direction_class,
    polar_binary_image,
)
import src.train_m8_direction_models as improved


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (20260710, 20260711, 20260712)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _example(trial: improved.LoadedTrial, rotation_index: int) -> tuple[np.ndarray, int, float]:
    sequence = improved._augment_sequence(
        trial.sequence, trial.columns, rotation_index
    )
    paper_sequence = sequence[
        [trial.columns.index(name) for name in ("tip_z_m", "Tx_Nm", "Ty_Nm")]
    ]
    angle = improved._direction_angle_deg(trial.metadata, rotation_index)
    return polar_binary_image(paper_sequence), direction_class(angle), angle


def _examples(
    trials: list[improved.LoadedTrial],
    groups: set[str],
    rotations: tuple[int, ...],
    *,
    sector_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = []
    for trial in trials:
        group = (
            improved._sector_pose_id(trial.metadata)
            if sector_groups
            else trial.metadata["base_pose_id"]
        )
        if group in groups:
            selected.extend(_example(trial, rotation) for rotation in rotations)
    images, labels, angles = zip(*selected)
    return (
        np.stack(images).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(angles, dtype=np.float32),
    )


def _concatenate(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.concatenate([part[index] for part in parts], axis=0) for index in range(3))  # type: ignore[return-value]


def _metrics(probability: np.ndarray, labels: np.ndarray, angles: np.ndarray) -> dict:
    prediction = np.argmax(probability, axis=1)
    predicted_angle = direction_centers_deg()[prediction]
    error = circular_error_deg(predicted_angle, angles)
    return {
        "samples": len(labels),
        "sector_top1_accuracy": float(np.mean(prediction == labels)),
        "angle_error_median_deg": float(np.median(error)),
        "angle_error_p90_deg": float(np.quantile(error, 0.90)),
        "angle_within_7p5_rate": float(np.mean(error <= 7.5 + 1.0e-9)),
        "angle_within_15_rate": float(np.mean(error <= 15.0 + 1.0e-9)),
        "catastrophic_gt90_rate": float(np.mean(error > 90.0)),
    }


def _probabilities(model: JinStylePolarCNN, arrays: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> np.ndarray:
    images = arrays[0]
    loader = DataLoader(TensorDataset(torch.from_numpy(images)), batch_size=256)
    output = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
    return np.vstack(output)


def _train_seed(
    seed: int,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Path, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = JinStylePolarCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    dataset = TensorDataset(torch.from_numpy(train[0]), torch.from_numpy(train[1]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    seed_dir = output / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / "polar_cnn_best.pt"
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in loader:
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability = _probabilities(model, validation, device)
        val_loss = float(
            -np.mean(
                np.log(
                    np.clip(
                        probability[np.arange(len(validation[1])), validation[1]],
                        1.0e-12,
                        1.0,
                    )
                )
            )
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
            **_metrics(probability, validation[1], validation[2]),
        }
        history.append(record)
        if val_loss < best_loss - 1.0e-5:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    probability = _probabilities(model, validation, device)
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation": _metrics(probability, validation[1], validation[2]),
        "history": history,
    }
    _atomic_json(seed_dir / "metrics.json", metrics)
    return checkpoint, metrics


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_seed = 20260710
    p0b = improved.load_trials(improved.P0B)
    p0d = improved.load_trials(improved.P0D)
    targeted = improved.load_trials(improved.P1V3_TARGETED)
    p0b_split = improved.grouped_split(p0b, split_seed)
    p0d_split = improved.grouped_sector_split(p0d, split_seed)
    targeted_split = improved.grouped_targeted_split(targeted, split_seed)

    train = _concatenate(
        [
            _examples(p0b, set(p0b_split["train"]), tuple(range(6))),
            _examples(
                p0d,
                set(p0d_split["train"]),
                (0,),
                sector_groups=True,
            ),
            _examples(targeted, set(targeted_split["train"]), (0,)),
        ]
    )
    validation = _concatenate(
        [
            _examples(p0b, set(p0b_split["val"]), (0,)),
            _examples(
                p0d,
                set(p0d_split["val"]),
                (0,),
                sector_groups=True,
            ),
            _examples(targeted, set(targeted_split["val"]), (0,)),
        ]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoints = []
    seed_metrics = []
    for seed in args.seeds:
        checkpoint, metrics = _train_seed(
            seed,
            train,
            validation,
            output,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=device,
        )
        checkpoints.append(checkpoint)
        seed_metrics.append(metrics)

    ensemble_models = []
    for checkpoint in checkpoints:
        model = JinStylePolarCNN().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        ensemble_models.append(model)
    ensemble_probability = np.mean(
        [_probabilities(model, validation, device) for model in ensemble_models],
        axis=0,
    )
    report = {
        "experiment": "M8_Jin_2021_style_polar_CNN_direction_proxy_training",
        "created_utc": _utc_now(),
        "device": str(device),
        "seeds": list(args.seeds),
        "split_seed": split_seed,
        "train_examples": len(train[1]),
        "validation_examples": len(validation[1]),
        "ensemble_validation": _metrics(
            ensemble_probability, validation[1], validation[2]
        ),
        "seed_metrics": seed_metrics,
        "paper_fixed_contract": {
            "probe": "15 degree complete tilt-then-rotate with constant downward force",
            "input_channels": ["Xz", "Mx", "My"],
            "source_samples": 2000,
            "moving_average_samples": 20,
            "polar_binary_canvas": [200, 200],
            "network_input": [3, 20, 20],
            "architecture": "two convolution, two max-pooling, one fully-connected layer",
        },
        "m8_proxy_choices_locked_before_evaluation": {
            "direction_classes": 12,
            "class_centers_deg": direction_centers_deg().tolist(),
            "normalization": "per-scan per-channel min-max",
            "raster": "radius=0.12+0.82*normalized, 2-pixel closed polyline",
            "downsampling": "deterministic 10x10 max pooling",
            "conv_channels": [16, 32],
            "kernel": "3x3 padding 1",
            "optimizer": "Adam lr=1e-3",
            "training_split": "same grouped P0b/P0d/P1v3 development split as improved model",
        },
        "claim_boundary": (
            "Jin-2021-style M8 proxy, not a faithful reproduction: the paper "
            "does not define hex classes, raster normalization, convolution "
            "widths/kernels, optimizer, or direct-slide controller. P1v5/P1v6 "
            "and strict direct-ray trials are not loaded during training."
        ),
    }
    _atomic_json(output / "training_report.json", report)
    _atomic_json(
        output / "runtime_contract.json",
        {
            "checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
            "combiner": "mean uncalibrated sector probabilities",
            "direction_classes": 12,
            "class_centers_deg": direction_centers_deg().tolist(),
            "paper_reference": "Jin et al. 2021, arXiv:2101.12467",
            "claim_boundary": report["claim_boundary"],
            "created_utc": report["created_utc"],
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered Jin-2021-style M8 direction proxy."
    )
    parser.add_argument("--output", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.seeds or args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("seeds, epochs, patience, and batch size must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
