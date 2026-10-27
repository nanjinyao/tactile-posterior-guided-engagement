#!/usr/bin/env python3
"""Train and select compact real-first M8/M10/M12 direction models.

Model selection uses only 48 ``clean_candidate`` samples and a strict
leave-one-size-out split.  C6 copies remain in their source training fold.
The already-consumed holdout, quarantine, and sentinel files are not opened.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from common import (
    ANALYSIS_ROOT,
    BUNDLE_ROOT,
    C6_TRANSFORM_ID,
    DATASET_ROOT,
    DIRECTION_CLASSES,
    FourierLinearModel,
    FrozenSimulationAdapter,
    MODEL_ID,
    REAL_STATE_TO_CANONICAL,
    SIZES,
    SIM_ROOT,
    SIZE_BIT_RADIUS_M,
    SimulationAdapterTailL2SP,
    STATE_TO_INDEX,
    TinyCircularCNN,
    aps_quantile,
    apply_normalization,
    c6_augment,
    circular_soft_targets,
    direction_class,
    direction_metrics,
    initialize_simulation_adapter,
    initialize_simulation_tail_l2sp,
    load_training_arrays,
    normalization,
    set_seed,
    sha256_file,
    softmax_numpy,
    state_metrics,
    write_json,
)
from export_real_multisize_bundle import export_bundle


TRAINING_ROOT = ANALYSIS_ROOT / "training"
SIM_MODEL_ROOT = (
    SIM_ROOT
    / "outputs"
    / "current"
    / "models"
    / "hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving"
)
SIM_SEED_NAMES = ("seed_20260720", "seed_20260721", "seed_20260722")
SIM_PROFILE_PATH = SIM_ROOT / "configs" / "real_calibrated_v1.json"
SIM_REPLAY_PATH = (
    ANALYSIS_ROOT
    / "sim_calibration_v1"
    / "calibrated_observation_replay_v1.npz"
)
DEVELOPMENT_SEED = 20260728
FINAL_SEEDS = (20260728, 20260729, 20260730)
L2SP_COEFFICIENT = 5.0e-3
CANDIDATE_ARCHITECTURES = (
    FourierLinearModel.architecture_type,
    FrozenSimulationAdapter.architecture_type,
    SimulationAdapterTailL2SP.architecture_type,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _soft_direction_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -torch.mean(
        torch.sum(
            target * functional.log_softmax(logits, dim=1),
            dim=1,
        )
    )


def _forward(
    model: nn.Module,
    sequence_raw: torch.Tensor,
    sequence_normalized: torch.Tensor,
    context_raw: torch.Tensor,
    context_normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, FrozenSimulationAdapter):
        return model(sequence_raw, sequence_normalized, context_raw)
    return model(sequence_normalized, context_normalized)


def _preserve_frozen_adapter(model: nn.Module) -> None:
    if isinstance(model, FrozenSimulationAdapter):
        model.encoder.eval()
        if not isinstance(model, SimulationAdapterTailL2SP):
            model.shared.eval()
        model.teacher_state_head.eval()
        model.teacher_direction_head.eval()


def _is_simulation_assisted(architecture: str) -> bool:
    return architecture in {
        FrozenSimulationAdapter.architecture_type,
        SimulationAdapterTailL2SP.architecture_type,
    }


def _predict(
    model: nn.Module,
    sequence_raw: np.ndarray,
    context_raw: np.ndarray,
    normalizer: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    sequence_normalized, context_normalized = apply_normalization(
        sequence_raw, context_raw, normalizer
    )
    model.eval()
    with torch.no_grad():
        state_logits, direction_logits = _forward(
            model,
            torch.from_numpy(sequence_raw),
            torch.from_numpy(sequence_normalized),
            torch.from_numpy(context_raw),
            torch.from_numpy(context_normalized),
        )
    return (
        softmax_numpy(state_logits.numpy()),
        softmax_numpy(direction_logits.numpy()),
    )


def _sim_source(seed_name: str) -> tuple[Path, Path]:
    root = SIM_MODEL_ROOT / "seeds" / seed_name
    return root / "circular_cnn_best.pt", root / "normalization.npz"


def _construct_model(
    architecture: str, simulation_seed_name: str
) -> nn.Module:
    if architecture == FourierLinearModel.architecture_type:
        return FourierLinearModel()
    if architecture == TinyCircularCNN.architecture_type:
        return TinyCircularCNN()
    if architecture == FrozenSimulationAdapter.architecture_type:
        checkpoint, normalizer = _sim_source(simulation_seed_name)
        return initialize_simulation_adapter(checkpoint, normalizer)
    if architecture == SimulationAdapterTailL2SP.architecture_type:
        checkpoint, normalizer = _sim_source(simulation_seed_name)
        return initialize_simulation_tail_l2sp(checkpoint, normalizer)
    raise ValueError(architecture)


def _l2sp_anchor(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, SimulationAdapterTailL2SP):
        return {}
    named_parameters = dict(model.named_parameters())
    missing = set(model.l2sp_parameter_names) - set(named_parameters)
    if missing:
        raise ValueError(f"L2SP parameters are absent: {sorted(missing)}")
    return {
        name: named_parameters[name].detach().clone()
        for name in model.l2sp_parameter_names
    }


def _l2sp_mean_square(
    model: nn.Module, anchor: dict[str, torch.Tensor]
) -> torch.Tensor:
    if not anchor:
        return torch.zeros((), dtype=torch.float32)
    named_parameters = dict(model.named_parameters())
    squared_sum = torch.zeros(
        (),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    element_count = 0
    for name, initial in anchor.items():
        current = named_parameters[name]
        squared_sum = squared_sum + torch.sum(
            torch.square(current - initial.to(current.device))
        )
        element_count += current.numel()
    return squared_sum / max(1, element_count)


def _fit(
    *,
    architecture: str,
    sequence_train: np.ndarray,
    context_train: np.ndarray,
    state_train: np.ndarray,
    direction_train_deg: np.ndarray,
    sequence_validation: np.ndarray,
    context_validation: np.ndarray,
    state_validation: np.ndarray,
    direction_validation_deg: np.ndarray,
    seed: int,
    simulation_seed_name: str,
    simulation_anchor: dict[str, np.ndarray] | None = None,
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, Any]]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    normalizer = normalization(sequence_train, context_train)
    train_sequence_normalized, train_context_normalized = apply_normalization(
        sequence_train, context_train, normalizer
    )
    validation_sequence_normalized, validation_context_normalized = (
        apply_normalization(
            sequence_validation, context_validation, normalizer
        )
    )
    model = _construct_model(architecture, simulation_seed_name)
    l2sp_anchor = _l2sp_anchor(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if architecture == FourierLinearModel.architecture_type:
        learning_rate, maximum_epochs, patience, batch_size = (
            2.0e-2,
            120,
            20,
            len(sequence_train),
        )
    elif architecture == TinyCircularCNN.architecture_type:
        learning_rate, maximum_epochs, patience, batch_size = (
            3.0e-3,
            55,
            12,
            32,
        )
    else:
        learning_rate, maximum_epochs, patience, batch_size = (
            2.0e-3,
            30,
            8,
            32,
        )
    if fixed_epochs is not None:
        maximum_epochs = int(fixed_epochs)
        patience = maximum_epochs + 1
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1.0e-3
    )
    state_tensor = torch.from_numpy(state_train)
    direction_tensor = torch.from_numpy(
        circular_soft_targets(direction_train_deg)
    )
    raw_sequence_tensor = torch.from_numpy(sequence_train)
    normalized_sequence_tensor = torch.from_numpy(
        train_sequence_normalized
    )
    raw_context_tensor = torch.from_numpy(context_train)
    normalized_context_tensor = torch.from_numpy(train_context_normalized)
    validation_direction_target = torch.from_numpy(
        circular_soft_targets(direction_validation_deg)
    )
    validation_state_tensor = torch.from_numpy(state_validation)
    if simulation_anchor is not None:
        simulation_sequence = simulation_anchor["sequence"]
        simulation_context = simulation_anchor["context"]
        simulation_state = simulation_anchor["state"]
        simulation_direction = simulation_anchor["direction_deg"]
        (
            simulation_sequence_normalized,
            simulation_context_normalized,
        ) = apply_normalization(
            simulation_sequence, simulation_context, normalizer
        )
        simulation_sequence_tensor = torch.from_numpy(simulation_sequence)
        simulation_sequence_normalized_tensor = torch.from_numpy(
            simulation_sequence_normalized
        )
        simulation_context_tensor = torch.from_numpy(simulation_context)
        simulation_context_normalized_tensor = torch.from_numpy(
            simulation_context_normalized
        )
        simulation_state_tensor = torch.from_numpy(simulation_state)
        simulation_direction_tensor = torch.from_numpy(
            circular_soft_targets(simulation_direction)
        )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    generator = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        _preserve_frozen_adapter(model)
        order = generator.permutation(len(sequence_train))
        if simulation_anchor is not None:
            simulation_order = generator.permutation(
                len(simulation_anchor["sequence"])
            )
        total_loss = 0.0
        for batch_index, start in enumerate(
            range(0, len(order), batch_size)
        ):
            indices = torch.from_numpy(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            state_logits, direction_logits = _forward(
                model,
                raw_sequence_tensor[indices],
                normalized_sequence_tensor[indices],
                raw_context_tensor[indices],
                normalized_context_tensor[indices],
            )
            state_loss = functional.cross_entropy(
                state_logits, state_tensor[indices]
            )
            direction_loss = _soft_direction_loss(
                direction_logits, direction_tensor[indices]
            )
            real_loss = 0.35 * state_loss + direction_loss
            if simulation_anchor is not None:
                simulation_start = (
                    batch_index * batch_size
                ) % len(simulation_order)
                simulation_indices_numpy = np.concatenate(
                    (simulation_order, simulation_order)
                )[simulation_start : simulation_start + len(indices)]
                simulation_indices = torch.from_numpy(
                    simulation_indices_numpy
                )
                simulation_state_logits, simulation_direction_logits = (
                    _forward(
                        model,
                        simulation_sequence_tensor[simulation_indices],
                        simulation_sequence_normalized_tensor[
                            simulation_indices
                        ],
                        simulation_context_tensor[simulation_indices],
                        simulation_context_normalized_tensor[
                            simulation_indices
                        ],
                    )
                )
                simulation_loss = (
                    0.35
                    * functional.cross_entropy(
                        simulation_state_logits,
                        simulation_state_tensor[simulation_indices],
                    )
                    + _soft_direction_loss(
                        simulation_direction_logits,
                        simulation_direction_tensor[simulation_indices],
                    )
                )
                loss = 0.80 * real_loss + 0.20 * simulation_loss
            else:
                loss = real_loss
            if l2sp_anchor:
                loss = loss + L2SP_COEFFICIENT * _l2sp_mean_square(
                    model, l2sp_anchor
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)

        model.eval()
        with torch.no_grad():
            validation_state_logits, validation_direction_logits = _forward(
                model,
                torch.from_numpy(sequence_validation),
                torch.from_numpy(validation_sequence_normalized),
                torch.from_numpy(context_validation),
                torch.from_numpy(validation_context_normalized),
            )
            validation_loss = float(
                (
                    0.35
                    * functional.cross_entropy(
                        validation_state_logits, validation_state_tensor
                    )
                    + _soft_direction_loss(
                        validation_direction_logits,
                        validation_direction_target,
                    )
                ).item()
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(sequence_train),
                "validation_loss": validation_loss,
            }
        )
        if fixed_epochs is not None:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            continue
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_l2sp_mean_square = float(
        _l2sp_mean_square(model, l2sp_anchor).detach().item()
    )
    return model, normalizer, {
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "best_validation_loss": best_loss,
        "elapsed_s": time.monotonic() - started,
        "history_tail": history[-5:],
        "real_supervision_weight": 0.80
        if simulation_anchor is not None
        else 1.0,
        "calibrated_simulation_anchor_weight": 0.20
        if simulation_anchor is not None
        else 0.0,
        "simulation_anchor_rows": (
            len(simulation_anchor["sequence"])
            if simulation_anchor is not None
            else 0
        ),
        "l2sp": {
            "enabled": bool(l2sp_anchor),
            "coefficient": (
                L2SP_COEFFICIENT if l2sp_anchor else 0.0
            ),
            "anchor_source": (
                "pretrained_simulation_checkpoint"
                if l2sp_anchor
                else "not_applicable"
            ),
            "anchor_parameter_names": sorted(l2sp_anchor),
            "final_mean_square_drift": final_l2sp_mean_square,
        },
    }


def _load_simulation_replay() -> dict[str, np.ndarray]:
    if not SIM_PROFILE_PATH.is_file() or not SIM_REPLAY_PATH.is_file():
        raise FileNotFoundError(
            "real-calibrated simulation profile/replay is unavailable"
        )
    with np.load(SIM_REPLAY_PATH, allow_pickle=False) as data:
        partition = np.asarray(data["partition"]).astype(str)
        mask = partition == "clean_candidate"
        sequence = np.asarray(
            data["sequence_physical"][mask], dtype=np.float32
        )
        size_index = np.asarray(data["size_index"][mask], dtype=np.int64)
        raw_states = np.asarray(data["contact_state"][mask]).astype(str)
        context = np.zeros((len(sequence), 5), dtype=np.float32)
        state = np.empty(len(sequence), dtype=np.int64)
        for index, size_value in enumerate(size_index):
            context[index, size_value] = 1.0
            context[index, 3] = 1.0
            size_name = SIZES[int(size_value)]
            context[index, 4] = (
                SIZE_BIT_RADIUS_M[size_name] / SIZE_BIT_RADIUS_M["m12"]
            )
            state[index] = STATE_TO_INDEX[
                REAL_STATE_TO_CANONICAL[raw_states[index]]
            ]
        output = {
            "sequence": sequence,
            "context": context,
            "state": state,
            "direction_deg": np.asarray(
                data["direction_deg"][mask], dtype=np.float32
            ),
            "size_index": size_index,
            "sample_id": np.asarray(data["sample_id"][mask]).astype(str),
        }
    if len(output["sequence"]) != 144:
        raise ValueError(
            f"expected 144 eligible simulation rows, got "
            f"{len(output['sequence'])}"
        )
    return output


def _subset_simulation(
    replay: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)[mask]
        for key, value in replay.items()
        if key not in ("sample_id", "size_index")
    }


def _split_manifest(arrays) -> dict[str, Any]:
    folds = []
    all_ids = set(arrays.sample_id.tolist())
    for held_size_index, held_size in enumerate(SIZES):
        validation_mask = arrays.size_index == held_size_index
        training_ids = arrays.sample_id[~validation_mask].tolist()
        validation_ids = arrays.sample_id[validation_mask].tolist()
        if set(training_ids) & set(validation_ids):
            raise ValueError("LOSO sample leakage")
        if set(training_ids) | set(validation_ids) != all_ids:
            raise ValueError("LOSO split does not cover clean samples")
        folds.append(
            {
                "fold_id": f"holdout_{held_size}",
                "held_out_size": held_size,
                "training_sample_ids": training_ids,
                "validation_sample_ids": validation_ids,
                "training_origin_count": len(training_ids),
                "training_rows_after_c6": 6 * len(training_ids),
                "validation_origin_count": len(validation_ids),
                "validation_rows": len(validation_ids),
                "validation_augmented": False,
            }
        )
    return {
        "schema_version": 1,
        "split_strategy": "leave_one_size_out",
        "candidate_selection_partition": "clean_candidate_only",
        "holdout_read_during_selection": False,
        "c6_transform_id": C6_TRANSFORM_ID,
        "c6_training_fold_only": True,
        "origin_groups_indivisible": True,
        "folds": folds,
    }


def _candidate_key(record: dict[str, Any]) -> tuple[float, ...]:
    direction = record["aggregate"]["direction"]
    state = record["aggregate"]["state"]
    return (
        float(direction["catastrophic_gt90_count"]),
        float(direction["p90_error_deg"]),
        float(direction["circular_soft_target_nll"]),
        -float(state["macro_f1"]),
    )


def train(output: Path = TRAINING_ROOT) -> dict[str, Any]:
    dataset_contract_path = DATASET_ROOT / "dataset_contract.json"
    training_npz = DATASET_ROOT / "train_clean.npz"
    if not dataset_contract_path.is_file() or not training_npz.is_file():
        raise FileNotFoundError(
            "run build_real_multisize_dataset.py before training"
        )
    contract = _json(dataset_contract_path)
    if contract["training_sample_count"] != 48:
        raise ValueError("training contract is not the 48-sample clean set")
    arrays = load_training_arrays(training_npz)
    if len(arrays.sequence) != 48 or np.any(arrays.state < 0):
        raise ValueError("training NPZ is not clean-only")
    simulation_replay = _load_simulation_replay()

    output.mkdir(parents=True, exist_ok=True)
    split_manifest = _split_manifest(arrays)
    split_manifest["dataset_contract_sha256"] = sha256_file(
        dataset_contract_path
    )
    split_manifest_path = output / "split_manifest.json"
    write_json(split_manifest_path, split_manifest)

    candidates = CANDIDATE_ARCHITECTURES
    candidate_records: list[dict[str, Any]] = []
    selected_oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for architecture in candidates:
        fold_records: list[dict[str, Any]] = []
        all_state_probability = np.zeros((48, 3), dtype=np.float64)
        all_direction_probability = np.zeros(
            (48, DIRECTION_CLASSES), dtype=np.float64
        )
        rejected_reason = ""
        try:
            if _is_simulation_assisted(architecture):
                for seed_name in SIM_SEED_NAMES:
                    checkpoint, normalizer_path = _sim_source(seed_name)
                    if not checkpoint.is_file() or not normalizer_path.is_file():
                        raise FileNotFoundError(
                            f"missing frozen simulation seed {seed_name}"
                        )
            for fold_index, held_size in enumerate(SIZES):
                validation_mask = arrays.size_index == fold_index
                training_mask = ~validation_mask
                training_indices = np.flatnonzero(training_mask)
                (
                    augmented_sequence,
                    augmented_direction,
                    augmented_state,
                    augmented_context,
                    augmented_origin,
                    rotation_index,
                ) = c6_augment(
                    arrays.sequence[training_mask],
                    arrays.direction_deg[training_mask],
                    arrays.state[training_mask],
                    arrays.context[training_mask],
                    training_indices,
                )
                if set(augmented_origin.tolist()) & set(
                    np.flatnonzero(validation_mask).tolist()
                ):
                    raise ValueError("C6 origin leaked into validation")
                model, normalizer, fit_record = _fit(
                    architecture=architecture,
                    sequence_train=augmented_sequence,
                    context_train=augmented_context,
                    state_train=augmented_state,
                    direction_train_deg=augmented_direction,
                    sequence_validation=arrays.sequence[validation_mask],
                    context_validation=arrays.context[validation_mask],
                    state_validation=arrays.state[validation_mask],
                    direction_validation_deg=arrays.direction_deg[
                        validation_mask
                    ],
                    seed=DEVELOPMENT_SEED + fold_index,
                    simulation_seed_name=SIM_SEED_NAMES[fold_index],
                    simulation_anchor=(
                        _subset_simulation(
                            simulation_replay,
                            simulation_replay["size_index"] != fold_index,
                        )
                        if _is_simulation_assisted(architecture)
                        else None
                    ),
                )
                state_probability, direction_probability = _predict(
                    model,
                    arrays.sequence[validation_mask],
                    arrays.context[validation_mask],
                    normalizer,
                )
                all_state_probability[validation_mask] = state_probability
                all_direction_probability[
                    validation_mask
                ] = direction_probability
                fold_records.append(
                    {
                        "fold_id": f"holdout_{held_size}",
                        "held_out_size": held_size,
                        "training_origins": int(np.sum(training_mask)),
                        "training_rows_after_c6": len(augmented_sequence),
                        "validation_origins": int(np.sum(validation_mask)),
                        "validation_rows": int(np.sum(validation_mask)),
                        "validation_augmented": False,
                        "rotation_ids_in_training": sorted(
                            set(rotation_index.tolist())
                        ),
                        "state": state_metrics(
                            state_probability, arrays.state[validation_mask]
                        ),
                        "direction": direction_metrics(
                            direction_probability,
                            arrays.direction_deg[validation_mask],
                        ),
                        "fit": fit_record,
                    }
                )
            aggregate = {
                "state": state_metrics(
                    all_state_probability, arrays.state
                ),
                "direction": direction_metrics(
                    all_direction_probability, arrays.direction_deg
                ),
            }
            candidate_records.append(
                {
                    "architecture_type": architecture,
                    "status": "EVALUATED",
                    "development_seed_policy": (
                        "one deterministic seed per LOSO fold"
                    ),
                    "folds": fold_records,
                    "aggregate": aggregate,
                    "selection_key": list(
                        _candidate_key(
                            {
                                "aggregate": aggregate,
                            }
                        )
                    ),
                    "simulation_anchor": (
                        {
                            "mode": (
                                (
                                    "frozen legacy simulation encoder and "
                                    "teacher heads plus trainable real input/"
                                    "output adapter and shared.0 tail "
                                    "unfreeze with L2SP"
                                )
                                if architecture
                                == SimulationAdapterTailL2SP.architecture_type
                                else (
                                    "frozen legacy simulation encoder/shared/"
                                    "teacher heads plus trainable real input/"
                                    "output adapter"
                                )
                            ),
                            "real_supervision_weight": 0.80,
                            "calibrated_simulation_anchor_weight": 0.20,
                            "tail_unfreeze": (
                                list(
                                    SimulationAdapterTailL2SP
                                    .l2sp_parameter_names
                                )
                                if architecture
                                == SimulationAdapterTailL2SP.architecture_type
                                else []
                            ),
                            "l2sp": {
                                "enabled": architecture
                                == SimulationAdapterTailL2SP.architecture_type,
                                "coefficient": (
                                    L2SP_COEFFICIENT
                                    if architecture
                                    == SimulationAdapterTailL2SP
                                    .architecture_type
                                    else 0.0
                                ),
                                "anchor_source": (
                                    "pretrained_simulation_checkpoint"
                                    if architecture
                                    == SimulationAdapterTailL2SP
                                    .architecture_type
                                    else "not_applicable"
                                ),
                            },
                            "source_root": str(SIM_MODEL_ROOT),
                            "source_contract_sha256": sha256_file(
                                SIM_MODEL_ROOT
                                / "ensemble_interface_contract.json"
                            ),
                            "real_calibrated_profile_path": str(
                                SIM_PROFILE_PATH
                            ),
                            "real_calibrated_profile_sha256": sha256_file(
                                SIM_PROFILE_PATH
                            ),
                            "calibrated_replay_path": str(SIM_REPLAY_PATH),
                            "calibrated_replay_sha256": sha256_file(
                                SIM_REPLAY_PATH
                            ),
                            "eligible_partition": "clean_candidate",
                            "held_out_size_simulation_rows_in_fold": 0,
                        }
                        if _is_simulation_assisted(architecture)
                        else None
                    ),
                }
            )
            selected_oof[architecture] = (
                all_state_probability,
                all_direction_probability,
            )
        except Exception as exception:
            rejected_reason = f"{type(exception).__name__}: {exception}"
            candidate_records.append(
                {
                    "architecture_type": architecture,
                    "status": "REJECTED",
                    "rejection_reason": rejected_reason,
                    "folds_completed": fold_records,
                }
            )

    evaluated = [
        record
        for record in candidate_records
        if record["status"] == "EVALUATED"
    ]
    if not evaluated:
        raise RuntimeError("all model candidates were rejected")
    selected = min(evaluated, key=_candidate_key)
    selected_architecture = selected["architecture_type"]
    selected_state_oof, selected_direction_oof = selected_oof[
        selected_architecture
    ]
    aps_value = aps_quantile(
        selected_direction_oof,
        direction_class(arrays.direction_deg),
        alpha=0.10,
    )
    for record in candidate_records:
        record["selected"] = (
            record["architecture_type"] == selected_architecture
        )
    candidate_metrics = {
        "schema_version": 2,
        "selection_rule": (
            "lexicographic: catastrophic >90 count, p90 direction error, "
            "circular soft-target NLL, then negative state macro-F1"
        ),
        "selection_data": (
            "48 clean samples only; LOSO by size; validation unaugmented"
        ),
        "candidate_order": list(CANDIDATE_ARCHITECTURES),
        "excluded_from_selection": [
            TinyCircularCNN.architecture_type,
        ],
        "candidates": candidate_records,
        "selected_architecture": selected_architecture,
        "selected_oof_aps_quantile_alpha_0p10": aps_value,
    }
    candidate_metrics_path = output / "candidate_metrics.json"
    write_json(candidate_metrics_path, candidate_metrics)

    oof_path = output / "selected_oof_predictions.csv"
    with oof_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "sample_id",
            "size",
            "true_state_index",
            "predicted_state_index",
            "true_direction_deg",
            "predicted_direction_deg",
            "direction_max_probability",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        predicted_class = np.argmax(selected_direction_oof, axis=1)
        predicted_angle = 3.75 + 7.5 * predicted_class
        for index, sample_id in enumerate(arrays.sample_id):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "size": SIZES[int(arrays.size_index[index])],
                    "true_state_index": int(arrays.state[index]),
                    "predicted_state_index": int(
                        np.argmax(selected_state_oof[index])
                    ),
                    "true_direction_deg": float(
                        arrays.direction_deg[index]
                    ),
                    "predicted_direction_deg": float(
                        predicted_angle[index] % 360.0
                    ),
                    "direction_max_probability": float(
                        np.max(selected_direction_oof[index])
                    ),
                }
            )

    selected_best_epochs = [
        int(fold["fit"]["best_epoch"]) for fold in selected["folds"]
    ]
    final_epochs = max(8, int(np.median(selected_best_epochs)))
    (
        all_sequence,
        all_direction,
        all_state,
        all_context,
        all_origin,
        all_rotation,
    ) = c6_augment(
        arrays.sequence,
        arrays.direction_deg,
        arrays.state,
        arrays.context,
        np.arange(len(arrays.sequence)),
    )
    final_records: list[dict[str, Any]] = []
    for final_index, seed in enumerate(FINAL_SEEDS):
        model, normalizer, fit_record = _fit(
            architecture=selected_architecture,
            sequence_train=all_sequence,
            context_train=all_context,
            state_train=all_state,
            direction_train_deg=all_direction,
            sequence_validation=arrays.sequence,
            context_validation=arrays.context,
            state_validation=arrays.state,
            direction_validation_deg=arrays.direction_deg,
            seed=seed,
            simulation_seed_name=SIM_SEED_NAMES[final_index],
            simulation_anchor=(
                _subset_simulation(
                    simulation_replay,
                    np.ones(
                        len(simulation_replay["sequence"]), dtype=bool
                    ),
                )
                if _is_simulation_assisted(selected_architecture)
                else None
            ),
            fixed_epochs=final_epochs,
        )
        state_probability, direction_probability = _predict(
            model, arrays.sequence, arrays.context, normalizer
        )
        seed_root = output / "final_seeds" / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        model_path = seed_root / "model.pt"
        torch.save(model.state_dict(), model_path)
        normalization_path = seed_root / "normalization.npz"
        np.savez(
            normalization_path,
            **normalizer,
            direction_class_angles_deg=(
                3.75 + 7.5 * np.arange(DIRECTION_CLASSES)
            ).astype(np.float32),
        )
        metrics = {
            "seed": seed,
            "architecture_type": selected_architecture,
            "fit": fit_record,
            "fit_data": "all 48 clean origins plus grouped C6 copies",
            "state_training_fit_diagnostic": state_metrics(
                state_probability, arrays.state
            ),
            "direction_training_fit_diagnostic": direction_metrics(
                direction_probability, arrays.direction_deg
            ),
            "selection_metrics_are_oof_not_this_training_fit": True,
        }
        metrics_path = seed_root / "metrics.json"
        write_json(metrics_path, metrics)
        state_dict_shapes = {
            key: list(value.shape)
            for key, value in model.state_dict().items()
        }
        config = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "architecture_type": selected_architecture,
            "constructor": (
                {"harmonics": model.harmonics}
                if isinstance(model, FourierLinearModel)
                else {
                    "legacy_context_features": 16,
                    "simulation_adapter_variant": (
                        "tail_l2sp"
                        if isinstance(model, SimulationAdapterTailL2SP)
                        else "frozen"
                    ),
                    "tail_unfreeze_parameter_names": (
                        list(model.l2sp_parameter_names)
                        if isinstance(model, SimulationAdapterTailL2SP)
                        else []
                    ),
                    "l2sp_coefficient": (
                        L2SP_COEFFICIENT
                        if isinstance(model, SimulationAdapterTailL2SP)
                        else 0.0
                    ),
                }
            ),
            "state_dict_shapes": state_dict_shapes,
            "state_dict_sha256": sha256_file(model_path),
            "normalization_sha256": sha256_file(normalization_path),
            "simulation_seed_source": (
                {
                    "seed": SIM_SEED_NAMES[final_index],
                    "checkpoint_sha256": sha256_file(
                        _sim_source(SIM_SEED_NAMES[final_index])[0]
                    ),
                    "normalization_sha256": sha256_file(
                        _sim_source(SIM_SEED_NAMES[final_index])[1]
                    ),
                }
                if isinstance(model, FrozenSimulationAdapter)
                else None
            ),
        }
        model_config_path = seed_root / "model_config.json"
        write_json(model_config_path, config)
        final_records.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "normalization_path": str(normalization_path),
                "metrics_path": str(metrics_path),
                "model_config_path": str(model_config_path),
            }
        )

    training_report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_contract_path": str(dataset_contract_path),
        "dataset_contract_sha256": sha256_file(dataset_contract_path),
        "training_npz_sha256": sha256_file(training_npz),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "common_source_sha256": sha256_file(
            Path(__file__).resolve().parent / "common.py"
        ),
        "exporter_source_sha256": sha256_file(
            Path(__file__).resolve().parent
            / "export_real_multisize_bundle.py"
        ),
        "requirements_lock_sha256": sha256_file(
            Path(__file__).resolve().parent / "requirements-lock.txt"
        ),
        "sample_authority": "REAL_DEVICE_CLEAN_ONLY",
        "clean_origin_count": 48,
        "c6_rows_final_fit": len(all_sequence),
        "c6_transform_id": C6_TRANSFORM_ID,
        "candidate_architectures": list(CANDIDATE_ARCHITECTURES),
        "candidate_exclusions": [TinyCircularCNN.architecture_type],
        "selected_architecture": selected_architecture,
        "selection_metrics": selected["aggregate"],
        "selection_key": selected["selection_key"],
        "aps_quantile_alpha_0p10": aps_value,
        "fixed_final_epochs": final_epochs,
        "final_seeds": list(FINAL_SEEDS),
        "holdout_used": False,
        "quarantine_used": False,
        "sentinel_used": False,
        "simulation_data_in_normalization": False,
        "simulation_assistance": (
            selected.get("simulation_anchor")
            if _is_simulation_assisted(selected_architecture)
            else "candidate compared but not selected"
        ),
        "claim_boundary": (
            "Development CV evidence for high-level direction/state and a "
            "CAD-bounded capped shortcut. No capture, insertion, terminal, "
            "or Pure-Outward Spiral authority."
        ),
    }
    training_report_path = output / "training_report.json"
    write_json(training_report_path, training_report)
    summary = (
        "# Real multisize offline learning v1\n\n"
        f"Selected `{selected_architecture}` from clean-only LOSO.\n\n"
        f"- state accuracy: {selected['aggregate']['state']['accuracy']:.3f}\n"
        "- direction median / p90: "
        f"{selected['aggregate']['direction']['median_error_deg']:.2f} / "
        f"{selected['aggregate']['direction']['p90_error_deg']:.2f} deg\n"
        "- C6 is grouped augmentation inside training folds only.\n"
        "- The 12 consumed holdouts were not opened for model selection.\n"
        "- Simulation assistance was explicitly compared at 0.8 real / "
        "0.2 frozen-simulation anchor.\n"
    )
    summary_path = output / "summary.md"
    temporary = summary_path.with_suffix(".md.tmp")
    temporary.write_text(summary, encoding="utf-8")
    temporary.replace(summary_path)

    bundle_report = export_bundle(
        selected_architecture=selected_architecture,
        seed_records=final_records,
        split_manifest_path=split_manifest_path,
        training_report_path=training_report_path,
        candidate_metrics_path=candidate_metrics_path,
        dataset_contract_path=dataset_contract_path,
        aps_quantile=aps_value,
        output=BUNDLE_ROOT,
    )
    training_report["bundle"] = bundle_report
    write_json(training_report_path, training_report)
    # Refresh the copied report and bundle hashes after adding the bundle record.
    bundle_report = export_bundle(
        selected_architecture=selected_architecture,
        seed_records=final_records,
        split_manifest_path=split_manifest_path,
        training_report_path=training_report_path,
        candidate_metrics_path=candidate_metrics_path,
        dataset_contract_path=dataset_contract_path,
        aps_quantile=aps_value,
        output=BUNDLE_ROOT,
    )
    return {
        "selected_architecture": selected_architecture,
        "selection_metrics": selected["aggregate"],
        "bundle": bundle_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TRAINING_ROOT)
    arguments = parser.parse_args()
    result = train(arguments.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
