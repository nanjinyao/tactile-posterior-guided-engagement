from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = OFFLINE_ROOT.parent
sys.path.insert(0, str(OFFLINE_ROOT))

from common import (  # noqa: E402
    BUNDLE_ROOT,
    C6_TRANSFORM_ID,
    DATASET_ROOT,
    FourierLinearModel,
    FrozenSimulationAdapter,
    SimulationAdapterTailL2SP,
    TinyCircularCNN,
    c6_augment,
    initialize_simulation_adapter,
    initialize_simulation_tail_l2sp,
    load_training_arrays,
)
from train_real_multisize_direction import (  # noqa: E402
    CANDIDATE_ARCHITECTURES,
    L2SP_COEFFICIENT,
    SIM_MODEL_ROOT,
    _l2sp_anchor,
    _l2sp_mean_square,
    _load_simulation_replay,
    _split_manifest,
)


def test_dataset_contract_and_partition_firewall() -> None:
    contract = json.loads(
        (DATASET_ROOT / "dataset_contract.json").read_text(encoding="utf-8")
    )
    assert contract["sample_count"] == 78
    assert contract["partition_counts"] == {
        "clean_candidate": 48,
        "holdout": 12,
        "quarantine": 12,
        "sentinel": 6,
    }
    assert contract["training_partition"] == "clean_candidate"
    clocking = contract["cad_clocking_contract"]
    assert clocking["design_revision"] == (
        "v1.5-integrated-flange-tools-yaw-datum"
    )
    assert clocking["shape_rotation_from_legacy_deg"] == -15.0
    assert clocking[
        "operator_taught_peg_flat_normal_in_fixture_deg"
    ] == 0.0
    assert "must not be applied a second time" in clocking["runtime_rule"]
    arrays = load_training_arrays(DATASET_ROOT / "train_clean.npz")
    assert arrays.sequence.shape == (48, 7, 360)
    assert arrays.context.shape == (48, 5)
    assert np.all(arrays.state >= 0)
    with np.load(
        DATASET_ROOT / "nontraining_diagnostics.npz", allow_pickle=False
    ) as diagnostic:
        assert diagnostic["sequence"].shape == (30, 7, 360)
        assert not set(arrays.sample_id) & set(
            diagnostic["sample_id"].astype(str)
        )


def test_c6_transform_is_grouped_and_matches_real_frame_convention() -> None:
    sequence = np.zeros((1, 7, 360), dtype=np.float32)
    sequence[:, 0] = 1.0
    output, angle, state, context, origin, rotation = c6_augment(
        sequence,
        np.asarray([3.75], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.zeros((1, 5), dtype=np.float32),
        np.asarray([17], dtype=np.int64),
    )
    assert output.shape == (6, 7, 360)
    assert np.allclose(output[1, 0], 0.5, atol=1.0e-6)
    assert np.allclose(output[1, 1], np.sqrt(3.0) / 2.0, atol=1.0e-6)
    assert angle.tolist() == pytest.approx(
        [3.75, 63.75, 123.75, 183.75, 243.75, 303.75]
    )
    assert state.tolist() == [1] * 6
    assert origin.tolist() == [17] * 6
    assert rotation.tolist() == list(range(6))
    assert C6_TRANSFORM_ID == "REAL_FRAME_C6_V2_RPOS_PHASE_NEG_CAD_V1P5"


def test_loso_has_no_origin_or_size_leakage() -> None:
    arrays = load_training_arrays(DATASET_ROOT / "train_clean.npz")
    manifest = _split_manifest(arrays)
    assert len(manifest["folds"]) == 3
    for fold in manifest["folds"]:
        training = set(fold["training_sample_ids"])
        validation = set(fold["validation_sample_ids"])
        assert len(training) == 32
        assert len(validation) == 16
        assert not training & validation
        assert fold["training_rows_after_c6"] == 192
        assert fold["validation_rows"] == 16
        assert fold["validation_augmented"] is False


def test_real_calibrated_simulation_replay_firewall() -> None:
    replay = _load_simulation_replay()
    assert replay["sequence"].shape == (144, 7, 360)
    assert set(replay["size_index"].tolist()) == {0, 1, 2}
    for held_size in range(3):
        training_mask = replay["size_index"] != held_size
        assert int(np.sum(training_mask)) == 96
        assert held_size not in set(replay["size_index"][training_mask])


def test_frozen_adapter_state_dict_roundtrip() -> None:
    source = SIM_MODEL_ROOT / "seeds" / "seed_20260720"
    first = initialize_simulation_adapter(
        source / "circular_cnn_best.pt", source / "normalization.npz"
    )
    second = initialize_simulation_adapter(
        source / "circular_cnn_best.pt", source / "normalization.npz"
    )
    second.load_state_dict(first.state_dict())
    first.eval()
    second.eval()
    sequence = torch.zeros(2, 7, 360)
    context = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 0.6], [0.0, 0.0, 1.0, 1.0, 1.0]]
    )
    with torch.no_grad():
        first_output = first(sequence, sequence, context)
        second_output = second(sequence, sequence, context)
    assert torch.equal(first_output[0], second_output[0])
    assert torch.equal(first_output[1], second_output[1])


def test_candidate_contract_has_exact_three_approved_models() -> None:
    assert CANDIDATE_ARCHITECTURES == (
        FourierLinearModel.architecture_type,
        FrozenSimulationAdapter.architecture_type,
        SimulationAdapterTailL2SP.architecture_type,
    )
    assert TinyCircularCNN.architecture_type not in CANDIDATE_ARCHITECTURES


def test_tail_candidate_unfreezes_only_adapters_and_l2sp_tail() -> None:
    source = SIM_MODEL_ROOT / "seeds" / "seed_20260720"
    model = initialize_simulation_tail_l2sp(
        source / "circular_cnn_best.pt", source / "normalization.npz"
    )
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "input_adapter.weight",
        "input_adapter.bias",
        "shared.0.weight",
        "shared.0.bias",
        "state_head.weight",
        "state_head.bias",
        "direction_head.weight",
        "direction_head.bias",
    }
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.")
        or name.startswith("teacher_state_head.")
        or name.startswith("teacher_direction_head.")
    )
    anchor = _l2sp_anchor(model)
    assert tuple(anchor) == model.l2sp_parameter_names
    assert float(_l2sp_mean_square(model, anchor)) == 0.0
    assert L2SP_COEFFICIENT > 0.0
    with torch.no_grad():
        model.shared[0].weight[0, 0] += 1.0
    assert float(_l2sp_mean_square(model, anchor)) > 0.0


def test_exported_bundle_hashes_and_authority_when_available() -> None:
    manifest_path = BUNDLE_ROOT / "deployment_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("training/export has not completed yet")
    deployment = json.loads(manifest_path.read_text(encoding="utf-8"))
    promotion = json.loads(
        (BUNDLE_ROOT / "promotion_contract.json").read_text(encoding="utf-8")
    )
    interface = json.loads(
        (BUNDLE_ROOT / "ensemble_interface_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert deployment["deployment_authority"] == "CAPPED_SHORTCUT"
    assert deployment["hardware_control_sizes"] == ["m12"]
    assert promotion["status"] == "QUALIFIED"
    assert promotion["qualification"] == "DEVELOPMENT_CV_CONTROL"
    assert promotion["authorization"]["capped_shortcut_hardware_control"]
    assert promotion["authorization"]["hardware_control_sizes"] == ["m12"]
    assert not promotion["authorization"]["capture_control"]
    assert interface["input"]["shape"] == [7, 360]
    assert interface["input"]["context_shape"] == [5]
    assert interface["output"]["state_names"] == [
        "supported_search",
        "double_overhang",
        "outer_partial_support",
    ]
    assert interface["policy_boundary"]["hardware_control_sizes"] == ["m12"]
    candidate_metrics = json.loads(
        (BUNDLE_ROOT / "candidate_metrics.json").read_text(encoding="utf-8")
    )
    assert candidate_metrics["candidate_order"] == list(
        CANDIDATE_ARCHITECTURES
    )
    assert candidate_metrics["excluded_from_selection"] == [
        TinyCircularCNN.architecture_type
    ]
    l2sp_candidate = next(
        item
        for item in candidate_metrics["candidates"]
        if item["architecture_type"]
        == SimulationAdapterTailL2SP.architecture_type
    )
    assert l2sp_candidate["simulation_anchor"]["l2sp"] == {
        "enabled": True,
        "coefficient": L2SP_COEFFICIENT,
        "anchor_source": "pretrained_simulation_checkpoint",
    }
    for fold in l2sp_candidate["folds"]:
        assert fold["fit"]["l2sp"]["enabled"]
        assert fold["fit"]["l2sp"]["anchor_parameter_names"] == [
            "shared.0.bias",
            "shared.0.weight",
        ]
        assert fold["fit"]["real_supervision_weight"] == 0.80
        assert fold["fit"]["calibrated_simulation_anchor_weight"] == 0.20
    hashes = json.loads(
        (BUNDLE_ROOT / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    import hashlib

    for relative, expected in hashes.items():
        digest = hashlib.sha256((BUNDLE_ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected
