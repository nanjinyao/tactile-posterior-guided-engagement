from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/reports/closed_loop_scientific_audit.json"


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    direction = _load(
        "outputs/current/evaluations/p1v5_m8_final_untouched_direction_audit/"
        "independent_direction_report.json"
    )
    state_cad = _load(
        "outputs/current/evaluations/m8_state_cad_fusion_audit/report.json"
    )
    policies = _load(
        "outputs/current/evaluations/m8_four_method_sensor_only_audit/report.json"
    )
    strict_m8_complete_loop = _load(
        "outputs/current/evaluations/m8_strict_24_complete_loop/report.json"
    )
    grid_coverage = _load(
        "outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json"
    )
    entropy_cue = _load(
        "outputs/current/evaluations/m8_entropy_contact_cue_calibration/report.json"
    )
    funnel = _load(
        "outputs/current/evaluations/m8_funnel_guard_ablation/report.json"
    )
    geometry = _load("outputs/current/reports/m8_m10_m12_model_inventory.json")
    cross_size = _load(
        "outputs/current/evaluations/m8_model_m10_m12_zero_shot/report.json"
    )
    cross_size_full_loop = _load(
        "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop/report.json"
    )
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "implemented_endpoint": "fixed Allen bit geometric engagement to CAD socket-bottom band",
            "not_implemented": [
                "measured camera residual prior",
                "KUKA iiwa full-arm hardware loop",
                "motor-current/torque coupling verification",
                "reverse unscrewing and screw extraction",
            ],
            "active_full_loop_evidence_size": "M8",
            "zero_shot_direction_interface_sizes": ["M10", "M12"],
            "full_loop_not_validated_sizes": ["M10", "M12"],
            "cross_size_full_loop_gate": "failed for both M10 and M12",
        },
        "geometry": geometry,
        "direction_interface": {
            "samples": direction["samples"],
            "metrics": direction["metrics"],
            "role": (
                "circular direction posterior and calibrated angular set; "
                "not center point or radial distance"
            ),
            "claim_boundary": direction["claim_boundary"],
        },
        "state_cad_interface": state_cad,
        "cross_size_direction_interface": cross_size,
        "cross_size_full_loop_S2": cross_size_full_loop,
        "strict_m8_complete_loop": strict_m8_complete_loop,
        "finite_grid_coverage_certificate": grid_coverage,
        "historical_capture_only_four_method_audit": policies,
        "research_and_diagnostic_branch": {
            "entropy_grid_contact_cue_audit": entropy_cue,
            "paper_facing_role": "transparent strong diagnostic and Proposed+ component ablation",
            "reason": (
                "not promoted as a superior main method; the strict complete-loop "
                "audit reports it so a strong information-grid comparator is not hidden"
            ),
        },
        "post_capture_guard_audit": funnel,
        "terminal_interface": {
            "online_true_center_or_contact_tags": False,
            "completion_channels": [
                "bottom_force_dwell",
                "cad_depth_low_wrench_dwell",
            ],
            "force_channel_requires_expected_axial_sign": True,
            "both_channels_bound_lateral_force_and_tilt_torque": True,
            "strict_terminal_mode_counts": {
                method: values["terminal_mode_counts"]
                for method, values in strict_m8_complete_loop["methods"].items()
            },
            "strict_successes_exercised_low_wrench_branch_only": True,
            "signed_axial_force_branch_batch_exercised": False,
            "claim_boundary": (
                "Simulation sensor/CAD geometric engagement verifier only; "
                "hardware thresholds and bit/socket coupling remain unvalidated."
            ),
        },
        "online_offline_boundary": {
            "online_controller_uses_true_center": False,
            "online_controller_uses_mujoco_contact_tags": False,
            "offline_scoring_uses_mujoco_geometry": True,
            "terminal_cad_depth_is_geometric_engagement_not_unscrewing": True,
            "preprobe_ambiguity_router_is_upstream_condition_not_evaluated": True,
            "negative_swept_area_update_uses_measured_tip_polyline": True,
            "active_budget_can_be_extended_by_fallback": False,
            "observation_model_advances_once_per_physics_step": True,
            "controller_belief_terminal_share_post_step_measured_sample": True,
            "raw_physical_wrench_drives_fast_hard_stop": True,
            "terminal_outcome_frozen_before_inactive_display_tail": True,
        },
        "proposed_plus_schedule_contract": strict_m8_complete_loop["design"].get(
            "proposed_plus_schedule_contract"
        ),
        "hardware_minimum_plan": {
            "no_contact_calibration": "vision covariance plus F/T frame, gravity, bias, covariance and delay",
            "safety_scans": 12,
            "adaptation_scans": 24,
            "untouched_random_test_scans": 12,
            "maximum_first_active_add": 12,
            "adaptation_scope": "normalization, temperature, small adapter/head before backbone",
        },
        "claim_boundary": (
            "Machine-readable snapshot of simulation evidence. It supports an M8 "
            "active-tactile direction shortcut, certified finite-grid geometry, "
            "a strict-budget CAD-bottom audit, and a post-capture defense layer. "
            "It is not hardware, passivity, wear, "
            "passed cross-size closed-loop transfer, or autonomous-unscrewing evidence."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
