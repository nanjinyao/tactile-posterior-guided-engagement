from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from src.mujoco_admittance_rollout import MuJoCoAdmittanceRollout
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "current" / "datasets" / "p0a_m8_symmetry"
DEFAULT_P0B_OUT = ROOT / "outputs" / "current" / "datasets" / "p0b_m8_observability"
DEFAULT_P0C_OUT = ROOT / "outputs" / "current" / "datasets" / "p0c_m8_collision_followup"
DEFAULT_P0D_OUT = ROOT / "outputs" / "current" / "datasets" / "p0d_m8_sector_anchors"
DEFAULT_P1V2_OUT = ROOT / "outputs" / "current" / "evaluations" / "p1v2_m8_independent_direction_audit"
DEFAULT_P1V3_TRAIN_OUT = ROOT / "outputs" / "current" / "datasets" / "p1v3_m8_targeted_direction_training"
DEFAULT_P1V4_AUDIT_OUT = ROOT / "outputs" / "current" / "evaluations" / "p1v4_m8_untouched_direction_audit"
DEFAULT_P1V5_AUDIT_OUT = ROOT / "outputs" / "current" / "evaluations" / "p1v5_m8_final_untouched_direction_audit"
DEFAULT_P1V6_JIN_COMPARISON_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "p1v6_m8_jin_comparison_direction_audit"
)
MODEL_PATH = ROOT / "models" / "allen_m8_source.xml"
SPEC_PATH = ROOT / "configs" / "allen_m8_source.json"

SIM_TIMESTEP_S = 0.0005
CONTROLLER_DT_S = 0.002
TARGET_DOWN_FORCE_N = 3.0
ALPHA_RAD = math.radians(15.0)
SETTLE_DURATION_S = 0.80
RAMP_DURATION_S = 10.0 / 15.0
SCAN_DURATION_S = 52.0 / 15.0
RETURN_DURATION_S = 22.0 / 15.0

# Offline dataset-quality gates.  These deliberately live in the sampler,
# rather than in the controller: ``contact_count`` is privileged MuJoCo state
# and must never be presented as a deployable real-robot observation.
CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION = 1
SUSTAINED_CONTACT_EVENT_MIN_S = 0.006
LOW_FORCE_TARGET_FRACTION = 0.35
LOW_FORCE_ABSOLUTE_FLOOR_N = 0.25
WITHIN_SCAN_PROGRESS_CHAMFER_FRACTION = 1.0

RAW_COLUMNS = (
    "time_s",
    "phase_rad",
    "Fx_N",
    "Fy_N",
    "Fz_N",
    "Tx_Nm",
    "Ty_Nm",
    "Tz_Nm",
    "tip_x_m",
    "tip_y_m",
    "tip_z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "cmd_dx_m",
    "cmd_dy_m",
    "cmd_dz_m",
    "cmd_droll_rad",
    "cmd_dpitch_rad",
    "cmd_dyaw_rad",
    "nominal_roll_rad",
    "nominal_pitch_rad",
    "nominal_yaw_rad",
    "contact_count",
)
PHASE_COLUMNS = RAW_COLUMNS[2:]


@dataclass(frozen=True)
class PilotTask:
    trial_id: str
    base_pose_id: str
    search_state: str
    radial_level: int
    radius_m: float
    base_angle_deg: float
    rotation_index: int
    rotation_deg: float
    x_m: float
    y_m: float
    nominal_tip_z_m: float
    outer_signed_distance_m: float
    inner_signed_distance_m: float
    nuisance_id: str = "nominal"
    friction_slide: float = 0.75
    contact_timeconst_s: float = 0.05
    target_down_force_n: float = TARGET_DOWN_FORCE_N
    yaw_rad: float = 0.0
    nuisance_seed: int = 0
    symmetry_orbit_id: str = ""
    # Optional pre-hardware observation-chain nuisance.  Historical datasets
    # leave this disabled; cross-size adaptation can explicitly match the
    # delayed/noisy full-loop sensing contract without changing contact
    # physics or labels.
    sensor_noise_seed: int | None = None
    sensor_delay_s: float = 0.0


_WORKER_MODEL: mujoco.MjModel | None = None
_WORKER_DATA: mujoco.MjData | None = None
_WORKER_SPEC: dict | None = None
_WORKER_PHASE_BINS = 360
_WORKER_OUT: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _rotate_xy(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return rotation @ np.asarray(vector, dtype=float)


def build_symmetry_tasks(spec: dict) -> list[PilotTask]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])

    outer_double_boundary = head_radius - bit_radius
    double_outer_boundary = socket_top_apothem + bit_radius
    if not 0.0 < outer_double_boundary < double_outer_boundary:
        raise ValueError("invalid M8 support-state radial boundaries")

    state_radii = {
        "supported_search": np.linspace(0.55, 0.90, 4) * outer_double_boundary,
        "double_overhang": np.linspace(0.20, 0.80, 4) * (
            double_outer_boundary - outer_double_boundary
        )
        + outer_double_boundary,
        "outer_partial": double_outer_boundary + np.array([0.00015, 0.00045, 0.00075, 0.00105]),
    }
    base_angles_deg = (7.5, 22.5, 37.5, 52.5)
    tasks: list[PilotTask] = []
    for state, radii in state_radii.items():
        for radial_level, (radius, base_angle_deg) in enumerate(zip(radii, base_angles_deg)):
            outer_signed = head_radius - (float(radius) + bit_radius)
            inner_signed = (float(radius) - bit_radius) - socket_top_apothem
            for rotation_index in range(6):
                rotation_deg = 60.0 * rotation_index
                angle_deg = base_angle_deg + rotation_deg
                angle_rad = math.radians(angle_deg)
                x_m = float(radius) * math.cos(angle_rad)
                y_m = float(radius) * math.sin(angle_rad)
                base_pose_id = f"{state}_r{radial_level:02d}"
                trial_id = f"{base_pose_id}_rot{rotation_index:02d}"
                tasks.append(
                    PilotTask(
                        trial_id=trial_id,
                        base_pose_id=base_pose_id,
                        search_state=state,
                        radial_level=radial_level,
                        radius_m=float(radius),
                        base_angle_deg=base_angle_deg,
                        rotation_index=rotation_index,
                        rotation_deg=rotation_deg,
                        x_m=x_m,
                        y_m=y_m,
                        nominal_tip_z_m=0.00010,
                        outer_signed_distance_m=outer_signed,
                        inner_signed_distance_m=inner_signed,
                    )
                )
    return tasks


def build_observability_tasks(spec: dict) -> list[PilotTask]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    outer_double_boundary = head_radius - bit_radius
    double_outer_boundary = socket_top_apothem + bit_radius

    state_radii = {
        "supported_search": np.linspace(0.50, 0.93, 5) * outer_double_boundary,
        "double_overhang": np.linspace(0.10, 0.90, 5) * (
            double_outer_boundary - outer_double_boundary
        )
        + outer_double_boundary,
        "outer_partial": double_outer_boundary + np.linspace(0.00010, 0.00110, 5),
    }
    angles_deg = np.linspace(3.75, 56.25, 8)
    count = len(state_radii) * 5 * len(angles_deg) * 3
    sobol = qmc.Sobol(d=4, scramble=True, seed=20260710).random_base2(9)[:count]
    nuisance_index = 0
    tasks: list[PilotTask] = []
    for state, radii in state_radii.items():
        for radial_level, radius in enumerate(radii):
            outer_signed = head_radius - (float(radius) + bit_radius)
            inner_signed = (float(radius) - bit_radius) - socket_top_apothem
            for angle_index, angle_deg in enumerate(angles_deg):
                angle_rad = math.radians(float(angle_deg))
                for repeat_index in range(3):
                    sample = sobol[nuisance_index]
                    nuisance_index += 1
                    friction = 0.55 + 0.40 * float(sample[0])
                    contact_timeconst = 0.040 + 0.030 * float(sample[1])
                    down_force = 2.5 + float(sample[2])
                    yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
                    base_pose_id = f"{state}_r{radial_level:02d}_a{angle_index:02d}"
                    trial_id = f"{base_pose_id}_n{repeat_index:02d}"
                    tasks.append(
                        PilotTask(
                            trial_id=trial_id,
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=float(angle_deg),
                            rotation_index=0,
                            rotation_deg=0.0,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"sobol_{repeat_index}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=nuisance_index - 1,
                        )
                    )
    return tasks


def build_sector_anchor_tasks(spec: dict) -> list[PilotTask]:
    """Pair nuisance values across six physical sectors to audit C6 residuals."""
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    outer_double_boundary = head_radius - bit_radius
    double_outer_boundary = socket_top_apothem + bit_radius
    all_radii = {
        "supported_search": np.linspace(0.50, 0.93, 5) * outer_double_boundary,
        "double_overhang": np.linspace(0.10, 0.90, 5)
        * (double_outer_boundary - outer_double_boundary)
        + outer_double_boundary,
        "outer_partial": double_outer_boundary + np.linspace(0.00010, 0.00110, 5),
    }
    radial_levels = (0, 2, 4)
    local_angles_deg = (3.75, 26.25, 48.75)
    orbit_count = len(all_radii) * len(radial_levels) * len(local_angles_deg) * 2
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260712).random_base2(6)[:orbit_count]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for state, radii in all_radii.items():
        for radial_level in radial_levels:
            radius = float(radii[radial_level])
            outer_signed = head_radius - (radius + bit_radius)
            inner_signed = (radius - bit_radius) - socket_top_apothem
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                for repeat_index in range(2):
                    sample = nuisance[nuisance_index]
                    nuisance_index += 1
                    friction = 0.55 + 0.40 * float(sample[0])
                    contact_timeconst = 0.040 + 0.030 * float(sample[1])
                    down_force = 2.5 + float(sample[2])
                    yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
                    orbit_id = (
                        f"{state}_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                    )
                    for sector_index in range(6):
                        rotation_deg = 60.0 * sector_index
                        physical_angle_deg = local_angle_deg + rotation_deg
                        physical_angle_rad = math.radians(physical_angle_deg)
                        base_pose_id = (
                            f"{state}_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                        )
                        trial_id = f"{orbit_id}_s{sector_index:02d}"
                        tasks.append(
                            PilotTask(
                                trial_id=trial_id,
                                base_pose_id=base_pose_id,
                                search_state=state,
                                radial_level=radial_level,
                                radius_m=radius,
                                base_angle_deg=local_angle_deg,
                                rotation_index=sector_index,
                                rotation_deg=rotation_deg,
                                x_m=radius * math.cos(physical_angle_rad),
                                y_m=radius * math.sin(physical_angle_rad),
                                nominal_tip_z_m=0.00010,
                                outer_signed_distance_m=outer_signed,
                                inner_signed_distance_m=inner_signed,
                                nuisance_id=f"sector_sobol_{repeat_index}",
                                friction_slide=friction,
                                contact_timeconst_s=contact_timeconst,
                                target_down_force_n=down_force,
                                yaw_rad=yaw_rad,
                                nuisance_seed=nuisance_index - 1,
                                symmetry_orbit_id=orbit_id,
                            )
                        )
    return tasks


def build_independent_direction_audit_tasks(spec: dict) -> list[PilotTask]:
    """Build a frozen deployment-weighted set that is never used for training."""
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    full_support_radius = head_radius - bit_radius
    visual_bound = 0.004895
    radial_strata = (
        ("nominal", 0.45 * full_support_radius),
        ("nominal", 0.75 * full_support_radius),
        ("nominal", 0.97 * full_support_radius),
        ("transition", full_support_radius + 0.00045),
        ("transition", 0.00455),
    )
    if radial_strata[-1][1] >= visual_bound:
        raise ValueError("independent audit must remain inside the declared visual bound")

    local_angles_deg = (15.0, 45.0)
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260713).random_base2(4)[:15]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, (band, radius) in enumerate(radial_strata):
        state = "supported_search" if band == "nominal" else "double_overhang"
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - socket_top_apothem
        for repeat_index in range(3):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = f"p1v2_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle_deg + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_audit_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=local_angle_deg,
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"p1v2_sobol_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=20000 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_targeted_direction_training_tasks(spec: dict) -> list[PilotTask]:
    """Cover difficult radial regimes with continuous off-grid physical angles."""
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    full_support_radius = head_radius - bit_radius
    radial_strata = (
        ("supported_search", 0.36 * full_support_radius),
        ("supported_search", 0.50 * full_support_radius),
        ("supported_search", 0.62 * full_support_radius),
        ("double_overhang", 0.00410),
        ("double_overhang", 0.00440),
        ("double_overhang", 0.00470),
    )
    angle_samples = qmc.Sobol(d=1, scramble=True, seed=20260714).random_base2(2)[:, 0]
    local_angles_deg = tuple(sorted(2.0 + 56.0 * float(value) for value in angle_samples))
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260714).random_base2(4)[:12]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, (state, radius) in enumerate(radial_strata):
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - socket_top_apothem
        for repeat_index in range(2):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = f"p1v3_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle_deg + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_target_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=float(local_angle_deg),
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"p1v3_sobol_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=30000 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_untouched_direction_audit_tasks(spec: dict) -> list[PilotTask]:
    """Build a second audit after the P1v3 model and gate have been frozen."""
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    radial_strata = (
        ("supported_search", 0.00122),
        ("supported_search", 0.00170),
        ("supported_search", 0.00260),
        ("double_overhang", 0.00378),
        ("double_overhang", 0.00432),
    )
    local_angles_deg = (23.0, 50.0)
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260715).random_base2(4)[:15]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, (state, radius) in enumerate(radial_strata):
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - socket_top_apothem
        for repeat_index in range(3):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = f"p1v4_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle_deg + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_audit2_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=local_angle_deg,
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"p1v4_sobol_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=40000 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_final_untouched_direction_audit_tasks(spec: dict) -> list[PilotTask]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    radial_strata = (
        ("supported_search", 0.00140),
        ("supported_search", 0.00195),
        ("supported_search", 0.00280),
        ("double_overhang", 0.00360),
        ("double_overhang", 0.00458),
    )
    local_angles_deg = (7.0, 41.0)
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260716).random_base2(4)[:15]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, (state, radius) in enumerate(radial_strata):
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - socket_top_apothem
        for repeat_index in range(3):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = f"p1v5_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle_deg + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_audit3_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=local_angle_deg,
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"p1v5_sobol_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=50000 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_jin_comparison_direction_audit_tasks(spec: dict) -> list[PilotTask]:
    """Build the frozen P1v6 paired direction set for both estimators.

    The physical conical sampler is unchanged.  Each Sobol nuisance draw is
    paired across both local angles and all six C6 sectors so the improved and
    Jin-2021-style estimators can consume byte-identical trials downstream.
    """
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    visual_bound_m = 0.004895
    radial_strata = (
        ("supported_search", 0.00132),
        ("supported_search", 0.00212),
        ("supported_search", 0.00295),
        ("double_overhang", 0.00388),
        ("double_overhang", 0.00472),
    )
    if radial_strata[-1][1] >= visual_bound_m:
        raise ValueError("P1v6 comparison audit must remain inside the declared visual bound")

    local_angles_deg = (17.0, 53.0)
    nuisance = qmc.Sobol(d=4, scramble=True, seed=20260732).random_base2(4)[:15]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, (state, radius) in enumerate(radial_strata):
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - socket_top_apothem
        for repeat_index in range(3):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-1.5 + 3.0 * float(sample[3]))
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = f"p1v6_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle_deg + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_jin_compare_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=local_angle_deg,
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"p1v6_sobol_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=60000 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_collision_tasks(spec: dict, report_path: Path | None = None) -> list[PilotTask]:
    source_report = report_path or (DEFAULT_P0B_OUT / "observability_report.json")
    report = json.loads(source_report.read_text(encoding="utf-8"))
    selected_ids: set[str] = set()
    for record in report["shape_plus_absolute"]["collision_records"]:
        if record["is_collision"]:
            selected_ids.add(record["base_pose_id"])
            selected_ids.add(record["nearest_base_pose_id"])
    base_tasks: dict[str, PilotTask] = {}
    for task in build_observability_tasks(spec):
        base_tasks.setdefault(task.base_pose_id, task)
    missing = selected_ids - set(base_tasks)
    if missing:
        raise ValueError(f"collision report references unknown groups: {sorted(missing)}")

    count = len(selected_ids) * 5
    power = int(math.ceil(math.log2(max(count, 1))))
    samples = qmc.Sobol(d=4, scramble=True, seed=20260711).random_base2(power)[:count]
    tasks = []
    sample_index = 0
    for base_pose_id in sorted(selected_ids):
        base = base_tasks[base_pose_id]
        for repeat_index in range(5):
            sample = samples[sample_index]
            sample_index += 1
            tasks.append(
                replace(
                    base,
                    trial_id=f"{base_pose_id}_n{repeat_index + 3:02d}",
                    nuisance_id=f"followup_{repeat_index}",
                    friction_slide=0.55 + 0.40 * float(sample[0]),
                    contact_timeconst_s=0.040 + 0.030 * float(sample[1]),
                    target_down_force_n=2.5 + float(sample[2]),
                    yaw_rad=math.radians(-1.5 + 3.0 * float(sample[3])),
                    nuisance_seed=10000 + sample_index - 1,
                )
            )
    return tasks


def _worker_init(
    output_dir: str,
    phase_bins: int,
    model_path: str | None = None,
    spec_path: str | None = None,
) -> None:
    global _WORKER_MODEL, _WORKER_DATA, _WORKER_SPEC, _WORKER_PHASE_BINS, _WORKER_OUT
    selected_spec = Path(spec_path) if spec_path is not None else SPEC_PATH
    selected_model = Path(model_path) if model_path is not None else MODEL_PATH
    _WORKER_SPEC = json.loads(selected_spec.read_text(encoding="utf-8"))
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(str(selected_model))
    _WORKER_MODEL.opt.timestep = SIM_TIMESTEP_S
    _WORKER_DATA = mujoco.MjData(_WORKER_MODEL)
    _WORKER_PHASE_BINS = int(phase_bins)
    _WORKER_OUT = Path(output_dir)


def _run_segment(
    rollout: MuJoCoAdmittanceRollout,
    nominal_tip: np.ndarray,
    duration_s: float,
    rpy_at_fraction,
    *,
    phase_at_fraction=None,
    record: bool = False,
) -> tuple[list[np.ndarray], bool, str]:
    steps = max(1, int(round(duration_s / float(rollout.model.opt.timestep))))
    rows: list[np.ndarray] = []
    invalid = False
    reason = ""
    for index in range(steps):
        fraction = (index + 1) / steps
        nominal_rpy = np.asarray(rpy_at_fraction(fraction), dtype=float)
        snapshot = rollout.step(nominal_tip, nominal_rpy)
        if record and (index + 1) % rollout.control_stride == 0:
            phase = float(phase_at_fraction(fraction))
            row = np.concatenate(
                (
                    np.array([fraction * duration_s, phase], dtype=float),
                    snapshot.measured_wrench,
                    snapshot.tip_position,
                    np.array([snapshot.roll, snapshot.pitch, snapshot.yaw], dtype=float),
                    snapshot.command.pose_offset,
                    nominal_rpy,
                    np.array([float(rollout.data.ncon)], dtype=float),
                )
            )
            rows.append(row)
        if snapshot.command.hard_stop or snapshot.state_invalid:
            invalid = True
            reason = "hard_stop" if snapshot.command.hard_stop else snapshot.failure_reason
            break
    return rows, invalid, reason


def _phase_resample(raw: np.ndarray, phase_bins: int) -> tuple[np.ndarray, np.ndarray]:
    phase_grid = np.linspace(0.0, 2.0 * math.pi, phase_bins, endpoint=False, dtype=float)
    if raw.shape[0] < 2:
        return phase_grid.astype(np.float32), np.full((len(PHASE_COLUMNS), phase_bins), np.nan, dtype=np.float32)
    phase = raw[:, 1]
    values = raw[:, 2:]
    order = np.argsort(phase)
    phase = phase[order]
    values = values[order]
    result = np.vstack([np.interp(phase_grid, phase, values[:, index]) for index in range(values.shape[1])])
    return phase_grid.astype(np.float32), result.astype(np.float32)


def _contiguous_true_runs(
    mask: np.ndarray,
    *,
    time_s: np.ndarray,
    phase_rad: np.ndarray,
    sample_dt_s: float,
) -> list[dict]:
    """Describe every contiguous true run without smoothing it away."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if len(values) == 0:
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    runs = []
    for start, stop in zip(starts, stops, strict=True):
        end = int(stop - 1)
        samples = int(stop - start)
        runs.append(
            {
                "start_sample": int(start),
                "end_sample_inclusive": end,
                "samples": samples,
                # Each recorded sample represents one controller interval.
                "duration_s": float(samples * sample_dt_s),
                "start_time_s": float(time_s[start]),
                "end_time_s": float(time_s[end] + sample_dt_s),
                "start_phase_rad": float(phase_rad[start]),
                "end_phase_rad": float(phase_rad[end]),
            }
        )
    return runs


def audit_contact_continuity(
    raw: np.ndarray,
    raw_columns: tuple[str, ...] | list[str],
    *,
    target_down_force_n: float,
    downward_progress_z_threshold_m: float | None,
    sample_dt_s: float | None = None,
    sustained_event_min_s: float = SUSTAINED_CONTACT_EVENT_MIN_S,
) -> dict:
    """Audit scan continuity using deployable signals and privileged sim truth.

    The force-norm branch is sensor-observable.  ``contact_count == 0`` is a
    sufficient no-contact condition available only offline in MuJoCo; a
    nonzero count is *not* treated as proof that the peg has load-bearing
    support.  The result quarantines training data and is neither an online
    phase-hold policy nor an insertion-success certificate.
    """

    values = np.asarray(raw, dtype=float)
    columns = tuple(map(str, raw_columns))
    if values.ndim != 2:
        raise ValueError("raw scan must be a 2-D array")
    if values.shape[1] != len(columns):
        raise ValueError("raw scan width does not match raw_columns")

    sample_count = int(values.shape[0])
    time_index = columns.index("time_s") if "time_s" in columns else None
    phase_index = columns.index("phase_rad") if "phase_rad" in columns else None
    if time_index is None:
        time_s = np.empty(sample_count, dtype=float)
    else:
        time_s = values[:, time_index].astype(float, copy=False)
    if phase_index is None:
        phase_rad = np.zeros(sample_count, dtype=float)
    else:
        phase_rad = values[:, phase_index].astype(float, copy=False)

    inferred_dt_s = float(sample_dt_s) if sample_dt_s is not None else math.nan
    if not np.isfinite(inferred_dt_s) or inferred_dt_s <= 0.0:
        if time_index is None:
            inferred_dt_s = CONTROLLER_DT_S
        else:
            finite_time = time_s[np.isfinite(time_s)]
            positive_steps = np.diff(finite_time)
            positive_steps = positive_steps[positive_steps > 0.0]
            inferred_dt_s = (
                float(np.median(positive_steps))
                if len(positive_steps)
                else CONTROLLER_DT_S
            )
    if time_index is None:
        time_s = np.arange(sample_count, dtype=float) * inferred_dt_s

    force_names = ("Fx_N", "Fy_N", "Fz_N")
    force_available = all(name in columns for name in force_names)
    low_force_threshold_n = max(
        LOW_FORCE_ABSOLUTE_FLOOR_N,
        LOW_FORCE_TARGET_FRACTION * abs(float(target_down_force_n)),
    )
    if force_available:
        force = values[:, [columns.index(name) for name in force_names]]
        force_norm_n = np.linalg.norm(force, axis=1)
        finite_force = np.all(np.isfinite(force), axis=1)
        low_force_mask = finite_force & (force_norm_n < low_force_threshold_n)
        force_min_n = (
            float(np.min(force_norm_n[finite_force]))
            if np.any(finite_force)
            else None
        )
    else:
        finite_force = np.ones(sample_count, dtype=bool)
        low_force_mask = np.zeros(sample_count, dtype=bool)
        force_min_n = None

    contact_count_available = "contact_count" in columns
    if contact_count_available:
        contact_count = values[:, columns.index("contact_count")]
        finite_contact_count = np.isfinite(contact_count)
        # data.ncon == 0 proves that no MuJoCo contact exists at this instant.
        contact_loss_mask = finite_contact_count & (contact_count <= 0.0)
    else:
        finite_contact_count = np.ones(sample_count, dtype=bool)
        contact_loss_mask = np.zeros(sample_count, dtype=bool)

    tip_z_available = (
        "tip_z_m" in columns
        and downward_progress_z_threshold_m is not None
    )
    if tip_z_available:
        tip_z_m = values[:, columns.index("tip_z_m")]
        finite_tip_z = np.isfinite(tip_z_m)
        downward_progress_mask = finite_tip_z & (
            tip_z_m <= float(downward_progress_z_threshold_m)
        )
        minimum_tip_z_m = (
            float(np.min(tip_z_m[finite_tip_z]))
            if np.any(finite_tip_z)
            else None
        )
    else:
        finite_tip_z = np.ones(sample_count, dtype=bool)
        downward_progress_mask = np.zeros(sample_count, dtype=bool)
        minimum_tip_z_m = None

    run_arguments = {
        "time_s": time_s,
        "phase_rad": phase_rad,
        "sample_dt_s": inferred_dt_s,
    }
    contact_loss_runs = _contiguous_true_runs(
        contact_loss_mask, **run_arguments
    )
    low_force_runs = _contiguous_true_runs(low_force_mask, **run_arguments)
    downward_progress_runs = _contiguous_true_runs(
        downward_progress_mask, **run_arguments
    )

    def sustained(runs: list[dict]) -> list[dict]:
        return [
            run
            for run in runs
            if float(run["duration_s"]) + 1.0e-12 >= sustained_event_min_s
        ]

    sustained_contact_loss_runs = sustained(contact_loss_runs)
    sustained_low_force_runs = sustained(low_force_runs)
    sustained_downward_progress_runs = sustained(downward_progress_runs)
    has_nonfinite_recorded_signal = bool(
        (force_available and not np.all(finite_force))
        or (contact_count_available and not np.all(finite_contact_count))
        or (tip_z_available and not np.all(finite_tip_z))
        or not np.all(np.isfinite(time_s))
        or not np.all(np.isfinite(phase_rad))
    )

    return {
        "schema_version": CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION,
        "scope": "offline_dataset_quarantine",
        "policy_boundary": (
            "uses recorded sensor-observable force plus privileged MuJoCo "
            "contact_count; not an online phase-hold policy and not an "
            "insertion-success certificate"
        ),
        "sample_count": sample_count,
        "sample_dt_s": inferred_dt_s,
        "sustained_event_min_s": float(sustained_event_min_s),
        "sensor_observable_force": {
            "available": force_available,
            "signal": "norm([Fx_N, Fy_N, Fz_N])",
            "low_force_threshold_n": float(low_force_threshold_n),
            "threshold_formula": (
                "max(0.25 N, 0.35 * abs(target_down_force_n))"
            ),
            "minimum_force_norm_n": force_min_n,
            "low_force_runs": low_force_runs,
            "sustained_low_force_runs": sustained_low_force_runs,
        },
        "mujoco_offline_ground_truth": {
            "available": contact_count_available,
            "signal": "data.ncon recorded as contact_count",
            "interpretation": (
                "contact_count == 0 is sufficient for no contact; "
                "contact_count > 0 does not certify load-bearing peg support"
            ),
            "contact_loss_runs": contact_loss_runs,
            "sustained_contact_loss_runs": sustained_contact_loss_runs,
        },
        "within_scan_downward_progress": {
            "available": tip_z_available,
            "signal": "tip_z_m",
            "downward_progress_z_threshold_m": (
                None
                if downward_progress_z_threshold_m is None
                else float(downward_progress_z_threshold_m)
            ),
            "threshold_semantics": (
                "full chamfer-depth crossing; stricter than the endpoint "
                "capture-routing heuristic so ordinary conical force "
                "unloading is not mislabeled as task progress"
            ),
            "minimum_tip_z_m": minimum_tip_z_m,
            "downward_progress_runs": downward_progress_runs,
            "sustained_downward_progress_runs": sustained_downward_progress_runs,
        },
        "has_sustained_contact_loss": bool(sustained_contact_loss_runs),
        "has_sustained_low_force": bool(sustained_low_force_runs),
        "has_sustained_downward_progress": bool(
            sustained_downward_progress_runs
        ),
        "has_nonfinite_recorded_signal": has_nonfinite_recorded_signal,
    }


def derive_direction_supervision_metadata(
    raw: np.ndarray,
    raw_columns: tuple[str, ...] | list[str],
    *,
    status: str,
    target_down_force_n: float,
    downward_progress_z_threshold_m: float | None,
    full_scan_complete: bool,
    early_capture: bool,
    capture_during_scan: bool,
    sample_dt_s: float | None = None,
    recorded_full_fingerprint_valid: bool | None = None,
    recorded_direction_supervision_valid: bool | None = None,
    revalidated_from_raw_on_load: bool = False,
) -> dict:
    """Derive one conservative, reason-coded direction-training decision."""

    continuity = audit_contact_continuity(
        raw,
        raw_columns,
        target_down_force_n=target_down_force_n,
        downward_progress_z_threshold_m=downward_progress_z_threshold_m,
        sample_dt_s=sample_dt_s,
    )
    capture_or_incomplete_reasons = []
    if status != "ok":
        capture_or_incomplete_reasons.append("simulation_failure")
    if not full_scan_complete:
        capture_or_incomplete_reasons.append("scan_incomplete")
    if early_capture:
        capture_or_incomplete_reasons.append("early_capture")
    if capture_during_scan:
        capture_or_incomplete_reasons.append("capture_during_scan_endpoint")
    if continuity["has_sustained_downward_progress"]:
        capture_or_incomplete_reasons.append(
            "within_scan_sustained_downward_progress"
        )

    quarantine_reasons = list(capture_or_incomplete_reasons)
    if continuity["has_sustained_contact_loss"]:
        quarantine_reasons.append(
            "sustained_contact_loss_offline_ground_truth"
        )
    diagnostic_flags = []
    if continuity["has_sustained_low_force"]:
        diagnostic_flags.append("sustained_low_force_sensor_observable")
        # Force unloading is itself an intended tactile cue.  Only use it as
        # a conservative fallback when the legacy raw trace lacks privileged
        # MuJoCo contact-count evidence.
        if not continuity["mujoco_offline_ground_truth"]["available"]:
            quarantine_reasons.append(
                "sustained_low_force_without_contact_ground_truth"
            )
    if continuity["has_nonfinite_recorded_signal"]:
        quarantine_reasons.append("nonfinite_recorded_signal")

    valid = not quarantine_reasons
    return {
        # Existing keys are retained; their value now includes the continuity
        # quarantine rather than only the endpoint capture heuristic.
        "full_fingerprint_valid": valid,
        "direction_supervision_valid": valid,
        "geometric_full_fingerprint_valid": bool(
            status == "ok"
            and full_scan_complete
            and not early_capture
            and not capture_during_scan
        ),
        "capture_or_incomplete_reasons": capture_or_incomplete_reasons,
        "direction_diagnostic_flags": diagnostic_flags,
        "direction_quarantine_reasons": quarantine_reasons,
        "contact_continuity_audit": continuity,
        "direction_supervision_audit": {
            "schema_version": CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION,
            "decision": "admit" if valid else "quarantine",
            "revalidated_from_raw_on_load": bool(
                revalidated_from_raw_on_load
            ),
            "recorded_values_before_revalidation": {
                "full_fingerprint_valid": recorded_full_fingerprint_valid,
                "direction_supervision_valid": (
                    recorded_direction_supervision_valid
                ),
            },
            "policy_boundary": (
                "offline direction-label hygiene only; online hardware must "
                "use sensor-rate event handling and independent terminal "
                "verification"
            ),
        },
    }


def capture_xy_shift_threshold_for_spec(spec: dict) -> tuple[float, str]:
    """Return a size-aware routing threshold, not a success certificate.

    Hardware-easy-v2 scenes expose their paired flat-normal guidance margin,
    so the routing heuristic follows 15% of that physical lead-in distance.
    Legacy/source scenes retain the historical M8 0.25-mm ratio, scaled by the
    circumradius of the hex bit for M10/M12 compatibility.
    """

    guidance = spec.get("paired_flat_normal_guidance_margin_m")
    if guidance is not None and float(guidance) > 0.0:
        return (
            0.15 * float(guidance),
            "0.15 * paired_flat_normal_guidance_margin_m",
        )
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    m8_bit_radius = 0.006 / math.sqrt(3.0)
    return (
        0.00025 * bit_radius / m8_bit_radius,
        "0.25 mm * (bit_radius / M8_bit_radius)",
    )


def _sample_task(task: PilotTask) -> dict:
    if _WORKER_MODEL is None or _WORKER_DATA is None or _WORKER_SPEC is None or _WORKER_OUT is None:
        raise RuntimeError("worker was not initialized")
    output_path = _WORKER_OUT / "trials" / f"{task.trial_id}.npz"
    if output_path.exists():
        return {"trial_id": task.trial_id, "status": "skipped", "path": str(output_path)}

    mujoco.mj_resetData(_WORKER_MODEL, _WORKER_DATA)
    _WORKER_MODEL.geom_friction[:, 0] = float(task.friction_slide)
    rollout = MuJoCoAdmittanceRollout(
        _WORKER_MODEL,
        _WORKER_DATA,
        float(_WORKER_SPEC["bit_length_m"]),
        float(_WORKER_SPEC["bit_mass_kg"]),
        wrench_reference_to_tip_m=(
            None
            if _WORKER_SPEC.get("wrench_reference_to_tip_m") is None
            else float(_WORKER_SPEC["wrench_reference_to_tip_m"])
        ),
        controller_dt_s=CONTROLLER_DT_S,
        target_down_force_n=float(task.target_down_force_n),
        contact_timeconst_s=float(task.contact_timeconst_s),
        wrench_observation=(
            None
            if task.sensor_noise_seed is None
            else WrenchObservationConfig(
                seed=int(task.sensor_noise_seed),
                delay_s=float(task.sensor_delay_s),
            )
        ),
    )
    nominal_tip = np.array([task.x_m, task.y_m, task.nominal_tip_z_m], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip, np.array([0.0, 0.0, task.yaw_rad], dtype=float))
    start = time.perf_counter()

    snapshot = rollout.advance(nominal_tip, np.zeros(3), SETTLE_DURATION_S)
    invalid = snapshot.command.hard_stop or snapshot.state_invalid
    failure_reason = "hard_stop" if snapshot.command.hard_stop else snapshot.failure_reason
    baseline_tip = snapshot.tip_position.copy()
    pre_scan_tip = baseline_tip.copy()

    if not invalid:
        _, invalid, failure_reason = _run_segment(
            rollout,
            nominal_tip,
            RAMP_DURATION_S,
            lambda f: np.array([0.0, ALPHA_RAD * _smoothstep(f), task.yaw_rad]),
        )
        pre_scan_tip, _ = rollout.actual_tip_pose()

    raw_rows: list[np.ndarray] = []
    if not invalid:
        raw_rows, invalid, failure_reason = _run_segment(
            rollout,
            nominal_tip,
            SCAN_DURATION_S,
            lambda f: np.array(
                [
                    ALPHA_RAD * math.sin(2.0 * math.pi * f),
                    ALPHA_RAD * math.cos(2.0 * math.pi * f),
                    task.yaw_rad,
                ]
            ),
            phase_at_fraction=lambda f: 2.0 * math.pi * f,
            record=True,
        )
    post_scan_tip, _ = rollout.actual_tip_pose()

    if not invalid:
        _, invalid, failure_reason = _run_segment(
            rollout,
            nominal_tip,
            RETURN_DURATION_S,
            lambda f: np.array([0.0, ALPHA_RAD * (1.0 - _smoothstep(f)), task.yaw_rad]),
        )

    raw = np.asarray(raw_rows, dtype=np.float32)
    if raw.size == 0:
        raw = np.empty((0, len(RAW_COLUMNS)), dtype=np.float32)
    phase_grid, phase_sequence = _phase_resample(raw, _WORKER_PHASE_BINS)
    final_tip, final_rpy = rollout.actual_tip_pose()
    force = raw[:, 2:5] if len(raw) else np.empty((0, 3))
    torque = raw[:, 5:8] if len(raw) else np.empty((0, 3))
    elapsed = time.perf_counter() - start
    chamfer_depth = float(_WORKER_SPEC.get("chamfer_depth_m", 0.0))
    capture_z_threshold = -0.25 * chamfer_depth
    downward_progress_z_threshold = (
        -WITHIN_SCAN_PROGRESS_CHAMFER_FRACTION * chamfer_depth
        if chamfer_depth > 0.0
        else None
    )
    capture_xy_shift_threshold, capture_xy_shift_formula = (
        capture_xy_shift_threshold_for_spec(_WORKER_SPEC)
    )
    baseline_xy_shift = float(np.linalg.norm(baseline_tip[:2] - nominal_tip[:2]))
    pre_scan_xy_shift = float(np.linalg.norm(pre_scan_tip[:2] - nominal_tip[:2]))
    post_scan_xy_shift = float(np.linalg.norm(post_scan_tip[:2] - nominal_tip[:2]))
    early_capture = bool(
        pre_scan_tip[2] <= capture_z_threshold
        or pre_scan_xy_shift >= capture_xy_shift_threshold
    )
    capture_during_scan = bool(
        post_scan_tip[2] <= capture_z_threshold
        or post_scan_xy_shift >= capture_xy_shift_threshold
    )
    full_scan_complete = bool(
        not invalid
        and len(raw) >= int(0.95 * _WORKER_PHASE_BINS)
        and (float(raw[-1, 1] / (2.0 * math.pi)) if len(raw) else 0.0) >= 0.98
    )
    direction_supervision = derive_direction_supervision_metadata(
        raw,
        RAW_COLUMNS,
        status="failed" if invalid else "ok",
        target_down_force_n=float(task.target_down_force_n),
        downward_progress_z_threshold_m=downward_progress_z_threshold,
        full_scan_complete=full_scan_complete,
        early_capture=early_capture,
        capture_during_scan=capture_during_scan,
        sample_dt_s=CONTROLLER_DT_S,
    )
    geometry_parameters = {
        key: _WORKER_SPEC.get(key)
        for key in (
            "key_af_m",
            "clearance_m",
            "chamfer_width_m",
            "chamfer_depth_m",
            "socket_depth_m",
            "screw_head_diameter_m",
            "bit_tip_chamfer_m",
            "bit_tip_scale",
            "bounded_socket_shell",
            "paired_flat_normal_guidance_margin_m",
        )
    }
    geometry_parameters.update(
        {
            "socket_wall_af_m": _WORKER_SPEC.get("derived", {}).get(
                "socket_wall_af_m"
            ),
            "socket_mouth_af_m": _WORKER_SPEC.get("derived", {}).get(
                "socket_top_af_m"
            ),
        }
    )
    metadata = {
        **asdict(task),
        "status": "failed" if invalid else "ok",
        "failure_reason": failure_reason,
        "dynamic_physics": True,
        "admittance_controlled": True,
        "alpha_deg": 15.0,
        "target_down_force_n": float(task.target_down_force_n),
        "simulation_timestep_s": SIM_TIMESTEP_S,
        "controller_dt_s": CONTROLLER_DT_S,
        "scan_duration_s": SCAN_DURATION_S,
        "phase_bins": _WORKER_PHASE_BINS,
        "raw_samples": int(len(raw)),
        "scan_fraction": float(raw[-1, 1] / (2.0 * math.pi)) if len(raw) else 0.0,
        "max_force_n": float(np.max(np.linalg.norm(force, axis=1))) if len(force) else 0.0,
        "max_torque_nm": float(np.max(np.linalg.norm(torque, axis=1))) if len(torque) else 0.0,
        "baseline_tip_xyz_m": baseline_tip.tolist(),
        "pre_scan_tip_xyz_m": pre_scan_tip.tolist(),
        "post_scan_tip_xyz_m": post_scan_tip.tolist(),
        "final_tip_xyz_m": final_tip.tolist(),
        "final_rpy_rad": list(map(float, final_rpy)),
        "final_xy_residual_m": float(np.linalg.norm(final_tip[:2] - nominal_tip[:2])),
        "baseline_xy_shift_m": baseline_xy_shift,
        "pre_scan_xy_shift_m": pre_scan_xy_shift,
        "post_scan_xy_shift_m": post_scan_xy_shift,
        "early_capture": early_capture,
        "capture_during_scan": capture_during_scan,
        "full_scan_complete": full_scan_complete,
        **direction_supervision,
        "capture_classifier": {
            "z_threshold_m": capture_z_threshold,
            "xy_shift_threshold_m": capture_xy_shift_threshold,
            "xy_shift_threshold_formula": capture_xy_shift_formula,
            "semantics": (
                "simulation routing heuristic; captured traces are task progress but are excluded from direction supervision"
            ),
        },
        "worker_elapsed_s": elapsed,
        "created_utc": _utc_now(),
        "geometry_name": str(_WORKER_SPEC.get("name", "unknown")),
        "geometry_profile": str(
            _WORKER_SPEC.get("geometry_profile", "frozen_source")
        ),
        "geometry_parameters": geometry_parameters,
        "claim_boundary": (
            f"{str(_WORKER_SPEC.get('name', 'unknown')).upper()} conical sample; "
            "not real-robot or center-recovery evidence"
        ),
        "wrench_observation": (
            "ideal"
            if task.sensor_noise_seed is None
            else WrenchObservationConfig(
                seed=int(task.sensor_noise_seed),
                delay_s=float(task.sensor_delay_s),
            ).to_dict()
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            raw_scan=raw,
            raw_columns=np.asarray(RAW_COLUMNS),
            phase_grid_rad=phase_grid,
            phase_sequence=phase_sequence,
            phase_columns=np.asarray(PHASE_COLUMNS),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    return {"trial_id": task.trial_id, "status": metadata["status"], "path": str(output_path), **metadata}


def _load_trial(path: Path) -> tuple[dict, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        sequence = np.asarray(data["phase_sequence"], dtype=float)
        columns = tuple(map(str, data["phase_columns"].tolist()))
        raw = (
            np.asarray(data["raw_scan"], dtype=float)
            if "raw_scan" in data.files
            else np.empty((0, 0), dtype=float)
        )
        raw_columns = (
            tuple(map(str, data["raw_columns"].tolist()))
            if "raw_columns" in data.files
            else ()
        )

    recorded_values = metadata.get("direction_supervision_audit", {}).get(
        "recorded_values_before_revalidation", {}
    )
    recorded_full = recorded_values.get("full_fingerprint_valid")
    if recorded_full is None:
        recorded_full = metadata.get("full_fingerprint_valid")
    recorded_direction = recorded_values.get("direction_supervision_valid")
    if recorded_direction is None:
        recorded_direction = metadata.get("direction_supervision_valid")
    capture_classifier = metadata.get("capture_classifier", {})
    capture_z_threshold = capture_classifier.get("z_threshold_m")
    chamfer_depth = metadata.get("geometry_parameters", {}).get(
        "chamfer_depth_m"
    )
    if capture_z_threshold is None and chamfer_depth is not None:
        capture_z_threshold = -0.25 * float(chamfer_depth)
    if chamfer_depth is not None:
        downward_progress_z_threshold = (
            -WITHIN_SCAN_PROGRESS_CHAMFER_FRACTION * float(chamfer_depth)
        )
    elif capture_z_threshold is not None:
        # Current capture metadata stores -0.25 * chamfer depth.
        downward_progress_z_threshold = 4.0 * float(capture_z_threshold)
    else:
        downward_progress_z_threshold = None

    # Legacy NPZs predate the contact-continuity keys.  Recompute the decision
    # from immutable raw samples in memory; do not rewrite the source artifact.
    if raw_columns and raw.shape[1] == len(raw_columns):
        if "full_scan_complete" in metadata:
            full_scan_complete = bool(metadata["full_scan_complete"])
        else:
            phase_bins = int(metadata.get("phase_bins", max(len(raw), 1)))
            scan_fraction = float(
                metadata.get(
                    "scan_fraction",
                    (
                        raw[-1, raw_columns.index("phase_rad")]
                        / (2.0 * math.pi)
                        if len(raw) and "phase_rad" in raw_columns
                        else 0.0
                    ),
                )
            )
            full_scan_complete = bool(
                metadata.get("status", "ok") == "ok"
                and len(raw) >= int(0.95 * phase_bins)
                and scan_fraction >= 0.98
            )
        derived = derive_direction_supervision_metadata(
            raw,
            raw_columns,
            status=str(metadata.get("status", "ok")),
            target_down_force_n=float(
                metadata.get("target_down_force_n", TARGET_DOWN_FORCE_N)
            ),
            downward_progress_z_threshold_m=(
                None
                if downward_progress_z_threshold is None
                else float(downward_progress_z_threshold)
            ),
            full_scan_complete=full_scan_complete,
            early_capture=bool(metadata.get("early_capture", False)),
            capture_during_scan=bool(
                metadata.get("capture_during_scan", False)
            ),
            sample_dt_s=float(
                metadata.get("controller_dt_s", CONTROLLER_DT_S)
            ),
            recorded_full_fingerprint_valid=(
                None if recorded_full is None else bool(recorded_full)
            ),
            recorded_direction_supervision_valid=(
                None
                if recorded_direction is None
                else bool(recorded_direction)
            ),
            revalidated_from_raw_on_load=True,
        )
        metadata.update(derived)
    else:
        # Without raw evidence, a stale positive label is not admissible for
        # new training.  The phase sequence remains loadable for diagnostics.
        unavailable_reasons = ["raw_scan_unavailable_for_revalidation"]
        if metadata.get("status", "ok") != "ok":
            unavailable_reasons.insert(0, "simulation_failure")
        if not bool(metadata.get("full_scan_complete", False)):
            unavailable_reasons.insert(0, "scan_incomplete")
        if bool(metadata.get("early_capture", False)):
            unavailable_reasons.insert(0, "early_capture")
        if bool(metadata.get("capture_during_scan", False)):
            unavailable_reasons.insert(0, "capture_during_scan_endpoint")
        metadata.update(
            {
                "full_fingerprint_valid": False,
                "direction_supervision_valid": False,
                "capture_or_incomplete_reasons": unavailable_reasons,
                "direction_quarantine_reasons": unavailable_reasons,
                "direction_supervision_audit": {
                    "schema_version": CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION,
                    "decision": "quarantine",
                    "revalidated_from_raw_on_load": True,
                    "recorded_values_before_revalidation": {
                        "full_fingerprint_valid": recorded_full,
                        "direction_supervision_valid": recorded_direction,
                    },
                    "policy_boundary": (
                        "offline direction-label hygiene only; raw evidence "
                        "was unavailable and the source NPZ was not rewritten"
                    ),
                },
            }
        )
    return metadata, sequence, columns


def audit_dataset_contact_quarantine(
    dataset_root: Path,
    *,
    output_path: Path | None = None,
) -> dict:
    """Summarize reason-coded quarantine decisions without rewriting trials."""

    root = Path(dataset_root)
    paths = sorted((root / "trials").glob("*.npz"))
    reason_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    records = []
    admitted = 0
    for path in paths:
        metadata, _, _ = _load_trial(path)
        reasons = list(metadata.get("direction_quarantine_reasons", []))
        diagnostics = list(metadata.get("direction_diagnostic_flags", []))
        valid = bool(metadata.get("direction_supervision_valid", False))
        admitted += int(valid)
        for reason in reasons:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        for diagnostic in diagnostics:
            diagnostic_counts[str(diagnostic)] = (
                diagnostic_counts.get(str(diagnostic), 0) + 1
            )
        continuity = metadata.get("contact_continuity_audit", {})
        records.append(
            {
                "trial_id": str(metadata.get("trial_id", path.stem)),
                "path": str(path),
                "direction_supervision_valid": valid,
                "direction_quarantine_reasons": reasons,
                "direction_diagnostic_flags": diagnostics,
                "has_sustained_contact_loss": bool(
                    continuity.get("has_sustained_contact_loss", False)
                ),
                "has_sustained_low_force": bool(
                    continuity.get("has_sustained_low_force", False)
                ),
                "has_sustained_downward_progress": bool(
                    continuity.get(
                        "has_sustained_downward_progress", False
                    )
                ),
            }
        )
    report = {
        "schema_version": CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION,
        "dataset_root": str(root),
        "scope": "offline_read_only_direction_supervision_audit",
        "source_trials_rewritten": False,
        "trial_count": len(paths),
        "admitted_count": admitted,
        "quarantined_count": len(paths) - admitted,
        "reason_counts": dict(sorted(reason_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "records": records,
        "created_utc": _utc_now(),
    }
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def _aligned_to_reference(sequence: np.ndarray, columns: tuple[str, ...], rotation_index: int) -> np.ndarray:
    bins = sequence.shape[1]
    shift = int(round(rotation_index * bins / 6.0))
    # y_(R psi delta)(theta) = T_psi y_delta(theta - psi).
    # Evaluate the rotated trace at theta + psi, then rotate vectors back.
    aligned = np.roll(sequence, shift, axis=1).copy()
    angle = -rotation_index * math.pi / 3.0
    for x_name, y_name in (("Fx_N", "Fy_N"), ("Tx_Nm", "Ty_Nm"), ("tip_x_m", "tip_y_m")):
        ix, iy = columns.index(x_name), columns.index(y_name)
        aligned[[ix, iy]] = _rotate_xy(aligned[[ix, iy]], angle)
    return aligned


def _smoothed_standardized(values: np.ndarray, window: int = 7) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if window <= 1:
        smoothed = vector.copy()
    else:
        half = window // 2
        extended = np.concatenate((vector[-half:], vector, vector[:half]))
        smoothed = np.convolve(extended, np.ones(window, dtype=float) / window, mode="valid")
    return (smoothed - np.mean(smoothed)) / max(float(np.std(smoothed)), 1.0e-9)


def audit_symmetry(output_dir: Path) -> dict:
    groups: dict[str, list[tuple[dict, np.ndarray, tuple[str, ...]]]] = {}
    for path in sorted((output_dir / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            continue
        groups.setdefault(metadata["base_pose_id"], []).append((metadata, sequence, columns))

    selected = ("Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm", "tip_z_m")
    primary_shape = ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")
    records: list[dict] = []
    for base_pose_id, items in sorted(groups.items()):
        by_rotation = {int(item[0]["rotation_index"]): item for item in items}
        if 0 not in by_rotation:
            continue
        ref_meta, ref_sequence, columns = by_rotation[0]
        ref = _aligned_to_reference(ref_sequence, columns, 0)
        indices = [columns.index(name) for name in selected]
        scale = np.sqrt(np.mean(ref[indices] ** 2, axis=1))
        scale = np.maximum(scale, np.array([0.10, 0.10, 0.10, 1e-4, 1e-4, 1e-4, 1e-5]))
        for rotation_index in range(1, 6):
            if rotation_index not in by_rotation:
                continue
            metadata, sequence, _ = by_rotation[rotation_index]
            aligned = _aligned_to_reference(sequence, columns, rotation_index)
            channel_nrmse = np.sqrt(np.mean((aligned[indices] - ref[indices]) ** 2, axis=1)) / scale
            shape_rmse: dict[str, float] = {}
            shape_correlation: dict[str, float] = {}
            for name in primary_shape:
                channel_index = columns.index(name)
                reference_shape = _smoothed_standardized(ref[channel_index])
                aligned_shape = _smoothed_standardized(aligned[channel_index])
                shape_rmse[name] = float(np.sqrt(np.mean((reference_shape - aligned_shape) ** 2)))
                shape_correlation[name] = float(np.corrcoef(reference_shape, aligned_shape)[0, 1])
            records.append(
                {
                    "base_pose_id": base_pose_id,
                    "search_state": ref_meta["search_state"],
                    "rotation_index": rotation_index,
                    "rotation_deg": metadata["rotation_deg"],
                    "channel_nrmse": dict(zip(selected, map(float, channel_nrmse))),
                    "mean_nrmse": float(np.mean(channel_nrmse)),
                    "max_nrmse": float(np.max(channel_nrmse)),
                    "primary_shape_rmse": shape_rmse,
                    "primary_shape_correlation": shape_correlation,
                    "mean_primary_shape_rmse": float(np.mean(list(shape_rmse.values()))),
                    "mean_primary_shape_correlation": float(np.mean(list(shape_correlation.values()))),
                }
            )
    means = np.asarray([record["mean_nrmse"] for record in records], dtype=float)
    shape_rmses = np.asarray([record["mean_primary_shape_rmse"] for record in records], dtype=float)
    shape_correlations = np.asarray(
        [record["mean_primary_shape_correlation"] for record in records], dtype=float
    )
    state_summary: dict[str, dict[str, float | int]] = {}
    for state in sorted({record["search_state"] for record in records}):
        state_records = [record for record in records if record["search_state"] == state]
        state_shape = np.asarray(
            [record["mean_primary_shape_correlation"] for record in state_records], dtype=float
        )
        state_summary[state] = {
            "comparisons": len(state_records),
            "mean_primary_shape_correlation": float(np.mean(state_shape)),
            "p05_primary_shape_correlation": float(np.quantile(state_shape, 0.05)),
        }
    result = {
        "stage": "P0a_C6_symmetry_audit",
        "complete_base_groups": sum(len(items) == 6 for items in groups.values()),
        "available_base_groups": len(groups),
        "comparisons": len(records),
        "mean_nrmse": float(np.mean(means)) if len(means) else None,
        "median_nrmse": float(np.median(means)) if len(means) else None,
        "p95_nrmse": float(np.quantile(means, 0.95)) if len(means) else None,
        "mean_primary_shape_rmse": float(np.mean(shape_rmses)) if len(shape_rmses) else None,
        "p95_primary_shape_rmse": float(np.quantile(shape_rmses, 0.95)) if len(shape_rmses) else None,
        "mean_primary_shape_correlation": (
            float(np.mean(shape_correlations)) if len(shape_correlations) else None
        ),
        "p05_primary_shape_correlation": (
            float(np.quantile(shape_correlations, 0.05)) if len(shape_correlations) else None
        ),
        "state_summary": state_summary,
        "records": records,
        "decision_boundary": (
            "Compare symmetry residual with repeated-nuisance residual before enabling C6 data augmentation; "
            "P0a alone does not prove hardware symmetry."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output_dir / "symmetry_audit.json", result)
    summary_lines = [
        "# M8 P0a C6 Symmetry Audit",
        "",
        f"Updated: {result['updated_utc']}",
        "",
        "## Result",
        "",
        f"- complete base groups: {result['complete_base_groups']} / {result['available_base_groups']}",
        f"- rotated comparisons: {result['comparisons']}",
        f"- raw selected-channel mean NRMSE: {result['mean_nrmse']:.4f}",
        f"- raw selected-channel p95 NRMSE: {result['p95_nrmse']:.4f}",
        f"- primary normalized-shape mean correlation: {result['mean_primary_shape_correlation']:.4f}",
        f"- primary normalized-shape p05 correlation: {result['p05_primary_shape_correlation']:.4f}",
        "",
        "Primary shape channels are `Fz`, `Tx`, `Ty`, and `tip_z`, circularly smoothed and standardized per trial.",
        "",
        "## State Breakdown",
        "",
        "| state | comparisons | mean shape correlation | p05 shape correlation |",
        "|---|---:|---:|---:|",
    ]
    for state, values in state_summary.items():
        summary_lines.append(
            f"| {state} | {values['comparisons']} | "
            f"{values['mean_primary_shape_correlation']:.4f} | "
            f"{values['p05_primary_shape_correlation']:.4f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The nominal MuJoCo data supports provisional C6 augmentation for the normalized direction/shape branch. Raw lateral wrench amplitudes are not sufficiently invariant to use rotated copies as independent evidence for radial distance or hardware transfer.",
            "",
            "P0a has no nuisance repeats. Compare these residuals with the P0b nuisance baseline before enabling C6 augmentation in the final training pipeline.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return result


def audit_sector_anchors(output_dir: Path) -> dict:
    groups: dict[str, list[tuple[dict, np.ndarray, tuple[str, ...]]]] = {}
    for path in sorted((output_dir / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            continue
        groups.setdefault(metadata["symmetry_orbit_id"], []).append(
            (metadata, sequence, columns)
        )

    selected = ("Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm", "tip_z_m")
    primary = ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")
    records = []
    for orbit_id, items in sorted(groups.items()):
        by_sector = {int(item[0]["rotation_index"]): item for item in items}
        if 0 not in by_sector:
            continue
        reference_meta, reference_sequence, columns = by_sector[0]
        reference = _aligned_to_reference(reference_sequence, columns, 0)
        indices = [columns.index(name) for name in selected]
        scale = np.sqrt(np.mean(reference[indices] ** 2, axis=1))
        scale = np.maximum(
            scale, np.array([0.10, 0.10, 0.10, 1e-4, 1e-4, 1e-4, 1e-5])
        )
        for sector_index in range(1, 6):
            if sector_index not in by_sector:
                continue
            metadata, sequence, _ = by_sector[sector_index]
            aligned = _aligned_to_reference(sequence, columns, sector_index)
            nrmse = np.sqrt(np.mean((aligned[indices] - reference[indices]) ** 2, axis=1)) / scale
            correlations = []
            for name in primary:
                channel_index = columns.index(name)
                correlations.append(
                    float(
                        np.corrcoef(
                            _smoothed_standardized(reference[channel_index]),
                            _smoothed_standardized(aligned[channel_index]),
                        )[0, 1]
                    )
                )
            records.append(
                {
                    "symmetry_orbit_id": orbit_id,
                    "search_state": reference_meta["search_state"],
                    "radial_level": reference_meta["radial_level"],
                    "local_angle_deg": reference_meta["base_angle_deg"],
                    "sector_index": sector_index,
                    "mean_nrmse": float(np.mean(nrmse)),
                    "mean_primary_shape_correlation": float(np.mean(correlations)),
                }
            )

    nrmse = np.asarray([record["mean_nrmse"] for record in records], dtype=float)
    correlations = np.asarray(
        [record["mean_primary_shape_correlation"] for record in records], dtype=float
    )
    state_summary = {}
    for state in sorted({record["search_state"] for record in records}):
        state_records = [record for record in records if record["search_state"] == state]
        values = np.asarray(
            [record["mean_primary_shape_correlation"] for record in state_records]
        )
        state_summary[state] = {
            "comparisons": len(state_records),
            "mean_primary_shape_correlation": float(np.mean(values)),
            "p05_primary_shape_correlation": float(np.quantile(values, 0.05)),
        }
    result = {
        "stage": "P0d_paired_sector_anchor_audit",
        "successful_trials": sum(len(items) for items in groups.values()),
        "complete_orbits": sum(len(items) == 6 for items in groups.values()),
        "available_orbits": len(groups),
        "comparisons": len(records),
        "mean_nrmse": float(np.mean(nrmse)) if len(nrmse) else None,
        "p95_nrmse": float(np.quantile(nrmse, 0.95)) if len(nrmse) else None,
        "mean_primary_shape_correlation": (
            float(np.mean(correlations)) if len(correlations) else None
        ),
        "p05_primary_shape_correlation": (
            float(np.quantile(correlations, 0.05)) if len(correlations) else None
        ),
        "state_summary": state_summary,
        "records": records,
        "claim_boundary": (
            "Paired bounded-nuisance MuJoCo sector residual audit. It tests whether C6 copies are a safe "
            "training shortcut; it is not a trained posterior, hardware symmetry proof, or control result."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output_dir / "sector_anchor_audit.json", result)
    lines = [
        "# M8 P0d Paired Sector Anchor Audit",
        "",
        f"Updated: {result['updated_utc']}",
        "",
        f"- complete paired orbits: {result['complete_orbits']} / {result['available_orbits']}",
        f"- rotated comparisons: {result['comparisons']}",
        f"- raw selected-channel mean / p95 NRMSE: {result['mean_nrmse']:.4f} / {result['p95_nrmse']:.4f}",
        f"- normalized primary-shape mean / p05 correlation: {result['mean_primary_shape_correlation']:.4f} / {result['p05_primary_shape_correlation']:.4f}",
        "",
        "| state | comparisons | mean shape correlation | p05 shape correlation |",
        "|---|---:|---:|---:|",
    ]
    for state, values in state_summary.items():
        lines.append(
            f"| {state} | {values['comparisons']} | "
            f"{values['mean_primary_shape_correlation']:.4f} | "
            f"{values['p05_primary_shape_correlation']:.4f} |"
        )
    lines.extend(["", "## Boundary", "", result["claim_boundary"], ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _fingerprint_features(
    metadata: dict, sequence: np.ndarray, columns: tuple[str, ...], bit_radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    primary = ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")
    shape_parts: list[np.ndarray] = []
    for name in primary:
        signal = _smoothed_standardized(sequence[columns.index(name)])
        spectrum = np.fft.rfft(signal) / len(signal)
        shape_parts.append(np.concatenate((spectrum[1:7].real, spectrum[1:7].imag)))

    force_scale = max(float(metadata["target_down_force_n"]), 1.0e-6)
    torque_scale = force_scale * bit_radius_m
    dimensionless = {
        "Fz_N": sequence[columns.index("Fz_N")] / force_scale,
        "Tx_Nm": sequence[columns.index("Tx_Nm")] / torque_scale,
        "Ty_Nm": sequence[columns.index("Ty_Nm")] / torque_scale,
        "tip_z_m": sequence[columns.index("tip_z_m")] / bit_radius_m,
    }
    absolute_parts = []
    for name in primary:
        values = dimensionless[name]
        absolute_parts.extend((np.mean(values), np.std(values), np.min(values), np.max(values)))
    return np.concatenate(shape_parts), np.asarray(absolute_parts, dtype=float)


def _angular_error_deg(first: float, second: float) -> float:
    difference = (first - second + 180.0) % 360.0 - 180.0
    return abs(difference)


def _analyze_feature_family(records: list[dict], feature_key: str, bit_radius_m: float) -> dict:
    features = np.vstack([record[feature_key] for record in records])
    group_ids = sorted({record["metadata"]["base_pose_id"] for record in records})
    grouped = {
        group_id: [record for record in records if record["metadata"]["base_pose_id"] == group_id]
        for group_id in group_ids
    }
    global_scale = np.std(features, axis=0)
    residuals = []
    for group_records in grouped.values():
        group_features = np.vstack([record[feature_key] for record in group_records])
        residuals.append(group_features - np.mean(group_features, axis=0))
    residual_matrix = np.vstack(residuals)
    nuisance_scale = np.sqrt(np.mean(residual_matrix**2, axis=0))
    nuisance_scale = np.maximum(nuisance_scale, 0.05 * np.maximum(global_scale, 1.0e-6))

    centroids = {
        group_id: np.mean(np.vstack([record[feature_key] for record in group_records]), axis=0)
        for group_id, group_records in grouped.items()
    }
    cv_results = []
    for record in records:
        own_group = record["metadata"]["base_pose_id"]
        own_others = [item for item in grouped[own_group] if item is not record]
        candidate_centroids = dict(centroids)
        candidate_centroids[own_group] = np.mean(
            np.vstack([item[feature_key] for item in own_others]), axis=0
        )
        distances = {
            group_id: float(np.linalg.norm((record[feature_key] - centroid) / nuisance_scale))
            for group_id, centroid in candidate_centroids.items()
        }
        predicted_group = min(distances, key=distances.get)
        predicted = grouped[predicted_group][0]["metadata"]
        true = record["metadata"]
        cv_results.append(
            {
                "true_group": own_group,
                "predicted_group": predicted_group,
                "state_correct": predicted["search_state"] == true["search_state"],
                "angle_error_deg": _angular_error_deg(
                    float(predicted["base_angle_deg"]), float(true["base_angle_deg"])
                ),
                "radial_error_m": abs(float(predicted["radius_m"]) - float(true["radius_m"])),
            }
        )

    within_distances = []
    for group_id, group_records in grouped.items():
        centroid = centroids[group_id]
        within_distances.extend(
            float(np.linalg.norm((record[feature_key] - centroid) / nuisance_scale))
            for record in group_records
        )
    collision_threshold = float(np.quantile(within_distances, 0.95))
    collision_records = []
    for group_id, centroid in centroids.items():
        metadata = grouped[group_id][0]["metadata"]
        alternatives = {
            other_id: float(np.linalg.norm((centroid - other_centroid) / nuisance_scale))
            for other_id, other_centroid in centroids.items()
            if other_id != group_id
        }
        if not alternatives:
            continue
        nearest_id = min(alternatives, key=alternatives.get)
        nearest_metadata = grouped[nearest_id][0]["metadata"]
        angle_error = _angular_error_deg(
            float(metadata["base_angle_deg"]), float(nearest_metadata["base_angle_deg"])
        )
        radial_error = abs(float(metadata["radius_m"]) - float(nearest_metadata["radius_m"]))
        action_conflict = (
            metadata["search_state"] != nearest_metadata["search_state"]
            or angle_error > 7.5
            or radial_error > 0.0004
        )
        collision_records.append(
            {
                "base_pose_id": group_id,
                "nearest_base_pose_id": nearest_id,
                "feature_distance": alternatives[nearest_id],
                "action_conflict": action_conflict,
                "is_collision": action_conflict and alternatives[nearest_id] <= collision_threshold,
                "angle_error_deg": angle_error,
                "radial_error_m": radial_error,
            }
        )

    lookup = {
        (
            record["metadata"]["search_state"],
            int(record["metadata"]["radial_level"]),
            round(float(record["metadata"]["base_angle_deg"]), 6),
        ): centroids[record["metadata"]["base_pose_id"]]
        for record in records
    }
    radii = {
        (record["metadata"]["search_state"], int(record["metadata"]["radial_level"])): float(
            record["metadata"]["radius_m"]
        )
        for record in records
    }
    angles = sorted({round(float(record["metadata"]["base_angle_deg"]), 6) for record in records})
    fisher_records = []
    for state in sorted({record["metadata"]["search_state"] for record in records}):
        for radial_level in range(1, 4):
            for angle_index in range(1, len(angles) - 1):
                angle_deg = angles[angle_index]
                angle_minus = angles[angle_index - 1]
                angle_plus = angles[angle_index + 1]
                needed = (
                    (state, radial_level - 1, angle_deg),
                    (state, radial_level + 1, angle_deg),
                    (state, radial_level, angle_minus),
                    (state, radial_level, angle_plus),
                )
                radius_keys = (
                    (state, radial_level - 1),
                    (state, radial_level),
                    (state, radial_level + 1),
                )
                if any(key not in lookup for key in needed) or any(key not in radii for key in radius_keys):
                    continue
                r_minus = radii[(state, radial_level - 1)]
                r_value = radii[(state, radial_level)]
                r_plus = radii[(state, radial_level + 1)]
                f_r_minus = lookup[(state, radial_level - 1, angle_deg)]
                f_r_plus = lookup[(state, radial_level + 1, angle_deg)]
                f_a_minus = lookup[(state, radial_level, angle_minus)]
                f_a_plus = lookup[(state, radial_level, angle_plus)]
                derivative_r = (f_r_plus - f_r_minus) / (r_plus - r_minus)
                derivative_phi = (f_a_plus - f_a_minus) / math.radians(angle_plus - angle_minus)
                jacobian = np.column_stack(
                    (
                        bit_radius_m * derivative_r,
                        (bit_radius_m / r_value) * derivative_phi,
                    )
                )
                whitened = jacobian / nuisance_scale[:, None]
                singular_values = np.linalg.svd(whitened, compute_uv=False)
                fisher_records.append(
                    {
                        "search_state": state,
                        "radial_level": radial_level,
                        "angle_deg": angle_deg,
                        "sigma_max": float(singular_values[0]),
                        "sigma_min": float(singular_values[-1]),
                        "condition_number": float(singular_values[0] / max(singular_values[-1], 1.0e-12)),
                    }
                )

    angle_errors = np.asarray([record["angle_error_deg"] for record in cv_results], dtype=float)
    radial_errors = np.asarray([record["radial_error_m"] for record in cv_results], dtype=float)
    sigma_min = np.asarray([record["sigma_min"] for record in fisher_records], dtype=float)
    return {
        "feature_family": feature_key,
        "dimension": int(features.shape[1]),
        "base_pose_groups": len(grouped),
        "nuisance_repeats_per_group": sorted({len(value) for value in grouped.values()}),
        "cv_state_accuracy": float(np.mean([record["state_correct"] for record in cv_results])),
        "cv_angle_error_median_deg": float(np.median(angle_errors)),
        "cv_angle_error_p90_deg": float(np.quantile(angle_errors, 0.90)),
        "cv_radial_mae_mm": float(1000.0 * np.mean(radial_errors)),
        "within_nuisance_p95_distance": collision_threshold,
        "action_collision_count": sum(record["is_collision"] for record in collision_records),
        "action_collision_rate": (
            float(np.mean([record["is_collision"] for record in collision_records]))
            if collision_records
            else None
        ),
        "fisher_interior_points": len(fisher_records),
        "fisher_sigma_min_median": float(np.median(sigma_min)) if len(sigma_min) else None,
        "fisher_sigma_min_p05": float(np.quantile(sigma_min, 0.05)) if len(sigma_min) else None,
        "cv_records": cv_results,
        "collision_records": collision_records,
        "fisher_records": fisher_records,
    }


def analyze_observability(output_dir: Path) -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    records = []
    for path in sorted((output_dir / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            continue
        shape, absolute = _fingerprint_features(metadata, sequence, columns, bit_radius_m)
        records.append(
            {
                "metadata": metadata,
                "shape": shape,
                "shape_plus_absolute": np.concatenate((shape, absolute)),
            }
        )
    result = {
        "stage": "P0b_nuisance_observability",
        "successful_trials": len(records),
        "shape": _analyze_feature_family(records, "shape", bit_radius_m),
        "shape_plus_absolute": _analyze_feature_family(records, "shape_plus_absolute", bit_radius_m),
        "claim_boundary": (
            "Pilot feature/Fisher/collision diagnostics under four bounded nuisances; not a trained CNN, "
            "formal global identifiability proof, hardware result, or safe-transport validation."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output_dir / "observability_report.json", result)
    lines = [
        "# M8 P0b Nuisance Observability Report",
        "",
        f"Updated: {result['updated_utc']}",
        "",
        f"Successful trials: {result['successful_trials']}",
        "",
        "| feature family | state accuracy | median / p90 angle error | radial MAE | action collisions | Fisher sigma_min median / p05 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("shape", "shape_plus_absolute"):
        item = result[key]
        fisher_text = (
            f"{item['fisher_sigma_min_median']:.2f} / {item['fisher_sigma_min_p05']:.2f}"
            if item["fisher_sigma_min_median"] is not None
            else "n/a"
        )
        lines.append(
            f"| {key} | {item['cv_state_accuracy']:.3f} | "
            f"{item['cv_angle_error_median_deg']:.2f} / {item['cv_angle_error_p90_deg']:.2f} deg | "
            f"{item['cv_radial_mae_mm']:.3f} mm | "
            f"{item['action_collision_count']} / {item['base_pose_groups']} | {fisher_text} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            result["claim_boundary"],
            "",
            "The first decision should be based on direction error and action collisions. Radial outputs remain optional until their nuisance-conditioned benefit is confirmed in downstream posterior transport.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _load_feature_records(output_dir: Path, bit_radius_m: float) -> list[dict]:
    records = []
    for path in sorted((output_dir / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            continue
        shape, absolute = _fingerprint_features(metadata, sequence, columns, bit_radius_m)
        records.append(
            {
                "metadata": metadata,
                "shape": shape,
                "shape_plus_absolute": np.concatenate((shape, absolute)),
            }
        )
    return records


def _independent_collision_metrics(train: list[dict], test: list[dict], feature_key: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in train:
        grouped.setdefault(record["metadata"]["base_pose_id"], []).append(record)
    centroids = {
        group_id: np.mean(np.vstack([record[feature_key] for record in records]), axis=0)
        for group_id, records in grouped.items()
    }
    train_features = np.vstack([record[feature_key] for record in train])
    global_scale = np.std(train_features, axis=0)
    residuals = []
    for group_id, records in grouped.items():
        values = np.vstack([record[feature_key] for record in records])
        residuals.append(values - centroids[group_id])
    nuisance_scale = np.sqrt(np.mean(np.vstack(residuals) ** 2, axis=0))
    nuisance_scale = np.maximum(nuisance_scale, 0.05 * np.maximum(global_scale, 1.0e-6))

    predictions = []
    for record in test:
        distances = {
            group_id: float(np.linalg.norm((record[feature_key] - centroid) / nuisance_scale))
            for group_id, centroid in centroids.items()
        }
        predicted_group = min(distances, key=distances.get)
        predicted = grouped[predicted_group][0]["metadata"]
        true = record["metadata"]
        angle_error = _angular_error_deg(
            float(predicted["base_angle_deg"]), float(true["base_angle_deg"])
        )
        radial_error_m = abs(float(predicted["radius_m"]) - float(true["radius_m"]))
        state_correct = predicted["search_state"] == true["search_state"]
        predictions.append(
            {
                "trial_id": true["trial_id"],
                "true_group": true["base_pose_id"],
                "predicted_group": predicted_group,
                "state_correct": state_correct,
                "angle_error_deg": angle_error,
                "radial_error_m": radial_error_m,
                "direction_acceptable": angle_error <= 7.5,
                "state_direction_acceptable": state_correct and angle_error <= 7.5,
                "full_action_proxy_acceptable": (
                    state_correct and angle_error <= 7.5 and radial_error_m <= 0.0004
                ),
            }
        )
    angles = np.asarray([record["angle_error_deg"] for record in predictions], dtype=float)
    radial = np.asarray([record["radial_error_m"] for record in predictions], dtype=float)
    return {
        "feature_family": feature_key,
        "train_trials": len(train),
        "test_trials": len(test),
        "test_groups": len({record["metadata"]["base_pose_id"] for record in test}),
        "base_group_accuracy": float(
            np.mean([record["true_group"] == record["predicted_group"] for record in predictions])
        ),
        "state_accuracy": float(np.mean([record["state_correct"] for record in predictions])),
        "angle_error_median_deg": float(np.median(angles)),
        "angle_error_p90_deg": float(np.quantile(angles, 0.90)),
        "radial_mae_mm": float(1000.0 * np.mean(radial)),
        "direction_acceptable_rate": float(
            np.mean([record["direction_acceptable"] for record in predictions])
        ),
        "state_direction_acceptable_rate": float(
            np.mean([record["state_direction_acceptable"] for record in predictions])
        ),
        "full_action_proxy_acceptable_rate": float(
            np.mean([record["full_action_proxy_acceptable"] for record in predictions])
        ),
        "predictions": predictions,
    }


def analyze_collision_followup(output_dir: Path) -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    train = _load_feature_records(DEFAULT_P0B_OUT, bit_radius_m)
    test = _load_feature_records(output_dir, bit_radius_m)
    result = {
        "stage": "P0c_independent_collision_followup",
        "shape": _independent_collision_metrics(train, test, "shape"),
        "shape_plus_absolute": _independent_collision_metrics(
            train, test, "shape_plus_absolute"
        ),
        "claim_boundary": (
            "Independent bounded-nuisance repeats on P0b-selected hard groups using fixed feature centroids; "
            "not a CNN result, calibrated posterior, hardware validation, or closed-loop action guarantee."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output_dir / "collision_followup_report.json", result)
    lines = [
        "# M8 P0c Independent Collision Follow-Up",
        "",
        f"Updated: {result['updated_utc']}",
        "",
        "| feature family | state accuracy | angle median / p90 | radial MAE | direction acceptable | state+direction acceptable | full action proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("shape", "shape_plus_absolute"):
        item = result[key]
        lines.append(
            f"| {key} | {item['state_accuracy']:.3f} | "
            f"{item['angle_error_median_deg']:.2f} / {item['angle_error_p90_deg']:.2f} deg | "
            f"{item['radial_mae_mm']:.3f} mm | {item['direction_acceptable_rate']:.3f} | "
            f"{item['state_direction_acceptable_rate']:.3f} | "
            f"{item['full_action_proxy_acceptable_rate']:.3f} |"
        )
    lines.extend(["", "## Boundary", "", result["claim_boundary"], ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _write_manifest(
    output_dir: Path, tasks: list[PilotTask], results: list[dict], stage_name: str
) -> None:
    result_by_id = {result["trial_id"]: result for result in results}
    records = []
    for task in tasks:
        result = result_by_id.get(task.trial_id, {})
        records.append({**asdict(task), "status": result.get("status", "existing_or_unknown")})
    stage_boundary = {
        "P0a_C6_symmetry_audit": "nominal M8 symmetry pilot",
        "P0b_nuisance_observability": "bounded-nuisance M8 observability pilot",
        "P0c_independent_collision_followup": "independent bounded-nuisance hard-case pilot",
        "P0d_paired_sector_anchor_audit": "paired bounded-nuisance physical-sector pilot",
    }.get(stage_name, "M8 conical pilot")
    _atomic_json(
        output_dir / "manifest.json",
        {
            "stage": stage_name,
            "tasks": len(tasks),
            "records": records,
            "raw_columns": RAW_COLUMNS,
            "phase_columns": PHASE_COLUMNS,
            "claim_boundary": stage_boundary + "; no real validation",
            "updated_utc": _utc_now(),
        },
    )


def run(args: argparse.Namespace) -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    is_symmetry = args.stage == "symmetry"
    is_observability = args.stage == "observability"
    is_sector_anchors = args.stage == "sector-anchors"
    if is_symmetry:
        tasks = build_symmetry_tasks(spec)
    elif is_observability:
        tasks = build_observability_tasks(spec)
    elif is_sector_anchors:
        tasks = build_sector_anchor_tasks(spec)
    else:
        tasks = build_collision_tasks(spec)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    default_output = (
        DEFAULT_OUT
        if is_symmetry
        else DEFAULT_P0B_OUT
        if is_observability
        else DEFAULT_P0D_OUT
        if is_sector_anchors
        else DEFAULT_P0C_OUT
    )
    output_dir = Path(args.output or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trials").mkdir(exist_ok=True)

    if args.audit_only:
        audit = (
            audit_symmetry(output_dir)
            if is_symmetry
            else analyze_observability(output_dir)
            if is_observability
            else audit_sector_anchors(output_dir)
            if is_sector_anchors
            else analyze_collision_followup(output_dir)
        )
        if is_symmetry:
            selected = {
                key: audit[key]
                for key in (
                    "comparisons",
                    "mean_nrmse",
                    "p95_nrmse",
                    "mean_primary_shape_correlation",
                    "p05_primary_shape_correlation",
                )
            }
        elif is_observability:
            selected = {
                "successful_trials": audit["successful_trials"],
                "shape": {
                    key: audit["shape"][key]
                    for key in (
                        "cv_state_accuracy",
                        "cv_angle_error_median_deg",
                        "cv_angle_error_p90_deg",
                        "action_collision_rate",
                    )
                },
            }
        elif is_sector_anchors:
            selected = {
                key: audit[key]
                for key in (
                    "successful_trials",
                    "complete_orbits",
                    "comparisons",
                    "mean_nrmse",
                    "p95_nrmse",
                    "mean_primary_shape_correlation",
                    "p05_primary_shape_correlation",
                )
            }
        else:
            selected = {
                "shape": {
                    key: audit["shape"][key]
                    for key in (
                        "state_accuracy",
                        "angle_error_p90_deg",
                        "direction_acceptable_rate",
                    )
                },
                "shape_plus_absolute": {
                    key: audit["shape_plus_absolute"][key]
                    for key in (
                        "state_accuracy",
                        "angle_error_p90_deg",
                        "direction_acceptable_rate",
                    )
                },
            }
        print(json.dumps(selected, indent=2))
        return 0

    pending = []
    existing_results: list[dict] = []
    for task in tasks:
        path = output_dir / "trials" / f"{task.trial_id}.npz"
        if args.resume and path.exists():
            existing_results.append({"trial_id": task.trial_id, "status": "existing", "path": str(path)})
        else:
            pending.append(task)

    started = time.perf_counter()
    results = list(existing_results)
    ok = sum(result["status"] in {"ok", "existing"} for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    progress_path = output_dir / "progress.json"

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        completed_new = len(results) - len(existing_results)
        rate = completed_new / elapsed
        remaining = max(len(tasks) - len(results), 0)
        _atomic_json(
            progress_path,
            {
                "stage": (
                    "P0a_C6_symmetry_audit"
                    if is_symmetry
                    else "P0b_nuisance_observability"
                    if is_observability
                    else "P0d_paired_sector_anchor_audit"
                    if is_sector_anchors
                    else "P0c_independent_collision_followup"
                ),
                "total": len(tasks),
                "completed": len(results),
                "succeeded_or_existing": ok,
                "failed": failed,
                "rate_trials_s": rate,
                "eta_s": remaining / rate if rate > 0.0 else None,
                "output_dir": str(output_dir),
                "updated_utc": _utc_now(),
            },
        )

    checkpoint()
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(str(output_dir), args.phase_bins),
        ) as pool:
            futures = {pool.submit(_sample_task, task): task for task in pending}
            bar = tqdm(
                total=len(tasks),
                initial=len(existing_results),
                desc=(
                    "P0a M8 C6"
                    if is_symmetry
                    else "P0b M8 obs"
                    if is_observability
                    else "P0d M8 sectors"
                    if is_sector_anchors
                    else "P0c M8 hard"
                ),
                unit="trial",
                dynamic_ncols=True,
                disable=args.no_progress,
            )
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"trial_id": task.trial_id, "status": "failed", "error": repr(exc)}
                results.append(result)
                if result["status"] in {"ok", "existing", "skipped"}:
                    ok += 1
                else:
                    failed += 1
                bar.update(1)
                bar.set_postfix(ok=ok, fail=failed, refresh=False)
                checkpoint()
            bar.close()

    stage_name = (
        "P0a_C6_symmetry_audit"
        if is_symmetry
        else "P0b_nuisance_observability"
        if is_observability
        else "P0d_paired_sector_anchor_audit"
        if is_sector_anchors
        else "P0c_independent_collision_followup"
    )
    _write_manifest(output_dir, tasks, results, stage_name)
    audit = (
        audit_symmetry(output_dir)
        if is_symmetry
        else analyze_observability(output_dir)
        if is_observability
        else audit_sector_anchors(output_dir)
        if is_sector_anchors
        else analyze_collision_followup(output_dir)
    )
    checkpoint()
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tasks": len(tasks),
                "ok_or_existing": ok,
                "failed": failed,
                **(
                    {
                        "symmetry_comparisons": audit["comparisons"],
                        "symmetry_mean_nrmse": audit["mean_nrmse"],
                        "symmetry_p95_nrmse": audit["p95_nrmse"],
                        "symmetry_mean_primary_shape_correlation": audit[
                            "mean_primary_shape_correlation"
                        ],
                        "symmetry_p05_primary_shape_correlation": audit[
                            "p05_primary_shape_correlation"
                        ],
                    }
                    if is_symmetry
                    else {
                        "shape_cv_state_accuracy": audit["shape"]["cv_state_accuracy"],
                        "shape_cv_angle_p90_deg": audit["shape"]["cv_angle_error_p90_deg"],
                        "shape_action_collision_rate": audit["shape"]["action_collision_rate"],
                    }
                    if is_observability
                    else {
                        "sector_complete_orbits": audit["complete_orbits"],
                        "sector_mean_nrmse": audit["mean_nrmse"],
                        "sector_p95_nrmse": audit["p95_nrmse"],
                        "sector_mean_shape_correlation": audit[
                            "mean_primary_shape_correlation"
                        ],
                        "sector_p05_shape_correlation": audit[
                            "p05_primary_shape_correlation"
                        ],
                    }
                    if is_sector_anchors
                    else {
                        "shape_test_state_accuracy": audit["shape"]["state_accuracy"],
                        "shape_test_angle_p90_deg": audit["shape"]["angle_error_p90_deg"],
                        "combined_test_state_accuracy": audit["shape_plus_absolute"]["state_accuracy"],
                        "combined_test_angle_p90_deg": audit["shape_plus_absolute"][
                            "angle_error_p90_deg"
                        ],
                    }
                ),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the M8 P0a conical C6 symmetry pilot.")
    parser.add_argument(
        "--stage",
        choices=("symmetry", "observability", "collisions", "sector-anchors"),
        default="symmetry",
    )
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=min(32, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins <= 0 or args.phase_bins % 6:
        parser.error("--phase-bins must be positive and divisible by 6")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
