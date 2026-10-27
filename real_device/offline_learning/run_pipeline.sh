#!/usr/bin/env bash
set -euo pipefail

offline_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
real_dir="$(cd "${offline_dir}/.." && pwd)"
group_dir="$(cd "${real_dir}/.." && pwd)"
python_bin="${real_dir}/.venv-offline-learning/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "offline environment missing; run ${offline_dir}/bootstrap_offline_learning.sh first" >&2
  exit 2
fi

"${python_bin}" - <<'PY'
import h5py
import joblib
import matplotlib
import numpy
import pandas
import pytest
import scipy
import sklearn
import torch
import yaml

expected = {
    "numpy": (numpy.__version__, "1.24.4"),
    "matplotlib": (matplotlib.__version__, "3.7.5"),
    "scipy": (scipy.__version__, "1.10.1"),
    "scikit-learn": (sklearn.__version__, "1.3.2"),
    "pandas": (pandas.__version__, "2.0.3"),
    "h5py": (h5py.__version__, "3.11.0"),
    "PyYAML": (yaml.__version__, "6.0.2"),
    "joblib": (joblib.__version__, "1.4.2"),
    "pytest": (pytest.__version__, "8.3.5"),
    "torch": (torch.__version__, "2.4.1+cpu"),
}
wrong = {key: value for key, value in expected.items() if value[0] != value[1]}
assert not wrong, wrong
print("offline environment: PASS")
PY

"${python_bin}" "${offline_dir}/build_real_multisize_dataset.py"
calibration_config="${group_dir}/mujoco_unscrewing_probe_new/configs/real_calibrated_v1.json"
calibration_root="${real_dir}/offline_analysis/real_multisize_learning_v1/sim_calibration_v1"
if [[ -f "${calibration_config}" && -f "${calibration_root}/manifest.json" ]]; then
  echo "real-calibrated simulation bundle exists; reuse immutable v1"
else
  "${python_bin}" \
    "${group_dir}/mujoco_unscrewing_probe_new/src/build_real_observation_calibration.py"
fi
"${python_bin}" "${offline_dir}/train_real_multisize_direction.py"
"${python_bin}" "${offline_dir}/evaluate_real_multisize_bundle.py"
"${python_bin}" -m pytest -q \
  "${offline_dir}/tests/test_offline_learning_contract.py" \
  "${real_dir}/tests/test_real_multisize_runtime_v2.py" \
  "${real_dir}/tests/test_m12_learned_policy_contract_static.py" \
  "${real_dir}/tests/test_model_runtime_authority.py" \
  "${group_dir}/mujoco_unscrewing_probe_new/tests/test_real_observation_calibration.py" \
  "${group_dir}/mujoco_unscrewing_probe_new/tests/test_real_cad_clocking_contract.py"
"${python_bin}" -m unittest discover \
  -s "${real_dir}/model/parametric_tooling" \
  -p "test_render_previews_yaw.py"

echo "real multisize offline pipeline: PASS"
