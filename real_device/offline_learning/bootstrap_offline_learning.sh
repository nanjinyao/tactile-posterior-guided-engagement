#!/usr/bin/env bash
set -euo pipefail

offline_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
real_dir="$(cd "${offline_dir}/.." && pwd)"
venv_dir="${real_dir}/.venv-offline-learning"

if ! python3 -m venv "${venv_dir}"; then
  # Ubuntu installations without python3.8-venv can still create the isolated
  # environment through the user-installed virtualenv package.
  python3 -m virtualenv --clear "${venv_dir}"
fi
"${venv_dir}/bin/python" -m pip install --upgrade "pip==24.3.1"
"${venv_dir}/bin/python" -m pip install \
  "numpy==1.24.4" \
  "matplotlib==3.7.5" \
  "scipy==1.10.1" \
  "scikit-learn==1.3.2" \
  "pandas==2.0.3" \
  "h5py==3.11.0" \
  "PyYAML==6.0.2" \
  "joblib==1.4.2" \
  "pytest==8.3.5"
"${venv_dir}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.4.1+cpu"
"${venv_dir}/bin/python" - <<'PY'
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
print("verified imports:", ", ".join(expected))
PY
echo "offline learning environment ready: ${venv_dir}"
