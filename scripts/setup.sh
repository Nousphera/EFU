#!/usr/bin/env bash
# Create a project-specific Python environment without changing system Python.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python3.12}"
venv_dir="${1:-.venv}"
"$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Python 3.11-3.13 required; 3.12 tested"'
# Prefer an installed standalone macOS toolchain; never accept licenses automatically.
if [[ "$(uname -s)" == Darwin && -d /Library/Developer/CommandLineTools ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
fi
if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
elif [[ ! -f "$venv_dir/pyvenv.cfg" || ! -x "$venv_dir/bin/python" ]]; then
  printf 'Refusing non-venv path: %s\n' "$venv_dir" >&2
  exit 2
fi
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -c constraints.txt -r requirements.txt
"$venv_dir/bin/python" -m pip check
printf 'Environment ready. Activate with: source "%s/bin/activate"\n' "$venv_dir"
