#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python3.11}"
if [[ "$(uname -s)" == Darwin && -d /Library/Developer/CommandLineTools ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
fi
if [[ ! -e .venv ]]; then
  "$python_bin" -m venv --prompt efu .venv
elif [[ ! -f .venv/pyvenv.cfg || ! -x .venv/bin/python ]]; then
  printf 'Refusing to overwrite a non-environment .venv path\n' >&2
  exit 2
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
