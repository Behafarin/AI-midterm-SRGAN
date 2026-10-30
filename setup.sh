#!/usr/bin/env bash
# One-time environment setup on the lab server (Linux + NVIDIA GPU). Uses uv.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "Installing uv ..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

python check_setup.py
