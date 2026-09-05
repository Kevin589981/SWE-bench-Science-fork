#!/usr/bin/env bash
set -euo pipefail

# The server installs Pier with uv.  PIER_PYTHON can be overridden for another
# deployment while keeping the adapter's default self-contained.
PIER_PYTHON="${PIER_PYTHON:-/root/.local/share/uv/tools/datacurve-pier/bin/python}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PIER_PYTHON" "$SCRIPT_DIR/pier_dynamic_ports.py" "$@"
