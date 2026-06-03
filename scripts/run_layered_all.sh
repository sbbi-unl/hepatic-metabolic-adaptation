#!/usr/bin/env bash
# macOS/Linux launcher for the layered manuscript executor.
# Run from the pn_pipeline folder.

set -euo pipefail

CONFIG="${CONFIG:-run_layered_config.json}"

# If you use conda on Linux/macOS, uncomment the next line:
# conda activate metabolic-modeling

python run_layered_executor.py --config "$CONFIG" "$@"
