#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible wrapper:
#   ./start_nav.sh true
#   ./start_nav.sh --nav2 true --rviz false ...
exec "${SCRIPT_DIR}/start_localization.sh" "$@"
