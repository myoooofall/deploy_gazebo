#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${RL_SAR_DIR}/.." && pwd)"
DEFAULT_BAG_BASE_DIR="${WS_DIR}/bag"

usage() {
  cat <<EOF
Usage: $(basename "$0") <raw_bag_path_or_name>

End-to-end presentation pipeline:
  1) run offline localization and record a localized bag with /odom
  2) crop to navigation window using /nav/goal_pred_map
  3) write a smoothed presentation bag with /odom_smooth and /odom_path_smooth

Only required argument is the raw bag path/name.

Defaults:
  output hz        20
  smooth window    0.8s
  replay rate      0.5
  rviz             true

Environment overrides:
  PLAY_RATE=0.5
  OUTPUT_HZ=20
  PATH_HZ=2
  SMOOTH_WINDOW=0.8
  MAX_STEP_DIST=0
  MAX_STEP_YAW=0
  MAX_OUTLIER_RUN=0
  ENABLE_RVIZ=true
  INIT_POSE_WAIT_SEC=8
  LOCALIZED_BAG=/abs/path/localized_bag
  DEMO_BAG=/abs/path/demo_bag
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

INPUT="$1"
if [[ -d "${INPUT}" ]]; then
  RAW_BAG="$(cd "${INPUT}" && pwd)"
else
  RAW_BAG="${DEFAULT_BAG_BASE_DIR}/${INPUT}"
fi

if [[ ! -d "${RAW_BAG}" ]]; then
  echo "[offline-demo] ERROR: raw bag not found: ${RAW_BAG}" >&2
  exit 2
fi

RUN_NAME="$(basename "${RAW_BAG}")"
LOCALIZED_BAG="${LOCALIZED_BAG:-${DEFAULT_BAG_BASE_DIR}/${RUN_NAME}_localized}"
DEMO_BAG="${DEMO_BAG:-${DEFAULT_BAG_BASE_DIR}/${RUN_NAME}_demo_smooth}"
OUTPUT_HZ="${OUTPUT_HZ:-20}"
PATH_HZ="${PATH_HZ:-2}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-0.8}"
MAX_STEP_DIST="${MAX_STEP_DIST:-0}"
MAX_STEP_YAW="${MAX_STEP_YAW:-0}"
MAX_OUTLIER_RUN="${MAX_OUTLIER_RUN:-0}"
MAX_INTERP_SPEED="${MAX_INTERP_SPEED:-1.5}"

echo "[offline-demo] raw=${RAW_BAG}"
echo "[offline-demo] localized=${LOCALIZED_BAG}"
echo "[offline-demo] demo=${DEMO_BAG}"

"${SCRIPT_DIR}/localize_record_bag.sh" "${RAW_BAG}" "${LOCALIZED_BAG}"

echo "[offline-demo] building smooth trajectory bag..."
"${SCRIPT_DIR}/make_smooth_trajectory_bag.py" \
  "${LOCALIZED_BAG}" \
  "${DEMO_BAG}" \
  --output-hz "${OUTPUT_HZ}" \
  --path-hz "${PATH_HZ}" \
  --smooth-window "${SMOOTH_WINDOW}" \
  --max-step-dist "${MAX_STEP_DIST}" \
  --max-step-yaw "${MAX_STEP_YAW}" \
  --max-outlier-run "${MAX_OUTLIER_RUN}" \
  --max-interp-speed "${MAX_INTERP_SPEED}" \
  --force

echo "[offline-demo] done."
echo "[offline-demo] Replay with:"
echo "  ros2 bag play ${DEMO_BAG} --rate 1.0"
