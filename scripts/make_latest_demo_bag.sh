#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${RL_SAR_DIR}/.." && pwd)"
BAG_BASE_DIR="${BAG_BASE_DIR:-${WS_DIR}/bag}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <run_name_or_localized_bag_path> [goal_forward goal_left goal_yaw]

If the argument is an existing bag directory, use it directly.
Otherwise first try the fixed output name, then fall back to older timestamped outputs:
  ${BAG_BASE_DIR}/<run_name>_localized
  ${BAG_BASE_DIR}/<run_name>_localized_*

Output defaults to:
  ${BAG_BASE_DIR}/<run_name>_demo

Examples:
  ./make_latest_demo_bag.sh easy_3_smooth
  ./make_latest_demo_bag.sh clutter_7_0_0 7 0 0
  ./make_latest_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth_localized

Environment overrides:
  DEMO_BAG=/abs/path/demo_bag
  OUTPUT_HZ=20
  PATH_HZ=2
  SMOOTH_WINDOW=0.8
  START_SEC=
  END_SEC=
  MAX_STEP_DIST=0
  MAX_STEP_YAW=0
  MAX_OUTLIER_RUN=0
  MAX_INTERP_SPEED=1.5
  REPAIR_INDEX_RANGE=74:77
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi
if [[ $# -ne 1 && $# -ne 4 ]]; then
  usage >&2
  exit 1
fi

INPUT="$1"
OUTPUT_HZ="${OUTPUT_HZ:-20}"
PATH_HZ="${PATH_HZ:-2}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-0.8}"
START_SEC="${START_SEC:-}"
END_SEC="${END_SEC:-}"
MAX_STEP_DIST="${MAX_STEP_DIST:-0}"
MAX_STEP_YAW="${MAX_STEP_YAW:-0}"
MAX_OUTLIER_RUN="${MAX_OUTLIER_RUN:-0}"
MAX_INTERP_SPEED="${MAX_INTERP_SPEED:-1.5}"
REPAIR_INDEX_RANGE="${REPAIR_INDEX_RANGE:-}"
GOAL_REL="${GOAL_REL:-}"
if [[ $# -eq 4 ]]; then
  GOAL_REL="$2 $3 $4"
fi

if [[ -d "${INPUT}" ]]; then
  LOCALIZED_BAG="$(cd "${INPUT}" && pwd)"
  RUN_NAME="$(basename "${LOCALIZED_BAG}")"
  RUN_NAME="${RUN_NAME%%_localized}"
  RUN_NAME="${RUN_NAME%%_localized_*}"
else
  RUN_NAME="${INPUT}"
  if [[ -d "${BAG_BASE_DIR}/${RUN_NAME}_localized" ]]; then
    LOCALIZED_BAG="${BAG_BASE_DIR}/${RUN_NAME}_localized"
  else
    mapfile -t candidates < <(find "${BAG_BASE_DIR}" -maxdepth 1 -type d -name "${RUN_NAME}_localized_*" | sort)
    if [[ ${#candidates[@]} -eq 0 ]]; then
      echo "[make-demo] ERROR: no localized bag found for run '${RUN_NAME}' in ${BAG_BASE_DIR}" >&2
      echo "[make-demo] Expected: ${BAG_BASE_DIR}/${RUN_NAME}_localized" >&2
      exit 2
    fi
    LOCALIZED_BAG="${candidates[-1]}"
  fi
fi

DEMO_BAG="${DEMO_BAG:-${BAG_BASE_DIR}/${RUN_NAME}_demo}"
if [[ -z "${REPAIR_INDEX_RANGE}" && "${RUN_NAME}" == "clutter_7_0_0" ]]; then
  REPAIR_INDEX_RANGE="74:77"
fi
if [[ -z "${REPAIR_INDEX_RANGE}" && "${RUN_NAME}" == "clutter_straight_easy" ]]; then
  REPAIR_INDEX_RANGE="36:36,60:65"
fi
if [[ "${RUN_NAME}" == "clutter_straight_easy" ]]; then
  START_SEC="${START_SEC:-5.59}"
  END_SEC="${END_SEC:-46.0}"
fi
if [[ "${RUN_NAME}" == "maze_65_6_0_smooth" ]]; then
  START_SEC="${START_SEC:-32.45}"
fi

echo "[make-demo] localized=${LOCALIZED_BAG}"
echo "[make-demo] output=${DEMO_BAG}"
echo "[make-demo] output_hz=${OUTPUT_HZ} path_hz=${PATH_HZ} smooth_window=${SMOOTH_WINDOW} start_sec=${START_SEC:-auto} end_sec=${END_SEC:-auto} max_step_dist=${MAX_STEP_DIST} max_step_yaw=${MAX_STEP_YAW} max_outlier_run=${MAX_OUTLIER_RUN} max_interp_speed=${MAX_INTERP_SPEED} repair_index_range=${REPAIR_INDEX_RANGE:-none} goal_rel=${GOAL_REL:-none}"

cmd=(
  "${SCRIPT_DIR}/make_smooth_trajectory_bag.py"
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
)
if [[ -n "${START_SEC}" ]]; then
  cmd+=(--start-sec "${START_SEC}")
fi
if [[ -n "${END_SEC}" ]]; then
  cmd+=(--end-sec "${END_SEC}")
fi
if [[ -n "${REPAIR_INDEX_RANGE}" ]]; then
  cmd+=(--repair-index-range "${REPAIR_INDEX_RANGE}")
fi
if [[ -n "${GOAL_REL}" ]]; then
  read -r goal_rel_x goal_rel_y goal_rel_yaw <<< "${GOAL_REL}"
  cmd+=(--goal-rel "${goal_rel_x}" "${goal_rel_y}" "${goal_rel_yaw}")
fi
"${cmd[@]}"

echo "[make-demo] done."
echo "[make-demo] replay:"
echo "  ros2 bag play ${DEMO_BAG} --rate 1.0"
