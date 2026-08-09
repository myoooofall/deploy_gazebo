#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${RL_SAR_DIR}/.." && pwd)"
BAG_BASE_DIR="${BAG_BASE_DIR:-${WS_DIR}/bag}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <demo_bag_path_or_name>

Open the complete presentation view:
  1) publish /map from lite3.yaml
  2) open RViz with smooth trajectory config
  3) play the demo bag

Examples:
  ./play_demo_bag.sh easy_3_smooth_demo
  ./play_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth_demo

Environment overrides:
  MAP_YAML=${WS_DIR}/map/map/lite3.yaml
  RVIZ_CONFIG=${SCRIPT_DIR}/demo_smooth.rviz
  PLAY_RATE=1.0
  PLAY_DELAY_SEC=5
  WAIT_FOR_ENTER=true
  ENABLE_RVIZ=true
  KEEP_OPEN=true
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

INPUT="$1"
if [[ -d "${INPUT}" ]]; then
  DEMO_BAG="$(cd "${INPUT}" && pwd)"
else
  DEMO_BAG="${BAG_BASE_DIR}/${INPUT}"
fi

if [[ ! -d "${DEMO_BAG}" ]]; then
  echo "[play-demo] ERROR: demo bag not found: ${DEMO_BAG}" >&2
  exit 2
fi

MAP_YAML="${MAP_YAML:-${WS_DIR}/map/map/lite3.yaml}"
RVIZ_CONFIG="${RVIZ_CONFIG:-${SCRIPT_DIR}/demo_smooth.rviz}"
PLAY_RATE="${PLAY_RATE:-1.0}"
PLAY_DELAY_SEC="${PLAY_DELAY_SEC:-5}"
WAIT_FOR_ENTER="${WAIT_FOR_ENTER:-true}"
ENABLE_RVIZ="${ENABLE_RVIZ:-true}"
KEEP_OPEN="${KEEP_OPEN:-true}"
MAP_PUBLISHER="${MAP_PUBLISHER:-${WS_DIR}/lite_cog_ros2/system/scripts/slam/publish_pgm_map.py}"

if [[ ! -f "${MAP_YAML}" ]]; then
  echo "[play-demo] ERROR: map yaml not found: ${MAP_YAML}" >&2
  exit 3
fi
if [[ ! -f "${RVIZ_CONFIG}" ]]; then
  echo "[play-demo] ERROR: rviz config not found: ${RVIZ_CONFIG}" >&2
  exit 4
fi
if [[ ! -x "${MAP_PUBLISHER}" ]]; then
  echo "[play-demo] ERROR: map publisher not found/executable: ${MAP_PUBLISHER}" >&2
  exit 5
fi

set +u
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
fi
if [[ -f "${WS_DIR}/lite_cog_ros2/nav/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/lite_cog_ros2/nav/install/setup.bash"
fi
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/install/setup.bash"
fi
set -u

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  echo
  echo "[play-demo] stopping background processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -INT "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait "${PIDS[@]:-}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[play-demo] bag=${DEMO_BAG}"
echo "[play-demo] map=${MAP_YAML}"
echo "[play-demo] rviz=${ENABLE_RVIZ} config=${RVIZ_CONFIG}"
echo "[play-demo] rate=${PLAY_RATE} delay=${PLAY_DELAY_SEC}s wait_for_enter=${WAIT_FOR_ENTER} keep_open=${KEEP_OPEN}"

python3 "${MAP_PUBLISHER}" --yaml "${MAP_YAML}" --topic /map --frame-id map &
PIDS+=("$!")
sleep 1

if [[ "${ENABLE_RVIZ}" == "true" ]]; then
  rviz2 -d "${RVIZ_CONFIG}" &
  PIDS+=("$!")
  sleep 2
fi

if [[ "${PLAY_DELAY_SEC}" != "0" ]]; then
  echo "[play-demo] waiting ${PLAY_DELAY_SEC}s before replay..."
  sleep "${PLAY_DELAY_SEC}"
fi

if [[ "${WAIT_FOR_ENTER}" == "true" ]]; then
  echo "[play-demo] adjust RViz/recording angle, then press Enter to start replay..."
  read -r _
fi

ros2 bag play "${DEMO_BAG}" --rate "${PLAY_RATE}" &
PLAY_PID="$!"
PIDS+=("${PLAY_PID}")

wait "${PLAY_PID}"
echo "[play-demo] replay finished."

if [[ "${KEEP_OPEN}" == "true" ]]; then
  echo "[play-demo] keeping RViz/map publisher alive. Press Ctrl+C to close them."
  while true; do
    sleep 3600
  done
fi
