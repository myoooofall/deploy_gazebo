#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENABLE_NAV2="false"
ENABLE_RVIZ="false"
ENABLE_MAP_SERVER="false"
USE_SIM_TIME="false"
PUBLISH_BASE2LIDAR_TF="true"
REG_METHOD="NDT_OMP"
DOWNSAMPLE_RESOLUTION="0.5"
POINTS_TOPIC="/rslidar_points"
IMU_TOPIC="/imu/data"
T_DIFF="0.25"

LITE_COG_ROOT="${LITE_COG_ROOT:-}"
MAP_PCD="${MAP_PCD:-}"
MAP_YAML="${MAP_YAML:-}"
RVIZ_CONFIG="${RVIZ_CONFIG:-}"

usage() {
  cat <<'EOF'
Usage:
  ./start_localization.sh [enable_nav2]
  ./start_localization.sh [options]

Options:
  --nav2 <true|false>               Enable dr_nav2 bringup (default: false)
  --rviz <true|false>               Enable rviz2 (default: false)
  --map-server <true|false>         Enable nav2_map_server (default: false)
  --sim-time <true|false>           Use /clock (default: false)
  --publish-base2lidar-tf <bool>    Publish static base_link->rslidar TF (default: true)
  --map-pcd <path>                  Global map .pcd path
  --map-yaml <path>                 Occupancy map .yaml path
  --rviz-config <path>              RViz config path
  --lite-cog-root <path>            lite_cog_ros2 root directory
  --points-topic <topic>            Input points topic (default: /rslidar_points)
  --imu-topic <topic>               Input imu topic (default: /imu/data)
  --reg-method <name>               hdl reg_method, e.g. NDT_OMP/NDT_CUDA_P2D
  --downsample-resolution <float>   hdl downsample resolution (default: 0.5)
  --t-diff <float>                  hdl t_diff (default: 0.25)
  -h, --help                        Show this help
EOF
}

is_bool() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  if ! is_bool "$1"; then
    echo "[ERR] invalid positional enable_nav2: $1 (expect true/false)" >&2
    exit 1
  fi
  ENABLE_NAV2="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nav2) ENABLE_NAV2="${2:-}"; shift 2 ;;
    --rviz) ENABLE_RVIZ="${2:-}"; shift 2 ;;
    --map-server) ENABLE_MAP_SERVER="${2:-}"; shift 2 ;;
    --sim-time) USE_SIM_TIME="${2:-}"; shift 2 ;;
    --publish-base2lidar-tf) PUBLISH_BASE2LIDAR_TF="${2:-}"; shift 2 ;;
    --map-pcd) MAP_PCD="${2:-}"; shift 2 ;;
    --map-yaml) MAP_YAML="${2:-}"; shift 2 ;;
    --rviz-config) RVIZ_CONFIG="${2:-}"; shift 2 ;;
    --lite-cog-root) LITE_COG_ROOT="${2:-}"; shift 2 ;;
    --points-topic) POINTS_TOPIC="${2:-}"; shift 2 ;;
    --imu-topic) IMU_TOPIC="${2:-}"; shift 2 ;;
    --reg-method) REG_METHOD="${2:-}"; shift 2 ;;
    --downsample-resolution) DOWNSAMPLE_RESOLUTION="${2:-}"; shift 2 ;;
    --t-diff) T_DIFF="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERR] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

for v in ENABLE_NAV2 ENABLE_RVIZ ENABLE_MAP_SERVER USE_SIM_TIME PUBLISH_BASE2LIDAR_TF; do
  val="${!v}"
  if ! is_bool "$val"; then
    echo "[ERR] ${v} must be true/false, got: ${val}" >&2
    exit 1
  fi
done

if [[ -z "${LITE_COG_ROOT}" ]]; then
  if [[ -d "${WS_DIR}/lite_cog_ros2" ]]; then
    LITE_COG_ROOT="${WS_DIR}/lite_cog_ros2"
  elif [[ -d "/home/ysc/lite_cog_ros2" ]]; then
    LITE_COG_ROOT="/home/ysc/lite_cog_ros2"
  fi
fi

if [[ -z "${MAP_PCD}" ]]; then
  if [[ -n "${LITE_COG_ROOT}" ]]; then
    MAP_PCD="${LITE_COG_ROOT}/system/map/lite3.pcd"
  else
    MAP_PCD="${WS_DIR}/lite_cog_ros2/system/map/lite3.pcd"
  fi
fi

if [[ -z "${MAP_YAML}" ]]; then
  if [[ -n "${LITE_COG_ROOT}" ]]; then
    MAP_YAML="${LITE_COG_ROOT}/system/map/lite3.yaml"
  else
    MAP_YAML="${WS_DIR}/lite_cog_ros2/system/map/lite3.yaml"
  fi
fi

if [[ -z "${RVIZ_CONFIG}" ]]; then
  RVIZ_CONFIG="${SCRIPT_DIR}/src/rl_sar/rviz/goal_only.rviz"
fi

set +u
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
fi
if [[ -n "${LITE_COG_ROOT}" && -f "${LITE_COG_ROOT}/navigation2-foxy/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${LITE_COG_ROOT}/navigation2-foxy/install/setup.bash"
fi
if [[ -n "${LITE_COG_ROOT}" && -f "${LITE_COG_ROOT}/nav/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${LITE_COG_ROOT}/nav/install/setup.bash"
fi
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/install/setup.bash"
fi
set -u

if [[ ! -f "${MAP_PCD}" ]]; then
  echo "[WARN] map pcd not found: ${MAP_PCD}"
fi
if [[ ! -f "${MAP_YAML}" ]]; then
  echo "[WARN] map yaml not found: ${MAP_YAML}"
fi
if [[ "${ENABLE_RVIZ}" == "true" && ! -f "${RVIZ_CONFIG}" ]]; then
  echo "[WARN] rviz config not found: ${RVIZ_CONFIG}"
fi

echo "[localization] LITE_COG_ROOT=${LITE_COG_ROOT:-<unset>}"
echo "[localization] map_pcd=${MAP_PCD}"
echo "[localization] map_yaml=${MAP_YAML}"
echo "[localization] nav2=${ENABLE_NAV2} rviz=${ENABLE_RVIZ} map_server=${ENABLE_MAP_SERVER} sim_time=${USE_SIM_TIME}"
echo "[localization] topics: points=${POINTS_TOPIC} imu=${IMU_TOPIC}"

exec ros2 launch "${SCRIPT_DIR}/lite_localization.launch.py" \
  enable_nav2:="${ENABLE_NAV2}" \
  enable_rviz:="${ENABLE_RVIZ}" \
  enable_map_server:="${ENABLE_MAP_SERVER}" \
  publish_base2lidar_tf:="${PUBLISH_BASE2LIDAR_TF}" \
  use_sim_time:="${USE_SIM_TIME}" \
  map_path:="${MAP_PCD}" \
  map_server_config_file:="${MAP_YAML}" \
  rviz_config:="${RVIZ_CONFIG}" \
  reg_method:="${REG_METHOD}" \
  downsample_resolution:="${DOWNSAMPLE_RESOLUTION}" \
  points_topic:="${POINTS_TOPIC}" \
  imu_topic:="${IMU_TOPIC}" \
  t_diff:="${T_DIFF}"
