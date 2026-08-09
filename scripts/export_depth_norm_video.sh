#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${RL_SAR_DIR}/.." && pwd)"
BAG_BASE_DIR="${BAG_BASE_DIR:-${WS_DIR}/bag}"
EXPORTER="${EXPORTER:-${RL_SAR_DIR}/src/rl_sar/scripts/offline/export_depth_video_from_bag.py}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <run_name_or_depth_bag_path> [output_mp4]

Export /camera/depth/processed_norm from a split bag to MP4.

Examples:
  ./export_depth_norm_video.sh test_depth
  ./export_depth_norm_video.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/test_depth
  ./export_depth_norm_video.sh test_depth /tmp/test_depth.mp4

Environment overrides:
  BAG_BASE_DIR=${BAG_BASE_DIR}
  FPS=10
  COLORMAP=none
  NAV_CROP=true
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

INPUT="$1"
OUT_ARG="${2:-}"

if [[ -d "${INPUT}" ]]; then
  INPUT_DIR="$(cd "${INPUT}" && pwd)"
elif [[ -d "${BAG_BASE_DIR}/${INPUT}" ]]; then
  INPUT_DIR="$(cd "${BAG_BASE_DIR}/${INPUT}" && pwd)"
else
  echo "[depth-video] ERROR: bag not found: ${INPUT}" >&2
  echo "[depth-video] Tried: ${BAG_BASE_DIR}/${INPUT}" >&2
  exit 2
fi

if [[ -d "${INPUT_DIR}/depth_norm" ]]; then
  DEPTH_BAG="${INPUT_DIR}/depth_norm"
  RUN_DIR="${INPUT_DIR}"
else
  DEPTH_BAG="${INPUT_DIR}"
  RUN_DIR="$(cd "${INPUT_DIR}/.." && pwd)"
fi

if [[ ! -f "${DEPTH_BAG}/metadata.yaml" ]]; then
  echo "[depth-video] ERROR: depth bag metadata not found: ${DEPTH_BAG}/metadata.yaml" >&2
  exit 3
fi

if [[ ! -x "${EXPORTER}" ]]; then
  echo "[depth-video] ERROR: exporter not found/executable: ${EXPORTER}" >&2
  exit 4
fi

FPS="${FPS:-10}"
COLORMAP="${COLORMAP:-none}"
NAV_CROP="${NAV_CROP:-true}"
OUT="${OUT_ARG:-${RUN_DIR}/depth_norm.mp4}"

echo "[depth-video] depth_bag=${DEPTH_BAG}"
echo "[depth-video] out=${OUT}"
echo "[depth-video] fps=${FPS} colormap=${COLORMAP} nav_crop=${NAV_CROP}"

cmd=(
  "${EXPORTER}" "${DEPTH_BAG}"
  --topic /camera/depth/processed_norm \
  --out "${OUT}" \
  --fps "${FPS}" \
  --colormap "${COLORMAP}" \
  --mode direct
)

if [[ "${NAV_CROP}" == "true" && -f "${RUN_DIR}/small/metadata.yaml" ]]; then
  cmd+=(
    --crop-bag "${RUN_DIR}/small"
    --crop-topic /nav/cmd_high
    --crop-topic /nav/goal_pred_map
    --crop-topic /nav/goal_error_body
  )
fi

exec "${cmd[@]}"
