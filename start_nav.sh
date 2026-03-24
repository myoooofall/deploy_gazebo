source /home/ysc/lite_cog_ros2/navigation2-foxy/install/setup.bash
source /home/ysc/lite_cog_ros2/nav/install/setup.bash

ENABLE_NAV2="${1:-false}"
if [[ "${ENABLE_NAV2}" != "true" && "${ENABLE_NAV2}" != "false" ]]; then
    echo "[WARN] Invalid enable_nav2='${ENABLE_NAV2}', fallback to false"
    ENABLE_NAV2="false"
fi

ros2 launch hdl_localization lite_localization.launch.py enable_nav2:="${ENABLE_NAV2}"
