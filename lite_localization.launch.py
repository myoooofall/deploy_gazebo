import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _try_get_package_share(pkg_name: str) -> str:
    try:
        return get_package_share_directory(pkg_name)
    except PackageNotFoundError:
        return ""


def _resolve_default_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_dir = os.path.dirname(script_dir)

    env_map_pcd = os.environ.get("LITE3_MAP_PCD", "")
    env_map_yaml = os.environ.get("LITE3_MAP_YAML", "")
    env_rviz = os.environ.get("LITE3_RVIZ_CONFIG", "")

    map_pcd_candidates = [
        env_map_pcd,
        os.path.join(ws_dir, "lite_cog_ros2", "system", "map", "lite3.pcd"),
        "/home/ysc/lite_cog_ros2/system/map/lite3.pcd",
    ]
    map_yaml_candidates = [
        env_map_yaml,
        os.path.join(ws_dir, "lite_cog_ros2", "system", "map", "lite3.yaml"),
        "/home/ysc/lite_cog_ros2/system/map/lite3.yaml",
    ]
    rviz_candidates = [
        env_rviz,
        os.path.join(script_dir, "src", "rl_sar", "rviz", "goal_only.rviz"),
    ]

    def first_existing(candidates):
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return candidates[1] if len(candidates) > 1 else ""

    map_pcd = first_existing(map_pcd_candidates)
    map_yaml = first_existing(map_yaml_candidates)

    dr_nav2_share = _try_get_package_share("dr_nav2")
    if dr_nav2_share:
        rviz_candidates.append(os.path.join(dr_nav2_share, "rviz", "dr_nav2.rviz"))
    rviz_cfg = first_existing(rviz_candidates)
    return map_pcd, map_yaml, rviz_cfg, dr_nav2_share


def _launch_map_server(condition=None):
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        condition=condition,
        parameters=[
            {"yaml_filename": LaunchConfiguration("map_server_config_file")},
            {"topic_name": "map"},
            {"frame_id": "map"},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )
    return map_server


def _launch_hdl_localization_composition():
    return Node(
        package="hdl_localization",
        executable="hdl_localization_composition",
        parameters=[
            {"globalmap_pcd": LaunchConfiguration("map_path")},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"convert_utm_to_local": True},
            {"odom_child_frame_id": "base_link"},
            {"use_imu": True},
            {"invert_acc": False},
            {"invert_gyro": False},
            {"cool_time_duration": 2.0},
            {"enable_robot_odometry_prediction": False},
            {"robot_odom_frame_id": "odom"},
            {"reg_method": LaunchConfiguration("reg_method")},
            {"ndt_neighbor_search_method": "DIRECT1"},
            {"ndt_neighbor_search_radius": 3.0},
            {"ndt_resolution": 1.5},
            {"downsample_resolution": LaunchConfiguration("downsample_resolution")},
            {"specify_init_pose": True},
            {"init_pos_x": 0.0},
            {"init_pos_y": 0.0},
            {"init_pos_z": 0.0},
            {"init_ori_w": 1.0},
            {"init_ori_x": 0.0},
            {"init_ori_y": 0.0},
            {"init_ori_z": 0.0},
            {"use_global_localization": False},
            {"t_diff": LaunchConfiguration("t_diff")},
        ],
        remappings=[
            ("/velodyne_points", LaunchConfiguration("points_topic")),
            ("/gpsimu_driver/imu_data", LaunchConfiguration("imu_topic")),
        ],
    )


def _launch_rviz():
    return Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )


def _launch_base2lidar_tf():
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base2lidar_tf_broadcaster",
        condition=IfCondition(LaunchConfiguration("publish_base2lidar_tf")),
        arguments=["0.16", "0.0", "0.47", "1.57", "0.0", "0.0", "base_link", "rslidar"],
    )


def generate_launch_description():
    default_map_pcd, default_map_yaml, default_rviz, dr_nav2_share = _resolve_default_paths()

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument("enable_nav2", default_value="false"))
    ld.add_action(DeclareLaunchArgument("enable_rviz", default_value="false"))
    ld.add_action(DeclareLaunchArgument("enable_map_server", default_value="false"))
    ld.add_action(DeclareLaunchArgument("publish_base2lidar_tf", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="false"))

    ld.add_action(DeclareLaunchArgument("map_path", default_value=default_map_pcd))
    ld.add_action(DeclareLaunchArgument("map_server_config_file", default_value=default_map_yaml))
    ld.add_action(DeclareLaunchArgument("rviz_config", default_value=default_rviz))

    ld.add_action(DeclareLaunchArgument("reg_method", default_value="NDT_OMP"))
    ld.add_action(DeclareLaunchArgument("downsample_resolution", default_value="0.5"))
    ld.add_action(DeclareLaunchArgument("t_diff", default_value="0.25"))
    ld.add_action(DeclareLaunchArgument("points_topic", default_value="/rslidar_points"))
    ld.add_action(DeclareLaunchArgument("imu_topic", default_value="/imu/data"))

    ld.add_action(_launch_hdl_localization_composition())
    ld.add_action(_launch_base2lidar_tf())

    if dr_nav2_share:
        ld.add_action(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(dr_nav2_share, "launch", "dr_nav2.launch.py")),
                condition=IfCondition(LaunchConfiguration("enable_nav2")),
            )
        )
    else:
        ld.add_action(LogInfo(msg="[lite_localization] dr_nav2 not found, enable_nav2 will be ignored."))

    ld.add_action(_launch_map_server(condition=IfCondition(LaunchConfiguration("enable_map_server"))))

    ld.add_action(_launch_rviz())
    return ld
