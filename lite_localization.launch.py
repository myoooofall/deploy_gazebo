# voa_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_map_server():
    declare_map_server_config_file_cmd = DeclareLaunchArgument(
        'map_server_config_file',
        default_value=os.path.join(
            '/home/ysc/lite_cog_ros2/system/map/lite3.yaml'
        )
    )
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[
            {"yaml_filename": LaunchConfiguration('map_server_config_file')},
            {"topic_name": "map"},
            {"frame_id": "map"},
        ],
    )
    return declare_map_server_config_file_cmd, map_server

def launch_hdl_localization_composition():
    hdl_localization_composition = Node(
        package='hdl_localization',
        executable='hdl_localization_composition',
        # prefix=['xterm -e gdb -ex run --args'],
        parameters=[
            {"globalmap_pcd": "/home/ysc/lite_cog_ros2/system/map/lite3.pcd"},
            {"convert_utm_to_local": True},

            {"odom_child_frame_id": "base_link"},
            # imu settings
            # during "cool_time", imu inputs are ignored
            {"use_imu": True},
            {"invert_acc": False},
            {"invert_gyro": False},
            {"cool_time_duration": 2.0},
            # robot odometry-based prediction
            {"enable_robot_odometry_prediction": False},
            {"robot_odom_frame_id": "odom"},
            # ndt settings
            # available reg_methods: NDT_OMP, NDT_CUDA_P2D, NDT_CUDA_D2D
            {"reg_method": "NDT_OMP"},
            # if NDT is slow for your PC, try DIRECT1 serach method, which is a bit unstable but extremely fast
            {"ndt_neighbor_search_method": "DIRECT1"},
            {"ndt_neighbor_search_radius": 3.0},
            {"ndt_resolution": 1.5},
            {"downsample_resolution": 0.5},
            # if "specify_init_pose" is true, pose estimator will be initialized with the following params
            # otherwise, you need to input an initial pose with "2D Pose Estimate" on rviz"
            {"specify_init_pose": True},
            {"init_pos_x": 0.0},
            {"init_pos_y": 0.0},
            {"init_pos_z": 0.0},
            {"init_ori_w": 1.0},
            {"init_ori_x": 0.0},
            {"init_ori_y": 0.0},
            {"init_ori_z": 0.0},

            {"use_global_localization": False},

            {"t_diff": 0.25}
        ],
        remappings=[
            ('/velodyne_points', '/rslidar_points'),
            ('/gpsimu_driver/imu_data', '/imu/data'),
        ],
        # prefix=['xterm -e gdb -ex run --args'],
    )
    return hdl_localization_composition

def launch_nav2(enable_nav2):
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("dr_nav2"), "launch", "dr_nav2.launch.py")
        ),
        condition=IfCondition(enable_nav2)
    )
    return included_launch

def launch_plot_status_py():
    return

def launch_rviz():
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': False}],  # RViz 默认参数，可根据需要修改
        arguments=['-d', os.path.join(get_package_share_directory('dr_nav2'), 'rviz', 'dr_nav2.rviz')]
    )
    return rviz

def launch_base2lidar_tf_broadcaster():
    base2lidar_tf_broadcaste = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base2lidar_tf_broadcaste',
        arguments=['0.16', '0.0', '0.47', '1.57', '0.0', '0.0', 'base_link', 'rslidar']
    )
    return base2lidar_tf_broadcaste


plot_estimation_errors = False
debug = False

def generate_launch_description():
    ld = LaunchDescription()
    declare_enable_nav2_cmd = DeclareLaunchArgument("enable_nav2", default_value="false", description="Enable Nav2 bringup (controller/planner stack)")
    ld.add_action(declare_enable_nav2_cmd)

    hdl_localization_composition = launch_hdl_localization_composition()
    ld.add_action(hdl_localization_composition)

    if debug:
        declare_map_server_config_file_cmd, map_server = launch_map_server()
        ld.add_action(declare_map_server_config_file_cmd)
        ld.add_action(map_server)

        rviz = launch_rviz()
        ld.add_action(rviz)
    else:
        nav2 = launch_nav2(LaunchConfiguration("enable_nav2"))
        ld.add_action(nav2)

    if plot_estimation_errors:
        pass

    return ld


