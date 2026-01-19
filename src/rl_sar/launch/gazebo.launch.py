# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import time
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory, get_package_prefix


def generate_launch_description():
    rname = LaunchConfiguration("rname")
    wname = LaunchConfiguration("wname")
    terrain = LaunchConfiguration("terrain")
    seed = LaunchConfiguration("seed")
    difficulty = LaunchConfiguration("difficulty")
    start_corner = LaunchConfiguration("start_corner")
    world_out = LaunchConfiguration("world_out")
    wall_thickness = LaunchConfiguration("wall_thickness")
    wall_height = LaunchConfiguration("wall_height")
    cell_size = LaunchConfiguration("cell_size")
    check_inflation = LaunchConfiguration("check_inflation")
    min_reachable = LaunchConfiguration("min_reachable")
    max_attempts = LaunchConfiguration("max_attempts")
    spawn_keepout = LaunchConfiguration("spawn_keepout")
    spawn_box = LaunchConfiguration("spawn_box")
    nav_goal_dialog = LaunchConfiguration("nav_goal_dialog")

    world_path = LaunchConfiguration("world_path")
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

    robot_name = ParameterValue(Command(["echo -n ", rname]), value_type=str)
    ros_namespace = ParameterValue(Command(["echo -n ", "/", rname, "_gazebo"]), value_type=str)
    gazebo_model_name = ParameterValue(Command(["echo -n ", rname, "_gazebo"]), value_type=str)

    robot_description = ParameterValue(
        Command([
            "xacro ",
            Command(["echo -n ", Command(["ros2 pkg prefix ", rname, "_description"])]),
            "/share/", rname, "_description/xacro/robot.xacro"
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # NOTE: gazebo_ros/launch/gazebo.launch.py does not forward arguments to gzserver.launch.py.
    # Include gzserver/gzclient directly so we can reliably load libgazebo_ros_state.so (publishes /model_states).
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            # "verbose": "true",
            # "pause": "true",
            "world": world_path,
            # Use the no-space form ("-slib...") because ExecuteProcess passes each list item as one argv token.
            "extra_gazebo_args": "-slibgazebo_ros_state.so",
        }.items(),
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzclient.launch.py")
        ),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "robot_model",
            "-x", spawn_x,
            "-y", spawn_y,
            "-Y", spawn_yaw,
            "-z", "1.0",
        ],
        output="screen",
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    robot_joint_controller_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["robot_joint_controller"],
        output="screen",
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'deadzone': 0.1,
            'autorepeat_rate': 0.0,
        }],
    )

    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
            "gazebo_model_name": gazebo_model_name,
        }],
    )

    def _setup(context):
        terrain_val = terrain.perform(context).strip()
        wname_val = wname.perform(context).strip()
        seed_val = seed.perform(context).strip()
        difficulty_val = difficulty.perform(context).strip()
        start_corner_val = start_corner.perform(context).strip()
        world_out_val = world_out.perform(context).strip()

        # Default spawn at world origin.
        spawn_x_val = "0.0"
        spawn_y_val = "0.0"
        spawn_yaw_val = "0.0"

        if terrain_val in ("maze", "navigation"):
            try:
                seed_int = int(seed_val)
            except ValueError:
                seed_int = 0
            if seed_int < 0:
                seed_int = int(time.time())

            if world_out_val:
                out_path = world_out_val
            else:
                safe_diff = difficulty_val.replace(".", "p")
                out_path = f"/tmp/rl_sar_{terrain_val}_seed{seed_int}_diff{safe_diff}_{start_corner_val}.world"

            script_path = os.path.join(get_package_prefix("rl_sar"), "lib", "rl_sar", "generate_terrain_world.py")
            if not os.path.exists(script_path):
                raise RuntimeError(f"Missing terrain generator script: {script_path}")

            cmd = [
                "python3",
                script_path,
                "--type", terrain_val,
                "--seed", str(seed_int),
                "--difficulty", str(difficulty_val),
                "--start-corner", str(start_corner_val),
                "--out", out_path,
                "--wall-thickness", str(wall_thickness.perform(context).strip()),
                "--wall-height", str(wall_height.perform(context).strip()),
                "--check-inflation", str(check_inflation.perform(context).strip()),
                "--spawn-keepout", str(spawn_keepout.perform(context).strip()),
                "--spawn-box", str(spawn_box.perform(context).strip()),
                "--min-reachable", str(min_reachable.perform(context).strip()),
                "--max-attempts", str(max_attempts.perform(context).strip()),
                "--print-json",
            ]

            cell_size_val = cell_size.perform(context).strip()
            if cell_size_val and cell_size_val.lower() != "none":
                cmd += ["--cell-size", cell_size_val]

            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(proc.stdout.strip().splitlines()[-1])

            world_path_val = str(payload["world_path"])
            spawn_x_val = str(payload["spawn_x"])
            spawn_y_val = str(payload["spawn_y"])
            spawn_yaw_val = str(payload.get("spawn_yaw", 0.0))
        else:
            if wname_val.startswith("/"):
                world_path_val = wname_val
            else:
                world_path_val = os.path.join(get_package_share_directory("rl_sar"), "worlds", wname_val)

        return [
            SetLaunchConfiguration("world_path", world_path_val),
            SetLaunchConfiguration("spawn_x", spawn_x_val),
            SetLaunchConfiguration("spawn_y", spawn_y_val),
            SetLaunchConfiguration("spawn_yaw", spawn_yaw_val),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "rname",
            description="Robot name (e.g., a1, go2)",
            default_value=TextSubstitution(text=""),
        ),
        DeclareLaunchArgument(
            "wname",
            description="World filename under rl_sar/worlds, or an absolute world path.",
            default_value=TextSubstitution(text="stairs.world"),
        ),
        DeclareLaunchArgument(
            "terrain",
            description="If set to 'maze' or 'navigation', generate a random world at launch-time (overrides wname).",
            default_value=TextSubstitution(text=""),
        ),
        DeclareLaunchArgument("seed", description="Random seed (use -1 for time-based).", default_value=TextSubstitution(text="0")),
        DeclareLaunchArgument("difficulty", description="Difficulty in [0,1].", default_value=TextSubstitution(text="0.5")),
        DeclareLaunchArgument("start_corner", description="Spawn corner: bl/br/tl/tr.", default_value=TextSubstitution(text="bl")),
        DeclareLaunchArgument(
            "world_out",
            description="Output world path for generated terrain (default: /tmp/rl_sar_<...>.world).",
            default_value=TextSubstitution(text=""),
        ),
        DeclareLaunchArgument("wall_thickness", description="Wall thickness (m).", default_value=TextSubstitution(text="0.3")),
        DeclareLaunchArgument("wall_height", description="Wall height (m).", default_value=TextSubstitution(text="0.5")),
        DeclareLaunchArgument("cell_size", description="Maze cell size (m) or empty.", default_value=TextSubstitution(text="")),
        DeclareLaunchArgument("check_inflation", description="Inflation for reachability check (m).", default_value=TextSubstitution(text="0.4")),
        DeclareLaunchArgument("spawn_keepout", description="Keepout radius around spawn (m).", default_value=TextSubstitution(text="0.6")),
        DeclareLaunchArgument("spawn_box", description="Spawn sampling box size in the corner (m).", default_value=TextSubstitution(text="1.0")),
        DeclareLaunchArgument("nav_goal_dialog", description="Start the nav goal GUI publisher.", default_value=TextSubstitution(text="true")),
        DeclareLaunchArgument("min_reachable", description="Minimum reachable free-space fraction.", default_value=TextSubstitution(text="0.98")),
        DeclareLaunchArgument("max_attempts", description="Max resample attempts.", default_value=TextSubstitution(text="20")),
        DeclareLaunchArgument("world_path", default_value=TextSubstitution(text="")),
        DeclareLaunchArgument("spawn_x", default_value=TextSubstitution(text="0.0")),
        DeclareLaunchArgument("spawn_y", default_value=TextSubstitution(text="0.0")),
        DeclareLaunchArgument("spawn_yaw", default_value=TextSubstitution(text="0.0")),
        OpaqueFunction(function=_setup),
        robot_state_publisher_node,
        gazebo_server,
        gazebo_client,
        spawn_entity,
        joint_state_broadcaster_node,
        # robot_joint_controller_node,  # Spawn in rl_sim.cpp
        joy_node,
        param_node,
        Node(
            package="rl_sar",
            executable="nav_goal_dialog.py",
            name="nav_goal_dialog",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(nav_goal_dialog),
        ),
    ])
