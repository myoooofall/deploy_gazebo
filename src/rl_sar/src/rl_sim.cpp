/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"

#if defined(USE_ROS2)
static geometry_msgs::msg::Quaternion YawToQuaternion(double yaw)
{
    geometry_msgs::msg::Quaternion q;
    const double half = 0.5 * yaw;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(half);
    q.w = std::cos(half);
    return q;
}
#endif

static double QuaternionToYaw(double x, double y, double z, double w)
{
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

static double WrapToPi(double a)
{
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static void RotateVecByQuat(double qx, double qy, double qz, double qw, double vx, double vy, double vz,
                            double &ox, double &oy, double &oz)
{
    // v' = v + w*(2*q.xyz x v) + (q.xyz x (2*q.xyz x v))
    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);

    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

static std::string BuildNavGoalMarkerSdf()
{
    // Visual-only marker (no collision), an arrow pointing along +X.
    // Spawned using /spawn_entity with reference_frame="robot_model".
    return R"(
<sdf version="1.6">
  <model name="nav_goal_marker">
    <static>true</static>
    <link name="link">
      <!-- Shaft -->
      <visual name="shaft">
        <pose>0.45 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.9 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <emissive>0.6 0 0 1</emissive>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Red</name>
          </script>
        </material>
        <cast_shadows>false</cast_shadows>
      </visual>

      <!-- Head (brighter + wider) -->
      <visual name="head">
        <pose>1.05 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.3 0.18 0.12</size></box>
        </geometry>
        <material>
          <ambient>1 1 0 1</ambient>
          <diffuse>1 1 0 1</diffuse>
          <emissive>1 1 0 1</emissive>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Yellow</name>
          </script>
        </material>
        <cast_shadows>false</cast_shadows>
      </visual>
    </link>
  </model>
</sdf>
)";
}

static std::string BuildNavPredMarkerSdf()
{
    return R"(
<sdf version="1.6">
  <model name="nav_pred_marker">
    <static>true</static>
    <link name="link">
      <visual name="shaft">
        <pose>0.45 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.9 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
          <emissive>0 0.6 0 1</emissive>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Green</name>
          </script>
        </material>
        <cast_shadows>false</cast_shadows>
      </visual>
      <visual name="head">
        <pose>1.05 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.3 0.18 0.12</size></box>
        </geometry>
        <material>
          <ambient>0 1 1 1</ambient>
          <diffuse>0 1 1 1</diffuse>
          <emissive>0 1 1 1</emissive>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Cyan</name>
          </script>
        </material>
        <cast_shadows>false</cast_shadows>
      </visual>
    </link>
  </model>
</sdf>
)";
}

RL_Sim::RL_Sim()
#if defined(USE_ROS2)
    : rclcpp::Node("rl_sim_node")
#endif
{
#if defined(USE_ROS1)
    this->ang_vel_type = "ang_vel_world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)
    this->ang_vel_type = "ang_vel_body";
    this->ros_namespace = this->get_namespace();
    // get params from param_node
    param_client = this->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(this->get_node_base_interface(), future, std::chrono::seconds(5));
    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
            std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
            std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
    }
#endif

    // read params from yaml
    this->ReadYamlBase(this->robot_name);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init torch
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);

    // init robot
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.num_of_dofs);
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.num_of_dofs);
    this->robot_state_subscriber_msg.motor_state.resize(this->params.num_of_dofs);
#endif
    this->InitOutputs();
    this->InitControl();

#if defined(USE_ROS1)
    this->StartJointController(this->ros_namespace, this->params.joint_controller_names);
    // publisher
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        const std::string &joint_controller_name = this->params.joint_controller_names[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        const std::string &joint_controller_name = this->params.joint_controller_names[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0;
        this->joint_velocities[joint_controller_name] = 0.0;
        this->joint_efforts[joint_controller_name] = 0.0;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    this->StartJointController(this->ros_namespace, this->params.joint_names);
    // publisher
    this->robot_command_publisher = this->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    // subscriber
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->gazebo_imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->GazeboImuCallback(msg);}
    );
    // Gazebo topics are often best-effort; use best-effort QoS to avoid mismatch and missing callbacks.
    const auto model_states_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    this->gazebo_model_states_subscriber = this->create_subscription<gazebo_msgs::msg::ModelStates>(
        "/gazebo/model_states", model_states_qos,
        [this] (const gazebo_msgs::msg::ModelStates::SharedPtr msg) { this->GazeboModelStatesCallback(msg); }
    );
    this->gazebo_model_states_subscriber_alt = this->create_subscription<gazebo_msgs::msg::ModelStates>(
        "/model_states", model_states_qos,
        [this] (const gazebo_msgs::msg::ModelStates::SharedPtr msg) { this->GazeboModelStatesCallback(msg); }
    );
    // Preferred source for robot world pose in ROS2: nav_msgs/Odometry from gazebo_ros_p3d plugin.
    const auto odom_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    this->nav_odom_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
        "/go2_gazebo/odom", odom_qos,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->NavOdomCallback(msg); }
    );
    this->nav_odom_subscriber_alt = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", odom_qos,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->NavOdomCallback(msg); }
    );
    this->robot_state_subscriber = this->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );

    this->depth_image_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/depth/image_rect_raw", rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Sim::DepthImageCallback, this, std::placeholders::_1));

    this->processed_depth_publisher = this->create_publisher<sensor_msgs::msg::Image>(
        "/camera/camera/depth/processed", rclcpp::SystemDefaultsQoS());
        depth_buffer = DepthBuffer(1, 60, 86, 3);  // 1个环境，3帧历史，最终尺寸60x86 (height=60, width=86)

    // hierarchical navigation: body-frame goal only (no odom dependency)
    this->nav_goal_body_subscriber = this->create_subscription<geometry_msgs::msg::Pose2D>(
        "/nav_goal_body", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Pose2D::SharedPtr msg) { this->NavGoalBodyCallback(msg); }
    );

	    // service
	    this->gazebo_pause_physics_client = this->create_client<std_srvs::srv::Empty>("/pause_physics");
	    this->gazebo_unpause_physics_client = this->create_client<std_srvs::srv::Empty>("/unpause_physics");
	    this->gazebo_reset_world_client = this->create_client<std_srvs::srv::Empty>("/reset_world");

    // gazebo goal marker services (optional)
    this->nav_goal_marker_spawn_client = this->create_client<gazebo_msgs::srv::SpawnEntity>("/spawn_entity");
    this->nav_goal_marker_delete_client = this->create_client<gazebo_msgs::srv::DeleteEntity>("/delete_entity");

	    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
	    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // init hierarchical nav policy (best-effort; safe to fail)
    this->InitHierarchicalNav();

	    // loop
	    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Sim::RobotControl, this));
	    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Sim::RunModel, this));
	    this->loop_navi = std::make_shared<LoopFunc>("loop_nav",this->nav_dt_,std::bind(&RL_Sim::RunHighLevel, this));
	    this->loop_control->start();
	    this->loop_rl->start();
	    this->loop_navi->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}
void RL_Sim::DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 只在每个时间步更新一次深度图
    if (this->motion_time % 5 == 0) {  // 每5个时间步更新一次
        torch::Tensor processed_depth = depth_buffer.process_depth_image(msg,
            this->processed_depth_publisher);
        // torch::Tensor processed_depth = depth_buffer.process_depth_image_old(msg);
        // processed_depth shape: [60, 86], insert函数会处理batch维度
        depth_buffer.insert(processed_depth);
        this->motion_time = 1;
    }
    this->motion_time++;
}
RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    this->loop_navi->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "robot_joint_controller_params.yaml";
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    // First, try to unload the controller if it already exists (using unspawner)
    pid_t pid_unload = fork();
    if (pid_unload == 0)
    {
        std::string unload_cmd = "ros2 run controller_manager unspawner robot_joint_controller 2>/dev/null || true";
        execlp("sh", "sh", "-c", unload_cmd.c_str(), nullptr);
        exit(0);
    }
    else if (pid_unload > 0)
    {
        int status;
        waitpid(pid_unload, &status, 0);  // Ignore errors from unload
        usleep(100000);  // Small delay (100ms) to ensure cleanup
    }

    // Now load the controller
    pid_t pid = fork();
    if (pid == 0)
    {
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "-p " + tmp_path.string() + " ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            throw std::runtime_error("Failed to start joint controller");
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}

void RL_Sim::GetState(RobotState<double> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    const auto &orientation = this->gazebo_imu.orientation;
    const auto &angular_velocity = this->gazebo_imu.angular_velocity;
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.joint_controller_names[this->params.joint_mapping[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.joint_controller_names[this->params.joint_mapping[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.joint_controller_names[this->params.joint_mapping[i]]];
#elif defined(USE_ROS2)
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.joint_mapping[i]].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.joint_mapping[i]].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.joint_mapping[i]].tau_est;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.joint_mapping[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.joint_mapping[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.joint_mapping[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.joint_mapping[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.joint_mapping[i]].tau = command->motor_command.tau[i];
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.joint_mapping[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.joint_mapping[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.joint_mapping[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.joint_mapping[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.joint_mapping[i]].tau = command->motor_command.tau[i];
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->joint_publishers[this->params.joint_controller_names[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{
    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
#if defined(USE_ROS1)
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (simulation_running)
    {
        this->motiontime++;

        if (this->nav_enabled_.load() && this->nav_models_loaded_.load())
        {
            this->control.x = this->nav_cmd_x_.load();
            this->control.y = this->nav_cmd_y_.load();
            this->control.yaw = this->nav_cmd_yaw_.load();
        }
        else
        {
            if (this->control.current_keyboard == Input::Keyboard::W)
            {
                this->control.x += 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::S)
            {
                this->control.x -= 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::A)
            {
                this->control.y += 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::D)
            {
                this->control.y -= 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::Q)
            {
                this->control.yaw += 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::E)
            {
                this->control.yaw -= 0.1;
                this->control.current_keyboard = this->control.last_keyboard;
            }
            if (this->control.current_keyboard == Input::Keyboard::Space)
            {
                this->control.x = 0;
                this->control.y = 0;
                this->control.yaw = 0;
                this->control.current_keyboard = this->control.last_keyboard;
            }
        }
        if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
        {
            this->control.navigation_mode = !this->control.navigation_mode;
            this->nav_enabled_.store(this->control.navigation_mode);
            std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
            this->control.current_keyboard = this->control.last_keyboard;
        }

        {
            std::lock_guard<std::mutex> lock(this->nav_state_mutex_);
            this->GetState(&this->robot_state);
            this->UpdateHighFrequencyObs();
        }
        this->StateController(&this->robot_state, &this->robot_command);
        this->SetCommand(&this->robot_command);
    }
}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
}
#endif

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}

void RL_Sim::JoyCallback(
#if defined(USE_ROS1)
    const sensor_msgs::Joy::ConstPtr &msg
#elif defined(USE_ROS2)
    const sensor_msgs::msg::Joy::SharedPtr msg
#endif
)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    if (this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    if (!this->nav_enabled_.load())
    {
        this->control.x = this->joy_msg.axes[1] * 1.5; // LY
        this->control.y = this->joy_msg.axes[0] * 1.5; // LX
        this->control.yaw = this->joy_msg.axes[3] * 1.5; // RX
    }
}

#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}

void RL_Sim::GazeboModelStatesCallback(const gazebo_msgs::msg::ModelStates::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }
    const auto &names = msg->name;
    const auto &poses = msg->pose;
    if (names.size() != poses.size())
    {
        return;
    }

    size_t idx = static_cast<size_t>(-1);
    auto pick_by_key = [&](const std::string &key) -> bool
    {
        if (key.empty()) return false;
        for (size_t i = 0; i < names.size(); ++i)
        {
            if (names[i] == key)
            {
                idx = i;
                return true;
            }
        }
        for (size_t i = 0; i < names.size(); ++i)
        {
            if (names[i].find(key) != std::string::npos)
            {
                idx = i;
                return true;
            }
        }
        return false;
    };
    (void)pick_by_key(this->gazebo_model_name);
    if (idx == static_cast<size_t>(-1)) (void)pick_by_key("robot_model");
    if (idx == static_cast<size_t>(-1)) (void)pick_by_key(this->robot_name);
    if (idx == static_cast<size_t>(-1)) (void)pick_by_key("robot");
    if (idx == static_cast<size_t>(-1))
    {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
        {
            std::cout << LOGGER::WARNING << "No matching model name in /gazebo/model_states; set gazebo_model_name param. First few names:";
            for (size_t i = 0; i < std::min<size_t>(names.size(), 8); ++i) std::cout << " " << names[i];
            std::cout << std::endl;
        }
        return;
    }

    const auto &p = poses[idx].position;
    const auto &q = poses[idx].orientation;
    const double yaw = QuaternionToYaw(q.x, q.y, q.z, q.w);
    this->nav_base_world_x_.store(p.x);
    this->nav_base_world_y_.store(p.y);
    this->nav_base_world_z_.store(p.z);
    this->nav_base_world_yaw_.store(yaw);
    this->nav_base_world_qx_.store(q.x);
    this->nav_base_world_qy_.store(q.y);
    this->nav_base_world_qz_.store(q.z);
    this->nav_base_world_qw_.store(q.w);
    this->nav_base_world_valid_.store(true);
}

void RL_Sim::NavOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }
    const auto &p = msg->pose.pose.position;
    const auto &q = msg->pose.pose.orientation;
    const double yaw = QuaternionToYaw(q.x, q.y, q.z, q.w);
    this->nav_base_world_x_.store(p.x);
    this->nav_base_world_y_.store(p.y);
    this->nav_base_world_z_.store(p.z);
    this->nav_base_world_yaw_.store(yaw);
    this->nav_base_world_qx_.store(q.x);
    this->nav_base_world_qy_.store(q.y);
    this->nav_base_world_qz_.store(q.z);
    this->nav_base_world_qw_.store(q.w);
    this->nav_base_world_valid_.store(true);
}

void RL_Sim::NavGoalBodyCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
{
    this->nav_goal_body_x_.store(msg->x);
    this->nav_goal_body_y_.store(msg->y);
    this->nav_goal_body_yaw_.store(msg->theta);
    this->nav_has_goal_.store(true);
    this->nav_goal_seq_.fetch_add(1);
    std::cout << LOGGER::INFO << "NavGoalBody: x=" << msg->x << " y=" << msg->y << " yaw=" << msg->theta << std::endl;

    // Latch the goal in world using the robot pose at goal time so we can compute live goal_body each step
    // like the training code (yaw-only).
    if (this->nav_base_world_valid_.load())
    {
        const double rx = this->nav_base_world_x_.load();
        const double ry = this->nav_base_world_y_.load();
        const double ryaw = this->nav_base_world_yaw_.load();
        const double c = std::cos(ryaw);
        const double s = std::sin(ryaw);
        const double gx = rx + c * msg->x - s * msg->y;
        const double gy = ry + s * msg->x + c * msg->y;
        const double gyaw = WrapToPi(ryaw + msg->theta);
        this->nav_goal_world_x_.store(gx);
        this->nav_goal_world_y_.store(gy);
        this->nav_goal_world_yaw_.store(gyaw);
        this->nav_goal_world_valid_.store(true);
    }
    else
    {
        this->nav_goal_world_valid_.store(false);
    }

    // Best-effort visualization in Gazebo: place/update a marker in world.
    this->UpdateNavGoalMarker(msg->x, msg->y, msg->theta);
}
#endif

void RL_Sim::UpdateNavGoalMarker(double goal_body_x, double goal_body_y, double goal_body_yaw)
{
#if defined(USE_ROS2)
    const char *kRobotEntity = "robot_model";

    if (!this->nav_goal_marker_spawn_client || !this->nav_goal_marker_delete_client)
    {
        return;
    }

    if (!this->nav_goal_marker_delete_client->wait_for_service(std::chrono::milliseconds(200)))
    {
        return;
    }
    if (!this->nav_goal_marker_spawn_client->wait_for_service(std::chrono::milliseconds(200)))
    {
        return;
    }

    // For robustness (no extra Gazebo state plugin required), respawn the marker at the goal pose.
    // Using reference_frame="robot_model" makes goal_body_* interpreted in the robot body frame at spawn time.
    auto spawn_req = std::make_shared<gazebo_msgs::srv::SpawnEntity::Request>();
    spawn_req->name = "nav_goal_marker";
    spawn_req->xml = BuildNavGoalMarkerSdf();
    spawn_req->robot_namespace = "";
    spawn_req->reference_frame = kRobotEntity;
    spawn_req->initial_pose.position.x = goal_body_x;
    spawn_req->initial_pose.position.y = goal_body_y;
    spawn_req->initial_pose.position.z = 0.4;
    spawn_req->initial_pose.orientation = YawToQuaternion(goal_body_yaw);

    auto del_req = std::make_shared<gazebo_msgs::srv::DeleteEntity::Request>();
    del_req->name = "nav_goal_marker";
    (void)this->nav_goal_marker_delete_client->async_send_request(
        del_req,
        [this, spawn_req](rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedFuture)
        {
            (void)this->nav_goal_marker_spawn_client->async_send_request(
                spawn_req,
                [this](rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedFuture future)
                {
                    try
                    {
                        const auto resp = future.get();
                        if (!resp->success)
                        {
                            std::cout << LOGGER::WARNING << "Spawn nav_goal_marker failed: " << resp->status_message << std::endl;
                            this->nav_goal_marker_spawned_.store(false);
                            return;
                        }
                        this->nav_goal_marker_spawned_.store(true);
                    }
                    catch (...)
                    {
                        this->nav_goal_marker_spawned_.store(false);
                    }
                });
        });
#else
    (void)goal_body_x;
    (void)goal_body_y;
    (void)goal_body_yaw;
#endif
}

void RL_Sim::UpdateNavPredMarker(double pred_body_x, double pred_body_y, double pred_body_yaw)
{
#if defined(USE_ROS2)
    // Single marker (green) only, updated via delete+spawn.
    if (!this->nav_goal_marker_delete_client || !this->nav_goal_marker_spawn_client)
    {
        return;
    }
    if (!this->nav_goal_marker_delete_client->service_is_ready() || !this->nav_goal_marker_spawn_client->service_is_ready())
    {
        return;
    }

    // Avoid building up an unbounded queue of delete/spawn requests at 10Hz.
    static std::atomic<bool> inflight{false};
    if (inflight.exchange(true))
    {
        return;
    }

    auto spawn_req = std::make_shared<gazebo_msgs::srv::SpawnEntity::Request>();
    spawn_req->name = "nav_pred_marker";
    spawn_req->xml = BuildNavPredMarkerSdf();
    spawn_req->robot_namespace = "";
    spawn_req->reference_frame = "robot_model";
    spawn_req->initial_pose.position.x = pred_body_x;
    spawn_req->initial_pose.position.y = pred_body_y;
    spawn_req->initial_pose.position.z = 0.4;
    spawn_req->initial_pose.orientation = YawToQuaternion(pred_body_yaw);

    auto del_req = std::make_shared<gazebo_msgs::srv::DeleteEntity::Request>();
    del_req->name = "nav_pred_marker";
    (void)this->nav_goal_marker_delete_client->async_send_request(
        del_req,
        [this, spawn_req](rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedFuture)
        {
            (void)this->nav_goal_marker_spawn_client->async_send_request(
                spawn_req,
                [](rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedFuture) { inflight.store(false); });
        });
#else
    (void)pred_body_x;
    (void)pred_body_y;
    (void)pred_body_yaw;
#endif
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        // this->obs.lin_vel = torch::tensor({{this->vel.linear.x, this->vel.linear.y, this->vel.linear.z}});
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        // Always feed the low-level policy with the active control command.
        // In navigation mode, KeyboardInterface already overwrites control.{x,y,yaw} with the high-level outputs.
        this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        this->obs.actions = this->Forward();
        {
            std::lock_guard<std::mutex> lock(this->nav_last_actions_mutex_);
            this->nav_last_actions_.resize(this->params.num_of_dofs);
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                this->nav_last_actions_[i] = this->obs.actions[0][i].item<float>();
            }
        }
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::zeros({1, this->params.num_of_dofs});
        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            tau_est[0][i] = this->joint_efforts[this->params.joint_controller_names[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Sim::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (this->params.observations_history.size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        // actions = this->model.forward({this->history_obs}).toTensor();
        // torch::Tensor depth_image = depth_buffer.get_depth_vec();
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(this->history_obs);    // [1, 45]
        std::vector<torch::jit::IValue> vision_inputs;
        // vision_inputs.push_back(depth_image);
        // torch::Tensor vision_tokens = this->vision_head.forward(vision_inputs).toTensor();
        // inputs.push_back(vision_tokens);
        actions = this->model.forward(inputs).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

bool RL_Sim::InitHierarchicalNav()
{
    const std::string robot = this->robot_name.empty() ? "go2" : this->robot_name;
    const std::string nav_dir = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot + "/navi";
    this->nav_config_path_ = nav_dir + "/config.yaml";

    YAML::Node config;
    try
    {
        config = YAML::LoadFile(this->nav_config_path_)[robot + "/navi"];
    }
    catch (const YAML::BadFile &)
    {
        std::cout << LOGGER::WARNING << "Nav config not found: " << this->nav_config_path_ << std::endl;
        this->nav_models_loaded_.store(false);
        return false;
    }
    catch (const YAML::Exception &e)
    {
        std::cout << LOGGER::WARNING << "Failed to parse nav config " << this->nav_config_path_ << ": " << e.what() << std::endl;
        this->nav_models_loaded_.store(false);
        return false;
    }

    if (!config)
    {
        std::cout << LOGGER::WARNING << "Nav config missing key '" << robot << "/navi' in " << this->nav_config_path_ << std::endl;
        this->nav_models_loaded_.store(false);
        return false;
    }

    const std::string high_name = config["high_model_name"] ? config["high_model_name"].as<std::string>() : "";
    const std::string vision_name = config["vision_model_name"] ? config["vision_model_name"].as<std::string>() : "";
    if (high_name.empty() || vision_name.empty())
    {
        std::cout << LOGGER::WARNING << "Nav config must contain 'high_model_name' and 'vision_model_name' in " << this->nav_config_path_ << std::endl;
        this->nav_models_loaded_.store(false);
        return false;
    }

    // optional params (safe defaults)
    if (config["nav_dt"]) this->nav_dt_ = config["nav_dt"].as<double>();
    if (config["nav_episode_length_s"]) this->nav_episode_length_s_ = config["nav_episode_length_s"].as<double>();
    if (config["clip_commands"]) this->nav_clip_commands_ = config["clip_commands"].as<double>();
    this->nav_timer_left_.store(this->nav_episode_length_s_);
    this->nav_time_io_.store(0.0);
    this->nav_time_io_hf_.store(0.0);

    this->nav_high_model_path_ = nav_dir + "/" + high_name;
    this->nav_vision_model_path_ = nav_dir + "/" + vision_name;

    try
    {
        this->nav_high_model_ = torch::jit::load(this->nav_high_model_path_);
        this->nav_vision_model_ = torch::jit::load(this->nav_vision_model_path_);
        this->nav_high_model_.eval();
        this->nav_vision_model_.eval();
    }
    catch (const c10::Error &e)
    {
        std::cout << LOGGER::WARNING << "Failed to load nav models: " << e.what() << std::endl;
        this->nav_models_loaded_.store(false);
        return false;
    }

    try
    {
        std::cout << LOGGER::INFO << "Nav high forward schema: " << this->nav_high_model_.get_method("forward").function().getSchema() << std::endl;
        std::cout << LOGGER::INFO << "Nav vision forward schema: " << this->nav_vision_model_.get_method("forward").function().getSchema() << std::endl;
    }
    catch (...)
    {
    }

    // training-aligned dims (go2)
    const int dof = this->params.num_of_dofs; // 12
    const int hf_dim = 1 + 3 + 3 + dof + dof + dof ;
    const int obs_dim = 3 + 3 + 3 + 1 + 3 + 3 + dof + dof + dof;
    const int obs_io_dim = 3 + 3 + 1 + 3 + 3 + dof + dof + dof;

    this->nav_highfreq_buf_ = ObservationBuffer(1, {hf_dim}, this->nav_highfreq_hist_len_, "time");
    this->nav_obs_hist_buf_ = ObservationBuffer(1, {obs_dim}, this->nav_obs_hist_len_, "time");
    this->nav_obs_io_hist_buf_ = ObservationBuffer(1, {obs_io_dim}, this->nav_obs_io_hist_len_, "time");

    this->nav_position_targets_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_high_command_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_last_actions_ = std::vector<float>(dof, 0.0f);

    this->nav_models_loaded_.store(true);
    return true;
}

void RL_Sim::UpdateHighFrequencyObs()
{
    if (!this->nav_models_loaded_.load())
    {
        return;
    }

    const double t = this->nav_time_io_hf_.load() + this->params.dt;
    this->nav_time_io_hf_.store(t);

    const int dof = this->params.num_of_dofs;
    torch::Tensor time_io = torch::tensor({{static_cast<float>(t)}});
    torch::Tensor base_ang_vel = torch::tensor({{
        static_cast<float>(this->robot_state.imu.gyroscope[0]),
        static_cast<float>(this->robot_state.imu.gyroscope[1]),
        static_cast<float>(this->robot_state.imu.gyroscope[2]),
    }}) * static_cast<float>(this->params.ang_vel_scale);
    torch::Tensor base_quat = torch::tensor({{
        static_cast<float>(this->robot_state.imu.quaternion[0]),
        static_cast<float>(this->robot_state.imu.quaternion[1]),
        static_cast<float>(this->robot_state.imu.quaternion[2]),
        static_cast<float>(this->robot_state.imu.quaternion[3]),
    }});
    torch::Tensor gravity_vec = torch::tensor({{0.0f, 0.0f, -1.0f}});
    torch::Tensor projected_gravity = this->QuatRotateInverse(base_quat, gravity_vec);

    torch::Tensor dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, dof).unsqueeze(0).to(torch::kFloat32);
    torch::Tensor dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, dof).unsqueeze(0).to(torch::kFloat32);
    torch::Tensor dof_pos_term = (dof_pos - this->params.default_dof_pos) * static_cast<float>(this->params.dof_pos_scale);
    torch::Tensor dof_vel_term = dof_vel * static_cast<float>(this->params.dof_vel_scale);

    torch::Tensor actions = torch::zeros({1, dof}, torch::dtype(torch::kFloat32));
    {
        std::lock_guard<std::mutex> lock(this->nav_last_actions_mutex_);
        if (static_cast<int>(this->nav_last_actions_.size()) == dof)
        {
            for (int i = 0; i < dof; ++i)
            {
                actions[0][i] = this->nav_last_actions_[i];
            }
        }
    }

    torch::Tensor hf = torch::cat({time_io, base_ang_vel, projected_gravity, dof_pos_term, dof_vel_term, actions}, 1);
    {
        std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
        this->nav_highfreq_buf_.insert(hf);
    }
}

	void RL_Sim::RunHighLevel()
	{
    if (!this->nav_models_loaded_.load() || !this->nav_enabled_.load())
    {
        return;
    }
    if (!this->nav_has_goal_.load() || !this->rl_init_done)
    {
        return;
    }

    const uint64_t goal_seq = this->nav_goal_seq_.load();
    const bool new_goal = (goal_seq != this->nav_active_goal_seq_.load());

	    const double goal_body_x_initial = this->nav_goal_body_x_.load();
	    const double goal_body_y_initial = this->nav_goal_body_y_.load();
	    const double goal_body_yaw_initial = this->nav_goal_body_yaw_.load();

		    if (new_goal)
		    {
		        this->nav_active_goal_seq_.store(goal_seq);
		        this->nav_timer_left_.store(this->nav_episode_length_s_);
			        this->nav_time_io_.store(0.0);
			        this->nav_time_io_hf_.store(0.0);
		        this->nav_goal_world_valid_.store(false);
	        this->nav_high_command_.zero_();
	        this->nav_cmd_x_.store(0.0);
	        this->nav_cmd_y_.store(0.0);
	        this->nav_cmd_yaw_.store(0.0);

	        // Initialize per-episode goal and spawn inputs (body frame).
	        this->nav_position_targets_body_initial_ = torch::tensor({{
	            static_cast<float>(goal_body_x_initial),
	            static_cast<float>(goal_body_y_initial),
	            static_cast<float>(goal_body_yaw_initial),
	        }});
		        this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
		    }

	    // For evaluation only: latch goal in world (once) if we have world pose and haven't latched yet.
	    // This does NOT affect model inputs (which use goal_body_initial).
	    if (!this->nav_goal_world_valid_.load() && this->nav_base_world_valid_.load())
	    {
	        const double rx = this->nav_base_world_x_.load();
	        const double ry = this->nav_base_world_y_.load();
	        const double ryaw = this->nav_base_world_yaw_.load();
	        const double c = std::cos(ryaw);
	        const double s = std::sin(ryaw);
	        const double gx = rx + c * goal_body_x_initial - s * goal_body_y_initial;
	        const double gy = ry + s * goal_body_x_initial + c * goal_body_y_initial;
	        const double gyaw = WrapToPi(ryaw + goal_body_yaw_initial);
	        this->nav_goal_world_x_.store(gx);
	        this->nav_goal_world_y_.store(gy);
	        this->nav_goal_world_yaw_.store(gyaw);
	        this->nav_goal_world_valid_.store(true);
	    }

	    // For evaluation only: compute live goal in body frame from latched world goal and current world pose.
	    bool goal_body_live_ok = false;
	    double goal_body_live_x = 0.0;
	    double goal_body_live_y = 0.0;
	    double goal_body_live_yaw = 0.0;
	    if (this->nav_goal_world_valid_.load() && this->nav_base_world_valid_.load())
	    {
	        const double rx = this->nav_base_world_x_.load();
	        const double ry = this->nav_base_world_y_.load();
	        const double ryaw = this->nav_base_world_yaw_.load();
	        const double gx = this->nav_goal_world_x_.load();
	        const double gy = this->nav_goal_world_y_.load();
	        const double gyaw = this->nav_goal_world_yaw_.load();

	        const double dx = gx - rx;
	        const double dy = gy - ry;
	        const double c = std::cos(-ryaw);
	        const double s = std::sin(-ryaw);
	        goal_body_live_x = dx * c - dy * s;
	        goal_body_live_y = dx * s + dy * c;
	        goal_body_live_yaw = WrapToPi(gyaw - ryaw);
	        goal_body_live_ok = true;
	    }

		    const double timer_left = this->nav_timer_left_.load();
		    const double timer_norm = std::max(0.0, timer_left) / std::max(1e-6, this->nav_episode_length_s_);
			    const double time_io = this->nav_time_io_.load();

    torch::Tensor timer_tensor = torch::tensor({{static_cast<float>(timer_norm)}});
    torch::Tensor time_io_tensor = torch::tensor({{static_cast<float>(time_io)}});

    const int dof = this->params.num_of_dofs;
    torch::Tensor base_ang_vel, projected_gravity, dof_pos_term, dof_vel_term;
    {
        std::lock_guard<std::mutex> lock(this->nav_state_mutex_);
        torch::Tensor base_quat = torch::tensor({{
            static_cast<float>(this->robot_state.imu.quaternion[0]),
            static_cast<float>(this->robot_state.imu.quaternion[1]),
            static_cast<float>(this->robot_state.imu.quaternion[2]),
            static_cast<float>(this->robot_state.imu.quaternion[3]),
        }});
        torch::Tensor gravity_vec = torch::tensor({{0.0f, 0.0f, -1.0f}});
        projected_gravity = this->QuatRotateInverse(base_quat, gravity_vec);
        base_ang_vel = torch::tensor({{
            static_cast<float>(this->robot_state.imu.gyroscope[0]),
            static_cast<float>(this->robot_state.imu.gyroscope[1]),
            static_cast<float>(this->robot_state.imu.gyroscope[2]),
        }}) * static_cast<float>(this->params.ang_vel_scale);

        torch::Tensor dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, dof).unsqueeze(0).to(torch::kFloat32);
        torch::Tensor dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, dof).unsqueeze(0).to(torch::kFloat32);
        dof_pos_term = (dof_pos - this->params.default_dof_pos) * static_cast<float>(this->params.dof_pos_scale);
        dof_vel_term = dof_vel * static_cast<float>(this->params.dof_vel_scale);
    }

    torch::Tensor actions = torch::zeros({1, dof}, torch::dtype(torch::kFloat32));
    {
        std::lock_guard<std::mutex> lock(this->nav_last_actions_mutex_);
        if (static_cast<int>(this->nav_last_actions_.size()) == dof)
        {
            for (int i = 0; i < dof; ++i)
            {
                actions[0][i] = this->nav_last_actions_[i];
            }
        }
    }
    torch::Tensor high_command_scaled = this->nav_high_command_ * this->params.commands_scale.to(torch::kFloat32);

    torch::Tensor obs_frame = torch::cat({
        this->nav_position_targets_body_initial_.to(torch::kFloat32),
        this->nav_spawn_positions_body_initial_.to(torch::kFloat32),
        high_command_scaled,
        timer_tensor,
        base_ang_vel,
        projected_gravity,
        dof_pos_term,
        dof_vel_term,
        actions,
    }, 1);

    torch::Tensor obs_io_frame = torch::cat({
        this->nav_position_targets_body_initial_.to(torch::kFloat32),
        this->nav_spawn_positions_body_initial_.to(torch::kFloat32),
        time_io_tensor,
        base_ang_vel,
        projected_gravity,
        dof_pos_term,
        dof_vel_term,
        actions,
    }, 1);

    torch::Tensor obs_io_frame_hf = torch::cat({
        time_io_tensor,
        base_ang_vel,
        projected_gravity,
        dof_pos_term,
        dof_vel_term,
        actions,
    }, 1);

    if (new_goal)
    {
        this->nav_obs_hist_buf_.reset_from_0({0}, obs_frame);
        this->nav_obs_io_hist_buf_.reset_from_0({0}, obs_io_frame);
        this->nav_highfreq_buf_.reset({0}, obs_io_frame_hf);
    }
    else{
        this->nav_obs_hist_buf_.insert(obs_frame);
        this->nav_obs_io_hist_buf_.insert(obs_io_frame);}
    
    std::vector<int> obs_ids_10;
    obs_ids_10.reserve(this->nav_obs_hist_len_);
    for (int i = this->nav_obs_hist_len_ - 1; i >= 0; --i) obs_ids_10.push_back(i);
    std::vector<int> obs_ids_20;
    obs_ids_20.reserve(this->nav_highfreq_hist_len_);
    for (int i = this->nav_highfreq_hist_len_ - 1; i >= 0; --i) obs_ids_20.push_back(i);

    torch::Tensor obs_hist = this->nav_obs_hist_buf_.get_obs_vec(obs_ids_10);
    torch::Tensor obs_io_hist = this->nav_obs_io_hist_buf_.get_obs_vec(obs_ids_10);
    torch::Tensor hf_hist;
    {
        std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
        hf_hist = this->nav_highfreq_buf_.get_obs_vec(obs_ids_20);
    }

    torch::Tensor vision_feat;
    try
    {
        torch::Tensor depth = depth_buffer.get_depth_vec().to(torch::kFloat32);
        vision_feat = this->nav_vision_model_.forward({depth}).toTensor();
    }
    catch (...)
    {
        vision_feat = torch::zeros({1, 0}, torch::dtype(torch::kFloat32));
    }

	    torch::Tensor cmd;
	    torch::Tensor pred_target_body;
	    
	    torch::jit::IValue out;
	    try
	    {
	        static int nav_input_print_count = 0;
	        if (nav_input_print_count < 5)
	        {
	            auto print_sizes = [](const c10::IntArrayRef &sizes)
	            {
	                std::cout << "[";
	                for (size_t i = 0; i < sizes.size(); ++i)
	                {
	                    std::cout << sizes[i] << (i + 1 < sizes.size() ? ", " : "");
	                }
	                std::cout << "]";
	            };

	            // std::cout << "\n[nav] high-level inputs dump #" << nav_input_print_count << std::endl;
	            // std::cout << "[nav] obs_hist sizes=";
	            // print_sizes(obs_hist.sizes());
	            // std::cout << "\n" << obs_hist << std::endl;

		            // std::cout << "[nav] obs_io_hist sizes=";
		            // print_sizes(obs_io_hist.sizes());
		            // std::cout << "\n" << obs_io_hist << std::endl;

		            std::cout << "[nav] hf_hist sizes=";
		            print_sizes(hf_hist.sizes());
		            std::cout << std::endl;

		            // hf_hist is concatenated over nav_highfreq_hist_len_ frames.
		            // Print only the first 7 dims of each frame:
		            //   [time_io, ang_vel(3), projected_gravity(3)]
		            const int hf_len = this->nav_highfreq_hist_len_;
		            const int hf_dim = static_cast<int>(obs_io_frame_hf.size(1));
		            torch::Tensor hf_mat = hf_hist.reshape({hf_len, hf_dim}).to(torch::kCPU);
		            std::cout << "[nav] hf_hist first7 per frame (oldest->newest):" << std::endl;
		            for (int t = 0; t < hf_len; ++t)
		            {
		                std::cout << "  [" << t << "] ";
		                for (int j = 0; j < 7; ++j)
		                {
		                    std::cout << hf_mat[t][j].item<float>() << (j + 1 < 7 ? " " : "");
		                }
		                std::cout << std::endl;
		            }

		            ++nav_input_print_count;
		        }

	        std::vector<torch::jit::IValue> inputs = {obs_frame, obs_hist, obs_io_hist, vision_feat, hf_hist};
	        out = this->nav_high_model_.forward(inputs);
	    }
	    catch (const c10::Error &e)
	    {
	        std::cout << LOGGER::WARNING << "Nav high forward failed: " << e.what() << std::endl;
	        return;
	    }

    // Unpack output: some models return a Tensor, some return a tuple/list:
    //   (actions, v, target_pos, spawn_pos)
    try
    {
        if (out.isTensor())
        {
            cmd = out.toTensor();
        }
        else if (out.isTuple())
        {
            auto elems = out.toTuple()->elements();
            if (!elems.empty() && elems[0].isTensor())
            {
                cmd = elems[0].toTensor();
            }
            if (elems.size() > 2 && elems[2].isTensor())
            {
                pred_target_body = elems[2].toTensor();
            }
        }
        else if (out.isList())
        {
            auto list = out.toList();
            if (list.size() > 0 && list.get(0).isTensor())
            {
                cmd = list.get(0).toTensor();
            }
            if (list.size() > 2 && list.get(2).isTensor())
            {
                pred_target_body = list.get(2).toTensor();
            }
        }
    }
    catch (...)
    {
        return;
    }

    if (!cmd.defined() || cmd.numel() < 3)
    {
        return;
    }

    const torch::Tensor cmd_raw = cmd.to(torch::kFloat32);

    // clip + momentum smoothing
    cmd = torch::clamp(cmd_raw, -static_cast<float>(this->nav_clip_commands_), static_cast<float>(this->nav_clip_commands_));

    
        static int dbg_tick = 0;
	    dbg_tick = (dbg_tick + 1) % 10; 
	    if (dbg_tick == 0 || new_goal)
	    {
	        const double rx = cmd_raw[0][0].item<double>();
        const double ry = cmd_raw[0][1].item<double>();
        const double rz = cmd_raw[0][2].item<double>();
	        const double cx = cmd[0][0].item<double>();
	        const double cy = cmd[0][1].item<double>();
	        const double cz = cmd[0][2].item<double>();
			        std::cout << LOGGER::INFO
			                  << "NavHigh raw:[" << rx << ", " << ry << ", " << rz << "]"
			                  << " clipped:[" << cx << ", " << cy << ", " << cz << "]"
			                  << " goal_body_initial:[" << this->nav_position_targets_body_initial_[0][0].item<double>()
			                  << ", " << this->nav_position_targets_body_initial_[0][1].item<double>()
			                  << ", " << this->nav_position_targets_body_initial_[0][2].item<double>() << "]"
			                  << std::endl;

			        if (!goal_body_live_ok)
			        {
			            static std::atomic<bool> warned{false};
			            if (!warned.exchange(true))
			            {
			                std::cout << LOGGER::WARNING
	                          << "No world pose received; goal_body_live (evaluation) is unavailable."
	                          << std::endl;
	            }
	        }
	    }

	    if (pred_target_body.defined() && pred_target_body.numel() >= 2)
	    {
	        const double tx = pred_target_body[0][0].item<double>();
	        const double ty = pred_target_body[0][1].item<double>();
	        const double tyaw = (pred_target_body.numel() >= 3) ? pred_target_body[0][2].item<double>() : 0.0;

	        if (dbg_tick == 0 || new_goal)
	        {
	            std::cout << LOGGER::INFO
	                      << " NavPred body(robot_model):[" << tx << ", " << ty << ", " << tyaw << "]";
	            if (goal_body_live_ok)
	            {
	                std::cout << " goal_body_live:[" << goal_body_live_x << ", " << goal_body_live_y << ", " << goal_body_live_yaw << "]";
	            }
	            else
	            {
	                std::cout << " goal_body_live:[NA]";
	            }
	            std::cout << std::endl;
	        }

        // Update markers at the same rate as high-level inference (typically 10Hz).
        this->UpdateNavPredMarker(tx, ty, tyaw);
    }

	    this->nav_cmd_x_.store(cmd[0][0].item<double>());
	    this->nav_cmd_y_.store(cmd[0][1].item<double>());
	    this->nav_cmd_yaw_.store(cmd[0][2].item<double>());
	    this->nav_high_command_ = cmd.to(torch::kFloat32);

		    this->nav_timer_left_.store(timer_left - this->nav_dt_);
		    this->nav_time_io_.store(time_io + this->nav_dt_);
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.joint_controller_names[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Sim rl_sar;
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Sim>());
    rclcpp::shutdown();
#endif
    return 0;
}
