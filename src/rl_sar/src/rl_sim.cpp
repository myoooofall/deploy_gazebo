/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"

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

    // hierarchical navigation: odometry + goal
    this->odom_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::SystemDefaultsQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->OdomCallback(msg); }
    );
    this->nav_goal_subscriber = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/nav_goal", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { this->NavGoalCallback(msg); }
    );

    // service
    this->gazebo_pause_physics_client = this->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = this->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = this->create_client<std_srvs::srv::Empty>("/reset_world");

    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // init hierarchical nav policy (best-effort; safe to fail)
    this->InitHierarchicalNav();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Sim::RunModel, this));
    this->loop_high = std::make_shared<LoopFunc>("loop_high", this->nav_dt_, std::bind(&RL_Sim::RunHighLevel, this));
    this->loop_control->start();
    this->loop_rl->start();
    this->loop_high->start();

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
    this->loop_high->shutdown();
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

void RL_Sim::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    this->nav_base_x_.store(msg->pose.pose.position.x);
    this->nav_base_y_.store(msg->pose.pose.position.y);
    const auto &q = msg->pose.pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    this->nav_base_yaw_.store(std::atan2(siny_cosp, cosy_cosp));
    this->nav_has_odom_.store(true);
}

static double YawFromQuaternionWXYZ(double w, double x, double y, double z)
{
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

void RL_Sim::NavGoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    this->nav_goal_x_.store(msg->pose.position.x);
    this->nav_goal_y_.store(msg->pose.position.y);
    const auto &q = msg->pose.orientation;
    this->nav_goal_yaw_.store(YawFromQuaternionWXYZ(q.w, q.x, q.y, q.z));
    this->nav_has_goal_.store(true);
    this->nav_goal_seq_.fetch_add(1);
}
#endif

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        // this->obs.lin_vel = torch::tensor({{this->vel.linear.x, this->vel.linear.y, this->vel.linear.z}});
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        if (this->control.navigation_mode)
        {
            this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
        }
        else
        {
            this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        }
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
    if (config["momentum_factor"]) this->nav_momentum_ = config["momentum_factor"].as<double>();
    this->nav_timer_left_.store(this->nav_episode_length_s_);
    this->nav_time_io_.store(0.0);

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
    const int hf_dim = 1 + 3 + 3 + dof + dof + dof + 4;
    const int obs_dim = 3 + 3 + 3 + 1 + 3 + 3 + dof + dof + dof + 4;
    const int obs_io_dim = 3 + 3 + 1 + 3 + 3 + dof + dof + dof + 4;

    this->nav_highfreq_buf_ = ObservationBuffer(1, {hf_dim}, this->nav_highfreq_hist_len_, "time");
    this->nav_obs_hist_buf_ = ObservationBuffer(1, {obs_dim}, this->nav_obs_hist_len_, "time");
    this->nav_obs_io_hist_buf_ = ObservationBuffer(1, {obs_io_dim}, this->nav_obs_io_hist_len_, "time");

    this->nav_position_targets_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_high_command_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_momentum_high_command_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
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

    const double t = this->nav_time_io_.load() + this->params.dt;
    this->nav_time_io_.store(t);

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

    // no foot contact sensor in this stack: use neutral value 0.0 (equivalent to (0.5-0.5)/2)
    torch::Tensor foot_contact = torch::zeros({1, 4}, torch::dtype(torch::kFloat32));

    torch::Tensor hf = torch::cat({time_io, base_ang_vel, projected_gravity, dof_pos_term, dof_vel_term, actions, foot_contact}, 1);
    {
        std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
        this->nav_highfreq_buf_.insert(hf);
    }
}

static double WrapToPi(double angle)
{
    constexpr double kPi = 3.14159265358979323846;
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

void RL_Sim::RunHighLevel()
{
    if (!this->nav_models_loaded_.load() || !this->nav_enabled_.load())
    {
        return;
    }
    if (!this->nav_has_goal_.load() || !this->nav_has_odom_.load() || !this->rl_init_done)
    {
        return;
    }

    const uint64_t goal_seq = this->nav_goal_seq_.load();
    const bool new_goal = (goal_seq != this->nav_active_goal_seq_.load());

    const double base_x = this->nav_base_x_.load();
    const double base_y = this->nav_base_y_.load();
    const double base_yaw = this->nav_base_yaw_.load();
    const double goal_x = this->nav_goal_x_.load();
    const double goal_y = this->nav_goal_y_.load();
    const double goal_yaw = this->nav_goal_yaw_.load();

    if (new_goal)
    {
        this->nav_active_goal_seq_.store(goal_seq);
        this->nav_timer_left_.store(this->nav_episode_length_s_);
        this->nav_time_io_.store(0.0);
        this->nav_high_command_.zero_();
        this->nav_momentum_high_command_.zero_();
        this->nav_cmd_x_.store(0.0);
        this->nav_cmd_y_.store(0.0);
        this->nav_cmd_yaw_.store(0.0);

        const double dx = goal_x - base_x;
        const double dy = goal_y - base_y;
        const double c = std::cos(-base_yaw);
        const double s = std::sin(-base_yaw);
        const double goal_body_x = dx * c - dy * s;
        const double goal_body_y = dx * s + dy * c;
        const double goal_body_yaw = WrapToPi(goal_yaw - base_yaw);
        this->nav_position_targets_body_initial_ = torch::tensor({{static_cast<float>(goal_body_x), static_cast<float>(goal_body_y), static_cast<float>(goal_body_yaw)}});
        this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
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
    torch::Tensor foot_contact = torch::zeros({1, 4}, torch::dtype(torch::kFloat32));

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
        foot_contact,
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
        foot_contact,
    }, 1);

    if (new_goal)
    {
        this->nav_obs_hist_buf_.reset({0}, obs_frame);
        this->nav_obs_io_hist_buf_.reset({0}, obs_io_frame);
    }
    else
    {
        this->nav_obs_hist_buf_.insert(obs_frame);
        this->nav_obs_io_hist_buf_.insert(obs_io_frame);
    }

    // ObservationBuffer stores history in time order internally (oldest -> newest),
    // but get_obs_vec() takes indices where 0 = newest.
    // Build ids in reverse so the concatenated vector is [oldest ... newest],
    // matching training code where new frames are appended at the tail.
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
    std::vector<torch::jit::IValue> inputs4 = {obs_hist, obs_io_hist, hf_hist, vision_feat};
    std::vector<torch::jit::IValue> inputs3 = {obs_hist, obs_io_hist, hf_hist};
    std::vector<torch::jit::IValue> inputs2 = {obs_hist, vision_feat};
    bool ok = false;
    try { cmd = this->nav_high_model_.forward(inputs4).toTensor(); ok = true; } catch (const c10::Error &) {}
    if (!ok)
    {
        try { cmd = this->nav_high_model_.forward(inputs3).toTensor(); ok = true; } catch (const c10::Error &) {}
    }
    if (!ok)
    {
        try { cmd = this->nav_high_model_.forward(inputs2).toTensor(); ok = true; } catch (const c10::Error &e) {
            std::cout << LOGGER::WARNING << "Nav high forward failed: " << e.what() << std::endl;
        }
    }
    if (!ok)
    {
        return;
    }

    if (!cmd.defined() || cmd.numel() < 3)
    {
        return;
    }

    // clip + momentum smoothing
    cmd = torch::clamp(cmd, -static_cast<float>(this->nav_clip_commands_), static_cast<float>(this->nav_clip_commands_));

    this->nav_cmd_x_.store(cmd[0][0].item<double>());
    this->nav_cmd_y_.store(cmd[0][1].item<double>());
    this->nav_cmd_yaw_.store(cmd[0][2].item<double>());
    this->nav_high_command_ = cmd.to(torch::kFloat32);

    this->nav_timer_left_.store(timer_left - this->nav_dt_);
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
