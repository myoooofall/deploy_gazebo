/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_lite3.hpp"

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

RL_Real::RL_Real()
#if defined(USE_ROS2) && defined(USE_ROS)
    : rclcpp::Node("rl_real_node")
#endif
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_type = "ang_vel_world";
    this->robot_name = "lite3";
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


    // Network init
    int local_port = 43987;
    int robot_port = 43893;
    std::string robot_ip = "192.168.2.1";
    // std::string robot_ip = "127.0.0.1";
    // init robot
    this->receiver_ = new Receiver();
    this->sender_ = new Sender(robot_ip, robot_port);
    this->sender_->RobotStateInit();
    this->InitOutputs();
    this->InitControl();
    this->receiver_->StartWork();
    this->robot_data_ = &(receiver_->GetState());

    // init gamepad
    this->gamepad_ptr_ = std::make_shared<RetroidGamepad>(12121);
    this->first_flag_ = true;
    this->gamepad_ptr_->StartDataThread();

#if defined(USE_ROS2) && defined(USE_ROS)
    // Preferred source for robot world pose (optional): nav_msgs/Odometry.
    const auto odom_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    this->nav_odom_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", odom_qos,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->NavOdomCallback(msg); }
    );

    // hierarchical navigation: body-frame goal only (no odom dependency)
    this->nav_goal_body_subscriber = this->create_subscription<geometry_msgs::msg::Pose2D>(
        "/nav_goal_body", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Pose2D::SharedPtr msg) { this->NavGoalBodyCallback(msg); }
    );

    this->depth_image_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/depth/image_rect_raw", rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Real::DepthImageCallback, this, std::placeholders::_1));
    this->processed_depth_publisher = this->create_publisher<sensor_msgs::msg::Image>(
        "/camera/camera/depth/processed", rclcpp::SystemDefaultsQoS());
        depth_buffer = DepthBuffer(1, 60, 86, 3);  // 1个环境，3帧历史，最终尺寸60x86 (height=60, width=86)
#endif

    // init hierarchical nav policy (best-effort; safe to fail)
    this->InitHierarchicalNav();

    // loop
    this->loop_udpRecv = std::make_shared<LoopFunc>("loop_udpRecv", 0.002, std::bind(&RL_Real::UDPRecv, this), 3);
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_navi = std::make_shared<LoopFunc>("loop_nav", this->nav_dt_, std::bind(&RL_Real::RunHighLevel, this));
    this->loop_udpRecv->start();
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
    this->loop_navi->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    this->loop_udpRecv->shutdown();
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    this->loop_navi->shutdown();
    this->gamepad_ptr_->StopDataThread();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    this->rt_keys_ = this->gamepad_ptr_->GetKeys();
    if(this->first_flag_){
        this->rt_keys_record_ = this->rt_keys_;
        this->first_flag_ = false;
    }
    if (this->rt_keys_.A != this->rt_keys_record_.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->rt_keys_.B != this->rt_keys_record_.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->rt_keys_.X != this->rt_keys_record_.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->rt_keys_.Y != this->rt_keys_record_.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->rt_keys_.L1 != this->rt_keys_record_.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->rt_keys_.R1 != this->rt_keys_record_.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->rt_keys_.up != this->rt_keys_record_.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->rt_keys_.down != this->rt_keys_record_.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->rt_keys_.left != this->rt_keys_record_.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->rt_keys_.right != this->rt_keys_record_.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.A != this->rt_keys_record_.A)) this->control.SetGamepad(Input::Gamepad::LB_A);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.B != this->rt_keys_record_.B)) this->control.SetGamepad(Input::Gamepad::LB_B);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.X != this->rt_keys_record_.X)) this->control.SetGamepad(Input::Gamepad::LB_X);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.Y != this->rt_keys_record_.Y)) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button)) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button)) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.up != this->rt_keys_record_.up)) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.down != this->rt_keys_record_.down)) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.left != this->rt_keys_record_.left)) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.right != this->rt_keys_record_.right)) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.A != this->rt_keys_record_.A)) this->control.SetGamepad(Input::Gamepad::RB_A);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.B != this->rt_keys_record_.B)) this->control.SetGamepad(Input::Gamepad::RB_B);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.X != this->rt_keys_record_.X)) this->control.SetGamepad(Input::Gamepad::RB_X);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.Y != this->rt_keys_record_.Y)) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button)) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button)) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.up != this->rt_keys_record_.up)) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.down != this->rt_keys_record_.down)) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.left != this->rt_keys_record_.left)) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.right != this->rt_keys_record_.right)) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.R1 != this->rt_keys_record_.R1)) this->control.SetGamepad(Input::Gamepad::LB_RB);

    if (!this->nav_enabled_.load())
    {
        this->control.x = this->rt_keys_.left_axis_y;
        this->control.y = -this->rt_keys_.left_axis_x;
        this->control.yaw = -this->rt_keys_.right_axis_x;
    }
       
    float q[4];
    EulerToQuaternion(this->robot_data_->imu.angle_roll, this->robot_data_->imu.angle_pitch, this->robot_data_->imu.angle_yaw, q);

    state->imu.quaternion[0] = q[0]; // w
    state->imu.quaternion[1] = q[1]; // x
    state->imu.quaternion[2] = q[2]; // y
    state->imu.quaternion[3] = q[3]; // z

    state->imu.gyroscope[0] = this->robot_data_->imu.angular_velocity_roll;
    state->imu.gyroscope[1] = this->robot_data_->imu.angular_velocity_pitch;
    state->imu.gyroscope[2] = this->robot_data_->imu.angular_velocity_yaw;

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].position;
        state->motor_state.dq[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].velocity;
        state->motor_state.tau_est[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].torque;
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].position = command->motor_command.q[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].velocity = command->motor_command.dq[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].kp = command->motor_command.kp[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].kd = command->motor_command.kd[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].torque = command->motor_command.tau[i];
    }

    this->sender_->SendCmd(robot_joint_cmd_);
}

void RL_Real::RobotControl()
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

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        // Always feed the low-level policy with the active control command.
        // In navigation mode, RobotControl overwrites control.{x,y,yaw} with the high-level outputs.
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
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
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

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].position);
        this->plot_target_joint_pos[i].push_back(this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].position);
        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

void RL_Real::UDPRecv()
{
    if (receiver_)
    {
        robot_data_ = &(receiver_->GetState());
    }
}

void RL_Real::EulerToQuaternion(float roll, float pitch, float yaw, float q[4])
{
    roll *= M_PI / 180.0f;
    pitch *= M_PI / 180.0f;
    yaw *= M_PI / 180.0f;

    float cr = cos(roll * 0.5f);
    float sr = sin(roll * 0.5f);
    float cp = cos(pitch * 0.5f);
    float sp = sin(pitch * 0.5f);
    float cy = cos(yaw * 0.5f);
    float sy = sin(yaw * 0.5f);

    q[0] = cr * cp * cy + sr * sp * sy;  // w
    q[1] = sr * cp * cy - cr * sp * sy;  // x
    q[2] = cr * sp * cy + sr * cp * sy;  // y
    q[3] = cr * cp * sy - sr * sp * cy;  // z
}

#if defined(USE_ROS2) && defined(USE_ROS)
void RL_Real::DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 只在每个时间步更新一次深度图
    if (this->motion_time % 5 == 0) {  // 每5个时间步更新一次
        torch::Tensor processed_depth = depth_buffer.process_depth_image(msg,
            this->processed_depth_publisher);
        // processed_depth shape: [60, 86], insert函数会处理batch维度
        depth_buffer.insert(processed_depth);
        this->motion_time = 1;
    }
    this->motion_time++;
}

void RL_Real::NavOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
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

void RL_Real::NavGoalBodyCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
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

    this->UpdateNavGoalMarker(msg->x, msg->y, msg->theta);
}
#endif

void RL_Real::UpdateNavGoalMarker(double goal_body_x, double goal_body_y, double goal_body_yaw)
{
    (void)goal_body_x;
    (void)goal_body_y;
    (void)goal_body_yaw;
}

void RL_Real::UpdateNavPredMarker(double pred_body_x, double pred_body_y, double pred_body_yaw)
{
    (void)pred_body_x;
    (void)pred_body_y;
    (void)pred_body_yaw;
}

bool RL_Real::InitHierarchicalNav()
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

void RL_Real::UpdateHighFrequencyObs()
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

	void RL_Real::RunHighLevel()
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

        this->UpdateNavPredMarker(tx, ty, tyaw);
    }

	    this->nav_cmd_x_.store(cmd[0][0].item<double>());
	    this->nav_cmd_y_.store(cmd[0][1].item<double>());
	    this->nav_cmd_yaw_.store(cmd[0][2].item<double>());
	    this->nav_high_command_ = cmd.to(torch::kFloat32);

		    this->nav_timer_left_.store(timer_left - this->nav_dt_);
		    this->nav_time_io_.store(time_io + this->nav_dt_);
}


#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar;
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Real>());
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    RL_Real rl_sar;
    while (1) { sleep(10); }
#endif
    return 0;
}
