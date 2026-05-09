/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_lite3.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

static std::atomic<bool> g_depth_frame_received{false};

static void ConfigureTorchJitForRealtime()
{
    static std::once_flag once;
    std::call_once(once, []() {
        torch::jit::setGraphExecutorOptimize(false);
        torch::jit::setTensorExprFuserEnabled(false);
        torch::jit::overrideCanFuseOnCPULegacy(false);
        torch::jit::getProfilingMode() = false;
        torch::jit::getExecutorMode() = false;
        std::cout << LOGGER::INFO
                  << "[NAV][TORCH] disabled JIT optimize/fuser/profiling executor (stability mode)"
                  << std::endl;
    });
}

static void ConfigureTorchThreadingForRealtime()
{
    // Keep CPU headroom for control/ROS threads on NX while nav models run on GPU.
    constexpr int kTorchIntraOpThreads = 2;
    constexpr int kTorchInterOpThreads = 1;
    torch::set_num_threads(kTorchIntraOpThreads);
    torch::set_num_interop_threads(kTorchInterOpThreads);
}

static double WrapToPi(double a)
{
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static std::string TensorShapeToString(const torch::Tensor &tensor)
{
    if (!tensor.defined())
    {
        return "undefined";
    }
    std::ostringstream oss;
    oss << "[";
    for (int64_t i = 0; i < tensor.dim(); ++i)
    {
        if (i > 0) oss << "x";
        oss << tensor.size(i);
    }
    oss << "]";
    return oss.str();
}

static std::string TensorFlatToString(const torch::Tensor &tensor, int max_elems = -1, int precision = 4)
{
    if (!tensor.defined())
    {
        return "undefined";
    }
    torch::Tensor flat = tensor.to(torch::kCPU, torch::kFloat32).contiguous().view({-1});
    const int64_t n = flat.size(0);
    int64_t shown = n;
    if (max_elems > 0 && static_cast<int64_t>(max_elems) < n)
    {
        shown = static_cast<int64_t>(max_elems);
    }
    const float *ptr = flat.data_ptr<float>();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << "[";
    for (int64_t i = 0; i < shown; ++i)
    {
        if (i > 0) oss << ", ";
        oss << ptr[i];
    }
    if (shown < n)
    {
        oss << ", ... (" << n << " elems)";
    }
    oss << "]";
    return oss.str();
}

static bool EnsureDirExistsRecursive(const std::string &dir, std::string *err_msg)
{
    if (dir.empty())
    {
        if (err_msg) *err_msg = "empty directory path";
        return false;
    }

    std::string current;
    if (!dir.empty() && dir.front() == '/')
    {
        current = "/";
    }

    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/'))
    {
        if (part.empty())
        {
            continue;
        }
        if (!current.empty() && current.back() != '/')
        {
            current += "/";
        }
        current += part;

        struct stat st {};
        if (::stat(current.c_str(), &st) == 0)
        {
            if (!S_ISDIR(st.st_mode))
            {
                if (err_msg) *err_msg = "path exists but is not a directory: " + current;
                return false;
            }
            continue;
        }

        if (errno != ENOENT)
        {
            if (err_msg)
            {
                *err_msg = "stat failed at " + current + ": " + std::strerror(errno);
            }
            return false;
        }

        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
        {
            if (err_msg)
            {
                *err_msg = "mkdir failed at " + current + ": " + std::strerror(errno);
            }
            return false;
        }
    }
    return true;
}

#if defined(USE_ROS2)
static geometry_msgs::msg::Quaternion YawToQuaternion(double yaw)
{
    geometry_msgs::msg::Quaternion q;
    const double half = yaw * 0.5;
    q.w = std::cos(half);
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(half);
    return q;
}

static double QuaternionToYaw(const geometry_msgs::msg::Quaternion &q)
{
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}
#endif

RL_Real::RL_Real()
#if defined(USE_ROS2)
    : rclcpp::Node("rl_real_node")
#endif
{
#if defined(USE_ROS1)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2)
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    // Real robot IMU angular velocity is consumed as body-frame angular velocity
    // by default (same convention as ROS2 sim path).
    this->ang_vel_type = "ang_vel_body";
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
    ConfigureTorchThreadingForRealtime();
    ConfigureTorchJitForRealtime();
    // Prefer GPU inference for navigation models; fallback to CPU if CUDA is unavailable.
    this->nav_infer_device_ = torch::Device(torch::kCPU);
    try
    {
        (void)torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        this->nav_infer_device_ = torch::Device(torch::kCUDA, 0);
    }
    catch (const c10::Error &)
    {
        this->nav_infer_device_ = torch::Device(torch::kCPU);
    }
    std::cout << LOGGER::INFO
              << "Navigation inference target device: "
              << this->nav_infer_device_.str()
              << std::endl;


    // Network init
    int local_port = 43987;
    int robot_port = 43893;
    std::string robot_ip = "192.168.1.120";
    // init robot
    this->receiver_ = new Receiver();
    this->sender_ = new Sender(robot_ip, robot_port);
    this->sender_->RobotStateInit();
    this->InitOutputs();
    this->InitControl();
    this->receiver_->StartWork();
    this->robot_data_ = &(receiver_->GetState());

    // init hierarchical nav config/models before optional runtime components.
    this->InitHierarchicalNav();

    // init gamepad only when explicitly enabled; the NX deployment normally uses keyboard/ROS goals.
    this->first_flag_ = true;
    if (this->nav_gamepad_enable_)
    {
        this->gamepad_ptr_ = std::make_shared<RetroidGamepad>(12121);
        this->gamepad_ptr_->StartDataThread();
    }

#if defined(USE_ROS2)
    // hierarchical navigation: body-frame goal only (no odom dependency)
    this->nav_goal_body_subscriber = this->create_subscription<geometry_msgs::msg::Pose2D>(
        "/nav_goal_body", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Pose2D::SharedPtr msg) { this->NavGoalBodyCallback(msg); }
    );

    this->depth_image_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/depth/image_rect_raw", rclcpp::SensorDataQoS().keep_last(1),
        std::bind(&RL_Real::DepthImageCallback, this, std::placeholders::_1));
    if (this->nav_depth_debug_publish_enable_)
    {
        this->processed_depth_publisher = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/depth/processed", rclcpp::SystemDefaultsQoS());
        this->processed_depth_norm_publisher = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/depth/processed_norm", rclcpp::SystemDefaultsQoS());
    }
    depth_buffer = DepthBuffer(1, 30, 43, this->nav_vision_channels_ + 1);

    this->sdk_imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(
        "/imu/data", rclcpp::SystemDefaultsQoS());

    if (this->nav_console_info_enable_)
    {
        std::cout << LOGGER::INFO
                  << "[NAV][DEPTH] subscribe=/camera/depth/image_rect_raw (cache_latest) "
                  << "debug_publish=" << (this->nav_depth_debug_publish_enable_ ? "ON" : "OFF")
                  << " (loop_vision@nav_dt)"
                  << '\n';
        std::cout << LOGGER::INFO
                  << "[SLAM][SDK2ROS] publish=/imu/data (sensor_msgs/Imu, source=lite3_sdk)"
                  << '\n';
    }

    this->odometry_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/Odometry", rclcpp::SystemDefaultsQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->OdomCallback(msg); });
    this->odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::SystemDefaultsQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->OdomCallback(msg); });

    this->nav_goal_actual_map_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/nav/goal_actual_map", rclcpp::SystemDefaultsQoS());
    this->nav_goal_pred_map_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/nav/goal_pred_map", rclcpp::SystemDefaultsQoS());
    this->nav_goal_compare_markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/nav/goal_compare_markers", rclcpp::SystemDefaultsQoS());
    this->nav_goal_error_body_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "/nav/goal_error_body", rclcpp::SystemDefaultsQoS());
    this->nav_cmd_high_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "/nav/cmd_high", rclcpp::SystemDefaultsQoS());
    this->nav_cmd_applied_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "/nav/cmd_applied", rclcpp::SystemDefaultsQoS());

    if (this->nav_console_info_enable_)
    {
        std::cout << LOGGER::INFO
                  << "[NAV][COMPARE] subscribe={/Odometry,/odom} publish={/nav/goal_actual_map,/nav/goal_pred_map,/nav/goal_compare_markers,/nav/goal_error_body,/nav/cmd_high,/nav/cmd_applied} (fallback: body-frame on same goal topics when both odom topics are absent)"
                  << '\n';
    }
#endif

    // loop
    this->loop_udpRecv = std::make_shared<LoopFunc>("loop_udpRecv", 0.002, std::bind(&RL_Real::UDPRecv, this), 3);
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::HandleKeyboard, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
#if defined(USE_ROS2)
    this->loop_vision = std::make_shared<LoopFunc>("loop_vision", this->nav_dt_, std::bind(&RL_Real::RunVision, this));
#endif
    this->loop_navi = std::make_shared<LoopFunc>("loop_nav", this->nav_dt_, std::bind(&RL_Real::RunHighLevel, this));
    this->loop_udpRecv->start();
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
#if defined(USE_ROS2)
    this->loop_vision->start();
#endif
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
#if defined(USE_ROS2)
    this->loop_vision->shutdown();
#endif
    this->loop_navi->shutdown();
    if (this->gamepad_ptr_)
    {
        this->gamepad_ptr_->StopDataThread();
    }
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    this->StopNavObsLogIfNeeded();
    this->StopPerfCsvIfNeeded();
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    if (this->nav_gamepad_enable_ && this->gamepad_ptr_)
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
        if (this->rt_keys_.L2 != this->rt_keys_record_.L2) this->control.SetGamepad(Input::Gamepad::L2);
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
        if (this->rt_keys_.L1 && this->rt_keys_.R1) this->control.SetGamepad(Input::Gamepad::LB_RB);

        if (!this->nav_enabled_.load())
        {
            // Allow keyboard and joystick to coexist:
            // only override command with joystick when the stick leaves deadzone.
            constexpr double kStickDeadzone = 0.08;
            const double joy_x = static_cast<double>(this->rt_keys_.left_axis_y);
            const double joy_y = -static_cast<double>(this->rt_keys_.left_axis_x);
            const double joy_yaw = -static_cast<double>(this->rt_keys_.right_axis_x);
            const bool joy_active = (std::fabs(joy_x) > kStickDeadzone) ||
                                    (std::fabs(joy_y) > kStickDeadzone) ||
                                    (std::fabs(joy_yaw) > kStickDeadzone);
            if (joy_active)
            {
                // Map normalized joystick input [-1, 1] to configured command range.
                this->control.x = joy_x * this->params.cmd_clip_x;
                this->control.y = joy_y * this->params.cmd_clip_y;
                this->control.yaw = joy_yaw * this->params.cmd_clip_yaw;
                this->joystick_override_active_ = true;
            }
            else if (this->joystick_override_active_)
            {
                // Stick returned to neutral after taking control: clear once to avoid stale velocity.
                this->control.x = 0.0;
                this->control.y = 0.0;
                this->control.yaw = 0.0;
                this->joystick_override_active_ = false;
            }
        }
        else
        {
            this->joystick_override_active_ = false;
        }
        this->rt_keys_record_ = this->rt_keys_;
    }
    else
    {
        this->joystick_override_active_ = false;
    }
       
    float q[4];
    EulerToQuaternion(this->robot_data_->imu.angle_roll, this->robot_data_->imu.angle_pitch, this->robot_data_->imu.angle_yaw, q);

    state->imu.quaternion[0] = q[0]; // w
    state->imu.quaternion[1] = q[1]; // x
    state->imu.quaternion[2] = q[2]; // y
    state->imu.quaternion[3] = q[3]; // z

    // SDK attitude angles are in degrees; convert them to radians for trigonometry.
    // On this deployment, SDK angular_velocity_{roll,pitch,yaw} is treated as rad/s.
    // When nav_sdk_gyro_is_euler_rate_ is enabled, they are interpreted as ZYX Euler
    // rates [roll_dot, pitch_dot, yaw_dot] and mapped to body gyro [wx, wy, wz].
    constexpr double kDeg2Rad = M_PI / 180.0;
    const double roll = static_cast<double>(this->robot_data_->imu.angle_roll) * kDeg2Rad;
    const double pitch = static_cast<double>(this->robot_data_->imu.angle_pitch) * kDeg2Rad;
    const double roll_rate = static_cast<double>(this->robot_data_->imu.angular_velocity_roll);
    const double pitch_rate = static_cast<double>(this->robot_data_->imu.angular_velocity_pitch);
    const double yaw_rate = static_cast<double>(this->robot_data_->imu.angular_velocity_yaw);

    if (this->nav_sdk_gyro_is_euler_rate_)
    {
        // ZYX Euler rates -> body angular velocity
        // [phi_dot, theta_dot, psi_dot] = E(roll,pitch) * [wx, wy, wz]
        // => [wx, wy, wz] = E^{-1} * [phi_dot, theta_dot, psi_dot]
        const double cr = std::cos(roll);
        const double sr = std::sin(roll);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);
        state->imu.gyroscope[0] = roll_rate - sp * yaw_rate;
        state->imu.gyroscope[1] = cr * pitch_rate + sr * cp * yaw_rate;
        state->imu.gyroscope[2] = -sr * pitch_rate + cr * cp * yaw_rate;
    }
    else
    {
        state->imu.gyroscope[0] = roll_rate;
        state->imu.gyroscope[1] = pitch_rate;
        state->imu.gyroscope[2] = yaw_rate;
    }

    if (this->nav_console_info_enable_ && this->nav_debug_enable_)
    {
        static auto last_imu_debug_tp = std::chrono::steady_clock::time_point{};
        const auto now_tp = std::chrono::steady_clock::now();
        if (last_imu_debug_tp.time_since_epoch().count() == 0 ||
            std::chrono::duration<double>(now_tp - last_imu_debug_tp).count() >= this->nav_debug_log_interval_s_)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "[NAV][CHK][IMU] mode_euler_rate="
                << (this->nav_sdk_gyro_is_euler_rate_ ? 1 : 0)
                << " rpy_deg=[" << this->robot_data_->imu.angle_roll << ", "
                << this->robot_data_->imu.angle_pitch << ", "
                << this->robot_data_->imu.angle_yaw << "]"
                << " raw_rpy_rate_sdk=[" << this->robot_data_->imu.angular_velocity_roll << ", "
                << this->robot_data_->imu.angular_velocity_pitch << ", "
                << this->robot_data_->imu.angular_velocity_yaw << "]"
                << " gyro_body_rad_s=[" << state->imu.gyroscope[0] << ", "
                << state->imu.gyroscope[1] << ", "
                << state->imu.gyroscope[2] << "]";
            std::cout << LOGGER::INFO << oss.str() << '\n';
            last_imu_debug_tp = now_tp;
        }
    }

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

void RL_Real::HandleKeyboard()
{
    if (this->nav_goal_input_active_.load())
    {
        static auto last_hint_tp = std::chrono::steady_clock::time_point{};
        const auto now_tp = std::chrono::steady_clock::now();
        if (last_hint_tp.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_hint_tp).count() >= 1)
        {
            std::cout << LOGGER::INFO
                      << "[NAV][INPUT] waiting... format: x y yaw(rad), then Enter"
                      << std::endl;
            last_hint_tp = now_tp;
        }
        return;
    }

    this->KeyboardInterface();

    if (this->control.current_keyboard == Input::Keyboard::G)
    {
        std::cout << std::endl
                  << LOGGER::INFO
                  << "[NAV][INPUT] G pressed. Enter goal as: x y yaw(rad)"
                  << std::endl;
        this->StartNavGoalInput();
        this->control.current_keyboard = this->control.last_keyboard;
    }
}

void RL_Real::SetNavGoalBody(double goal_x, double goal_y, double goal_yaw, const char *source)
{
    const double yaw_wrapped = WrapToPi(goal_yaw);
    this->nav_goal_body_x_.store(goal_x);
    this->nav_goal_body_y_.store(goal_y);
    this->nav_goal_body_yaw_.store(yaw_wrapped);
    this->nav_goal_seq_.fetch_add(1);
    this->nav_enabled_.store(true);

    std::cout << LOGGER::INFO
              << "NavGoalBody(" << (source ? source : "unknown") << "): x=" << goal_x
              << " y=" << goal_y << " yaw=" << yaw_wrapped << std::endl;
}

[[noreturn]] void RL_Real::DisableNavigationWithError(const std::string &stage, const std::string &detail)
{
    const std::string msg = "[NAV][FATAL] " + stage + ": " + detail;
#if defined(USE_ROS2)
    RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
#endif
    std::cout << LOGGER::ERROR << msg << std::endl;
    throw std::runtime_error(msg);
}

void RL_Real::StartPerfCsvIfNeeded()
{
    if (!this->nav_perf_csv_enable_)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->nav_perf_csv_mutex_);
    if (this->nav_perf_csv_stream_.is_open())
    {
        return;
    }

    std::string log_dir = this->nav_perf_csv_dir_;
    if (log_dir.empty())
    {
        log_dir = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/logs/perf";
    }

    std::string mkdir_err;
    if (!EnsureDirExistsRecursive(log_dir, &mkdir_err))
    {
        log_dir = "/tmp/rl_sar_perf";
        mkdir_err.clear();
        EnsureDirExistsRecursive(log_dir, &mkdir_err);
    }
    if (!mkdir_err.empty())
    {
        std::cout << LOGGER::WARNING
                  << "[PERFCSV] failed to create log dir: " << log_dir
                  << " err=" << mkdir_err
                  << std::endl;
        this->nav_perf_csv_enable_ = false;
        return;
    }

    auto now_sys = std::chrono::system_clock::now();
    std::time_t now_tt = std::chrono::system_clock::to_time_t(now_sys);
    std::tm tm_buf{};
    localtime_r(&now_tt, &tm_buf);
    std::ostringstream ts_oss;
    ts_oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");

    std::ostringstream file_oss;
    file_oss << log_dir << "/rl_real_perf_" << ts_oss.str() << ".csv";
    this->nav_perf_csv_path_ = file_oss.str();
    this->nav_perf_csv_stream_.open(this->nav_perf_csv_path_, std::ios::out | std::ios::trunc);
    if (!this->nav_perf_csv_stream_.is_open())
    {
        std::cout << LOGGER::WARNING
                  << "[PERFCSV] failed to open: " << this->nav_perf_csv_path_
                  << std::endl;
        this->nav_perf_csv_path_.clear();
        this->nav_perf_csv_enable_ = false;
        return;
    }

    this->nav_perf_csv_start_tp_ = std::chrono::steady_clock::now();
    this->nav_perf_csv_last_flush_tp_ = this->nav_perf_csv_start_tp_;
    this->nav_perf_csv_stream_
        << "steady_time_s,event,nav_enabled,rl_init_done,"
        << "dt_ms,exec_ms,"
        << "sdk_imu_stamp_ms,sdk_imu_stamp_delta_ms,"
        << "imu_pub_dt_ms,imu_header_stamp_ms,imu_header_delta_ms,"
        << "rl_tick_ms,rl_infer_ms,rl_total_ms,"
        << "vision_tick_ms,vision_process_ms,"
        << "nav_tick_ms,nav_vision_ms,nav_high_ms,nav_total_ms\n";
    this->nav_perf_csv_stream_.flush();

    std::cout << LOGGER::INFO
              << "[PERFCSV] started path=" << this->nav_perf_csv_path_
              << std::endl;
}

void RL_Real::StopPerfCsvIfNeeded()
{
    std::lock_guard<std::mutex> lock(this->nav_perf_csv_mutex_);
    if (this->nav_perf_csv_stream_.is_open())
    {
        this->nav_perf_csv_stream_.flush();
        this->nav_perf_csv_stream_.close();
    }
}

void RL_Real::WritePerfCsvRow(
    const char *event,
    double dt_ms,
    double exec_ms,
    double sdk_imu_stamp_ms,
    double sdk_imu_stamp_delta_ms,
    double imu_pub_dt_ms,
    double imu_header_stamp_ms,
    double imu_header_delta_ms,
    double rl_tick_ms,
    double rl_infer_ms,
    double rl_total_ms,
    double vision_tick_ms,
    double vision_process_ms,
    double nav_tick_ms,
    double nav_vision_ms,
    double nav_high_ms,
    double nav_total_ms)
{
    if (!this->nav_perf_csv_enable_)
    {
        return;
    }

    const auto now_tp = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(this->nav_perf_csv_mutex_);
    if (!this->nav_perf_csv_stream_.is_open())
    {
        return;
    }

    const double steady_time_s = std::chrono::duration<double>(now_tp - this->nav_perf_csv_start_tp_).count();
    this->nav_perf_csv_stream_ << std::fixed << std::setprecision(6)
        << steady_time_s << ","
        << (event ? event : "unknown") << ","
        << (this->nav_enabled_.load() ? 1 : 0) << ","
        << (this->rl_init_done ? 1 : 0) << ","
        << dt_ms << ","
        << exec_ms << ","
        << sdk_imu_stamp_ms << ","
        << sdk_imu_stamp_delta_ms << ","
        << imu_pub_dt_ms << ","
        << imu_header_stamp_ms << ","
        << imu_header_delta_ms << ","
        << rl_tick_ms << ","
        << rl_infer_ms << ","
        << rl_total_ms << ","
        << vision_tick_ms << ","
        << vision_process_ms << ","
        << nav_tick_ms << ","
        << nav_vision_ms << ","
        << nav_high_ms << ","
        << nav_total_ms << "\n";

    if (std::chrono::duration<double>(now_tp - this->nav_perf_csv_last_flush_tp_).count() >= this->nav_perf_csv_flush_interval_s_)
    {
        this->nav_perf_csv_stream_.flush();
        this->nav_perf_csv_last_flush_tp_ = now_tp;
    }
}

void RL_Real::StartNavGoalInput()
{
    if (this->nav_goal_input_active_.exchange(true))
    {
        std::cout << LOGGER::WARNING << "Goal input already active." << std::endl;
        return;
    }

    std::thread([this]() {
        std::cout << LOGGER::INFO
                  << "[NAV][INPUT] Input goal in body frame: x y yaw(rad), then Enter."
                  << std::endl;

        std::string line;
        if (!std::getline(std::cin, line))
        {
            std::cout << LOGGER::WARNING << "[NAV] Read goal input failed." << std::endl;
            this->nav_goal_input_active_.store(false);
            return;
        }
        if (line.empty())
        {
            if (!std::getline(std::cin, line))
            {
                std::cout << LOGGER::WARNING << "[NAV] Empty goal input." << std::endl;
                this->nav_goal_input_active_.store(false);
                return;
            }
        }

        std::istringstream iss(line);
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
        if (!(iss >> x >> y >> yaw))
        {
            std::cout << LOGGER::WARNING
                      << "[NAV][INPUT] Invalid format. Expect: x y yaw(rad)."
                      << std::endl;
            this->nav_goal_input_active_.store(false);
            return;
        }

        this->SetNavGoalBody(x, y, yaw, "keyboard");
        this->nav_goal_input_active_.store(false);
    }).detach();
}

void RL_Real::RobotControl()
{
    this->motiontime++;
    using SteadyClock = std::chrono::steady_clock;
    static uint64_t last_nav_hl_beat_seq = 0;
    static SteadyClock::time_point last_nav_hl_beat_tp{};
    static SteadyClock::time_point last_stale_warn_tp{};
    static bool perf_has_last_tick = false;
    static SteadyClock::time_point perf_last_tick_tp;

    const auto cycle_begin_tp = SteadyClock::now();
    double tick_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_tick)
    {
        tick_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_tick_tp).count();
    }
    perf_last_tick_tp = cycle_begin_tp;
    perf_has_last_tick = true;

    if (this->nav_enabled_.load())
    {
        this->control.x = this->nav_cmd_x_.load();
        this->control.y = this->nav_cmd_y_.load();
        this->control.yaw = this->nav_cmd_yaw_.load();

        // Safety watchdog: once high-level produced at least one output, if beat seq
        // stops changing while navigation is ON, clear stale commands.
        const uint64_t beat_seq = this->nav_hl_beat_seq_.load();
        if (beat_seq > 0 && beat_seq != last_nav_hl_beat_seq)
        {
            last_nav_hl_beat_seq = beat_seq;
            last_nav_hl_beat_tp = SteadyClock::now();
        }

        // Only watchdog in RL running stage. During GetUp/GetDown/Passive, rl_init_done=false
        // and high-level heartbeat is expected to stop.
        // Require at least 2 high-level beats before watchdog is armed.
        // The first beat after a new goal can be noticeably slower (cold start),
        // and should not be treated as a stale-command fault.
        const bool can_watchdog = this->rl_init_done && beat_seq >= 2;
        if (can_watchdog && last_nav_hl_beat_tp.time_since_epoch().count() != 0)
        {
            const auto now_tp = SteadyClock::now();
            const int64_t kMaxNoBeatMs = static_cast<int64_t>(this->nav_watchdog_timeout_ms_);
            const int64_t no_beat_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - last_nav_hl_beat_tp).count();
            if (no_beat_ms > kMaxNoBeatMs)
            {
                this->control.x = 0.0;
                this->control.y = 0.0;
                this->control.yaw = 0.0;
                this->nav_cmd_x_.store(0.0);
                this->nav_cmd_y_.store(0.0);
                this->nav_cmd_yaw_.store(0.0);
                if (this->nav_high_command_.defined())
                {
                    this->nav_high_command_.zero_();
                }

                if (last_stale_warn_tp.time_since_epoch().count() == 0 ||
                    std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_stale_warn_tp).count() >= 1)
                {
                    std::cout << LOGGER::WARNING
                              << "[NAV][SAFE] high-level heartbeat stale (" << no_beat_ms
                              << "ms), command cleared"
                              << std::endl;
                    last_stale_warn_tp = now_tp;
                }
            }
        }
    }
    else
    {
        last_nav_hl_beat_seq = this->nav_hl_beat_seq_.load();
        last_nav_hl_beat_tp = SteadyClock::time_point{};
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
        const bool next_enabled = !this->nav_enabled_.load();
        this->nav_enabled_.store(next_enabled);
        if (!next_enabled)
        {
            // Stop immediately when leaving navigation mode.
            this->control.x = 0.0;
            this->control.y = 0.0;
            this->control.yaw = 0.0;
            this->nav_cmd_x_.store(0.0);
            this->nav_cmd_y_.store(0.0);
            this->nav_cmd_yaw_.store(0.0);
            if (this->nav_high_command_.defined())
            {
                this->nav_high_command_.zero_();
            }
        }
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (next_enabled ? "ON" : "OFF") << std::endl;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    {
        std::lock_guard<std::mutex> lock(this->nav_state_mutex_);
        this->GetState(&this->robot_state);
    }
#if defined(USE_ROS2)
    this->PublishSlamImuFromSdk(this->robot_state);
#endif
    this->UpdateHighFrequencyObs();

    // Command magnitude safety clamp for low-level policy inputs.
    // Applied after GetState so both keyboard and gamepad commands are constrained.
    this->control.x = std::max(-this->params.cmd_clip_x, std::min(this->params.cmd_clip_x, this->control.x));
    this->control.y = std::max(-this->params.cmd_clip_y, std::min(this->params.cmd_clip_y, this->control.y));
    this->control.yaw = std::max(-this->params.cmd_clip_yaw, std::min(this->params.cmd_clip_yaw, this->control.yaw));
#if defined(USE_ROS2)
    // Debug output: command actually fed to low-level policy/control.
    static int cmd_applied_pub_decim = 0;
    if (++cmd_applied_pub_decim >= 20)
    {
        cmd_applied_pub_decim = 0;
        if (this->nav_cmd_applied_publisher_)
        {
            geometry_msgs::msg::Vector3 msg;
            msg.x = this->control.x;
            msg.y = this->control.y;
            msg.z = this->control.yaw;
            this->nav_cmd_applied_publisher_->publish(msg);
        }
    }
#endif

    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);

    const double exec_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
    this->WritePerfCsvRow("control", tick_ms, exec_ms);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        using SteadyClock = std::chrono::steady_clock;
        static bool perf_inited = false;
        static bool perf_has_last_tick = false;
        static SteadyClock::time_point perf_last_tick_tp;
        static SteadyClock::time_point perf_last_report_tp;
        static double perf_sum_infer_ms = 0.0;
        static double perf_sum_total_ms = 0.0;
        static double perf_sum_tick_ms = 0.0;
        static double perf_max_infer_ms = 0.0;
        static double perf_max_total_ms = 0.0;
        static double perf_max_tick_ms = 0.0;
        static uint64_t perf_samples = 0;
        static uint64_t perf_tick_samples = 0;
        static uint64_t perf_over_budget = 0;
        static double perf_max_abs_roll_deg = 0.0;
        static double perf_max_abs_pitch_deg = 0.0;
        static double perf_max_cmd_norm = 0.0;

        const auto cycle_begin_tp = SteadyClock::now();
        const double loop_budget_ms = this->params.dt * this->params.decimation * 1000.0;
        double tick_ms = std::numeric_limits<double>::quiet_NaN();
        if (!perf_inited)
        {
            perf_last_report_tp = cycle_begin_tp;
            perf_inited = true;
        }

        if (perf_has_last_tick)
        {
            tick_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_tick_tp).count();
            perf_sum_tick_ms += tick_ms;
            perf_max_tick_ms = std::max(perf_max_tick_ms, tick_ms);
            perf_tick_samples += 1;
        }
        perf_last_tick_tp = cycle_begin_tp;
        perf_has_last_tick = true;

        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        // Always feed the low-level policy with the active control command.
        // In navigation mode, RobotControl overwrites control.{x,y,yaw} with the high-level outputs.
        this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        {
            const double w = this->robot_state.imu.quaternion[0];
            const double x = this->robot_state.imu.quaternion[1];
            const double y = this->robot_state.imu.quaternion[2];
            const double z = this->robot_state.imu.quaternion[3];

            const double sinr_cosp = 2.0 * (w * x + y * z);
            const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
            const double roll_rad = std::atan2(sinr_cosp, cosr_cosp);

            const double sinp = 2.0 * (w * y - z * x);
            const double pitch_rad = std::asin(std::max(-1.0, std::min(1.0, sinp)));

            const double rad2deg = 57.29577951308232;
            perf_max_abs_roll_deg = std::max(perf_max_abs_roll_deg, std::abs(roll_rad) * rad2deg);
            perf_max_abs_pitch_deg = std::max(perf_max_abs_pitch_deg, std::abs(pitch_rad) * rad2deg);
        }
        {
            const double cmd_norm = std::sqrt(this->control.x * this->control.x +
                                              this->control.y * this->control.y +
                                              this->control.yaw * this->control.yaw);
            perf_max_cmd_norm = std::max(perf_max_cmd_norm, cmd_norm);
        }

        const auto infer_begin_tp = SteadyClock::now();
        this->obs.actions = this->Forward();
        const auto infer_end_tp = SteadyClock::now();
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

        const auto cycle_end_tp = SteadyClock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end_tp - infer_begin_tp).count();
        const double total_ms = std::chrono::duration<double, std::milli>(cycle_end_tp - cycle_begin_tp).count();
        this->WritePerfCsvRow(
            "rl_model",
            tick_ms,
            total_ms,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            tick_ms,
            infer_ms,
            total_ms);
        perf_sum_infer_ms += infer_ms;
        perf_sum_total_ms += total_ms;
        perf_max_infer_ms = std::max(perf_max_infer_ms, infer_ms);
        perf_max_total_ms = std::max(perf_max_total_ms, total_ms);
        perf_samples += 1;
        if (infer_ms > loop_budget_ms)
        {
            perf_over_budget += 1;
        }

        if (std::chrono::duration<double>(cycle_end_tp - perf_last_report_tp).count() >= 1.0 && perf_samples > 0)
        {
            if (this->nav_perf_log_enable_)
            {
                const double avg_infer_ms = perf_sum_infer_ms / static_cast<double>(perf_samples);
                const double avg_total_ms = perf_sum_total_ms / static_cast<double>(perf_samples);
                const double avg_tick_ms = (perf_tick_samples > 0)
                    ? perf_sum_tick_ms / static_cast<double>(perf_tick_samples)
                    : 0.0;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << "[RL][PERF] samples=" << perf_samples
                    << " budget_ms=" << loop_budget_ms
                    << " infer_ms(avg/max)=[" << avg_infer_ms << "/" << perf_max_infer_ms << "]"
                    << " total_ms(avg/max)=[" << avg_total_ms << "/" << perf_max_total_ms << "]"
                    << " tick_ms(avg/max)=[" << avg_tick_ms << "/" << perf_max_tick_ms << "]"
                    << " infer_over_budget=" << perf_over_budget;
                std::cout << LOGGER::INFO << oss.str() << '\n';
            }

            perf_sum_infer_ms = 0.0;
            perf_sum_total_ms = 0.0;
            perf_sum_tick_ms = 0.0;
            perf_max_infer_ms = 0.0;
            perf_max_total_ms = 0.0;
            perf_max_tick_ms = 0.0;
            perf_samples = 0;
            perf_tick_samples = 0;
            perf_over_budget = 0;
            perf_max_abs_roll_deg = 0.0;
            perf_max_abs_pitch_deg = 0.0;
            perf_max_cmd_norm = 0.0;
            perf_last_report_tp = cycle_end_tp;
        }
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
    using SteadyClock = std::chrono::steady_clock;
    static bool perf_has_last_tick = false;
    static SteadyClock::time_point perf_last_tick_tp;
    static bool perf_has_last_sdk_stamp = false;
    static int64_t perf_last_sdk_stamp_ms = 0;

    const auto cycle_begin_tp = SteadyClock::now();
    double tick_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_tick)
    {
        tick_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_tick_tp).count();
    }
    perf_last_tick_tp = cycle_begin_tp;
    perf_has_last_tick = true;

    if (receiver_)
    {
        robot_data_ = &(receiver_->GetState());
    }

    double sdk_stamp_ms = std::numeric_limits<double>::quiet_NaN();
    double sdk_delta_ms = std::numeric_limits<double>::quiet_NaN();
    if (this->robot_data_ != nullptr)
    {
        const int64_t stamp_ms = static_cast<int64_t>(this->robot_data_->imu.timestamp);
        sdk_stamp_ms = static_cast<double>(stamp_ms);
        if (stamp_ms > 0 && perf_has_last_sdk_stamp)
        {
            sdk_delta_ms = static_cast<double>(stamp_ms - perf_last_sdk_stamp_ms);
        }
        if (stamp_ms > 0)
        {
            perf_last_sdk_stamp_ms = stamp_ms;
            perf_has_last_sdk_stamp = true;
        }
    }

    const double exec_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
    this->WritePerfCsvRow(
        "udp_recv",
        tick_ms,
        exec_ms,
        sdk_stamp_ms,
        sdk_delta_ms);
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

#if defined(USE_ROS2)
void RL_Real::PublishSlamImuFromSdk(const RobotState<double> &state)
{
    if (this->sdk_imu_publisher_ == nullptr || this->robot_data_ == nullptr)
    {
        return;
    }

    using SteadyClock = std::chrono::steady_clock;
    static bool perf_has_last_pub = false;
    static SteadyClock::time_point perf_last_pub_tp;
    static bool perf_has_last_header = false;
    static double perf_last_header_stamp_ms = 0.0;

    const auto cycle_begin_tp = SteadyClock::now();
    double pub_dt_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_pub)
    {
        pub_dt_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_pub_tp).count();
    }
    perf_last_pub_tp = cycle_begin_tp;
    perf_has_last_pub = true;

    sensor_msgs::msg::Imu imu_msg;
    // Use Lite3 SDK IMU timestamp (ms) so IMU/LiDAR stay in the same time domain.
    const int64_t imu_stamp_ms = static_cast<int64_t>(this->robot_data_->imu.timestamp);
    if (imu_stamp_ms > 0)
    {
        imu_msg.header.stamp.sec = static_cast<int32_t>(imu_stamp_ms / 1000);
        imu_msg.header.stamp.nanosec = static_cast<uint32_t>((imu_stamp_ms % 1000) * 1000000);
    }
    else
    {
        // Fallback to ROS clock only when SDK timestamp is unavailable.
        imu_msg.header.stamp = this->get_clock()->now();
    }
    imu_msg.header.frame_id = "base_link";

    imu_msg.orientation.w = state.imu.quaternion[0];
    imu_msg.orientation.x = state.imu.quaternion[1];
    imu_msg.orientation.y = state.imu.quaternion[2];
    imu_msg.orientation.z = state.imu.quaternion[3];

    imu_msg.angular_velocity.x = state.imu.gyroscope[0];
    imu_msg.angular_velocity.y = state.imu.gyroscope[1];
    imu_msg.angular_velocity.z = state.imu.gyroscope[2];

    imu_msg.linear_acceleration.x = static_cast<double>(this->robot_data_->imu.acc_x);
    imu_msg.linear_acceleration.y = static_cast<double>(this->robot_data_->imu.acc_y);
    imu_msg.linear_acceleration.z = static_cast<double>(this->robot_data_->imu.acc_z);

    this->sdk_imu_publisher_->publish(imu_msg);

    const double header_stamp_ms =
        static_cast<double>(imu_msg.header.stamp.sec) * 1000.0 +
        static_cast<double>(imu_msg.header.stamp.nanosec) / 1000000.0;
    double header_delta_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_header)
    {
        header_delta_ms = header_stamp_ms - perf_last_header_stamp_ms;
    }
    perf_last_header_stamp_ms = header_stamp_ms;
    perf_has_last_header = true;

    const double exec_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
    this->WritePerfCsvRow(
        "imu_pub",
        pub_dt_ms,
        exec_ms,
        static_cast<double>(imu_stamp_ms),
        header_delta_ms,
        pub_dt_ms,
        header_stamp_ms,
        header_delta_ms);

    if (this->sdk_imu_pub_started_ == false)
    {
        this->sdk_imu_pub_started_ = true;
        if (this->nav_console_info_enable_)
        {
            std::cout << LOGGER::INFO
                      << "[SLAM][SDK2ROS] first /imu/data published from Lite3 SDK, stamp_ms="
                      << imu_stamp_ms
                      << '\n';
        }
    }
}

void RL_Real::DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->depth_raw_msg_mutex_);
    this->depth_raw_latest_msg_ = msg;
}

void RL_Real::RunVision()
{
    using SteadyClock = std::chrono::steady_clock;
    static bool perf_has_last_tick = false;
    static SteadyClock::time_point perf_last_tick_tp;

    const auto cycle_begin_tp = SteadyClock::now();
    double tick_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_tick)
    {
        tick_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_tick_tp).count();
    }
    perf_last_tick_tp = cycle_begin_tp;
    perf_has_last_tick = true;

    sensor_msgs::msg::Image::SharedPtr latest_msg;
    {
        std::lock_guard<std::mutex> lock(this->depth_raw_msg_mutex_);
        latest_msg = this->depth_raw_latest_msg_;
    }

    if (!latest_msg)
    {
        const double exec_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
        this->WritePerfCsvRow(
            "vision_no_msg",
            tick_ms,
            exec_ms,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            tick_ms,
            std::numeric_limits<double>::quiet_NaN());
        return;
    }

    const auto proc_begin_tp = SteadyClock::now();
    torch::Tensor processed_depth = depth_buffer.process_depth_image(
        latest_msg,
        this->nav_depth_debug_publish_enable_ ? this->processed_depth_publisher : nullptr,
        this->nav_depth_debug_publish_enable_ ? this->processed_depth_norm_publisher : nullptr);
    {
        std::lock_guard<std::mutex> lock(this->depth_buffer_mutex_);
        // processed_depth shape: [30, 43], insert函数会处理batch维度
        depth_buffer.insert(processed_depth);
    }
    const double proc_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - proc_begin_tp).count();

    const bool first_depth_frame = !g_depth_frame_received.exchange(true);
    if (first_depth_frame)
    {
        if (this->nav_console_info_enable_)
        {
            std::cout << LOGGER::INFO
                      << "[NAV][DEPTH] first processed depth frame ready from loop_vision (source=/camera/depth/image_rect_raw)"
                      << '\n';
        }
    }

    (void)proc_ms;
    const double exec_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
    this->WritePerfCsvRow(
        "vision",
        tick_ms,
        exec_ms,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        tick_ms,
        proc_ms);
}

void RL_Real::NavGoalBodyCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
{
    this->SetNavGoalBody(msg->x, msg->y, msg->theta, "ros_topic");
}

void RL_Real::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    const auto &pose = msg->pose.pose;
    const double yaw = QuaternionToYaw(pose.orientation);

    std::lock_guard<std::mutex> lock(this->odom_pose_mutex_);
    this->odom_map_x_ = pose.position.x;
    this->odom_map_y_ = pose.position.y;
    this->odom_map_yaw_ = yaw;
    this->odom_stamp_ = msg->header.stamp;
    this->odom_pose_received_ = true;
}

bool RL_Real::TryProjectBodyTargetToMap(
    double body_x,
    double body_y,
    double body_yaw,
    double base_x,
    double base_y,
    double base_yaw,
    geometry_msgs::msg::PoseStamped *pose_out) const
{
    if (pose_out == nullptr)
    {
        return false;
    }

    const double c = std::cos(base_yaw);
    const double s = std::sin(base_yaw);
    pose_out->pose.position.x = base_x + c * body_x - s * body_y;
    pose_out->pose.position.y = base_y + s * body_x + c * body_y;
    pose_out->pose.position.z = 0.0;
    pose_out->pose.orientation = YawToQuaternion(WrapToPi(base_yaw + body_yaw));
    return true;
}

void RL_Real::PublishNavGoalComparison(
    const torch::Tensor &pred_target_body,
    uint64_t goal_seq,
    bool new_goal)
{
    double base_x = 0.0;
    double base_y = 0.0;
    double base_yaw = 0.0;
    rclcpp::Time base_stamp(0, 0, RCL_ROS_TIME);

    bool has_odom = false;
    {
        std::lock_guard<std::mutex> lock(this->odom_pose_mutex_);
        if (this->odom_pose_received_)
        {
            has_odom = true;
            base_x = this->odom_map_x_;
            base_y = this->odom_map_y_;
            base_yaw = this->odom_map_yaw_;
            base_stamp = this->odom_stamp_;
        }
    }

    if (!has_odom)
    {
        const rclcpp::Time now_stamp = this->get_clock()->now();
        const double actual_body_x = this->nav_goal_body_x_.load();
        const double actual_body_y = this->nav_goal_body_y_.load();
        const double actual_body_yaw = this->nav_goal_body_yaw_.load();

        geometry_msgs::msg::PoseStamped actual_pose;
        actual_pose.header.frame_id = "base_link";
        actual_pose.header.stamp = now_stamp;
        actual_pose.pose.position.x = actual_body_x;
        actual_pose.pose.position.y = actual_body_y;
        actual_pose.pose.position.z = 0.0;
        float q_actual[4];
        this->EulerToQuaternion(0.0f, 0.0f, static_cast<float>(actual_body_yaw), q_actual);
        actual_pose.pose.orientation.w = q_actual[0];
        actual_pose.pose.orientation.x = q_actual[1];
        actual_pose.pose.orientation.y = q_actual[2];
        actual_pose.pose.orientation.z = q_actual[3];
        this->nav_goal_actual_map_publisher_->publish(actual_pose);

        if (!pred_target_body.defined() || pred_target_body.numel() < 2)
        {
            return;
        }

        torch::Tensor pred_flat = pred_target_body.to(torch::kFloat32).view({-1});

        if (pred_flat.numel() < 2)
        {
            return;
        }

        const double pred_body_x = pred_flat[0].item<double>();
        const double pred_body_y = pred_flat[1].item<double>();
        const double pred_body_yaw = (pred_flat.numel() >= 3) ? pred_flat[2].item<double>() : 0.0;

        geometry_msgs::msg::PoseStamped pred_pose;
        pred_pose.header.frame_id = "base_link";
        pred_pose.header.stamp = now_stamp;
        pred_pose.pose.position.x = pred_body_x;
        pred_pose.pose.position.y = pred_body_y;
        pred_pose.pose.position.z = 0.0;
        float q_pred[4];
        this->EulerToQuaternion(0.0f, 0.0f, static_cast<float>(pred_body_yaw), q_pred);
        pred_pose.pose.orientation.w = q_pred[0];
        pred_pose.pose.orientation.x = q_pred[1];
        pred_pose.pose.orientation.y = q_pred[2];
        pred_pose.pose.orientation.z = q_pred[3];
        this->nav_goal_pred_map_publisher_->publish(pred_pose);

        geometry_msgs::msg::Vector3 error_body;
        error_body.x = pred_body_x - actual_body_x;
        error_body.y = pred_body_y - actual_body_y;
        error_body.z = WrapToPi(pred_body_yaw - actual_body_yaw);
        this->nav_goal_error_body_publisher_->publish(error_body);
        return;
    }

    if (new_goal || !this->nav_goal_actual_map_valid_ || this->nav_goal_actual_map_goal_seq_ != goal_seq)
    {
        geometry_msgs::msg::PoseStamped actual_pose;
        if (!this->TryProjectBodyTargetToMap(
                this->nav_goal_body_x_.load(),
                this->nav_goal_body_y_.load(),
                this->nav_goal_body_yaw_.load(),
                base_x,
                base_y,
                base_yaw,
                &actual_pose))
        {
            return;
        }

        actual_pose.header.frame_id = "map";
        actual_pose.header.stamp = base_stamp;
        this->nav_goal_actual_map_pose_ = actual_pose;
        this->nav_goal_actual_map_valid_ = true;
        this->nav_goal_actual_map_goal_seq_ = goal_seq;
    }

    if (!this->nav_goal_actual_map_valid_)
    {
        return;
    }

    this->nav_goal_actual_map_pose_.header.stamp = base_stamp;
    this->nav_goal_actual_map_publisher_->publish(this->nav_goal_actual_map_pose_);

    if (!pred_target_body.defined() || pred_target_body.numel() < 2)
    {
        static auto last_warn_tp = std::chrono::steady_clock::time_point{};
        const auto now_tp = std::chrono::steady_clock::now();
        if (last_warn_tp.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_warn_tp).count() >= 1)
        {
            std::cout << LOGGER::WARNING
                      << "[NAV][COMPARE] pred_target_body unavailable, skip compare publish"
                      << std::endl;
            last_warn_tp = now_tp;
        }
        return;
    }

    torch::Tensor pred_flat = pred_target_body.to(torch::kFloat32).view({-1});

    if (pred_flat.numel() < 2)
    {
        return;
    }

    const double pred_body_x = pred_flat[0].item<double>();
    const double pred_body_y = pred_flat[1].item<double>();
    const double pred_body_yaw = (pred_flat.numel() >= 3) ? pred_flat[2].item<double>() : 0.0;

    geometry_msgs::msg::PoseStamped pred_pose;
    if (!this->TryProjectBodyTargetToMap(
            pred_body_x,
            pred_body_y,
            pred_body_yaw,
            base_x,
            base_y,
            base_yaw,
            &pred_pose))
    {
        return;
    }
    pred_pose.header.frame_id = "map";
    pred_pose.header.stamp = base_stamp;
    this->nav_goal_pred_map_publisher_->publish(pred_pose);

    const double actual_map_yaw = QuaternionToYaw(this->nav_goal_actual_map_pose_.pose.orientation);
    const double dx_map = this->nav_goal_actual_map_pose_.pose.position.x - base_x;
    const double dy_map = this->nav_goal_actual_map_pose_.pose.position.y - base_y;
    const double c = std::cos(base_yaw);
    const double s = std::sin(base_yaw);
    const double actual_body_x = c * dx_map + s * dy_map;
    const double actual_body_y = -s * dx_map + c * dy_map;
    const double actual_body_yaw = WrapToPi(actual_map_yaw - base_yaw);

    geometry_msgs::msg::Vector3 error_body;
    error_body.x = pred_body_x - actual_body_x;
    error_body.y = pred_body_y - actual_body_y;
    error_body.z = WrapToPi(pred_body_yaw - actual_body_yaw);
    this->nav_goal_error_body_publisher_->publish(error_body);

    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker actual_marker;
    actual_marker.header.frame_id = "map";
    actual_marker.header.stamp = base_stamp;
    actual_marker.ns = "nav_goal_compare";
    actual_marker.id = 0;
    actual_marker.type = visualization_msgs::msg::Marker::SPHERE;
    actual_marker.action = visualization_msgs::msg::Marker::ADD;
    actual_marker.pose = this->nav_goal_actual_map_pose_.pose;
    actual_marker.scale.x = 0.25;
    actual_marker.scale.y = 0.25;
    actual_marker.scale.z = 0.25;
    actual_marker.color.r = 0.0f;
    actual_marker.color.g = 1.0f;
    actual_marker.color.b = 0.0f;
    actual_marker.color.a = 0.9f;
    markers.markers.push_back(actual_marker);

    visualization_msgs::msg::Marker pred_marker;
    pred_marker.header.frame_id = "map";
    pred_marker.header.stamp = base_stamp;
    pred_marker.ns = "nav_goal_compare";
    pred_marker.id = 1;
    pred_marker.type = visualization_msgs::msg::Marker::SPHERE;
    pred_marker.action = visualization_msgs::msg::Marker::ADD;
    pred_marker.pose = pred_pose.pose;
    pred_marker.scale.x = 0.22;
    pred_marker.scale.y = 0.22;
    pred_marker.scale.z = 0.22;
    pred_marker.color.r = 1.0f;
    pred_marker.color.g = 0.5f;
    pred_marker.color.b = 0.0f;
    pred_marker.color.a = 0.9f;
    markers.markers.push_back(pred_marker);

    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = base_stamp;
    line_marker.ns = "nav_goal_compare";
    line_marker.id = 2;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.06;
    line_marker.color.r = 1.0f;
    line_marker.color.g = 1.0f;
    line_marker.color.b = 1.0f;
    line_marker.color.a = 0.9f;

    geometry_msgs::msg::Point p_actual;
    p_actual.x = this->nav_goal_actual_map_pose_.pose.position.x;
    p_actual.y = this->nav_goal_actual_map_pose_.pose.position.y;
    p_actual.z = 0.05;
    geometry_msgs::msg::Point p_pred;
    p_pred.x = pred_pose.pose.position.x;
    p_pred.y = pred_pose.pose.position.y;
    p_pred.z = 0.05;
    line_marker.points.push_back(p_actual);
    line_marker.points.push_back(p_pred);
    markers.markers.push_back(line_marker);

    this->nav_goal_compare_markers_publisher_->publish(markers);
}
#endif

double RL_Real::GetNavEpisodeElapsedSec() const
{
    using SteadyClock = std::chrono::steady_clock;
    const int64_t start_ns = this->nav_episode_start_ns_.load(std::memory_order_relaxed);
    if (start_ns == 0)
    {
        return 0.0;
    }

    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now().time_since_epoch()).count();
    return std::max(0.0, static_cast<double>(now_ns - start_ns) * 1e-9);
}

void RL_Real::ResetNavEpisodeClock()
{
    using SteadyClock = std::chrono::steady_clock;
    const int64_t start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now().time_since_epoch()).count();
    this->nav_episode_start_ns_.store(start_ns, std::memory_order_relaxed);
    this->nav_time_io_.store(0.0);
    this->nav_time_io_hf_.store(0.0);
}

void RL_Real::ResetNavModelStates()
{
    try
    {
        this->nav_high_model_.get_method("reset_memory")({});
    }
    catch (const c10::Error &e)
    {
        this->DisableNavigationWithError(
            "model_reset",
            std::string("nav_high_model.reset_memory() failed: ") + e.what());
    }
    catch (const std::exception &e)
    {
        this->DisableNavigationWithError(
            "model_reset",
            std::string("nav_high_model.reset_memory() failed: ") + e.what());
    }
}

void RL_Real::WarmupNavModels(int obs_dim, int obs_io_dim, int hf_dim)
{
    torch::NoGradGuard no_grad;
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(this->nav_infer_device_);
    torch::Tensor vision_input = torch::zeros({1, this->nav_vision_channels_, 30, 43}, opts);
    torch::Tensor obs_frame = torch::zeros({1, obs_dim}, opts);
    torch::Tensor obs_io_hist = torch::zeros({1, obs_io_dim * this->nav_obs_io_hist_len_}, opts);
    torch::Tensor hf_hist = torch::zeros({1, hf_dim * this->nav_highfreq_hist_len_}, opts);

    const int warmup_iters = 20;
    for (int i = 0; i < warmup_iters; ++i)
    {
        const auto warmup_begin_tp = std::chrono::steady_clock::now();
        torch::Tensor vision_feat = this->nav_vision_model_.forward({vision_input}).toTensor();
        std::vector<torch::jit::IValue> inputs = {obs_frame, obs_io_hist, vision_feat, hf_hist};
        (void)this->nav_high_model_.forward(inputs);
        const double warmup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - warmup_begin_tp).count();
        if (this->nav_console_info_enable_)
        {
            std::cout << LOGGER::INFO
                      << "[NAV][INIT] warmup " << (i + 1) << "/" << warmup_iters
                      << " took " << std::fixed << std::setprecision(2) << warmup_ms << " ms"
                      << '\n';
        }
    }

    this->ResetNavModelStates();
    this->nav_high_command_.zero_();
    this->nav_cmd_x_.store(0.0);
    this->nav_cmd_y_.store(0.0);
    this->nav_cmd_yaw_.store(0.0);
    this->nav_hl_beat_seq_.store(0);
    this->nav_timer_left_.store(this->nav_episode_length_s_);
    this->nav_episode_start_ns_.store(0, std::memory_order_relaxed);
    this->nav_time_io_.store(0.0);
    this->nav_time_io_hf_.store(0.0);
    {
        std::lock_guard<std::mutex> lock(this->nav_last_actions_mutex_);
        std::fill(this->nav_last_actions_.begin(), this->nav_last_actions_.end(), 0.0f);
    }
    this->nav_runtime_prime_done_.store(false);
}

void RL_Real::PrimeNavRuntimeOnce()
{
    if (this->nav_runtime_prime_done_.load())
    {
        return;
    }
    if (!g_depth_frame_received.load())
    {
        return;
    }
    if (this->nav_obs_dim_ <= 0 || this->nav_obs_io_dim_ <= 0 || this->nav_hf_dim_ <= 0)
    {
        return;
    }

    torch::Tensor depth;
    {
        std::lock_guard<std::mutex> lock(this->depth_buffer_mutex_);
        depth = depth_buffer.get_depth_vec();
    }
    if (!depth.defined())
    {
        return;
    }
    depth = depth.to(torch::kFloat32);
    if (depth.dim() != 4)
    {
        return;
    }
    const int64_t channels = depth.size(1);
    if (channels != static_cast<int64_t>(this->nav_vision_channels_))
    {
        return;
    }

    try
    {
        torch::NoGradGuard no_grad;
        torch::InferenceMode infer_mode;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(this->nav_infer_device_);
        const torch::Tensor obs_frame = torch::zeros({1, this->nav_obs_dim_}, opts);
        const torch::Tensor obs_io_hist = torch::zeros({1, this->nav_obs_io_dim_ * this->nav_obs_io_hist_len_}, opts);
        const torch::Tensor hf_hist = torch::zeros({1, this->nav_hf_dim_ * this->nav_highfreq_hist_len_}, opts);
        torch::Tensor depth_model = depth;
        if (depth_model.device() != this->nav_infer_device_)
        {
            depth_model = depth_model.to(this->nav_infer_device_);
        }

        const auto begin_tp = std::chrono::steady_clock::now();
        const auto vision_begin_tp = std::chrono::steady_clock::now();
        torch::Tensor vision_feat = this->nav_vision_model_.forward({depth_model}).toTensor();
        const double vision_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - vision_begin_tp).count();

        const auto high_begin_tp = std::chrono::steady_clock::now();
        std::vector<torch::jit::IValue> inputs = {obs_frame, obs_io_hist, vision_feat, hf_hist};
        (void)this->nav_high_model_.forward(inputs);
        const double high_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - high_begin_tp).count();
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin_tp).count();

        this->ResetNavModelStates();
        this->nav_runtime_prime_done_.store(true);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "[NAV][INIT] runtime-prime done total_ms=" << total_ms
            << " vision_ms=" << vision_ms
            << " high_ms=" << high_ms;
        if (this->nav_console_info_enable_)
        {
            std::cout << LOGGER::INFO << oss.str() << '\n';
        }
    }
    catch (const c10::Error &e)
    {
        this->DisableNavigationWithError(
            "runtime_prime",
            std::string("runtime prime failed: ") + e.what());
    }
    catch (const std::exception &e)
    {
        this->DisableNavigationWithError(
            "runtime_prime",
            std::string("runtime prime failed: ") + e.what());
    }
}

void RL_Real::StartNavObsLogIfNeeded(uint64_t goal_seq)
{
    if (!this->nav_obs_log_enable_)
    {
        return;
    }
    if (this->nav_obs_log_active_ && this->nav_obs_log_goal_seq_ == goal_seq && this->nav_obs_log_stream_.is_open())
    {
        return;
    }
    this->StopNavObsLogIfNeeded();

    std::string log_dir = this->nav_obs_log_dir_;
    if (log_dir.empty())
    {
        log_dir = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/logs/nav_obs";
    }

    std::string mkdir_err;
    if (!EnsureDirExistsRecursive(log_dir, &mkdir_err))
    {
        log_dir = "/tmp/rl_sar_nav_obs";
        mkdir_err.clear();
        EnsureDirExistsRecursive(log_dir, &mkdir_err);
    }
    if (!mkdir_err.empty())
    {
        std::cout << LOGGER::WARNING
                  << "[NAV][OBSLOG] failed to create log dir: " << log_dir
                  << " err=" << mkdir_err
                  << std::endl;
        return;
    }

    auto now_sys = std::chrono::system_clock::now();
    std::time_t now_tt = std::chrono::system_clock::to_time_t(now_sys);
    std::tm tm_buf{};
    localtime_r(&now_tt, &tm_buf);
    std::ostringstream ts_oss;
    ts_oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");

    std::ostringstream file_oss;
    file_oss << log_dir
             << "/nav_obs_goal_" << goal_seq
             << "_" << ts_oss.str() << ".csv";
    this->nav_obs_log_path_ = file_oss.str();
    this->nav_obs_log_stream_.open(this->nav_obs_log_path_, std::ios::out | std::ios::trunc);
    if (!this->nav_obs_log_stream_.is_open())
    {
        std::cout << LOGGER::WARNING
                  << "[NAV][OBSLOG] failed to open: " << this->nav_obs_log_path_
                  << std::endl;
        this->nav_obs_log_path_.clear();
        return;
    }

    std::ostringstream header;
    header
        << "goal_seq,new_goal,time_io,timer_norm,"
        << "goal_init_x,goal_init_y,goal_init_yaw,"
        << "pred_body_x,pred_body_y,pred_body_yaw,"
        << "cmd_raw_x,cmd_raw_y,cmd_raw_yaw,"
        << "cmd_filtered_x,cmd_filtered_y,cmd_filtered_yaw,"
        << "prev_high_cmd_scaled_x,prev_high_cmd_scaled_y,prev_high_cmd_scaled_yaw,"
        << "imu_rpy_deg_roll,imu_rpy_deg_pitch,imu_rpy_deg_yaw,"
        << "imu_raw_rate_roll,imu_raw_rate_pitch,imu_raw_rate_yaw,"
        << "imu_gyro_body_x,imu_gyro_body_y,imu_gyro_body_z,"
        << "ang_vel_scaled_x,ang_vel_scaled_y,ang_vel_scaled_z,"
        << "proj_g_x,proj_g_y,proj_g_z";
    for (int i = 0; i < this->params.num_of_dofs; ++i) header << ",dof_pos_raw_" << i;
    for (int i = 0; i < this->params.num_of_dofs; ++i) header << ",dof_pos_term_" << i;
    for (int i = 0; i < this->params.num_of_dofs; ++i) header << ",dof_vel_raw_" << i;
    for (int i = 0; i < this->params.num_of_dofs; ++i) header << ",dof_vel_term_" << i;
    for (int i = 0; i < this->params.num_of_dofs; ++i) header << ",actions_" << i;

    this->nav_obs_log_stream_ << header.str() << "\n";
    this->nav_obs_log_stream_.flush();
    this->nav_obs_log_goal_seq_ = goal_seq;
    this->nav_obs_log_last_time_io_ = -1.0;
    this->nav_obs_log_active_ = true;

    if (this->nav_console_info_enable_)
    {
        std::cout << LOGGER::INFO
                  << "[NAV][OBSLOG] started goal_seq=" << goal_seq
                  << " path=" << this->nav_obs_log_path_
                  << " interval_s=" << this->nav_obs_log_interval_s_
                  << '\n';
    }
}

void RL_Real::StopNavObsLogIfNeeded()
{
    if (this->nav_obs_log_stream_.is_open())
    {
        this->nav_obs_log_stream_.flush();
        this->nav_obs_log_stream_.close();
    }
    if (this->nav_obs_log_active_)
    {
        if (this->nav_console_info_enable_)
        {
            std::cout << LOGGER::INFO
                      << "[NAV][OBSLOG] stopped goal_seq=" << this->nav_obs_log_goal_seq_
                      << " path=" << this->nav_obs_log_path_
                      << '\n';
        }
    }
    this->nav_obs_log_active_ = false;
    this->nav_obs_log_goal_seq_ = 0;
    this->nav_obs_log_last_time_io_ = -1.0;
    this->nav_obs_log_path_.clear();
}

void RL_Real::WriteNavObsSemanticLog(
    uint64_t goal_seq,
    bool new_goal,
    double time_io,
    double timer_norm,
    const torch::Tensor &goal_init_body,
    const torch::Tensor &pred_target_body,
    const torch::Tensor &cmd_raw,
    const torch::Tensor &cmd_filtered,
    const torch::Tensor &prev_high_cmd_scaled,
    const torch::Tensor &base_ang_vel_scaled,
    const torch::Tensor &projected_gravity,
    const torch::Tensor &dof_pos_raw,
    const torch::Tensor &dof_pos_term,
    const torch::Tensor &dof_vel_raw,
    const torch::Tensor &dof_vel_term,
    const torch::Tensor &actions)
{
    if (!this->nav_obs_log_enable_ || !this->nav_obs_log_active_ || !this->nav_obs_log_stream_.is_open())
    {
        return;
    }
    if (this->nav_obs_log_goal_seq_ != goal_seq)
    {
        return;
    }
    if (!new_goal && this->nav_obs_log_interval_s_ > 0.0 && this->nav_obs_log_last_time_io_ >= 0.0)
    {
        if ((time_io - this->nav_obs_log_last_time_io_) < this->nav_obs_log_interval_s_)
        {
            return;
        }
    }
    this->nav_obs_log_last_time_io_ = time_io;

    const auto flat_cpu = [](const torch::Tensor &t) -> torch::Tensor {
        if (!t.defined())
        {
            return torch::Tensor();
        }
        return t.to(torch::kCPU, torch::kFloat32).contiguous().view({-1});
    };
    const auto value_at = [](const torch::Tensor &flat, int64_t idx, double fallback = std::numeric_limits<double>::quiet_NaN()) -> double {
        if (!flat.defined() || idx < 0 || idx >= flat.numel())
        {
            return fallback;
        }
        return static_cast<double>(flat[idx].item<float>());
    };

    const torch::Tensor goal_init_flat = flat_cpu(goal_init_body);
    const torch::Tensor pred_flat = flat_cpu(pred_target_body);
    const torch::Tensor cmd_raw_flat = flat_cpu(cmd_raw);
    const torch::Tensor cmd_filtered_flat = flat_cpu(cmd_filtered);
    const torch::Tensor prev_high_flat = flat_cpu(prev_high_cmd_scaled);
    const torch::Tensor ang_vel_scaled_flat = flat_cpu(base_ang_vel_scaled);
    const torch::Tensor proj_g_flat = flat_cpu(projected_gravity);
    const torch::Tensor dof_pos_raw_flat = flat_cpu(dof_pos_raw);
    const torch::Tensor dof_pos_term_flat = flat_cpu(dof_pos_term);
    const torch::Tensor dof_vel_raw_flat = flat_cpu(dof_vel_raw);
    const torch::Tensor dof_vel_term_flat = flat_cpu(dof_vel_term);
    const torch::Tensor actions_flat = flat_cpu(actions);

    double imu_rpy_deg_roll = std::numeric_limits<double>::quiet_NaN();
    double imu_rpy_deg_pitch = std::numeric_limits<double>::quiet_NaN();
    double imu_rpy_deg_yaw = std::numeric_limits<double>::quiet_NaN();
    double imu_raw_rate_roll = std::numeric_limits<double>::quiet_NaN();
    double imu_raw_rate_pitch = std::numeric_limits<double>::quiet_NaN();
    double imu_raw_rate_yaw = std::numeric_limits<double>::quiet_NaN();
    if (this->robot_data_ != nullptr)
    {
        imu_rpy_deg_roll = static_cast<double>(this->robot_data_->imu.angle_roll);
        imu_rpy_deg_pitch = static_cast<double>(this->robot_data_->imu.angle_pitch);
        imu_rpy_deg_yaw = static_cast<double>(this->robot_data_->imu.angle_yaw);
        imu_raw_rate_roll = static_cast<double>(this->robot_data_->imu.angular_velocity_roll);
        imu_raw_rate_pitch = static_cast<double>(this->robot_data_->imu.angular_velocity_pitch);
        imu_raw_rate_yaw = static_cast<double>(this->robot_data_->imu.angular_velocity_yaw);
    }

    std::ostringstream row;
    row << std::fixed << std::setprecision(6)
        << goal_seq << ","
        << (new_goal ? 1 : 0) << ","
        << time_io << ","
        << timer_norm << ","
        << value_at(goal_init_flat, 0) << ","
        << value_at(goal_init_flat, 1) << ","
        << value_at(goal_init_flat, 2) << ","
        << value_at(pred_flat, 0) << ","
        << value_at(pred_flat, 1) << ","
        << value_at(pred_flat, 2) << ","
        << value_at(cmd_raw_flat, 0) << ","
        << value_at(cmd_raw_flat, 1) << ","
        << value_at(cmd_raw_flat, 2) << ","
        << value_at(cmd_filtered_flat, 0) << ","
        << value_at(cmd_filtered_flat, 1) << ","
        << value_at(cmd_filtered_flat, 2) << ","
        << value_at(prev_high_flat, 0) << ","
        << value_at(prev_high_flat, 1) << ","
        << value_at(prev_high_flat, 2) << ","
        << imu_rpy_deg_roll << ","
        << imu_rpy_deg_pitch << ","
        << imu_rpy_deg_yaw << ","
        << imu_raw_rate_roll << ","
        << imu_raw_rate_pitch << ","
        << imu_raw_rate_yaw << ","
        << this->robot_state.imu.gyroscope[0] << ","
        << this->robot_state.imu.gyroscope[1] << ","
        << this->robot_state.imu.gyroscope[2] << ","
        << value_at(ang_vel_scaled_flat, 0) << ","
        << value_at(ang_vel_scaled_flat, 1) << ","
        << value_at(ang_vel_scaled_flat, 2) << ","
        << value_at(proj_g_flat, 0) << ","
        << value_at(proj_g_flat, 1) << ","
        << value_at(proj_g_flat, 2);

    for (int i = 0; i < this->params.num_of_dofs; ++i) row << "," << value_at(dof_pos_raw_flat, i);
    for (int i = 0; i < this->params.num_of_dofs; ++i) row << "," << value_at(dof_pos_term_flat, i);
    for (int i = 0; i < this->params.num_of_dofs; ++i) row << "," << value_at(dof_vel_raw_flat, i);
    for (int i = 0; i < this->params.num_of_dofs; ++i) row << "," << value_at(dof_vel_term_flat, i);
    for (int i = 0; i < this->params.num_of_dofs; ++i) row << "," << value_at(actions_flat, i);

    this->nav_obs_log_stream_ << row.str() << "\n";
    this->nav_obs_log_stream_.flush();
}

bool RL_Real::InitHierarchicalNav()
{
    const std::string robot = this->robot_name.empty() ? "go2" : this->robot_name;
    const std::string nav_dir = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot + "/navi";
    this->nav_config_path_ = nav_dir + "/config.yaml";

    YAML::Node config = YAML::LoadFile(this->nav_config_path_)[robot + "/navi"];

    if (!config)
    {
        std::ostringstream oss;
        oss << "Nav config missing key '" << robot << "/navi' in " << this->nav_config_path_;
        throw std::runtime_error(oss.str());
    }

    const std::string high_name = config["high_model_name"] ? config["high_model_name"].as<std::string>() : "";
    const std::string vision_name = config["vision_model_name"] ? config["vision_model_name"].as<std::string>() : "";
    if (high_name.empty() || vision_name.empty())
    {
        std::ostringstream oss;
        oss << "Nav config must contain 'high_model_name' and 'vision_model_name' in " << this->nav_config_path_;
        throw std::runtime_error(oss.str());
    }

    // optional params (safe defaults)
    if (config["nav_dt"]) this->nav_dt_ = config["nav_dt"].as<double>();
    if (config["nav_episode_length_s"]) this->nav_episode_length_s_ = config["nav_episode_length_s"].as<double>();
    if (config["goal_stop_radius"])
    {
        this->nav_goal_stop_radius_ = config["goal_stop_radius"].as<double>();
    }
    this->nav_goal_stop_radius_ = std::max(0.0, this->nav_goal_stop_radius_);
    if (!config["clip_commands_vx"] || !config["clip_commands_vy"] || !config["clip_commands_w"])
    {
        std::ostringstream oss;
        oss << "Nav config must contain clip_commands_vx/clip_commands_vy/clip_commands_w in "
            << this->nav_config_path_;
        throw std::runtime_error(oss.str());
    }
    this->nav_command_clip_x_ = config["clip_commands_vx"].as<double>();
    this->nav_command_clip_y_ = config["clip_commands_vy"].as<double>();
    this->nav_command_clip_yaw_ = config["clip_commands_w"].as<double>();
    this->nav_command_clip_x_ = std::fabs(this->nav_command_clip_x_);
    this->nav_command_clip_y_ = std::fabs(this->nav_command_clip_y_);
    this->nav_command_clip_yaw_ = std::fabs(this->nav_command_clip_yaw_);

    if (config["high_command_filter_alpha"])
    {
        this->nav_high_command_filter_alpha_ = config["high_command_filter_alpha"].as<double>();
    }
    if (this->nav_high_command_filter_alpha_ < 0.0) this->nav_high_command_filter_alpha_ = 0.0;
    if (this->nav_high_command_filter_alpha_ > 1.0) this->nav_high_command_filter_alpha_ = 1.0;
    if (config["high_command_max_step_lin_x"])
    {
        this->nav_high_command_max_step_x_ = config["high_command_max_step_lin_x"].as<double>();
    }
    if (config["high_command_max_step_lin_y"])
    {
        this->nav_high_command_max_step_y_ = config["high_command_max_step_lin_y"].as<double>();
    }
    if (config["high_command_max_step_yaw"])
    {
        this->nav_high_command_max_step_yaw_ = config["high_command_max_step_yaw"].as<double>();
    }
    this->nav_high_command_max_step_x_ = std::fabs(this->nav_high_command_max_step_x_);
    this->nav_high_command_max_step_y_ = std::fabs(this->nav_high_command_max_step_y_);
    this->nav_high_command_max_step_yaw_ = std::fabs(this->nav_high_command_max_step_yaw_);
    if (config["vision_channels"])
    {
        const int channels = config["vision_channels"].as<int>();
        this->nav_vision_channels_ = (channels > 0) ? channels : 1;
    }
    if (config["watchdog_timeout_ms"])
    {
        const int timeout_ms = config["watchdog_timeout_ms"].as<int>();
        this->nav_watchdog_timeout_ms_ = (timeout_ms > 0) ? timeout_ms : 1200;
    }
    if (config["debug_enable"])
    {
        this->nav_debug_enable_ = config["debug_enable"].as<bool>();
    }
    if (config["debug_log_interval_s"])
    {
        const double interval_s = config["debug_log_interval_s"].as<double>();
        this->nav_debug_log_interval_s_ = (interval_s > 0.0) ? interval_s : 1.0;
    }
    this->nav_perf_log_interval_s_ = this->nav_debug_log_interval_s_;
    this->nav_console_info_enable_ = this->nav_debug_enable_;
    this->nav_loop_overrun_log_enable_ = this->nav_debug_enable_;
    this->nav_loop_lifecycle_log_enable_ = this->nav_debug_enable_;
    this->nav_depth_console_log_enable_ = this->nav_debug_enable_;
    this->nav_perf_log_enable_ = this->nav_debug_enable_;
    if (config["depth_debug_publish_enable"])
    {
        this->nav_depth_debug_publish_enable_ = config["depth_debug_publish_enable"].as<bool>();
    }
    if (config["gamepad_enable"])
    {
        this->nav_gamepad_enable_ = config["gamepad_enable"].as<bool>();
    }
    if (config["perf_csv_enable"])
    {
        this->nav_perf_csv_enable_ = config["perf_csv_enable"].as<bool>();
    }
    if (config["perf_csv_flush_interval_s"])
    {
        const double interval_s = config["perf_csv_flush_interval_s"].as<double>();
        this->nav_perf_csv_flush_interval_s_ = (interval_s > 0.0) ? interval_s : 1.0;
    }
    if (config["perf_csv_dir"])
    {
        this->nav_perf_csv_dir_ = config["perf_csv_dir"].as<std::string>();
    }
    if (config["obs_log_enable"])
    {
        this->nav_obs_log_enable_ = config["obs_log_enable"].as<bool>();
    }
    if (config["obs_log_interval_s"])
    {
        const double interval_s = config["obs_log_interval_s"].as<double>();
        this->nav_obs_log_interval_s_ = (interval_s > 0.0) ? interval_s : 0.1;
    }
    if (config["obs_log_dir"])
    {
        this->nav_obs_log_dir_ = config["obs_log_dir"].as<std::string>();
    }
    if (config["sdk_gyro_is_euler_rate"])
    {
        this->nav_sdk_gyro_is_euler_rate_ = config["sdk_gyro_is_euler_rate"].as<bool>();
    }
    if (config["ang_vel_type"])
    {
        const std::string configured_ang_vel_type = config["ang_vel_type"].as<std::string>();
        if (configured_ang_vel_type == "ang_vel_body" || configured_ang_vel_type == "ang_vel_world")
        {
            this->ang_vel_type = configured_ang_vel_type;
        }
        else
        {
            std::cout << LOGGER::WARNING
                      << "Invalid ang_vel_type='" << configured_ang_vel_type
                      << "', fallback to '" << this->ang_vel_type
                      << "'. Expected 'ang_vel_body' or 'ang_vel_world'."
                      << std::endl;
        }
    }
    if (this->nav_vision_channels_ < 1)
    {
        this->nav_vision_channels_ = 1;
    }

    LoopFunc::SetOverrunLogEnabled(this->nav_loop_overrun_log_enable_);
    LoopFunc::SetLifecycleLogEnabled(this->nav_loop_lifecycle_log_enable_);
    DepthBuffer::SetConsoleLogEnabled(this->nav_depth_console_log_enable_);
    this->StartPerfCsvIfNeeded();

    // Keep one extra newest frame in buffer and drop it at inference time for one-frame delay.
    depth_buffer = DepthBuffer(1, 30, 43, this->nav_vision_channels_ + 1);
    if (this->nav_console_info_enable_)
    {
        std::cout << LOGGER::INFO
                  << "Nav vision_channels=" << this->nav_vision_channels_
                  << ", depth_history_steps=" << (this->nav_vision_channels_ + 1)
                  << ", cmd_clip=["
                  << this->nav_command_clip_x_ << ", "
                  << this->nav_command_clip_y_ << ", "
                  << this->nav_command_clip_yaw_ << "]"
                  << ", cmd_filter_alpha=" << this->nav_high_command_filter_alpha_
                  << ", cmd_step_limits=["
                  << this->nav_high_command_max_step_x_ << ", "
                  << this->nav_high_command_max_step_y_ << ", "
                  << this->nav_high_command_max_step_yaw_ << "]"
                  << ", goal_stop_radius=" << this->nav_goal_stop_radius_
                  << ", depth_target_hz=" << ((this->nav_dt_ > 0.0) ? (1.0 / this->nav_dt_) : 10.0)
                  << ", watchdog_timeout_ms=" << this->nav_watchdog_timeout_ms_
                  << ", perf_log_enable=" << (this->nav_perf_log_enable_ ? "true" : "false")
                  << ", perf_log_interval_s=" << this->nav_perf_log_interval_s_
                  << ", console_info_enable=" << (this->nav_console_info_enable_ ? "true" : "false")
                  << ", loop_overrun_log_enable=" << (this->nav_loop_overrun_log_enable_ ? "true" : "false")
                  << ", loop_lifecycle_log_enable=" << (this->nav_loop_lifecycle_log_enable_ ? "true" : "false")
                  << ", depth_console_log_enable=" << (this->nav_depth_console_log_enable_ ? "true" : "false")
                  << ", depth_debug_publish_enable=" << (this->nav_depth_debug_publish_enable_ ? "true" : "false")
                  << ", debug_enable=" << (this->nav_debug_enable_ ? "true" : "false")
                  << ", debug_log_interval_s=" << this->nav_debug_log_interval_s_
                  << ", gamepad_enable=" << (this->nav_gamepad_enable_ ? "true" : "false")
                  << ", perf_csv_enable=" << (this->nav_perf_csv_enable_ ? "true" : "false")
                  << ", perf_csv_dir=" << (this->nav_perf_csv_dir_.empty() ? "(default)" : this->nav_perf_csv_dir_)
                  << ", obs_log_enable=" << (this->nav_obs_log_enable_ ? "true" : "false")
                  << ", obs_log_interval_s=" << this->nav_obs_log_interval_s_
                  << ", obs_log_dir=" << (this->nav_obs_log_dir_.empty() ? "(default)" : this->nav_obs_log_dir_)
                  << ", sdk_gyro_is_euler_rate=" << (this->nav_sdk_gyro_is_euler_rate_ ? "true" : "false")
                  << ", nav_infer_device=" << this->nav_infer_device_.str()
                  << ", low_level_ang_vel_type=" << this->ang_vel_type
                  << '\n';
    }

    this->nav_timer_left_.store(this->nav_episode_length_s_);
    this->nav_episode_start_ns_.store(0, std::memory_order_relaxed);
    this->nav_time_io_.store(0.0);
    this->nav_time_io_hf_.store(0.0);

    this->nav_high_model_path_ = nav_dir + "/" + high_name;
    this->nav_vision_model_path_ = nav_dir + "/" + vision_name;

    this->nav_high_model_ = torch::jit::load(this->nav_high_model_path_);
    this->nav_vision_model_ = torch::jit::load(this->nav_vision_model_path_);
    try
    {
        this->nav_high_model_.to(this->nav_infer_device_);
        this->nav_vision_model_.to(this->nav_infer_device_);
        this->nav_high_model_.eval();
        this->nav_vision_model_.eval();
    }
    catch (const c10::Error &e)
    {
        std::cout << LOGGER::WARNING
                  << "Failed to move nav models to target device (" << this->nav_infer_device_.str()
                  << "), fallback to CPU. reason: " << e.what()
                  << std::endl;
        this->nav_infer_device_ = torch::Device(torch::kCPU);
        this->nav_high_model_.to(this->nav_infer_device_);
        this->nav_vision_model_.to(this->nav_infer_device_);
        this->nav_high_model_.eval();
        this->nav_vision_model_.eval();
    }
    std::cout << LOGGER::INFO
              << "Navigation inference device: " << this->nav_infer_device_.str()
              << std::endl;

    // training-aligned dims (go2)
    const int dof = this->params.num_of_dofs; // 12
    const int hf_dim = 1 + 3 + 3 + dof + dof + dof ;
    const int obs_dim = 3 + 3 + 3 + 1 + 3 + 3 + dof + dof + dof ;
    const int obs_io_dim = 3 + 3 + 3 + 1 + 3 + 3 + dof + dof + dof ;
    this->nav_hf_dim_ = hf_dim;
    this->nav_obs_dim_ = obs_dim;
    this->nav_obs_io_dim_ = obs_io_dim;
    this->nav_runtime_prime_done_.store(false);

    this->nav_highfreq_buf_ = ObservationBuffer(1, {hf_dim}, this->nav_highfreq_hist_len_, "time");
    this->nav_obs_hist_buf_ = ObservationBuffer(1, {obs_dim}, this->nav_obs_hist_len_, "time");
    this->nav_obs_io_hist_buf_ = ObservationBuffer(1, {obs_io_dim}, this->nav_obs_io_hist_len_, "time");

    this->nav_position_targets_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_high_command_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    this->nav_last_actions_ = std::vector<float>(dof, 0.0f);

    this->WarmupNavModels(obs_dim, obs_io_dim, hf_dim);

    return true;
}

void RL_Real::UpdateHighFrequencyObs()
{
    const double t = this->GetNavEpisodeElapsedSec();
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
    using SteadyClock = std::chrono::steady_clock;
    static bool perf_has_last_tick = false;
    static SteadyClock::time_point perf_last_tick_tp;
    static bool perf_inited = false;
    static SteadyClock::time_point perf_last_report_tp;
    static double perf_sum_vision_ms = 0.0;
    static double perf_sum_high_ms = 0.0;
    static double perf_max_vision_ms = 0.0;
    static double perf_max_high_ms = 0.0;
    static uint64_t perf_samples = 0;
    static uint64_t timeio_check_goal_seq = 0;
    static int timeio_check_count = 0;
    const auto cycle_begin_tp = SteadyClock::now();
    double tick_ms = std::numeric_limits<double>::quiet_NaN();
    if (perf_has_last_tick)
    {
        tick_ms = std::chrono::duration<double, std::milli>(cycle_begin_tp - perf_last_tick_tp).count();
    }
    perf_last_tick_tp = cycle_begin_tp;
    perf_has_last_tick = true;

    const auto write_nav_perf_row = [this, cycle_begin_tp, tick_ms](
        const char *event,
        double nav_vision_ms,
        double nav_high_ms) {
        const double nav_total_ms =
            std::chrono::duration<double, std::milli>(SteadyClock::now() - cycle_begin_tp).count();
        this->WritePerfCsvRow(
            event,
            tick_ms,
            nav_total_ms,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            tick_ms,
            nav_vision_ms,
            nav_high_ms,
            nav_total_ms);
    };

    this->PrimeNavRuntimeOnce();

    if (!this->nav_enabled_.load())
    {
        this->StopNavObsLogIfNeeded();
        write_nav_perf_row(
            "nav_idle",
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        return;
    }
    static auto last_gate_log_tp = std::chrono::steady_clock::time_point{};
    const auto maybe_log_gate = [this](const char *reason, uint64_t goal_seq_now) {
        if (!this->nav_console_info_enable_)
        {
            return;
        }
        const auto now_tp = std::chrono::steady_clock::now();
        if (last_gate_log_tp.time_since_epoch().count() != 0 &&
            std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_gate_log_tp).count() < 1)
        {
            return;
        }
        std::cout << LOGGER::INFO
                  << "[NAV][GATE] mode=ON blocked_by=" << reason
                  << " rl_init_done=" << (this->rl_init_done ? 1 : 0)
                  << " goal_seq=" << goal_seq_now
                  << '\n';
        last_gate_log_tp = now_tp;
    };
    const auto clear_nav_cmd = [this]() {
        this->nav_cmd_x_.store(0.0);
        this->nav_cmd_y_.store(0.0);
        this->nav_cmd_yaw_.store(0.0);
        if (this->nav_high_command_.defined())
        {
            this->nav_high_command_.zero_();
        }
    };
    if (!this->rl_init_done)
    {
        clear_nav_cmd();
        maybe_log_gate("rl_not_ready", this->nav_goal_seq_.load());
        write_nav_perf_row(
            "nav_gate_rl_not_ready",
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        return;
    }

    const uint64_t goal_seq = this->nav_goal_seq_.load();
    if (goal_seq == 0)
    {
        clear_nav_cmd();
        this->StopNavObsLogIfNeeded();
        maybe_log_gate("no_goal", goal_seq);
        write_nav_perf_row(
            "nav_gate_no_goal",
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        return;
    }
    const bool new_goal = (goal_seq != this->nav_active_goal_seq_.load());
    this->StartNavObsLogIfNeeded(goal_seq);

    const double goal_body_x_initial = this->nav_goal_body_x_.load();
    const double goal_body_y_initial = this->nav_goal_body_y_.load();
    const double goal_body_yaw_initial = this->nav_goal_body_yaw_.load();

    if (new_goal)
    {
        this->ResetNavEpisodeClock();
        this->nav_active_goal_seq_.store(goal_seq);
        this->nav_timer_left_.store(this->nav_episode_length_s_);
        this->nav_hl_beat_seq_.store(0);
        this->nav_high_command_.zero_();
        this->nav_cmd_x_.store(0.0);
        this->nav_cmd_y_.store(0.0);
        this->nav_cmd_yaw_.store(0.0);

        this->nav_position_targets_body_initial_ = torch::tensor({{
            static_cast<float>(goal_body_x_initial),
            static_cast<float>(goal_body_y_initial),
            static_cast<float>(goal_body_yaw_initial),
        }});
        this->nav_spawn_positions_body_initial_ = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    }

    const double timer_left = this->nav_timer_left_.load();
    const double timer_norm = std::max(0.0, timer_left) / std::max(1e-6, this->nav_episode_length_s_);
    torch::Tensor timer_tensor = torch::tensor({{static_cast<float>(timer_norm)}});

    const int dof = this->params.num_of_dofs;
    std::array<float, 4> imu_quat_snapshot{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 3> imu_gyro_snapshot{0.0f, 0.0f, 0.0f};
    std::array<float, 32> dof_pos_snapshot{};
    std::array<float, 32> dof_vel_snapshot{};
    {
        std::lock_guard<std::mutex> lock(this->nav_state_mutex_);
        for (int i = 0; i < 4; ++i)
        {
            imu_quat_snapshot[i] = static_cast<float>(this->robot_state.imu.quaternion[i]);
        }
        for (int i = 0; i < 3; ++i)
        {
            imu_gyro_snapshot[i] = static_cast<float>(this->robot_state.imu.gyroscope[i]);
        }
        const int dof_copy = std::min(dof, 32);
        for (int i = 0; i < dof_copy; ++i)
        {
            dof_pos_snapshot[i] = static_cast<float>(this->robot_state.motor_state.q[i]);
            dof_vel_snapshot[i] = static_cast<float>(this->robot_state.motor_state.dq[i]);
        }
    }
    torch::Tensor base_quat = torch::tensor({{
        imu_quat_snapshot[0],
        imu_quat_snapshot[1],
        imu_quat_snapshot[2],
        imu_quat_snapshot[3],
    }});
    torch::Tensor gravity_vec = torch::tensor({{0.0f, 0.0f, -1.0f}});
    torch::Tensor projected_gravity = this->QuatRotateInverse(base_quat, gravity_vec);
    torch::Tensor base_ang_vel = torch::tensor({{
        imu_gyro_snapshot[0],
        imu_gyro_snapshot[1],
        imu_gyro_snapshot[2],
    }}) * static_cast<float>(this->params.ang_vel_scale);
    torch::Tensor dof_pos_raw = torch::from_blob(
        dof_pos_snapshot.data(),
        {1, dof},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    torch::Tensor dof_vel_raw = torch::from_blob(
        dof_vel_snapshot.data(),
        {1, dof},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    torch::Tensor dof_pos_term = (dof_pos_raw - this->params.default_dof_pos) * static_cast<float>(this->params.dof_pos_scale);
    torch::Tensor dof_vel_term = dof_vel_raw * static_cast<float>(this->params.dof_vel_scale);

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
    torch::Tensor prev_high_command = torch::zeros({1, 3}, torch::dtype(torch::kFloat32));
    if (this->nav_high_command_.defined() && this->nav_high_command_.numel() >= 3)
    {
        prev_high_command = this->nav_high_command_.to(torch::kFloat32);
    }
    torch::Tensor prev_high_command_scaled =
        prev_high_command * this->params.commands_scale.to(torch::kFloat32);
    if (this->nav_console_info_enable_ && this->nav_debug_enable_)
    {
        static auto last_obs_debug_tp = std::chrono::steady_clock::time_point{};
        const auto now_tp = std::chrono::steady_clock::now();
        if (last_obs_debug_tp.time_since_epoch().count() == 0 ||
            std::chrono::duration<double>(now_tp - last_obs_debug_tp).count() >= this->nav_debug_log_interval_s_)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "[NAV][CHK][OBS]"
                << " ang_vel_raw_rad_s=["
                << imu_gyro_snapshot[0] << ", "
                << imu_gyro_snapshot[1] << ", "
                << imu_gyro_snapshot[2] << "]"
                << " ang_vel_scaled=["
                << base_ang_vel[0][0].item<double>() << ", "
                << base_ang_vel[0][1].item<double>() << ", "
                << base_ang_vel[0][2].item<double>() << "]"
                << " proj_g=["
                << projected_gravity[0][0].item<double>() << ", "
                << projected_gravity[0][1].item<double>() << ", "
                << projected_gravity[0][2].item<double>() << "]";
            std::cout << LOGGER::INFO << oss.str() << '\n';
            last_obs_debug_tp = now_tp;
        }
    }
    // Sample low-frequency time as late as possible before composing model inputs,
    // so it stays aligned with hf_hist timestamps in the same nav tick.
    const double time_io = this->GetNavEpisodeElapsedSec();
    this->nav_time_io_.store(time_io);
    torch::Tensor time_io_tensor = torch::tensor({{static_cast<float>(time_io)}});
    torch::Tensor obs_frame = torch::cat({
        this->nav_position_targets_body_initial_.to(torch::kFloat32),
        this->nav_spawn_positions_body_initial_.to(torch::kFloat32),
        prev_high_command_scaled,
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
        prev_high_command_scaled,
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
        {
            std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
            this->nav_highfreq_buf_.reset({0}, obs_io_frame_hf);
        }
    }
    else
    {
        this->nav_obs_hist_buf_.insert(obs_frame);
        this->nav_obs_io_hist_buf_.insert(obs_io_frame);
    }

    // Align HF timeline with current nav tick using the current loop_nav snapshot.
    // This avoids reacquiring nav_state_mutex_ and re-reading shared state.
    {
        std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
        this->nav_highfreq_buf_.insert(obs_io_frame_hf);
    }

    std::vector<int> obs_ids_10;
    obs_ids_10.reserve(this->nav_obs_hist_len_);
    for (int i = this->nav_obs_hist_len_ - 1; i >= 0; --i)
    {
        obs_ids_10.push_back(i);
    }
    std::vector<int> obs_ids_20;
    obs_ids_20.reserve(this->nav_highfreq_hist_len_);
    for (int i = this->nav_highfreq_hist_len_ - 1; i >= 0; --i)
    {
        obs_ids_20.push_back(i);
    }

    torch::Tensor obs_io_hist = this->nav_obs_io_hist_buf_.get_obs_vec(obs_ids_10);
    torch::Tensor hf_hist;
    {
        std::lock_guard<std::mutex> lock(this->nav_highfreq_mutex_);
        hf_hist = this->nav_highfreq_buf_.get_obs_vec(obs_ids_20);
    }

    if (new_goal || goal_seq != timeio_check_goal_seq)
    {
        timeio_check_goal_seq = goal_seq;
        timeio_check_count = 0;
    }
    if (this->nav_console_info_enable_ && this->nav_debug_enable_ && timeio_check_count < 3)
    {
        timeio_check_count++;
        const double time_io_hf_now = this->nav_time_io_hf_.load();
        const int64_t obs_io_dim = obs_io_frame.size(1);
        const int64_t hf_dim = obs_io_frame_hf.size(1);
        // lite3 obs_io_frame layout:
        // [goal_init(3), spawn_init(3), prev_high_cmd_scaled(3), time_io(1), ...]
        const int64_t obs_io_time_index = 3 + 3 + 3;
        const int64_t obs_io_latest_offset = obs_io_dim * (this->nav_obs_io_hist_len_ - 1);
        const double obs_io_hist_latest_time =
            (obs_io_hist.defined() &&
             obs_io_hist.numel() >= (obs_io_latest_offset + obs_io_time_index + 1))
                ? obs_io_hist[0][obs_io_latest_offset + obs_io_time_index].item<double>()
                : std::numeric_limits<double>::quiet_NaN();
        const double hf_hist_oldest_time =
            (hf_hist.defined() && hf_hist.numel() >= hf_dim)
                ? hf_hist[0][0].item<double>()
                : std::numeric_limits<double>::quiet_NaN();
        const double hf_hist_latest_time =
            (hf_hist.defined() && hf_hist.numel() >= hf_dim * this->nav_highfreq_hist_len_)
                ? hf_hist[0][hf_dim * (this->nav_highfreq_hist_len_ - 1)].item<double>()
                : std::numeric_limits<double>::quiet_NaN();

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4)
            << "[NAV][CHK][TIMEIO][" << timeio_check_count << "/3]"
            << " goal_seq=" << goal_seq
            << " new_goal=" << (new_goal ? 1 : 0)
            << " time_io=" << time_io
            << " time_io_hf_now=" << time_io_hf_now
            << " obs_io_hist_latest_time=" << obs_io_hist_latest_time
            << " hf_hist_oldest_time=" << hf_hist_oldest_time
            << " hf_hist_latest_time=" << hf_hist_latest_time
            << " hf_minus_lf_now=" << (time_io_hf_now - time_io)
            << " hf_hist_latest_minus_lf=" << (hf_hist_latest_time - time_io);
        std::cout << LOGGER::INFO << oss.str() << '\n';
    }

    torch::Tensor vision_feat;
    const auto vision_begin_tp = SteadyClock::now();
    if (!g_depth_frame_received.load())
    {
        this->DisableNavigationWithError("vision_input", "no processed depth received yet on /camera/depth/image_rect_raw");
    }
    torch::Tensor depth;
    {
        std::lock_guard<std::mutex> lock(this->depth_buffer_mutex_);
        depth = depth_buffer.get_depth_vec();
    }
    if (!depth.defined())
    {
        this->DisableNavigationWithError("vision_input", "depth tensor is undefined");
    }
    depth = depth.to(torch::kFloat32);
    if (depth.dim() != 4)
    {
        std::ostringstream oss;
        oss << "depth tensor rank mismatch, expected 4D [B,C,H,W], got dim=" << depth.dim();
        this->DisableNavigationWithError("vision_input", oss.str());
    }
    const int64_t channels = depth.size(1);
    if (channels != static_cast<int64_t>(this->nav_vision_channels_))
    {
        std::ostringstream oss;
        oss << "depth channels mismatch, expected " << this->nav_vision_channels_
            << ", got " << channels
            << " (history_steps=" << (this->nav_vision_channels_ + 1) << ")";
        this->DisableNavigationWithError("vision_input", oss.str());
    }
    torch::Tensor depth_model = depth;
    if (depth_model.device() != this->nav_infer_device_)
    {
        depth_model = depth_model.to(this->nav_infer_device_);
    }
    torch::Tensor obs_frame_model = obs_frame;
    torch::Tensor obs_io_hist_model = obs_io_hist;
    torch::Tensor hf_hist_model = hf_hist;
    if (obs_frame_model.device() != this->nav_infer_device_)
    {
        obs_frame_model = obs_frame_model.to(this->nav_infer_device_);
        obs_io_hist_model = obs_io_hist_model.to(this->nav_infer_device_);
        hf_hist_model = hf_hist_model.to(this->nav_infer_device_);
    }
    vision_feat = this->nav_vision_model_.forward({depth_model}).toTensor();
    const double vision_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - vision_begin_tp).count();

    torch::Tensor cmd;
    torch::Tensor pred_target_body;
    const auto high_begin_tp = SteadyClock::now();
    std::vector<torch::jit::IValue> inputs = {obs_frame_model, obs_io_hist_model, vision_feat, hf_hist_model};
    torch::jit::IValue out = this->nav_high_model_.forward(inputs);
    const double high_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - high_begin_tp).count();
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

    if (!cmd.defined() || cmd.numel() < 3)
    {
        this->DisableNavigationWithError("output_validate", "invalid high-level output: command tensor missing or too short");
    }
    if (!cmd.device().is_cpu())
    {
        cmd = cmd.to(torch::kCPU, torch::kFloat32);
    }
    if (pred_target_body.defined() && !pred_target_body.device().is_cpu())
    {
        pred_target_body = pred_target_body.to(torch::kCPU, torch::kFloat32);
    }

    if (this->nav_perf_log_enable_)
    {
        if (!perf_inited)
        {
            perf_last_report_tp = SteadyClock::now();
            perf_inited = true;
        }
        perf_sum_vision_ms += vision_ms;
        perf_sum_high_ms += high_ms;
        perf_max_vision_ms = std::max(perf_max_vision_ms, vision_ms);
        perf_max_high_ms = std::max(perf_max_high_ms, high_ms);
        perf_samples += 1;

        const auto now_tp = SteadyClock::now();
        if (std::chrono::duration<double>(now_tp - perf_last_report_tp).count() >= this->nav_perf_log_interval_s_ &&
            perf_samples > 0)
        {
            const double avg_vision_ms = perf_sum_vision_ms / static_cast<double>(perf_samples);
            const double avg_high_ms = perf_sum_high_ms / static_cast<double>(perf_samples);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << "[NAV][PERF] samples=" << perf_samples
                << " vision_ms(avg/max)=[" << avg_vision_ms << "/" << perf_max_vision_ms << "]"
                << " high_ms(avg/max)=[" << avg_high_ms << "/" << perf_max_high_ms << "]";
            std::cout << LOGGER::INFO << oss.str() << '\n';

            perf_sum_vision_ms = 0.0;
            perf_sum_high_ms = 0.0;
            perf_max_vision_ms = 0.0;
            perf_max_high_ms = 0.0;
            perf_samples = 0;
            perf_last_report_tp = now_tp;
        }
    }

    torch::Tensor source_command = cmd.to(torch::kFloat32);
    if (source_command.dim() == 1)
    {
        source_command = source_command.view({1, -1});
    }
    source_command = source_command.slice(1, 0, 3);
    const auto cmd_device = source_command.device();

    torch::Tensor last_high_command = torch::zeros(
        {1, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(cmd_device));
    if (this->nav_high_command_.defined() && this->nav_high_command_.numel() >= 3)
    {
        last_high_command = this->nav_high_command_.to(
            torch::TensorOptions().dtype(torch::kFloat32).device(cmd_device));
        if (last_high_command.dim() == 1)
        {
            last_high_command = last_high_command.view({1, -1});
        }
        last_high_command = last_high_command.slice(1, 0, 3);
    }

    const float filter_alpha = static_cast<float>(this->nav_high_command_filter_alpha_);
    torch::Tensor filtered_command =
        filter_alpha * last_high_command + (1.0f - filter_alpha) * source_command;
    const torch::Tensor high_command_step_limits = torch::tensor(
        {static_cast<float>(this->nav_high_command_max_step_x_),
         static_cast<float>(this->nav_high_command_max_step_y_),
         static_cast<float>(this->nav_high_command_max_step_yaw_)},
        torch::TensorOptions().dtype(torch::kFloat32).device(cmd_device)).view({1, 3});
    const torch::Tensor filtered_delta = filtered_command - last_high_command;
    filtered_command = last_high_command + torch::max(
        torch::min(filtered_delta, high_command_step_limits),
        -high_command_step_limits);

    const torch::Tensor command_clip_limits = torch::tensor(
        {static_cast<float>(this->nav_command_clip_x_),
         static_cast<float>(this->nav_command_clip_y_),
         static_cast<float>(this->nav_command_clip_yaw_)},
        torch::TensorOptions().dtype(torch::kFloat32).device(cmd_device)).view({1, 3});
    cmd = torch::max(torch::min(filtered_command, command_clip_limits), -command_clip_limits);
    const torch::Tensor cmd_raw = source_command;

    static auto last_nav_run_log_tp = std::chrono::steady_clock::time_point{};
    const auto now_tp = std::chrono::steady_clock::now();
    const bool due_log = (last_nav_run_log_tp.time_since_epoch().count() == 0) ||
                         (std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_nav_run_log_tp).count() >= 1);
    if (this->nav_console_info_enable_ && (new_goal || due_log))
    {
        double pred_x = std::numeric_limits<double>::quiet_NaN();
        double pred_y = std::numeric_limits<double>::quiet_NaN();
        double pred_yaw = std::numeric_limits<double>::quiet_NaN();
        bool has_pred = false;
        if (pred_target_body.defined() && pred_target_body.numel() >= 2)
        {
            has_pred = true;
            pred_x = pred_target_body[0][0].item<double>();
            pred_y = pred_target_body[0][1].item<double>();
            pred_yaw = (pred_target_body.numel() >= 3) ? pred_target_body[0][2].item<double>() : 0.0;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "[NAV][RUN] mode=ON"
            << " goal_init:[" << this->nav_position_targets_body_initial_[0][0].item<double>() << ", "
            << this->nav_position_targets_body_initial_[0][1].item<double>() << ", "
            << this->nav_position_targets_body_initial_[0][2].item<double>() << "]"
            << " pred_body:";
        if (has_pred)
        {
            oss << "[" << pred_x << ", " << pred_y << ", " << pred_yaw << "]";
        }
        else
        {
            oss << "[NA]";
        }
        oss << " cmd_raw:[" << cmd_raw[0][0].item<double>() << ", "
            << cmd_raw[0][1].item<double>() << ", "
            << cmd_raw[0][2].item<double>() << "]"
            << " cmd:[" << cmd[0][0].item<double>() << ", "
            << cmd[0][1].item<double>() << ", "
            << cmd[0][2].item<double>() << "]";
        std::cout << LOGGER::INFO << oss.str() << '\n';
        last_nav_run_log_tp = now_tp;
    }

    this->WriteNavObsSemanticLog(
        goal_seq,
        new_goal,
        time_io,
        timer_norm,
        this->nav_position_targets_body_initial_,
        pred_target_body,
        cmd_raw,
        cmd,
        prev_high_command_scaled,
        base_ang_vel,
        projected_gravity,
        dof_pos_raw,
        dof_pos_term,
        dof_vel_raw,
        dof_vel_term,
        actions);

    if (pred_target_body.defined() && pred_target_body.numel() >= 2)
    {
        const double pred_x = pred_target_body[0][0].item<double>();
        const double pred_y = pred_target_body[0][1].item<double>();
        const double goal_pred_radius = std::hypot(pred_x, pred_y);
        if (goal_pred_radius <= this->nav_goal_stop_radius_)
        {
            this->nav_enabled_.store(false);
            this->control.x = 0.0;
            this->control.y = 0.0;
            this->control.yaw = 0.0;
            clear_nav_cmd();
            this->StopNavObsLogIfNeeded();
            if (this->nav_console_info_enable_)
            {
                std::cout << LOGGER::INFO
                          << "[NAV][STOP] Navigation mode: OFF (goal reached, pred_radius="
                          << goal_pred_radius << " <= " << this->nav_goal_stop_radius_
                          << ")"
                          << '\n';
            }
            write_nav_perf_row("nav_goal_reached", vision_ms, high_ms);
            return;
        }
    }

#if defined(USE_ROS2)
    this->PublishNavGoalComparison(pred_target_body, goal_seq, new_goal);
#endif

    this->nav_cmd_x_.store(cmd[0][0].item<double>());
    this->nav_cmd_y_.store(cmd[0][1].item<double>());
    this->nav_cmd_yaw_.store(cmd[0][2].item<double>());
    {
        c10::InferenceMode normal_mode(false);
        this->nav_high_command_ = cmd.to(torch::kFloat32).clone();
    }
    this->nav_hl_beat_seq_.fetch_add(1);
#if defined(USE_ROS2)
    // Debug output: raw high-level command (before filter/step-limit/clip), aligned with training source_command.
    if (this->nav_cmd_high_publisher_)
    {
        geometry_msgs::msg::Vector3 msg;
        msg.x = cmd_raw[0][0].item<double>();
        msg.y = cmd_raw[0][1].item<double>();
        msg.z = cmd_raw[0][2].item<double>();
        this->nav_cmd_high_publisher_->publish(msg);
    }
#endif

    this->nav_timer_left_.store(std::max(0.0, timer_left - this->nav_dt_));
    write_nav_perf_row("nav_high_level", vision_ms, high_ms);

}


#if !defined(USE_CMAKE) && (defined(USE_ROS1) || defined(USE_ROS2))
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

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
    RL_Real rl_sar;
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Real>());
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || (!defined(USE_ROS1) && !defined(USE_ROS2))
    RL_Real rl_sar;
    while (1) { sleep(10); }
#endif
    return 0;
}
