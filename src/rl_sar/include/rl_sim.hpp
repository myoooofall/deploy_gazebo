/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "fsm.hpp"

#include <csignal>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <atomic>
#include <mutex>

#if defined(USE_ROS1)
#include <ros/ros.h>
#include "std_srvs/Empty.h"
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <gazebo_msgs/ModelStates.h>
#include "robot_msgs/MotorCommand.h"
#include "robot_msgs/MotorState.h"
#elif defined(USE_ROS2)
#include "robot_msgs/msg/robot_command.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <gazebo_msgs/srv/spawn_entity.hpp>
#include <gazebo_msgs/srv/delete_entity.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Sim : public RL
#if defined(USE_ROS2)
    , public rclcpp::Node
#endif
{
public:
    RL_Sim();
    ~RL_Sim();

 private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_navi;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // ros interface
    std::string ros_namespace;

    //depth
    int motion_time = 1;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_subscriber;
    void DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_depth_publisher;
    bool no_depth_check_ = false; // Flag for depth image check
    bool no_depth_forward = false; // Flag for depth image forwarding
#if defined(USE_ROS1)
    geometry_msgs::Twist vel;
    geometry_msgs::Pose pose;
    geometry_msgs::Twist cmd_vel;
    sensor_msgs::Joy joy_msg;
    ros::Subscriber model_state_subscriber;
    ros::Subscriber cmd_vel_subscriber;
    ros::Subscriber joy_subscriber;
    ros::ServiceClient gazebo_pause_physics_client;
    ros::ServiceClient gazebo_unpause_physics_client;
    ros::ServiceClient gazebo_reset_world_client;
    std::map<std::string, ros::Publisher> joint_publishers;
    std::map<std::string, ros::Subscriber> joint_subscribers;
    std::vector<robot_msgs::MotorCommand> joint_publishers_commands;
    void ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg);
    void JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name);
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
    void JoyCallback(const sensor_msgs::Joy::ConstPtr &msg);
#elif defined(USE_ROS2)
    sensor_msgs::msg::Imu gazebo_imu;
    geometry_msgs::msg::Twist cmd_vel;
    sensor_msgs::msg::Joy joy_msg;
    robot_msgs::msg::RobotCommand robot_command_publisher_msg;
    robot_msgs::msg::RobotState robot_state_subscriber_msg;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr gazebo_model_states_subscriber;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr gazebo_model_states_subscriber_alt;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr nav_odom_subscriber;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr nav_odom_subscriber_alt;
    std::atomic<double> nav_base_world_x_{0.0};
    std::atomic<double> nav_base_world_y_{0.0};
    std::atomic<double> nav_base_world_z_{0.0};
    std::atomic<double> nav_base_world_yaw_{0.0};
    std::atomic<double> nav_base_world_qx_{0.0};
    std::atomic<double> nav_base_world_qy_{0.0};
    std::atomic<double> nav_base_world_qz_{0.0};
    std::atomic<double> nav_base_world_qw_{1.0};
    std::atomic<bool> nav_base_world_valid_{false};
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gazebo_imu_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_pause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_unpause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_reset_world_client;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr nav_goal_body_subscriber;
    void GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void GazeboModelStatesCallback(const gazebo_msgs::msg::ModelStates::SharedPtr msg);
    void NavOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void NavGoalBodyCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg);
#endif

    // others
    std::string gazebo_model_name;
    int motiontime = 0;
    std::map<std::string, double> joint_positions;
    std::map<std::string, double> joint_velocities;
    std::map<std::string, double> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);

    // hierarchical navigation (high-level policy @ 10Hz)
    void RunHighLevel();
    bool InitHierarchicalNav();
    void UpdateHighFrequencyObs();

    // nav state shared across loops
    std::atomic<bool> nav_enabled_{false};
    std::atomic<bool> nav_models_loaded_{false};
    std::atomic<bool> nav_has_goal_{false};
    std::atomic<uint64_t> nav_goal_seq_{0};

    std::atomic<double> nav_goal_body_x_{0.0};
    std::atomic<double> nav_goal_body_y_{0.0};
    std::atomic<double> nav_goal_body_yaw_{0.0};
    // Goal in world (latched from body goal + robot pose at goal time); used for evaluation/visualization.
    std::atomic<double> nav_goal_world_x_{0.0};
    std::atomic<double> nav_goal_world_y_{0.0};
    std::atomic<double> nav_goal_world_yaw_{0.0};
    std::atomic<bool> nav_goal_world_valid_{false};

    std::atomic<double> nav_cmd_x_{0.0};
    std::atomic<double> nav_cmd_y_{0.0};
    std::atomic<double> nav_cmd_yaw_{0.0};

    // gazebo goal marker (visualization)
    std::atomic<bool> nav_goal_marker_spawned_{false};
    rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedPtr nav_goal_marker_spawn_client;
    rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedPtr nav_goal_marker_delete_client;
    void UpdateNavGoalMarker(double goal_body_x, double goal_body_y, double goal_body_yaw);
    void UpdateNavPredMarker(double pred_body_x, double pred_body_y, double pred_body_yaw);

    // buffers and models (guarded as needed)
    torch::jit::script::Module nav_high_model_;
    torch::jit::script::Module nav_vision_model_;
    std::string nav_config_path_;
    std::string nav_high_model_path_;
    std::string nav_vision_model_path_;

    int nav_obs_hist_len_ = 10;
    int nav_obs_io_hist_len_ = 10;
    int nav_highfreq_hist_len_ = 20;
    int nav_vision_channels_ = 2;       // number of depth frames consumed by vision model
    double nav_dt_ = 0.1;               // 10Hz
    double nav_episode_length_s_ = 30;  // default if not specified
    double nav_clip_commands_ = 3.0;    // default clip


    std::mutex nav_highfreq_mutex_;
    std::mutex nav_state_mutex_;
    ObservationBuffer nav_highfreq_buf_;
    ObservationBuffer nav_obs_hist_buf_;
    ObservationBuffer nav_obs_io_hist_buf_;

    torch::Tensor nav_position_targets_body_initial_;
    torch::Tensor nav_spawn_positions_body_initial_;
    torch::Tensor nav_high_command_;

    std::atomic<uint64_t> nav_active_goal_seq_{0};
    // Time since current nav episode start for high-level policy (10Hz, advanced by nav_dt_).
    std::atomic<double> nav_time_io_{0.0};
    // Time since current nav episode start for high-frequency buffer (advanced by params.dt).
    std::atomic<double> nav_time_io_hf_{0.0};
    std::atomic<double> nav_timer_left_{0.0};

    std::mutex nav_last_actions_mutex_;
    std::vector<float> nav_last_actions_;
};

#endif // RL_SIM_HPP
