#ifndef OBSERVATION_BUFFER_HPP
#define OBSERVATION_BUFFER_HPP

#include <torch/torch.h>
#include <vector>
#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp> // For OpenCV functions like inpaint, cvtColor, resize, etc.
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp> 

class ObservationBuffer
{
public:
    ObservationBuffer(int num_envs, int num_obs, int include_history_steps);
    ObservationBuffer();

    void reset(std::vector<int> reset_idxs, torch::Tensor new_obs);
    void insert(torch::Tensor new_obs);
    torch::Tensor get_obs_vec(std::vector<int> obs_ids);

private:
    int num_envs;
    int num_obs;
    int include_history_steps;
    int num_obs_total;
    torch::Tensor obs_buf;
};
class DepthBuffer
{
public:
    DepthBuffer(int num_envs, int height, int width, int include_history_steps);
    DepthBuffer();

    void reset(std::vector<int> reset_idxs, torch::Tensor new_depth);
    void insert(torch::Tensor new_depth);
    torch::Tensor get_depth_vec();
    torch::Tensor process_depth_image_old(const sensor_msgs::msg::Image::SharedPtr msg);
    torch::Tensor process_depth_image(
        const sensor_msgs::msg::Image::SharedPtr msg,
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_publisher,
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher
        );


private:
    int num_envs;
    int height;
    int width;
    int include_history_steps;
    torch::Tensor depth_buf;
};

#endif // OBSERVATION_BUFFER_HPP
