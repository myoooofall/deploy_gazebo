#ifndef OBSERVATION_BUFFER_HPP
#define OBSERVATION_BUFFER_HPP

#include <torch/torch.h>
#include <vector>
#include <sensor_msgs/msg/image.hpp>

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
    torch::Tensor process_depth_image(const sensor_msgs::msg::Image::SharedPtr msg);

private:
    int num_envs;
    int height;
    int width;
    int include_history_steps;
    torch::Tensor depth_buf;
};

#endif // OBSERVATION_BUFFER_HPP
