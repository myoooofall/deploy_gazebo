#include "observation_buffer.hpp"

ObservationBuffer::ObservationBuffer() {}

ObservationBuffer::ObservationBuffer(int num_envs,
                                     int num_obs,
                                     int include_history_steps)
    : num_envs(num_envs),
      num_obs(num_obs),
      include_history_steps(include_history_steps)
{
    num_obs_total = num_obs * include_history_steps;
    obs_buf = torch::zeros({num_envs, num_obs_total}, torch::dtype(torch::kFloat32));
}

void ObservationBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_obs)
{
    std::vector<torch::indexing::TensorIndex> indices;
    for (int idx : reset_idxs)
    {
        indices.push_back(torch::indexing::Slice(idx));
    }
    obs_buf.index_put_(indices, new_obs.repeat({1, include_history_steps}));
}

void ObservationBuffer::insert(torch::Tensor new_obs)
{
    // Shift observations back.
    torch::Tensor shifted_obs = obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(num_obs, num_obs * include_history_steps)}).clone();
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, num_obs * (include_history_steps - 1))}) = shifted_obs;

    // Add new observation.
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-num_obs, torch::indexing::None)}) = new_obs;
}

torch::Tensor ObservationBuffer::get_obs_vec(std::vector<int> obs_ids)
{
    std::vector<torch::Tensor> obs;
    for (int i = obs_ids.size() - 1; i >= 0; --i)
    {
        int obs_id = obs_ids[i];
        int slice_idx = include_history_steps - obs_id - 1;
        obs.push_back(obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(slice_idx * num_obs, (slice_idx + 1) * num_obs)}));
    }
    return torch::cat(obs, -1);
}
// DepthBuffer implementation
DepthBuffer::DepthBuffer() {}

DepthBuffer::DepthBuffer(int num_envs,
                        int height,
                        int width,
                        int include_history_steps)
    : num_envs(num_envs),
      height(height),
      width(width),
      include_history_steps(include_history_steps)
{
    depth_buf = torch::zeros({num_envs, include_history_steps, height, width}, torch::dtype(torch::kFloat32));
}

void DepthBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_depth)
{
    std::vector<torch::indexing::TensorIndex> indices;
    for (int idx : reset_idxs)
    {
        indices.push_back(torch::indexing::Slice(idx));
    }
    depth_buf.index_put_(indices, new_depth.repeat({1, include_history_steps, 1, 1}));
}

void DepthBuffer::insert(torch::Tensor new_depth)
{
    // Shift observations back.
    torch::Tensor shifted_depth = depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(include_history_steps - 1, torch::indexing::None), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}).clone();
    depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, include_history_steps - 1), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = shifted_depth;

    // Add new observation.
    depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-1), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
}

torch::Tensor DepthBuffer::get_depth_vec()
{
    return depth_buf;
}
torch::Tensor DepthBuffer::process_depth_image_old(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 打印消息格式信息
    std::cout << "深度图编码格式: " << msg->encoding << std::endl;
    std::cout << "每个像素的字节数: " << msg->step / msg->width << std::endl;
    std::cout << "是否大端序: " << msg->is_bigendian << std::endl;
    
    // 打印原始图像尺寸
    std::cout << "原始图像尺寸: " << msg->width << "x" << msg->height << std::endl;
    std::cout << "目标图像尺寸: " << width << "x" << height << std::endl;
    
    // 正确读取16位深度数据
    std::vector<uint16_t> depth_data;
    depth_data.reserve(msg->width * msg->height);
    const uint8_t* data_ptr = msg->data.data();
    
    for (size_t i = 0; i < msg->data.size(); i += 2) {
        uint16_t depth;
        if (msg->is_bigendian) {
            depth = (static_cast<uint16_t>(data_ptr[i]) << 8) | static_cast<uint16_t>(data_ptr[i + 1]);
        } else {
            depth = (static_cast<uint16_t>(data_ptr[i + 1]) << 8) | static_cast<uint16_t>(data_ptr[i]);
        }
        depth_data.push_back(depth);
    }
    
    torch::Tensor depth_tensor = torch::from_blob(depth_data.data(), 
        {msg->height, msg->width}, torch::kInt16).clone();
    
    // 打印原始深度值范围
    std::cout << "原始深度值范围: [" << depth_tensor.min().item<int16_t>() << ", " << depth_tensor.max().item<int16_t>() << "]" << std::endl;
    
    // 转换为float类型并转换为米
    depth_tensor = depth_tensor.to(torch::kFloat32) / 1000.0;  // 转换为米
    
    // 打印转换为米后的范围
    std::cout << "转换为米后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 将深度值裁剪到0.2-2.0米范围
    depth_tensor = torch::clamp(depth_tensor, 0.2, 2.0);
    
    // 打印裁剪后的范围
    std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 归一化到-0.5到0.5范围
    depth_tensor = (depth_tensor - 1) / 2;  // (x - 1.1) / 1.8 将0.2-2.0映射到-0.5-0.5
    
    // 打印归一化后的范围
    std::cout << "归一化后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 调整大小到目标尺寸
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // 添加batch和channel维度
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    
    // 打印调整后的深度值范围
    std::cout << "调整后的深度值范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    return depth_tensor.squeeze(0).squeeze(0);  // 移除batch和channel维度
}
torch::Tensor DepthBuffer::process_depth_image(
    const sensor_msgs::msg::Image::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_publisher,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher)
{
    // 打印消息格式信息
    // std::cout << "深度图编码格式: " << msg->encoding << std::endl;
    // std::cout << "每个像素的字节数: " << msg->step / msg->width << std::endl;
    // std::cout << "是否大端序: " << msg->is_bigendian << std::endl;
    
    // 打印原始图像尺寸
    // std::cout << "原始图像尺寸: " << msg->width << "x" << msg->height << std::endl;
    // std::cout << "目标图像尺寸: " << width << "x" << height << std::endl;
    
    // Convert ROS Image message to OpenCV Mat
    cv_bridge::CvImagePtr cv_ptr;
    try {
        if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } else if (msg->encoding == sensor_msgs::image_encodings::MONO16) {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16);
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("DepthBuffer"), "Unsupported depth image encoding: %s", msg->encoding.c_str());
            // Return an empty tensor or handle error appropriately
            return torch::empty({}); 
        }
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("DepthBuffer"), "cv_bridge exception: %s", e.what());
        return torch::empty({});
    }

    cv::Mat depth_mat_16UC1 = cv_ptr->image;

    // --- Hole Filling / Denoising ---
    // Create a mask for invalid depth values (0 in 16UC1 for Realsense)
    cv::Mat mask;
    cv::compare(depth_mat_16UC1, 0, mask, cv::CMP_EQ);

    // Apply inpainting (hole filling)
    cv::Mat filtered_depth_mat;
    // For inpainting, depth_mat_16UC1 needs to be 8-bit or float.
    // Let's convert to float for inpainting, then convert back to 16UC1 if needed for publishing.
    cv::Mat depth_float;
    depth_mat_16UC1.convertTo(depth_float, CV_32FC1); // Convert to float (meters if scaled, or mm if not)

    // Inpaint invalid depth values. `5` is the radius of circular neighborhood of each inpainted point.
    // `INPAINT_TELEA` or `INPAINT_NS` are available algorithms.
    cv::inpaint(depth_float, mask, filtered_depth_mat, 5, cv::INPAINT_TELEA);

    // If you need to publish the filtered image in a specific format, convert it.
    // For visualization, converting to 8-bit grayscale is common.
    cv::Mat filtered_depth_vis;
    filtered_depth_mat.convertTo(filtered_depth_vis, CV_8UC1, 255.0 / 2000.0); // Scale to 0-255 for visualization (assuming max depth around 2m = 2000mm)
    cv::cvtColor(filtered_depth_vis, filtered_depth_vis, cv::COLOR_GRAY2BGR); // Convert to BGR for typical image viewers

    // Publish the filtered depth image
    if (filtered_publisher) {
        sensor_msgs::msg::Image filtered_img_msg;
        cv_bridge::CvImage filtered_cv_img;
        filtered_cv_img.header = msg->header; // Use original timestamp and frame_id
        filtered_cv_img.encoding = sensor_msgs::image_encodings::BGR8; // Or MONO8 if you prefer
        filtered_cv_img.image = filtered_depth_vis;
        filtered_publisher->publish(*filtered_cv_img.toImageMsg());
        std::cout << "Published filtered depth image." << std::endl;
    }

    // Convert filtered_depth_mat (CV_32FC1, already in meters if original was mm/1000) to torch::Tensor
    torch::Tensor depth_tensor = torch::from_blob(filtered_depth_mat.data, 
                                                  {filtered_depth_mat.rows, filtered_depth_mat.cols}, 
                                                  torch::kFloat32).clone();
    
    // 打印原始深度值范围
    // std::cout << "原始深度值范围: [" << depth_tensor.min().item<int16_t>() << ", " << depth_tensor.max().item<int16_t>() << "]" << std::endl;
    
    // 转换为float类型并转换为米
    // depth_tensor = depth_tensor.to(torch::kFloat32) / 1000.0;  // 转换为米
    
    // 打印转换为米后的范围
    // std::cout << "转换为米后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 将深度值裁剪到0.2-2.0米范围
    // depth_tensor = torch::clamp(depth_tensor, 0.2, 2.0);
    
    // 打印裁剪后的范围
    // std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 归一化到-0.5到0.5范围
    // depth_tensor = (depth_tensor - 1) / 2;  // (x - 1.1) / 1.8 将0.2-2.0映射到-0.5-0.5
    
    // 打印归一化后的范围
    // std::cout << "归一化后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // Resize to target dimensions
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // 添加batch和channel维度
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    
    // Print adjusted depth value range
    // std::cout << "Adjusted depth value range: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;

    // --- Publish the final processed image ---
    if (processed_publisher) {
        // Convert the final processed torch::Tensor back to OpenCV Mat for publishing
        torch::Tensor final_processed_tensor_cpu = depth_tensor.squeeze(0).squeeze(0).cpu();
        cv::Mat final_processed_mat(final_processed_tensor_cpu.size(0), final_processed_tensor_cpu.size(1), CV_32FC1, final_processed_tensor_cpu.data_ptr<float>());

        // For visualization, normalize to 0-255 and convert to 8UC1
        cv::Mat final_processed_vis;
        // The tensor is in range -0.5 to 0.5. Scale to 0-255 for visualization.
        final_processed_mat.convertTo(final_processed_vis, CV_8UC1, 255.0); // Scale -0.5 to 0.5 to 0-255. Note: will map -0.5 to -127.5, so add 0.5 before multiplying by 255, or clamp to 0-1 and then multiply.
                                                                           // A better visualization for -0.5 to 0.5: `(final_processed_mat + 0.5) * 255.0`
        final_processed_vis = (final_processed_mat + 0.5) * 255.0;
        final_processed_vis.convertTo(final_processed_vis, CV_8UC1);
        cv::cvtColor(final_processed_vis, final_processed_vis, cv::COLOR_GRAY2BGR);

        sensor_msgs::msg::Image processed_img_msg;
        cv_bridge::CvImage processed_cv_img;
        processed_cv_img.header = msg->header;
        processed_cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        processed_cv_img.image = final_processed_vis;
        processed_publisher->publish(*processed_cv_img.toImageMsg());
        std::cout << "Published final processed depth image." << std::endl;
    }
    
    return depth_tensor.squeeze(0).squeeze(0); // Remove batch and channel dimensions
}