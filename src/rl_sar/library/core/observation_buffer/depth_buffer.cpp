#include "depth_buffer.hpp"
#include <opencv2/highgui.hpp>
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
    // reset后，如果所有环境都被reset，则重置initialized标志
    // 这里简化处理：reset后保持initialized状态，因为buffer已经被填充了
}

void DepthBuffer::insert(torch::Tensor new_depth)
{
    // new_depth shape: [height, width] = [60, 86] (没有batch维度)
    // depth_buf shape: [num_envs, include_history_steps, height, width] = [1, 3, 60, 86]
    // Queue: index 0 (最老) -> index 1 -> index 2 (最新)
    // depth_buf.index({0, i, ...}) 返回 shape: [60, 86]
    
    if (!initialized) {
        // 第一次插入：用第一帧复制满整个buffer (3帧)
        for (int i = 0; i < include_history_steps; ++i) {
            depth_buf.index({0, i, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
        }
        initialized = true;
    } else {
        // 后续插入：FIFO队列逻辑
        // 将索引0和1的内容前移到索引0和1（索引0的内容被丢弃）
        // 将索引1的内容移到索引0
        depth_buf.index({0, 0, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = 
            depth_buf.index({0, 1, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
        // 将索引2的内容移到索引1
        depth_buf.index({0, 1, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = 
            depth_buf.index({0, 2, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
        // 新帧插入到索引2（队尾）
        depth_buf.index({0, 2, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
    }
}

torch::Tensor DepthBuffer::get_depth_vec()
{
    // 只返回前两帧（索引0和1），索引2（最新帧）不使用，实现一帧延迟
    // depth_buf shape: [1, 3, 60, 86]
    // 返回 shape: [1, 2, 60, 86]
    return depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 2), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
}

torch::Tensor DepthBuffer::process_depth_image(const sensor_msgs::msg::Image::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher)  
{
    // // 打印消息格式信息
    // std::cout << "深度图编码格式: " << msg->encoding << std::endl;
    // std::cout << "每个像素的字节数: " << msg->step / msg->width << std::endl;
    // std::cout << "是否大端序: " << msg->is_bigendian << std::endl;
    
    // // 打印原始图像尺寸
    // std::cout << "原始图像尺寸: " << msg->width << "x" << msg->height << std::endl;
    // std::cout << "目标图像尺寸: " << width << "x" << height << std::endl;
    
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
    
    // // 打印原始深度值范围
    // std::cout << "原始深度值范围: [" << depth_tensor.min().item<int16_t>() << ", " << depth_tensor.max().item<int16_t>() << "]" << std::endl;
    
    // 转换为float类型并转换为米
    depth_tensor = depth_tensor.to(torch::kFloat32) / 1000.0;  // 转换为米
    
    // First resize to intermediate size (60, 106): height=60, width=106
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // Add batch and channel dims for interpolate
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{60, 106})
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(0).squeeze(0);  // Remove dims back to [60, 106]
    
    // Crop width: remove first 10 and last 10 columns [10:-10], keeping height unchanged
    // Result: [60, 86]
    depth_tensor = depth_tensor.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(10, -10)});  // [60, 86]
    
    // // 打印裁剪后的范围
    // std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 将深度值裁剪到0.2-2.0米范围
    depth_tensor = torch::clamp(depth_tensor, 0.2, 2.0);
    
    // 发布用于可视化的深度图（在归一化之前保存原始值）
    if (processed_publisher) {
        // depth_tensor shape is [60, 86] at this point
        torch::Tensor processed_tensor = depth_tensor.contiguous();
        
        // Get dimensions (height=60, width=86)
        int h = processed_tensor.size(0);
        int w = processed_tensor.size(1);
        
        // 创建cv::Mat，此时数据是实际的深度值（0.2~2.0米）
        // 使用32FC1格式（32位浮点数），单位米，无需转换
        cv::Mat depth_mat(h, w, CV_32FC1, processed_tensor.data_ptr<float>());
        
        // 使用32FC1编码发布，单位是米，rqt可以正确处理
        auto image_msg = cv_bridge::CvImage(msg->header, "32FC1", depth_mat).toImageMsg();
        processed_publisher->publish(*image_msg);
    }
    
    // 归一化到-0.5到0.5范围 (用于推理)
    // depth_normalized = (depth_m - 1) / 2 将0.2-2.0映射到-0.5-0.5
    depth_tensor = (depth_tensor - 1.0) / 2.0;
    
    // depth_tensor shape is already [60, 86] at this point, no need to resize
    // 打印调整后的深度值范围
    // std::cout << "调整后的深度值范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    return depth_tensor;  // Shape: [60, 86]
}