class RLSim : public rclcpp::Node
{
public:
    RLSim() : Node("rl_sim")
    {
        // 创建发布者
        processed_depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/processed_depth", 10);
            
        // ... 其他初始化代码 ...
        
        // 创建DepthBuffer时传入发布者
        depth_buffer = DepthBuffer(num_envs, depth_height, depth_width, include_history_steps,
                                 processed_depth_pub_);
    }
    
private:
    // ... 其他成员变量 ...
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_depth_pub_;
}; 