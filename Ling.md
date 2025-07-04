不设置domain id的话运行程序会报dds错误  
打开第一个终端，设置domain id为1，启动realsense节点，在ros_ws目录，source后直接run就可以  
然后再开一个终端，设置domain id为1，然后直接run rl_real_go2  
现在会如果深度相机断联会触发急停