## 🛠️ Installation & Build Instructions

1. Clone this repo inside your ROS 2 workspace:
   ```bash
   cd ~/turtlebot3_ws/src
   git clone https://github.com/yourusername/dwa_planner_ros2.git
   
2. Install dependencies (if any):
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   
4. Build the workspace:
   ```bash
   cd ~/turtlebot3_ws
   colcon build
   source install/setup.bash
   
6. Launch Gazebo & rviz:
   ```bash
   export TURTLEBOT3_MODEL=burger
   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

   ![DWA Visualization Result](rviz.png)
