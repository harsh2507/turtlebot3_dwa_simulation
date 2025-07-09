from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Paths
    pkg_dwa = get_package_share_directory('custom_dwa_planner')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    rviz_config_path = os.path.join(pkg_dwa, 'rviz', 'dwa_config.rviz')
    tb3_world_launch = os.path.join(pkg_tb3_gazebo, 'launch', 'turtlebot3_world.launch.py')

    return LaunchDescription([
        # Set TurtleBot3 model
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),

        # Launch Gazebo with TurtleBot3 world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tb3_world_launch)
        ),

        # Launch RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),

        # Launch DWA planner node
        Node(
            package='custom_dwa_planner',
            executable='dwa_planner',
            name='dwa_planner_node',
            output='screen'
        ),

        # Launch goal publisher node
        Node(
            package='custom_dwa_planner',
            executable='goal_publisher',
            name='goal_publisher',
            output='screen'
        ),
    ])

