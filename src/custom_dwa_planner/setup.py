from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'custom_dwa_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch and rviz directories
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harsh',
    maintainer_email='harsh@todo.todo',
    description='Custom DWA local planner for TurtleBot3 in ROS 2 Humble',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa_planner = custom_dwa_planner.dwa_planner:main',
            'goal_publisher = custom_dwa_planner.goal_publisher:main',
        ],
    },
)
