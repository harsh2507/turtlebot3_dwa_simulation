import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/harsh/turtlebot3_ws/install/custom_dwa_planner'
