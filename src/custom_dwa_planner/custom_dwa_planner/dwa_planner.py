# # # # # import rclpy
# # # # # from rclpy.node import Node
# # # # # from geometry_msgs.msg import Twist, PoseStamped
# # # # # from nav_msgs.msg import Odometry, Path
# # # # # from sensor_msgs.msg import LaserScan
# # # # # import numpy as np
# # # # # import math

# # # # # class DWAPlanner(Node):
# # # # #     def __init__(self):
# # # # #         super().__init__('dwa_planner')

# # # # #         self.declare_parameter('robot_radius', 0.15)
# # # # #         self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value

# # # # #         self.max_speed = 0.26
# # # # #         self.min_speed = 0.05
# # # # #         self.max_yawrate = 1.82
# # # # #         self.accel = 0.5
# # # # #         self.yaw_accel = 2.5
# # # # #         self.dt = 0.1
# # # # #         self.predict_time = 2.0

# # # # #         self.pose = None
# # # # #         self.scan = None
# # # # #         self.goal = None
# # # # #         self.global_path = []
# # # # #         self.velocity = (0.0, 0.0)

# # # # #         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
# # # # #         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
# # # # #         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
# # # # #         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
# # # # #         self.create_subscription(Path, '/global_path', self.path_callback, 10)
# # # # #         self.timer = self.create_timer(self.dt, self.plan)

# # # # #     def odom_callback(self, msg):
# # # # #         self.pose = msg.pose.pose
# # # # #         self.velocity = (
# # # # #             msg.twist.twist.linear.x,
# # # # #             msg.twist.twist.angular.z
# # # # #         )

# # # # #     def scan_callback(self, msg):
# # # # #         self.scan = msg

# # # # #     def goal_callback(self, msg):
# # # # #         self.goal = msg.pose

# # # # #     def path_callback(self, msg):
# # # # #         self.global_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

# # # # #     def plan(self):
# # # # #         if self.pose is None or self.scan is None or self.goal is None:
# # # # #             return

# # # # #         best_score = float('-inf')
# # # # #         best_cmd = Twist()

# # # # #         v_range = np.linspace(
# # # # #             self.min_speed,
# # # # #             self.max_speed,
# # # # #             5)

# # # # #         w_range = np.linspace(
# # # # #             -self.max_yawrate,
# # # # #             self.max_yawrate,
# # # # #             10)

# # # # #         for v in v_range:
# # # # #             for w in w_range:
# # # # #                 traj = self.simulate_trajectory(v, w)
# # # # #                 if self.check_collision(traj):
# # # # #                     continue
# # # # #                 score = self.evaluate(traj, v)
# # # # #                 if score > best_score:
# # # # #                     best_score = score
# # # # #                     best_cmd.linear.x = v
# # # # #                     best_cmd.angular.z = w

# # # # #         self.cmd_pub.publish(best_cmd)

# # # # #     def simulate_trajectory(self, v, w):
# # # # #         x = self.pose.position.x
# # # # #         y = self.pose.position.y
# # # # #         yaw = self.get_yaw()

# # # # #         traj = []
# # # # #         for _ in np.arange(0, self.predict_time, self.dt):
# # # # #             x += v * math.cos(yaw) * self.dt
# # # # #             y += v * math.sin(yaw) * self.dt
# # # # #             yaw += w * self.dt
# # # # #             traj.append((x, y))
# # # # #         return traj

# # # # #     def check_collision(self, traj):
# # # # #         if self.scan is None:
# # # # #             return False

# # # # #         ranges = np.array(self.scan.ranges)
# # # # #         angle_min = self.scan.angle_min
# # # # #         angle_increment = self.scan.angle_increment

# # # # #         robot_x = self.pose.position.x
# # # # #         robot_y = self.pose.position.y
# # # # #         yaw = self.get_yaw()

# # # # #         for traj_x, traj_y in traj:
# # # # #             for i, r in enumerate(ranges):
# # # # #                 if np.isinf(r) or np.isnan(r):
# # # # #                     continue
# # # # #                 angle = angle_min + i * angle_increment
# # # # #                 obs_x = robot_x + r * math.cos(angle + yaw)
# # # # #                 obs_y = robot_y + r * math.sin(angle + yaw)
# # # # #                 dist = math.hypot(traj_x - obs_x, traj_y - obs_y)
# # # # #                 if dist <= self.radius:
# # # # #                     return True
# # # # #         return False

# # # # #     def evaluate(self, traj, v):
# # # # #         x_end, y_end = traj[-1]
# # # # #         goal_dist = math.hypot(self.goal.position.x - x_end, self.goal.position.y - y_end)

# # # # #         path_cost = float('inf')
# # # # #         if self.global_path:
# # # # #             for px, py in self.global_path:
# # # # #                 path_cost = min(path_cost, math.hypot(px - x_end, py - y_end))
# # # # #         else:
# # # # #             path_cost = 0.0

# # # # #         score = -1.0 * goal_dist - 0.8 * path_cost + 1.0 * v
# # # # #         return score

# # # # #     def get_yaw(self):
# # # # #         o = self.pose.orientation
# # # # #         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
# # # # #         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
# # # # #         return math.atan2(siny_cosp, cosy_cosp)


# # # # # def main(args=None):
# # # # #     rclpy.init(args=args)
# # # # #     node = DWAPlanner()
# # # # #     rclpy.spin(node)
# # # # #     rclpy.shutdown()


# # # # ###########################################################################################################################################################

# # # #                                                         # code 2
# # # # ##############################################################################################################################################################

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# class Parameters:
#     def __init__(self):
#         self.dt = 0.1
#         self.goal = np.array([2, 1])
#         self.v = 0.2
#         self.r0 = 0.15
#         self.R = 0.2
#         self.r_buffer = self.R
#         self.v_max = 0.26
#         self.omega_min = -1.82
#         self.omega_max = 1.82
#         self.n_omega = 20
#         self.prediction_horizon = 100

# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')
#         self.parms = Parameters()
#         self.pose = None
#         self.scan = None

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

#         self.timer = self.create_timer(self.parms.dt, self.plan)

#     def odom_callback(self, msg):
#         self.pose = msg.pose.pose

#     def scan_callback(self, msg):
#         self.scan = msg

#     def goal_callback(self, msg):
#         self.parms.goal = np.array([msg.pose.position.x, msg.pose.position.y])

#     def plan(self):
#         if self.pose is None or self.scan is None:
#             return

#         x = self.pose.position.x
#         y = self.pose.position.y
#         theta = self.get_yaw()

#         r = np.linalg.norm(self.parms.goal - np.array([x, y]))
#         if r < self.parms.r0:
#             self.cmd_pub.publish(Twist())
#             return

#         omega = self.dwa(x, y, theta, self.parms.v)
#         cmd = Twist()
#         cmd.linear.x = self.parms.v
#         cmd.angular.z = omega
#         self.cmd_pub.publish(cmd)

#     def dwa(self, x0, y0, theta0, v):
#         n_omega = self.parms.n_omega
#         prediction_horizon = self.parms.prediction_horizon
#         omega_all = np.linspace(self.parms.omega_min, self.parms.omega_max, n_omega)
#         x_goal, y_goal = self.parms.goal
#         h = self.parms.dt

#         scan_ranges = np.array(self.scan.ranges)
#         angle_min = self.scan.angle_min
#         angle_inc = self.scan.angle_increment

#         best_score = float('inf')
#         best_omega = 0.0

#         for omega in omega_all:
#             z0 = [x0, y0, theta0]
#             z = np.array([z0])
#             cost = 0.0
#             valid = True

#             for _ in range(prediction_horizon):
#                 z0 = self.euler_integration(z0, [v, omega])
#                 z = np.vstack([z, z0])

#             for j in range(prediction_horizon + 1):
#                 x, y = z[j, 0], z[j, 1]
#                 goal_cost = np.linalg.norm(np.array([x_goal, y_goal]) - np.array([x, y]))
#                 cost += goal_cost

#                 for i, r in enumerate(scan_ranges):
#                     if np.isinf(r) or np.isnan(r):
#                         continue
#                     angle = angle_min + i * angle_inc
#                     obs_x = x0 + r * math.cos(angle + theta0)
#                     obs_y = y0 + r * math.sin(angle + theta0)
#                     dist_obstacle = np.linalg.norm(np.array([x, y]) - np.array([obs_x, obs_y])) - self.parms.r_buffer
#                     if dist_obstacle < self.parms.R:
#                         valid = False
#                         break
#                     cost += 0.1 * (1 / (dist_obstacle + 1e-2))

#                 if not valid:
#                     break

#             if valid and cost < best_score:
#                 best_score = cost
#                 best_omega = omega

#         return best_omega

#     def euler_integration(self, z0, u):
#         v, omega = u
#         h = self.parms.dt
#         x0, y0, theta0 = z0

#         x1 = x0 + v * math.cos(theta0) * h
#         y1 = y0 + v * math.sin(theta0) * h
#         theta1 = theta0 + omega * h

#         return [x1, y1, theta1]

#     def get_yaw(self):
#         o = self.pose.orientation
#         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DWAPlanner()
#     rclpy.spin(node)
#     rclpy.shutdown()

#     Got it. You want to implement a stopping criterion when the robot reaches the goal, without altering the existing DWA logic or parameter management.


# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# class Parameters:
#     def __init__(self):
#         self.dt = 0.1
#         self.goal = np.array([2, 1])
#         self.v = 0.2
#         self.r0 = 0.15 # This parameter is used for your existing stopping condition
#         self.R = 0.2
#         self.r_buffer = self.R
#         self.v_max = 0.26
#         self.omega_min = -1.82
#         self.omega_max = 1.82
#         self.n_omega = 20
#         self.prediction_horizon = 100
#         # --- NEW: Goal Tolerance Parameter ---
#         self.goal_tolerance = 0.1 # meters, how close to goal to consider reached


# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')
#         self.parms = Parameters()
#         self.pose = None
#         self.scan = None

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

#         self.timer = self.create_timer(self.parms.dt, self.plan)
#         self.get_logger().info("DWA Planner initialized") # Added init log

#     def odom_callback(self, msg):
#         self.pose = msg.pose.pose

#     def scan_callback(self, msg):
#         self.scan = msg

#     def goal_callback(self, msg):
#         self.parms.goal = np.array([msg.pose.position.x, msg.pose.position.y])
#         self.get_logger().info(f"New goal received: ({self.parms.goal[0]:.2f}, {self.parms.goal[1]:.2f})") # Added goal log

#     def plan(self):
#         if self.pose is None or self.scan is None or self.parms.goal is None: # Added check for self.parms.goal
#             # self.get_logger().debug("Waiting for pose, scan, or goal.") # Optional: enable for debugging
#             return

#         x = self.pose.position.x
#         y = self.pose.position.y
#         theta = self.get_yaw()

#         # --- NEW: Stopping Criteria Check ---
#         distance_to_goal = np.linalg.norm(self.parms.goal - np.array([x, y]))
#         if distance_to_goal < self.parms.goal_tolerance:
#             self.get_logger().info(f"Goal reached! Stopping robot. Distance: {distance_to_goal:.2f}m")
#             self.cmd_pub.publish(Twist()) # Publish zero Twist to stop the robot
#             self.parms.goal = None # Clear the goal to stop planning and prevent re-engagement
#             return # Exit the plan function for this cycle

#         # Your original stopping condition (which will now be superseded by the new one for actual goal reaching)
#         r = np.linalg.norm(self.parms.goal - np.array([x, y]))
#         if r < self.parms.r0: # This was your original close-to-goal stop, which might be a bit ambiguous now
#             # self.get_logger().debug(f"Robot very close to goal (within r0). Distance: {r:.2f}m")
#             # If the new goal_tolerance is larger than r0, this block might not be hit for final stop
#             # If r0 is meant as a region where final adjustments happen before full stop, keep it.
#             # Otherwise, the new goal_tolerance handles the ultimate stop.
#             pass # We rely on the goal_tolerance check above for the final stop.
#                  # If you want this to still issue a stop command *before* the final goal_tolerance,
#                  # uncomment the line below. However, it's usually better to have one clear stopping condition.
#             # self.cmd_pub.publish(Twist()) # This would stop it if it's within r0 but not yet goal_tolerance
#             # return # Return if you want this to act as a definitive stop at r0

#         omega = self.dwa(x, y, theta, self.parms.v)
#         cmd = Twist()
#         cmd.linear.x = self.parms.v
#         cmd.angular.z = omega
#         self.cmd_pub.publish(cmd)

#     def dwa(self, x0, y0, theta0, v):
#         n_omega = self.parms.n_omega
#         prediction_horizon = self.parms.prediction_horizon
#         omega_all = np.linspace(self.parms.omega_min, self.parms.omega_max, n_omega)
#         x_goal, y_goal = self.parms.goal
#         h = self.parms.dt

#         scan_ranges = np.array(self.scan.ranges)
#         angle_min = self.scan.angle_min
#         angle_inc = self.scan.angle_increment

#         best_score = float('inf') # Assuming minimizing cost, so best_score starts high
#         best_omega = 0.0

#         # --- PRE-COMPUTE GLOBAL OBSTACLE POINTS ONCE PER DWA CALL ---
#         # This is the crucial fix for your collision detection logic that was discussed previously.
#         # This part of the code was not modified in the previous request (as you asked for minimal changes)
#         # but it's a fundamental issue if your collision avoidance isn't working as expected.
#         # It assumes laser is at robot's center. Adjust if laser has a static offset.
#         current_robot_pose_for_scan = np.array([x0, y0, theta0])
#         global_obstacle_points = []
#         for i, r in enumerate(scan_ranges):
#             # Only consider valid ranges within the max limit
#             if np.isinf(r) or np.isnan(r) or r <= self.scan.range_min or r >= self.scan.range_max:
#                 continue # Skip invalid or out-of-range readings
            
#             angle_in_robot_frame = angle_min + i * angle_inc
            
#             # Point in robot's current local frame (relative to robot's center)
#             px_local = r * math.cos(angle_in_robot_frame)
#             py_local = r * math.sin(angle_in_robot_frame)

#             # Transform to global frame based on current robot pose
#             ox_global = current_robot_pose_for_scan[0] + (px_local * math.cos(current_robot_pose_for_scan[2]) - py_local * math.sin(current_robot_pose_for_scan[2]))
#             oy_global = current_robot_pose_for_scan[1] + (px_local * math.sin(current_robot_pose_for_scan[2]) + py_local * math.cos(current_robot_pose_for_scan[2]))
#             global_obstacle_points.append(np.array([ox_global, oy_global]))


#         for omega in omega_all:
#             z0_sim = [x0, y0, theta0] # Current robot state for this simulation
#             # z = np.array([z0_sim]) # No need to store full trajectory if just for cost
#             cost = 0.0
#             valid = True

#             for step in range(prediction_horizon):
#                 z0_sim = self.euler_integration(z0_sim, [v, omega]) # Simulate one step
#                 x_sim, y_sim, _ = z0_sim

#                 # Calculate goal cost for this simulated point
#                 goal_cost = np.linalg.norm(np.array([x_goal, y_goal]) - np.array([x_sim, y_sim]))
#                 cost += goal_cost

#                 # Collision check against pre-computed global obstacle points
#                 min_dist_to_obstacle_at_step = float('inf')
#                 for obs_point in global_obstacle_points:
#                     dist_obstacle_center_to_sim_robot = np.linalg.norm(np.array([x_sim, y_sim]) - obs_point)
#                     min_dist_to_obstacle_at_step = min(min_dist_to_obstacle_at_step, dist_obstacle_center_to_sim_robot)
                
#                 # Check for actual collision: robot perimeter touches obstacle
#                 collision_dist_threshold = self.parms.R + self.parms.r_buffer # Assuming R is robot radius, r_buffer is safety margin
#                 if min_dist_to_obstacle_at_step < collision_dist_threshold:
#                     valid = False
#                     break # Trajectory is invalid due to collision

#                 # Add obstacle cost - penalize getting too close
#                 # This formulation can be risky if min_dist_to_obstacle_at_step becomes 0 or negative
#                 # A safer cost function might involve a sigmoid or direct penalty
#                 # For example, cost += max(0, (collision_dist_threshold - min_dist_to_obstacle_at_step) * some_large_weight)
#                 # For now, keeping your original structure but adding a check for negative distance
#                 if min_dist_to_obstacle_at_step > 0: # Only add cost if not already in collision
#                     cost += 0.1 * (1 / (min_dist_to_obstacle_at_step + 1e-2)) # Added 1e-2 for stability

#             if valid and cost < best_score:
#                 best_score = cost
#                 best_omega = omega
#             elif not valid: # Give a very high cost to invalid trajectories
#                 # This ensures invalid trajectories are always worse than valid ones
#                 cost = float('inf') # Or some very large number
#                 # If you want to still consider angular velocity for recovery (e.g. spin in place)
#                 # when no valid path is found, you would need more sophisticated logic here
#                 # but for simply avoiding invalid trajectories, setting cost to inf is fine.

#         return best_omega

#     def euler_integration(self, z0, u):
#         v, omega = u
#         h = self.parms.dt
#         x0, y0, theta0 = z0

#         x1 = x0 + v * math.cos(theta0) * h
#         y1 = y0 + v * math.sin(theta0) * h
#         theta1 = theta0 + omega * h

#         return [x1, y1, theta1]

#     def get_yaw(self):
#         if self.pose is None: # Added check for pose to prevent error if not yet received
#             return 0.0
#         o = self.pose.orientation
#         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DWAPlanner()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("DWA Planner node stopped by user.")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# # # import rclpy
# # # from rclpy.node import Node
# # # from geometry_msgs.msg import Twist, PoseStamped
# # # from nav_msgs.msg import Odometry
# # # from sensor_msgs.msg import LaserScan
# # # import numpy as np
# # # import math

# # # class DWAPlanner(Node):
# # #     def __init__(self):
# # #         super().__init__('dwa_planner')

# # #         self.declare_parameter('robot_radius', 0.15)
# # #         self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value
# # #         self.safety_buffer = 0.05

# # #         self.max_speed = 0.26
# # #         self.max_yawrate = 1.82
# # #         self.accel = 0.5
# # #         self.yaw_accel = 2.5
# # #         self.dt = 0.1
# # #         self.predict_time = 2.0

# # #         self.alpha = 2.0  # weight for heading
# # #         self.beta = 0.2   # weight for clearance
# # #         self.gamma = 0.2  # weight for speed

# # #         self.pose = None
# # #         self.scan = None
# # #         self.goal = None
# # #         self.velocity = (0.0, 0.0)

# # #         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
# # #         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
# # #         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
# # #         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
# # #         self.timer = self.create_timer(self.dt, self.plan)

# # #     def odom_callback(self, msg):
# # #         self.pose = msg.pose.pose
# # #         self.velocity = (
# # #             msg.twist.twist.linear.x,
# # #             msg.twist.twist.angular.z
# # #         )

# # #     def scan_callback(self, msg):
# # #         self.scan = msg

# # #     def goal_callback(self, msg):
# # #         self.goal = msg.pose

# # #     def plan(self):
# # #         if self.pose is None or self.scan is None or self.goal is None:
# # #             return

# # #         # Stop if near goal
# # #         dx = self.goal.position.x - self.pose.position.x
# # #         dy = self.goal.position.y - self.pose.position.y
# # #         if math.hypot(dx, dy) < 0.2:
# # #             self.cmd_pub.publish(Twist())
# # #             return

# # #         best_score = -float('inf')
# # #         best_v = 0.0
# # #         best_w = 0.0

# # #         v_range = np.linspace(0.0, self.max_speed, 5)
# # #         w_range = np.linspace(-self.max_yawrate, self.max_yawrate, 15)

# # #         # Precompute obstacle positions
# # #         obstacles = self.get_obstacle_positions()

# # #         for v in v_range:
# # #             for w in w_range:
# # #                 traj, min_dist = self.simulate_trajectory(v, w, obstacles)
# # #                 if traj is None:
# # #                     continue

# # #                 score = self.evaluate(traj, v, w, min_dist)
# # #                 if score > best_score:
# # #                     best_score = score
# # #                     best_v = v
# # #                     best_w = w

# # #         cmd = Twist()
# # #         cmd.linear.x = best_v
# # #         cmd.angular.z = best_w
# # #         self.cmd_pub.publish(cmd)
# # #         self.get_logger().info(f"Published cmd: v={best_v:.2f}, w={best_w:.2f}, score={best_score:.2f}")

# # #     def simulate_trajectory(self, v, w, obstacles):
# # #         x = self.pose.position.x
# # #         y = self.pose.position.y
# # #         yaw = self.get_yaw()

# # #         traj = []
# # #         min_dist = float('inf')
# # #         t = 0.0

# # #         while t <= self.predict_time:
# # #             x += v * math.cos(yaw) * self.dt
# # #             y += v * math.sin(yaw) * self.dt
# # #             yaw += w * self.dt
# # #             traj.append((x, y))

# # #             for ox, oy in obstacles:
# # #                 dist = math.hypot(x - ox, y - oy)
# # #                 if dist <= self.radius + self.safety_buffer:
# # #                     return None, 0.0  # Collision
# # #                 min_dist = min(min_dist, dist)
# # #             t += self.dt

# # #         return traj, min_dist

# # #     def get_obstacle_positions(self):
# # #         if self.scan is None:
# # #             return []

# # #         ranges = np.array(self.scan.ranges)
# # #         angle_min = self.scan.angle_min
# # #         angle_increment = self.scan.angle_increment
# # #         yaw = self.get_yaw()
# # #         x0 = self.pose.position.x
# # #         y0 = self.pose.position.y

# # #         obstacles = []
# # #         angle = angle_min
# # #         for r in ranges:
# # #             if np.isinf(r) or np.isnan(r) or r > self.scan.range_max:
# # #                 angle += angle_increment
# # #                 continue
# # #             ox = x0 + r * math.cos(yaw + angle)
# # #             oy = y0 + r * math.sin(yaw + angle)
# # #             obstacles.append((ox, oy))
# # #             angle += angle_increment
# # #         return obstacles

# # #     def evaluate(self, traj, v, w, clearance):
# # #         x_end, y_end = traj[-1]
# # #         dx = self.goal.position.x - x_end
# # #         dy = self.goal.position.y - y_end
# # #         goal_angle = math.atan2(dy, dx)

# # #         traj_heading = math.atan2(traj[-1][1] - traj[0][1], traj[-1][0] - traj[0][0])
# # #         heading_error = abs(self.angle_diff(goal_angle, traj_heading))

# # #         score = (self.alpha * (math.pi - heading_error) +
# # #                  self.beta * clearance +
# # #                  self.gamma * v)
# # #         return score

# # #     def angle_diff(self, a, b):
# # #         diff = a - b
# # #         while diff > math.pi:
# # #             diff -= 2.0 * math.pi
# # #         while diff < -math.pi:
# # #             diff += 2.0 * math.pi
# # #         return diff

# # #     def get_yaw(self):
# # #         o = self.pose.orientation
# # #         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
# # #         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
# # #         return math.atan2(siny_cosp, cosy_cosp)

# # # def main(args=None):
# # #     rclpy.init(args=args)
# # #     node = DWAPlanner()
# # #     rclpy.spin(node)
# # #     rclpy.shutdown()

# # ####################################################################################################################################

# #                                                             #code 3


# # #############################################################################################################################

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')

#         self.declare_parameter('robot_radius', 0.15)
#         self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value
#         self.safety_buffer = 0.1

#         self.max_speed = 0.3
#         self.max_yawrate = 1.5
#         self.accel = 0.6
#         self.yaw_accel = 3.0
#         self.dt = 0.1
#         self.predict_time = 2.5

#         self.alpha = 2.0 #h
#         self.beta = 0.5  #c
#         self.gamma = 0.2 #v

#         self.pose = None
#         self.scan = None
#         self.goal = None
#         self.velocity = (0.0, 0.0)

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
#         self.timer = self.create_timer(self.dt, self.plan)

#     def odom_callback(self, msg):
#         self.pose = msg.pose.pose
#         self.velocity = (
#             msg.twist.twist.linear.x,
#             msg.twist.twist.angular.z
#         )

#     def scan_callback(self, msg):
#         self.scan = msg

#     def goal_callback(self, msg):
#         self.goal = msg.pose

#     def plan(self):
#         if self.pose is None or self.scan is None or self.goal is None:
#             return

#         dx = self.goal.position.x - self.pose.position.x
#         dy = self.goal.position.y - self.pose.position.y
#         if math.hypot(dx, dy) < 0.2:
#             self.cmd_pub.publish(Twist())
#             self.get_logger().info("Goal reached.")
#             return

#         best_score = -float('inf')
#         best_cmd = Twist()

#         v_range = np.linspace(0.0, self.max_speed, 8)
#         w_range = np.linspace(-self.max_yawrate, self.max_yawrate, 21)

#         obstacles = self.get_obstacle_positions()

#         for v in v_range:
#             for w in w_range:
#                 traj, min_dist = self.simulate_trajectory(v, w, obstacles)
#                 if traj is None:
#                     continue

#                 score = self.evaluate(traj, v, w, min_dist)
#                 if score > best_score:
#                     best_score = score
#                     best_cmd.linear.x = v
#                     best_cmd.angular.z = w

#         if best_score == -float('inf'):
#             self.get_logger().warn("No valid trajectory. Rotating to find path.")
#             best_cmd.angular.z = 0.3  # force turn to escape

#         self.cmd_pub.publish(best_cmd)

#     def simulate_trajectory(self, v, w, obstacles):
#         x = self.pose.position.x
#         y = self.pose.position.y
#         yaw = self.get_yaw()

#         traj = []
#         min_dist = float('inf')
#         t = 0.0

#         while t <= self.predict_time:
#             x += v * math.cos(yaw) * self.dt
#             y += v * math.sin(yaw) * self.dt
#             yaw += w * self.dt
#             traj.append((x, y))

#             for ox, oy in obstacles:
#                 dist = math.hypot(x - ox, y - oy)
#                 if dist <= self.radius + self.safety_buffer:
#                     return None, 0.0
#                 min_dist = min(min_dist, dist)

#             dx = self.goal.position.x - x
#             dy = self.goal.position.y - y
#             if math.hypot(dx, dy) < 0.2:
#                 break

#             t += self.dt

#         return traj, min_dist

#     def get_obstacle_positions(self):
#         if self.scan is None:
#             return []

#         ranges = np.array(self.scan.ranges)
#         angle_min = self.scan.angle_min
#         angle_increment = self.scan.angle_increment
#         yaw = self.get_yaw()
#         x0 = self.pose.position.x
#         y0 = self.pose.position.y

#         obstacles = []
#         for i, r in enumerate(ranges):
#             if np.isinf(r) or np.isnan(r) or r > self.scan.range_max:
#                 continue
#             angle = angle_min + i * angle_increment
#             ox = x0 + r * math.cos(yaw + angle)
#             oy = y0 + r * math.sin(yaw + angle)
#             obstacles.append((ox, oy))
#         return obstacles

#     def evaluate(self, traj, v, w, clearance):
#         x_end, y_end = traj[-1]
#         dx = self.goal.position.x - x_end
#         dy = self.goal.position.y - y_end
#         goal_angle = math.atan2(dy, dx)

#         traj_heading = math.atan2(traj[-1][1] - traj[0][1], traj[-1][0] - traj[0][0])
#         heading_error = abs(self.angle_diff(goal_angle, traj_heading))

#         score = (self.alpha * (math.pi - heading_error) +
#                  self.beta * clearance +
#                  self.gamma * v)
#         return score

#     def angle_diff(self, a, b):
#         diff = a - b
#         while diff > math.pi:
#             diff -= 2.0 * math.pi
#         while diff < -math.pi:
#             diff += 2.0 * math.pi
#         return diff

#     def get_yaw(self):
#         o = self.pose.orientation
#         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DWAPlanner()
#     rclpy.spin(node)
#     rclpy.shutdown()
################################################# 

################################################### working code ###################################################################################

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')

#         self.declare_parameter('robot_radius', 0.15)
#         self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value
#         self.safety_buffer = 0.02  # reduced for tighter but safer maneuvering

#         self.max_speed = 0.3
#         self.max_yawrate = 1.5
#         self.accel = 0.6
#         self.yaw_accel = 3.0
#         self.dt = 0.1
#         self.predict_time = 2.5

#         # Tuned cost weights
#         self.alpha = 1.0  # heading weight
#         self.beta = 1.5   # clearance weight
#         self.gamma = 1.0  # velocity weight

#         self.pose = None
#         self.scan = None
#         self.goal = None
#         self.velocity = (0.0, 0.0)

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
#         self.timer = self.create_timer(self.dt, self.plan)

#     def odom_callback(self, msg):
#         self.pose = msg.pose.pose
#         self.velocity = (
#             msg.twist.twist.linear.x,
#             msg.twist.twist.angular.z
#         )

#     def scan_callback(self, msg):
#         self.scan = msg

#     def goal_callback(self, msg):
#         self.goal = msg.pose

#     def plan(self):
#         if self.pose is None or self.scan is None or self.goal is None:
#             return

#         dx = self.goal.position.x - self.pose.position.x
#         dy = self.goal.position.y - self.pose.position.y
#         if math.hypot(dx, dy) < 0.2:
#             self.cmd_pub.publish(Twist())
#             self.get_logger().info("Goal reached.")
#             return

#         best_score = -float('inf')
#         best_cmd = Twist()

#         v_range = np.linspace(-0.05, self.max_speed, 12)  # include slight reverse motion
#         w_range = np.linspace(-self.max_yawrate, self.max_yawrate, 21)

#         obstacles = self.get_obstacle_positions()

#         for v in v_range:
#             for w in w_range:
#                 traj, min_dist = self.simulate_trajectory(v, w, obstacles)
#                 if traj is None:
#                     self.get_logger().debug(f"Rejected v={v:.2f}, w={w:.2f} due to collision.")
#                     continue

#                 score = self.evaluate(traj, v, w, min_dist)
#                 self.get_logger().debug(f"v={v:.2f}, w={w:.2f}, score={score:.2f}, min_dist={min_dist:.2f}")

#                 if score > best_score:
#                     best_score = score
#                     best_cmd.linear.x = v
#                     best_cmd.angular.z = w

#         if best_score == -float('inf'):
#             self.get_logger().warn("No valid trajectory. Rotating slowly to recover.")
#             best_cmd.angular.z = 0.3
#             best_cmd.linear.x = 0.0

#         self.get_logger().info(f"Selected: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f}, score={best_score:.2f}")
#         self.cmd_pub.publish(best_cmd)

#     def simulate_trajectory(self, v, w, obstacles):
#         x = self.pose.position.x
#         y = self.pose.position.y
#         yaw = self.get_yaw()

#         traj = []
#         min_dist = float('inf')
#         t = 0.0

#         while t <= self.predict_time:
#             x += v * math.cos(yaw) * self.dt
#             y += v * math.sin(yaw) * self.dt
#             yaw += w * self.dt
#             traj.append((x, y))

#             for ox, oy in obstacles:
#                 dist = math.hypot(x - ox, y - oy)
#                 if dist <= self.radius + self.safety_buffer:
#                     return None, 0.0  # collision
#                 min_dist = min(min_dist, dist)

#             dx = self.goal.position.x - x
#             dy = self.goal.position.y - y
#             if math.hypot(dx, dy) < 0.2:
#                 break

#             t += self.dt

#         return traj, min_dist

#     def get_obstacle_positions(self):
#         if self.scan is None:
#             return []

#         ranges = np.array(self.scan.ranges)
#         angle_min = self.scan.angle_min
#         angle_increment = self.scan.angle_increment
#         yaw = self.get_yaw()
#         x0 = self.pose.position.x
#         y0 = self.pose.position.y

#         obstacles = []
#         for i, r in enumerate(ranges):
#             if np.isinf(r) or np.isnan(r) or r > self.scan.range_max:
#                 continue
#             angle = angle_min + i * angle_increment
#             ox = x0 + r * math.cos(yaw + angle)
#             oy = y0 + r * math.sin(yaw + angle)
#             obstacles.append((ox, oy))
#         return obstacles

#     def evaluate(self, traj, v, w, clearance):
#         x_end, y_end = traj[-1]
#         dx = self.goal.position.x - x_end
#         dy = self.goal.position.y - y_end
#         goal_angle = math.atan2(dy, dx)

#         traj_heading = math.atan2(traj[-1][1] - traj[0][1], traj[-1][0] - traj[0][0])
#         heading_error = abs(self.angle_diff(goal_angle, traj_heading))

#         # Normalize heading error to [0, 1]
#         heading_score = 1.0 - heading_error / math.pi

#         score = (self.alpha * heading_score +
#                  self.beta * clearance +
#                  self.gamma * v)
#         return score

#     def angle_diff(self, a, b):
#         diff = a - b
#         while diff > math.pi:
#             diff -= 2.0 * math.pi
#         while diff < -math.pi:
#             diff += 2.0 * math.pi
#         return diff

#     def get_yaw(self):
#         o = self.pose.orientation
#         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DWAPlanner()
#     rclpy.spin(node)
#     rclpy.shutdown()


################################################################ modified working code  (better)###################################################

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')

#         self.declare_parameter('robot_radius', 0.15)
#         self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value
#         self.safety_buffer = 0.02

#         self.max_speed = 0.3
#         self.max_yawrate = 1.5
#         self.accel = 0.6
#         self.yaw_accel = 3.0
#         self.dt = 0.1
#         self.predict_time = 2.5

#         # Tuned weights
#         self.alpha = 1.2  # heading weight
#         self.beta = 0.8  # clearance weight
#         self.gamma = 2.0  # velocity weight

#         self.pose = None
#         self.scan = None
#         self.goal = None
#         self.velocity = (0.0, 0.0)

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
#         self.timer = self.create_timer(self.dt, self.plan)

#     def odom_callback(self, msg):
#         self.pose = msg.pose.pose
#         self.velocity = (
#             msg.twist.twist.linear.x,
#             msg.twist.twist.angular.z
#         )

#     def scan_callback(self, msg):
#         self.scan = msg

#     def goal_callback(self, msg):
#         self.goal = msg.pose

#     def plan(self):
#         if self.pose is None or self.scan is None or self.goal is None:
#             return

#         dx = self.goal.position.x - self.pose.position.x
#         dy = self.goal.position.y - self.pose.position.y
#         distance_to_goal = math.hypot(dx, dy)

#         if distance_to_goal < 0.2:
#             self.cmd_pub.publish(Twist())
#             self.get_logger().info("Goal reached.")
#             return

#         best_score = -float('inf')
#         best_cmd = Twist()

#         # Avoid near-zero and reverse motion
#         v_range = np.linspace(0.05, self.max_speed, 10)
#         w_range = np.linspace(-self.max_yawrate, self.max_yawrate, 21)

#         obstacles = self.get_obstacle_positions()

#         for v in v_range:
#             for w in w_range:
#                 traj, min_dist = self.simulate_trajectory(v, w, obstacles)
#                 if traj is None:
#                     self.get_logger().debug(f"Rejected v={v:.2f}, w={w:.2f} due to collision.")
#                     continue

#                 score = self.evaluate(traj, v, w, min_dist)
#                 self.get_logger().debug(f"v={v:.2f}, w={w:.2f}, score={score:.2f}, min_dist={min_dist:.2f}")

#                 if score > best_score:
#                     best_score = score
#                     best_cmd.linear.x = v
#                     best_cmd.angular.z = w

#         if best_score == -float('inf'):
#             self.get_logger().warn("No valid trajectory. Rotating slowly to recover.")
#             best_cmd.angular.z = 0.3
#             best_cmd.linear.x = 0.0

#         self.get_logger().info(f"Selected: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f}, score={best_score:.2f}")
#         self.cmd_pub.publish(best_cmd)

#     def simulate_trajectory(self, v, w, obstacles):
#         x = self.pose.position.x
#         y = self.pose.position.y
#         yaw = self.get_yaw()

#         traj = []
#         min_dist = float('inf')
#         t = 0.0

#         while t <= self.predict_time:
#             x += v * math.cos(yaw) * self.dt
#             y += v * math.sin(yaw) * self.dt
#             yaw += w * self.dt
#             traj.append((x, y))

#             for ox, oy in obstacles:
#                 dist = math.hypot(x - ox, y - oy)
#                 if dist <= self.radius + self.safety_buffer:
#                     return None, 0.0
#                 min_dist = min(min_dist, dist)

#             dx = self.goal.position.x - x
#             dy = self.goal.position.y - y
#             if math.hypot(dx, dy) < 0.2:
#                 break

#             t += self.dt

#         return traj, min_dist

#     def get_obstacle_positions(self):
#         if self.scan is None:
#             return []

#         ranges = np.array(self.scan.ranges)
#         angle_min = self.scan.angle_min
#         angle_increment = self.scan.angle_increment
#         yaw = self.get_yaw()
#         x0 = self.pose.position.x
#         y0 = self.pose.position.y

#         obstacles = []
#         for i, r in enumerate(ranges):
#             if np.isinf(r) or np.isnan(r) or r > self.scan.range_max:
#                 continue
#             angle = angle_min + i * angle_increment
#             ox = x0 + r * math.cos(yaw + angle)
#             oy = y0 + r * math.sin(yaw + angle)
#             obstacles.append((ox, oy))
#         return obstacles

#     def evaluate(self, traj, v, w, clearance):
#         x_end, y_end = traj[-1]
#         dx = self.goal.position.x - x_end
#         dy = self.goal.position.y - y_end
#         goal_angle = math.atan2(dy, dx)

#         traj_heading = math.atan2(traj[-1][1] - traj[0][1], traj[-1][0] - traj[0][0])
#         heading_error = abs(self.angle_diff(goal_angle, traj_heading))
#         heading_score = 1.0 - heading_error / math.pi  # normalize to [0,1]

#         # Optional: discourage aggressive spinning
#         angular_penalty = 0.3 * abs(w)

#         score = (self.alpha * heading_score +
#                  self.beta * clearance +
#                  self.gamma * v -
#                  angular_penalty)
#         return score

#     def angle_diff(self, a, b):
#         diff = a - b
#         while diff > math.pi:
#             diff -= 2.0 * math.pi
#         while diff < -math.pi:
#             diff += 2.0 * math.pi
#         return diff

#     def get_yaw(self):
#         o = self.pose.orientation
#         siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DWAPlanner()
#     rclpy.spin(node)
#     rclpy.shutdown()

##################################################################### modified third iteration ############################################

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class DWAPlanner(Node):
    def __init__(self):
        super().__init__('dwa_planner')

        self.declare_parameter('robot_radius', 0.15)
        self.radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.safety_buffer = 0.05  # increased buffer

        self.max_speed = 0.3
        self.max_yawrate = 1.5
        self.accel = 0.6
        self.yaw_accel = 3.0
        self.dt = 0.1
        self.predict_time = 2.5

        # Tuned weights
        self.alpha = 1.2   # heading
        self.beta = 0.8    # clearance (increased)
        self.gamma = 2.0   # velocity

        self.pose = None
        self.scan = None
        self.goal = None
        self.velocity = (0.0, 0.0)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.timer = self.create_timer(self.dt, self.plan)

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        self.velocity = (
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        )

    def scan_callback(self, msg):
        self.scan = msg

    def goal_callback(self, msg):
        self.goal = msg.pose

    def plan(self):
        if self.pose is None or self.scan is None or self.goal is None:
            return

        dx = self.goal.position.x - self.pose.position.x
        dy = self.goal.position.y - self.pose.position.y
        distance_to_goal = math.hypot(dx, dy)

        if distance_to_goal < 0.2:
            self.cmd_pub.publish(Twist())
            self.get_logger().info("Goal reached.")
            return

        best_score = -float('inf')
        best_cmd = Twist()

        v_range = np.linspace(0.05, self.max_speed, 10)
        w_range = np.linspace(-self.max_yawrate, self.max_yawrate, 21)

        obstacles = self.get_obstacle_positions()

        for v in v_range:
            for w in w_range:
                traj, min_dist = self.simulate_trajectory(v, w, obstacles)
                if traj is None:
                    continue

                score = self.evaluate(traj, v, w, min_dist)

                if score > best_score:
                    best_score = score
                    best_cmd.linear.x = v
                    best_cmd.angular.z = w

        if best_score == -float('inf'):
            self.get_logger().warn("No valid trajectory. Rotating slowly to recover.")
            best_cmd.angular.z = 0.3
            best_cmd.linear.x = 0.0

        self.get_logger().info(f"Selected: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f}, score={best_score:.2f}")
        self.cmd_pub.publish(best_cmd)

    def simulate_trajectory(self, v, w, obstacles):
        x = self.pose.position.x
        y = self.pose.position.y
        yaw = self.get_yaw()

        traj = []
        min_dist = float('inf')
        t = 0.0

        while t <= self.predict_time:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            traj.append((x, y))

            for ox, oy in obstacles:
                dist = math.hypot(x - ox, y - oy)
                if dist <= self.radius + self.safety_buffer:
                    return None, 0.0
                min_dist = min(min_dist, dist)

            t += self.dt

        return traj, min_dist

    def get_obstacle_positions(self):
        if self.scan is None:
            return []

        ranges = np.array(self.scan.ranges)
        angle_min = self.scan.angle_min
        angle_increment = self.scan.angle_increment
        yaw = self.get_yaw()
        x0 = self.pose.position.x
        y0 = self.pose.position.y

        obstacles = []
        for i, r in enumerate(ranges):
            if np.isinf(r) or np.isnan(r) or r < self.scan.range_min or r > self.scan.range_max:
                continue
            r = min(r, 2.5)  # cap range to 2.5m
            angle = angle_min + i * angle_increment
            ox = x0 + r * math.cos(yaw + angle)
            oy = y0 + r * math.sin(yaw + angle)
            obstacles.append((ox, oy))
        return obstacles

    def evaluate(self, traj, v, w, clearance):
        x_end, y_end = traj[-1]
        dx = self.goal.position.x - x_end
        dy = self.goal.position.y - y_end
        goal_angle = math.atan2(dy, dx)

        traj_heading = math.atan2(traj[-1][1] - traj[0][1], traj[-1][0] - traj[0][0])
        heading_error = abs(self.angle_diff(goal_angle, traj_heading))
        heading_score = 1.0 - heading_error / math.pi

        clearance = max(clearance, 0.05)  # clamp to avoid too low

        angular_penalty = 0.3 * abs(w)
        obstacle_penalty = 5.0 if clearance < 0.15 else 0.0

        score = (self.alpha * heading_score +
                 self.beta * clearance +
                 self.gamma * v -
                 angular_penalty -
                 obstacle_penalty)
        return score

    def angle_diff(self, a, b):
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff

    def get_yaw(self):
        o = self.pose.orientation
        siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = DWAPlanner()
    rclpy.spin(node)
    rclpy.shutdown()




# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import Twist, PoseStamped
# # from nav_msgs.msg import Odometry
# # from sensor_msgs.msg import LaserScan
# # import numpy as np
# # import math

# # class DWAPlanner(Node):
# #     def __init__(self):
# #         super().__init__('dwa_planner')

# #         # ===== TurtleBot3 Burger Specific Parameters =====
# #         self.robot_radius = 0.105  # Measured physical radius
# #         self.safety_buffer = 0.07   # Tight but safe buffer
        
# #         # Velocity limits (conservative for Burger)
# #         self.max_speed = 0.22      # Max linear (m/s)
# #         self.min_speed = -0.02     # Small reverse
# #         self.max_yawrate = 1.5      # Reduced from 2.84 for stability
# #         self.max_accel = 0.08        # Gentle acceleration
# #         self.max_dyawrate = 1.0     # Smoother turns
        
# #         # DWA Configuration
# #         self.dt = 0.1              # Time step
# #         self.predict_time = 1.0     # Shorter horizon for quick reactions
        
# #         # Scoring weights (tuned through testing)
# #         self.goal_weight = 2.5    # Strong goal attraction
# #         self.speed_weight = 0.1    # Mild speed preference
# #         self.obstacle_weight = 0.7 # Balanced obstacle avoidance

# #         # ===== State Management =====
# #         self.pose = None
# #         self.scan = None
# #         self.goal = None
# #         self.current_v = 0.0
# #         self.current_w = 0.0
# #         self.last_cmd = Twist()
        
# #         # ===== ROS Setup =====
# #         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
# #         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
# #         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
# #         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
# #         self.timer = self.create_timer(self.dt, self.plan)

# #     # ===== Core Callbacks =====
# #     def odom_callback(self, msg):
# #         self.pose = msg.pose.pose
# #         self.current_v = msg.twist.twist.linear.x
# #         self.current_w = msg.twist.twist.angular.z

# #     def scan_callback(self, msg):
# #         # Clean scan data (Burger's LDS-01 specific)
# #         ranges = np.array(msg.ranges)
# #         ranges[np.isnan(ranges)] = msg.range_max
# #         ranges[ranges < 0.05] = msg.range_max  # Remove too-close noise
# #         msg.ranges = ranges.tolist()
# #         self.scan = msg

# #     def goal_callback(self, msg):
# #         self.goal = msg.pose
# #         self.get_logger().info(f"New goal received: ({self.goal.position.x:.2f}, {self.goal.position.y:.2f})")

# #     # ===== Main Planning Loop =====
# #     def plan(self):
# #         # Check for missing data
# #         if None in [self.pose, self.scan, self.goal]:
# #             return

# #         # Goal check (with hysteresis)
# #         dx = self.goal.position.x - self.pose.position.x
# #         dy = self.goal.position.y - self.pose.position.y
# #         if math.hypot(dx, dy) < 0.1:
# #             self.stop_robot()
# #             return

# #         # ===== Dynamic Window Calculation =====
# #         # Expanded window for better escape options
# #         v_samples = np.linspace(
# #             max(self.min_speed, self.current_v - self.max_accel * self.dt * 2),
# #             min(self.max_speed, self.current_v + self.max_accel * self.dt * 2),
# #             10
# #         )
# #         w_samples = np.linspace(
# #             max(-self.max_yawrate, self.current_w - self.max_dyawrate * self.dt * 2),
# #             min(self.max_yawrate, self.current_w + self.max_dyawrate * self.dt * 2),
# #             20
# #         )

# #         # ===== Trajectory Evaluation =====
# #         obstacles = self.get_obstacle_positions()
# #         best_score = -float('inf')
# #         best_cmd = Twist()
        
# #         for v in v_samples:
# #             for w in w_samples:
# #                 # Skip near-zero velocities
# #                 if abs(v) < 0.01 and abs(w) < 0.01:
# #                     continue
                    
# #                 traj, min_dist = self.simulate_trajectory(v, w, obstacles)
# #                 if traj is None:  # Collision
# #                     continue

# #                 # ===== Scoring =====
# #                 # Goal distance score (exponential decay)
# #                 goal_dist = math.hypot(
# #                     self.goal.position.x - traj[-1][0],
# #                     self.goal.position.y - traj[-1][1]
# #                 )
# #                 goal_score = math.exp(-goal_dist)
                
# #                 # Heading alignment score
# #                 goal_angle = math.atan2(dy, dx)
# #                 traj_angle = math.atan2(
# #                     traj[-1][1] - self.pose.position.y,
# #                     traj[-1][0] - self.pose.position.x
# #                 )
# #                 heading_score = 1.0 - abs(self.normalize_angle(goal_angle - traj_angle)) / math.pi
                
# #                 # Clearance score (cap at 1m)
# #                 clearance_score = min(min_dist, 1.0)
                
# #                 # Speed score (prefer forward motion)
# #                 speed_score = max(0, v) / self.max_speed
                
# #                 # Combined weighted score
# #                 total_score = (
# #                     self.goal_weight * goal_score +
# #                     self.goal_weight * 0.5 * heading_score +
# #                     self.speed_weight * speed_score +
# #                     self.obstacle_weight * clearance_score
# #                 )

# #                 # Debug logging
# #                 if total_score > best_score:
# #                     best_score = total_score
# #                     best_cmd.linear.x = v
# #                     best_cmd.angular.z = w

# #         # ===== Fallback Behavior =====
# #         if best_score == -float('inf'):
# #             self.get_logger().warn("No valid path - executing recovery")
# #             best_cmd.linear.x = -0.02  # Tiny reverse
# #             best_cmd.angular.z = 0.5   # Aggressive turn
# #             if self.last_cmd.angular.z != 0:
# #                 best_cmd.angular.z *= 1.2  # Increase turn if already rotating

# #         # Publish and remember last command
# #         self.last_cmd = best_cmd
# #         self.cmd_pub.publish(best_cmd)
# #         self.get_logger().info(
# #             f"CMD: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f} | "
# #             f"Goal: {goal_dist:.2f}m | "
# #             f"Obstacles: {len(obstacles)}"
# #         )

# #     # ===== Helper Methods =====
# #     def simulate_trajectory(self, v, w, obstacles):
# #         x, y = self.pose.position.x, self.pose.position.y
# #         yaw = self.get_yaw()
# #         min_dist = float('inf')
        
# #         for _ in range(int(self.predict_time / self.dt)):
# #             # Kinematic model
# #             x += v * math.cos(yaw) * self.dt
# #             y += v * math.sin(yaw) * self.dt
# #             yaw += w * self.dt
            
# #             # Collision check with dynamic buffer
# #             for ox, oy in obstacles:
# #                 dist = math.hypot(x - ox, y - oy)
# #                 if dist < self.robot_radius + self.safety_buffer * (1 + abs(v)):  # Speed-adaptive buffer
# #                     return None, 0.0
# #                 min_dist = min(min_dist, dist)
            
# #             # Early termination near goal
# #             if math.hypot(x - self.goal.position.x, y - self.goal.position.y) < 0.15:
# #                 break
                
# #         return (x, y), min_dist

# #     def get_obstacle_positions(self):
# #         if self.scan is None:
# #             return []

# #         obstacles = []
# #         angle = self.scan.angle_min
# #         for r in self.scan.ranges:
# #             if r < self.scan.range_max * 0.8:  # Ignore distant obstacles
# #                 ox = self.pose.position.x + r * math.cos(self.get_yaw() + angle)
# #                 oy = self.pose.position.y + r * math.sin(self.get_yaw() + angle)
# #                 obstacles.append((ox, oy))
# #             angle += self.scan.angle_increment
# #         return obstacles

# #     def stop_robot(self):
# #         cmd = Twist()
# #         self.cmd_pub.publish(cmd)
# #         self.get_logger().info("Goal reached - robot stopped")

# #     @staticmethod
# #     def normalize_angle(angle):
# #         return math.atan2(math.sin(angle), math.cos(angle))

# #     def get_yaw(self):
# #         q = self.pose.orientation
# #         return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = DWAPlanner()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         node.stop_robot()
# #     finally:
# #         rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# import numpy as np
# import math

# # Optional: For visualizing trajectories in RViz
# from nav_msgs.msg import Path
# from geometry_msgs.msg import PoseStamped as Ros2PoseStamped # Rename to avoid conflict
# import time # For unique timestamp in debug messages

# class DWAPlanner(Node):
#     def __init__(self):
#         super().__init__('dwa_planner')
        
#         # Parameters (tuned for TurtleBot3 Burger)
#         self.declare_parameters(
#             namespace='',
#             parameters=[
#                 ('robot_radius', 0.105),
#                 ('safety_buffer', 0.08),
#                 ('max_speed', 0.22), # m/s
#                 ('max_yaw_rate', 1.5), # rad/s
#                 ('max_accel_v', 0.5), # m/s^2 - Added explicit acceleration limits
#                 ('max_accel_w', 2.0), # rad/s^2 - Added explicit acceleration limits
#                 ('sim_time', 1.5), # seconds, how far to simulate
#                 ('time_step', 0.1), # seconds, simulation granularity
#                 ('v_samples', 7), # Number of linear velocity samples in dynamic window
#                 ('w_samples', 15), # Number of angular velocity samples in dynamic window
#                 ('heading_weight', 1.5),
#                 ('dist_weight', 1.0),
#                 ('clearance_weight', 0.8),
#                 ('speed_weight', 0.1)
#             ]
#         )
        
#         # ROS2 interfaces
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        
#         # Debugging / Visualization publishers (optional but highly recommended)
#         self.debug_path_pub = self.create_publisher(Path, '/dwa_debug_path', 1)
#         self.best_trajectory_pub = self.create_publisher(Path, '/dwa_best_trajectory', 1)

#         # State variables
#         self.current_pose = None
#         self.current_vel = [0.0, 0.0]  # [v, w]
#         self.scan_data = None
#         self.goal = None
        
#         # Main control loop
#         # Accessing parameter for timer directly here is fine
#         self.control_timer = self.create_timer(
#             self.get_parameter('time_step').value, 
#             self.execute_control_loop
#         )
        
#         self.get_logger().info("DWA Planner initialized")

#     def odom_cb(self, msg):
#         """Handle odometry updates"""
#         self.current_pose = msg.pose.pose
#         self.current_vel = [
#             msg.twist.twist.linear.x,
#             msg.twist.twist.angular.z
#         ]

#     def scan_cb(self, msg):
#         """Process laser scan data"""
#         # Convert infinite values to max range
#         ranges = np.array(msg.ranges)
#         # Ensure all invalid ranges (inf or NaN) are set to max_range
#         ranges[np.isinf(ranges)] = msg.range_max
#         ranges[np.isnan(ranges)] = msg.range_max 
#         self.scan_data = msg
#         self.scan_data.ranges = ranges.tolist() # Update the ranges in the message object

#     def goal_cb(self, msg):
#         """Handle new goal poses"""
#         if self.current_pose is None:
#             self.get_logger().warn("Received goal before localization! Please wait for /odom.")
#             return
            
#         self.goal = msg.pose
#         dx = self.goal.position.x - self.current_pose.position.x
#         dy = self.goal.position.y - self.current_pose.position.y
#         self.get_logger().info(f"New goal accepted! Distance: {math.hypot(dx, dy):.2f}m")

#     def execute_control_loop(self):
#         """Main DWA control loop"""
#         # Check for required data
#         if None in [self.current_pose, self.scan_data, self.goal]:
#             # self.get_logger().info("Waiting for data: current_pose, scan_data, goal")
#             return

#         # Calculate dynamic window
#         dw = self.calculate_dynamic_window()
        
#         # Evaluate trajectories and select best command
#         best_cmd, best_score, best_trajectory_points = self.evaluate_trajectories(dw)
        
#         # Execute command
#         self.execute_command(best_cmd, best_score)

#         # Publish best trajectory for visualization (optional)
#         if best_trajectory_points:
#             path_msg = Path()
#             path_msg.header.stamp = self.get_clock().now().to_msg()
#             # It's usually best to use the 'map' or 'odom' frame for paths,
#             # depending on what frame your current_pose is in.
#             # Assuming current_pose is in the 'odom' frame for TurtleBot3
#             path_msg.header.frame_id = self.current_pose.header.frame_id if hasattr(self.current_pose, 'header') else 'odom' 
#             for point in best_trajectory_points:
#                 pose_stamped = Ros2PoseStamped()
#                 pose_stamped.header.stamp = path_msg.header.stamp
#                 pose_stamped.header.frame_id = path_msg.header.frame_id
#                 pose_stamped.pose.position.x = point[0]
#                 pose_stamped.pose.position.y = point[1]
#                 # Orientation is not explicitly stored in path points here, but could be added
#                 path_msg.poses.append(pose_stamped)
#             self.best_trajectory_pub.publish(path_msg)


#     def calculate_dynamic_window(self):
#         """Generate admissible velocity space"""
#         # Retrieve parameters as a list of Parameter objects
#         raw_params = self.get_parameters([
#             'max_speed', 'max_yaw_rate', 'max_accel_v', 'max_accel_w', 'time_step',
#             'v_samples', 'w_samples'
#         ])
        
#         # Convert the list of Parameter objects into a dictionary for easier access
#         params = {p.name: p.value for p in raw_params}
        
#         dt = params['time_step'] # Now you can access it by string key, e.g., params['time_step']
        
#         # 1. Robot Capabilities Limits (absolute limits of the robot)
#         v_robot_min = 0.0 # Assuming robot cannot move backward for now
#         v_robot_max = params['max_speed']
#         w_robot_min = -params['max_yaw_rate']
#         w_robot_max = params['max_yaw_rate']
        
#         # 2. Acceleration Limits (kinematic window - what speeds are reachable in the next dt)
#         v_accel_min = self.current_vel[0] - params['max_accel_v'] * dt
#         v_accel_max = self.current_vel[0] + params['max_accel_v'] * dt
#         w_accel_min = self.current_vel[1] - params['max_accel_w'] * dt
#         w_accel_max = self.current_vel[1] + params['max_accel_w'] * dt

#         # Combine all limits to get the final dynamic window
#         # The true dynamic window is the intersection of these limits
#         v_min_dw = max(v_robot_min, v_accel_min)
#         v_max_dw = min(v_robot_max, v_accel_max)
#         w_min_dw = max(w_robot_min, w_accel_min)
#         w_max_dw = min(w_robot_max, w_accel_max)

#         # Ensure that min is not greater than max due to aggressive limits or floating point issues
#         if v_min_dw > v_max_dw: v_min_dw = v_max_dw
#         if w_min_dw > w_max_dw: w_min_dw = w_max_dw

#         # Generate samples within the dynamic window
#         v_samples_array = np.linspace(v_min_dw, v_max_dw, params['v_samples'])
#         w_samples_array = np.linspace(w_min_dw, w_max_dw, params['w_samples'])
        
#         # self.get_logger().info(f"DW: v_range=[{v_min_dw:.2f}, {v_max_dw:.2f}], w_range=[{w_min_dw:.2f}, {w_max_dw:.2f}]")
        
#         return {
#             'v_samples': v_samples_array,
#             'w_samples': w_samples_array
#         }

#     def evaluate_trajectories(self, dw):
#         """Score all possible trajectories in dynamic window"""
#         best_cmd = Twist()
#         best_score = -float('inf')
#         best_trajectory_points = None # To store points for visualization

#         # Retrieve weights from parameters
#         raw_weights = self.get_parameters([
#             'heading_weight', 'dist_weight', 'clearance_weight', 'speed_weight'
#         ])
#         weights = {p.name: p.value for p in raw_weights}
        
#         # For debug visualization of all trajectories (can be heavy on performance)
#         # debug_path_msg = Path()
#         # debug_path_msg.header.stamp = self.get_clock().now().to_msg()
#         # debug_path_msg.header.frame_id = self.current_pose.header.frame_id if hasattr(self.current_pose, 'header') else 'odom'

#         for v in dw['v_samples']:
#             for w in dw['w_samples']:
#                 # Simulate trajectory
#                 traj_end_pose, min_clearance_along_traj, trajectory_points = self.simulate_trajectory(v, w)
                
#                 # if trajectory_points: # For debugging all trajectories
#                 #     # Append these points to debug_path_msg or publish separate markers
#                 #     for point in trajectory_points:
#                 #         pose_stamped = Ros2PoseStamped()
#                 #         pose_stamped.header.stamp = debug_path_msg.header.stamp
#                 #         pose_stamped.header.frame_id = debug_path_msg.header.frame_id
#                 #         pose_stamped.pose.position.x = point[0]
#                 #         pose_stamped.pose.position.y = point[1]
#                 #         debug_path_msg.poses.append(pose_stamped)

#                 if traj_end_pose is None:  # Collision detected or simulation failed
#                     # self.get_logger().debug(f"Collision for v={v:.2f}, w={w:.2f}")
#                     continue
                
#                 # Calculate score
#                 # Pass the weights dictionary to calculate_score
#                 score = self.calculate_score(traj_end_pose, v, w, min_clearance_along_traj, weights)
                
#                 if score > best_score:
#                     best_score = score
#                     best_cmd.linear.x = v
#                     best_cmd.angular.z = w
#                     best_trajectory_points = trajectory_points # Store for visualization
        
#         # if debug_path_msg.poses: # For debugging all trajectories
#         #     self.debug_path_pub.publish(debug_path_msg)

#         return best_cmd, best_score, best_trajectory_points

#     def simulate_trajectory(self, v, w):
#         """Predict robot motion for given velocities and check for collisions.
#            Returns (final_x, final_y), min_distance_along_path, list_of_points_on_path
#         """
#         # Retrieve parameters needed for simulation
#         robot_radius = self.get_parameter('robot_radius').value
#         safety_buffer = self.get_parameter('safety_buffer').value
#         sim_time = self.get_parameter('sim_time').value
#         time_step = self.get_parameter('time_step').value
        
#         if self.scan_data is None or self.current_pose is None:
#             return None, 0.0, None # Cannot simulate without required data
            
#         x_current = self.current_pose.position.x
#         y_current = self.current_pose.position.y
#         yaw_current = self.get_yaw()

#         # Convert current laser scan points to global coordinates (once per simulation call)
#         # These are static obstacles for the current simulation.
#         obstacle_points_global = []
#         # Adjusting max range filtering slightly for robust obstacle detection
#         effective_max_range = self.scan_data.range_max * 0.95 
#         for i, r in enumerate(self.scan_data.ranges):
#             # Only consider valid ranges within the min and effective max limits
#             if r >= self.scan_data.range_min and r <= effective_max_range:
#                 # Angle of the laser beam in the robot's current frame
#                 angle_in_robot_frame = self.scan_data.angle_min + i * self.scan_data.angle_increment
                
#                 # Point in robot's current local frame
#                 px_local = r * math.cos(angle_in_robot_frame)
#                 py_local = r * math.sin(angle_in_robot_frame)

#                 # Rotate and translate to global frame based on current robot pose
#                 # Transformation: P_global = R_z(yaw_current) * P_local + P_current_pose_translation
#                 ox_global = x_current + (px_local * math.cos(yaw_current) - py_local * math.sin(yaw_current))
#                 oy_global = y_current + (px_local * math.sin(yaw_current) + py_local * math.cos(yaw_current))
#                 obstacle_points_global.append((ox_global, oy_global))

#         # Initialize simulated pose
#         x_sim, y_sim, yaw_sim = x_current, y_current, yaw_current
#         min_clearance_along_traj = float('inf') # Tracks minimum distance to any obstacle along the path
#         trajectory_points = [] # Stores points for visualization

#         num_sim_steps = int(sim_time / time_step)
        
#         # Add current pose to trajectory points for visualization
#         trajectory_points.append((x_sim, y_sim))

#         for step in range(num_sim_steps):
#             # Kinematic model update for the simulated robot
#             x_sim += v * math.cos(yaw_sim) * time_step
#             y_sim += v * math.sin(yaw_sim) * time_step
#             yaw_sim += w * time_step
            
#             trajectory_points.append((x_sim, y_sim))

#             # --- Collision Check for current simulated robot pose ---
#             current_step_min_dist_to_obstacle = float('inf')
#             # Only check if there are obstacles
#             if obstacle_points_global:
#                 for obs_x, obs_y in obstacle_points_global:
#                     dist_to_obs = math.hypot(x_sim - obs_x, y_sim - obs_y)
#                     current_step_min_dist_to_obstacle = min(current_step_min_dist_to_obstacle, dist_to_obs)
#             else: # No obstacles, so assume infinite clearance
#                 current_step_min_dist_to_obstacle = float('inf') 
            
#             # Update the overall minimum distance found along the entire trajectory
#             min_clearance_along_traj = min(min_clearance_along_traj, current_step_min_dist_to_obstacle)

#             # If a collision is detected at this point in the trajectory, return None
#             # Collision if robot's current simulated position is too close to an obstacle point
#             if current_step_min_dist_to_obstacle < (robot_radius + safety_buffer):
#                 # self.get_logger().debug(f"Collision detected for v={v:.2f}, w={w:.2f} at step {step} with dist {current_step_min_dist_to_obstacle:.3f}")
#                 return None, 0.0, None # Collision detected, invalid trajectory

#         # If no collision occurred, return the end pose of the trajectory and the minimum clearance
#         return (x_sim, y_sim), min_clearance_along_traj, trajectory_points

#     def calculate_score(self, traj_end_pose, v, w, clearance, weights):
#         """Evaluate trajectory quality"""
#         # Get weights from the passed dictionary
#         heading_weight = weights['heading_weight']
#         dist_weight = weights['dist_weight']
#         clearance_weight = weights['clearance_weight']
#         speed_weight = weights['speed_weight']
#         max_speed = self.get_parameter('max_speed').value # Retrieve max_speed again here

#         # Goal distance score: inversely proportional to distance to goal
#         dx = self.goal.position.x - traj_end_pose[0]
#         dy = self.goal.position.y - traj_end_pose[1]
#         dist_to_goal = math.hypot(dx, dy)
#         dist_score = 1.0 / (1.0 + dist_to_goal) # Max 1.0 when at goal
        
#         # Heading alignment score: how well the trajectory points towards the goal
#         # Angle from current robot pose to goal
#         goal_angle = math.atan2(self.goal.position.y - self.current_pose.position.y, 
#                                 self.goal.position.x - self.current_pose.position.x)
        
#         # Angle from current robot pose to simulated trajectory end point
#         traj_angle = math.atan2(traj_end_pose[1] - self.current_pose.position.y, 
#                                traj_end_pose[0] - self.current_pose.position.x)
        
#         angle_diff = self.normalize_angle(goal_angle - traj_angle)
#         heading_score = 1.0 - (abs(angle_diff) / math.pi) # Max 1.0 when perfectly aligned, 0.0 when 180 deg off

#         # Clearance score: penalize trajectories that get too close to obstacles
#         # Cap clearance at a reasonable value (e.g., 1.0m) to prevent distant obstacles from dominating
#         # Normalize by a maximum expected useful clearance, e.g., 1.0 meter
#         clearance_score = min(clearance, 1.0) # Max 1.0 when clearance >= 1m
        
#         # Ensure clearance score is not NaN if clearance is inf
#         if math.isinf(clearance_score):
#             clearance_score = 1.0 # Treat infinite clearance as perfect

#         # Speed score: incentivize higher speeds towards the goal
#         speed_score = v / max_speed # Normalized, max 1.0 at max speed
        
#         # Weighted sum of scores
#         total_score = (
#             dist_weight * dist_score + 
#             heading_weight * heading_score + 
#             clearance_weight * clearance_score + 
#             speed_weight * speed_score
#         )
        
#         # self.get_logger().info(f"v={v:.2f}, w={w:.2f} Scores: d={dist_score:.2f}, h={heading_score:.2f}, c={clearance_score:.2f}, s={speed_score:.2f} -> Total: {total_score:.2f}")

#         return total_score

#     def execute_command(self, cmd, score):
#         """Publish commands with safety checks"""
#         # Retrieve parameters needed for command execution
#         max_speed = self.get_parameter('max_speed').value
#         max_yaw_rate = self.get_parameter('max_yaw_rate').value

#         # Check if the goal is reached
#         if self.current_pose and self.goal:
#             dx_goal = self.goal.position.x - self.current_pose.position.x
#             dy_goal = self.goal.position.y - self.current_pose.position.y
#             distance_to_goal = math.hypot(dx_goal, dy_goal)
            
#             # Simple goal threshold, can be made more sophisticated (e.g., orientation)
#             if distance_to_goal < 0.1: # meters
#                 cmd.linear.x = 0.0
#                 cmd.angular.z = 0.0
#                 self.get_logger().info("Goal reached!")
#                 self.goal = None # Clear the goal so it stops trying to plan
#                 self.cmd_vel_pub.publish(cmd)
#                 return # Stop and exit

#         if score == -float('inf'):
#             # Recovery behavior: No valid trajectory found (likely blocked)
#             cmd_recovery = Twist()
#             cmd_recovery.linear.x = 0.0
#             cmd_recovery.angular.z = 0.3 # Rotate slowly to find an opening
#             self.get_logger().warn(f"[{time.time():.2f}] No valid trajectory found. Executing recovery turn.")
#             self.cmd_vel_pub.publish(cmd_recovery)
#         else:
#             # Apply speed limits (already handled by dynamic window, but good final clip)
#             cmd.linear.x = np.clip(
#                 cmd.linear.x,
#                 0.0, # Assuming robot cannot move backward
#                 max_speed
#             )
#             cmd.angular.z = np.clip(
#                 cmd.angular.z,
#                 -max_yaw_rate,
#                 max_yaw_rate
#             )
#             self.cmd_vel_pub.publish(cmd)

#     @staticmethod
#     def normalize_angle(angle):
#         """Normalize angle to [-π, π]"""
#         return math.atan2(math.sin(angle), math.cos(angle))

#     def get_yaw(self):
#         """Extract yaw from quaternion"""
#         if self.current_pose is None:
#             return 0.0 # Default if pose isn't available yet
#         q = self.current_pose.orientation
#         # Yaw (z-axis rotation) from quaternion
#         siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
#         cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
#         return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     planner = DWAPlanner()
    
#     try:
#         rclpy.spin(planner)
#     except KeyboardInterrupt:
#         planner.get_logger().info("Shutting down planner...")
#     finally:
#         planner.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()