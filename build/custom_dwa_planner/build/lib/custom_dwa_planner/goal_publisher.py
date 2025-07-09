import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')

        # Publish goal
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Publish goal marker
        self.marker_pub = self.create_publisher(Marker, '/goal_marker', 10)

        # Publish trajectory marker
        self.trajectory_pub = self.create_publisher(Marker, '/trajectory_marker', 10)

        # Subscribe to robot's odometry
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.timer = self.create_timer(2.0, self.publish_goal)

        self.trajectory_points = []

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.pose.position.x = 2.0
        goal.pose.position.y = 1.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.publish_goal_marker(goal.pose.position.x, goal.pose.position.y)
        self.get_logger().info("Published goal and marker")

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # red
        marker.lifetime.sec = 0
        self.marker_pub.publish(marker)

    def odom_callback(self, msg):
        pt = Point()
        pt.x = msg.pose.pose.position.x
        pt.y = msg.pose.pose.position.y
        pt.z = 0.05

        self.trajectory_points.append(pt)
        if len(self.trajectory_points) > 500:
            self.trajectory_points.pop(0)

        self.publish_trajectory_marker()

    def publish_trajectory_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.ns = "trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # line width
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # green
        marker.points = self.trajectory_points
        marker.lifetime.sec = 0
        self.trajectory_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    rclpy.shutdown()
