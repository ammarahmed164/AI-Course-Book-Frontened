# Isaac ROS Navigation Example
# This demonstrates basic navigation concepts using Isaac ROS

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class SimpleNavigationController(Node):
    def __init__(self):
        super().__init__('simple_navigation_controller')
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Navigation parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.3  # rad/s
        self.safe_distance = 0.5  # meters
        
        # Robot state
        self.obstacle_detected = False
        self.scan_ranges = []

    def scan_callback(self, msg):
        """Process laser scan data."""
        self.scan_ranges = msg.ranges
        
        # Check for obstacles in front of the robot
        front_ranges = self.scan_ranges[len(self.scan_ranges)//2-30:len(self.scan_ranges)//2+30]
        min_front_distance = min([r for r in front_ranges if not math.isinf(r) and not math.isnan(r)], default=float('inf'))
        
        self.obstacle_detected = min_front_distance < self.safe_distance

    def control_loop(self):
        """Main navigation control loop."""
        cmd_vel = Twist()
        
        if self.obstacle_detected:
            # Stop and rotate to find clear path
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed
        else:
            # Move forward safely
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleNavigationController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()