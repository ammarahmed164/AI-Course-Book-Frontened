# Chapter 2: ROS Pipelines for Perception and Visual SLAM (VSLAM)

## Learning Objectives

After completing this chapter, you will be able to:
- Implement perception pipelines using ROS and Isaac ROS
- Understand Visual SLAM (VSLAM) concepts and applications
- Configure and use Isaac ROS for perception tasks
- Integrate perception results with navigation systems
- Evaluate perception pipeline performance

## Introduction to Perception in Robotics

Perception is the ability of a robot to interpret its environment through sensor data. In robotics, perception systems enable:
- Object detection and recognition
- Environment mapping
- Localization
- Scene understanding

### Perception Pipeline Overview

A typical robotics perception pipeline includes:
1. **Data Acquisition**: Collecting sensor data (cameras, LIDAR, etc.)
2. **Preprocessing**: Filtering, calibration, synchronization
3. **Feature Extraction**: Identifying key features in the data
4. **Interpretation**: Understanding the scene and objects
5. **Decision Making**: Using perception data for navigation or interaction

## Isaac ROS Framework

Isaac ROS is NVIDIA's collection of hardware-accelerated perception packages for ROS and ROS2, designed to run on NVIDIA Jetson platforms and GPUs.

### Key Isaac ROS Packages

1. **Isaac ROS Image Pipeline**: Hardware-accelerated image preprocessing
2. **Isaac ROS AprilTag**: Marker detection and pose estimation
3. **Isaac ROS DNN Inference**: Deep learning model inference
4. **Isaac ROS Visual SLAM**: Visual Simultaneous Localization and Mapping
5. **Isaac ROS ISAAC ROS Manipulator**: Manipulation and grasping algorithms

### Installing Isaac ROS

```bash
# Add NVIDIA's ROS2 repository
sudo apt update && sudo apt install curl gnupg2 lsb-release
curl -sL https://nvidia.github.io/nvidia-ros2-repos/debSh/setup.sh | sudo bash

# Install Isaac ROS packages
sudo apt update
sudo apt install nvidia-isaac-ros-nitros
sudo apt install nvidia-isaac-ros-visual-slam
sudo apt install nvidia-isaac-ros-test
```

## Isaac ROS Image Pipeline

The Isaac ROS Image Pipeline provides hardware-accelerated image preprocessing:

```python
# image_pipeline_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImagePipelineNode(Node):
    def __init__(self):
        super().__init__('image_pipeline_node')
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.processed_pub = self.create_publisher(
            Image,
            '/camera/rgb/processed',
            10
        )

        self.get_logger().info("Image Pipeline Node Started")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform image processing (e.g., noise reduction, contrast enhancement)
            processed_image = self.process_image(cv_image)

            # Convert back to ROS Image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header  # Preserve header

            # Publish processed image
            self.processed_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def process_image(self, image):
        # Example processing: apply Gaussian blur and adjust contrast
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Adjust contrast (alpha controls contrast, beta controls brightness)
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 0     # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

        return adjusted

def main(args=None):
    rclpy.init(args=args)
    node = ImagePipelineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Visual SLAM (VSLAM) Concepts

Visual SLAM is the process of estimating the pose of a camera simultaneously with mapping the environment using only visual input.

### Key Components of VSLAM

1. **Feature Detection**: Identifying distinctive points in images
2. **Feature Matching**: Finding correspondences across frames
3. **Pose Estimation**: Computing camera pose using matched features
4. **Mapping**: Building a map of the environment
5. **Loop Closure**: Detecting when the robot revisits a location

### Isaac ROS Visual SLAM

```python
# visual_slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np

class VisualSLAMNode(Node):
    def __init__(self):
        super().__init__('visual_slam_node')
        self.bridge = CvBridge()

        # Subscribers for stereo camera or RGBD
        self.left_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect',
            self.left_image_callback,
            10
        )

        self.right_sub = self.create_subscription(
            Image,
            '/camera/right/image_rect',
            self.right_image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/left/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for robot pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/visual_slam/pose',
            10
        )

        # Initialize VSLAM parameters
        self.camera_matrix = None
        self.last_image = None
        self.pose = np.eye(4)  # Homogeneous transformation matrix

        self.get_logger().info("Visual SLAM Node Started")

    def camera_info_callback(self, msg):
        """Callback to get camera calibration parameters."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def left_image_callback(self, msg):
        """Process left camera image for VSLAM."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.process_frame(cv_image, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"Error processing left image: {e}")

    def right_image_callback(self, msg):
        """Process right camera image for stereo depth."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            # In a complete implementation, this would be used for depth estimation
            pass
        except Exception as e:
            self.get_logger().error(f"Error processing right image: {e}")

    def process_frame(self, image, timestamp):
        """Process current frame and update pose estimate."""
        if self.last_image is None:
            self.last_image = image
            return

        # Feature matching between current and previous frame
        kp1, desc1 = self.extract_features(self.last_image)
        kp2, desc2 = self.extract_features(image)

        # Match features
        matches = self.match_features(desc1, desc2)

        if len(matches) > 10:  # Need enough matches for robust estimation
            # Estimate motion using matched features
            motion = self.estimate_motion(kp1, kp2, matches)

            # Update pose
            self.pose = self.pose @ motion

            # Publish pose
            self.publish_pose(timestamp)

        self.last_image = image

    def extract_features(self, image):
        """Extract ORB features from image."""
        orb = cv2.ORB_create(nfeatures=1000)
        kp, desc = orb.detectAndCompute(image, None)
        if desc is not None:
            return kp, desc
        else:
            return [], np.array([])

    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors."""
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        # Sort matches by distance (best first)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def estimate_motion(self, kp1, kp2, matches):
        """Estimate camera motion from matched features."""
        if len(matches) < 10:
            return np.eye(4)

        # Get matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix (assuming calibrated camera)
        E, mask = cv2.findEssentialMat(
            src_pts, dst_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            threshold=1.0
        )

        if E is not None:
            # Recover pose from essential matrix
            _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            return T

        return np.eye(4)

    def publish_pose(self, timestamp):
        """Publish current pose estimate."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"

        # Extract position and orientation from transformation matrix
        pose_msg.pose.position.x = self.pose[0, 3]
        pose_msg.pose.position.y = self.pose[1, 3]
        pose_msg.pose.position.z = self.pose[2, 3]

        # Convert rotation matrix to quaternion
        import tf_transformations
        quat = tf_transformations.quaternion_from_matrix(self.pose)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisualSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Visual SLAM Package

NVIDIA provides a hardware-accelerated Visual SLAM package optimized for Jetson platforms:

```python
# launch/visual_slam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_vslam'),
        'config',
        'vslam_config.yaml'
    )

    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        name='visual_slam',
        parameters=[config],
        remappings=[
            ('/visual_slam/camera/left/image', '/camera/left/image_rect'),
            ('/visual_slam/camera/right/image', '/camera/right/image_rect'),
            ('/visual_slam/camera/left/camera_info', '/camera/left/camera_info'),
            ('/visual_slam/camera/right/camera_info', '/camera/right/camera_info'),
        ],
        output='screen'
    )

    return LaunchDescription([
        visual_slam_node,
    ])
```

## Deep Learning-based Perception

### Isaac ROS DNN Inference Pipeline

```python
# dnn_perception_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class DNNPerceptionNode(Node):
    def __init__(self):
        super().__init__('dnn_perception_node')
        self.bridge = CvBridge()

        # Load pre-trained model (e.g., YOLOv5, Detectron2)
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/dnn_perception/detections',
            10
        )

        self.get_logger().info("DNN Perception Node Started")

    def load_model(self):
        """Load a pre-trained model."""
        # Load YOLOv5 model (example)
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run inference
            detections = self.run_inference(cv_image)

            # Publish detections
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error in perception pipeline: {e}")

    def run_inference(self, image):
        """Run DNN inference on the input image."""
        # Convert image for model input
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            results = self.model(img_tensor)

        # Process results
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():  # Bounding boxes
            detection = {
                'bbox': [int(val) for val in xyxy],
                'confidence': conf,
                'class_id': int(cls)
            }
            detections.append(detection)

        return detections

    def publish_detections(self, detections, header):
        """Publish detection results."""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            # In a complete implementation, we would create Detection2D messages
            # This is a simplified example
            pass

        self.detections_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = DNNPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Navigation Systems

Perception results are often used by navigation systems for path planning and obstacle avoidance:

```python
# perception_to_navigation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import binary_dilation

class PerceptionToNavigation(Node):
    def __init__(self):
        super().__init__('perception_to_navigation')

        # Subscribe to perception outputs
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/dnn_perception/detections',
            self.detections_callback,
            10
        )

        # Subscribe to VSLAM pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/visual_slam/pose',
            self.pose_callback,
            10
        )

        # Publish to navigation stack
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/perception/costmap',
            10
        )

        # Initialize costmap
        self.costmap = np.zeros((100, 100))  # 10m x 10m grid, 10cm resolution
        self.current_pose = None

        self.get_logger().info("Perception-to-Navigation Bridge Started")

    def pose_callback(self, msg):
        """Update current robot pose."""
        self.current_pose = [
            msg.pose.position.x,
            msg.pose.position.y
        ]

    def detections_callback(self, msg):
        """Process detections and update costmap."""
        if self.current_pose is None:
            return

        # Clear old detections
        self.costmap *= 0.8  # Decay old obstacle information

        # Add new detections to costmap
        for detection in msg.detections:
            # Convert detection position to costmap coordinates
            # This assumes we have depth information or stereo estimates
            obstacle_x = detection.bbox.center.x  # Simplified
            obstacle_y = detection.bbox.center.y  # Simplified

            # Calculate grid coordinates (relative to robot)
            grid_x = int((obstacle_x - self.current_pose[0]) / 0.1 + 50)  # 0.1m/cell, centered
            grid_y = int((obstacle_y - self.current_pose[1]) / 0.1 + 50)  # 0.1m/cell, centered

            # Update costmap if coordinates are valid
            if 0 <= grid_x < 100 and 0 <= grid_y < 100:
                # Add obstacle with confidence-based cost
                self.costmap[grid_y, grid_x] = min(100, self.costmap[grid_y, grid_x] +
                    int(detection.results[0].score * 100))

        # Dilate obstacles to add safety margin
        obstacle_map = self.costmap > 80
        dilated_map = binary_dilation(obstacle_map, iterations=2)
        self.costmap[dilated_map] = np.maximum(self.costmap[dilated_map], 75)

        # Publish updated costmap
        self.publish_costmap()

    def publish_costmap(self):
        """Publish costmap as OccupancyGrid."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = 0.1  # 10cm/cell
        msg.info.width = 100
        msg.info.height = 100

        # Set origin (robot position-centered)
        if self.current_pose:
            msg.info.origin.position.x = self.current_pose[0] - 5.0  # 5m offset
            msg.info.origin.position.y = self.current_pose[1] - 5.0  # 5m offset

        # Flatten costmap for message
        msg.data = self.costmap.flatten().astype(np.int8)

        self.costmap_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionToNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Evaluation

### Perception Pipeline Metrics

When implementing perception pipelines, it's important to evaluate performance:

1. **Accuracy Metrics**:
   - Precision and recall for object detection
   - Positional accuracy for localization
   - Map quality metrics

2. **Efficiency Metrics**:
   - Processing time per frame
   - Memory usage
   - Power consumption (especially on edge devices)

3. **Robustness Metrics**:
   - Performance under different lighting conditions
   - Performance with partial occlusions
   - Failure recovery capabilities

### Benchmarking Tools

```python
# perception_benchmark.py
import time
import numpy as np
from collections import deque

class PerceptionBenchmark:
    def __init__(self, window_size=100):
        self.processing_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()

    def start_timing(self):
        self.frame_start = time.time()

    def end_timing(self):
        if hasattr(self, 'frame_start'):
            processing_time = time.time() - self.frame_start
            self.processing_times.append(processing_time)
            self.frame_count += 1

    def get_fps(self):
        """Calculate frames per second."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

    def get_avg_processing_time(self):
        """Calculate average processing time."""
        return np.mean(self.processing_times) if self.processing_times else 0

    def get_throughput_stats(self):
        """Get comprehensive throughput statistics."""
        if not self.processing_times:
            return {}

        return {
            'fps': self.get_fps(),
            'avg_processing_time': self.get_avg_processing_time(),
            'min_processing_time': min(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'std_processing_time': np.std(self.processing_times)
        }
```

## Practical Exercise: Implement a Simple VSLAM Pipeline

1. **Set up Isaac ROS Visual SLAM** with your robot's camera
2. **Create a launch file** that starts the VSLAM node
3. **Visualize the results** in RViz
4. **Collect performance metrics** during operation
5. **Analyze mapping accuracy** by comparing with ground truth if available

## Summary

This chapter covered perception pipelines and Visual SLAM in ROS, specifically with NVIDIA's Isaac ROS framework. Perception is a critical component of robotic autonomy, enabling robots to understand and navigate their environment. Isaac ROS provides hardware-accelerated perception packages that are essential for real-time robotics applications on NVIDIA platforms.

## Diagrams and Visual Aids

![Perception Pipeline](/img/perception-pipeline.png)

*Figure 1: ROS perception pipeline with Isaac ROS components*

![VSLAM Process](/img/vslam-process.png)

*Figure 2: Visual SLAM process flow*

![Isaac ROS Components](/img/isaac-ros-components.png)

*Figure 3: Isaac ROS perception stack components*