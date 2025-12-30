# Chapter 1: ROS 2 Architecture

## Learning Objectives

After completing this chapter, you will be able to:
- Understand the fundamental architecture of ROS 2
- Explain the differences between ROS 1 and ROS 2
- Describe the core concepts: nodes, topics, services, and actions
- Identify when to use each communication pattern

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Improvements over ROS 1

ROS 2 addresses many of the limitations of ROS 1:
- **Real-Time Support**: ROS 2 supports real-time capabilities, essential for many robotic applications
- **Multi-Robot Systems**: Better support for multiple robots working together
- **Distributed Architecture**: No single point of failure like the ROS Master in ROS 1
- **Security**: Built-in security features for safe operation
- **Platform Support**: Expanded platform support including Windows and macOS

## Core Architecture Concepts

### Nodes

Nodes are the fundamental building blocks of a ROS 2 system. A node is a process that performs computation. Every ROS 2 program consists of one or more nodes.

```python
# Example: Basic ROS 2 Node in Python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data formats that travel between nodes via topics.

- **Pub/Sub Pattern**: One-to-many communication pattern
- **Asynchronous**: Publishers and subscribers don't need to run simultaneously
- **Typed**: Messages have a defined structure

### Services

Services provide request/response communication between nodes (Request/Reply pattern):
- Synchronous communication
- One-to-one communication
- Request is sent, then response is received

### Actions

Actions provide goal-based communication for long-running tasks:
- Goal → Result pattern
- Includes feedback during execution
- Cancelation capability
- For tasks that take significant time to complete

## DDS: The Middleware

ROS 2 uses DDS (Data Distribution Service) as its middleware. DDS provides the underlying communication layer and implements the publish/subscribe, service, and action patterns.

### DDS Implementation Options

ROS 2 supports multiple DDS implementations:
- **Fast DDS** (formerly Fast RTPS) - Default in recent ROS 2 versions
- **Cyclone DDS** - Eclipse Foundation implementation
- **RTI Connext DDS** - Commercial solution
- **OpenSplice DDS** - ADLINK implementation

## Quality of Service (QoS)

ROS 2 introduces Quality of Service settings that allow fine-tuning communication behavior:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Example QoS configuration
my_qos_profile = QoSProfile(
    history=QoSHistoryPolicy.RMW_QOS_HISTORY_KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE
)
```

## Practical Exercise

Create a simple ROS 2 package with a publisher and subscriber to understand the basic communication pattern.

### Step-by-Step Workflow

1. **Create a new ROS 2 package**
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_pkg
   cd my_robot_pkg
   ```

2. **Create the publisher node**
   - Create file: `my_robot_pkg/my_robot_pkg/publisher_member_function.py`
   - Add the publisher code as shown in the code examples
   - Make sure the file is executable: `chmod +x my_robot_pkg/publisher_member_function.py`

3. **Create the subscriber node**
   - Create file: `my_robot_pkg/my_robot_pkg/subscriber_member_function.py`
   - Add the subscriber code as shown in the code examples
   - Make sure the file is executable: `chmod +x my_robot_pkg/subscriber_member_function.py`

4. **Update setup.py**
   - Add entries for your nodes in the entry_points section
   - Example:
     ```python
     entry_points={
         'console_scripts': [
             'publisher_member_function = my_robot_pkg.publisher_member_function:main',
             'subscriber_member_function = my_robot_pkg.subscriber_member_function:main',
         ],
     },
     ```

5. **Build and source the workspace**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_pkg
   source install/setup.bash
   ```

6. **Test the communication between nodes**
   - In one terminal: `ros2 run my_robot_pkg publisher_member_function`
   - In another terminal: `ros2 run my_robot_pkg subscriber_member_function`

## Diagrams and Visual Aids

![ROS 2 Architecture Diagram](/img/ros2-architecture.png)

*Figure 1: High-level overview of ROS 2 architecture showing nodes, topics, services, and actions*

![DDS Communication Layers](/img/dds-layers.png)

*Figure 2: DDS middleware communication layers in ROS 2*