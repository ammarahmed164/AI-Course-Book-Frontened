---
sidebar_position: 4
title: "Building ROS 2 Packages with Python"
---

# Building ROS 2 Packages with Python

This lesson covers how to create and structure ROS 2 packages using Python.

## Learning Objectives

After completing this lesson, you will be able to:
- Create ROS 2 packages with Python
- Structure packages according to ROS 2 conventions
- Implement nodes with proper error handling
- Use ROS 2 client libraries in Python

## ROS 2 Package Structure

A well-structured ROS 2 package includes these elements:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration (for C++)
├── package.xml             # Package manifest
├── setup.py                # Python package setup
├── setup.cfg               # Configuration for ament_cmake_python
├── my_robot_package/       # Python module
│   ├── __init__.py
│   └── my_module.py
├── launch/                 # Launch files
│   └── robot.launch.py
├── config/                 # Configuration files
│   └── params.yaml
├── urdf/                   # Robot description files
│   └── robot.urdf.xacro
└── test/                   # Test files
    └── test_my_module.py
```

### Creating a package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot</description>
  <maintainer email="maintainer@todo.com">maintainer</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Creating a Python Package

### Basic Package Setup

1. **Create the package**:
   ```bash
   ros2 pkg create --build-type ament_python my_robot_package
   ```

2. **Set up the Python module**:
   ```python
   # my_robot_package/my_robot_package/__init__.py
   # This file can be empty or contain package initialization code
   ```

### Creating a Python Node

```python
# my_robot_package/my_robot_package/simple_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        
        # Create a publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)
        
        # Create a timer
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.i = 0
        
        # Log that the node is ready
        self.get_logger().info('SimpleNode initialized')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    
    simple_node = SimpleNode()
    
    try:
        rclpy.spin(simple_node)
    except KeyboardInterrupt:
        pass
    finally:
        simple_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Python Node Patterns

### Using Parameters

```python
# my_robot_package/my_robot_package/parameter_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with descriptions and ranges
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('safety_limits', [1.0, 2.0, 3.0])
        
        # Get parameter values
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value
        self.safety_limits = self.get_parameter('safety_limits').value
        
        # Create publisher
        self.publisher = self.create_publisher(String, 'parameter_output', 10)
        
        # Create timer with parameterized frequency
        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
        
        self.get_logger().info(
            f'ParameterNode initialized with frequency: {self.frequency}, '
            f'robot_name: {self.robot_name}, safety_limits: {self.safety_limits}'
        )

    def timer_callback(self):
        msg = String()
        msg.data = f'Data from {self.robot_name}'
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    parameter_node = ParameterNode()
    
    try:
        rclpy.spin(parameter_node)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Error Handling and Lifecycle Management

```python
# my_robot_package/my_robot_package/robust_node.py
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import String

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        
        # Use a callback group to manage execution
        self.cb_group = MutuallyExclusiveCallbackGroup()
        
        # Create publisher
        self.publisher = self.create_publisher(String, 'robust_chatter', 10)
        
        # Create timer with error handling
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(
            timer_period, 
            self.timer_callback, 
            callback_group=self.cb_group
        )
        
        self.i = 0
        
        self.get_logger().info('RobustNode initialized')

    def timer_callback(self):
        try:
            msg = String()
            msg.data = f'Robust message: {self.i}'
            
            # Add error handling for publisher
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: "{msg.data}"')
            self.i += 1
            
        except Exception as e:
            self.get_logger().error(f'Error in timer_callback: {str(e)}')

    def destroy_node(self):
        self.get_logger().info('Cleaning up RobustNode resources...')
        # Perform any cleanup here
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    robust_node = RobustNode()
    
    # Use a multi-threaded executor for better performance
    executor = MultiThreadedExecutor()
    executor.add_node(robust_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robust_node.destroy_node()
        rclpy.shutdown()
        executor.shutdown()

if __name__ == '__main__':
    main()
```

## Setting up setup.py for Python Packages

```python
# setup.py
from setuptools import setup

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package for my robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_node = my_robot_package.simple_node:main',
            'parameter_node = my_robot_package.parameter_node:main',
            'robust_node = my_robot_package.robust_node:main',
        ],
    },
)
```

## Testing ROS 2 Python Packages

### Unit Tests

```python
# test/test_my_robot_package.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_package.simple_node import SimpleNode

class TestSimpleNode(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = SimpleNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        self.assertEqual(self.node.get_name(), 'simple_node')
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
```

## Best Practices for Python ROS 2 Packages

1. **Follow Python naming conventions** (PEP 8)
2. **Use proper logging** instead of print statements
3. **Handle exceptions gracefully**
4. **Clean up resources** in destroy_node method
5. **Use parameters for configuration**
6. **Write tests** for your nodes
7. **Document your code** with docstrings
8. **Follow ROS 2 naming conventions** for topics, services, etc.

## Practical Exercise

Create a complete ROS 2 package that includes:
1. A publisher node that publishes sensor data
2. A subscriber node that processes the data
3. A service server that provides calculations
4. Proper package structure and configuration
5. Unit tests for your nodes

## Summary

Creating ROS 2 packages with Python involves understanding the package structure, properly configuring setup files, implementing nodes with error handling, and following best practices. Python provides a flexible and powerful environment for developing ROS 2 nodes.

## Next Steps

In the next lesson, we'll explore launch files and parameter management in ROS 2.