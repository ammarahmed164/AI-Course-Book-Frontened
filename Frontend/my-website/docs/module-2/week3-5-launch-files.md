---
sidebar_position: 5
title: "Launch Files and Parameter Management"
---

# Launch Files and Parameter Management

This lesson covers how to create and use launch files and manage parameters in ROS 2.

## Learning Objectives

After completing this lesson, you will be able to:
- Create launch files to start multiple nodes with specific configurations
- Manage parameters using YAML files and command-line arguments
- Use conditional logic in launch files
- Configure complex systems with launch arguments

## Introduction to Launch Files

Launch files allow you to start multiple nodes with specific configurations in a single command. ROS 2 uses Python-based launch files that provide more flexibility than ROS 1's XML-based launch files.

## Creating Launch Files

### Basic Launch File

```python
# launch/simple_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='simple_node',
            name='simple_node_1',
            output='screen',
        ),
        Node(
            package='my_robot_package',
            executable='simple_node',
            name='simple_node_2',
            output='screen',
        ),
    ])
```

### Launch Files with Parameters

```python
# launch/parameter_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='5.0',
        description='Frequency for the parameter node'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name for the robot'
    )

    return LaunchDescription([
        # Include the declared arguments
        frequency_arg,
        robot_name_arg,
        
        # Launch node with parameters
        Node(
            package='my_robot_package',
            executable='parameter_node',
            name='parameter_node',
            parameters=[
                {
                    'frequency': LaunchConfiguration('frequency'),
                    'robot_name': LaunchConfiguration('robot_name'),
                    'safety_limits': [1.0, 2.0, 3.0]
                }
            ],
            output='screen',
        ),
    ])
```

## Parameter Management

### Parameter Files (YAML)

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    frequency: 10.0
    robot_name: "my_robot"
    safety_limits:
      - 1.0
      - 2.0
      - 3.0
    sensors:
      lidar_enabled: true
      camera_enabled: true
      imu_enabled: true
    navigation:
      max_velocity: 0.5
      min_distance: 0.5
```

### Loading Parameters from YAML

```python
# launch/param_file_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'config',
            'robot_params.yaml'
        ]),
        description='Path to parameters file'
    )

    return LaunchDescription([
        params_file_arg,
        
        Node(
            package='my_robot_package',
            executable='parameter_node',
            name='parameter_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
    ])
```

## Advanced Launch Concepts

### Conditional Launch

```python
# launch/conditional_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera',
        default_value='true',
        description='Enable camera node if true'
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_camera_arg,
        
        # Launch node with conditional parameters
        Node(
            package='my_robot_package',
            executable='parameter_node',
            name='parameter_node',
            parameters=[
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }
            ],
            output='screen',
        ),
        
        # Conditionally launch camera node
        Node(
            package='my_robot_package',
            executable='camera_node',
            name='camera_node',
            condition=IfCondition(LaunchConfiguration('enable_camera')),
            output='screen',
        ),
    ])
```

### Composable Nodes

For better performance, you can run multiple nodes in the same process using composable nodes:

```python
# launch/composable_launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='my_robot_package',
                plugin='my_robot_package::SensorProcessor',
                name='sensor_processor'
            ),
            ComposableNode(
                package='my_robot_package',
                plugin='my_robot_package::Controller',
                name='controller'
            )
        ],
        output='screen',
    )

    return LaunchDescription([container])
```

## Parameter Techniques

### Using Parameter Descriptions

```python
# my_robot_package/parameter_node_with_descriptions.py
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterException

class ParameterNodeWithDescriptions(Node):
    def __init__(self):
        super().__init__('parameter_node_with_descriptions')
        
        # Declare parameters with descriptions
        self.declare_parameter(
            'frequency', 
            10.0,
            ParameterDescriptor(
                description='Frequency for the timer callback',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        
        self.declare_parameter(
            'robot_name',
            'my_robot',
            ParameterDescriptor(
                description='Name of the robot',
                type=ParameterType.PARAMETER_STRING
            )
        )
        
        # Get parameters
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value
        
        # Create timer with parameterized frequency
        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
        
        self.get_logger().info(
            f'Node initialized with frequency: {self.frequency}, '
            f'robot_name: {self.robot_name}'
        )

    def timer_callback(self):
        self.get_logger().info(f'Callback running for {self.robot_name}')
```

### Parameter Validation

```python
# my_robot_package/validated_parameter_node.py
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterException
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class ValidatedParameterNode(Node):
    def __init__(self):
        super().__init__('validated_parameter_node')
        
        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Declare parameters
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('robot_name', 'my_robot')

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'frequency':
                if param.type == ParameterType.PARAMETER_DOUBLE:
                    if param.value < 0.1 or param.value > 100.0:
                        return SetParametersResult(
                            successful=False,
                            reason='Frequency must be between 0.1 and 100.0'
                        )
                else:
                    return SetParametersResult(
                        successful=False,
                        reason='Frequency must be a double value'
                    )
            
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        
        return SetParametersResult(successful=True)
```

## Practical Exercise: Complete System Launch

Create a complete system launch file that includes:

1. A robot state publisher
2. A parameter server
3. Multiple sensor nodes
4. A controller node
5. Conditional launching based on arguments

### Complete Example Launch File

```python
# launch/complete_robot_system_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    robot_description_path_arg = DeclareLaunchArgument(
        'robot_description_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_description'),
            'urdf',
            'robot.urdf.xacro'
        ]),
        description='Path to robot description file'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'config',
            'robot_params.yaml'
        ]),
        description='Path to parameters file'
    )

    return LaunchDescription([
        # Declare arguments
        use_sim_time_arg,
        robot_description_path_arg,
        params_file_arg,
        
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_description': 
                    Command(['xacro ', LaunchConfiguration('robot_description_path')])
                }
            ]
        ),
        
        # Parameter server
        Node(
            package='my_robot_package',
            executable='parameter_node',
            name='parameter_server',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
        
        # Sensor nodes
        Node(
            package='my_robot_package',
            executable='lidar_node',
            name='lidar_driver',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
        
        Node(
            package='my_robot_package',
            executable='camera_node',
            name='camera_driver',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
        
        # Controller node
        Node(
            package='my_robot_package',
            executable='controller_node',
            name='robot_controller',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
    ])
```

## Running Launch Files

### Basic Launch Command

```bash
# Launch with default parameters
ros2 launch my_robot_package simple_launch.py

# Launch with specific parameters
ros2 launch my_robot_package parameter_launch.py frequency:=2.0 robot_name:=test_robot

# Launch with parameter file
ros2 launch my_robot_package param_file_launch.py params_file:=/path/to/params.yaml
```

## Best Practices

1. **Use descriptive names** for launch arguments
2. **Provide meaningful defaults** for all parameters
3. **Document parameters** with descriptions
4. **Validate parameters** when possible
5. **Use YAML files** for complex parameter sets
6. **Group related nodes** in launch files
7. **Use conditional launching** for optional components
8. **Test launch files** thoroughly

## Summary

Launch files and parameter management are crucial for configuring and running complex ROS 2 systems. Launch files allow you to start multiple nodes with specific configurations, while parameters provide a flexible way to configure node behavior. Understanding these concepts is essential for developing professional ROS 2 applications.

## Next Steps

With a foundation in ROS 2 fundamentals, we'll move on to Module 3, where we'll explore robot simulation and the NVIDIA Isaac platform.