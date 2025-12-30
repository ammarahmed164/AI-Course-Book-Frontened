# Chapter 3: Python rclpy, URDF Modeling, and Package & Launch Workflows

## Learning Objectives

After completing this chapter, you will be able to:
- Use rclpy to develop complex ROS 2 nodes in Python
- Create and work with URDF (Unified Robot Description Format) files
- Build and structure ROS 2 packages with proper dependencies
- Create launch files to start multiple nodes with specific configurations
- Implement advanced patterns like timers, callbacks, and parameter management

## Advanced rclpy Patterns

### Timers and Callbacks

Timers allow you to execute code at regular intervals:

```python
import rclpy
from rclpy.node import Node

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')

        # Create a timer that calls a callback every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.get_logger().info(f'Timer callback {self.counter}')
        self.counter += 1
```

### Advanced Parameter Handling

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterException

class AdvancedParameterNode(Node):
    def __init__(self):
        super().__init__('advanced_parameter_node')

        # Declare parameters with descriptions and ranges
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('safety_limits', [1.0, 2.0, 3.0])

        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        return SetParametersResult(successful=True)
```

### Custom Message Types

To create custom messages, you need to define them in a package's `msg` directory:

```
# In msg/BatteryState.msg
float32 voltage
float32 current
float32 charge_percentage
bool is_charging
```

Then use them in your Python code:

```python
from my_robot_msgs.msg import BatteryState

class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')
        self.publisher = self.create_publisher(BatteryState, 'battery_state', 10)
        self.timer = self.create_timer(1.0, self.publish_battery_status)

    def publish_battery_status(self):
        msg = BatteryState()
        msg.voltage = 12.6
        msg.current = 0.5
        msg.charge_percentage = 85.0
        msg.is_charging = False
        self.publisher.publish(msg)
```

## URDF: Unified Robot Description Format

URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS. It defines robot structure, kinematics, dynamics, visual, and collision properties.


### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Define materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <!-- Define base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Define wheel links -->
  <link name="wheel_front_left">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
      <origin rpy="0 1.57075 0" xyz="0 0 0"/>
    </visual>
  </link>

  <!-- Define joints between links -->
  <joint name="front_left_wheel_hinge" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_front_left"/>
    <origin xyz="0.2 0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### URDF with Xacro

Xacro is a macro language for URDF that allows for more readable and maintainable robot descriptions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot_xacro">

  <!-- Define properties -->
  <xacro:property name="base_radius" value="0.2" />
  <xacro:property name="base_length" value="0.6" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_length" value="0.1" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="suffix parent x y z">
    <link name="${suffix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_length}"/>
        </geometry>
        <origin rpy="0 1.57075 0" xyz="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_length}"/>
        </geometry>
        <origin rpy="0 1.57075 0" xyz="0 0 0"/>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${suffix}_wheel_hinge" type="continuous">
      <parent link="${parent}"/>
      <child link="${suffix}_wheel"/>
      <origin xyz="${x} ${y} ${z}"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Robot base -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_length}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Create wheels using macro -->
  <xacro:wheel suffix="front_left" parent="base_link" x="0.2" y="0.3" z="0"/>
  <xacro:wheel suffix="front_right" parent="base_link" x="0.2" y="-0.3" z="0"/>
  <xacro:wheel suffix="back_left" parent="base_link" x="-0.2" y="0.3" z="0"/>
  <xacro:wheel suffix="back_right" parent="base_link" x="-0.2" y="-0.3" z="0"/>

</robot>
```

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

## Launch Files

Launch files allow you to start multiple nodes with specific configurations in a single command.

### Python Launch Files

```python
# launch/robot.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # Launch robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_description':
                    Command(['xacro ', FindFile('my_robot_package', 'urdf/robot.urdf.xacro')])
                }
            ]
        ),

        # Launch the main robot controller
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        )
    ])
```

### Composable Launch Files

For better performance, you can run multiple nodes in the same process using composable nodes:

```python
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

## Practical Exercise: Complete Robot Package

Create a complete ROS 2 package that includes:
1. A URDF model of a simple robot with sensors
2. Python nodes for sensor processing and control
3. Launch files to start the entire system
4. Configuration files for parameters
5. Custom message types if needed

Structure:
1. Create the package with proper directory structure
2. Define the robot in URDF/Xacro
3. Implement sensor publisher and controller nodes
4. Create launch files to start the system
5. Test the system with RViz for visualization

## Summary

This chapter covered essential ROS 2 development practices:
- Advanced rclpy patterns including timers, parameters, and custom messages
- URDF for describing robot models, with Xacro for more complex models
- Proper ROS 2 package structure and conventions
- Launch files for starting complex systems with appropriate configurations

These components form the foundation of well-structured ROS 2 applications that can be easily maintained and extended.

## Diagrams and Visual Aids

![Package Structure](/img/package-structure.png)

*Figure 1: Standard ROS 2 package structure*

![URDF Robot Model](/img/urdf-model.png)

*Figure 2: Example of a URDF robot model visualization*

![Launch File Workflow](/img/launch-workflow.png)

*Figure 3: How launch files coordinate multiple nodes*