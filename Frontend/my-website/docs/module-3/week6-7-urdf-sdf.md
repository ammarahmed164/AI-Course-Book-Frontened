---
sidebar_position: 4
title: "URDF and SDF Robot Description Formats"
---

# URDF and SDF Robot Description Formats

This lesson covers the two primary robot description formats used in robotics: URDF (Unified Robot Description Format) and SDF (Simulation Description Format).

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the differences between URDF and SDF
- Create robot models using URDF
- Convert URDF to SDF for simulation
- Use Xacro to simplify complex robot descriptions
- Integrate robot models with Gazebo simulation

## Introduction to Robot Description Formats

Robot description formats are XML-based languages that define the physical and kinematic properties of robots. They specify:
- Links (rigid parts of the robot)
- Joints (connections between links)
- Visual and collision properties
- Inertial properties
- Sensors and actuators

## URDF (Unified Robot Description Format)

URDF is primarily used in ROS and focuses on kinematic and geometric properties of robots.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
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

  <!-- Example joint and link -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.3 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

### URDF Elements

#### Links
- **visual**: How the link looks in simulation
- **collision**: How the link interacts physically
- **inertial**: Mass and inertia properties

#### Joints
- **fixed**: No movement
- **continuous**: Continuous rotation
- **revolute**: Limited rotation
- **prismatic**: Linear movement
- **floating**: 6 DOF movement
- **planar**: Planar movement

### URDF Joint Types

```xml
<!-- Fixed joint -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
</joint>

<!-- Revolute joint -->
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
</joint>

<!-- Continuous joint -->
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>

<!-- Prismatic joint -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="10" velocity="1"/>
</joint>
```

## SDF (Simulation Description Format)

SDF is used by Gazebo and provides more features for simulation than URDF.

### Basic SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="my_robot">
    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0.5 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 0.8 1</ambient>
          <diffuse>0 0 0.8 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Joint -->
    <joint name="base_to_wheel" type="revolute">
      <parent>base_link</parent>
      <child>wheel_link</child>
      <pose>0 0.3 -0.1 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
        </limit>
      </axis>
    </joint>

    <link name="wheel_link">
      <pose>0 0.3 -0.1 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

### SDF Extensions for Gazebo

SDF supports Gazebo-specific elements:

```xml
<sdf version="1.6">
  <model name="my_robot">
    <!-- Standard model definition -->
    <link name="base_link">
      <!-- Link definition -->
    </link>

    <!-- Gazebo-specific extensions -->
    <gazebo reference="base_link">
      <material>Gazebo/Blue</material>
      <mu1>0.5</mu1>
      <mu2>0.5</mu2>
    </gazebo>

    <!-- Sensor definition -->
    <gazebo reference="camera_link">
      <sensor name="camera" type="camera">
        <pose>0.1 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <ros>
            <namespace>camera</namespace>
            <remapping>image_raw:=image</remapping>
          </ros>
        </plugin>
      </sensor>
    </gazebo>
  </model>
</sdf>
```

## Converting URDF to SDF

ROS provides tools to convert URDF to SDF for use in Gazebo:

```bash
# Using gz sdf tool
gz sdf -p robot.urdf > robot.sdf

# Using xacro to preprocess before conversion
ros2 run xacro xacro robot.xacro > robot.urdf
gz sdf -p robot.urdf > robot.sdf
```

### Programmatic Conversion in Launch Files

```python
# launch/robot_spawn.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('my_robot_description')
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', LaunchConfiguration('robot_name')
        ],
        output='screen'
    )
    
    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                os.path.join(pkg_dir, 'urdf', 'robot.urdf')
            ).read()
        }]
    )
    
    return LaunchDescription([
        robot_name_arg,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Xacro: XML Macros for Robot Description

Xacro is a macro language that simplifies complex URDF definitions.

### Basic Xacro Features

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot_xacro">

  <!-- Properties -->
  <xacro:property name="base_radius" value="0.2" />
  <xacro:property name="base_length" value="0.6" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_length" value="0.1" />
  <xacro:property name="PI" value="3.1415926535897931" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="suffix parent x y z">
    <joint name="${suffix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${suffix}_wheel"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 ${PI/2} 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <link name="${suffix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_length}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Robot base -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_length}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
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

  <!-- Include other xacro files -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro" />
  <xacro:include filename="$(find my_robot_description)/urdf/transmission.xacro" />

</robot>
```

### Advanced Xacro Features

```xml
<!-- Conditional statements -->
<xacro:if value="true">
  <link name="conditional_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</xacro:if>

<!-- Mathematical expressions -->
<xacro:property name="mass" value="2.0" />
<xacro:property name="volume" value="${mass * 0.1}" />

<!-- File inclusion -->
<xacro:include filename="$(find my_robot_description)/urdf/sensors/camera.xacro" />

<!-- Inverse kinematics properties -->
<xacro:macro name="arm_segment" params="name parent_link length mass">
  <joint name="${name}_joint" type="revolute">
    <parent link="${parent_link}"/>
    <child link="${name}_link"/>
    <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="1"/>
  </joint>

  <link name="${name}_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="${length}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="${length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass}"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</xacro:macro>
```

## Integrating with Gazebo

### Adding Gazebo-Specific Elements to URDF

```xml
<!-- In your URDF file -->
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<!-- Gazebo-specific elements -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <pose>0.05 0 0 0 0 0</pose>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>camera</namespace>
        <remapping>image_raw:=image</remapping>
      </ros>
      <output_type>sensor_msgs/Image</output_type>
    </plugin>
  </sensor>
</gazebo>

<!-- Physics properties -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>
```

## Practical Exercise: Create a Simple Robot Model

Create a simple differential drive robot model:

1. **Create a URDF file** (`diff_drive_robot.urdf`):
   - Define a base link
   - Add two wheels using joints
   - Include a caster wheel for stability
   - Add a camera sensor

2. **Create a Xacro version** to simplify the definition

3. **Test the robot** in Gazebo using the spawn_entity tool

4. **Add Gazebo plugins** for differential drive control

```xml
<!-- diff_drive_robot.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="diff_drive_robot">
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_length" value="0.4" />
  <xacro:property name="base_height" value="0.1" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.02" />
  <xacro:property name="wheel_pos" value="0.1" />

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 ${base_length/2 + wheel_width/2} -${wheel_radius}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -${base_length/2 + wheel_width/2} -${wheel_radius}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Caster wheel -->
  <joint name="caster_wheel_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="${base_width/2} 0 -${wheel_radius}" rpy="0 0 0"/>
  </joint>

  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="${wheel_radius/2}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${wheel_radius/2}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>diff_drive</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
    </plugin>
  </gazebo>

  <gazebo reference="base_link">
    <material>Gazebo/Orange</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
  </gazebo>
</robot>
```

## Best Practices

1. **Use Xacro** for complex robots to reduce duplication
2. **Validate URDF/SDF** files before using them
3. **Start simple** and add complexity gradually
4. **Match inertial properties** to real robot when possible
5. **Use appropriate collision meshes** for performance
6. **Test in simulation** before implementing on real robots
7. **Document your robot model** with comments

## Troubleshooting Common Issues

### URDF Validation
```bash
# Validate URDF
check_urdf /path/to/robot.urdf

# Use robot_state_publisher to check for errors
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="..."
```

### Common Problems
- Joint limits causing issues
- Mass/inertia values too small/large
- Invalid geometries
- Missing materials

## Summary

URDF and SDF are essential for defining robot models in ROS and Gazebo. URDF is used for kinematic descriptions in ROS, while SDF is used for full simulation in Gazebo. Xacro simplifies complex robot definitions and reduces duplication. Understanding these formats is crucial for robotics development.

## Next Steps

In the next lesson, we'll explore physics simulation and sensor simulation in Gazebo.