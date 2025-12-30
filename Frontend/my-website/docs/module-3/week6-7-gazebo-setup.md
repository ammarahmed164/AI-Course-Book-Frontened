---
sidebar_position: 3
title: "Gazebo Environment Setup"
---

# Gazebo Environment Setup

This lesson covers how to set up and configure Gazebo simulation environments for robot testing and development.

## Learning Objectives

After completing this lesson, you will be able to:
- Create custom world files for Gazebo
- Configure physics properties for different scenarios
- Set up lighting and visual environments
- Add and configure models in the simulation
- Launch Gazebo with custom configurations

## Creating Custom World Files

World files define the simulation environment using SDF (Simulation Description Format). Let's create a custom world for a robot navigation task.

### Basic World Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="navigation_world">
    <!-- Physics properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models -->
    <model name="robot">
      <!-- Robot definition will go here -->
    </model>

    <!-- Custom objects -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Physics Configuration

### Physics Engines
Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default, good for general use
- **Bullet**: Good for complex contact scenarios
- **Simbody**: Good for biomechanics applications
- **DART**: Good for articulated bodies

### Physics Parameters

```xml
<physics type="ode" name="default_physics">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Tuning Physics for Different Scenarios

For **precision tasks** (manipulation):
- Smaller step size: 0.0001
- More solver iterations: 100
- Higher update rate: 10000

For **real-time simulation**:
- Larger step size: 0.01
- Fewer solver iterations: 10
- Lower update rate: 100

## Lighting and Visual Environment

### Types of Lights
- **Directional**: Sun-like, parallel rays
- **Point**: Omnidirectional light source
- **Spot**: Conical light beam

### Example Lighting Configuration

```xml
<!-- Directional light (sun) -->
<light name="sun" type="directional">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.6 0.4 -0.8</direction>
</light>

<!-- Point light -->
<light name="lamp" type="point">
  <pose>0 0 5 0 0 0</pose>
  <diffuse>1 1 1 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.2</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
</light>
```

## Adding Models to the Environment

### Including Pre-built Models

```xml
<!-- Include a ground plane -->
<include>
  <uri>model://ground_plane</uri>
</include>

<!-- Include a robot model -->
<include>
  <uri>model://turtlebot3_waffle</uri>
  <pose>0 0 0 0 0 0</pose>
</include>

<!-- Include a custom model -->
<include>
  <uri>model://my_custom_robot</uri>
  <pose>1 1 0 0 0 0</pose>
</include>
```

### Creating Custom Models

```xml
<model name="custom_table">
  <pose>5 5 0 0 0 0</pose>
  <link name="table_top">
    <pose>0 0 0.75 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>1.5 1 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.5 1 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.8 0.6 0.4 1</diffuse>
        <specular>0.1 0.1 0.1 1</specular>
      </material>
    </visual>
    <inertial>
      <mass>10</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
  
  <!-- Table legs -->
  <link name="leg1">
    <pose>-0.6 -0.4 0.3 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.05</radius>
          <length>0.6</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.05</radius>
          <length>0.6</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>0.3 0.3 0.3 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

## Terrain and Environment

### Creating Terrain

```xml
<model name="terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### Building Indoor Environments

```xml
<!-- Walls for a room -->
<model name="wall_1">
  <pose>0 5 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>10 0.2 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>10 0.2 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.8 0.8 1</ambient>
        <diffuse>0.9 0.9 0.9 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

## Launching Gazebo with Custom Configurations

### Using ROS Launch Files

```python
# launch/custom_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('my_robot_gazebo')
    
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_dir, 'worlds', 'custom_world.world'),
        description='SDF world file'
    )
    
    # Launch Gazebo
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', LaunchConfiguration('world'), '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    
    return LaunchDescription([
        world_arg,
        gazebo,
    ])
```

### Launching from Command Line

```bash
# Launch with custom world
gazebo /path/to/my_world.world

# Launch with custom parameters
gazebo --verbose /path/to/my_world.world

# Launch with paused simulation
gazebo --pause /path/to/my_world.world
```

## Advanced Environment Features

### Wind Effects

```xml
<world name="windy_world">
  <physics type="ode">
    <gravity>0 0 -9.8</gravity>
  </physics>
  
  <!-- Wind effect -->
  <wind>
    <linear_velocity>0.5 0 0</linear_velocity>
  </wind>
  
  <!-- Include models -->
  <include>
    <uri>model://ground_plane</uri>
  </include>
</world>
```

### Atmospheric Effects

```xml
<scene>
  <ambient>0.4 0.4 0.4 1</ambient>
  <background>0.7 0.7 0.7 1</background>
  <shadows>true</shadows>
  <grid>false</grid>
  <origin_visual>false</origin_visual>
  <sky>
    <time>14:00</time>
    <sun_direction>0.7 0.3 0.5</sun_direction>
    <clouds>
      <speed>0.6</speed>
      <direction>0.3 0</direction>
      <humidity>0.5</humidity>
      <mean_size>0.5</mean_size>
    </clouds>
  </sky>
</scene>
```

## Practical Exercise: Create a Navigation World

Create a custom world for testing robot navigation:

1. **Create a world file** (`navigation_world.world`):
   - Add a ground plane
   - Create a simple maze using boxes
   - Add a starting position marker
   - Include a goal position marker

2. **Configure physics** for realistic movement

3. **Add lighting** for good visibility

4. **Test the world** by launching Gazebo with your custom world file

```bash
gazebo navigation_world.world
```

## Best Practices for Environment Setup

1. **Start Simple**: Begin with basic environments and add complexity gradually
2. **Use Realistic Physics**: Match physics parameters to real-world conditions
3. **Validate Visually**: Ensure the environment looks realistic
4. **Test Performance**: Ensure simulation runs at acceptable speeds
5. **Document Environments**: Keep notes on environment parameters
6. **Version Control**: Store world files in version control systems

## Troubleshooting Common Issues

### Simulation Runs Slowly
- Reduce visual complexity
- Increase step size (may reduce accuracy)
- Simplify collision meshes
- Reduce number of objects

### Physics Issues
- Adjust solver parameters
- Verify mass and inertia values
- Check for intersecting geometries
- Verify joint limits

## Summary

Setting up Gazebo environments requires understanding of SDF, physics parameters, and visual configuration. A well-designed environment enables effective testing and development of robotic systems. Proper configuration of physics, lighting, and objects is crucial for realistic simulation.

## Next Steps

In the next lesson, we'll explore URDF and SDF robot description formats in detail.