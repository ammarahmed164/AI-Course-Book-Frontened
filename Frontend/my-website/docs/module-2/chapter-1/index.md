```text
# Chapter 1: Physics Simulation with Gazebo

```
## Learning Objectives
```

After completing this chapter, you will be able to:
- Set up and configure Gazebo for physics simulation
- Create and modify simulation worlds
- Understand the physics engine parameters and their effects
- Spawn and manipulate robot models in simulation
- Configure simulation parameters for realistic behavior

```text
## Introduction to Gazebo
```

Gazebo is a powerful physics simulator that provides realistic rendering, physics, and sensor simulation capabilities. It's widely used in robotics research and development for testing algorithms before deployment on real robots.

```text
### Key Features of Gazebo
```python
```
- **Realistic Physics**: Based on ODE (Open Dynamics Engine), Bullet, and Simbody physics engines
```
- **High-quality Rendering**: Uses OGRE for realistic image rendering
- **Extensive Sensor Support**: LIDAR, cameras, IMU, force-torque sensors, and more
- **Robot Modeling**: Supports URDF and SDF for robot descriptions
- **Plugin Architecture**: Extensible through custom plugins

## Installing and Setting Up Gazebo

### Gazebo Garden Installation

Gazebo Garden is the latest version of the Gazebo simulation family. It can be installed as part of the ROS 2 installation or separately:

```bash
# Update package list
```
sudo apt update

```text
# Install Gazebo Garden
```
sudo apt install ros-humble-gazebo-*

```python
# Install Gazebo Classic (if needed for legacy projects)
```
sudo apt install ros-humble-gazebo-dev
```

### Basic Gazebo Launch

```bash
# Launch Gazebo Classic with an empty world
```
ros2 launch gazebo_ros empty_world.launch.py

```text
# Launch with a specific world file
```
ros2 launch gazebo_ros empty_world.launch.py world:=/path/to/world_file.world
```

## Creating Simulation Worlds

### World File Structure

```python
Gazebo uses SDF (Simulation Description Format) to define worlds. Here's a basic world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a model from the model database -->
    <include>
      <name>ground_plane</name>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a light source -->
    <include>
      <name>sun</name>
      <uri>model://sun</uri>
    </include>

    <!-- Define a box model directly -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

```text
### Physics Engine Configuration
```

The physics engine can be configured in the world file:

```xml
<world name="simple_world">
  <!-- Physics configuration -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <gravity>0 0 -9.8</gravity>
  </physics>
  
  <!-- Rest of the world definition -->
</world>
```

```text
## Robot Simulation in Gazebo

```
### Spawning Robots
```

Robots are typically spawned into simulation using ROS 2 launch files:

```python
# launch/robot_spawn.launch.py
```python
from launch import LaunchDescription
```
```
```python
from launch.actions import IncludeLaunchDescription
```
```
```python
from launch.launch_description_sources import PythonLaunchDescriptionSource
```
```
```python
from launch.substitutions import PathJoinSubstitution
```
```
```python
from launch_ros.actions import Node
```
```
```python
from launch_ros.substitutions import FindPackageShare

```
```
```text
def generate_launch_description
```():
```
```
```
    # Launch Gazebo with an empty world
```python
    gazebo = IncludeLaunchDescription(
```
```
```
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
```python
                FindPackageShare('gazebo_ros'),
```
```
                'launch',
                'empty_world.launch.py'
```
            ])
        ]),
```python
        launch_arguments={
```
```
            'world': PathJoinSubstitution([
```python
                FindPackageShare('my_robot_gazebo'),
```
```
                'worlds',
                'simple_room.world'
```
            ])
```python
        }.items()
```
    )

```text
    # Spawn robot in Gazebo
```python
    spawn_entity = Node(
```
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
```
```
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
```
        ],
```python
        output='screen'
```
```
    )

```python
    return LaunchDescription([
```
```
        gazebo,
        spawn_entity,
    ])
```

### Robot-World Interaction

The robot model should be configured for simulation in its URDF/Xacro:

```xml
<!-- In robot URDF file -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>false</self_collide>
</gazebo>

<!-- For joints with transmissions -->
<transmission name="wheel_front_left_trans" type="transmission_interface/SimpleTransmission">
  <actuator name="wheel_front_left_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
  <joint name="front_left_wheel_hinge">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
</transmission>

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
  </plugin>
</gazebo>
```

## Sensor Simulation

Gazebo provides realistic sensor simulation including:
- **Camera sensors**: RGB, depth, and stereo cameras
- **LIDAR sensors**: 2D and 3D LIDAR
- **IMU sensors**: Inertial measurement units
- **Force/Torque sensors**: For grippers and contact points

### Camera Sensor Example

```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Practical Exercise: Setting Up a Basic Simulation

1. **Create a simple world file** with a ground plane and a few objects
2. **Prepare your robot model** with proper Gazebo plugins
3. **Write a launch file** to spawn the robot in the world
4. **Test the simulation** by controlling the robot with ROS 2

### Step-by-Step Workflow

1. **Create the world file** (`my_robot_gazebo/worlds/simple.world`)

   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_world">
       <!-- Ground plane -->
       <include>
         <uri>model://ground_plane</uri>
       </include>

       <!-- Sun light -->
       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Simple box obstacle -->
       <model name="box_obstacle">
         <pose>2 0 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box><size>1 1 1</size></box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box><size>1 1 1</size></box>
             </geometry>
             <material>
               <ambient>1 0 0 1</ambient>
               <diffuse>1 0 0 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. **Prepare your robot model** with proper Gazebo plugins:

   In your robot's URDF/Xacro file, add the Gazebo-specific plugins:
   ```xml
   <!-- Transmission for ROS Control -->
   <transmission name="wheel_front_left_trans" type="transmission_interface/SimpleTransmission">
     <actuator name="wheel_front_left_motor">
       <mechanicalReduction>1</mechanicalReduction>
     </actuator>
     <joint name="front_left_wheel_hinge">
       <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
     </joint>
   </transmission>

   <!-- Gazebo ROS Control Plugin -->
   <gazebo>
     <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
       <robotNamespace>/my_robot</robotNamespace>
       <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
     </plugin>
   </gazebo>
   ```

3. **Create a launch file** to spawn the robot in the world:

   ```python
   # launch/robot_spawn.launch.py
```python
```
   from launch import LaunchDescription
```
```
```
```python
   from launch.actions import IncludeLaunchDescription
```
```
```
```python
   from launch.launch_description_sources import PythonLaunchDescriptionSource
```
```
```
```python
   from launch.substitutions import PathJoinSubstitution
```
```
```
```python
   from launch_ros.actions import Node
```
```
```
```python
   from launch_ros.substitutions import FindPackageShare

```
```
```
   def generate_launch_description():
```
```
       # Launch Gazebo with our custom world
```python
       gazebo = IncludeLaunchDescription(
```
```
           PythonLaunchDescriptionSource([
               PathJoinSubstitution([
```
```python
                   FindPackageShare('gazebo_ros'),
```
                   'launch',
                   'empty_world.launch.py'
```
               ])
           ]),
```python
           launch_arguments={
```
               'world': PathJoinSubstitution([
```python
                   FindPackageShare('my_robot_gazebo'),
```
                   'worlds',
                   'simple.world'
```
               ]),
```json
               'verbose': 'true'
```python
```
           }.items()
```
       )

       # Spawn robot in Gazebo
```python
       spawn_entity = Node(
```
           package='gazebo_ros',
           executable='spawn_entity.py',
           arguments=[
```
               '-topic', 'robot_description',
               '-entity', 'my_robot',
               '-x', '0.0',
               '-y', '0.0',
               '-z', '0.5'
```
           ],
```python
           output='screen'
```
       )

```python
       return LaunchDescription([
```
```
           gazebo,
           spawn_entity,
       ])
   ```

4. **Test the simulation** with a simple controller:

   ```python
   # scripts/simple_controller.py
   #!/usr/bin/env python3

```python
   import rclpy
```
```
```python
```
   from rclpy.node import Node
```
```
```python
   from geometry_msgs.msg import Twist
```
```
```python
   import sys, select, termios, tty

```
```
   msg = """
```
```
```
```
   Control Your Robot!
   -------------------
   Moving around:
      u    i    o
      j    k    l
      m    ,    .

   i: go straight
   o/k: rotate right/left
   ?: force stop

   CTRL-C to quit
```json
   """

```python
   moveBindings = `{`
```
```
```
       'i': (1, 0),
       'o': (1, -1),
       'j': (0, 1),
       'l': (0, -1),
       'u': (1, 1),
       ',': (-1, 0),
       '.': (-1, 1),
       'm': (-1, -1),
```
   }

```python
   class SimpleController(Node):
       def __init__(self):
```
           super().__init__('simple_controller')
```
```
```
           self.pub = self.create_publisher(Twist, '/my_robot/cmd_vel_unstamped', 10)

```python
       def getKey(self):
```
```
```
           tty.setraw(sys.stdin.fileno())
           select.select([sys.stdin], [], [], 0)
```python
           key = sys.stdin.read(1)
```
```
```
           termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
```python
           return key

```
```python
   def main():
```
```
```
       rclpy.init()
```python
       controller = SimpleController()
```
```
```
       controller.settings = termios.tcgetattr(sys.stdin)

```python
       print(msg)

       try:
           while True:
               key = controller.getKey()
               if key in moveBindings.keys():
                   twist = Twist()
```
```
```
                   twist.linear.x = moveBindings[key][0]
                   twist.angular.z = moveBindings[key][1]
```python
                   controller.pub.publish(twist)
```python
               elif key == '?':
```
```
```python
                   twist = Twist()
```
```
                   controller.pub.publish(twist)
```python
               else:
```
```
```python
                   if (key == '\x03'):  # Ctrl-C
```
```
```
                       break
```python
       except Exception as e:
           print(e)

```
```
       # Stop the robot
```python
       twist = Twist()
```
```
```
       controller.pub.publish(twist)
       termios.tcsetattr(sys.stdin, termios.TCSADRAIN, controller.settings)

       controller.destroy_node()
       rclpy.shutdown()
   ```

5. **Execute the workflow**:
   ```bash
   # Terminal 1: Launch the simulation
```
   ros2 launch my_robot_gazebo robot_spawn.launch.py

```python
   # Terminal 2: Control the robot (after the simulation starts)
```
   python3 scripts/simple_controller.py
   ```

### Step-by-Step Setup

1. **Create the world file** (`my_robot_gazebo/worlds/simple.world`)

2. **Verify your robot URDF** has proper Gazebo plugins for physics simulation

3. **Launch the simulation**:
   ```bash
   ros2 launch my_robot_gazebo robot_spawn.launch.py
   ```

4. **Test robot control** by publishing to joint command topics

## Summary

Gazebo provides a comprehensive physics simulation environment for robotics development. Understanding how to configure and use Gazebo effectively is essential for testing robotics algorithms before deployment on real hardware. The simulation allows for rapid prototyping and testing under various conditions without the risk or cost of physical hardware.

## Diagrams and Visual Aids

![Gazebo Simulation Environment](/img/gazebo-environment.png)

*Figure 1: Basic Gazebo simulation environment with robot and obstacles*

![Gazebo Architecture](/img/gazebo-architecture.png)

*Figure 2: Architecture of Gazebo simulation system with ROS integration*