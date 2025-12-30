---
sidebar_position: 5
title: "Physics Simulation and Sensor Simulation"
---

# Physics Simulation and Sensor Simulation

This lesson covers how to configure physics properties and simulate sensors in Gazebo.

## Learning Objectives

After completing this lesson, you will be able to:
- Configure physics properties for realistic simulation
- Implement sensor simulation in Gazebo
- Understand the differences between simulated and real sensors
- Calibrate simulation parameters to match real hardware
- Troubleshoot common physics and sensor issues

## Physics Simulation in Gazebo

Physics simulation is crucial for creating realistic robot behavior in simulation. Gazebo uses the Open Dynamics Engine (ODE) by default, but also supports other physics engines.

### Physics Engine Configuration

The physics engine parameters are defined in the world file:

```xml
<physics type="ode" name="default_physics">
  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>
  
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  
  <!-- ODE-specific settings -->
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

### Understanding Physics Parameters

#### max_step_size
- **Purpose**: Size of each simulation time step
- **Typical values**: 0.001s for precise simulation, 0.01s for faster performance
- **Trade-offs**: Smaller steps = more accuracy but slower performance

#### real_time_factor
- **Purpose**: Maximum ratio of simulation time to real time
- **Value of 1**: Simulation runs at real-time speed
- **Higher values**: Simulation runs faster than real-time

#### Solver Parameters
- **iters**: Number of iterations for constraint solving
- **sor**: Successive over-relaxation parameter (damping)
- **Higher values**: More stable but slower

### Tuning Physics for Different Scenarios

#### For Manipulation Tasks
```xml
<physics type="ode">
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000</real_time_update_rate>
  <ode>
    <solver>
      <iters>100</iters>
      <sor>1.2</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>
      <erp>0.8</erp>
      <contact_max_correcting_vel>10</contact_max_correcting_vel>
      <contact_surface_layer>0.005</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

#### For Navigation Tasks
```xml
<physics type="ode">
  <max_step_size>0.01</max_step_size>
  <real_time_update_rate>100</real_time_update_rate>
  <ode>
    <solver>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>1e-3</cfm>
      <erp>0.2</erp>
    </constraints>
  </ode>
</physics>
```

## Sensor Simulation

Gazebo provides realistic simulation of various sensor types commonly used in robotics.

### Camera Sensors

Camera sensors simulate RGB cameras with realistic distortion and noise parameters.

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>camera</namespace>
      <remapping>image_raw:=image</remapping>
      <remapping>camera_info:=camera_info</remapping>
    </ros>
    <camera_name>camera</camera_name>
    <image_topic_name>image</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>camera_link</frame_name>
  </plugin>
</sensor>
```

### LIDAR Sensors

LIDAR sensors simulate 2D and 3D laser range finders with realistic noise and accuracy.

#### 2D LIDAR (Ray Sensor)
```xml
<sensor name="lidar_2d" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
    <ros>
      <namespace>lidar</namespace>
      <remapping>scan:=scan</remapping>
    </ros>
    <topic_name>scan</topic_name>
    <frame_name>lidar_link</frame_name>
  </plugin>
</sensor>
```

#### 3D LIDAR (GPU Ray Sensor)
```xml
<sensor name="lidar_3d" type="gpu_ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <gpu_ray>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
        <max_angle>0.1745</max_angle>   <!-- 10 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </gpu_ray>
  <plugin name="gpu_lidar_controller" filename="libgazebo_ros_gpu_laser.so">
    <ros>
      <namespace>gpu_lidar</namespace>
      <remapping>points:=points</remapping>
    </ros>
    <topic_name>points</topic_name>
    <frame_name>lidar_link</frame_name>
  </plugin>
</sensor>
```

### IMU Sensors

IMU sensors simulate accelerometers, gyroscopes, and magnetometers.

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>  <!-- ~0.1 deg/s -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00017</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00017</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00017</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>  <!-- ~0.0017g -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>1.7e-3</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>1.7e-3</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>1.7e-3</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>imu</namespace>
      <remapping>imu:=data</remapping>
    </ros>
    <topic_name>data</topic_name>
    <frame_name>imu_link</frame_name>
    <body_name>imu_link</body_name>
  </plugin>
</sensor>
```

### Force/Torque Sensors

Force/Torque sensors simulate load cells and other force measurement devices.

```xml
<sensor name="force_torque_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_controller" filename="libgazebo_ros_ft_sensor.so">
    <ros>
      <namespace>ft_sensor</namespace>
      <remapping>wrench:=wrench</remapping>
    </ros>
    <topic_name>wrench</topic_name>
    <frame_name>ft_sensor_link</frame_name>
  </plugin>
</sensor>
```

## Physics Properties for Links

Each link can have specific physics properties that affect how it interacts with the simulation.

### Collision Properties

```xml
<link name="my_link">
  <collision name="collision">
    <geometry>
      <box>
        <size>0.1 0.1 0.1</size>
      </box>
    </geometry>
    <!-- Surface properties -->
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
          <fdir1>0 0 0</fdir1>
          <slip1>0</slip1>
          <slip2>0</slip2>
        </ode>
        <torsional>
          <coefficient>1.0</coefficient>
          <use_patch_radius>true</use_patch_radius>
          <surface_radius>0.01</surface_radius>
          <patch_radius>0.01</patch_radius>
        </torsional>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <collide_without_contact>false</collide_without_contact>
        <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
        <collide_bitmask>1</collide_bitmask>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1e+13</kp>
          <kd>1</kd>
          <max_vel>0.01</max_vel>
          <min_depth>0</min_depth>
        </ode>
        <bullet>
          <split_impulse>true</split_impulse>
          <split_impulse_penetration_threshold>-0.01</split_impulse_penetration_threshold>
        </bullet>
      </contact>
    </surface>
  </collision>
</link>
```

### Inertial Properties

```xml
<link name="my_link">
  <inertial>
    <mass>0.5</mass>
    <inertia>
      <ixx>0.001</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.001</iyy>
      <iyz>0</iyz>
      <izz>0.001</izz>
    </inertia>
  </inertial>
</link>
```

## Sensor Calibration and Validation

### Matching Real Hardware

To make simulation more realistic, calibrate sensor parameters to match real hardware:

1. **Camera Calibration**: Use camera calibration tools to get intrinsic parameters
2. **LIDAR Calibration**: Compare range measurements between real and simulated sensors
3. **IMU Calibration**: Compare noise characteristics and bias values
4. **Actuator Calibration**: Match motor dynamics and response times

### Validation Techniques

#### Quantitative Validation
- Compare sensor data from real and simulated robots
- Measure timing differences
- Validate kinematic and dynamic behavior

#### Qualitative Validation
- Visual inspection of robot behavior
- Comparison of trajectories and motion patterns
- Assessment of sensor data quality

## Practical Exercise: Sensor Integration

Create a robot model with multiple sensors and validate their behavior:

1. **Create a robot model** with:
   - RGB camera
   - 2D LIDAR
   - IMU
   - Force/torque sensor

2. **Configure physics parameters** for realistic movement

3. **Test the robot** in a simple environment

4. **Compare sensor outputs** with expected values

## Common Issues and Troubleshooting

### Physics Issues
- **Robot falls through ground**: Check collision geometries and static properties
- **Unstable simulation**: Reduce step size or adjust solver parameters
- **Joints behave strangely**: Check joint limits and damping parameters

### Sensor Issues
- **No sensor data**: Verify plugin configuration and ROS topics
- **Incorrect sensor readings**: Check sensor pose and calibration parameters
- **Performance issues**: Reduce update rate or simplify sensor models

## Best Practices

1. **Start simple**: Begin with basic physics and add complexity gradually
2. **Validate parameters**: Ensure simulation parameters match real hardware
3. **Monitor performance**: Balance accuracy with simulation speed
4. **Use realistic noise**: Add appropriate noise models to sensor data
5. **Test regularly**: Validate simulation behavior against real-world expectations

## Summary

Physics and sensor simulation are critical components of effective robot simulation. Proper configuration of physics parameters and sensor models ensures realistic robot behavior and accurate sensor data. Understanding how to tune these parameters for specific applications is essential for effective robotics development.

## Next Steps

In the next lesson, we'll explore Unity for robot visualization and how it compares to Gazebo.