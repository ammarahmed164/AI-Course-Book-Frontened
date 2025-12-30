---
sidebar_position: 4
title: "Bipedal Locomotion and Balance Control"
---

# Bipedal Locomotion and Balance Control

This lesson covers the principles of bipedal walking and balance control in humanoid robots.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the mechanics of bipedal locomotion
- Implement basic walking pattern generation
- Design balance control systems for humanoid robots
- Analyze the Zero Moment Point (ZMP) and its role in balance
- Understand the differences between static and dynamic walking
- Implement basic balance recovery strategies

## Introduction to Bipedal Locomotion

Bipedal locomotion is one of the most challenging aspects of humanoid robotics. Unlike wheeled robots or quadrupeds, bipedal robots have a small support base and must continuously maintain balance while walking.

### Challenges of Bipedal Locomotion

1. **Dynamic Stability**: Must maintain balance during movement
2. **Small Support Base**: Only two feet provide support
3. **Complex Control**: Many degrees of freedom to coordinate
4. **Energy Efficiency**: Human-like efficiency is difficult to achieve
5. **Terrain Adaptation**: Navigating uneven surfaces
6. **Obstacle Avoidance**: Moving around obstacles while walking

### Phases of Walking

A walking cycle consists of several phases:

#### Single Support Phase
- Only one foot is in contact with the ground
- Center of mass is typically moving over the support foot
- Requires active balance control

#### Double Support Phase
- Both feet are in contact with the ground
- Typically occurs during the transition between steps
- Provides stable support for CoM transfer

#### Swing Phase
- Non-support leg moves forward
- Requires coordination to avoid obstacles
- Affects the robot's overall balance

## Mathematical Models for Bipedal Locomotion

### Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model is a simplified representation of bipedal walking:

```
ẍ = ω²(x - x₀)
```

Where:
- `x` is the Center of Mass (CoM) position
- `x₀` is the Zero Moment Point (ZMP) position
- `ω² = g/h`, where `g` is gravity and `h` is CoM height

This model assumes:
- Constant CoM height
- No angular momentum about the CoM
- Point mass approximation

### Implementation of LIPM

```python
import numpy as np
import math

class LinearInvertedPendulumModel:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = math.sqrt(gravity / com_height)
        
        # Current state
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_z = com_height
        self.com_dx = 0.0
        self.com_dy = 0.0
        self.com_dz = 0.0
    
    def update_com_position(self, zmp_x, zmp_y, dt):
        """
        Update CoM position based on ZMP position
        """
        # Current CoM state
        x, y = self.com_x, self.com_y
        dx, dy = self.com_dx, self.com_dy
        
        # Update using LIPM equations
        # ẍ = ω²(x - zmp_x)
        # ÿ = ω²(y - zmp_y)
        ddx = self.omega**2 * (x - zmp_x)
        ddy = self.omega**2 * (y - zmp_y)
        
        # Integrate to get new velocities
        new_dx = dx + ddx * dt
        new_dy = dy + ddy * dt
        
        # Integrate to get new positions
        new_x = x + new_dx * dt
        new_y = y + new_dy * dt
        
        # Update state
        self.com_x, self.com_y = new_x, new_y
        self.com_dx, self.com_dy = new_dx, new_dy
    
    def calculate_zmp_trajectory(self, step_length, step_duration, n_steps):
        """
        Calculate ZMP trajectory for a simple walking pattern
        """
        zmp_trajectory = []
        
        for i in range(n_steps):
            t = i * step_duration / n_steps
            
            # Simple walking pattern: ZMP moves from one foot to another
            if t < step_duration / 2:
                # First half: ZMP under left foot
                zmp_x = -step_length / 2
                zmp_y = 0.1  # Slightly to the side for stability
            else:
                # Second half: ZMP under right foot
                zmp_x = step_length / 2
                zmp_y = 0.1
            
            zmp_trajectory.append((zmp_x, zmp_y))
        
        return zmp_trajectory

# Example usage
lipm = LinearInvertedPendulumModel(com_height=0.8)  # 80cm CoM height
zmp_trajectory = lipm.calculate_zmp_trajectory(step_length=0.3, step_duration=1.0, n_steps=100)

# Simulate walking
dt = 0.01  # 100Hz control
for zmp_x, zmp_y in zmp_trajectory:
    lipm.update_com_position(zmp_x, zmp_y, dt)
```

### Capture Point Model

The Capture Point is the location where a biped can come to a stop:

```
Capture Point = CoM + CoM_velocity / ω
```

This point indicates where the foot should be placed to stop the robot's momentum.

```python
def calculate_capture_point(com_pos, com_vel, omega):
    """
    Calculate the capture point for stopping
    """
    capture_point_x = com_pos[0] + com_vel[0] / omega
    capture_point_y = com_pos[1] + com_vel[1] / omega
    
    return np.array([capture_point_x, capture_point_y])

def is_stable(com_pos, zmp_pos, support_polygon):
    """
    Check if the robot is in a stable state
    """
    # Check if ZMP is within the support polygon
    return point_in_polygon(zmp_pos, support_polygon)

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
```

## Walking Pattern Generation

### Footstep Planning

Footstep planning involves determining where and when to place the feet:

```python
class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        
        # Current foot positions
        self.left_foot = np.array([0.0, step_width/2, 0.0])
        self.right_foot = np.array([0.0, -step_width/2, 0.0])
        
        # Walking state
        self.swing_foot = "right"  # Which foot is swinging next
        self.step_count = 0
    
    def plan_next_step(self, walk_direction, step_size=0.3):
        """
        Plan the next step based on walking direction
        """
        # Determine which foot to move
        if self.swing_foot == "left":
            current_foot_pos = self.left_foot
            support_foot_pos = self.right_foot
        else:
            current_foot_pos = self.right_foot
            support_foot_pos = self.left_foot
        
        # Calculate new foot position
        new_foot_pos = current_foot_pos.copy()
        
        # Move in the walking direction
        new_foot_pos[0] += step_size * walk_direction[0]
        new_foot_pos[1] = support_foot_pos[1]  # Alternate sides
        
        # Add step height during swing phase
        new_foot_pos[2] = self.step_height  # Swing height
        
        # Update state
        if self.swing_foot == "left":
            self.left_foot = new_foot_pos
            self.swing_foot = "right"
        else:
            self.right_foot = new_foot_pos
            self.swing_foot = "left"
        
        self.step_count += 1
        
        return new_foot_pos
    
    def generate_walk_pattern(self, steps, direction="forward"):
        """
        Generate a sequence of footsteps
        """
        if direction == "forward":
            walk_dir = np.array([1.0, 0.0])
        elif direction == "backward":
            walk_dir = np.array([-1.0, 0.0])
        elif direction == "left":
            walk_dir = np.array([0.0, 1.0])
        elif direction == "right":
            walk_dir = np.array([0.0, -1.0])
        else:
            walk_dir = np.array([1.0, 0.0])  # Default forward
        
        footsteps = []
        for i in range(steps):
            step_pos = self.plan_next_step(walk_dir)
            footsteps.append(step_pos)
        
        return footsteps
```

### Trajectory Generation

Generating smooth trajectories for feet, CoM, and joints:

```python
class WalkingTrajectoryGenerator:
    def __init__(self, com_height=0.8, step_duration=1.0):
        self.com_height = com_height
        self.step_duration = step_duration
        self.lipm = LinearInvertedPendulumModel(com_height)
        
        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.2
        self.step_height = 0.05
    
    def generate_foot_trajectory(self, start_pos, end_pos, step_height, n_points=50):
        """
        Generate a smooth trajectory for foot movement
        """
        trajectory = []
        
        # Linear interpolation for x, y
        x_traj = np.linspace(start_pos[0], end_pos[0], n_points)
        y_traj = np.linspace(start_pos[1], end_pos[1], n_points)
        
        # Parabolic trajectory for z (step over)
        z_traj = []
        for i in range(n_points):
            t = i / (n_points - 1)  # Normalized time (0 to 1)
            
            if t < 0.5:
                # Rising phase (0 to 1)
                z_pos = start_pos[2] + (step_height - start_pos[2]) * (2 * t)**2
            else:
                # Falling phase (1 to 0)
                z_pos = start_pos[2] + (step_height - start_pos[2]) * (2 * (1 - t))**2
            
            z_traj.append(z_pos)
        
        # Combine into full trajectory
        for x, y, z in zip(x_traj, y_traj, z_traj):
            trajectory.append(np.array([x, y, z]))
        
        return trajectory
    
    def generate_com_trajectory(self, zmp_trajectory, dt=0.01):
        """
        Generate CoM trajectory based on ZMP trajectory
        """
        com_trajectory = []
        self.lipm = LinearInvertedPendulumModel(self.com_height)
        
        for zmp_pos in zmp_trajectory:
            self.lipm.update_com_position(zmp_pos[0], zmp_pos[1], dt)
            com_pos = np.array([self.lipm.com_x, self.lipm.com_y, self.lipm.com_z])
            com_trajectory.append(com_pos)
        
        return com_trajectory
    
    def generate_zmp_trajectory(self, footsteps, support_time_ratio=0.8):
        """
        Generate ZMP trajectory based on footsteps
        """
        zmp_trajectory = []
        
        for i, foot_pos in enumerate(footsteps):
            # Calculate duration for this step
            step_duration = int(self.step_duration / 0.01)  # Convert to number of steps at 100Hz
            
            # Determine support foot (alternates with each step)
            if i % 2 == 0:
                # Left foot support
                support_pos = foot_pos
            else:
                # Right foot support
                support_pos = foot_pos
            
            # Generate ZMP for this step
            for j in range(step_duration):
                t = j / step_duration
                
                if t < support_time_ratio:
                    # During support phase: ZMP at support foot
                    zmp_x = support_pos[0]
                    zmp_y = support_pos[1]
                else:
                    # During transition phase: move ZMP toward next support
                    next_support = footsteps[i+1] if i+1 < len(footsteps) else support_pos
                    transition_ratio = (t - support_time_ratio) / (1 - support_time_ratio)
                    zmp_x = support_pos[0] + (next_support[0] - support_pos[0]) * transition_ratio
                    zmp_y = support_pos[1] + (next_support[1] - support_pos[1]) * transition_ratio
                
                zmp_trajectory.append(np.array([zmp_x, zmp_y]))
        
        return zmp_trajectory
```

## Balance Control Systems

### PID-Based Balance Control

```python
class BalanceController:
    def __init__(self, kp=10.0, ki=1.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Error accumulation
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Sensor inputs
        self.current_com = np.zeros(3)
        self.desired_com = np.zeros(3)
        self.current_zmp = np.zeros(2)
        self.desired_zmp = np.zeros(2)
        
        # Control outputs
        self.joint_corrections = np.zeros(28)  # Example for 28 DOF robot
    
    def update_sensors(self, com_position, zmp_position):
        """
        Update sensor readings
        """
        self.current_com = np.array(com_position)
        self.current_zmp = np.array(zmp_position)
    
    def compute_balance_control(self, dt=0.01):
        """
        Compute balance control corrections
        """
        # Calculate errors
        com_error = self.desired_com[:2] - self.current_com[:2]  # X, Y only
        zmp_error = self.desired_zmp - self.current_zmp
        
        # Combine errors (may weight differently)
        total_error = com_error + 0.5 * zmp_error
        
        # Update integral term
        self.integral_error += total_error * dt
        
        # Calculate derivative term
        derivative_error = (total_error - self.previous_error) / dt
        
        # PID control
        control_output = (self.kp * total_error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
        
        # Store current error for next iteration
        self.previous_error = total_error.copy()
        
        # Convert control output to joint corrections
        # This would involve inverse kinematics to determine
        # appropriate joint angle adjustments
        self.joint_corrections = self.map_control_to_joints(control_output)
        
        return self.joint_corrections
    
    def map_control_to_joints(self, control_output):
        """
        Map balance control output to joint adjustments
        """
        # This is a simplified mapping
        # In reality, this would use whole-body control
        joint_corrections = np.zeros(28)
        
        # Map CoM errors to appropriate joints
        # Example mappings (would be more complex in reality):
        joint_corrections[0] = control_output[0] * 0.1  # Ankle pitch
        joint_corrections[1] = control_output[1] * 0.1  # Ankle roll
        joint_corrections[2] = control_output[0] * 0.05  # Hip pitch
        joint_corrections[3] = control_output[1] * 0.05  # Hip roll
        
        return joint_corrections
    
    def set_desired_state(self, desired_com, desired_zmp):
        """
        Set desired CoM and ZMP positions
        """
        self.desired_com = desired_com
        self.desired_zmp = desired_zmp
```

### Whole-Body Balance Control

```python
class WholeBodyBalanceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.balance_controller = BalanceController()
        self.com_estimator = CenterOfMassEstimator(robot_model)
        self.zmp_estimator = ZMPEstimator(robot_model)
        
        # Support polygon
        self.support_polygon = []
        self.update_support_polygon()
    
    def update_support_polygon(self):
        """
        Update the support polygon based on contact points
        """
        # Determine which feet are in contact
        left_contact = self.robot_model.is_foot_in_contact("left")
        right_contact = self.robot_model.is_foot_in_contact("right")
        
        if left_contact and right_contact:
            # Double support: polygon connecting both feet
            left_pos = self.robot_model.get_foot_position("left")
            right_pos = self.robot_model.get_foot_position("right")
            
            # Simplified polygon - in reality would consider foot shapes
            self.support_polygon = [
                [left_pos[0], left_pos[1] + 0.1],   # Left foot top
                [right_pos[0], right_pos[1] + 0.1], # Right foot top
                [right_pos[0], right_pos[1] - 0.1], # Right foot bottom
                [left_pos[0], left_pos[1] - 0.1]    # Left foot bottom
            ]
        elif left_contact:
            # Single support: polygon around left foot
            left_pos = self.robot_model.get_foot_position("left")
            self.support_polygon = [
                [left_pos[0] - 0.1, left_pos[1] - 0.05],
                [left_pos[0] + 0.1, left_pos[1] - 0.05],
                [left_pos[0] + 0.1, left_pos[1] + 0.05],
                [left_pos[0] - 0.1, left_pos[1] + 0.05]
            ]
        elif right_contact:
            # Single support: polygon around right foot
            right_pos = self.robot_model.get_foot_position("right")
            self.support_polygon = [
                [right_pos[0] - 0.1, right_pos[1] - 0.05],
                [right_pos[0] + 0.1, right_pos[1] - 0.05],
                [right_pos[0] + 0.1, right_pos[1] + 0.05],
                [right_pos[0] - 0.1, right_pos[1] + 0.05]
            ]
        else:
            # No support - emergency state
            self.support_polygon = []
    
    def compute_balance_control(self, dt=0.01):
        """
        Compute whole-body balance control
        """
        # Update sensor readings
        current_com = self.com_estimator.estimate_com()
        current_zmp = self.zmp_estimator.estimate_zmp()
        
        self.balance_controller.update_sensors(current_com, current_zmp)
        
        # Check if robot is in stable state
        if not self.is_stable(current_zmp):
            # Emergency balance recovery
            return self.emergency_balance_recovery(current_com, current_zmp)
        
        # Normal balance control
        return self.balance_controller.compute_balance_control(dt)
    
    def is_stable(self, zmp_position):
        """
        Check if the robot is in a stable state
        """
        if not self.support_polygon:
            return False  # No support is unstable
        
        return point_in_polygon(zmp_position[:2], self.support_polygon)
    
    def emergency_balance_recovery(self, current_com, current_zmp):
        """
        Execute emergency balance recovery
        """
        # Determine the direction to move the CoM back to stability
        center_of_support = np.mean(self.support_polygon, axis=0)
        
        # Calculate required CoM movement
        recovery_vector = center_of_support - current_com[:2]
        
        # Generate appropriate joint commands to move CoM
        joint_commands = self.generate_recovery_commands(recovery_vector)
        
        return joint_commands
    
    def generate_recovery_commands(self, recovery_vector):
        """
        Generate joint commands for balance recovery
        """
        # This would implement specific recovery strategies
        # such as stepping, crouching, or arm movements
        recovery_commands = np.zeros(28)
        
        # Example: Move hip joints to shift CoM
        recovery_commands[0] = recovery_vector[0] * 0.5  # Hip forward/back
        recovery_commands[1] = recovery_vector[1] * 0.5  # Hip lateral
        
        # Example: Move arms to create counter-moment
        recovery_commands[10] = -recovery_vector[1] * 0.2  # Left arm
        recovery_commands[15] = -recovery_vector[1] * 0.2  # Right arm
        
        return recovery_commands

class CenterOfMassEstimator:
    def __init__(self, robot_model):
        self.robot_model = robot_model
    
    def estimate_com(self):
        """
        Estimate the Center of Mass position
        """
        # Get all link masses and positions
        total_mass = 0.0
        weighted_sum = np.zeros(3)
        
        for link in self.robot_model.links:
            mass = self.robot_model.get_link_mass(link)
            pos = self.robot_model.get_link_position(link)
            
            total_mass += mass
            weighted_sum += mass * pos
        
        if total_mass > 0:
            com = weighted_sum / total_mass
        else:
            com = np.zeros(3)
        
        return com

class ZMPEstimator:
    def __init__(self, robot_model):
        self.robot_model = robot_model
    
    def estimate_zmp(self):
        """
        Estimate the Zero Moment Point
        """
        # Get ground reaction forces from force sensors
        left_force = self.robot_model.get_foot_force("left")
        right_force = self.robot_model.get_foot_force("right")
        
        left_pos = self.robot_model.get_foot_position("left")
        right_pos = self.robot_model.get_foot_position("right")
        
        # Calculate ZMP based on ground reaction forces
        total_fz = left_force[2] + right_force[2]
        
        if abs(total_fz) > 1e-6:  # Avoid division by zero
            zmp_x = (left_pos[0] * left_force[2] + right_pos[0] * right_force[2]) / total_fz
            zmp_y = (left_pos[1] * left_force[2] + right_pos[1] * right_force[2]) / total_fz
        else:
            # Use average foot position if no ground contact
            zmp_x = (left_pos[0] + right_pos[0]) / 2
            zmp_y = (left_pos[1] + right_pos[1]) / 2
        
        return np.array([zmp_x, zmp_y])
```

## Walking Control Strategies

### Preview Control

Preview control uses future ZMP references to generate stable walking:

```python
class PreviewController:
    def __init__(self, preview_time=2.0, com_height=0.8, dt=0.01):
        self.preview_time = preview_time
        self.com_height = com_height
        self.dt = dt
        self.omega = math.sqrt(9.81 / com_height)
        
        # Calculate preview gains
        self.kx, self.ky = self.calculate_preview_gains()
    
    def calculate_preview_gains(self):
        """
        Calculate preview control gains
        """
        # Simplified gain calculation
        # In reality, this would involve more complex mathematics
        kx = self.omega
        ky = self.omega
        
        return kx, ky
    
    def calculate_control(self, current_com, current_com_vel, future_zmp_refs):
        """
        Calculate control input using preview control
        """
        # Current error
        zmp_ref_now = future_zmp_refs[0]
        error_x = current_com[0] - zmp_ref_now[0]
        error_y = current_com[1] - zmp_ref_now[1]
        
        # Preview term (sum of future errors weighted by exponentially decaying function)
        preview_sum_x = 0.0
        preview_sum_y = 0.0
        
        for i, zmp_ref in enumerate(future_zmp_refs):
            time_ahead = i * self.dt
            weight = math.exp(-self.omega * time_ahead)
            
            preview_sum_x += weight * (current_com[0] - zmp_ref[0])
            preview_sum_y += weight * (current_com[1] - zmp_ref[1])
        
        # Control law
        control_x = self.kx * error_x + self.omega * preview_sum_x * self.dt
        control_y = self.ky * error_y + self.omega * preview_sum_y * self.dt
        
        return np.array([control_x, control_y])
```

### Model Predictive Control (MPC)

Model Predictive Control optimizes future behavior over a prediction horizon:

```python
class ModelPredictiveController:
    def __init__(self, prediction_horizon=10, dt=0.01):
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        
        # Weights for the cost function
        self.state_weight = 1.0
        self.control_weight = 0.1
        self.terminal_weight = 5.0
    
    def optimize_walking(self, initial_state, reference_trajectory):
        """
        Optimize walking control using MPC
        """
        # This would implement an optimization algorithm
        # to minimize a cost function over the prediction horizon
        
        # Simplified implementation using linear quadratic regulator (LQR)
        # In practice, this would use more sophisticated optimization
        
        # System matrices for LIPM
        A = np.array([[1, self.dt, self.dt**2/2], 
                      [0, 1, self.dt], 
                      [0, 0, 1]])  # State transition matrix
        B = np.array([[self.dt**3/6], [self.dt**2/2], [self.dt]])  # Control matrix
        
        # Solve Riccati equation for optimal control
        # (Simplified - full implementation would be more complex)
        
        # For now, return a simple control based on current error
        current_error = initial_state[:2] - reference_trajectory[0][:2]
        control_input = -0.5 * current_error  # Simplified feedback
        
        return control_input
```

## Balance Recovery Strategies

### Stepping Strategy

When balance is lost, stepping can help recover:

```python
class BalanceRecovery:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.footstep_planner = FootstepPlanner()
    
    def should_step(self, com_pos, com_vel, zmp_pos):
        """
        Determine if a step is needed for balance recovery
        """
        # Calculate capture point
        omega = math.sqrt(9.81 / self.robot_model.get_com_height())
        capture_point = calculate_capture_point(com_pos, com_vel, omega)
        
        # Check if capture point is outside support polygon
        support_polygon = self.get_current_support_polygon()
        
        if not point_in_polygon(capture_point, support_polygon):
            # Capture point outside support - need to step
            return True, capture_point
        
        return False, capture_point
    
    def calculate_recover_step(self, capture_point):
        """
        Calculate where to place the recovery step
        """
        # Determine which foot to use for recovery
        # (simplified - in reality would consider multiple factors)
        
        # Calculate step location - typically at or beyond capture point
        step_location = capture_point.copy()
        
        # Add safety margin
        com_pos = self.robot_model.get_com_position()
        direction_to_capture = step_location - com_pos[:2]
        direction_to_capture = direction_to_capture / np.linalg.norm(direction_to_capture)
        
        # Place step beyond capture point for safety
        step_location = capture_point + 0.1 * direction_to_capture
        
        return step_location
```

## Practical Exercise: Implementing Simple Walking Controller

Create a simplified walking controller:

1. **Implement a basic LIPM** for CoM control
2. **Create a footstep planner** for simple walking
3. **Implement a balance controller** to maintain stability
4. **Test with simple walking patterns**
5. **Add basic recovery mechanisms**

### Complete Walking Controller Example

```python
class SimpleWalkingController:
    def __init__(self, com_height=0.8):
        self.com_height = com_height
        self.lipm = LinearInvertedPendulumModel(com_height)
        self.footstep_planner = FootstepPlanner()
        self.balance_controller = BalanceController()
        self.trajectory_generator = WalkingTrajectoryGenerator(com_height)
        
        # Walking state
        self.is_walking = False
        self.walk_speed = 0.0
        self.walk_direction = np.array([1.0, 0.0])
        
        # Control parameters
        self.control_freq = 100  # Hz
        self.dt = 1.0 / self.control_freq
    
    def start_walking(self, speed=0.3, direction="forward"):
        """
        Start walking with specified speed and direction
        """
        self.is_walking = True
        self.walk_speed = speed
        
        if direction == "forward":
            self.walk_direction = np.array([1.0, 0.0])
        elif direction == "backward":
            self.walk_direction = np.array([-1.0, 0.0])
        elif direction == "left":
            self.walk_direction = np.array([0.0, 1.0])
        elif direction == "right":
            self.walk_direction = np.array([0.0, -1.0])
        else:
            self.walk_direction = np.array([1.0, 0.0])
    
    def stop_walking(self):
        """
        Stop walking
        """
        self.is_walking = False
        self.walk_speed = 0.0
    
    def update(self):
        """
        Main control loop - call at control frequency
        """
        if not self.is_walking:
            return np.zeros(28)  # No movement when stopped
        
        # Calculate desired ZMP based on walking pattern
        desired_zmp = self.calculate_walking_zmp()
        
        # Update LIPM
        self.lipm.update_com_position(desired_zmp[0], desired_zmp[1], self.dt)
        
        # Update balance controller
        current_com = np.array([self.lipm.com_x, self.lipm.com_y, self.lipm.com_z])
        current_zmp = desired_zmp
        
        self.balance_controller.update_sensors(current_com, current_zmp)
        
        # Calculate balance corrections
        balance_corrections = self.balance_controller.compute_balance_control(self.dt)
        
        # Apply walking pattern + balance corrections
        joint_commands = balance_corrections  # Simplified
        
        return joint_commands
    
    def calculate_walking_zmp(self):
        """
        Calculate desired ZMP for walking
        """
        # Simplified walking pattern
        # In reality, this would be more sophisticated
        time_in_step = (self.get_current_time() % 1.0)  # Assume 1s step cycle
        
        if time_in_step < 0.5:
            # Left foot support
            zmp_x = -0.1  # Slightly to the left
        else:
            # Right foot support
            zmp_x = 0.1   # Slightly to the right
        
        # Move forward with walking
        zmp_x += self.get_current_time() * self.walk_speed
        
        return np.array([zmp_x, 0.0])
    
    def get_current_time(self):
        """
        Get current simulation time
        """
        # In a real implementation, this would come from the robot's clock
        import time
        return time.time() % 1000  # Simplified time tracking
```

## Summary

Bipedal locomotion and balance control is one of the most challenging aspects of humanoid robotics. It requires understanding of dynamics, control theory, and biomechanics. The Linear Inverted Pendulum Model provides a useful simplification for understanding balance, while ZMP control offers a practical approach to maintaining stability.

Key challenges include maintaining balance during dynamic movement, adapting to terrain variations, and implementing effective recovery strategies when balance is lost. Successful walking requires coordination of multiple control systems working together.

## Next Steps

In the next lesson, we'll explore manipulation and grasping with humanoid hands, which also requires sophisticated control and understanding of dynamics.