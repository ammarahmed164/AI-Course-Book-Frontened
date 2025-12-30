---
sidebar_position: 3
title: "Humanoid Robot Kinematics and Dynamics"
---

# Humanoid Robot Kinematics and Dynamics

This lesson covers the kinematic and dynamic principles that govern humanoid robot movement and control.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand forward and inverse kinematics for humanoid robots
- Apply kinematic models to solve manipulation and locomotion problems
- Analyze the dynamic properties of humanoid robots
- Implement basic kinematic and dynamic calculations
- Understand the relationship between kinematics and control

## Introduction to Humanoid Kinematics

Kinematics is the study of motion without considering the forces that cause it. For humanoid robots, kinematics is crucial for:
- Planning arm movements to reach specific positions
- Controlling leg movements for walking
- Coordinating multiple limbs for complex tasks
- Understanding the relationship between joint angles and end-effector positions

### Key Concepts in Humanoid Kinematics

#### Degrees of Freedom (DOF)
The number of independent movements a robot can make. Humanoid robots typically have:
- **Upper body**: 20-30 DOF (including arms, hands, and head)
- **Lower body**: 12-16 DOF (legs and feet)
- **Total**: 30-50+ DOF for full humanoid robots

#### Kinematic Chains
Humanoid robots consist of multiple kinematic chains:
- **Right arm**: Shoulder → Elbow → Wrist → Hand
- **Left arm**: Shoulder → Elbow → Wrist → Hand
- **Right leg**: Hip → Knee → Ankle → Foot
- **Left leg**: Hip → Knee → Ankle → Foot
- **Head/neck**: Yaw → Pitch (and sometimes roll)

## Forward Kinematics

Forward kinematics calculates the position and orientation of the end-effector given the joint angles.

### Mathematical Representation

For each joint in a chain, we define a transformation matrix that describes how coordinates transform from one joint to the next:

```
T_i^(i-1) = [R_i^(i-1)  p_i^(i-1)]
            [   0^T       1      ]
```

Where:
- `R_i^(i-1)` is the 3×3 rotation matrix
- `p_i^(i-1)` is the 3×1 position vector
- `T_i^(i-1)` is the 4×4 transformation matrix

### Forward Kinematics Example

```python
import numpy as np
import math

def rotation_matrix_z(theta):
    """Rotation matrix around Z-axis"""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0,               0,               1]
    ])

def translation_vector(dx, dy, dz):
    """Translation vector"""
    return np.array([dx, dy, dz])

def dh_transform(a, alpha, d, theta):
    """Denavit-Hartenberg transformation matrix"""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def forward_kinematics(joint_angles, dh_params):
    """
    Calculate forward kinematics for a simple arm
    joint_angles: list of joint angles [theta1, theta2, ...]
    dh_params: list of DH parameters [(a1, alpha1, d1, q_offset1), ...]
    """
    T = np.eye(4)  # Identity matrix
    
    for i, (theta, (a, alpha, d, q_offset)) in enumerate(zip(joint_angles, dh_params)):
        # Add joint angle to the parameter
        T_link = dh_transform(a, alpha, d, theta + q_offset)
        T = T @ T_link
    
    return T

# Example: Simple 2-DOF planar arm
# DH parameters: (a, alpha, d, q_offset)
dh_params = [
    (0.3, 0, 0, 0),   # Link 1: length 0.3m
    (0.3, 0, 0, 0)    # Link 2: length 0.3m
]

joint_angles = [math.pi/4, math.pi/6]  # 45° and 30°
end_effector_pose = forward_kinematics(joint_angles, dh_params)

print("End-effector position:", end_effector_pose[:3, 3])
```

### Humanoid-Specific Considerations

For humanoid robots, we need to consider multiple kinematic chains simultaneously:

```python
class HumanoidKinematics:
    def __init__(self):
        # Define DH parameters for each limb
        self.right_arm_dh = [
            (0.15, -math.pi/2, 0, 0),      # Shoulder joint 1
            (0,     math.pi/2,  0.2, 0),    # Shoulder joint 2
            (0.3,   0,          0, 0),      # Upper arm
            (0,    -math.pi/2,  0.3, 0),    # Forearm
        ]
        
        self.left_arm_dh = self.right_arm_dh  # Symmetric
        
        self.right_leg_dh = [
            (0,     math.pi/2,  0.1, 0),    # Hip joint 1
            (0,    -math.pi/2,  0.05, 0),   # Hip joint 2
            (0.4,   0,          0, 0),      # Thigh
            (0.4,   0,          0, 0),      # Shin
        ]
    
    def calculate_arm_pose(self, arm, joint_angles):
        """Calculate pose for right or left arm"""
        if arm == "right":
            dh_params = self.right_arm_dh
        elif arm == "left":
            dh_params = self.left_arm_dh
        else:
            raise ValueError("Arm must be 'right' or 'left'")
        
        return forward_kinematics(joint_angles, dh_params)
    
    def calculate_leg_pose(self, leg, joint_angles):
        """Calculate pose for right or left leg"""
        if leg == "right":
            dh_params = self.right_leg_dh
        elif leg == "left":
            dh_params = self.right_leg_dh  # Simplified - in reality, left leg would be different
        else:
            raise ValueError("Leg must be 'right' or 'left'")
        
        return forward_kinematics(joint_angles, dh_params)
```

## Inverse Kinematics

Inverse kinematics (IK) calculates the joint angles required to achieve a desired end-effector position and orientation.

### Analytical vs. Numerical Solutions

#### Analytical Solutions
- Exact solutions for simple kinematic chains
- Fast computation
- Limited to specific configurations
- Good for redundant manipulators with known geometry

#### Numerical Solutions
- Approximate solutions for complex chains
- Slower computation
- Applicable to any configuration
- Better for redundant systems

### Jacobian-Based IK

The Jacobian matrix relates joint velocities to end-effector velocities:

```
v_e = J(q) * q_dot
```

Where:
- `v_e` is the end-effector velocity
- `J(q)` is the Jacobian matrix
- `q_dot` is the joint velocity

For inverse kinematics:
```
q_dot = J^(-1) * v_e
```

When the Jacobian is not square (redundant system), we use the pseudoinverse:
```
q_dot = J^T * (J * J^T)^(-1) * v_e
```

### Implementation Example

```python
def jacobian(robot_config, joint_angles, link_index):
    """
    Calculate the Jacobian matrix for a robotic arm
    """
    n = len(joint_angles)
    J = np.zeros((6, n))  # 6 DOF (3 translation + 3 rotation)
    
    # Calculate end-effector position and orientation
    T_total = forward_kinematics(joint_angles, robot_config)
    end_effector_pos = T_total[:3, 3]
    
    # Calculate Jacobian columns
    for i in range(n):
        # Calculate transformation up to joint i
        T_to_i = forward_kinematics(joint_angles[:i+1], robot_config[:i+1])
        joint_pos = T_to_i[:3, 3]
        
        # Calculate z-axis of joint i in global frame
        z_i = T_to_i[:3, 2]  # Third column is z-axis
        
        # Position part of Jacobian
        J[:3, i] = np.cross(z_i, end_effector_pos - joint_pos)
        
        # Orientation part of Jacobian
        J[3:, i] = z_i
    
    return J

def inverse_kinematics(target_pose, current_joints, robot_config, max_iterations=100, tolerance=1e-4):
    """
    Solve inverse kinematics using Jacobian transpose method
    """
    current_pose = forward_kinematics(current_joints, robot_config)
    error = target_pose - current_pose
    
    for i in range(max_iterations):
        if np.linalg.norm(error) < tolerance:
            break
            
        # Calculate Jacobian
        J = jacobian(robot_config, current_joints, len(current_joints)-1)
        
        # Calculate pose error
        pose_error = np.zeros(6)
        pose_error[:3] = target_pose[:3, 3] - current_pose[:3, 3]  # Position error
        # For orientation, we'd calculate the rotation error
        pose_error[3:] = rotation_error(target_pose[:3, :3], current_pose[:3, :3])
        
        # Update joint angles
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ pose_error
        current_joints += dq * 0.1  # Learning rate
        
        # Recalculate current pose
        current_pose = forward_kinematics(current_joints, robot_config)
        error = target_pose - current_pose
    
    return current_joints

def rotation_error(R1, R2):
    """
    Calculate rotation error between two rotation matrices
    """
    R_error = R2.T @ R1
    angle_axis = rotation_matrix_to_angle_axis(R_error)
    return angle_axis

def rotation_matrix_to_angle_axis(R):
    """
    Convert rotation matrix to angle-axis representation
    """
    angle = math.acos((np.trace(R) - 1) / 2)
    if abs(angle) < 1e-6:
        return np.zeros(3)
    
    scale = angle / (2 * math.sin(angle))
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) * scale
    
    return axis
```

### Humanoid IK Challenges

Humanoid robots present special challenges for IK:

#### Redundancy
- Multiple solutions for the same end-effector pose
- Need for optimization criteria (e.g., minimize joint effort)
- Consideration of joint limits and obstacles

#### Multiple Constraints
- Maintaining balance while reaching
- Coordinating both arms for bimanual tasks
- Avoiding self-collisions

#### Real-time Requirements
- Fast computation for dynamic movements
- Smooth transitions between poses
- Handling of singularities

## Dynamics of Humanoid Robots

Dynamics deals with the forces and torques required to produce motion. For humanoid robots, dynamics is crucial for:
- Balance control during walking
- Smooth and energy-efficient movements
- Interaction with the environment
- Safety during operation

### Equation of Motion

The equation of motion for a humanoid robot is given by the Euler-Lagrange equation:

```
M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = τ
```

Where:
- `M(q)` is the mass matrix
- `C(q, q_dot)` is the Coriolis and centrifugal matrix
- `g(q)` is the gravity vector
- `q` is the joint position vector
- `q_dot` is the joint velocity vector
- `q_ddot` is the joint acceleration vector
- `τ` is the joint torque vector

### Inverse Dynamics

Inverse dynamics calculates the required joint torques for a given motion:

```python
def inverse_dynamics(trajectory, robot_params):
    """
    Calculate required joint torques for a given trajectory
    """
    n_points = len(trajectory['positions'])
    n_joints = len(trajectory['positions'][0])
    
    torques = np.zeros((n_points, n_joints))
    
    for i in range(n_points):
        q = trajectory['positions'][i]
        q_dot = trajectory['velocities'][i]
        q_ddot = trajectory['accelerations'][i]
        
        # Calculate mass matrix
        M = calculate_mass_matrix(q, robot_params)
        
        # Calculate Coriolis and centrifugal terms
        C = calculate_coriolis_matrix(q, q_dot, robot_params)
        
        # Calculate gravity terms
        g = calculate_gravity_vector(q, robot_params)
        
        # Calculate required torques
        torques[i] = M @ q_ddot + C @ q_dot + g
    
    return torques

def calculate_mass_matrix(q, robot_params):
    """
    Calculate the mass matrix using the composite rigid body algorithm
    """
    # This is a simplified implementation
    # In practice, this would use the robot's kinematic structure
    # and link properties to build the mass matrix
    n = len(q)
    M = np.zeros((n, n))
    
    # For each joint, calculate its contribution to the mass matrix
    for i in range(n):
        # Simplified calculation - in reality, this involves
        # complex calculations with link masses, inertias, and kinematics
        M[i, i] = robot_params['link_masses'][i]
    
    return M

def calculate_coriolis_matrix(q, q_dot, robot_params):
    """
    Calculate the Coriolis and centrifugal matrix
    """
    n = len(q)
    C = np.zeros((n, n))
    
    # Simplified calculation
    # In practice, this requires the partial derivatives of the mass matrix
    for i in range(n):
        for j in range(n):
            C[i, j] = robot_params['coriolis_coefficients'][i][j] * q_dot[j]
    
    return C

def calculate_gravity_vector(q, robot_params):
    """
    Calculate the gravity vector
    """
    n = len(q)
    g = np.zeros(n)
    
    # Calculate gravity effects for each joint
    for i in range(n):
        g[i] = robot_params['link_masses'][i] * 9.81 * math.sin(q[i])
    
    return g
```

### Center of Mass (CoM) and Zero Moment Point (ZMP)

For balance control, two important concepts are:

#### Center of Mass
The weighted average position of all mass in the robot:

```
CoM = Σ(m_i * r_i) / Σ(m_i)
```

#### Zero Moment Point
A point where the moment of the ground reaction force is zero:

```
ZMP_x = (Σ(m_i * g * x_i) - Σ(F_ix * z_i)) / Σ(m_i * g - F_iz)
```

### Balance Control Example

```python
class BalanceController:
    def __init__(self, robot_mass, gravity=9.81):
        self.robot_mass = robot_mass
        self.gravity = gravity
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_z = 0.0
        self.zmp_x = 0.0
        self.zmp_y = 0.0
        
        # PID controller parameters for balance
        self.kp = 10.0
        self.ki = 1.0
        self.kd = 1.0
        self.integral_error = 0.0
        self.previous_error = 0.0
    
    def update_com(self, com_position):
        """Update current center of mass position"""
        self.com_x, self.com_y, self.com_z = com_position
    
    def calculate_zmp(self, ground_forces, ground_moments):
        """Calculate Zero Moment Point"""
        # ZMP calculation based on ground reaction forces and moments
        fz_total = ground_forces[2]  # Total vertical force
        mx, my, _ = ground_moments   # Moments around x and y
        
        if abs(fz_total) > 1e-6:  # Avoid division by zero
            self.zmp_x = self.com_x - mx / fz_total
            self.zmp_y = self.com_y - my / fz_total
        else:
            self.zmp_x = self.com_x
            self.zmp_y = self.com_y
    
    def balance_control(self, desired_zmp, current_zmp):
        """Calculate corrective torques for balance"""
        # Calculate error
        error = desired_zmp - current_zmp
        
        # PID control
        self.integral_error += error * 0.01  # Assuming 100Hz control
        derivative_error = (error - self.previous_error) / 0.01
        
        control_output = (self.kp * error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
        
        self.previous_error = error
        
        return control_output
    
    def compute_stabilizing_motions(self, desired_com_position):
        """Compute motions to stabilize the robot"""
        # This would involve computing appropriate joint motions
        # to move the CoM to a stable position
        com_error = desired_com_position - np.array([self.com_x, self.com_y, self.com_z])
        
        # Simplified response - in reality, this would involve
        # complex whole-body control algorithms
        joint_corrections = np.zeros(28)  # Example for 28 DOF humanoid
        
        return joint_corrections
```

## Humanoid-Specific Kinematic Challenges

### Whole-Body Kinematics

Humanoid robots require coordination of multiple kinematic chains:

```python
class WholeBodyKinematics:
    def __init__(self):
        self.kinematics = HumanoidKinematics()
        self.balance_controller = BalanceController(robot_mass=50.0)
    
    def compute_reaching_motion(self, target_position, support_foot="left"):
        """
        Compute whole-body motion for reaching while maintaining balance
        """
        # Calculate required hand position
        hand_joints = self.inverse_kinematics_arm(target_position)
        
        # Calculate required CoM position to maintain balance
        required_com = self.calculate_balance_com(target_position, support_foot)
        
        # Coordinate leg movements to achieve required CoM
        leg_joints = self.inverse_kinematics_legs(required_com, support_foot)
        
        # Coordinate head/neck to look at target
        head_joints = self.inverse_kinematics_head(target_position)
        
        # Combine all joint angles
        full_body_joints = np.concatenate([head_joints, hand_joints, leg_joints])
        
        return full_body_joints
    
    def inverse_kinematics_arm(self, target_position):
        """Compute arm IK solution"""
        # Implementation would use the IK methods discussed earlier
        pass
    
    def inverse_kinematics_legs(self, target_com, support_foot):
        """Compute leg IK to achieve target CoM"""
        # Implementation would consider balance constraints
        pass
    
    def inverse_kinematics_head(self, target_position):
        """Compute head/neck IK to look at target"""
        # Implementation would point camera toward target
        pass
    
    def calculate_balance_com(self, reaching_position, support_foot):
        """Calculate CoM position for stable reaching"""
        # Calculate CoM that maintains ZMP within support polygon
        support_polygon = self.calculate_support_polygon(support_foot)
        
        # Find CoM position that keeps ZMP in safe region
        safe_com = self.find_safe_com_position(support_polygon, reaching_position)
        
        return safe_com
    
    def calculate_support_polygon(self, support_foot):
        """Calculate the support polygon based on contact points"""
        # For single support: polygon around the supporting foot
        # For double support: polygon connecting both feet
        pass
    
    def find_safe_com_position(self, support_polygon, reaching_position):
        """Find a safe CoM position given reaching task"""
        # Optimization to find CoM that allows reaching
        # while maintaining balance
        pass
```

## Practical Exercise: Implementing Simple Humanoid IK

Create a simplified humanoid robot and implement basic inverse kinematics:

1. **Define a simple humanoid model** with basic DH parameters
2. **Implement forward kinematics** for arms and legs
3. **Implement inverse kinematics** using Jacobian transpose
4. **Test with simple reaching motions**
5. **Add basic balance constraints**

## Summary

Humanoid robot kinematics and dynamics form the mathematical foundation for controlling these complex systems. Forward kinematics allows us to predict end-effector positions from joint angles, while inverse kinematics enables us to calculate the required joint angles for desired positions. Dynamics provides the understanding of forces and torques needed for motion.

The challenges of humanoid kinematics include managing redundancy, coordinating multiple limbs, and maintaining balance. These require sophisticated mathematical tools and computational approaches to solve effectively.

Understanding these principles is essential for developing effective control strategies for humanoid robots.

## Next Steps

In the next lesson, we'll explore bipedal locomotion and balance control, which builds on the kinematic and dynamic principles covered here.