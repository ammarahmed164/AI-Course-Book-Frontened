---
sidebar_position: 5
title: "Manipulation and Grasping with Humanoid Hands"
---

# Manipulation and Grasping with Humanoid Hands

This lesson covers the principles and techniques for manipulation and grasping using humanoid robot hands.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the challenges of dexterous manipulation in humanoid robots
- Analyze different hand designs and their capabilities
- Implement basic grasping strategies
- Plan manipulation trajectories
- Design control systems for manipulation tasks
- Understand tactile sensing and its role in grasping

## Introduction to Humanoid Manipulation

Humanoid manipulation refers to the ability of humanoid robots to interact with objects in their environment using anthropomorphic hands and arms. This capability is essential for humanoid robots to perform tasks in human environments.

### Challenges of Humanoid Manipulation

1. **Dexterity**: Achieving human-like manipulation capabilities
2. **Grasp Planning**: Determining how to grasp objects of various shapes
3. **Force Control**: Applying appropriate forces without damaging objects
4. **Tactile Sensing**: Understanding object properties through touch
5. **Real-time Control**: Responding quickly to changes in the environment
6. **Multi-finger Coordination**: Coordinating multiple fingers effectively

### Applications of Humanoid Manipulation

- **Household tasks**: Cooking, cleaning, organizing
- **Industrial tasks**: Assembly, quality inspection
- **Healthcare**: Assisting patients, handling medical instruments
- **Service industries**: Serving food, handling customer items

## Humanoid Hand Design

### Degrees of Freedom and Design Considerations

Human hands have approximately 27 degrees of freedom, though most humanoid hands have fewer:

#### Minimal Design (3-4 DOF)
- **Thumb**: 1-2 DOF
- **Fingers**: 1 DOF each (flexion)
- **Advantages**: Simple, robust, cost-effective
- **Disadvantages**: Limited dexterity

#### Moderate Design (8-12 DOF)
- **Thumb**: 2-3 DOF
- **Fingers**: 2-3 DOF each
- **Advantages**: Good balance of dexterity and complexity
- **Disadvantages**: More complex control

#### Full Design (15+ DOF)
- **Thumb**: 4-5 DOF
- **Fingers**: 3-4 DOF each
- **Advantages**: High dexterity, human-like capabilities
- **Disadvantages**: Complex, expensive, difficult to control

### Common Hand Designs

#### Underactuated Hands
- **Principle**: Fewer actuators than DOF
- **Mechanism**: Tendon-based or linkage systems
- **Advantages**: Self-adaptive, robust, lightweight
- **Disadvantages**: Limited independent control

```python
class UnderactuatedHand:
    def __init__(self, n_fingers=5):
        self.n_fingers = n_fingers
        self.actuators = n_fingers  # One actuator per finger
        self.dof_per_finger = 3     # Typical for underactuated designs
        
        # Finger positions (simplified)
        self.finger_angles = [0.0] * n_fingers
        self.finger_positions = [0.0] * n_fingers
    
    def grasp_object(self, object_shape):
        """
        Adapt grasp based on object shape
        """
        # Underactuated hands naturally adapt to object shape
        grasp_configuration = self.calculate_adaptive_grasp(object_shape)
        return grasp_configuration
    
    def calculate_adaptive_grasp(self, object_shape):
        """
        Calculate grasp based on object shape
        """
        # Simplified model - in reality would use more complex algorithms
        grasp_config = []
        
        for i in range(self.n_fingers):
            # Calculate appropriate finger position based on object shape
            finger_pos = self.calculate_finger_position(i, object_shape)
            grasp_config.append(finger_pos)
        
        return grasp_config
    
    def calculate_finger_position(self, finger_idx, object_shape):
        """
        Calculate position for a specific finger
        """
        # Simplified calculation
        # In reality, would use geometric analysis or learning-based methods
        return 0.5  # Mid-range position
```

#### Fully Actuated Hands
- **Principle**: Independent actuation for each DOF
- **Mechanism**: Motor for each joint
- **Advantages**: Full control, precise manipulation
- **Disadvantages**: Complex, heavy, expensive

```python
class FullyActuatedHand:
    def __init__(self, n_fingers=5, dof_per_finger=3):
        self.n_fingers = n_fingers
        self.dof_per_finger = dof_per_finger
        self.total_dof = n_fingers * dof_per_finger
        
        # Joint positions
        self.joint_positions = [0.0] * self.total_dof
        self.joint_velocities = [0.0] * self.total_dof
        self.joint_torques = [0.0] * self.total_dof
    
    def move_to_pregrasp(self, object_pose):
        """
        Move hand to pregrasp position
        """
        # Calculate inverse kinematics to move fingertips to pregrasp positions
        pregrasp_config = self.calculate_pregrasp_configuration(object_pose)
        return self.move_to_configuration(pregrasp_config)
    
    def grasp_object(self, object_pose, grasp_type="cylindrical"):
        """
        Execute grasp based on object properties
        """
        # Plan grasp trajectory
        grasp_config = self.plan_grasp_trajectory(object_pose, grasp_type)
        
        # Execute grasp with force control
        success = self.execute_grasp_with_force_control(grasp_config)
        return success
    
    def plan_grasp_trajectory(self, object_pose, grasp_type):
        """
        Plan trajectory for executing grasp
        """
        # This would involve complex grasp planning algorithms
        # For now, return a simple grasp configuration
        
        if grasp_type == "cylindrical":
            # Cylindrical grasp - wrap around object
            return self.plan_cylindrical_grasp(object_pose)
        elif grasp_type == "lateral":
            # Lateral grasp - pinch between thumb and finger
            return self.plan_lateral_grasp(object_pose)
        elif grasp_type == "tip":
            # Tip grasp - precision pinch
            return self.plan_tip_grasp(object_pose)
        else:
            # Default grasp
            return self.plan_power_grasp(object_pose)
    
    def plan_cylindrical_grasp(self, object_pose):
        """
        Plan cylindrical grasp for cylindrical objects
        """
        # Calculate finger positions around the object
        grasp_config = [0.0] * self.total_dof
        
        # Move fingers to wrap around object
        for i in range(self.n_fingers):
            finger_start_idx = i * self.dof_per_finger
            # Close fingers around object
            for j in range(self.dof_per_finger):
                grasp_config[finger_start_idx + j] = 0.8  # Close fingers
        
        # Position thumb opposable to other fingers
        thumb_start_idx = 4 * self.dof_per_finger  # Assuming thumb is last
        grasp_config[thumb_start_idx] = 0.6  # Move thumb to oppose
        grasp_config[thumb_start_idx + 1] = 0.3  # Adjust thumb angle
        grasp_config[thumb_start_idx + 2] = 0.7  # Final thumb position
        
        return grasp_config
```

### Hand Mechanisms

#### Tendon-Driven Systems
- **Mechanism**: Cables or tendons connect actuators to joints
- **Advantages**: Compact, lightweight, backdrivable
- **Disadvantages**: Nonlinear transmission, wear over time

#### Linkage Systems
- **Mechanism**: Mechanical linkages connect actuators to joints
- **Advantages**: Precise control, predictable behavior
- **Disadvantages**: Bulkier, more complex

#### Pneumatic Systems
- **Mechanism**: Air pressure actuates joints
- **Advantages**: Compliant, safe interaction
- **Disadvantages**: Requires air supply, less precise

## Grasp Planning

### Grasp Types

#### Power Grasps
- **Cylindrical grasp**: Wrap fingers around object
- **Spherical grasp**: Enclose object with curved fingers
- **Hook grasp**: Use fingertips to hook object

#### Precision Grasps
- **Lateral grasp**: Pinch between thumb and side of index finger
- **Tip grasp**: Pinch between fingertips
- **Tripod grasp**: Use thumb, index, and middle fingers

### Grasp Stability Analysis

```python
class GraspAnalyzer:
    def __init__(self):
        self.friction_coefficient = 0.8  # Typical for rubber-like materials
    
    def analyze_grasp_stability(self, contact_points, object_mass, grasp_type):
        """
        Analyze if a grasp is stable
        """
        # Calculate forces at contact points
        gravity_force = object_mass * 9.81
        
        # Check if grasp can resist gravity
        friction_force = self.calculate_total_friction_force(contact_points)
        
        # Stability condition: friction force > gravitational force
        if friction_force > gravity_force:
            return True, "Stable grasp"
        else:
            required_force = gravity_force - friction_force
            return False, f"Unstable grasp, requires {required_force:.2f}N more force"
    
    def calculate_total_friction_force(self, contact_points):
        """
        Calculate total friction force available at contact points
        """
        total_friction = 0.0
        
        for contact in contact_points:
            normal_force = contact['normal_force']
            friction_force = self.friction_coefficient * normal_force
            total_friction += friction_force
        
        return total_friction
    
    def find_grasp_points(self, object_mesh):
        """
        Find suitable grasp points on an object
        """
        # This would implement geometric algorithms to find grasp points
        # For now, return a simple set of points
        
        grasp_points = []
        
        # Find surface points suitable for grasping
        for face in object_mesh.faces:
            # Calculate face properties
            center = self.calculate_face_center(face)
            normal = self.calculate_face_normal(face)
            
            # Check if face is suitable for grasping
            if self.is_suitable_for_grasping(face, normal):
                grasp_points.append({
                    'position': center,
                    'normal': normal,
                    'surface_area': self.calculate_surface_area(face)
                })
        
        return grasp_points
    
    def calculate_face_center(self, face):
        """
        Calculate center of a face
        """
        # Simplified calculation
        return [sum(coord) / len(face) for coord in zip(*face)]
    
    def calculate_face_normal(self, face):
        """
        Calculate normal vector of a face
        """
        # Simplified calculation
        # In reality, would use cross product of edges
        return [0, 0, 1]  # Assume upward normal
    
    def is_suitable_for_grasping(self, face, normal):
        """
        Check if a surface is suitable for grasping
        """
        # Check if surface is flat enough
        surface_area = self.calculate_surface_area(face)
        if surface_area < 0.001:  # 1 cm² minimum
            return False
        
        # Check if normal is favorable for grasp
        # For power grasp, normal should be roughly perpendicular to gravity
        gravity_vector = [0, 0, -1]
        dot_product = sum(a*b for a, b in zip(normal, gravity_vector))
        
        # If normal is roughly horizontal (perpendicular to gravity), good for power grasp
        if abs(dot_product) < 0.5:
            return True
        
        return False
    
    def calculate_surface_area(self, face):
        """
        Calculate surface area of a face
        """
        # Simplified calculation
        return 0.01  # 1 cm² default
```

### Grasp Planning Algorithm

```python
class GraspPlanner:
    def __init__(self):
        self.analyzer = GraspAnalyzer()
        self.kinematics = HumanoidHandKinematics()
    
    def plan_grasp(self, object_info):
        """
        Plan optimal grasp for an object
        """
        # Get object properties
        object_mesh = object_info['mesh']
        object_mass = object_info['mass']
        object_com = object_info['center_of_mass']
        
        # Find potential grasp points
        grasp_points = self.analyzer.find_grasp_points(object_mesh)
        
        # Evaluate grasp configurations
        best_grasp = None
        best_score = float('-inf')
        
        for grasp_config in self.generate_grasp_configurations(grasp_points):
            score = self.evaluate_grasp_configuration(
                grasp_config, object_mass, object_com
            )
            
            if score > best_score:
                best_score = score
                best_grasp = grasp_config
        
        return best_grasp
    
    def generate_grasp_configurations(self, grasp_points):
        """
        Generate possible grasp configurations
        """
        configurations = []
        
        # Generate different grasp types
        for grasp_type in ["power", "precision", "cylindrical"]:
            config = self.generate_grasp_for_type(grasp_points, grasp_type)
            configurations.append({
                'type': grasp_type,
                'configuration': config
            })
        
        return configurations
    
    def generate_grasp_for_type(self, grasp_points, grasp_type):
        """
        Generate grasp configuration for specific grasp type
        """
        if grasp_type == "power":
            return self.generate_power_grasp(grasp_points)
        elif grasp_type == "precision":
            return self.generate_precision_grasp(grasp_points)
        elif grasp_type == "cylindrical":
            return self.generate_cylindrical_grasp(grasp_points)
        else:
            return self.generate_power_grasp(grasp_points)  # Default
    
    def generate_power_grasp(self, grasp_points):
        """
        Generate power grasp configuration
        """
        # Find points that allow wrapping around object
        wrap_points = [p for p in grasp_points if self.is_wrap_point(p)]
        
        if len(wrap_points) >= 3:
            # Use 3 points: 2 for fingers, 1 for thumb opposition
            return {
                'finger_points': wrap_points[:2],
                'thumb_point': wrap_points[2],
                'joint_angles': self.calculate_joint_angles_for_wrap(wrap_points[:3])
            }
        
        return None
    
    def is_wrap_point(self, point):
        """
        Check if point is suitable for wrapping grasp
        """
        # Check if surface is curved appropriately
        return point['surface_area'] > 0.002  # More than 2 cm²
    
    def calculate_joint_angles_for_wrap(self, points):
        """
        Calculate joint angles to achieve wrap grasp
        """
        # This would solve inverse kinematics for the hand
        # to position fingertips at the specified points
        return [0.5] * 15  # Simplified: mid-range joint angles
```

## Tactile Sensing and Feedback

### Importance of Tactile Sensing

Tactile sensing is crucial for:
- Detecting object contact
- Estimating object properties (texture, hardness, weight)
- Adjusting grasp force appropriately
- Detecting slip and preventing object drops
- Enabling delicate manipulation

### Tactile Sensor Types

#### Force/Torque Sensors
- **Location**: Often at fingertips or in joints
- **Capabilities**: Measure normal and tangential forces
- **Resolution**: High precision for force control

#### Tactile Arrays
- **Location**: On fingertips and palm
- **Capabilities**: Spatial distribution of contact forces
- **Resolution**: High spatial resolution

#### Slip Sensors
- **Location**: On fingertips
- **Capabilities**: Detect early signs of object slip
- **Response**: Fast reaction to prevent drops

### Tactile Control Implementation

```python
class TactileController:
    def __init__(self):
        # Tactile sensor data
        self.fingertip_sensors = {
            'thumb': {'force': [0, 0, 0], 'slip': False},
            'index': {'force': [0, 0, 0], 'slip': False},
            'middle': {'force': [0, 0, 0], 'slip': False},
            'ring': {'force': [0, 0, 0], 'slip': False},
            'pinky': {'force': [0, 0, 0], 'slip': False}
        }
        
        # Control parameters
        self.target_force = 5.0  # Newtons
        self.slip_threshold = 0.1  # Normalized slip detection
        self.force_tolerance = 0.5  # Acceptable force deviation
    
    def update_tactile_sensors(self, sensor_data):
        """
        Update tactile sensor readings
        """
        for finger, data in sensor_data.items():
            self.fingertip_sensors[finger]['force'] = data['force']
            self.fingertip_sensors[finger]['slip'] = data['slip']
    
    def adjust_grasp_force(self):
        """
        Adjust grasp force based on tactile feedback
        """
        adjustments = {}
        
        for finger, sensor_data in self.fingertip_sensors.items():
            current_force = np.linalg.norm(sensor_data['force'])
            
            # Check for slip
            if sensor_data['slip']:
                # Increase force rapidly to prevent drop
                adjustments[finger] = min(1.0, current_force + 2.0)
            elif current_force < self.target_force - self.force_tolerance:
                # Increase force gradually
                adjustments[finger] = min(1.0, current_force + 0.1)
            elif current_force > self.target_force + self.force_tolerance:
                # Decrease force gradually
                adjustments[finger] = max(0.1, current_force - 0.1)
            else:
                # Force is appropriate
                adjustments[finger] = current_force
        
        return adjustments
    
    def detect_object_properties(self):
        """
        Infer object properties from tactile data
        """
        properties = {
            'stiffness': self.estimate_stiffness(),
            'texture': self.estimate_texture(),
            'weight': self.estimate_weight(),
            'shape': self.estimate_shape()
        }
        
        return properties
    
    def estimate_stiffness(self):
        """
        Estimate object stiffness from force-displacement data
        """
        # Simplified stiffness estimation
        # In reality, would use more sophisticated models
        total_force = 0
        for finger_data in self.fingertip_sensors.values():
            total_force += np.linalg.norm(finger_data['force'])
        
        # Stiffness is roughly proportional to force
        return min(1.0, total_force / 50.0)  # Normalize to 0-1
    
    def estimate_texture(self):
        """
        Estimate object texture from tactile patterns
        """
        # Simplified texture estimation
        # In reality, would use machine learning on tactile data
        force_variability = 0
        for finger_data in self.fingertip_sensors.values():
            force_var = np.var(finger_data['force'])
            force_variability += force_var
        
        if force_variability < 0.1:
            return "smooth"
        elif force_variability < 0.5:
            return "medium"
        else:
            return "rough"
    
    def estimate_weight(self):
        """
        Estimate object weight from grasp force
        """
        # Estimate weight based on required grasp force
        total_grasp_force = sum(
            np.linalg.norm(data['force']) 
            for data in self.fingertip_sensors.values()
        )
        
        # Rough estimate: weight ≈ grasp_force / 4 (assuming 4 contact points)
        estimated_weight = total_grasp_force / 4.0
        return estimated_weight
    
    def estimate_shape(self):
        """
        Estimate object shape from contact pattern
        """
        # Determine contact points and their spatial relationship
        contact_points = []
        for finger, data in self.fingertip_sensors.items():
            if np.linalg.norm(data['force']) > 0.1:  # Significant contact
                contact_points.append(finger)
        
        if len(contact_points) == 2:
            return "cylindrical"  # Between two fingers
        elif len(contact_points) >= 3:
            return "irregular"  # Multiple contact points
        else:
            return "unknown"
```

## Manipulation Trajectory Planning

### Pregrasp to Grasp Motion

```python
class ManipulationPlanner:
    def __init__(self):
        self.kinematics = HumanoidHandKinematics()
        self.tactile_controller = TactileController()
        self.grasp_planner = GraspPlanner()
    
    def plan_manipulation_task(self, object_pose, target_pose, task_type="pick_and_place"):
        """
        Plan complete manipulation task
        """
        if task_type == "pick_and_place":
            return self.plan_pick_and_place(object_pose, target_pose)
        elif task_type == "reposition":
            return self.plan_reposition(object_pose, target_pose)
        else:
            return self.plan_pick_and_place(object_pose, target_pose)
    
    def plan_pick_and_place(self, object_pose, target_pose):
        """
        Plan pick and place manipulation
        """
        trajectory = []
        
        # 1. Move to pregrasp position
        pregrasp_poses = self.calculate_pregrasp_poses(object_pose)
        trajectory.extend(self.generate_approach_trajectory(pregrasp_poses))
        
        # 2. Execute grasp
        grasp_configuration = self.grasp_planner.plan_grasp(object_pose)
        trajectory.append(("grasp", grasp_configuration))
        
        # 3. Lift object
        lift_trajectory = self.generate_lift_trajectory()
        trajectory.extend(lift_trajectory)
        
        # 4. Move to target location
        transport_trajectory = self.generate_transport_trajectory(
            object_pose, target_pose
        )
        trajectory.extend(transport_trajectory)
        
        # 5. Place object
        place_trajectory = self.generate_place_trajectory(target_pose)
        trajectory.extend(place_trajectory)
        
        # 6. Release and retract
        trajectory.append(("release", {}))
        retract_trajectory = self.generate_retract_trajectory()
        trajectory.extend(retract_trajectory)
        
        return trajectory
    
    def calculate_pregrasp_poses(self, object_pose):
        """
        Calculate pregrasp poses for fingers
        """
        # Calculate approach positions for each finger
        # This would involve inverse kinematics to position
        # fingertips at appropriate locations around the object
        
        pregrasp_poses = {}
        
        # Example: Position fingertips 2cm away from grasp points
        grasp_config = self.grasp_planner.plan_grasp(object_pose)
        
        for i, finger_name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
            if i < len(grasp_config['finger_points']):
                grasp_point = grasp_config['finger_points'][i % len(grasp_config['finger_points'])]
                
                # Calculate approach direction (typically opposite to surface normal)
                approach_dir = np.array([0, 0, -1])  # From above
                pregrasp_pos = grasp_point['position'] - 0.02 * approach_dir  # 2cm away
                
                pregrasp_poses[finger_name] = {
                    'position': pregrasp_pos,
                    'orientation': grasp_point['normal']
                }
        
        return pregrasp_poses
    
    def generate_approach_trajectory(self, pregrasp_poses):
        """
        Generate trajectory to move from current position to pregrasp
        """
        trajectory = []
        
        # Move each finger to its pregrasp position
        for finger_name, pose in pregrasp_poses.items():
            # Calculate joint trajectory using inverse kinematics
            joint_trajectory = self.kinematics.inverse_kinematics(
                pose['position'], pose['orientation']
            )
            
            trajectory.append(("move_finger", {
                'finger': finger_name,
                'trajectory': joint_trajectory
            }))
        
        return trajectory
    
    def generate_lift_trajectory(self):
        """
        Generate trajectory to lift object after grasping
        """
        # Move entire hand/arm upward
        lift_trajectory = [
            ("lift", {'direction': [0, 0, 0.05], 'duration': 1.0})  # Lift 5cm in 1s
        ]
        
        return lift_trajectory
    
    def generate_transport_trajectory(self, start_pose, end_pose):
        """
        Generate trajectory to move object from start to end
        """
        # Plan path that avoids obstacles and maintains grasp
        transport_trajectory = []
        
        # Simplified: straight line movement
        # In reality, would use path planning algorithms
        midpoint = (start_pose['position'] + end_pose['position']) / 2
        midpoint[2] += 0.1  # Lift higher to avoid obstacles
        
        transport_trajectory.append(("move_object", {
            'path': [start_pose['position'], midpoint, end_pose['position']],
            'orientation': start_pose['orientation']
        }))
        
        return transport_trajectory
    
    def generate_place_trajectory(self, target_pose):
        """
        Generate trajectory to place object at target
        """
        place_trajectory = []
        
        # Move to placement position
        place_trajectory.append(("move_to_place", {
            'position': target_pose['position'],
            'orientation': target_pose['orientation']
        }))
        
        # Lower object
        place_trajectory.append(("lower_object", {
            'distance': 0.02,  # Lower 2cm
            'duration': 0.5
        }))
        
        return place_trajectory
    
    def generate_retract_trajectory(self):
        """
        Generate trajectory to retract after placing
        """
        retract_trajectory = [
            ("retract", {'direction': [0, 0, 0.05], 'duration': 1.0})  # Retract upward
        ]
        
        return retract_trajectory
```

## Force Control for Manipulation

### Impedance Control

Impedance control allows the robot to behave like a spring-damper system, providing compliant behavior during manipulation:

```python
class ImpedanceController:
    def __init__(self, stiffness=1000, damping=20, mass=1):
        self.stiffness = stiffness  # Spring constant
        self.damping = damping      # Damping coefficient
        self.mass = mass           # Equivalent mass
        
        # Desired position (equilibrium point)
        self.desired_pos = np.zeros(3)
        self.desired_vel = np.zeros(3)
        
        # Current state
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
    
    def update(self, current_pos, current_vel, dt):
        """
        Update impedance controller
        """
        self.current_pos = current_pos
        self.current_vel = current_vel
        
        # Calculate position and velocity errors
        pos_error = self.desired_pos - self.current_pos
        vel_error = self.desired_vel - self.current_vel
        
        # Calculate forces based on impedance model
        spring_force = self.stiffness * pos_error
        damping_force = self.damping * vel_error
        
        # Total force
        total_force = spring_force + damping_force
        
        # Convert to joint torques (simplified)
        joint_torques = self.cartesian_to_joint_torques(total_force)
        
        return joint_torques
    
    def cartesian_to_joint_torques(self, force):
        """
        Convert Cartesian forces to joint torques using Jacobian
        """
        # Simplified conversion
        # In reality, would use full Jacobian transpose method
        return np.array([force[0]*0.1, force[1]*0.1, force[2]*0.1] + [0.0]*12)  # Example for 15 DOF hand
    
    def set_desired_pose(self, pos, vel=None):
        """
        Set desired pose for impedance control
        """
        self.desired_pos = np.array(pos)
        if vel is not None:
            self.desired_vel = np.array(vel)
        else:
            self.desired_vel = np.zeros(3)

class ForceController:
    def __init__(self, target_force=5.0):
        self.target_force = target_force
        self.current_force = 0.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # PID gains
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05
    
    def update(self, measured_force, dt):
        """
        Update force controller
        """
        self.current_force = measured_force
        error = self.target_force - measured_force
        
        # PID control
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0
        
        output = (self.kp * error + 
                 self.ki * self.integral_error + 
                 self.kd * derivative_error)
        
        self.previous_error = error
        
        return output
```

## Practical Exercise: Implementing a Simple Grasping Controller

Create a complete grasping system:

1. **Implement a hand kinematics model**
2. **Create a grasp planner**
3. **Add tactile sensing and force control**
4. **Implement a manipulation planner**
5. **Test with simple objects**

### Complete Grasping System Example

```python
class CompleteGraspingSystem:
    def __init__(self):
        self.hand_kinematics = HumanoidHandKinematics()
        self.grasp_planner = GraspPlanner()
        self.tactile_controller = TactileController()
        self.manipulation_planner = ManipulationPlanner()
        self.force_controller = ForceController()
        
        # System state
        self.is_object_grasped = False
        self.current_object = None
    
    def grasp_object(self, object_info):
        """
        Execute complete grasp of an object
        """
        print("Starting grasp sequence...")
        
        # 1. Plan grasp
        print("Planning grasp...")
        grasp_config = self.grasp_planner.plan_grasp(object_info)
        
        if grasp_config is None:
            print("Could not find stable grasp configuration")
            return False
        
        # 2. Move to pregrasp position
        print("Moving to pregrasp position...")
        pregrasp_trajectory = self.manipulation_planner.calculate_pregrasp_poses(object_info)
        self.execute_trajectory(
            self.manipulation_planner.generate_approach_trajectory(pregrasp_trajectory)
        )
        
        # 3. Execute grasp
        print("Executing grasp...")
        self.execute_grasp(grasp_config)
        
        # 4. Verify grasp stability
        print("Verifying grasp stability...")
        if self.verify_grasp_stability():
            self.is_object_grasped = True
            self.current_object = object_info
            print("Grasp successful!")
            return True
        else:
            print("Grasp verification failed")
            self.release_object()
            return False
    
    def execute_grasp(self, grasp_config):
        """
        Execute the grasp configuration
        """
        # Move fingers to grasp positions
        for finger, angle in enumerate(grasp_config['joint_angles']):
            self.move_finger_to_position(finger, angle)
        
        # Apply appropriate grasp force
        self.apply_grasp_force()
    
    def verify_grasp_stability(self):
        """
        Verify that the grasp is stable
        """
        # Get tactile feedback
        tactile_data = self.get_tactile_feedback()
        self.tactile_controller.update_tactile_sensors(tactile_data)
        
        # Check if forces are appropriate
        object_weight = self.current_object['mass'] * 9.81
        total_grip_force = self.calculate_total_grip_force()
        
        # Stability condition: grip force should be sufficient but not excessive
        if total_grip_force > object_weight * 1.5 and total_grip_force < object_weight * 5:
            return True
        else:
            return False
    
    def execute_trajectory(self, trajectory):
        """
        Execute a manipulation trajectory
        """
        for step in trajectory:
            command, params = step
            if command == "move_finger":
                self.move_finger_to_position(
                    params['finger'], 
                    params['trajectory']
                )
            elif command == "grasp":
                self.execute_grasp(params)
            # Add other command types as needed
    
    def move_finger_to_position(self, finger_id, target_position):
        """
        Move a finger to a target position
        """
        # Implementation would use inverse kinematics
        # and joint control to move the finger
        pass
    
    def apply_grasp_force(self):
        """
        Apply appropriate grasp force
        """
        # Use force control to apply the right amount of force
        target_force = self.calculate_appropriate_grasp_force()
        self.force_controller.target_force = target_force
    
    def calculate_appropriate_grasp_force(self):
        """
        Calculate appropriate grasp force based on object properties
        """
        if self.current_object:
            object_weight = self.current_object['mass'] * 9.81
            # Use safety factor
            return object_weight * 2.0
        else:
            return 5.0  # Default force
    
    def calculate_total_grip_force(self):
        """
        Calculate total grip force from tactile sensors
        """
        tactile_data = self.get_tactile_feedback()
        total_force = 0
        
        for finger_data in tactile_data.values():
            force_magnitude = np.linalg.norm(finger_data['force'])
            total_force += force_magnitude
        
        return total_force
    
    def get_tactile_feedback(self):
        """
        Get current tactile sensor readings
        """
        # In simulation, return synthetic data
        # In real robot, read from actual sensors
        return {
            'thumb': {'force': [1.0, 0.2, 2.0], 'slip': False},
            'index': {'force': [0.8, 0.1, 1.8], 'slip': False},
            'middle': {'force': [0.9, 0.15, 1.9], 'slip': False},
            'ring': {'force': [0.7, 0.1, 1.7], 'slip': False},
            'pinky': {'force': [0.5, 0.05, 1.5], 'slip': False}
        }
    
    def release_object(self):
        """
        Release the currently grasped object
        """
        # Open fingers to release object
        for finger in range(5):  # Assuming 5 fingers
            self.move_finger_to_position(finger, 0.0)  # Open position
        
        self.is_object_grasped = False
        self.current_object = None
        print("Object released")
```

## Summary

Humanoid manipulation and grasping is a complex field that combines mechanical design, control theory, and sensor feedback. Successful manipulation requires careful consideration of hand design, grasp planning, force control, and tactile feedback.

The challenges include achieving human-like dexterity, planning stable grasps for various objects, and controlling the many degrees of freedom involved. Modern approaches combine model-based planning with learning-based methods to achieve robust manipulation capabilities.

Understanding these principles is essential for developing humanoid robots that can effectively interact with their environment through manipulation tasks.

## Next Steps

In the next lesson, we'll explore natural human-robot interaction design, which builds on the manipulation capabilities to create intuitive interaction experiences.