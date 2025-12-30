---
sidebar_position: 11
title: "Sim-to-Real Transfer Techniques"
---

# Sim-to-Real Transfer Techniques

This lesson covers techniques for transferring robot control policies and behaviors from simulation to real-world robots.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the challenges of sim-to-real transfer
- Apply domain randomization techniques
- Implement system identification methods
- Evaluate sim-to-real transfer performance
- Implement safety measures for real-world deployment

## Introduction to Sim-to-Real Transfer

Sim-to-real transfer is the process of taking policies, controllers, or behaviors learned or developed in simulation and successfully deploying them on real robots. This is a critical challenge in robotics because simulations, despite their accuracy, never perfectly match the real world.

### The Reality Gap

The "reality gap" refers to the differences between simulation and the real world:

- **Physical properties**: Mass, friction, compliance
- **Sensor noise**: Different noise characteristics
- **Actuator dynamics**: Delays, backlash, nonlinearities
- **Environmental conditions**: Lighting, temperature, disturbances
- **Modeling inaccuracies**: Simplifications in simulation

### Why Sim-to-Real Transfer Matters

- **Cost reduction**: Training in simulation is cheaper than real-world trials
- **Safety**: Dangerous scenarios can be tested safely in simulation
- **Speed**: Simulation can run faster than real-time
- **Repeatability**: Same conditions can be recreated exactly
- **Scalability**: Multiple robots can be trained simultaneously

## Domain Randomization

Domain randomization is a technique that randomizes simulation parameters to improve the robustness of policies to real-world variations.

### Principles of Domain Randomization

The core idea is to train on a wide variety of simulated environments so the policy learns to adapt to variations:

```python
# domain_randomizer.py
import numpy as np

class DomainRandomizer:
    def __init__(self):
        # Define ranges for various parameters
        self.param_ranges = {
            'mass': [0.5, 1.5],  # ±50% mass variation
            'friction': [0.1, 10.0],  # Wide friction range
            'com_offset': [-0.05, 0.05],  # Center of mass offset
            'joint_damping': [0.01, 0.1],  # Joint damping variation
            'camera_noise': [0.0, 0.1],  # Camera noise level
            'lighting': [0.5, 2.0],  # Lighting intensity variation
        }
    
    def randomize_environment(self, env):
        """Apply randomization to environment parameters"""
        # Randomize physical properties
        self._randomize_masses(env)
        self._randomize_friction(env)
        self._randomize_com_offset(env)
        self._randomize_joint_damping(env)
        
        # Randomize sensor properties
        self._randomize_camera_noise(env)
        
        # Randomize environmental properties
        self._randomize_lighting(env)
        self._randomize_texture(env)
    
    def _randomize_masses(self, env):
        """Randomize robot masses"""
        for link_name in env.get_link_names():
            original_mass = env.get_original_mass(link_name)
            random_factor = np.random.uniform(*self.param_ranges['mass'])
            new_mass = original_mass * random_factor
            env.set_link_mass(link_name, new_mass)
    
    def _randomize_friction(self, env):
        """Randomize friction coefficients"""
        for link_name in env.get_link_names():
            random_friction = np.random.uniform(*self.param_ranges['friction'])
            env.set_link_friction(link_name, random_friction)
    
    def _randomize_com_offset(self, env):
        """Randomize center of mass offset"""
        for link_name in env.get_link_names():
            offset = np.random.uniform(
                low=self.param_ranges['com_offset'][0],
                high=self.param_ranges['com_offset'][1],
                size=3
            )
            env.set_link_com_offset(link_name, offset)
    
    def _randomize_joint_damping(self, env):
        """Randomize joint damping"""
        for joint_name in env.get_joint_names():
            random_damping = np.random.uniform(*self.param_ranges['joint_damping'])
            env.set_joint_damping(joint_name, random_damping)
    
    def _randomize_camera_noise(self, env):
        """Randomize camera sensor noise"""
        noise_level = np.random.uniform(*self.param_ranges['camera_noise'])
        env.set_camera_noise_level(noise_level)
    
    def _randomize_lighting(self, env):
        """Randomize lighting conditions"""
        intensity = np.random.uniform(*self.param_ranges['lighting'])
        env.set_lighting_intensity(intensity)
    
    def _randomize_texture(self, env):
        """Randomize surface textures"""
        # Apply random textures to objects
        for obj_name in env.get_object_names():
            texture = self._get_random_texture()
            env.set_object_texture(obj_name, texture)
    
    def _get_random_texture(self):
        """Generate or select a random texture"""
        # This could involve selecting from a database of textures
        # or procedurally generating textures
        return np.random.choice(['rough', 'smooth', 'bumpy', 'textured'])
```

### Adaptive Domain Randomization

Adaptive domain randomization adjusts the randomization based on performance:

```python
# adaptive_domain_randomizer.py
class AdaptiveDomainRandomizer(DomainRandomizer):
    def __init__(self):
        super().__init__()
        self.performance_history = []
        self.current_difficulty = 0.5  # Start at medium difficulty
    
    def update_difficulty(self, performance):
        """Update randomization difficulty based on performance"""
        self.performance_history.append(performance)
        
        # Calculate recent performance trend
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            previous_performance = np.mean(self.performance_history[-20:-10])
            
            # If performance is improving, increase difficulty
            if recent_performance > previous_performance:
                self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
            # If performance is degrading, decrease difficulty
            elif recent_performance < previous_performance:
                self.current_difficulty = max(0.1, self.current_difficulty - 0.05)
    
    def randomize_environment(self, env):
        """Apply randomization with adaptive difficulty"""
        # Adjust parameter ranges based on difficulty
        adjusted_ranges = self._adjust_ranges_by_difficulty()
        
        # Apply randomization with adjusted ranges
        self._randomize_with_ranges(env, adjusted_ranges)
    
    def _adjust_ranges_by_difficulty(self):
        """Adjust parameter ranges based on current difficulty"""
        adjusted_ranges = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            center = (min_val + max_val) / 2
            range_size = (max_val - min_val) * self.current_difficulty
            adjusted_ranges[param] = [
                center - range_size/2,
                center + range_size/2
            ]
        return adjusted_ranges
    
    def _randomize_with_ranges(self, env, ranges):
        """Apply randomization using specific ranges"""
        # Implementation similar to base class but using ranges parameter
        pass
```

## System Identification

System identification involves measuring real-world robot parameters to improve simulation accuracy.

### Parameter Estimation

```python
# system_identifier.py
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.sim_params = {}
        self.real_params = {}
    
    def collect_step_response_data(self, real_robot, joint_idx, input_torque, duration=2.0):
        """Collect step response data from real robot"""
        # Apply step input
        real_robot.apply_torque(joint_idx, input_torque)
        
        # Collect data over time
        dt = 0.01  # 100Hz sampling
        time_steps = int(duration / dt)
        time_data = []
        position_data = []
        velocity_data = []
        
        for i in range(time_steps):
            time_data.append(i * dt)
            pos, vel = real_robot.get_joint_state(joint_idx)
            position_data.append(pos)
            velocity_data.append(vel)
        
        # Remove input
        real_robot.apply_torque(joint_idx, 0.0)
        
        return {
            'time': np.array(time_data),
            'position': np.array(position_data),
            'velocity': np.array(velocity_data)
        }
    
    def simulate_system(self, params, input_torque, time_data):
        """Simulate system with given parameters"""
        # Define system dynamics
        def system_dynamics(t, state, params, torque):
            # state = [position, velocity]
            pos, vel = state
            
            # Simplified dynamics: m*ddot{x} + b*dot{x} + k*x = torque
            mass = params['mass']
            damping = params['damping']
            stiffness = params['stiffness']
            
            acc = (torque - damping * vel - stiffness * pos) / mass
            return [vel, acc]
        
        # Initial conditions
        initial_state = [0.0, 0.0]  # Starting at rest
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, state: system_dynamics(t, state, params, input_torque),
            [time_data[0], time_data[-1]],
            initial_state,
            t_eval=time_data,
            method='RK45'
        )
        
        return {
            'position': sol.y[0],
            'velocity': sol.y[1]
        }
    
    def identify_parameters(self, real_robot, joint_idx):
        """Identify system parameters by minimizing error"""
        # Collect data from real robot
        input_torque = 1.0  # 1 Nm step input
        real_data = self.collect_step_response_data(real_robot, joint_idx, input_torque)
        
        # Objective function to minimize
        def objective(params_list):
            # Convert list to parameter dictionary
            params = {
                'mass': params_list[0],
                'damping': params_list[1],
                'stiffness': params_list[2]
            }
            
            # Simulate with these parameters
            sim_data = self.simulate_system(params, input_torque, real_data['time'])
            
            # Calculate error between real and simulated responses
            pos_error = np.mean((real_data['position'] - sim_data['position'])**2)
            vel_error = np.mean((real_data['velocity'] - sim_data['velocity'])**2)
            
            total_error = pos_error + vel_error
            return total_error
        
        # Initial guess for parameters
        initial_guess = [1.0, 0.1, 10.0]  # mass, damping, stiffness
        
        # Optimize parameters
        result = minimize(objective, initial_guess, method='BFGS')
        
        # Extract identified parameters
        identified_params = {
            'mass': result.x[0],
            'damping': result.x[1],
            'stiffness': result.x[2]
        }
        
        return identified_params
```

### Closed-Loop System Identification

```python
# closed_loop_identification.py
class ClosedLoopSystemIdentifier(SystemIdentifier):
    def __init__(self, robot_model):
        super().__init__(robot_model)
        self.controller = None  # PID or other controller
    
    def collect_closed_loop_data(self, real_robot, reference_signal):
        """Collect data with controller running"""
        time_data = []
        position_data = []
        velocity_data = []
        reference_data = []
        
        dt = 0.01  # 100Hz
        duration = 10.0  # 10 seconds of data
        steps = int(duration / dt)
        
        for i in range(steps):
            t = i * dt
            
            # Get current state
            pos, vel = real_robot.get_state()
            
            # Get reference for this time
            ref = reference_signal(t)
            
            # Compute control action
            control_action = self.controller.compute_control(pos, ref, vel)
            
            # Apply control
            real_robot.apply_control(control_action)
            
            # Store data
            time_data.append(t)
            position_data.append(pos)
            velocity_data.append(vel)
            reference_data.append(ref)
        
        return {
            'time': np.array(time_data),
            'position': np.array(position_data),
            'velocity': np.array(velocity_data),
            'reference': np.array(reference_data)
        }
```

## Dynamics Randomization

Dynamics randomization focuses on randomizing the dynamic properties of the system:

```python
# dynamics_randomizer.py
class DynamicsRandomizer:
    def __init__(self):
        self.dynamics_params = {
            'mass_variance': 0.1,  # ±10% mass variation
            'inertia_variance': 0.15,  # ±15% inertia variation
            'friction_variance': 0.2,  # ±20% friction variation
            'gear_ratio_variance': 0.05,  # ±5% gear ratio variation
            'motor_constant_variance': 0.1,  # ±10% motor constant variation
        }
    
    def randomize_dynamics(self, robot):
        """Randomize the dynamics of the robot"""
        # Randomize mass properties
        self._randomize_mass_properties(robot)
        
        # Randomize friction parameters
        self._randomize_friction_properties(robot)
        
        # Randomize actuator properties
        self._randomize_actuator_properties(robot)
    
    def _randomize_mass_properties(self, robot):
        """Randomize mass and inertia properties"""
        for link in robot.links:
            # Randomize mass
            original_mass = robot.get_link_mass(link)
            mass_factor = 1.0 + np.random.normal(0, self.dynamics_params['mass_variance'])
            robot.set_link_mass(link, original_mass * max(0.1, mass_factor))
            
            # Randomize inertia
            original_inertia = robot.get_link_inertia(link)
            inertia_factor = 1.0 + np.random.normal(0, self.dynamics_params['inertia_variance'])
            robot.set_link_inertia(link, original_inertia * max(0.1, inertia_factor))
    
    def _randomize_friction_properties(self, robot):
        """Randomize friction parameters"""
        for joint in robot.joints:
            # Randomize static and dynamic friction
            static_friction = robot.get_joint_static_friction(joint)
            dynamic_friction = robot.get_joint_dynamic_friction(joint)
            
            static_factor = 1.0 + np.random.normal(0, self.dynamics_params['friction_variance'])
            dynamic_factor = 1.0 + np.random.normal(0, self.dynamics_params['friction_variance'])
            
            robot.set_joint_friction(
                joint, 
                static_friction * max(0.01, static_factor),
                dynamic_friction * max(0.01, dynamic_factor)
            )
    
    def _randomize_actuator_properties(self, robot):
        """Randomize actuator properties"""
        for joint in robot.joints:
            # Randomize gear ratio
            original_gear_ratio = robot.get_joint_gear_ratio(joint)
            gear_factor = 1.0 + np.random.normal(0, self.dynamics_params['gear_ratio_variance'])
            robot.set_joint_gear_ratio(joint, original_gear_ratio * gear_factor)
            
            # Randomize motor constants
            original_motor_constant = robot.get_joint_motor_constant(joint)
            motor_factor = 1.0 + np.random.normal(0, self.dynamics_params['motor_constant_variance'])
            robot.set_joint_motor_constant(joint, original_motor_constant * motor_factor)
```

## Sensor Randomization

Sensor randomization helps policies become robust to sensor noise and variations:

```python
# sensor_randomizer.py
class SensorRandomizer:
    def __init__(self):
        self.sensor_params = {
            'camera_noise_range': [0.0, 0.1],  # Noise level range
            'imu_noise_range': [0.0, 0.05],   # IMU noise range
            'lidar_noise_range': [0.0, 0.02], # LIDAR noise range
            'delay_range': [0.0, 0.05],       # Sensor delay range (s)
            'bias_range': [-0.01, 0.01],      # Sensor bias range
        }
    
    def randomize_sensors(self, robot):
        """Randomize sensor properties"""
        # Randomize camera sensors
        self._randomize_camera_sensors(robot)
        
        # Randomize IMU sensors
        self._randomize_imu_sensors(robot)
        
        # Randomize LIDAR sensors
        self._randomize_lidar_sensors(robot)
    
    def _randomize_camera_sensors(self, robot):
        """Randomize camera sensor properties"""
        for camera in robot.cameras:
            # Randomize noise level
            noise_level = np.random.uniform(*self.sensor_params['camera_noise_range'])
            robot.set_camera_noise_level(camera, noise_level)
            
            # Randomize delay
            delay = np.random.uniform(*self.sensor_params['delay_range'])
            robot.set_camera_delay(camera, delay)
    
    def _randomize_imu_sensors(self, robot):
        """Randomize IMU sensor properties"""
        for imu in robot.imus:
            # Randomize noise level
            noise_level = np.random.uniform(*self.sensor_params['imu_noise_range'])
            robot.set_imu_noise_level(imu, noise_level)
            
            # Randomize bias
            bias = np.random.uniform(*self.sensor_params['bias_range'])
            robot.set_imu_bias(imu, bias)
    
    def _randomize_lidar_sensors(self, robot):
        """Randomize LIDAR sensor properties"""
        for lidar in robot.lidars:
            # Randomize noise level
            noise_level = np.random.uniform(*self.sensor_params['lidar_noise_range'])
            robot.set_lidar_noise_level(lidar, noise_level)
            
            # Randomize delay
            delay = np.random.uniform(*self.sensor_params['delay_range'])
            robot.set_lidar_delay(lidar, delay)
```

## Transfer Validation and Safety

### Safety Measures

Before transferring to real robots, implement safety measures:

```python
# safety_checker.py
class SafetyChecker:
    def __init__(self, robot_limits):
        self.position_limits = robot_limits['position']
        self.velocity_limits = robot_limits['velocity']
        self.torque_limits = robot_limits['torque']
        self.safety_margin = 0.1  # 10% safety margin
    
    def check_action_safe(self, action, current_state):
        """Check if an action is safe to execute"""
        # Check position limits
        next_positions = current_state['position'] + action * 0.01  # Assuming 10ms timestep
        if not self._check_position_limits(next_positions):
            return False, "Position limit violation"
        
        # Check velocity limits
        estimated_velocity = action / 0.01  # Rough velocity estimate
        if not self._check_velocity_limits(estimated_velocity):
            return False, "Velocity limit violation"
        
        # Check torque limits
        if not self._check_torque_limits(action):
            return False, "Torque limit violation"
        
        return True, "Action is safe"
    
    def _check_position_limits(self, positions):
        """Check if positions are within limits"""
        for i, pos in enumerate(positions):
            min_limit = self.position_limits[i][0] * (1 - self.safety_margin)
            max_limit = self.position_limits[i][1] * (1 + self.safety_margin)
            if pos < min_limit or pos > max_limit:
                return False
        return True
    
    def _check_velocity_limits(self, velocities):
        """Check if velocities are within limits"""
        for i, vel in enumerate(velocities):
            max_limit = self.velocity_limits[i] * (1 + self.safety_margin)
            if abs(vel) > max_limit:
                return False
        return True
    
    def _check_torque_limits(self, torques):
        """Check if torques are within limits"""
        for i, torque in enumerate(torques):
            max_limit = self.torque_limits[i] * (1 + self.safety_margin)
            if abs(torque) > max_limit:
                return False
        return True
```

### Gradual Transfer Protocol

```python
# gradual_transfer.py
class GradualTransfer:
    def __init__(self, safety_checker):
        self.safety_checker = safety_checker
        self.transfer_stage = 0  # 0: simulation only, 1: safe real-world, 2: full real-world
        self.performance_thresholds = [0.7, 0.85, 0.95]  # Performance thresholds for each stage
    
    def evaluate_policy_performance(self, policy, env):
        """Evaluate policy performance in environment"""
        total_reward = 0
        episodes = 10
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / episodes
    
    def can_progress_stage(self, policy, real_env):
        """Check if we can progress to the next transfer stage"""
        current_performance = self.evaluate_policy_performance(policy, real_env)
        threshold = self.performance_thresholds[self.transfer_stage]
        
        if current_performance >= threshold:
            return True, f"Performance {current_performance:.3f} meets threshold {threshold:.3f}"
        else:
            return False, f"Performance {current_performance:.3f} below threshold {threshold:.3f}"
    
    def transfer_policy(self, policy, sim_env, real_env):
        """Gradually transfer policy from sim to real"""
        while self.transfer_stage < 3:
            print(f"Attempting transfer to stage {self.transfer_stage + 1}")
            
            can_progress, message = self.can_progress_stage(policy, real_env)
            print(f"Stage {self.transfer_stage + 1} evaluation: {message}")
            
            if can_progress:
                self.transfer_stage += 1
                print(f"Progressed to stage {self.transfer_stage}")
                
                if self.transfer_stage == 1:
                    print("Stage 1: Safe real-world execution with safety limits")
                elif self.transfer_stage == 2:
                    print("Stage 2: Full real-world execution")
                elif self.transfer_stage == 3:
                    print("Transfer complete!")
                    break
            else:
                print(f"Cannot progress, need to improve policy performance")
                # In practice, this would involve more training or fine-tuning
                break
```

## Practical Exercise: Sim-to-Real Transfer

Implement a complete sim-to-real transfer pipeline:

1. **Create a simulated robot environment** with domain randomization
2. **Train a policy** in the randomized simulation
3. **Identify real robot parameters** using system identification
4. **Apply safety checks** before real-world execution
5. **Gradually transfer** the policy to the real robot

### Complete Transfer Pipeline

```python
# complete_transfer_pipeline.py
def complete_sim_to_real_pipeline():
    """Complete pipeline for sim-to-real transfer"""
    
    # 1. Set up simulation with domain randomization
    sim_env = create_randomized_simulation()
    domain_randomizer = DomainRandomizer()
    
    # 2. Train policy in simulation
    policy = train_policy_in_simulation(sim_env, domain_randomizer)
    
    # 3. Connect to real robot
    real_robot = connect_to_real_robot()
    
    # 4. Identify real robot parameters
    system_identifier = SystemIdentifier(robot_model=sim_env.model)
    identified_params = system_identifier.identify_parameters(real_robot, joint_idx=0)
    
    # 5. Update simulation with identified parameters
    update_simulation_with_real_params(sim_env, identified_params)
    
    # 6. Apply safety checks
    safety_checker = SafetyChecker(real_robot.get_limits())
    
    # 7. Gradual transfer
    gradual_transfer = GradualTransfer(safety_checker)
    gradual_transfer.transfer_policy(policy, sim_env, real_robot)
    
    print("Sim-to-real transfer completed successfully!")

def create_randomized_simulation():
    """Create a simulation environment with domain randomization"""
    # Implementation depends on the specific simulator (Isaac, PyBullet, etc.)
    pass

def train_policy_in_simulation(sim_env, domain_randomizer):
    """Train a policy in the randomized simulation environment"""
    # Implementation would use RL training algorithms
    pass

def connect_to_real_robot():
    """Connect to the real robot"""
    # Implementation depends on the specific robot hardware
    pass

def update_simulation_with_real_params(sim_env, params):
    """Update simulation with real-world parameters"""
    # Update the simulation model with identified parameters
    pass
```

## Best Practices for Sim-to-Real Transfer

1. **Start Conservative**: Begin with wide parameter ranges to ensure robustness
2. **Validate in Simulation**: Ensure the policy works across the randomization range
3. **Use System Identification**: Measure real robot parameters when possible
4. **Implement Safety**: Always have safety checks before real-world execution
5. **Gradual Transfer**: Progress from simulation to real-world gradually
6. **Monitor Performance**: Track performance metrics during transfer
7. **Have Fallback Plans**: Be ready to switch to safe controllers if needed

## Summary

Sim-to-real transfer is a critical challenge in robotics that requires careful consideration of the differences between simulation and reality. Domain randomization, system identification, and safety measures are key techniques for successful transfer. The process requires iterative refinement and validation to ensure policies trained in simulation can operate safely and effectively on real robots.

## Next Steps

With a foundation in simulation and transfer techniques, we'll move on to Module 4, where we'll explore humanoid robot development and conversational robotics.