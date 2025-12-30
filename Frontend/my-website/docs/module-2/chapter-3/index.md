# Chapter 3: Unity Visualization & Human-Robot Interaction (HRI)

## Learning Objectives

After completing this chapter, you will be able to:
- Understand Unity's role in robotics visualization and simulation
- Set up Unity for robotics applications using ROS# or Unity Robotics Hub
- Implement Human-Robot Interaction (HRI) concepts in Unity
- Design intuitive interfaces for robot teleoperation
- Create immersive AR/VR experiences for robot monitoring and control

## Introduction to Unity for Robotics

Unity is a powerful real-time 3D development platform that has found significant applications in robotics simulation and visualization. Unlike traditional physics simulators like Gazebo, Unity excels in creating photorealistic environments and intuitive human interfaces for robot interaction.

### Unity Robotics Integration Options

1. **Unity Robotics Hub**: A comprehensive package that provides tools for robotics simulation
2. **ROS# (ROS Sharp)**: A Unity package for ROS communication
3. **Unity ML-Agents**: For training robot behaviors using reinforcement learning

## Setting Up Unity for Robotics

### Installing Unity Robotics Hub

Unity Robotics Hub provides official tools and samples for robotics applications:

1. Install Unity Hub and Unity Editor (2021.3 LTS or later recommended)
2. Through Unity Hub, install the Unity Robotics Hub
3. Create a new project or import Robotics samples

### Basic ROS Communication Setup

```csharp
// RobotController.cs - Basic ROS communication in Unity
using UnityEngine;
using RosSharp.RosBridgeClient;

public class RobotController : MonoBehaviour
{
    private RosSocket rosSocket;
    private string robotName = "my_robot";

    void Start()
    {
        // Initialize connection to ROS
        RosConnector rosConnector = GetComponent<RosConnector>();
        rosSocket = rosConnector.RosSocket;

        // Subscribe to robot joint states
        rosSocket.Subscribe<sensor_msgs.JointState>(
            "/joint_states",
            ReceiveJointStates
        );

        // Set up publisher for velocity commands
        InvokeRepeating("PublishVelocityCommand", 0.1f, 0.1f);
    }

    void ReceiveJointStates(sensor_msgs.JointState jointState)
    {
        // Update robot visualization based on joint states
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];

            // Find and update the corresponding joint in Unity
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint visualization
                UpdateJoint(jointTransform, jointPosition);
            }
        }
    }

    void PublishVelocityCommand()
    {
        // Publish velocity commands to ROS
        geometry_msgs.Twist twist = new geometry_msgs.Twist();
        twist.linear.x = 0.5f; // Move forward at 0.5 m/s
        twist.angular.z = 0.2f; // Rotate at 0.2 rad/s

        rosSocket.Publish("/cmd_vel", twist);
    }

    Transform FindJointByName(string name)
    {
        // Implementation depends on your robot model hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }

    void UpdateJoint(Transform joint, float position)
    {
        // Update rotation or position based on joint value
        joint.Rotate(Vector3.up, position * Mathf.Rad2Deg);
    }
}
```

## Photorealistic Simulation with Unity

### Creating Realistic Environments

Unity's physically-based rendering (PBR) pipeline enables creation of photorealistic environments:

1. **Material Setup**:
   - Use Standard Shader with proper metallic/roughness maps
   - Set up environment lighting with HDRI
   - Configure post-processing effects (bloom, color grading)

2. **Lighting Configuration**:
   - Use realistic light intensities and colors
   - Configure shadows with appropriate settings
   - Set up reflection probes for accurate reflections

### Environment Design Principles

- **Scale Accuracy**: Maintain real-world scale for accurate simulation
- **Physics Properties**: Configure realistic friction, bounciness, etc.
- **Visual Fidelity**: Match textures and materials to real-world counterparts

```csharp
// EnvironmentFidelity.cs - Ensuring environment accuracy
using UnityEngine;

public class EnvironmentFidelity : MonoBehaviour
{
    [Header("Physical Properties")]
    [Tooltip("Real-world scale of the environment (1 unit = 1 meter)")]
    public float realWorldScale = 1.0f;

    [Header("Physics Configuration")]
    public PhysicMaterial floorMaterial;

    void Start()
    {
        ConfigureEnvironment();
    }

    void ConfigureEnvironment()
    {
        // Ensure proper scale
        transform.localScale = Vector3.one * realWorldScale;

        // Configure physics materials
        if (floorMaterial != null)
        {
            floorMaterial.staticFriction = 0.5f;   // Typical for concrete
            floorMaterial.dynamicFriction = 0.4f;  // Typical for concrete
            floorMaterial.bounciness = 0.1f;       // Minimal bounce
        }

        // Configure lighting to match real conditions
        SetupLighting();
    }

    void SetupLighting()
    {
        // Configure sun light to match real-world conditions
        Light[] lights = FindObjectsOfType<Light>();
        foreach (Light light in lights)
        {
            if (light.type == LightType.Directional) // Sun-like light
            {
                light.intensity = 1.0f; // Realistic sun intensity in Unity
                light.color = Color.white;
            }
        }
    }
}
```

## Human-Robot Interaction (HRI) in Unity

### Designing Intuitive Control Interfaces

Unity excels at creating intuitive interfaces for robot control and monitoring:

1. **Visual Feedback Systems**:
   - Real-time visualization of sensor data
   - Path planning visualization
   - Status indicators

2. **Teleoperation Interfaces**:
   - Gamepad/keyboard control mapping
   - VR controller integration
   - Touch screen interfaces

### Unity HRI Implementation Example

```csharp
// HRInterfaceManager.cs - Human-Robot Interface Management
using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class HRInterfaceManager : MonoBehaviour
{
    [Header("Control Mapping")]
    public KeyCode forwardKey = KeyCode.W;
    public KeyCode backwardKey = KeyCode.S;
    public KeyCode leftKey = KeyCode.A;
    public KeyCode rightKey = KeyCode.D;
    public KeyCode actionKey = KeyCode.Space;

    [Header("UI Elements")]
    public Slider speedSlider;
    public Text statusText;
    public Image sensorVisualization;

    private float baseSpeed = 0.2f;
    private bool isControlling = false;

    void Update()
    {
        HandleUserInput();
        UpdateUI();
    }

    void HandleUserInput()
    {
        float linearVel = 0f;
        float angularVel = 0f;

        if (Input.GetKey(forwardKey))
            linearVel = baseSpeed;
        else if (Input.GetKey(backwardKey))
            linearVel = -baseSpeed;

        if (Input.GetKey(leftKey))
            angularVel = baseSpeed;
        else if (Input.GetKey(rightKey))
            angularVel = -baseSpeed;

        // Publish command if there's input
        if (linearVel != 0 || angularVel != 0)
        {
            PublishRobotCommand(linearVel, angularVel);
            isControlling = true;
        }
        else
        {
            isControlling = false;
        }

        // Handle action key (e.g., for gripper control)
        if (Input.GetKeyDown(actionKey))
        {
            ExecuteActionCommand();
        }
    }

    void PublishRobotCommand(float linear, float angular)
    {
        // In a real implementation, this would publish to ROS
        Debug.Log($"Sending cmd: linear={linear}, angular={angular}");
        // rosSocket.Publish("/cmd_vel", CreateTwistCommand(linear, angular));
    }

    void ExecuteActionCommand()
    {
        // Execute special action (e.g., gripper, camera capture)
        Debug.Log("Executing action command");
        // rosSocket.Publish("/action_cmd", CreateActionMessage());
    }

    void UpdateUI()
    {
        // Update status text
        statusText.text = isControlling ? "Controlling Robot" : "Idle";

        // Update based on sensor data if available
        UpdateSensorVisualization();
    }

    void UpdateSensorVisualization()
    {
        // Example: Update based on LiDAR data
        // This would be connected to actual sensor data in a real implementation
        if (Random.value > 0.8f) // Simulated sensor data
        {
            sensorVisualization.color = Color.red; // Obstacle detected
        }
        else
        {
            sensorVisualization.color = Color.green; // Path clear
        }
    }
}
```

## AR/VR Integration for Robotics

### Virtual Reality for Robot Teleoperation

VR provides immersive control interfaces that can enhance teleoperation:

```csharp
// VRControlInterface.cs - VR-based robot control
using UnityEngine;
using UnityEngine.XR;

public class VRControlInterface : MonoBehaviour
{
    [Header("VR Controllers")]
    public Transform leftController;
    public Transform rightController;

    [Header("Robot Configuration")]
    public float translationSpeed = 0.5f;
    public float rotationSpeed = 45.0f; // degrees per second

    void Update()
    {
        HandleVRInput();
    }

    void HandleVRInput()
    {
        // Get controller inputs
        if (leftController != null)
        {
            // Use left controller for movement
            Vector3 joyInput = GetLeftControllerStickInput();
            MoveRobot(joyInput.x, joyInput.y);
        }

        if (rightController != null)
        {
            // Use right controller for camera/robot orientation
            Vector3 joyInput = GetRightControllerStickInput();
            RotateRobot(joyInput.x, joyInput.y);
        }
    }

    Vector3 GetLeftControllerStickInput()
    {
        // Implementation depends on VR SDK
        // For OpenXR or Oculus, this would use input actions
        return new Vector3(0, 0, 0); // placeholder
    }

    Vector3 GetRightControllerStickInput()
    {
        // Implementation depends on VR SDK
        return new Vector3(0, 0, 0); // placeholder
    }

    void MoveRobot(float xInput, float zInput)
    {
        // Move robot based on stick input
        Vector3 movement = new Vector3(xInput, 0, zInput) * translationSpeed * Time.deltaTime;

        // Publish movement command to robot
        Debug.Log($"VR Movement: {movement}");
    }

    void RotateRobot(float xInput, float yInput)
    {
        // Rotate robot based on stick input
        float rotation = yInput * rotationSpeed * Time.deltaTime;

        // Publish rotation command to robot
        Debug.Log($"VR Rotation: {rotation}");
    }
}
```

## Unity-ROS Bridge Configuration

### Setting up ROS Communication

The Unity-RosBridge connection allows Unity to communicate with ROS systems:

```csharp
// RosBridgeManager.cs - Manager for ROS Bridge communication
using UnityEngine;
using RosSharp.RosBridgeClient;

public class RosBridgeManager : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosBridgeServerUrl = "ws://127.0.0.1:9090";

    [Header("Topics")]
    public string cmdVelTopic = "/cmd_vel";
    public string jointStatesTopic = "/joint_states";

    private RosSocket rosSocket;

    void Start()
    {
        ConnectToRosBridge();
    }

    void ConnectToRosBridge()
    {
        rosSocket = new RosSocket(new WebSocketNetProtocol(rosBridgeServerUrl));

        // Subscribe to robot feedback topics
        rosSocket.Subscribe<sensor_msgs.JointState>(
            jointStatesTopic,
            OnJointStatesReceived
        );

        Debug.Log("Connected to ROS Bridge: " + rosBridgeServerUrl);
    }

    void OnJointStatesReceived(sensor_msgs.JointState jointState)
    {
        // Process joint state data
        Debug.Log($"Received joint state for {jointState.name.Count} joints");

        // Update robot visualization in Unity
        UpdateRobotVisualization(jointState);
    }

    void UpdateRobotVisualization(sensor_msgs.JointState jointState)
    {
        // Update the robot model based on joint positions
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            double jointPosition = jointState.position[i];

            // Find and update corresponding joint in Unity
            GameObject jointObject = GameObject.Find(jointName);
            if (jointObject != null)
            {
                // Apply joint position to Unity object
                jointObject.transform.localRotation =
                    Quaternion.AngleAxis((float)jointPosition * Mathf.Rad2Deg, Vector3.up);
            }
        }
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (rosSocket != null)
        {
            geometry_msgs.Twist twist = new geometry_msgs.Twist();
            twist.linear = new geometry_msgs.Vector3 { x = linearX };
            twist.angular = new geometry_msgs.Vector3 { z = angularZ };

            rosSocket.Publish(cmdVelTopic, twist);
        }
    }

    void OnApplicationQuit()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

## Practical Exercise: Unity Robot Visualization

Create a Unity scene that visualizes a real robot's state:

1. **Import ROS# Package** into your Unity project
2. **Create a robot model** with articulated joints
3. **Set up ROS bridge connection** to your robot
4. **Implement joint state subscription** to update robot visualization
5. **Add UI elements** to show robot status and sensor data

### Step-by-Step Implementation

1. **Create Unity Project** with 3D objects representing robot links
2. **Add ROS connection scripts** to manage communication
3. **Configure ROS topics** for joint states and other feedback
4. **Test with a real robot or simulation** to verify visualization accuracy
5. **Develop HRI interface** for commanding the robot through Unity

## Summary

Unity provides powerful capabilities for robotics visualization and human-robot interaction. Its photorealistic rendering, intuitive interface design tools, and AR/VR support make it ideal for creating immersive and effective robot teleoperation and monitoring systems.

When combined with ROS through the Unity-RosBridge, Unity becomes a comprehensive platform for visualizing robot state, creating intuitive control interfaces, and developing advanced HRI applications in virtual and augmented reality environments.

## Diagrams and Visual Aids

![Unity Robotics Environment](/img/unity-robotics.png)

*Figure 1: Unity environment with robot visualization*

![HRI Interface Design](/img/hri-interface.png)

*Figure 2: Example of Human-Robot Interaction interface*

![VR Teleoperation](/img/vr-teleop.png)

*Figure 3: VR-based robot teleoperation setup*