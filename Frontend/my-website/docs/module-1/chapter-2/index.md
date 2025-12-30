# Chapter 2: Nodes, Topics, Services, and Actions

## Learning Objectives

After completing this chapter, you will be able to:
- Implement ROS 2 nodes using Python and rclpy
- Create publishers and subscribers for topic-based communication
- Implement request/reply communication using services
- Design goal-based workflows using actions
- Choose the appropriate communication pattern for different scenarios

## Nodes in Depth

Nodes are processes that perform computation. In ROS 2, nodes are organized into packages, which are groups of nodes that work together to perform a specific function.

### Creating a Node with rclpy

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        self.get_logger().info('MyNode has been initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)  # Keep the node alive
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Parameters

Nodes can accept parameters at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare a parameter with default value
        self.declare_parameter('my_parameter', 'default_value')

        # Get the parameter value
        param_value = self.get_parameter('my_parameter').value
        self.get_logger().info(f'Parameter value: {param_value}')
```

## Topics: Publish/Subscribe Communication

Topics enable asynchronous, one-to-many communication using the publish/subscribe pattern.

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        
        self.publisher = self.create_publisher(String, 'my_topic', 10)
        self.timer = self.create_timer(0.5, self.publish_message)
        self.i = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.i += 1
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'my_topic',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

## Services: Request/Reply Communication

Services enable synchronous, one-to-one communication using the request/reply pattern.

### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServerNode(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response
```

### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions: Goal-Based Communication

Actions are used for long-running tasks that require feedback and the ability to cancel.

### Creating an Action Server

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')
        return result
```

## Communication Pattern Selection Guide

| Pattern | Communication Type | Use Case | Example |
|---------|-------------------|----------|---------|
| Topics | Publish/Subscribe | Continuous data streams | Sensor data, robot pose |
| Services | Request/Reply | One-time queries | Getting robot status |
| Actions | Goal-Based | Long-running tasks | Navigation, manipulation |

## Practical Exercise

Implement a complete ROS 2 system that demonstrates all three communication patterns:
1. A publisher that publishes temperature sensor readings
2. A subscriber that logs these readings
3. A service that returns the average temperature over the last 10 readings
4. An action that runs a heating/cooling cycle for a specified duration

## Summary

ROS 2 provides three primary communication patterns:
- **Topics**: For continuous data streaming with publisher/subscriber pattern
- **Services**: For request/reply communication for immediate responses
- **Actions**: For goal-based communication with feedback and cancellation for long-running tasks

Each pattern serves different communication needs in robotic systems. Understanding when and how to use each pattern is crucial for designing effective robotic architectures.

## Diagrams and Visual Aids

![Communication Patterns Comparison](/img/communication-patterns.png)

*Figure 1: Comparison of Topics, Services, and Actions communication patterns*

![Node Communication Example](/img/node-communication.png)

*Figure 2: Example of nodes communicating via different patterns*