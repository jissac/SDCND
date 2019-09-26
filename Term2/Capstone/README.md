# Capstone Project: Building a self-driving car
In this final project, I built a ROS-enabled system that allows Carla (Udacity's autonomous vehicle) to drive around a test track while avoiding obstacles and stopping at traffic lights. 

The project was divided into three distinct parts or subsystems: perception, control, and planning. **Perception** involved building a traffic light detection node and a traffic light classification node. **Planning** involved coding the the waypoint updater, which sets the target velocity for each waypoint based on the upcoming traffic lights and obstacles. **Control** involved the drive-by-wire ROS node that takes target trajectory as input and sends control commands to navigate the vehicle. Combining these three parts into one system ensured that the vehicle was able to effectively navigate around the test track and stop at traffic lights.

![image1](./capstone_overview.png)


## Implementation
I implemented the project in the following steps:

### Traffic light detection
Using the vehicle's location and the (x, y) coordinates for traffic lights allows us to find the nearest visible traffic light ahead of the vehicle. This takes place in the `process_traffic_lights` method of `tl_detector.py`. Using these waypoint indices (`get_closest_waypoint()`), I determined which light is ahead of the vehicle along the list of waypoints. Furthermore, I used the camera image data to classify the color of the traffic light. The core functionality of this step takes place in the `get_light_state` method of `tl_detector.py`.

### Waypoint Updater Node
The purpose of this node is to publish a fixed number of waypoints ahead of the vehicle with the correct target velocities, depending on traffic lights and obstacles. Refer to `./ros/src/waypoint_updater/` for more details.

### Drive-by-Wire (DBW) Node
The DBW node contains the ROS message definitions and acts as the bridge between hardware and software by allowing ROS to actuate the car's steering and throttle. Refer to `./ros/src/twist_controller/` for more details.


