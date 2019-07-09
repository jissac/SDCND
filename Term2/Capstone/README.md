# Capstone Project: Building a self-driving car
In this final project, I built a ROS-enabled system that allows Carla (Udacity's autonomous vehicle) to drive around a test track while avoiding obstacles and stopping at traffic lights. 

The project was divided into three distinct parts: perception, control, and planning. **Perception** involved building a traffic light detection node and a traffic light classification node. **Planning** node called the waypoint updater, which sets the target velocity for each waypoint based on the upcoming traffic lights and obstacles. **Control** drive-by-wire ROS that takes target trajectory as input and sends control commands to navigate the vehicle.

## Implementation
I implemented the project in the following steps:
### Waypoint Updater Node
The purpose of this node is to publish a fixed number of waypoints ahead of the vehicle with the correct target velocities, depending on traffic lights and obstacles. 

### Drive-by-Wire (DBW) Node
The DBW node contains the ROS message definitions and acts as the bridge between hardware and software by running in the car and allowing ROS to actuate the car's steering and throttle.
