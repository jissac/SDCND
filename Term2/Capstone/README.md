# Capstone Project: Building a self-driving car
In this final project, I built a ROS-enabled system that allows Carla (Udacity's autonomous vehicle) to drive around a test track while avoiding obstacles and stopping at traffic lights. 

The project was divided into three distinct parts: perception, control, and planning.

## Perception
Prediction involved building a w21traffic light detection node and a traffic light classification node. 

## Planning
I implemented a node called the waypoint updater, which sets the target velocity for each waypoint based on the upcoming traffic lights and obstacles. 

## Control
I implemented a drive-by-wire ROS that takes target trajectory as input and sends control commands to navigate the vehicle.
