# Capstone Project: Building a self-driving car
In this final project, I built a ROS-enabled system that allows Carla (Udacity's autonomous vehicle) to drive around a test track while avoiding obstacles and stopping at traffic lights. 

The perception, control, and planning subsystems are essential to building a successful self-driving car. 

## Perception
I implemented traffic light detection and obstacle detection 

## Planning
I implemented a node called the waypoint updater, which sets the target velocity for each waypoint based on the upcoming traffic lights and obstacles. 

## Control
I implemented a drive-by-wire ROS that takes target trajectory as input and sends control commands to navigate the vehicle.
