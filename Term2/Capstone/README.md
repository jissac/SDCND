# Capstone Project: Building a self-driving car
The goal of this project is to enable Karla (the Udacity car) to drive around a test track using waypoint navigation, while avoiding obstacles and stopping at traffic lights. I implemented the components of the perception, control, and planning subsystems in order to accomplish this task.

## Perception
I implemented traffic light detection and obstacle detection 

## Planning
I implemented a node called the waypoint updater, which sets the target velocity for each waypoint based on the upcoming traffic lights and obstacles. 

## Control
I implemented a drive-by-wire ROS that takes target trajectory as input and sends control commands to navigate the vehicle.
