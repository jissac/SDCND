# **PID Controller Project** 

### In this project, I implemented a proportional–integral–derivative (PID) controller in C++ to maneuver the vehicle around the track.

## Overview
Using the car's distance from the lane's center, or cross track error (CTE) and the speed (mph) provided by the Udacity simulator, I computed the appropriate steering angle necessary for the vehicle to traverse the track. Compare this method to the deep learning technique used to drive the car around the track in my [Behavioral Cloning project](https://github.com/jissac/SDCND/tree/master/Term1/Behavioral_Cloning).

## Control
Control is how we use the steering, throttle, and brakes to make the car go where we want it to go - a PID controller is a type of feedback controller that uses past, present, and future error to control a desired system. Check out [this video](https://www.youtube.com/watch?v=wkfEZmsQqiA) for a good intuition about how a PID controller works. 

### Proportional (P) Component
The proportional portion of the PID controller uses the CTE at the present moment to steer the car left or right depending on how far the car is from the lane center. Increases or decreasing the gain value adjusts how fast we reach the center. 

### Integral (I) Component
Using just a proportional controller results in steady-state error, whereas using an integral counteracts the bias in the CTE. The integral portion of the PID controller uses the past error to sum the inputs over time, a sort of memory of what has happened before. 

### Differential (D) Component
The differential portion of the PID controller measures the rate of change of the error and protects for the future error by counteracting the P component's tendency to overshoot the center line.

### Hyperparameter tuning
I manually tuned the hyperparameters of each P, I, and D component in order to have the car drive properly around the track. The final hyperparameter values I settled with for the P, I, D components were 0.18, 0.0004, and 2.9, respectively. Further tuning would make the driving smoother, as there are portions of the track where the vehicle swings wildly (while still staying on the driveable portion of the track).
