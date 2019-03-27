# **Extended Kalman Filter Project** 


### In this project, I implemented an Extended Kalman Filter (EKF) in C++ and used it to detect a bicycle that travels around the vehicle. I used the Kalman filter, lidar measurements, and radar measurements to accurately track the bicycle's position and velocity.

---

The following files are included in the `src` directory:
* `main.cpp` - communicates with the Udacity Simulator web socket - receives data measurements, calls a function to run the Kalman filter, calls a function to calculate RMSE
* `FusionEKF.cpp` - initializes the filter, calls the predict function, calls the update function
* `kalman_filter.cpp` - defines the predict function, the update function for lidar, and the update function for radar
* `tools.cpp` - function to calculate RMSE and the Jacobian matrix

A visualization is shown below.

(insert gif)

[//]: # (Image References)

[image0a]: ./imgs/center.jpg "center"
[image0b]: ./imgs/left.jpg "left"
[image0c]: ./imgs/right.jpg "right"
