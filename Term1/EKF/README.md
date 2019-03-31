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

### Overview
The lidar sensor measures position with high accuracy. Using the doppler effect, the radar sensor can measure radial velocity (the component of velocity moving towards or away from the sensor) of a moving object. However, radar has lower spatial accuracy than lidar. Sensor fusion allows us to combine both lidar and radar measurements and gives us a more accurate location and velocity update.

### Radar Measurments
The radar provides us with three key measurements: range (rho), bearing (phi), and radial velocity (rho dot). This measurement is in the polar coordinate system.

### Lidar Measurements

### The Kalman Filter
initializing Kalman filter variables
predicting where our object is going to be after a time step \Delta{t}Δt
updating where our object is based on sensor measurements

### The Extended Kalman Filter

Jacobian

### Root Mean Squared Error
In order to check how far the estimated result is from ground truth result, we use the Root Mean Squared Error (RMSE) calculation - an accuracy metric used to measure the deviation of the estimated state from the true state. The lower the RMSE, the higher the estimation accuracy. The formula is given by:

