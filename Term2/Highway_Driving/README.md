# Highway Driving

### In this project, I wrote a path planning algorithm in C++ that implements SAE Level 4 functionality. 

---

**Outline**

There are three core components to path planning

* Predicting what other vehicles on the road will do next
* Deciding on what maneuvers to execute given our goals and predicitons about the other vehicles
* Building a trajectory to execute the maneuvers we decide on

The code for this project is contained in the `src` directory. 

### Goals
In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

### Prediction
The prediction step is responsible for predicting what the vehicle will do depending on sensor information and making decisions on how to get to the destination succesfully. Prediction can be categorized into two approaches: the model-based approach, which uses physical models to predict future behavior or the data-driven approach, which uses lots of observed behavior data to predict what will happen next.
### Behavior Planning
The process of finding a path from start location to goal location is called 'planning'. In other words, given a map of the world, a starting and ending location, and a cost function, or the time required to drive a certain route, the goal is to find the minimum cost path. The behavior planning step is responsible for providing guidance to the trajectory planner about what sorts of maneuvers they should plan trajectories for.
### Trajectory Generation
Generating a continuous path for the car to follow is the goal of this step. We need continuity in position and speed (and also acceleration and jerk). As jerk is the most noticeable factor when in the car, generating a jerk minimizing trajectory is important for a comfortable ride. 

### Implmentation
I used the spline function provided by http://kluge.in-chemnitz.de/opensource/spline/ to generate trajectories that the vehicle would be able to follow. Using a list of waypoints, I generated a path that's tangent to the angle of the car. Furthermore, transforming the world coordinates to the car's local coordinates was useful in calculating the vehicle trajectory.

Using the position of other cars in the road provided by the `sensor_fusion` data (a list of all other car's attributes on the same side of the road), I determined where the other cars are, how fast they are going, and how the host vehicle should behave as a result. To avoid hitting a car in front of us, I check whether there is a car in front of the host vehicle, and whether to change lanes or reduce speed and stay in the current lane (depending on the vehicles around us). Using Frenet coordinates was useful in determining lane position (using `d`) and lane distance (using `s`). 
