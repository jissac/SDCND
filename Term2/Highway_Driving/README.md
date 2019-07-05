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

## Prediction

## Behavior Planning
The process of finding a path from start location to goal location is called 'planning'. In other words, given a map of the world, a starting and ending location, and a cost function, or the time required to drive a certain route, the goal is to find the minimum cost path. 
## Trajectory Generation

I used the spline function provided by http://kluge.in-chemnitz.de/opensource/spline/ to generate trajectories that the vehicle would be able to follow.
