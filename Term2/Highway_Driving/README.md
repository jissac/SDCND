# Highway Driving

### In this project, I wrote a path planning algorithm in C++ that implements SAE Level 4 functionality. 

---

**Outline**

There are three core components to path planning

* Predicting what other vehicles on the road will do next
* Deciding on what maneuvers to execute given our goals and predicitons about the other vehicles
* Building a trajectory to execute the maneuvers we decide on


An example of the final output is shown below. The code for this project is contained in the `src` directory. 

![](includeagif.gif)

[//]: # (Image References)

[image1]: ./img/output_images/cal_undist.jpg "cal_undist"
[image2]: ./img/output_images/test_undist.jpg "test_undist"
[image3]: ./img/output_images/test_binary.jpg "test_binary"
[image4]: ./img/output_images/test_roi.jpg "test_roi"
[image5]: ./img/output_i

## Path Planning
The process of finding a path from start location to goal location is called 'planning'. In other words, given a map of the world, a starting and ending location, and a cost function, or the time required to drive a certain route, the goal is to find the minimum cost path. We can formulate the path planning problem as a search problem
