# **Kidnapped Vehicle Project** 


### In this project, I implemented a 2 dimensional particle filter in C++. The particle filter is given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter also gets observation and control data.

### Overview
The robot has been 'kidnapped' and transported to a new location. Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data. Given this information, 

The following files are included in the 'src' directory:


### Particle Filters
Unlike the Kalman filter which is a single Gaussian, the particle filter can be used to represent multi-modal distributions. The particle filter uses particles to represent a discrete guess (X and Y coordinates, heading) as to where the robot might be. The set of several thousand of these guesses forms the filter - depending on how consistent the particles are with the sensor measurements, the correct set of particles will survive. Those thousands of particles that are then clustered together at a single location form the approximate belief of the robot as it localizes itself.
