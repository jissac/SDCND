/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::discrete_distribution;
int NUM_PARTICLES = 100;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = NUM_PARTICLES;  // Set the number of particles
  weights.resize(num_particles); // initialize weights variable
  default_random_engine rand_gen;
  
  // Define normal distributions
  normal_distribution<double> std_x_init(x,std[0]);
  normal_distribution<double> std_y_init(y,std[1]);
  normal_distribution<double> std_yaw_init(theta,std[2]);
  
  // Initialize all particles to Gaussian distribution around first position
  for (int i=0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = std_x_init(rand_gen);
    particle.y = std_y_init(rand_gen);
    particle.theta = std_yaw_init(rand_gen);
    particle.weight = 1;
    
    // append to set of current particles
    particles.push_back(particle);
  }                            
  is_initialized = true;
  //cout << "init OK"<<endl;
} 

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine rand_gen;
  // normal distributions
  normal_distribution<double> std_x(0,std_pos[0]);
  normal_distribution<double> std_y(0,std_pos[1]);
  normal_distribution<double> std_yaw(0,std_pos[2]);
  // find new position of car
  for (int i=0;i<num_particles;i++) {//cout << "predict.2 OK"<<endl;
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity*delta_t*cos(particles[i].theta);
      particles[i].y += velocity*delta_t*sin(particles[i].theta);
    } else {//cout << "predict.3 OK"<<endl;
      particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    }
    // add noise
    particles[i].x += std_x(rand_gen);
    particles[i].y += std_y(rand_gen);
    particles[i].theta += std_yaw(rand_gen);
  }
  //cout << "predict OK"<<endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
  int num_pred = predicted.size();
  int num_obs = observations.size();
  // for each observation, check what is the closest predicted measurement 
  // using Euclidean distance
  for(int i=0;i<num_obs;i++){
    LandmarkObs obs = observations[i];
    LandmarkObs pred = predicted[0];
    double min_dis = dist(obs.x, obs.y, pred.x, pred.y);
    observations[i].id = pred.id;
    for(int j=1;j<num_pred;j++){
      LandmarkObs pred = predicted[j];
      double dis = dist(obs.x, obs.y, pred.x, pred.y);
      if(dis< min_dis){
        min_dis = dis;
        observations[i].id = pred.id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  //cout<<"updateWeight0 OK"<<endl;
  // Landmark measurement uncertainty [x [m], y [m]]]
  double std_landmark_x = std_landmark[0];
  double std_landmark_y = std_landmark[1];
  
  // Define multi-variate Gaussian constant terms for later
  const double MVG_const1 = 1/(2*M_PI*std_landmark_x*std_landmark_y);
  
  // for each particle...
  for (int i = 0; i<num_particles; ++i) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    // find the nearest landmark
    vector<LandmarkObs> nearestLandmarks;
    for(unsigned int k=0;k<map_landmarks.landmark_list.size();++k) { 
      float landmark_x = map_landmarks.landmark_list[k].x_f;
      float landmark_y = map_landmarks.landmark_list[k].y_f;
      int landmark_mapId = map_landmarks.landmark_list[k].id_i;
      double dist_landmark= dist(landmark_x,landmark_y,p_x,p_y);
      // is the landmark's distance within the vehicle sensor's range?
      if (dist_landmark <= sensor_range) {
        LandmarkObs obs;
        obs.id = landmark_mapId;
        obs.x = landmark_x;
        obs.y = landmark_y;
        // if so, add the landmark to the list of nearest landmarks
        nearestLandmarks.push_back(obs);
      } 
    } // now we have the list of the nearest observed landmarks to the particle
    // next, transform each observation
    vector<LandmarkObs> transformedObservations;
    for (int j = 0; j< observations.size(); ++j) {
      // transform each observation from vehicle coordinates to map coordinates
      int obs_id = observations[j].id;
      float obs_x = observations[j].x;
      float obs_y = observations[j].y;
      double transformed_x = p_x + cos(p_theta)*obs_x - sin(p_theta)*obs_y;
      double transformed_y = p_y + sin(p_theta)*obs_x + cos(p_theta)*obs_y;
      transformedObservations.push_back(LandmarkObs{obs_id,transformed_x,transformed_y});
    }
    // Now that the vehicle observations have been transformed into the map's coordinate space, 
    // the next step is to associate each transformed observation with a land mark identifier
    dataAssociation(nearestLandmarks,transformedObservations); 
    
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    particles[i].weight = 1.0;
    for (unsigned int ii=0;ii<transformedObservations.size();++ii) {
      //cout<<"mvg for loop OK"<<ii<<endl;
      // coordinates of observed landmarks 
      double x = transformedObservations[ii].x;
      double y = transformedObservations[ii].y;
      int obs_id = transformedObservations[ii].id; //associated id
      // placeholder for coordinates of nearest landmarks
      double mew_x; //
      double mew_y;// 
      // find the x,y coordinates of the nearest landmark to the associated observation
      for (unsigned int iii=0;iii<nearestLandmarks.size();++iii) {
        //cout<<"nearestlandmark association ok"<<iii<<endl;
        if (nearestLandmarks[iii].id == obs_id) {
          mew_x = nearestLandmarks[iii].x;
          mew_y = nearestLandmarks[iii].y;
        }
      }
      // exponent terms of MVG
      double exp_x = pow((x-mew_x),2) / (2*pow(std_landmark_x,2));
      double exp_y = pow((y-mew_y),2) / (2*pow(std_landmark_y,2));
      // MVG
      double weight;
      weight = MVG_const1*exp(-(exp_x+exp_y));
      if (weight == 0) {
        particles[i].weight  *= -.000001;
      }else {
        particles[i].weight  *= weight;
      }
      associations.push_back(obs_id);
      sense_x.push_back(x);
      sense_y.push_back(y);
    }
    SetAssociations(particles[i],associations,sense_x,sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // populate weights
  weights.clear();
  for(int i=0; i<num_particles; ++i){
    weights.push_back(particles[i].weight);
  }
  
  // Probability distributions based on particle weights
  discrete_distribution<int> part_distribution(weights.begin(),weights.end());  
  
  // Use calculated weights to update particles to Bayesian posterior distribution
  vector <Particle> resampled_particles;
  resampled_particles.resize(num_particles);
  for(int i=0;i<num_particles;++i) {
    resampled_particles[i] = particles[part_distribution(gen)];
  }  
  // replace old particles
  particles = resampled_particles;
 // cout << "resample OK"<<endl;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}