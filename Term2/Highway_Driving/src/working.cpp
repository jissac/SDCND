#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

// start in lane 1
int lane = 1; // 0: far left, 1: middle, 2: far right

// Have a reference velocity (mph) to target
double ref_vel = 0.0; //mph

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }
  
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
          
          
//////////////////////// 
          
          // what the previous path size was
          int prev_size = previous_path_x.size();
          
          // ## Collision avoidance using sensor fusion data ## //
          if(prev_size >0)
          {
            car_s = end_path_s;
          }
          
          bool too_close = false;
          
          // find ref_vel to use
          for(int i = 0; i <sensor_fusion.size(); i++)
          {
            // car is in my lane
            float d = sensor_fusion[i][2+4*lane];
            if(d < (2+4*lane+2) && d > (2+4*lane-2))
            {
              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];
              double check_speed = sqrt(vx*vx+vy*vy);
              double check_car_s = sensor_fusion[i][5];
              
              // if using previous points we can project s values outwards in time
              check_car_s += ((double)prev_size*0.02*check_speed);
              // check if our car's s is close to the other car's s 
              if ((check_car_s > car_s) && ((check_car_s-car_s) <30))
                  {
                    // if our car in the future is going to be within 30 m of the 
                    // other car in the future, lower speed or change lanes, etc.
                    // ref_vel = 29.5; // mph
                    too_close = true;
                    // add logic to change lanes (left, right, stay) if too close to another vehicle
                
                    if(lane > 0)
                    {
                      lane = lane-1;
                    }
                  }
            }
          }
          
          if (too_close)
            // reduce speed
          {
            ref_vel -= 0.224;
          }
          else if(ref_vel < 49.5)
            // speed up
          {
            ref_vel += 0.424;
          }
                  
          // ## Path planning ## //       
          // create a liste of widely spaced (x,y) waypoints, that are evenly spaced
          // at 30m. These wapoints will be interpolated with a spline and filled in 
          // with more points that control speed
          vector<double> x_pts;
          vector<double> y_pts;
          
          // reference x,y, and yaw states
          // the starting point will be defined as where the car is or at the previous path's end-point
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);
          
          // what is previous path size, if empty use the car as starting refernence
          if(prev_size < 2)
          {
            // use two points that make the path tangent to the car
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);
            
            //generate two points to make sure that path is tangent to car
            x_pts.push_back(prev_car_x);
            x_pts.push_back(car_x);
            
            y_pts.push_back(prev_car_y);
            y_pts.push_back(car_y);
          }
          // use the previous path's end points as the starting reference
          // what are the last couple of points car was following, and then
          // calculating the angle the car was heading in using the last couple of points
          else
          {
            //redifine the reference state as previous paht end point
            ref_x = previous_path_x[prev_size-1];
            ref_y = previous_path_y[prev_size-1];
            
            double ref_x_prev = previous_path_x[prev_size-2];
            double ref_y_prev = previous_path_y[prev_size-2];
            ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);
            
            // use two points that make the path tangent to the previous path's end points
            x_pts.push_back(ref_x_prev);
            x_pts.push_back(ref_x);
            
            y_pts.push_back(ref_y_prev);
            y_pts.push_back(ref_y);
          }
          
          // In Frenet coordinates, add evenly spaced points 30m ahead of the starting reference
          vector<double> next_wp0 = getXY(car_s+30, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s+60, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s+90, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          
          x_pts.push_back(next_wp0[0]);
          x_pts.push_back(next_wp1[0]);
          x_pts.push_back(next_wp2[0]);

          y_pts.push_back(next_wp0[1]);
          y_pts.push_back(next_wp1[1]);
          y_pts.push_back(next_wp2[1]);
          
          // transformation to local car's coordinates
          for (int i =0; i < x_pts.size(); i++)
          {
            // shift car's reference angle to 0 degrees
            double shift_x = x_pts[i]-ref_x;
            double shift_y = y_pts[i]-ref_y;
            
            x_pts[i] = (shift_x *cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
            y_pts[i] = (shift_x *sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
          }
          
          // create a spline
          tk::spline s;
          
          // set (x,y) points to the spline
          s.set_points(x_pts,y_pts);
//////////////////////// 
          
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */
          
          // start with all of the previous path points
          for(int i =0; i < previous_path_x.size(); i++)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }
          
          // calculate how to break up spline points so that car travel's at desired velocity
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));
          
          double x_add_on = 0;
          
          // fill up the rest of the path planner after filling it with previous points
          for (int i =1; i <= 50-previous_path_x.size(); i++)
          {
            double N = (target_dist/(.02*ref_vel/2.24));
            double x_point = x_add_on+(target_x)/N;
            double y_point = s(x_point);
            
            x_add_on = x_point; // adding points that are along spline
            
            double x_ref = x_point;
            double y_ref = y_point;
            
            // rotate from local to world coordinates
            x_point = (x_ref *cos(ref_yaw)-y_ref*sin(ref_yaw));
            y_point = (x_ref *sin(ref_yaw)+y_ref*cos(ref_yaw));
            
            x_point += ref_x;
            y_point += ref_y;
            
            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }
          
//           double dist_inc = 0.3;
//           for(int i=0;i<50;i++) {
//             double next_s = car_s+(i+1)*dist_inc;
//             double next_d = 6;
//             vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
//             next_x_vals.push_back(xy[0]);
//             next_y_vals.push_back(xy[1]);

//           }
          
          json msgJson;
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}