#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  // PID hyperparamters
  PID::Kp = Kp_;
  PID::Ki = Ki_;
  PID::Kd = Kd_;
//   Kp = Kp_;
//   Ki = Ki_;
//   Kd = Kd_;
  
  p_error = d_error = i_error = 0.0;
  // Previous cte.
  //prev_cte = 0.0;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
  
  //prev_cte = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return -(p_error*Kp + i_error*Ki + d_error*Kd);  // TODO: Add your total error calc here!
}