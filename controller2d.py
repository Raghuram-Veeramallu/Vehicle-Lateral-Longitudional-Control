#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        
        # PID controls
        Kp = 0.2
        Ki = 0.05
        Kd = 0.01

        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('a_previous', 0.0)
        self.vars.create_var('err_previous', 0.0)
        self.vars.create_var('integral_err_previous', 0.0)
        self.vars.create_var('steering_angle_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################

            ### Implementing a PID control

            # desired speed is the reference speed
            # no low level conroller required

            # Propotional Control (based on speed errors - to ensure direction is correct)
            # Kp * (dx_ref - dx_current)
            propotional_term = Kp * (v_desired - v)

            # Integral Control (based on accumulated path errors)
            # Ki * ((dx_ref - dx_current) * delta_t)
            integral_term = Ki * (v_desired - v) * (t - self.vars.t_previous)

            # Derivative Control (dampens the overshoot caused by integral term)
            # Kd * ((dx_ref - dx_current) / delta_t)
            derivative_term = Kd * (v_desired - v) / (t - self.vars.t_previous)

            # desired acceleration
            desired_acceleration = propotional_term + integral_term + derivative_term

            # due to simple motion in the coursework, no specific breaking condition is required

            if desired_acceleration >= 0:
                throttle = desired_acceleration
                breaking = 0
            else:
                throttle = 0
                breaking = - desired_acceleration

            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            throttle_output = throttle
            brake_output    = breaking

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################


            ### Stanley Controller
            
            # we need the trajectory line between the two waypoints
            # (x1, y1) => previous waypoint 
            # (x2, y2) => next waypoint
            # line equation => (y - y2) = m * (x - x2), where m = (y2 - y1)/(x2 - x1)
            # line equation => ax + by + c = 0 ==> (y2 - y1) x + (x1 - x2) y + (y2 * (x2 - x1) - x2 * (y2 - y1)) = 0
            # so,
            # a = y2 - y1
            # b = x1 - x2
            # c = (y2 * (x2 - x1) - x2 * (y2 - y1))

            k_e = 0.3

            x1 = waypoints[0][0]
            y1 = waypoints[0][1]
            x2 = waypoints[-1][0]
            y2 = waypoints[-1][1]

            a = y2 - y1
            b = x1 - x2
            c = (y2 * (x2 - x1) - x2 * (y2 - y1))

            # # heading error
            yaw_path = np.arctan2(a, -b)
            heading_error = yaw_path - yaw
            if heading_error > np.pi:
                heading_error -= 2 * np.pi
            if heading_error < - np.pi:
                heading_error += 2 * np.pi

            # crosstrack error
            # e = ((a * x_current) + (b * y_current) + c) / ((a^2 + b ^2) ^ 1/2)
            # e = ((a * x) + (b * y) + c)/np.sqrt((a ** 2) + (b ** 2))

            current_position = np.array([x, y])
            crosstrack_error = np.min(np.sum((current_position - np.array(waypoints)[:, :2])**2, axis=1))
            
            # yaw rate error
            yaw_cross_track = np.arctan2(y - y1, x - x1)
            yaw_path_diff = yaw_path - yaw_cross_track

            if yaw_path_diff > np.pi:
                yaw_path_diff -= 2 * np.pi
            if yaw_path_diff < - np.pi:
                yaw_path_diff += 2 * np.pi

            # crosstrack error
            crosstrack_error = np.sign(yaw_path_diff) * abs(crosstrack_error)

            # yaw crosstrack error
            yaw_crosstrack_error = np.arctan2(k_e * crosstrack_error, v)

            # steering angle from crosstrack and heading error
            steering_angle = yaw_crosstrack_error + heading_error
            if steering_angle > np.pi:
                steering_angle -= 2 * np.pi
            if steering_angle < - np.pi:
                steering_angle += 2 * np.pi

            # limiting the steering angle range to prevent harsh turnings
            steering_angle = min(1.22, steering_angle)
            steering_angle = max(-1.22, steering_angle)

            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral controller. 
            steer_output    = steering_angle

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.a_previous = desired_acceleration
        self.vars.t_previous = t
        self.vars.steering_angle_previous = steer_output

