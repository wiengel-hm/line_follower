import rclpy
from rclpy.node import Node

import numpy as np
import time
from collections import deque
from ament_index_python.packages import get_package_prefix
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import Int16MultiArray
from ackermann_msgs.msg import AckermannDriveStamped

import os
from geometry_msgs.msg import PoseStamped

from ros2_numpy import pose_to_np, to_ackermann
from nav_msgs.msg import Path
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
# from smart_parking.utils import get_mapping, replay_bagfile

from odometry.call_service import call_reset_odometry_service  

''' NavigationNode Summary

Inputs:
- Int16MultiArray message from sensor arrays (Camera, and Ultrasonic)
- PoseStamped message of ball and goal position (received from internal function)

Outputs:
- PoseStamped Message of ball and goal position
- AckermannDriveStamped Message to PIDcontrollerNode

Features:
- Tracks ball position



''' 

class NavigationNode(Node):
    def __init__(self, bagfile_path):
        super().__init__('navigation_node')

        if not os.path.exists(bagfile_path):
            raise FileNotFoundError(f"Bagfile not found: {bagfile_path}")

        self.bagfile_path = bagfile_path

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # --- ROS Subscribers ---
        self.create_subscription(PoseStamped, '/position', self.location_callback, qos_profile)
        self.create_subscription(Int16MultiArray, '/uss_sensors', self.uss_callback, qos_profile)
        self.create_subscription(AckermannDriveStamped, '/rc/ackermann_cmd', self.ackermann_callback, qos_profile)
        self.create_subscription(Joy, '/rc/joy', self.joy_callback, qos_profile)
        self.obj_sub = self.create_subscription(PoseStamped, '/object', self.location_callback, qos_profile)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan',self.lidar_callback, qos_profile)

        
        # --- ROS Publisher ---
        self.autonomous_pub = self.create_publisher(AckermannDriveStamped, '/autonomous/ackermann_cmd', qos_profile)
        self.strike_pub = self.create_publisher(PoseStamped, '/strike', qos_profile)
        self.status_pub = self.create_publisher(PoseStamped, '/navigation/status', qos_profile)


        # --- Program States ---
        self.POSITIONING = 0
        self.BALL_FOLLOW = 1
        self.SCANNING = 2
        self.HIT_TOWARD_GOAL = 3

        # --- Ball Positions Relative to Goal ---
        self.LEFT = 0
        self.IN_LINE = 1
        self.RIGHT = 2

        self.ball_wrt_goal = None

        # --- Mapping Spacing Distances ---
        '''
        Based on a subtraction of goal position - ball position. Positive numbers mean ball is left of goal,
        Negative numbers mean ball is right of goal. Boundaries are set up to decide which direction to hit.
        '''
        self.left_reqd_spacing = -10 # EDIT: change to pixels to boundary in captured image
        self.right_reqd_spacing = 10 #

        # --- Mode Initialization ---
        self.mode = None
        self.status = self.POSITIONING

        self.brake = lambda: self.autonomous_pub.publish(to_ackermann(.0, .0)) 

        self.get_logger().info("Navigation Node has been started.")

        # --- Object Detection Initialization
        self.len_history = 10
        self.max_out = 3
        self.ball_success = deque([False] * self.max_out, maxlen=self.max_out)
        self.goal_success = deque([False] * self.max_out, maxlen=self.max_out)

        # --- INitialize Positions to clear ---
        self.ball_x = 100
        self.ball_y = 100
        self.goal_x = 100
        self.goal_y = 100

        # TODO: add the correct object labels for this 
        self.id2target = {0: 'ball', 1: 'goal'}

        self.target= {
            lbl: {
                'id': id, 
                'history': deque([False] * self.len_history, maxlen = self.len_history)
            }
            for id, lbl in self.id2target.items()
            }

        self.index2frame = {
        0: 'USS_SRF',  # Side Right Front
        1: 'USS_SRB',  # Side Right Back
        2: 'USS_BR"',  # Back Right
        3: 'USS_BC',   # Back Center
        4: 'USS_BL',   # Back Left
        5: 'USS_SLB',  # Side Left Back
        6: 'USS_SLF',  # Side Left Front
        7: 'USS_FL',   # Front Left
        8: 'USS_FC',   # Front Center
        9: 'USS_FR'    # Front Right
    }


    def ackermann_callback(self, msg: AckermannDriveStamped):
        return
    def joy_callback(self, msg: Joy):
        return

    def lidar_callback(self, msg: LaserScan):
        # msg.ranges is a list of float distances in meters (length = num_beams)
        ranges = np.array(msg.ranges)
        valid = np.isfinite(ranges) & (ranges > 0.02)  # remove NaNs, zero/negative
        if np.any(valid):
            min_dist = np.min(ranges[valid])
            avg_dist = np.mean(ranges[valid])
            self.get_logger().info(f"[LiDAR] Min: {min_dist:.2f} m | Avg: {avg_dist:.2f} m")
        else:
            self.get_logger().info("[LiDAR] No valid range data.")

    def uss_callback(self, msg: Int16MultiArray):
        # msg.data = [sensor0, sensor1, ..., sensor9] in centimeters
        msg_trimmed = np.array(msg.data)
        uss_distances_m = msg_trimmed / 100.0  # convert to meters

        # Ignore invalid readings (e.g., zero or negative)
        valid = uss_distances_m > 0.02
        if np.any(valid):
            # Find index of minimum (closest) valid value
            # Note: np.argmin gives the first min if multiple; that's standard
            min_distance = np.min(uss_distances_m[valid])
            # Get the original index in the msg.data array
            min_index = np.where(uss_distances_m == min_distance)[0][0]
            sensor_name = self.index2frame.get(min_index, str(min_index))
            valid_sensors = ['USS_SLF', 'USS_FL', 'USS_FC', 'USS_FR', 'USS_SRF']
            if sensor_name in valid_sensors:
                #self.get_logger().info(
                #    f"[Ultrasonic] Closest: {min_distance:.2f} m (Sensor {sensor_name}, index {min_index})"
                #)
                self.trim_min_index = min_index
                self.trim_sensor_name = sensor_name
                self.trim_min_distance = min_distance

            # Now you can use min_distance and min_index in your control logic!
        #else:
            #self.get_logger().info("[Ultrasonic] No valid readings.")

    def location_callback(self, msg: PoseStamped):
    # TODO: check to see what the incoming heading data type is. Is it one value, and can be an integer? is it xyz headings?
        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)
        id = int(point[-1]) # Extracts the last coordinate, which we know to be z coordinate, containing ID

        label = self.id2target[id]
        data = self.target[label]

        # Positive Heading means object is to left of car, negative is to right of car (pi --> -pi)
        for lbl, dta in self.target.items():
            if np.all(np.array(dta['history']) == True) and len(np.array(dta['history'])) >= self.len_history:
                if lbl == 'ball':
                    if label == 'ball':
                        self.ball_heading = np.arctan2(point[1], point[0])
                        ball_x = float(point[0]/100) # extracts first coordinate, which we know to be x pos
                        ball_y = float(point[1]/100) # extracts second coordinate, which we know to be y pos
                        self.get_logger().info(f"ball_x: {ball_x}")
                        self.get_logger().info(f"ball_heading: {self.ball_heading}")
                        self.update_state(self.ball_heading, lbl, ball_x, ball_y)

                elif lbl == 'goal':
                    if label == 'goal':
                        self.goal_heading = np.arctan2(point[1], point[0])
                        goal_x = float(point[0]/100) # extracts first coordinate, which we know to be x pos
                        goal_y = float(point[1]/100) # extracts second coordinate, which we know to be y pos
                        self.get_logger().info(f"goal_x: {goal_x}")
                        self.get_logger().info(f"goal_heading: {self.goal_heading}")
                        self.update_state(self.goal_heading, lbl, goal_x, goal_y)
                else:
                    self.update_state(float('nan'), float('nan'), float('nan'), float('nan'))

        if np.isnan(point).any():
            data['history'].append(False)
        else:
            data['history'].append(True)

    def update_state(self, heading, label, x, y):
        self.status_msg = PoseStamped()
        self.status_msg.pose.position.z = float(self.status)
        self.status_pub.publish(self.status_msg)

        if self.status == self.POSITIONING:
            '''
            We understand positioning is a difficult task if ball is not in a favorable position.
            So, we are allowing ourselves to take favorable starting position, but we are 
            creating the architecture for a future advanced positioning setup.

            '''
            self.get_logger().info("Positioning in line with ball and goal")
            self.status = self.BALL_FOLLOW
        
        if self.status == self.BALL_FOLLOW:
            if label == 'ball':
                x_pos = x
                y_pos = y
                distance_to_ball = np.sqrt(x_pos ** 2 + y_pos ** 2)
                self.get_logger().info(f"Following ball: Current Distance: {distance_to_ball}")

                if distance_to_ball < 0.175:
                    self.get_logger().info("Ball is close. Switching to scanning phase.")
                    self.status = self.SCANNING


        if self.status == self.SCANNING:
            self.get_logger().info("Scanning state entered, braking while searching")
            self.brake()

            # Log each location append since we entered the scanning state
            if label == 'ball':
                self.ball_success.append(True)
                self.theta_1 = heading
                self.get_logger().info(f"Successful ball confirm")

            if label == 'goal':
                self.goal_success.append(True)
                self.theta_2 = heading
                self.get_logger().info(f"Successful goal confirm")


            # Check to see if three successful location appends have been made while in the scanning state. If so, we can
            # say that the car has been stopped and has an accurate idea of the location of the ball
            if np.all(np.array(self.ball_success) == True) and np.all(np.array(self.goal_success) == True):
                self.get_logger().info(f"Confirmed Location of Goal and Ball. Theta 1 = {self.theta_1}, Theta 2 = {self.theta_2}")
                # Make decision about where ball is in reference to goal
                # Equation: angular difference = goal angle - ball angle
                self.delta_theta = self.theta_2 - self.theta_1
                self.delta_theta_deg = np.rad2deg(self.delta_theta)
                self.get_logger().info(f"DTheta DEG {self.delta_theta_deg}")

                # LEFT
                if self.delta_theta_deg <= self.left_reqd_spacing:
                    self.get_logger().info(f"Estimated Angular Separation = {self.delta_theta_deg} . Ball is left of goal.")
                    self.ball_wrt_goal = self.LEFT

                # CENTER 
                elif self.delta_theta_deg < self.right_reqd_spacing and self.delta_theta_deg > self.left_reqd_spacing:
                    self.get_logger().info(f"Estimated Angular Separation = {self.delta_theta_deg} . Ball in line with goal.")
                    self.ball_wrt_goal = self.IN_LINE

                # RIGHT
                else: 
                    self.get_logger().info(f"Estimated Angular Separation = {self.delta_theta_deg} . Ball is right of goal.")
                    self.ball_wrt_goal = self.RIGHT
                # Decision made, returning decision and changing state. Resets success in case we re-enter scanning
                self.ball_success.clear()
                self.ball_success.extend([False] * self.max_out)

                self.goal_success.clear()
                self.goal_success.extend([False] * self.max_out)

                self.ball_strike_msg = PoseStamped()
                self.ball_strike_msg.pose.position.x = float(self.ball_wrt_goal)
                self.strike_pub.publish(self.ball_strike_msg)
                self.status = self.HIT_TOWARD_GOAL

        if self.status == self.HIT_TOWARD_GOAL:
            self.get_logger().info(f"Strike Command Sent")
            time.sleep(1)
            self.brake()
            self.ball_success.clear()
            self.goal_success.clear()
            self.ball_success = deque([False] * self.max_out, maxlen=self.max_out)
            self.goal_success = deque([False] * self.max_out, maxlen=self.max_out)

            self.status = self.POSITIONING
            self.get_logger().info(f"Heading back to positioing state!")

def main(args=None):
    rclpy.init(args=args)

    # Not sure if below line is necessary
    pkg_dir = get_package_prefix('rocket_league').replace('install', 'src') #  /mxck2_ws/install/smart_parking → /mxck2_ws/src/smart_parking
    
    # NEED TO INSERT BAGFILE PATH HERE FOR TEST
    bagfile_path = pkg_dir + '/bagfiles/'
    
    node = NavigationNode(bagfile_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Navigation Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()