#Third-Party Libraries
import os
import cv2
import numpy as np

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import to_surface_coordinates, read_transform_config, draw_circle, forkline_visible, count_people, calculate_drive_waypoint
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_image
from ultralytics import YOLO
from collections import deque


class TrolleyProblem(Node):
    def __init__(self, model_path, config_path):
        super().__init__('trolley_problem')

        # Define a message to send when the line tracker has lost track
        self.stop_msg = PoseStamped()
        self.stop_msg.pose.position.x = self.stop_msg.pose.position.y = self.stop_msg.pose.position.z = float('nan')

        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")


        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(config_path)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Publisher to send calculated waypoints
        self.publisher = self.create_publisher(PoseStamped, '/waypoint', qos_profile)
        
        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(Image, '/result', qos_profile)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        id2label = self.model.names

        ##########################                 BEGIN MODIFIED                   ##########################
        ### Specified center and car because that's all we need for ACC.
        targets = ['dashed_line', 'left_side', 'people', 'right_side'] # Classes to track and publish as /object PoseStamped messages
        
        self.id2target = {id: lbl for id, lbl in id2label.items() if lbl in targets}

        ### Need this from the midterm colab to differentiate between objects!!!
        self.label2id = {label: id for id, label in id2label.items() if label in targets}
        ##########################                  END MODIFIED                    ##########################

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Trolley Problem Node started. Custom YOLO ONNX model loaded successfully.")

        self.way_cx = 640
        self.way_cy = 360
        self.x = 0.0
        self.y = 0.0
        self.serial_killer_mode = False
        self.filter_size = 3
        self.y_queue = deque([], maxlen=self.filter_size)


    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args['imgsz'] # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        print("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im, verbose = False)  
        print("Model initialization complete.")

        return model

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)


        # Run YOLO inference
        predictions = self.model(image)
        
        # Hard coded this because onnx is different from pt.
        screen_x_center = 320

        # Draw results on the image
        plot = predictions[0].plot()
        
        # get the dashed line centroid
        fork_success, cx, cy = forkline_visible(predictions)

        # Flag telling the car whether it should continue straight or turn left
        go_straight = True
        send_waypoint = False
        # If we can see the fork, we need to determine whether or not to continue straight or fork
        if fork_success: # forkline is visble

            # Boolean to determine if the fork is on the left or right of us
            fork_on_right = False

            # If the dashed line is on the right side of the screen, assume fork is on the right
            # otherwise it is on the left
            if cx > screen_x_center:
                fork_on_right = True
            # Count the number of people on the screen and get their centroids in the x
            centers = count_people(predictions)
            if len(centers) == 0: # no people
                pass
            # if we have people...
            # 1) count the number of people to the left and right of the dashed line
            # 2) use this to determine if we go straight or fork
            #   a) if there more people on the left than right and the dashed line is on the right,
            #      we need to fork to hit less people
            #   b) if there more people on the right than left and the dashed line is on the right,
            #      continue straight by following the line
            #   c) if there more people on the left than right and the dashed line is on the left,
            #      continue straight by following the line 
            #   d) if there more people on the right than left and the dashed line is on the left,
            #      we need to fork to hit less people
            # Default for the trolley problem, we should just go straight (something something pulling
            # the lever) and so if we have equal numbers of people we assunme going straight
            else:
                cnt_left = np.sum(centers < cx)
                cnt_right = np.sum(centers >= cx)
                self.get_logger().info(f"{cnt_left}vs{cnt_right}")
                if cnt_left > cnt_right and fork_on_right:
                    go_straight = False
                elif cnt_left <= cnt_right and fork_on_right:
                    go_straight = True
                elif cnt_left >= cnt_right and not fork_on_right:
                    go_straight = True
                else:
                    go_straight = False

        # Fun thing, if we want to be a bit homicidal we can just tell the car to do the opposite of 
        # the above logic
        if self.serial_killer_mode:
            go_straight = not go_straight

        # Get the waypoint based on the lane lines
        way_success, way_cx, way_cy = calculate_drive_waypoint(predictions)

        # If we can see both the lane lines and the fork, do whatever the above logic suggests
        if way_success and fork_success:
            if go_straight:
                self.way_cx = way_cx
                self.way_cy = way_cy
                self.get_logger().info("FOLLOWING TRACK")
                plot = draw_circle(plot, self.way_cx, self.way_cy)
            else:
                self.way_cx = cx
                self.way_cy = cy
                self.get_logger().info("FORKING")
                plot = draw_circle(plot, self.way_cx, self.way_cy, color=(255, 0, 0))
            self.x, self.y = self.to_surface_coordinates(self.way_cx, self.way_cy)
            send_waypoint = True

        # If we can only see the lane lines just follow them
        elif way_success and not fork_success:
            self.way_cx = way_cx
            self.way_cy = way_cy
            self.get_logger().info("FOLLOWING TRACK")
            plot = draw_circle(plot, self.way_cx, self.way_cy)
            self.x, self.y = self.to_surface_coordinates(self.way_cx, self.way_cy)
            send_waypoint = True

        # Weird case that came up in testing.  If we see no lane lines but we do see 
        # both the dashed line and people, use the logic only to follow the fork.
        # Otherwise assume we cant see anything
        elif not way_success and fork_success and len(centers) > 0 and not go_straight:
            way_cx = cx
            way_cy = cy
            self.get_logger().info("FORKING")
            plot = draw_circle(plot, way_cx, way_cy, color=(0, 0, 255))
            self.x, self.y = to_surface_coordinates(way_cx, way_cy)
            send_waypoint = True
        else:
            self.publisher.publish(self.stop_msg)
            self.get_logger().info("Lost track!")
        
        if send_waypoint:
            self.y_queue.append(self.y)
            self.y = np.median(self.y_queue)
            timestamp = msg.header.stamp
            pose_msg = np_to_pose(np.array([self.x, self.y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)
            self.get_logger().info(f"steering angle = {-self.y}")
        # Convert back to ROS2 Image and publish
        im_msg = np_to_image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

        # Publish predictions
        self.im_publisher.publish(im_msg)


def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    model_path = pkg_path + '/models/best.pt'
    rclpy.init(args=args)
    node = TrolleyProblem(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()