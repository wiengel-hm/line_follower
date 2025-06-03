# Third-Party Libraries
import os
import cv2
import numpy as np

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import to_surface_coordinates, read_transform_config, parse_predictions, get_base, detect_bbox_center, draw_circle
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_image
from ultralytics import YOLO

class LineFollower(Node):
    def __init__(self, model_path, config_path):
        super().__init__('line_tracker')

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

        # Publisher to send 3d object positions
        self.obj_publisher = self.create_publisher(PoseStamped, '/moose_sign', qos_profile)
        
        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(Image, '/result', qos_profile)

        #Publisher to send value representing which line is detected for visualization
        self.offset_publisher = self.create_publisher(Int16, '/offset', qos_profile)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        #Confidence threshhold for identfying center lines
        self.center_thresh = 0.5

        #Total width of center track (in meters)
        self.lane_width = 0.85

        #Intialize seen both lines variable
        self.seen_both = False
        self.last_offset = 0

        # Map class IDs to labels and labels to IDs
        id2label = self.model.names
        label2id = {label: id for id, label in id2label.items()}
        self.center_id = label2id['center']
        self.moose_sign_id = label2id['moose_sign']
        self.get_logger().info(f'Class Ids: {self.center_id=} {self.moose_sign_id=}')
        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO ONNX model loaded successfully.")

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
        
        #Extract timestamp from message
        timestamp = msg.header.stamp

        ### A NOTE ABOUT OFFSET MESSAGES
        #We publish an integer to the /offset whenever we set an offset for the car
        #0: Both center lines are detected so no offset is applied
        #1: Only left line was detected and appropriate offset was applied
        #-1: Only right lane was detected and appropraite offset was applied


        # Run YOLO inference
        predictions = self.model(image)

        p = predictions[0]

        bboxs = p.boxes
        ids = bboxs.cls.cpu().numpy()        # Class IDs e.g., center(0), stop(1)
        confidences = bboxs.conf.cpu().numpy() # Confidence scores (not used here)


        
        #p = predictions # move to CPU
        #ids = p.boxes.cls.numpy() # Class IDs
        #confidences = p.boxes.conf.numpy() #Confidences
        #Discard ids with low confidences
        ids = np.array([pred for pred, conf in zip(ids, confidences) if conf >= self.center_thresh])
        self.get_logger().info(f'{ids=}')

        #Get mask as usual
        success, mask = parse_predictions(predictions, self.center_id)

        if success:
            #Get bottom center of road mask
            cx, cy = get_base(mask)
            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)
            #Did we detect more than 1 line?
            offset_msg = Int16()
            if np.sum(ids == self.center_id) < 2: #If less than 2 lines detected, determine if line is on left or right of frame and add apropriate offset
            #More white pixels on left or right?
                left = mask[:,:320]
                right = mask[:,320:]
                if not self.seen_both:
                    self.get_logger().info('Applying last offset')
                    y = y+self.last_offset
                elif np.sum(left)>np.sum(right):
                    y = y+self.lane_width/2
                    self.last_offset = self.lane_width/2
                    self.seen_both = False
                    self.get_logger().info(f'Only left line detected. Offset {y}')
                    offset_msg.data = 1
                    self.offset_publisher.publish(offset_msg)
                else:
                    y = y-self.lane_width/2
                    self.last_offset = -self.lane_width/2
                    self.seen_both = False
                    self.get_logger().info(f'Only right line detected. Offset {y}')
                    offset_msg.data = -1
                    self.offset_publisher.publish(offset_msg)  
            else:
                self.get_logger().info(f'No offset {y}')
                self.seen_both = True
                offset_msg.data = 0
                self.offset_publisher.publish(offset_msg)          
            #Publish potentially offset coordinates
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
        else:
            self.get_logger().info('Lost Track')
            self.seen_both = True
            pose_msg = self.stop_msg
        self.publisher.publish(pose_msg)

        # Draw results on the image
        plot = predictions[0].plot()

       
        # Compute 3D positions of all targets (In this case, just the moose sign)
        detected, u, v = detect_bbox_center(predictions, self.moose_sign_id)
        if detected:
            # Draw a circle at the bottom center of the bounding box
            plot = draw_circle(plot, u, v)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(u, v)

            # Publish object as Pose message
            pose_msg = np_to_pose(np.array([x, y, self.moose_sign_id]), 0.0, timestamp=timestamp)
            self.get_logger().warn('KILLER MOOSE IN THE AREA')
        else:
            pose_msg = self.stop_msg # Attention: this creates a reference — use deepcopy() if you want self.stop_msg to remain unchanged
            pose_msg.pose.position.z = float(self.moose_sign_id)

        self.obj_publisher.publish(pose_msg)

       
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
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()