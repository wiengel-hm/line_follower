# Third-Party Libraries
import os
import cv2
import numpy as np

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from vision_msgs.msg import Detection3DArray, LabelInfo
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import (
    to_surface_coordinates, read_transform_config, parse_predictions,
    get_base, draw_circle, get_bounding_boxes, pixels_in_box, display_distances
)
from ros2_numpy import image_to_np, np_to_compressedimage, scan_to_np, to_detection3d_array, to_label_info
from ultralytics import YOLO
from line_follower.fusion import LidarToImageProjector

class LineFollower(Node):
    def __init__(self, model_path, config_path):
        super().__init__('line_tracker')

        # Check if both the model and config paths exist
        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")

        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(config_path)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message


        # Publisher to send out 3D detections
        # All detections (e.g., stop signs, waypoints) are published as a Detection3DArray.
        # Each object is represented as a Detection3D message with a label (e.g., 'center'),
        # a confidence score (e.g., 0.99), and a 3D pose (x, y, z).
        # Optionally, it can include orientation and 3D bounding box info.
        self.detection3d_pub = self.create_publisher(
            Detection3DArray,
            '/objects_3d',
            qos_profile
        )

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Subscriber to receive LiDAR scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback, 
            qos_profile
        )

        # QoS profile that ensures the last published message is saved and sent to new subscribers.
        # This is useful for static or infrequently changing data like label maps or calibration info.
        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.label_pub = self.create_publisher(
            LabelInfo, # {0: 'car', 1: 'center', ...}
            '/label_mapping',
            qos_transient
        )

        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(CompressedImage, '/result', qos_profile)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        self.id2label = self.model.names # {0: 'car', 1: 'center', ...}
        self.label2id = {lbl: id for id, lbl in self.id2label.items()} # {'car: 0, 'center': 1, ...}

        info_msg = to_label_info(self.id2label) # Convert it to a LabelInfo message
        self.label_pub.publish(info_msg) # Publish it

        # Initialize default detections with one entry per class the model can detect.
        # If nothing is visible for a class, it still gets included with a score of 0.0.
        # So if the model knows 8 classes, this dictionary will always have 8 entries.
        # Format: {id: [label, score, x, y, z]}
        # Example:
        # self.detections = {
        #     0: ['car', 0.98, 0.0, 0.0, 0.0],
        #     1: ['center', 0.0, 0.0, 0.0, 0.0], # not detected
        #     ...
        # }
        self.detections = {id: [lbl, 0.0, 0.0, 0.0, 0.0] for id, lbl in self.id2label.items()}

        # Define the classes we want to process based on how we estimate their 3D position.
        # We use two different techniques for this: 
        # - Homography (for flat objects on the ground like lane lines)
        # - Camera-LiDAR fusion (for 3D objects like cars or stop signs)

        # Classes processed using homography.
        # This only works for flat objects on the track surface, like the center line.
        objects_2d = ['center']
        self.objects_2d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_2d}

        # Classes processed using camera-LiDAR fusion.
        # This method is used for tall 3D objects like cars or stop signs.
        # It uses a projector that transforms the LiDAR point cloud into the camera frame
        # and projects the 3D points onto the image to estimate object distance and position.
        # Make sure objects are tall enough — our single-layer LiDAR may miss low objects.
        objects_3d = ['car']  # Add other 3D objects like 'stop_sign' if needed
        self.objects_3d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_3d}

        # Stores the latest 3D LiDAR points in vehicle coordinates (x: forward, y: left, z: up)
        self.pts = None

        # LiDAR-to-camera projection tool used for camera-LiDAR fusion
        self.projector = LidarToImageProjector()

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO model loaded successfully.")

    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args['imgsz'] # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        print("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im, verbose = False)  
        print("Model initialization complete.")

        return model
    
    def scan_callback(self, msg):
        # Convert the incoming LiDAR scan message to NumPy format (x, y, intensity)
        xyi, timestamp_unix = scan_to_np(msg)
        x, y, intensity = xyi[:, 0], xyi[:, 1], xyi[:, 2]

        # Store the 3D point cloud in vehicle coordinates.
        # Since we use a 2D LiDAR (single layer), all z-values are set to 0.
        self.pts = np.column_stack([x, y, np.zeros_like(x)])  # Shape: (N, 3)

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Create a copy of the default detection list to reset or start fresh
        # Format: [label, score, x, y, z] — matches Detection3DArray format
        detections = self.detections.copy()

        # Run YOLO inference
        predictions = self.model(image)

        # If there are no predictions, skip processing this frame
        if len(predictions) == 0:
            return

        # Draw results on the image
        plot = predictions[0].plot()

        """ STEP 1: 3D Object Detection (Camera-LiDAR Fusion)
        Detect tall objects like cars or signs using LiDAR projected into the camera image.
        Good for 3D objects that can be hit by the LiDAR beam. """

        success, boxes = get_bounding_boxes(predictions, self.objects_3d)
        
        # Proceed only if an object was successfully detected
        # and LiDAR data has been received
        if success and self.pts is not None:

            
            # Project 3D points to 2D image coordinates
            pixels, depth, x_values, y_values = self.projector.project_points_to_image(self.pts)

            distance_dict = {}  # Store distances per label for optional visualization

            for id, box in boxes.items():
                # Get the bounding box corners (x1, y1, x2, y2)
                corners = box['corners']
                label = box['label']
                score = box['score']

                # Check which projected LiDAR points fall inside the bounding box on the image
                mask = pixels_in_box(pixels, corners)

                # Use the mask to select the corresponding 3D LiDAR x and y values
                # Then take the median to get a stable estimate of the object's position
                x_med = np.median(x_values[mask]).item()
                y_med = np.median(y_values[mask]).item()

                # Compute the 2D Euclidean distance from the vehicle to the detected object
                distance = np.hypot(x_med, y_med)

                # Log the estimated x, y position and the 2D distance to the object
                self.get_logger().info(f"{label}: x={x_med:.2f}, y={y_med:.2f}, distance={distance:.2f}")

                # Store the detection as [label, score, x, y, z]
                detections[id] = [label, score, x_med, y_med, 0.0]  # z = 0 since LiDAR is 2D

                # Save the distance using the object's label as key (e.g., 'car': 3.42)
                distance_dict[label] = distance

            # Draw all label: distance entries onto the image
            plot = display_distances(plot, distance_dict)

        """ Step 2: 2D Object Detection (Homography)
        Detect flat objects like lane lines and project them to the ground plane using homography.
        Ideal for surface-level features."""

        # Identify the next waypoint along the lane line
        # This function checks if any of the 2D object classes (e.g., 'center') are present in the prediction mask.
        # 'success' will be True if at least one of the specified 2D object classes is detected in the mask.

        # The prediction mask uses 0 for background, and (class_id + 1) for each class.
        # For example, if 'center' has class_id = 1, then all 'center' pixels in the mask will have value 2.

        # 'scores' is a dictionary with the highest score per detected class.
        # Format: {id: {'label': 'center', 'score': 0.98, ...}}

        success, mask, scores = parse_predictions(predictions, self.objects_2d)


        # Right now, we only detect the center line.
        # In future lane-keeping setups, you might have two separate classes: 'left' and 'right' lane lines.
        # The mask will then include three values: background (0), left (class_id + 1), and right (class_id + 1).
        # For example, if left = 2 and right = 5, the mask values will be: left = 3, right = 6.
        # You’ll need to decide how to compute a good waypoint (e.g., midpoint between lines).
        if success:
            
            # Get the base point (e.g., bottom-most pixel) from the mask to use as the waypoint
            cx, cy = get_base(mask)

            # Draw a visual marker (e.g., a circle) at the selected waypoint position on the image
            plot = draw_circle(plot, cx, cy)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            for id, info in scores.items():
                label = info['label']
                score = info['score']

                detections[id] = [label, score, x, y, 0.0]

        # Convert the detections into a Detection3DArray message and publish it
        msg = to_detection3d_array(detections, timestamp_unix)
        self.detection3d_pub.publish(msg)

        # Convert back to ROS2 Image and publish
        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

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