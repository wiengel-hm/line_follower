# Third-Party Libraries
import os
import cv2
import numpy as np

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection3DArray, LabelInfo
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import (
    to_surface_coordinates, read_transform_config, parse_predictions, 
    get_base, detect_bbox_center, draw_circle,
    get_bounding_boxes, pixels_in_box, display_distances
)
from ros2_numpy import image_to_np, np_to_image, scan_to_np,  to_detection3d_array, to_label_info
from ultralytics import YOLO 
# from ultralytics import YOLO
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
            LabelInfo, # {0: 'crosswalk', 1: 'left_lane', ...}
            '/label_mapping',
            qos_transient
        )

        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(Image, '/result', qos_profile)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        self.id2label = self.model.names # {0: 'crosswalk', 1: 'left_lane', ...}
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
        objects_2d = ['crosswalk', 'left_lane', 'right_lane']
        self.objects_2d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_2d}

        # Classes processed using camera-LiDAR fusion.
        # This method is used for tall 3D objects like cars or stop signs.
        # It uses a projector that transforms the LiDAR point cloud into the camera frame
        # and projects the 3D points onto the image to estimate object distance and position.
        # Make sure objects are tall enough — our single-layer LiDAR may miss low objects.
        objects_3d = ['pedcrossing_sign', 'person']  # Add other 3D objects like 'stop_sign' if needed
        self.objects_3d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_3d}

        # Stores the latest 3D LiDAR points in vehicle coordinates (x: forward, y: left, z: up)
        self.pts = None

        # LiDAR-to-camera projection tool used for camera-LiDAR fusion
        self.projector = LidarToImageProjector()

        # Navigation parameters for left-only mode
        self.lane_width_pixels = 200  # Estimated lane width in pixels (adjust based on your setup)
        self.min_curve_threshold = 0.1  # Minimum curvature to consider it a curve
        self.straight_offset_ratio = 1  # For straight lanes, offset = lane_width * ratio
        self.curve_offset_ratio = 0.8  # For curves, use smaller offset to stay safer
        
        # Store previous left lane positions for curvature calculation
        self.left_lane_history = []
        self.history_length = 5  # Number of frames to keep for curvature analysis

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
    
    def calculate_curvature(self, left_points): # am I overthinking this? 
        """Calculate curvature of the left lane line from recent positions"""
        if len(left_points) < 3:
            return 0.0  # There's not enough points to calculate curvature
        
        # Use the last 3 points to estimate curvature (maybe 5?)
        points = np.array(left_points[-3:])
        
        # Calculate the change in direction
        if len(points) >= 3:
            # Vector from first to second point
            v1 = points[1] - points[0]
            # Vector from second to third point  
            v2 = points[2] - points[1]
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                # Curvature is inversely related to the angle (this math may be incorrect)
                curvature = 1.0 - (angle / np.pi)
                return curvature
        
        return 0.0

    def calculate_left_only_waypoint(self, left_cx, left_cy, mask):
        """Calculate navigation waypoint using only left lane line"""
        
        # Add current left position to history
        self.left_lane_history.append([left_cx, left_cy])
        if len(self.left_lane_history) > self.history_length:
            self.left_lane_history.pop(0)
        
        # Calculate curvature to determine if we're on a curve
        curvature = self.calculate_curvature(self.left_lane_history)
        
        # Determine if this is a curve or straight section
        is_curve = curvature > self.min_curve_threshold
        
        # Choose offset based on road type (this may be unecessarily complicated but still)
        if is_curve:
            # On curves, use smaller offset to stay safer and avoid cutting corners
            offset_ratio = self.curve_offset_ratio
            offset_pixels = int(self.lane_width_pixels * offset_ratio)
            navigation_mode = "left_only_curve"
        else:
            # On straight sections, use larger offset to center better
            offset_ratio = self.straight_offset_ratio  
            offset_pixels = int(self.lane_width_pixels * offset_ratio)
            navigation_mode = "left_only_straight"
        
        # Calculate waypoint by offsetting to the right of the left lane
        # In image coordinates, moving right means increasing x
        waypoint_cx = left_cx + offset_pixels
        waypoint_cy = left_cy  # Keep same y-coordinate (distance ahead)
        
        # Ensure waypoint doesn't go outside image bounds
        image_width = mask.shape[1]
        waypoint_cx = min(waypoint_cx, image_width - 1)
        waypoint_cx = max(waypoint_cx, 0)
        
        # Debugging stuff to see if our curvature stuff is correct 
        self.get_logger().info(f"Left-only: curvature:{curvature:.3f}, "
                              f"is_curve:{is_curve}, offset:{offset_pixels}px, "
                              f"left:({left_cx},{left_cy}), waypoint:({waypoint_cx},{waypoint_cy})")
        
        return waypoint_cx, waypoint_cy, navigation_mode

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
        predictions = self.model(image, verbose = False)

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

        # Now we detect both left and right lane lines separately
        # The mask will include three values: background (0), left_lane (class_id + 1), and right_lane (class_id + 1).
        if success:

            # Initialize waypoint coordinates
            left_waypoint = None
            right_waypoint = None
            center_waypoint = None
            
            # Check if left lane line is detected
            if 'left_lane' in [scores[id]['label'] for id in scores]:
                # Extract only the left lane pixels from the mask
                left_class_id = self.label2id['left_lane']
                left_mask = (mask == (left_class_id + 1))
                if np.any(left_mask):
                    # Get the base point (e.g., bottom-most pixel) from the left lane mask
                    left_cx, left_cy = get_base(left_mask.astype(np.uint8))
                    left_waypoint = (left_cx, left_cy)
                    # Draw a visual marker for the left waypoint
                    plot = draw_circle(plot, left_cx, left_cy)
            
            # Check if right lane line is detected
            if 'right_lane' in [scores[id]['label'] for id in scores]:
                # Extract only the right lane pixels from the mask
                right_class_id = self.label2id['right_lane']
                right_mask = (mask == (right_class_id + 1))
                if np.any(right_mask):
                    # Get the base point from the right lane mask
                    right_cx, right_cy = get_base(right_mask.astype(np.uint8))
                    right_waypoint = (right_cx, right_cy)
                    # Draw a visual marker for the right waypoint
                    plot = draw_circle(plot, right_cx, right_cy)
            
            # Compute center waypoint based on available lane lines
            if left_waypoint and right_waypoint:
                # Both lanes detected - use center between them
                center_cx = (left_waypoint[0] + right_waypoint[0]) // 2
                center_cy = (left_waypoint[1] + right_waypoint[1]) // 2
                center_waypoint = (center_cx, center_cy)
                navigation_mode = "center_between_lanes"
                # Draw center waypoint marker
                plot = draw_circle(plot, center_cx, center_cy, color=(255, 0, 0))
                
            elif left_waypoint and not right_waypoint:
                # Only left lane detected - calculate waypoint using offset
                left_cx, left_cy = left_waypoint
                center_cx, center_cy, navigation_mode = self.calculate_left_only_waypoint(left_cx, left_cy, mask)
                center_waypoint = (center_cx, center_cy)
                # Draw calculated center waypoint marker in different color
                plot = draw_circle(plot, center_cx, center_cy, color=(0, 255, 255))  # Yellow for left-only mode
                
            elif right_waypoint and not left_waypoint:
                # Only right lane detected - use right lane position directly
                # (You could implement a similar offset calculation for right-only mode)
                center_cx, center_cy = right_waypoint
                center_waypoint = (center_cx, center_cy)
                navigation_mode = "right_lane_only"
                # Draw right waypoint marker
                plot = draw_circle(plot, center_cx, center_cy, color=(0, 255, 0))  # Green for right-only
            else:
                # Fallback - no clear lane detection
                self.get_logger().warn("No clear lane lines detected")
                navigation_mode = "no_lanes"
            
            # Store the final waypoint detection if we have one
            if center_waypoint:
                cx, cy = center_waypoint
                # Transform from pixel coordinates to surface coordinates using homography
                x, y = self.to_surface_coordinates(cx, cy)
                
                # Determine the score based on detection mode
                if navigation_mode == "center_between_lanes":
                    left_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'left_lane'), 0.0)
                    right_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'right_lane'), 0.0)
                    center_score = max(left_score, right_score)
                elif "left_only" in navigation_mode:
                    center_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'left_lane'), 0.0)
                elif navigation_mode == "right_lane_only":
                    center_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'right_lane'), 0.0)
                else:
                    center_score = 0.5  # Default score for calculated waypoints
                
                # Store as a virtual 'center' detection
                center_id = len(self.id2label)  # Use next available ID for virtual center
                detections[center_id] = ['center', center_score, x, y, 0.0]
            
            # Also store individual lane detections if they exist
            if left_waypoint:
                left_cx, left_cy = left_waypoint
                x, y = self.to_surface_coordinates(left_cx, left_cy)
                left_id = self.label2id['left_lane']
                left_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'left_lane'), 0.0)
                detections[left_id] = ['left_lane', left_score, x, y, 0.0]
                
            if right_waypoint:
                right_cx, right_cy = right_waypoint
                x, y = self.to_surface_coordinates(right_cx, right_cy)
                right_id = self.label2id['right_lane']
                right_score = next((scores[id]['score'] for id in scores if scores[id]['label'] == 'right_lane'), 0.0)
                detections[right_id] = ['right_lane', right_score, x, y, 0.0]
            
            # Log detection status
            self.get_logger().info(f"Lane detection - Left: {'✓' if left_waypoint else '✗'}, "
                                  f"Right: {'✓' if right_waypoint else '✗'}, "
                                  f"Mode: {navigation_mode}")

        # Convert the detections into a Detection3DArray message and publish it
        msg = to_detection3d_array(detections, timestamp_unix)
        self.detection3d_pub.publish(msg)

        # Convert back to ROS2 Image and publish
        im_msg = np_to_image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

        # Publish predictions
        self.im_publisher.publish(im_msg)


def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    # model_path = pkg_path + '/models/crosswalk.onnx'
    model_path = pkg_path + '/models/crosswalk.pt'


            
    rclpy.init(args=args)
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()