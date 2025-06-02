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
from line_follower.utils import to_surface_coordinates, read_transform_config, parse_predictions, get_base, detect_bbox_center, draw_circle, get_onnx_boxes
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_image, scan_to_np
from yolo_onnx_runner import YOLO # pip install git+https://github.com/william-mx/yolo-onnx-runner.git
from line_follower.conversions import to_detection3d_array, to_label_info
from line_follower.fusion import LidarToImageProjector

class LineFollower(Node):
    def __init__(self, model_path, config_path):
        super().__init__('line_tracker')

        # Stores the latest 3D LiDAR points in vehicle coordinates (x: forward, y: left, z: up)
        self.pts = None

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

        # Publisher to send out 3D detections
        self.detection3d_pub = self.create_publisher(Detection3DArray, '/objects_3d', qos_profile)

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Subscriber to receive LiDAR scan data
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)

        # QoS profile that ensures the last published message is saved and sent to new subscribers.
        # This is useful for static or infrequently changing data like label maps or calibration info.
        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.label_pub = self.create_publisher(
            LabelInfo, # {0: 'car', 1: 'ceter', ...}
            '/label_mapping',
            qos_transient
        )

        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(Image, '/result', qos_profile)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = YOLO(model_path, conf_thres=0.1)

        # Map class IDs to labels and labels to IDs
        self.id2label = self.model.names
        self.label2id = {lbl: id for id, lbl in self.id2label.items()}

        info_msg = to_label_info(self.id2label) # Convert it to a LabelInfo message
        self.label_pub.publish(info_msg) # Publish it

        # Initialize default detections with all scores set to 0.0 (i.e., not detected)
        # Format: [label, score, x, y, z]
        self.detections = {id: [lbl, 0.0, 0.0, 0.0, 0.0] for id, lbl in self.id2label.items()}

        objects_3d = ['car'] # Classes to track 
        self.objects_3d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_3d}

        objects_2d = ['center']
        self.objects_2d = {id: lbl for id, lbl in self.id2label.items() if lbl in objects_2d}

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO ONNX model loaded successfully.")

        self.projector = LidarToImageProjector()

    def scan_callback(self, msg):
        xyi, timestamp_unix = scan_to_np(msg)
        x, y, intensity = xyi[:, 0], xyi[:, 1], xyi[:, 2]
        self.pts = np.column_stack([x, y, np.zeros_like(x)]) # (N, 3)

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Create a copy of the default detection list to reset or start fresh
        # Format: [label, score, x, y, z] — matches Detection3DArray format
        detections = self.detections.copy()

        # Run YOLO inference
        predictions = self.model(image)


        # === Step 1: 3D Object Detection and Fusion ===
        # - Extract object predictions from the model output
        # - Fuse predictions with depth from camera-LiDAR projection
        # - Add all detections to a Detection3DArray for publishing

        success, boxes = get_onnx_boxes(predictions, self.objects_3d)
        
        # Proceed only if an object was successfully detected
        # and LiDAR data has been received
        if success and self.pts is not None:

            
            # Project 3D points to 2D image coordinates
            pixels, depth, x_values, y_values = self.projector.project_points_to_image(self.pts)
            u, v = pixels.T
            
            
            for id, box in boxes.items():

                x1, y1, x2, y2 = box['corners']

                # Normalize box corners in case x1 > x2 or y1 > y2
                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                # Create a mask for pixels inside the bounding box
                mask = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)

                # Select corresponding depth values
                d = np.median(depth[mask]).item()
                x = -np.median(x_values[mask]).item() # invert axis
                y = np.median(y_values[mask]).item()

                self.get_logger().info(f"x={x:.2f}, y={y:.2f}, depth={d:.2f}, hypot={np.hypot(x, y):.2f}")

                detections[id] = [box['label'], box['score'], x, y, 0.0]


        # === Step 2: 2D Object Detection and Distance Estimation ===
        # - Extract object predictions from the model output
        # - Estimate 3D position using homography transformation
        # - Add all detections to a Detection3DArray for publishing

        # Identify the next waypoint along the lane line
        success, mask, scores = parse_predictions(predictions, self.objects_2d)

        if success:
            
            cx, cy = get_base(mask)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            for id, info in scores.items():
                label = info['label']
                score = info['score']

                detections[id] = [label, score, x, y, 0.0]

        msg = to_detection3d_array(detections, timestamp_unix)
        
        self.detection3d_pub.publish(msg)

        # Draw results on the image
        plot = predictions[0].plot()

        # Convert back to ROS2 Image and publish
        im_msg = np_to_image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

        # Publish predictions
        self.im_publisher.publish(im_msg)



def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    model_path = pkg_path + '/models/best.onnx'
            
    rclpy.init(args=args)
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()