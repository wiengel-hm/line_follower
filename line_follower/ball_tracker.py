import os
import cv2
import numpy as np
from yolo_onnx_runner import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from rocket_league.utils import (
    to_surface_coordinates,
    read_transform_config,
    # ──────────────────────────────────────────────────────────────────────────
    parse_predictions,    # ── CHANGED: now needed for segmentation
    get_base,             # ── CHANGED: used to compute centroid of mask
    # ──────────────────────────────────────────────────────────────────────────
    # detect_bbox_center,  # ── NO LONGER USED (segmentation replaces it)
)
from ros2_numpy import image_to_np, np_to_image, np_to_pose

class BallTracker(Node):
    def __init__(self, model_path, config_path):
        super().__init__('ball_tracker')

        # Prepare stop message
        self.stop_msg = PoseStamped()
        self.stop_msg.pose.position.x = self.stop_msg.pose.position.y = float('nan')
        # (z will be set per‐object when publishing)

        # Verify files
        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")

        # Load homography matrix
        H = read_transform_config(config_path)
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        qos_profile = rclpy.qos.qos_profile_sensor_data
        qos_profile.depth = 1

        # Subscribers and publishers
        self.im_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile
        )
        self.way_pub = self.create_publisher(PoseStamped, '/waypoint', qos_profile)
        self.obj_pub = self.create_publisher(PoseStamped, '/object', qos_profile)
        self.im_pub = self.create_publisher(Image, '/result', qos_profile)

        # ──────────────────────────────────────────────────────────────────────────
        # CHANGED: instantiate YOLO in segmentation mode by passing float thresholds
        #         (remove the bogus 'detectt' argument you had before)
        #
        # Original line (wrong):   self.model = YOLO(model_path, 'detectt')
        #
        # New:
        self.model = YOLO(
            model_path,
            conf_thres=0.25,   # must be a float, not a string
            iou_thres=0.45     # likewise, a float
        )
        # ──────────────────────────────────────────────────────────────────────────

        # figure out the class‐ID for your target labels
        targets = ['ball', 'goal']
        id2label = self.model.names  # e.g. {0:'ball', 1:'goal', ...}

        # keep only the IDs we care about
        self.id2target = {cid: lbl for cid, lbl in id2label.items() if lbl in targets}

        self.get_logger().info("Ball Tracker Node started. YOLO segmentation model loaded.")

    def image_callback(self, msg: Image):
        # 1) Convert ROS Image → NumPy BGR image + grab timestamp
        image_bgr, timestamp = image_to_np(msg)

        # 2) Run YOLO inference (segmentation)
        try:
            predictions = self.model(image_bgr)
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")
            return

        # 3) Overlay masks/contours on the BGR image
        try:
            annotated_bgr = predictions[0].plot()
        except Exception as e:
            self.get_logger().error(f"Error plotting segmentation results: {e}")
            annotated_bgr = image_bgr.copy()

        # 4) Convert annotated BGR → RGB and publish as sensor_msgs/Image on /result
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        img_msg = np_to_image(annotated_rgb)      # ros2_numpy helper
        img_msg.header.stamp = msg.header.stamp
        self.im_pub.publish(img_msg)

        # ──────────────────────────────────────────────────────────────────────────
        # 5) “Waypoint” logic: use parse_predictions to get mask for “ball” (ID=0)
        ball_id = next((cid for cid, lbl in self.id2target.items() if lbl == 'ball'), None)
        if ball_id is not None:
            try:
                found_ball, ball_mask = parse_predictions(predictions, [ball_id])
            except Exception as e:
                self.get_logger().error(f"Error parsing predictions for ball: {e}")
                found_ball, ball_mask = False, None

            if found_ball and (ball_mask is not None):
                cx, cy = get_base(ball_mask)
                x_b, y_b = self.to_surface_coordinates(cx, cy)
                pose_msg = np_to_pose(np.array([x_b, y_b]), 0.0, timestamp=msg.header.stamp)
                self.get_logger().info("Found Ball!")
                self.way_pub.publish(pose_msg)
            else:
                self.way_pub.publish(self.stop_msg)
                self.get_logger().info("Lost Ball!")
        else:
            # If “ball” wasn’t in the model.names → id2target, just publish stop
            self.way_pub.publish(self.stop_msg)

        # ──────────────────────────────────────────────────────────────────────────
        # 6) “Object” loop: do the same for each target (‘ball’ and ‘goal’)
        for cid, lbl in self.id2target.items():
            try:
                detected, obj_mask = parse_predictions(predictions, [cid])
            except Exception as e:
                self.get_logger().error(f"Error parsing predictions for {lbl}: {e}")
                detected, obj_mask = False, None

            if detected and (obj_mask is not None):
                cx_o, cy_o = get_base(obj_mask)
                x_o, y_o = self.to_surface_coordinates(cx_o, cy_o)
                obj_msg = np_to_pose(np.array([x_o, y_o, float(cid)]), 0.0, timestamp=msg.header.stamp)
            else:
                obj_msg = PoseStamped()
                obj_msg.pose.position.x = float('nan')
                obj_msg.pose.position.y = float('nan')
                obj_msg.pose.position.z = float(cid)
                obj_msg.header.stamp = msg.header.stamp

            self.obj_pub.publish(obj_msg)

def main(args=None):
    rclpy.init(args=args)

    cfg = get_package_share_directory('rocket_league') + '/config/transform_config_640x360.yaml'
    pkg = get_package_prefix('rocket_league').replace('install', 'src')
    model = pkg + '/models/best.onnx'  # ─── point at your segmentation ONNX here

    node = BallTracker(model, cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
