# Third-Party Libraries
import os
import numpy as np
from line_follower.trt_engine_loader import PilotNet

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_prefix

# Project-Specific Imports
from ros2_numpy import image_to_np, to_ackermann

class LineFollower(Node):
    def __init__(self, model_path):
        super().__init__('line_tracker')

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message
        
        # Create a publisher for sending AckermannDriveStamped messages to the '/autonomous/ackermann_cmd' topic
        self.publisher = self.create_publisher(AckermannDriveStamped, '/autonomous/ackermann_cmd', qos_profile)


        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The file at '{model_path}' was not found.")


        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Initialize the TensorRT model
        self.model = PilotNet(model_path)

        self.speed = 1.0

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. PilotNet trt engine loaded successfully.")

    
    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)


        # Run PilotNt inference
        steering_angle = self.model.predict(image)

        # Get the timestamp from the message header
        timestamp = msg.header.stamp

        # Create an Ackermann drive message with speed and steering angle
        kp = 1.3
        ackermann_msg = to_ackermann(self.speed, steering_angle * kp, timestamp)

        # Publish the message to the vehicle
        self.publisher.publish(ackermann_msg)


def main(args=None):


    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    run = 'classic-pine-17'
    model_path = pkg_path + f'/models/{run}/pilotnet.trt'
            
    rclpy.init(args=args)
    node = LineFollower(model_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()