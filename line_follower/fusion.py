import numpy as np


class LidarToImageProjector:

    def __init__(self, im_w = 640, im_h = 360):
        self.K = self.get_camera_intrinsics()
        self.T = self.get_transformation_matrix()
        self.im_w = im_w
        self.im_h = im_h


    def project_points_to_image(self, pts):
        # Add homogeneous coordinate
        pts_homo = np.hstack((pts, np.ones((pts.shape[0], 1))))

        # Transform from Lidar to Camera frame
        pts_cam = (self.T @ pts_homo.T).T

        # Remove Lidar points that are behind the camera
        # Otherwise they would be incorrectly projected onto the image
        valid = pts_cam[:, 2] > 1e-6
        pts_cam = pts_cam[valid]

        # Extract depth values (z-coordinate) in the camera coordinate system
        depth = pts_cam[:, 2]

        # Project to image plane
        pts_2d = self.K @ pts_cam[:, :3].T

        # Perspectiv devision: f * x / z, f * y / z
        pixels = (pts_2d[:2] / pts_2d[2]).T

        # Remove pixels that fall outside the image boundaries
        u, v = pixels.T
        mask = (u >= 0) & (u < self.im_w) & (v >= 0) & (v < self.im_h)

        # Keep only valid pixel projections and corresponding depths
        pixels = pixels[mask]

        # Depth in camera coordinates (along the optical axis, i.e., z-direction)
        depth = depth[mask]

        # Filter corresponding original LiDAR points (in vehicle coordinates)
        # x_values: forward direction in vehicle frame
        # y_values: left direction in vehicle frame
        x_values = pts[valid][mask, 0]
        y_values = pts[valid][mask, 1]

        return pixels, depth, x_values, y_values

    @staticmethod
    def get_camera_intrinsics():
        """
        Returns the hardcoded 3x3 camera intrinsic matrix (K).
        """
        K = np.array([
            [455.21691895,   0.0,         324.14334106],
            [  0.0,         455.26907349, 188.91212463],
            [  0.0,           0.0,           1.0      ]
        ])
        return K

    @staticmethod
    def get_transformation_matrix():
        """
        Returns a fixed 3x4 transformation matrix from LiDAR to camera frame.

        LiDAR frame: x-forward, y-left, z-down (left-handed)
        Camera frame: x-right, y-down, z-forward (OpenCV/ROS optical frame)
        """
        # Base rotation to align LiDAR frame with camera frame
        R_base = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])

        # Additional LiDAR rotation: ~228 degrees clockwise around Z-axis
        theta = np.deg2rad(228)
        R_rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        # Combined rotation
        R = R_base @ R_rot

        # Translation vector (in camera coordinates)
        t = np.array([[-0.0015, -0.06, -0.04]])  # shape: (1, 3)

        # Combine rotation and translation into a 3x4 matrix
        T = np.hstack((R, t.T))
        return T
