from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

'''
Overall Launch Package for Ball Follower Launch

Launch Tree:
ball_follower_launch
- ball_tracker_launch
    - ball_tracker
    - rqt_image_view_node
- soccer_pid_launch
    - soccer_pid_node

'''

def generate_launch_description():
    # Get package paths
    ball_tracker_pkg = FindPackageShare("rocket_league")
    controller_pkg = FindPackageShare("controller")

    # Group for line_tracker
    ball_tracker_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([ball_tracker_pkg, "/launch/ball_tracker_launch.py"])
            )
        ]
    )

    # Group for PID control
    soccer_pid_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([controller_pkg, "/launch/soccer_pid_launch.py"])
            )
        ]
    )

    return LaunchDescription([
        ball_tracker_launch,
        soccer_pid_launch
    ])
