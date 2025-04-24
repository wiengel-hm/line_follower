from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package paths
    line_tracker_pkg = FindPackageShare("line_follower")
    controller_pkg = FindPackageShare("controller")

    # Group for line_tracker
    line_tracker_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([line_tracker_pkg, "/launch/line_tracker_launch.py"])
            )
        ]
    )

    # Group for PID control
    pid_control_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([controller_pkg, "/launch/pid_control_launch.py"])
            )
        ]
    )

    return LaunchDescription([
        line_tracker_launch,
        pid_control_launch
    ])
