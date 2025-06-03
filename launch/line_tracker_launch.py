import launch
import launch_ros.actions
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Follower node declaration
    line_tracker = Node(
        package='line_follower',
        executable='line_tracker',
        name='line_tracker',
        output='screen'
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        line_tracker
    ])
