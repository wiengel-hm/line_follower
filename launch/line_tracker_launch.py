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

    # Launch rqt_image_view to visualize image topics
    rqt_image_view_node = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_image_view",
        output="screen"
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        line_tracker,
        rqt_image_view_node
    ])
