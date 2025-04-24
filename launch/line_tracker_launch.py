import launch
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the configuration file for the line follower node
    config_file_path = get_package_share_directory('line_follower') + '/config/config_lane_640x360.yaml'
    
    # Follower node declaration
    line_tracker = launch_ros.actions.Node(
        package='line_follower',
        executable='line_tracker',
        name='line_tracker',
        output='screen',
        parameters=[config_file_path]
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        line_tracker
    ])
