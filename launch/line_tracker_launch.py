import launch
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the configuration file for the line follower node
    config_file_path = get_package_share_directory('line_follower') + '/config/config_lane_640x360.yaml'

    # Define a substitution that reads the 'ROBOT_NAMESPACE' environment variable.
    # If the environment variable is not set, it defaults to an empty string (global namespace).
    robot_namespace_env = EnvironmentVariable('ROBOT_NAMESPACE', default_value = '')

    # Follower node declaration
    line_tracker = Node(
        package='line_follower',
        executable='line_tracker',
        name='line_tracker',
        output='screen',
        namespace=robot_namespace_env,
        parameters=[config_file_path]
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        line_tracker
    ])
