import launch
import launch_ros.actions
from launch.substitutions import EnvironmentVariable
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the configuration file for the line follower node
    # CHANGE THIS  ----------------------------------------------------------VVVV
    config_file_path = get_package_share_directory('line_follower') + '/config/pid_gains.yaml'

    # Define a substitution that reads the 'ROBOT_NAMESPACE' environment variable.
    # If the environment variable is not set, it defaults to an empty string (global namespace).
    robot_namespace_env = EnvironmentVariable('ROBOT_NAMESPACE', default_value = '')

    ball_tracker = launch_ros.actions.Node(
        package='line_follower',
        executable='ball_tracker',
        name='ball_tracker',
        namespace=robot_namespace_env,
        output='screen',
        parameters=[config_file_path],
        remappings=[
            ('/object', [robot_namespace_env, '/object']),
            ('/waypoint', [robot_namespace_env, '/waypoint']),
            ('/result', [robot_namespace_env, '/result'])
        ]
    )
    

    # Return launch description including both nodes
    return launch.LaunchDescription([
        ball_tracker
    ])
