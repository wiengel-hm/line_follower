import launch
import launch_ros.actions
from launch.substitutions import EnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Define a substitution that reads the 'ROBOT_NAMESPACE' environment variable.
    # If the environment variable is not set, it defaults to an empty string (global namespace).
    robot_namespace_env = EnvironmentVariable('ROBOT_NAMESPACE', default_value = '')
    ball_follower_pkg = FindPackageShare("line_follower")

    navigation = launch_ros.actions.Node(
        package='line_follower',
        executable='navigation',
        name='navigation',
        namespace=robot_namespace_env,
        output='screen',
    )

    ball_follower_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([ball_follower_pkg, "/launch/ball_follower_launch.py"])
            )
        ]
    )

    # Return launch description including both nodes
    return launch.LaunchDescription([
        navigation,
        ball_follower_launch
    ])
