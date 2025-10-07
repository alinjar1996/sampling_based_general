import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Declare launch arguments with defaults
    use_config_file_arg = DeclareLaunchArgument('use_config_file', default_value='false')
    use_hardware_arg = DeclareLaunchArgument('use_hardware', default_value='false')
    idx_arg = DeclareLaunchArgument('idx', default_value='0')
    record_data_arg = DeclareLaunchArgument('record_data', default_value='false')
    num_batch_arg = DeclareLaunchArgument('num_batch', default_value='500')
    num_steps_arg = DeclareLaunchArgument('num_steps', default_value='15')
    maxiter_cem_arg = DeclareLaunchArgument('maxiter_cem', default_value='1')
    maxiter_projection_arg = DeclareLaunchArgument('maxiter_projection', default_value='5')
    num_elite_arg = DeclareLaunchArgument('num_elite', default_value='0.05')
    timestep_arg = DeclareLaunchArgument('timestep', default_value='0.1')

    # Launch configurations to fetch launch args
    use_config_file = LaunchConfiguration('use_config_file')
    use_hardware = LaunchConfiguration('use_hardware')
    idx = LaunchConfiguration('idx')
    record_data = LaunchConfiguration('record_data')
    num_batch = LaunchConfiguration('num_batch')
    num_steps = LaunchConfiguration('num_steps')
    maxiter_cem = LaunchConfiguration('maxiter_cem')
    maxiter_projection = LaunchConfiguration('maxiter_projection')
    num_elite = LaunchConfiguration('num_elite')
    timestep = LaunchConfiguration('timestep')

    # params = os.path.join(
    #         get_package_share_directory('real_demo'),
    #         'config',
    #         'planner_parameters.yaml'
    #     )
    
    params = {
        'use_hardware': use_hardware,
        'record_data': record_data,
        'idx': idx,
        'num_batch': num_batch,
        'num_steps': num_steps,
        'maxiter_cem': maxiter_cem,
        'maxiter_projection': maxiter_projection,
        'num_elite': num_elite,
        'timestep': timestep,
    }

    return LaunchDescription([
        use_config_file_arg,
        use_hardware_arg,
        idx_arg,
        record_data_arg,
        num_batch_arg,
        num_steps_arg,
        maxiter_cem_arg,
        maxiter_projection_arg,
        num_elite_arg,
        timestep_arg,
        Node(
            package='real_demo',
            executable='pendulum_demo',
            name='planner',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
            parameters=[params],
        )
    ])