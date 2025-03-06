import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    td3_rl_path = get_package_share_directory('td3_rl')
    
    if "GZ_SIM_RESOURCE_PATH" in os.environ:
        gz_sim_resource_path = os.environ["GZ_SIM_RESOURCE_PATH"]

        if "SDF_PATH" in os.environ:
            sdf_path = os.environ["SDF_PATH"]
            os.environ["SDF_PATH"] = sdf_path + ":" + gz_sim_resource_path
        else:
            os.environ["SDF_PATH"] = gz_sim_resource_path

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('gui', default_value='false', description='Launch Gazebo GUI'),
        DeclareLaunchArgument('world_name', default_value=os.path.join(td3_rl_path, 'worlds/TD3.world'), description='World file'),
        
        # Launch Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    ros_gz_sim,
                    "launch",
                    "gz_sim.launch.py"
                ])
            ),
            launch_arguments={
                "gz_args": [
                    "-v4 -s -r ",
                    LaunchConfiguration('world_name')
                ]
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                f'{Path(ros_gz_sim) / "launch" / "gz_sim.launch.py"}'
            ),
            launch_arguments={"gz_args": "-v4 -g"}.items(),
            condition=IfCondition(LaunchConfiguration("gui")),
        )
    ])