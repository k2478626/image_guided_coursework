#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Load MoveIt config from your MoveIt config package (update if yours is named differently)
    moveit_config = (
        MoveItConfigsBuilder("neuro_arm", package_name="my_neuro_arm_in_moveit")
        .to_moveit_configs()
    )

    # Static transform publisher: world -> base_link
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"]
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    # Your custom Python node to run MoveItPy logic
    my_robot_goal_node = Node(
        package="my_neuro_arm_goal",
        executable="my_robot_goal",
        output="screen"
    )

    return LaunchDescription([
        static_tf,
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        my_robot_goal_node,
    ])
